#!/usr/bin/env python3
"""
Phase 2: Ridge regression of model attention against regressive saccade patterns.

"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ConstantInputWarning, pearsonr, zscore
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore", category=ConstantInputWarning)
pd.options.mode.chained_assignment = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    ATTENTION_DIR,
    EVENTS_TEMPLATE,
    EYETRACK_SUBJECTS,
    N_ARTICLES,
    RESULTS_DIR,
    RIDGE_ALPHAS,
    SENTS_PER_ARTICLE,
    load_words_list,
)

SACCADE_DIR = RESULTS_DIR / "saccade"
CUMS = np.cumsum(SENTS_PER_ARTICLE)  # [31,62,90,118,148]

_VALID_SID_RE = r"^t\.\d+\.\d+$"


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def _first_valid_article(df: pd.DataFrame) -> int | None:
    """Return the article number from the first valid SentenceID row."""
    sids = df["SentenceID"].astype(str)
    valid = sids[sids.str.match(_VALID_SID_RE)]
    if valid.empty:
        return None
    return int(valid.iloc[0].split(".")[1])


def load_subject_events(subj_id: int) -> pd.DataFrame | None:
    """Load and concatenate events for all 5 articles.

    Matches each run to its article via SentenceID 
    """
    sub = f"{subj_id:02d}"
    events_all: list[pd.DataFrame] = []
    for article in range(1, N_ARTICLES + 1):
        for run in range(1, 6):
            fpath = EVENTS_TEMPLATE.format(sub=sub, run=run)
            if not Path(fpath).exists():
                continue
            df = pd.read_csv(fpath, delimiter="\t")
            df = df[df["onset"].notna()].reset_index(drop=True)
            if _first_valid_article(df) == article:
                df["article"] = article
                events_all.append(df)
                break
    if not events_all:
        return None
    return pd.concat(events_all, ignore_index=True)


# ---------------------------------------------------------------------------
# Saccade matrix construction
# ---------------------------------------------------------------------------


def _snt_id(article: int, sent_i: int) -> str:
    return f"t.{article:02d}.{sent_i:02d}"


def build_saccade_vectors(
    events: pd.DataFrame, words_list: list[int]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build per-article saccade vectors in article-sequential order.

    Returns
    -------
    y_num_by_article, y_dur_by_article : list of 5 arrays
        Each element is the concatenated lower-triangle vector for one article.
    """
    y_num_by_article: list[np.ndarray] = []
    y_dur_by_article: list[np.ndarray] = []

    for article in range(1, N_ARTICLES + 1):
        art_num: list[float] = []
        art_dur: list[float] = []
        n_snts = SENTS_PER_ARTICLE[article - 1]
        for sent_i in range(1, n_snts + 1):
            sid = (sent_i - 1) if article == 1 else CUMS[article - 2] + (sent_i - 1)
            sent_len = words_list[sid]
            tril_idx = np.tril_indices(sent_len, k=-1)
            sac_num = np.zeros((sent_len, sent_len))
            sac_dur = np.zeros((sent_len, sent_len))

            event = events[
                events["SentenceID"].astype(str) == _snt_id(article, sent_i)
            ].copy()
            # Keep only rows with a numeric AOI (drop '.' placeholders)
            aoi = pd.to_numeric(event["CURRENT_FIX_INTEREST_AREA_ID"], errors="coerce")
            event = event[aoi.notna()].reset_index(drop=True)

            if len(event) >= 2:
                dup_mask = event["CURRENT_FIX_INTEREST_AREA_ID"].eq(
                    event["CURRENT_FIX_INTEREST_AREA_ID"].shift()
                )
                dup_idx = event[dup_mask].index.tolist()
                for idx in dup_idx:
                    if idx > 0:
                        event.at[idx - 1, "duration"] += event.at[idx, "duration"]
                event = event.drop(dup_idx).reset_index(drop=True)

                for ind in range(len(event) - 1):
                    row = int(event.iloc[ind]["CURRENT_FIX_INTEREST_AREA_ID"]) - 1
                    col = int(event.iloc[ind + 1]["CURRENT_FIX_INTEREST_AREA_ID"]) - 1
                    if 0 <= row < sent_len and 0 <= col < sent_len:
                        sac_num[row, col] += 1
                        sac_dur[row, col] += (
                            event.iloc[ind]["duration"]
                            + event.iloc[ind + 1]["duration"]
                        )

            art_num.extend(sac_num[tril_idx])
            art_dur.extend(sac_dur[tril_idx])

        y_num_by_article.append(np.array(art_num, dtype=np.float32))
        y_dur_by_article.append(np.array(art_dur, dtype=np.float32))

    return y_num_by_article, y_dur_by_article


def _safe_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson r with constant-vector guard."""
    if np.nanstd(pred) == 0 or np.nanstd(target) == 0:
        return float("nan")
    r, _ = pearsonr(pred, target)
    return float(r)


# ---------------------------------------------------------------------------
# Per-subject regression worker
# ---------------------------------------------------------------------------


def _process_subject(
    subj_id: int,
    X_by_article: list[np.ndarray],
    words_list: list[int],
) -> tuple[float, float] | None:
    """Run LOAO ridge regression for one subject.

    Returns
    -------
    (mean_r_num, mean_r_dur) across 5 leave-one-article-out folds.
    """
    events = load_subject_events(subj_id)
    if events is None:
        print(f"  sub-{subj_id:02d}: no events — skipped")
        return None

    y_num_by_article, y_dur_by_article = build_saccade_vectors(events, words_list)

    y_num_full = np.concatenate(y_num_by_article, axis=0)
    y_dur_full = np.concatenate(y_dur_by_article, axis=0)
    y_num_full = np.nan_to_num(zscore(y_num_full, nan_policy="omit"))
    y_dur_full = np.nan_to_num(zscore(y_dur_full, nan_policy="omit"))

    art_sizes = [len(v) for v in y_num_by_article]
    offsets = np.cumsum([0] + art_sizes)
    y_num_z = [y_num_full[offsets[i]:offsets[i + 1]] for i in range(N_ARTICLES)]
    y_dur_z = [y_dur_full[offsets[i]:offsets[i + 1]] for i in range(N_ARTICLES)]

    fold_r_num: list[float] = []
    fold_r_dur: list[float] = []
    for test_art in range(N_ARTICLES):
        X_test = X_by_article[test_art]
        X_train = np.concatenate(
            [X_by_article[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )
        y_num_test = y_num_z[test_art]
        y_num_train = np.concatenate(
            [y_num_z[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )
        y_dur_test = y_dur_z[test_art]
        y_dur_train = np.concatenate(
            [y_dur_z[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )

        if X_train.shape[0] != len(y_num_train) or X_test.shape[0] != len(y_num_test):
            print(
                f"  sub-{subj_id:02d}: fold{test_art + 1} dimension mismatch "
                f"(X_train={X_train.shape[0]}, y_train={len(y_num_train)}, "
                f"X_test={X_test.shape[0]}, y_test={len(y_num_test)}) — skipped"
            )
            return None

        clf_num = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, y_num_train)
        clf_dur = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, y_dur_train)
        fold_r_num.append(_safe_pearson(clf_num.predict(X_test), y_num_test))
        fold_r_dur.append(_safe_pearson(clf_dur.predict(X_test), y_dur_test))

    r_num = float(np.nanmean(fold_r_num)) if np.isfinite(fold_r_num).any() else float("nan")
    r_dur = float(np.nanmean(fold_r_dur)) if np.isfinite(fold_r_dur).any() else float("nan")
    return float(np.nan_to_num(r_num)), float(np.nan_to_num(r_dur))


# ---------------------------------------------------------------------------
# Attention flattening
# ---------------------------------------------------------------------------


def build_X_by_article(layer_attn: np.ndarray, words_list: list[int]) -> list[np.ndarray]:
    """Flatten attention to per-article lower-triangle feature matrices.

    Parameters
    ----------
    layer_attn : (5, 31, n_heads, 16, 16)
    words_list : 148 sentence lengths

    Returns
    -------
    X_by_article : list of 5 arrays
        Each array has shape (total_pairs_in_article, n_heads).
    """
    by_article: list[np.ndarray] = []
    sent_idx = 0
    for art_idx, n_sents in enumerate(SENTS_PER_ARTICLE):
        art_parts: list[np.ndarray] = []
        for s in range(n_sents):
            w = words_list[sent_idx]
            tril_r, tril_c = np.tril_indices(w, k=-1)
            # (n_heads, w, w) → (n_heads, n_pairs) → (n_pairs, n_heads)
            feat = layer_attn[art_idx, s, :, :w, :w][:, tril_r, tril_c].T
            art_parts.append(feat)
            sent_idx += 1
        by_article.append(np.concatenate(art_parts, axis=0))
    return by_article


def _append_skip_log(log_path: Path, lines: list[str]) -> None:
    """Append skipped-layer messages to a run log."""
    if not lines:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] skip_nan_layers run\n")
        for ln in lines:
            fh.write(f"  {ln}\n")
        fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: ridge regression of attention vs saccades."
    )
    parser.add_argument("--model", required=True, help="Model key from models.py")
    parser.add_argument("--layer", type=int, default=None, help="Single layer index")
    parser.add_argument("--all_layers", action="store_true", help="Run all layers")
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject IDs, e.g. 1,2,3,4",
    )
    parser.add_argument(
        "--all_subjects", action="store_true", help="Run all eye-tracking subjects"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="Parallel jobs for subjects (default 4)"
    )
    parser.add_argument(
        "--skip_nan_layers",
        action="store_true",
        help="Skip layers whose attention file contains NaN values (and log them).",
    )
    args = parser.parse_args()

    # Resolve subjects
    if args.all_subjects:
        subjects = EYETRACK_SUBJECTS
    elif args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        subjects = EYETRACK_SUBJECTS[:4]  # default: first 4 valid subjects

    # Resolve layers
    attn_dir = ATTENTION_DIR / args.model
    all_npy = sorted(attn_dir.glob("rb_layer*.npy"))
    n_layers = len(all_npy)
    if n_layers == 0:
        print(f"No .npy files found in {attn_dir}. Run Phase 1 first.")
        sys.exit(1)

    if args.all_layers:
        layers = list(range(n_layers))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [0]

    words_list = load_words_list()
    save_dir = SACCADE_DIR / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Phase 2: Saccade Regression  [{args.model}]")
    print(f"  Subjects : {subjects}")
    print(f"  Layers   : {layers}")
    print("  CV       : leave-one-article-out (5 folds)")
    print(f"  n_jobs   : {args.n_jobs}")
    print(f"  skip_nan : {args.skip_nan_layers}")
    print("=" * 64)

    t0 = time.time()
    skipped_layers: list[str] = []
    for layer in layers:
        layer_path = attn_dir / f"rb_layer{layer}.npy"
        if not layer_path.exists():
            print(f"  Layer {layer}: file not found — skipped")
            skipped_layers.append(f"layer{layer:02d}: file not found")
            continue

        layer_attn = np.load(layer_path)  # (5, 31, n_heads, 16, 16)
        nan_count = int(np.isnan(layer_attn).sum())
        if nan_count > 0:
            msg = f"layer{layer:02d}: NaN detected in attention tensor ({nan_count} cells)"
            if args.skip_nan_layers:
                print(f"  {msg} — skipped")
                skipped_layers.append(msg)
                continue
            raise ValueError(
                f"{msg}. Re-run extraction or pass --skip_nan_layers to continue."
            )

        X_by_article = build_X_by_article(layer_attn, words_list)

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_process_subject)(sid, X_by_article, words_list)
            for sid in subjects
        )

        valid = [(sid, r) for sid, r in zip(subjects, results) if r is not None]
        valid_subjects = np.array([sid for sid, _ in valid], dtype=np.int32)
        r_num = np.array([r[0] for _, r in valid], dtype=np.float32)
        r_dur = np.array([r[1] for _, r in valid], dtype=np.float32)

        np.save(save_dir / f"layer{layer:02d}_num.npy", r_num)
        np.save(save_dir / f"layer{layer:02d}_dur.npy", r_dur)
        np.save(save_dir / f"layer{layer:02d}_subjects.npy", valid_subjects)

        print(
            f"  Layer {layer:2d}: "
            f"r_num={r_num.mean():.4f}±{r_num.std():.4f}  "
            f"r_dur={r_dur.mean():.4f}±{r_dur.std():.4f}  "
            f"(n={len(r_num)} subjects, ids saved)"
        )

    if skipped_layers:
        log_path = save_dir / "skipped_layers.log"
        _append_skip_log(log_path, skipped_layers)
        print("\nSkipped layers:")
        for ln in skipped_layers:
            print(f"  - {ln}")
        print(f"Skip log saved to: {log_path}")

    print(f"\nDone in {time.time() - t0:.1f}s  →  {save_dir}/")


if __name__ == "__main__":
    main()
