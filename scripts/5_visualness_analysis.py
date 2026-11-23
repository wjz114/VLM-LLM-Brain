#!/usr/bin/env python3
"""
Phase 5: Visualness moderation analysis.

"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import DATA_DIR, N_ARTICLES, RESULTS_DIR, SENTS_PER_ARTICLE, SENTENCES_FILE, WORDS_CSV

DEFAULT_LANCASTER = _SCRIPT_DIR.parent / "Lancaster_Sensorimotor_norms.csv"
DEFAULT_OUT_DIR = RESULTS_DIR / "visualness"
DEFAULT_FMRI_GLOBAL_BEST = RESULTS_DIR / "comparison" / "fmri_global_best_layer.csv"

_SID_RE = re.compile(r"^t\.(\d{2})\.(\d{2})$")
_NORM_RE = re.compile(r"[^A-Za-z0-9]+")
_EPS = 1e-6

STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BEEN", "BEING", "BUT", "BY",
    "FOR", "FROM", "HAD", "HAS", "HAVE", "HE", "HER", "HERS", "HIM", "HIS",
    "I", "IF", "IN", "INTO", "IS", "IT", "ITS", "ME", "MY", "OF", "ON", "OR",
    "OUR", "OURS", "SHE", "SO", "THAN", "THAT", "THE", "THEIR", "THEIRS",
    "THEM", "THEN", "THERE", "THESE", "THEY", "THIS", "THOSE", "TO", "UP",
    "US", "WAS", "WE", "WERE", "WHAT", "WHEN", "WHICH", "WHO", "WHOM", "WHY",
    "WILL", "WITH", "YOU", "YOUR", "YOURS",
}


def _parse_sid(sid: str) -> tuple[int, int]:
    m = _SID_RE.match(str(sid).strip())
    if not m:
        raise ValueError(f"invalid SentenceID: {sid}")
    return int(m.group(1)), int(m.group(2))


def _normalize_token(token: str) -> str:
    return _NORM_RE.sub("", str(token).upper())


def _lemma_candidates(token: str) -> list[str]:
    cands = [token]
    if len(token) > 4 and token.endswith("IES"):
        cands.append(token[:-3] + "Y")
    if len(token) > 3 and token.endswith("ES"):
        cands.append(token[:-2])
        cands.append(token[:-1])
    if len(token) > 2 and token.endswith("S"):
        cands.append(token[:-1])
    if len(token) > 5 and token.endswith("ING"):
        cands.append(token[:-3])
        cands.append(token[:-3] + "E")
    if len(token) > 4 and token.endswith("ED"):
        cands.append(token[:-2])
        cands.append(token[:-1])
    out: list[str] = []
    for x in cands:
        if x and x not in out:
            out.append(x)
    return out


def _is_content(norm_token: str, keep_numbers: bool) -> bool:
    if not norm_token:
        return False
    if not keep_numbers and norm_token.isdigit():
        return False
    if norm_token in STOPWORDS:
        return False
    if len(norm_token) == 1 and norm_token.isalpha():
        return False
    return True


def load_lancaster(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "Word" not in df.columns or "Visual.mean" not in df.columns:
        raise ValueError(f"{path} must contain 'Word' and 'Visual.mean' columns")
    df = df[["Word", "Visual.mean"]].dropna()
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = _normalize_token(row["Word"])
        if key:
            out[key] = float(row["Visual.mean"])
    return out


def load_sentence_texts(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    blocks = text.split("\n\n")
    if len(blocks) != N_ARTICLES:
        raise ValueError(f"{path} expected {N_ARTICLES} article blocks, got {len(blocks)}")
    out: dict[str, str] = {}
    for article_i, block in enumerate(blocks, start=1):
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        expected = SENTS_PER_ARTICLE[article_i - 1]
        if len(lines) != expected:
            raise ValueError(f"article {article_i}: expected {expected} lines, got {len(lines)}")
        for sent_i, sentence in enumerate(lines, start=1):
            out[f"t.{article_i:02d}.{sent_i:02d}"] = sentence
    return out


def score_sentence(
    words: list[str],
    lexicon: dict[str, float],
    *,
    keep_numbers: bool,
    use_bigrams: bool,
) -> dict[str, float | int]:
    norms = [_normalize_token(w) for w in words]
    content_ids = [i for i, t in enumerate(norms) if _is_content(t, keep_numbers)]
    n_content = len(content_ids)
    matched_words = 0
    scores: list[float] = []

    i = 0
    while i < len(content_ids):
        j = content_ids[i]
        tok = norms[j]
        if use_bigrams and i + 1 < len(content_ids):
            j2 = content_ids[i + 1]
            if j2 == j + 1:
                phrase = f"{norms[j]} {norms[j2]}"
                if phrase in lexicon:
                    scores.append(lexicon[phrase])
                    matched_words += 2
                    i += 2
                    continue
        found = False
        for cand in _lemma_candidates(tok):
            if cand in lexicon:
                scores.append(lexicon[cand])
                matched_words += 1
                found = True
                break
        if not found:
            pass
        i += 1

    coverage = (matched_words / n_content) if n_content > 0 else 0.0
    mean_score = float(np.mean(scores)) if scores else np.nan
    max_score = float(np.max(scores)) if scores else np.nan
    return {
        "n_words": int(len(words)),
        "n_content_words": int(n_content),
        "n_matched_words": int(matched_words),
        "coverage": float(coverage),
        "visualness_mean": mean_score,
        "visualness_max": max_score,
    }


def build_visualness_metadata(
    words_csv: Path,
    sentences_txt: Path,
    lexicon: dict[str, float],
    *,
    coverage_threshold: float,
    min_content_words: int,
    top_fraction: float,
    keep_numbers: bool,
    use_bigrams: bool,
    min_matched_words: int = 3,
) -> pd.DataFrame:
    words_df = pd.read_csv(words_csv, dtype={"SentenceID": str, "WordPosition": int, "Word": str})
    req = {"SentenceID", "WordPosition", "Word"}
    if not req.issubset(set(words_df.columns)):
        raise ValueError(f"{words_csv} missing required columns: {sorted(req - set(words_df.columns))}")

    sent_text = load_sentence_texts(sentences_txt)
    rows: list[dict] = []
    grouped = words_df.groupby("SentenceID", sort=False)
    for sid, g in grouped:
        article_id, _ = _parse_sid(sid)
        g = g.sort_values("WordPosition")
        words = g["Word"].astype(str).tolist()
        score = score_sentence(
            words,
            lexicon,
            keep_numbers=keep_numbers,
            use_bigrams=use_bigrams,
        )
        rows.append({
            "article_id": article_id,
            "fold": article_id,
            "sentence_id": sid,
            "sentence_text": sent_text.get(sid, ""),
            **score,
            "visualness_z_article": np.nan,
            "rank_within_article": np.nan,
            "k_top_bottom": 0,
            "visualness_group": "excluded",
        })

    df = pd.DataFrame(rows)
    if len(df) != sum(SENTS_PER_ARTICLE):
        raise ValueError(f"expected {sum(SENTS_PER_ARTICLE)} sentences, got {len(df)}")

    for article in range(1, N_ARTICLES + 1):
        mask_art = df["article_id"] == article
        idx_art = df.index[mask_art]
        eligible = (
            mask_art
            & (df["coverage"] >= coverage_threshold)
            & (df["n_content_words"] >= min_content_words)
            & (df["n_matched_words"] >= min_matched_words)
            & df["visualness_mean"].notna()
        )
        idx_eligible = df.index[eligible].tolist()
        k_nominal = int(np.floor(top_fraction * len(idx_eligible)))
        k_eff = min(k_nominal, len(idx_eligible) // 2)
        df.loc[idx_art, "k_top_bottom"] = k_eff

        if not idx_eligible:
            continue

        vals = df.loc[idx_eligible, "visualness_mean"].to_numpy(dtype=np.float64)
        if len(vals) > 1 and float(np.std(vals, ddof=1)) > 0:
            z = (vals - float(np.mean(vals))) / float(np.std(vals, ddof=1))
        else:
            z = np.zeros_like(vals)
        df.loc[idx_eligible, "visualness_z_article"] = z

        ranked = (
            df.loc[idx_eligible]
            .sort_values(["visualness_mean", "sentence_id"], ascending=[False, True], kind="mergesort")
            .index.tolist()
        )
        for rk, ix in enumerate(ranked, start=1):
            df.at[ix, "rank_within_article"] = int(rk)
            df.at[ix, "visualness_group"] = "middle"
        if k_eff > 0:
            top_idx = ranked[:k_eff]
            bottom_idx = ranked[-k_eff:]
            df.loc[top_idx, "visualness_group"] = "top"
            df.loc[bottom_idx, "visualness_group"] = "bottom"

    cols = [
        "article_id",
        "fold",
        "sentence_id",
        "sentence_text",
        "n_words",
        "n_content_words",
        "n_matched_words",
        "coverage",
        "visualness_mean",
        "visualness_max",
        "visualness_z_article",
        "rank_within_article",
        "k_top_bottom",
        "visualness_group",
    ]
    return df[cols].sort_values(["article_id", "sentence_id"]).reset_index(drop=True)


def _mode_int(series: pd.Series) -> int | None:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    x = x[x >= 0]
    if x.empty:
        return None
    vc = x.value_counts()
    top_count = int(vc.max())
    return int(sorted(vc[vc == top_count].index.tolist())[0])


def build_fixed_layers(
    global_best_csv: Path,
    llm_key: str,
    vlm_key: str,
    *,
    domain: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    pair = f"{llm_key}__vs__{vlm_key}"
    if not global_best_csv.exists():
        for key in (llm_key, vlm_key):
            rows.append({
                "model_pair": pair,
                "model_key": key,
                "layer": -1,
                "selection_source": str(global_best_csv),
                "selection_metric": "missing_source_csv",
                "domain": domain,
                "n_subjects_used": 0,
            })
        return pd.DataFrame(rows)

    df = pd.read_csv(global_best_csv)
    need = {"domain", "model", "best_layer"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{global_best_csv} missing columns: {sorted(need - set(df.columns))}")

    for key in (llm_key, vlm_key):
        sub = df[(df["domain"] == domain) & (df["model"] == key)]
        layer = _mode_int(sub["best_layer"]) if not sub.empty else None
        rows.append({
            "model_pair": pair,
            "model_key": key,
            "layer": int(layer) if layer is not None else -1,
            "selection_source": str(global_best_csv),
            "selection_metric": "mode(best_layer)_from_phase4_global_best_layer",
            "domain": domain,
            "n_subjects_used": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _onesided_signflip(vec: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    x = np.asarray(vec, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n_subjects": 0, "mean_obs": np.nan, "p_one_sided": np.nan}
    obs = float(np.mean(x))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, x.size))
    perm_means = np.mean(signs * x[None, :], axis=1)
    p = float((1 + np.sum(perm_means >= obs)) / (n_perm + 1))
    return {"n_subjects": int(x.size), "mean_obs": obs, "p_one_sided": p}


def _collect_oof_stats_files(path: Path) -> list[Path]:
    p = path.resolve()
    if p.is_file():
        if p.suffix.lower() != ".npz":
            raise ValueError(f"--oof_stats file must be .npz, got: {p}")
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("*_oof_stats.npz"))
        if not files:
            raise ValueError(f"--oof_stats directory has no *_oof_stats.npz files: {p}")
        return files
    raise ValueError(f"--oof_stats path does not exist: {p}")


def _parse_subject_ids(s: str) -> set[int]:
    s = s.strip()
    if not s:
        return set()
    out: set[int] = set()
    for tok in s.replace(",", " ").split():
        out.add(int(tok))
    return out


def _npz_scalar(npz: np.lib.npyio.NpzFile, key: str) -> object:
    arr = np.asarray(npz[key])
    if arr.size != 1:
        raise ValueError(f"NPZ key '{key}' must contain a scalar, got shape={arr.shape}")
    return arr.reshape(-1)[0].item()


def _corr_from_sufficient_stats(
    n_rows: int,
    sum_y: np.ndarray,
    sum_pred: np.ndarray,
    sum_y2: np.ndarray,
    sum_pred2: np.ndarray,
    sum_y_pred: np.ndarray,
) -> np.ndarray:
    n = float(n_rows)
    if n < 2:
        return np.full_like(sum_y, np.nan, dtype=np.float64)
    num = n * sum_y_pred - (sum_y * sum_pred)
    den_y = n * sum_y2 - (sum_y * sum_y)
    den_pred = n * sum_pred2 - (sum_pred * sum_pred)
    den = np.sqrt(np.maximum(den_y, 0.0) * np.maximum(den_pred, 0.0))
    r = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den > 0)
    r[~np.isfinite(r)] = np.nan
    return r


def run_oof_analysis(
    oof_stats_path: Path,
    sent_meta: pd.DataFrame,
    *,
    llm_key: str,
    vlm_key: str,
    n_perm: int,
    seed: int,
    exclude_subjects: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = _collect_oof_stats_files(oof_stats_path)
    meta = sent_meta[["sentence_id", "visualness_group"]].drop_duplicates("sentence_id")
    group_lookup = meta.set_index("sentence_id")["visualness_group"]

    req = {
        "sentence_id",
        "article_id",
        "fold",
        "n_rows",
        "sum_y",
        "sum_pred",
        "sum_y2",
        "sum_pred2",
        "sum_y_pred",
        "unit_id",
        "subject",
        "layer",
        "model_key",
    }
    unit_chunks: list[pd.DataFrame] = []
    n_files_skipped_subject = 0
    for fp in files:
        with np.load(fp, allow_pickle=False) as npz:
            if not req.issubset(set(npz.files)):
                missing = sorted(req - set(npz.files))
                raise ValueError(f"{fp} missing required keys: {missing}")

            subject = int(_npz_scalar(npz, "subject"))
            if subject in exclude_subjects:
                n_files_skipped_subject += 1
                continue

            sentence_id = np.asarray(npz["sentence_id"]).astype(str)
            n_rows = np.asarray(npz["n_rows"]).astype(np.int64)
            unit_id = np.asarray(npz["unit_id"]).astype(np.int32)
            sum_y = np.asarray(npz["sum_y"], dtype=np.float64)
            sum_pred = np.asarray(npz["sum_pred"], dtype=np.float64)
            sum_y2 = np.asarray(npz["sum_y2"], dtype=np.float64)
            sum_pred2 = np.asarray(npz["sum_pred2"], dtype=np.float64)
            sum_y_pred = np.asarray(npz["sum_y_pred"], dtype=np.float64)
            layer = int(_npz_scalar(npz, "layer"))
            model_key = str(_npz_scalar(npz, "model_key"))

        n_sent = int(sentence_id.shape[0])
        n_units = int(unit_id.shape[0])
        if n_rows.shape != (n_sent,):
            raise ValueError(f"{fp} invalid n_rows shape: expected {(n_sent,)}, got {n_rows.shape}")
        for name, arr in {
            "sum_y": sum_y,
            "sum_pred": sum_pred,
            "sum_y2": sum_y2,
            "sum_pred2": sum_pred2,
            "sum_y_pred": sum_y_pred,
        }.items():
            if arr.shape != (n_sent, n_units):
                raise ValueError(
                    f"{fp} invalid {name} shape: expected {(n_sent, n_units)}, got {arr.shape}"
                )

        groups = group_lookup.reindex(sentence_id)
        if groups.isna().any():
            missing = sorted(set(groups[groups.isna()].index.tolist()))
            preview = ", ".join(missing[:5])
            raise ValueError(f"{fp} has sentence_id missing in metadata: {preview}")

        for group_name in ("top", "bottom"):
            mask = (groups == group_name).to_numpy()
            if not mask.any():
                r = np.full(n_units, np.nan, dtype=np.float64)
                n_total = 0
            else:
                n_total = int(np.sum(n_rows[mask]))
                r = _corr_from_sufficient_stats(
                    n_total,
                    np.sum(sum_y[mask], axis=0, dtype=np.float64),
                    np.sum(sum_pred[mask], axis=0, dtype=np.float64),
                    np.sum(sum_y2[mask], axis=0, dtype=np.float64),
                    np.sum(sum_pred2[mask], axis=0, dtype=np.float64),
                    np.sum(sum_y_pred[mask], axis=0, dtype=np.float64),
                )
            unit_chunks.append(pd.DataFrame({
                "subject": np.full(n_units, subject, dtype=np.int16),
                "unit_id": unit_id,
                "model_key": model_key,
                "layer": np.full(n_units, layer, dtype=np.int16),
                "visualness_group": group_name,
                "n_rows": np.full(n_units, n_total, dtype=np.int32),
                "r": r,
            }))

    if not unit_chunks:
        raise ValueError(f"no OOF stats files were loaded from: {oof_stats_path}")
    unit_long = pd.concat(unit_chunks, ignore_index=True)

    layer_counts = unit_long.groupby("model_key")["layer"].nunique(dropna=False)
    bad_layers = layer_counts[layer_counts != 1]
    if not bad_layers.empty:
        details: list[str] = []
        for model_key in bad_layers.index.tolist():
            layers = sorted(unit_long.loc[unit_long["model_key"] == model_key, "layer"].unique().tolist())
            details.append(f"{model_key} -> {layers}")
        raise ValueError(
            "OOF stats include multiple layers per model_key. "
            "Pass only the predetermined best-layer stats. "
            + "; ".join(details)
        )

    pivot = (
        unit_long
        .pivot_table(
            index=["subject", "unit_id", "visualness_group"],
            columns="model_key",
            values="r",
            aggfunc="first",
        )
        .reset_index()
    )
    if llm_key not in pivot.columns or vlm_key not in pivot.columns:
        raise ValueError(f"OOF table must include both model keys: {llm_key} and {vlm_key}")

    core = pivot.dropna(subset=[llm_key, vlm_key]).copy()
    core["z_llm"] = np.arctanh(np.clip(core[llm_key].astype(np.float64), -1 + _EPS, 1 - _EPS))
    core["z_vlm"] = np.arctanh(np.clip(core[vlm_key].astype(np.float64), -1 + _EPS, 1 - _EPS))
    core["A"] = core["z_vlm"] - core["z_llm"]

    wide = (
        core.pivot_table(index=["subject", "unit_id"], columns="visualness_group", values="A", aggfunc="first")
        .reset_index()
        .rename(columns={"top": "A_top", "bottom": "A_bottom"})
    )
    wide["H"] = wide["A_top"] - wide["A_bottom"]
    wide = wide.dropna(subset=["A_top", "A_bottom", "H"])

    subj = (
        wide.groupby("subject", as_index=False)
        .agg(
            n_units=("unit_id", "nunique"),
            A_top_mean=("A_top", "mean"),
            A_bottom_mean=("A_bottom", "mean"),
            H_mean=("H", "mean"),
        )
        .sort_values("subject")
        .reset_index(drop=True)
    )

    q1_stats = {
        "llm_key": llm_key,
        "vlm_key": vlm_key,
        "excluded_subjects": sorted(int(s) for s in exclude_subjects),
        "n_oof_files_seen": int(len(files)),
        "n_oof_files_skipped_excluded_subject": int(n_files_skipped_subject),
        "layers_by_model_key": {
            k: int(v) for k, v in unit_long.groupby("model_key")["layer"].first().to_dict().items()
        },
        "n_units": int(len(wide)),
        "n_subjects": int(len(subj)),
        "mean_A_top": float(subj["A_top_mean"].mean()) if len(subj) else np.nan,
        "mean_A_bottom": float(subj["A_bottom_mean"].mean()) if len(subj) else np.nan,
        "mean_H": float(subj["H_mean"].mean()) if len(subj) else np.nan,
        "test_top_vs_zero": _onesided_signflip(
            subj["A_top_mean"].to_numpy(dtype=np.float64), n_perm=n_perm, seed=seed
        ),
        "test_modulation_H": _onesided_signflip(
            subj["H_mean"].to_numpy(dtype=np.float64), n_perm=n_perm, seed=seed + 1
        ),
    }
    return wide, subj, q1_stats


def plot_figure1_visualness_distribution(metadata: pd.DataFrame, out_dir: Path) -> None:
    """Pooled histogram of article-normalized visualness (V*) by top / middle / bottom."""
    plot_df = metadata[metadata["visualness_group"].isin(["top", "middle", "bottom"])].copy()
    plot_df = plot_df[np.isfinite(plot_df["visualness_z_article"].to_numpy(dtype=np.float64))]
    if plot_df.empty:
        print("Warning: no rows for Figure 1 (need finite visualness_z_article in top/middle/bottom).")
        return

    colors = {"top": "#c44e52", "middle": "#8172b3", "bottom": "#4c72b0"}
    order = ["bottom", "middle", "top"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = np.linspace(
        float(plot_df["visualness_z_article"].min()),
        float(plot_df["visualness_z_article"].max()),
        35,
    )
    for g in order:
        vals = plot_df.loc[plot_df["visualness_group"] == g, "visualness_z_article"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            alpha=0.45,
            label=f"{g} (n={len(vals)})",
            color=colors[g],
            density=True,
        )
    ax.set_xlabel(r"Article-normalized visualness $V_i^\star$ (z within article, eligible sentences)")
    ax.set_ylabel("Density")
    ax.set_title("Visualness distribution and groups (pooled across articles)")
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.5,
        0.01,
        "Groups are assigned within each article (top / middle / bottom), then pooled here.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out_dir / "figure1_visualness_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def plot_figure_simple_high_low_contrast(metadata: pd.DataFrame, out_dir: Path) -> None:
    """Intuitive 2-bar contrast: mean sentence-level Lancaster visualness in low vs high groups.

    Uses ``visualness_mean`` (raw norm), not z-scores. Error bars: SEM across sentences.
    """
    def _group_stats(group: str) -> tuple[float, float, int]:
        v = (
            metadata.loc[metadata["visualness_group"] == group, "visualness_mean"]
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if v.size == 0:
            return (float("nan"), float("nan"), 0)
        m = float(np.mean(v))
        se = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        return (m, se, int(v.size))

    mean_lo, se_lo, n_lo = _group_stats("bottom")
    mean_hi, se_hi, n_hi = _group_stats("top")
    if n_lo == 0 or n_hi == 0:
        print("Warning: need both bottom and top sentences for the simple high/low figure.")
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    x = np.array([0, 1], dtype=np.float64)
    means = np.array([mean_lo, mean_hi], dtype=np.float64)
    ses = np.array([se_lo, se_hi], dtype=np.float64)
    ax.bar(
        x,
        means,
        yerr=ses,
        width=0.55,
        capsize=5,
        color=["#4c72b0", "#c44e52"],
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )
    # Middle group mean as a light reference (descriptive)
    mid = (
        metadata.loc[metadata["visualness_group"] == "middle", "visualness_mean"]
        .dropna()
        .to_numpy(dtype=np.float64)
    )
    if mid.size:
        ax.axhline(
            float(np.mean(mid)),
            color="#8172b3",
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            zorder=1,
            label=rf"Middle group mean (n={len(mid)})",
        )
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Low\n(n={n_lo})", f"High\n(n={n_hi})"],
    )
    ax.set_ylabel("Mean sentence visualness (Lancaster Visual.mean)")
    ax.set_title("High vs low visualness (sentence-level means)")
    ax.set_xlim(-0.6, 1.6)
    d = mean_hi - mean_lo
    fig.text(
        0.5,
        0.02,
        rf"High $-$ Low = {d:.3f}. Groups: within-article bottom vs top, then pooled across articles.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = out_dir / "figure_simple_high_low_visualness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def plot_figure2_advantage_top_bottom(
    subj_df: pd.DataFrame,
    stats: dict,
    out_dir: Path,
    *,
    llm_key: str,
    vlm_key: str,
) -> None:
    """Subject-level $\\bar A_{s,g}$ for bottom vs top; annotate means and permutation p."""
    if subj_df.empty:
        print("Warning: no subject rows for Figure 2.")
        return
    need = {"A_bottom_mean", "A_top_mean", "H_mean"}
    if not need.issubset(set(subj_df.columns)):
        raise ValueError(f"subject summary missing columns: {sorted(need - set(subj_df.columns))}")

    n = len(subj_df)
    x0, x1 = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    for _, row in subj_df.iterrows():
        ax.plot(
            [x0, x1],
            [row["A_bottom_mean"], row["A_top_mean"]],
            color="#cccccc",
            linewidth=0.8,
            zorder=1,
        )
    ax.scatter(
        np.full(n, x0),
        subj_df["A_bottom_mean"].to_numpy(dtype=np.float64),
        s=36,
        color="#4c72b0",
        zorder=3,
        label=r"$\bar{A}_{s,\mathrm{bottom}}$",
    )
    ax.scatter(
        np.full(n, x1),
        subj_df["A_top_mean"].to_numpy(dtype=np.float64),
        s=36,
        color="#c44e52",
        zorder=3,
        label=r"$\bar{A}_{s,\mathrm{top}}$",
    )

    mean_b = float(stats.get("mean_A_bottom", np.nan))
    mean_t = float(stats.get("mean_A_top", np.nan))
    mean_h = float(stats.get("mean_H", np.nan))
    p_h = stats.get("test_modulation_H", {})
    p_val = p_h.get("p_one_sided", np.nan) if isinstance(p_h, dict) else np.nan

    ax.hlines(mean_b, x0 - 0.12, x0 + 0.12, colors="#4c72b0", linestyles="--", linewidth=1.2, alpha=0.9, zorder=2)
    ax.hlines(mean_t, x1 - 0.12, x1 + 0.12, colors="#c44e52", linestyles="--", linewidth=1.2, alpha=0.9, zorder=2)

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["bottom", "top"])
    ax.set_ylabel(r"Normalized mean VLM–LLM advantage")
    ax.set_title(f"{vlm_key} vs {llm_key}")
    ax.legend(frameon=False, loc="best")

    txt = (
        f"mean H      = {mean_h:.4f}\n"
        f"p           = {p_val:.4g}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    p = out_dir / "figure2_vlm_llm_advantage_top_bottom.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)


def run_smoke(metadata: pd.DataFrame) -> None:
    expected = sum(SENTS_PER_ARTICLE)
    if len(metadata) != expected:
        raise RuntimeError(f"smoke failed: expected {expected} metadata rows, got {len(metadata)}")
    by_article = (
        metadata.groupby(["article_id", "visualness_group"]).size().unstack(fill_value=0)
        .reset_index()
        .sort_values("article_id")
    )
    print("\n[smoke] sentence group counts by article:")
    print(by_article.to_string(index=False))
    if (by_article[["top", "bottom"]].sum(axis=1) == 0).any():
        raise RuntimeError("smoke failed: at least one article has no top/bottom sentence")
    print("[smoke] metadata checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 visualness analysis.")
    parser.add_argument("--words_csv", type=Path, default=WORDS_CSV)
    parser.add_argument("--sentences_txt", type=Path, default=SENTENCES_FILE)
    parser.add_argument("--lancaster_csv", type=Path, default=DEFAULT_LANCASTER)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--coverage_threshold", type=float, default=0.75)
    parser.add_argument(
        "--min_content_words",
        type=int,
        default=3,
        help="Exclude sentences with fewer content words before top/bottom grouping (default: 3).",
    )
    parser.add_argument("--min_matched_words", type=int, default=3,
                        help="Minimum Lancaster-matched content words (default: 3).")
    parser.add_argument("--top_fraction", type=float, default=0.2)
    parser.add_argument("--keep_numbers", action="store_true", help="Count numeric tokens as content words.")
    parser.add_argument("--no_bigrams", action="store_true", help="Disable two-word lexicon matching.")
    parser.add_argument("--llm", default="qwen2-7b")
    parser.add_argument("--vlm", default="qwen2-vl-7b")
    parser.add_argument("--global_best_csv", type=Path, default=DEFAULT_FMRI_GLOBAL_BEST)
    parser.add_argument("--best_layer_domain", default="LR_pooled")
    parser.add_argument("--skip_fixed_layers", action="store_true")
    parser.add_argument(
        "--oof_stats",
        type=Path,
        default=None,
        help="Optional OOF stats input: one *_oof_stats.npz file or a directory containing many.",
    )
    parser.add_argument("--oof_csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--exclude_subjects",
        default="",
        help="Comma-separated fMRI subject IDs to exclude from OOF Q1 analysis (default: none).",
    )
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip writing figure outputs (metadata and tables still run).",
    )
    args = parser.parse_args()

    if not (0 < args.top_fraction < 0.5):
        raise ValueError("--top_fraction must be in (0, 0.5)")
    if not (0 <= args.coverage_threshold <= 1):
        raise ValueError("--coverage_threshold must be in [0, 1]")
    if args.min_content_words < 0:
        raise ValueError("--min_content_words must be >= 0")
    if args.min_matched_words < 0:
        raise ValueError("--min_matched_words must be >= 0")
    if args.n_perm <= 0:
        raise ValueError("--n_perm must be > 0")
    exclude_subjects = _parse_subject_ids(args.exclude_subjects)

    t0 = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 5: Visualness analysis")
    print(f"  words_csv          : {args.words_csv}")
    print(f"  sentences_txt      : {args.sentences_txt}")
    print(f"  lancaster_csv      : {args.lancaster_csv}")
    print(f"  out_dir            : {out_dir}")
    print(f"  coverage_threshold : {args.coverage_threshold}")
    print(f"  min_content_words  : {args.min_content_words}")
    print(f"  min_matched_words  : {args.min_matched_words}")
    print(f"  top_fraction       : {args.top_fraction}")
    print(f"  use_bigrams        : {not args.no_bigrams}")
    print(f"  keep_numbers       : {args.keep_numbers}")
    print(f"  exclude_subjects   : {sorted(exclude_subjects)}")
    print("=" * 72)

    lexicon = load_lancaster(args.lancaster_csv)
    metadata = build_visualness_metadata(
        args.words_csv,
        args.sentences_txt,
        lexicon,
        coverage_threshold=args.coverage_threshold,
        min_content_words=args.min_content_words,
        min_matched_words=args.min_matched_words,
        top_fraction=args.top_fraction,
        keep_numbers=args.keep_numbers,
        use_bigrams=(not args.no_bigrams),
    )
    meta_csv = out_dir / "sentence_visualness_groups.csv"
    metadata.to_csv(meta_csv, index=False, float_format="%.6f")
    print(f"Saved: {meta_csv}")

    cfg = {
        "top_fraction": args.top_fraction,
        "coverage_threshold": args.coverage_threshold,
        "min_content_words": int(args.min_content_words),
        "min_matched_words": int(args.min_matched_words),
        "lexicon_file": str(args.lancaster_csv),
        "lexicon_score_column": "Visual.mean",
        "lexicon_term_column": "Word",
        "use_bigrams": bool(not args.no_bigrams),
        "keep_numbers_as_content": bool(args.keep_numbers),
        "exclude_subjects": sorted(int(s) for s in exclude_subjects),
        "llm_fallback": False,
        "sentence_ordering": "article_then_sentence",
        "random_seed": int(args.seed),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = out_dir / "visualness_analysis_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved: {cfg_path}")

    if not args.no_figures:
        plot_figure1_visualness_distribution(metadata, out_dir)
        plot_figure_simple_high_low_contrast(metadata, out_dir)

    if not args.skip_fixed_layers:
        fixed_df = build_fixed_layers(
            args.global_best_csv,
            llm_key=args.llm,
            vlm_key=args.vlm,
            domain=args.best_layer_domain,
        )
        fixed_csv = out_dir / "fixed_layers.csv"
        fixed_df.to_csv(fixed_csv, index=False)
        print(f"Saved: {fixed_csv}")
        unresolved = fixed_df[fixed_df["layer"] < 0]
        if not unresolved.empty:
            models = ", ".join(unresolved["model_key"].astype(str).tolist())
            print(
                f"Warning: unresolved fixed layer for model(s): {models}. "
                "Check --llm/--vlm and --global_best_csv."
            )

    oof_input = args.oof_stats
    if args.oof_csv is not None:
        if oof_input is not None:
            raise ValueError("use only one of --oof_stats or --oof_csv")
        oof_input = args.oof_csv
        print("Warning: --oof_csv is deprecated; interpreting it as --oof_stats.")

    if oof_input is not None:
        unit_df, subj_df, stats = run_oof_analysis(
            oof_input,
            metadata,
            llm_key=args.llm,
            vlm_key=args.vlm,
            n_perm=args.n_perm,
            seed=args.seed,
            exclude_subjects=exclude_subjects,
        )
        unit_csv = out_dir / "visualness_unit_advantage.csv"
        subj_csv = out_dir / "visualness_subject_summary.csv"
        stats_json = out_dir / "visualness_q1_stats.json"
        unit_df.to_csv(unit_csv, index=False, float_format="%.6f")
        subj_df.to_csv(subj_csv, index=False, float_format="%.6f")
        stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved: {unit_csv}")
        print(f"Saved: {subj_csv}")
        print(f"Saved: {stats_json}")
        if not args.no_figures:
            plot_figure2_advantage_top_bottom(
                subj_df, stats, out_dir, llm_key=args.llm, vlm_key=args.vlm
            )

    if args.smoke_test:
        run_smoke(metadata)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
