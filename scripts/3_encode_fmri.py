#!/usr/bin/env python3
"""
Phase 3: Per-vertex ridge regression of model attention against BOLD matrices.

"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
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
    FMRI_SUBJECTS,
    GIFTI_TEMPLATE,
    N_ARTICLES,
    N_VERTICES,
    RESULTS_DIR,
    RIDGE_ALPHAS,
    SENTS_PER_ARTICLE,
    TR,
    HRF_DELAY,
    load_words_list,
)

FMRI_DIR = RESULTS_DIR / "fmri"
FMRI_OOF_DIR = FMRI_DIR / "oof"
FMRI_CACHE_DIR = RESULTS_DIR / "fmri_cache"
CUMS = np.cumsum(SENTS_PER_ARTICLE)   # [31,62,90,118,148]

_VALID_SID_RE = r"^t\.\d+\.\d+$"


# ---------------------------------------------------------------------------
# Event / surface loading
# ---------------------------------------------------------------------------


def _first_valid_article(df: pd.DataFrame) -> int | None:
    sids = df["SentenceID"].astype(str)
    valid = sids[sids.str.match(_VALID_SID_RE)]
    if valid.empty:
        return None
    return int(valid.iloc[0].split(".")[1])


def load_surface_and_events(
    subj_id: int, hemi: str
) -> tuple[list[np.ndarray], pd.DataFrame] | None:
    """Load fsaverage5 BOLD and events for all 5 articles.

    Returns
    -------
    surf : list of 5 (N_VERTICES, n_trs) float32 arrays (z-scored)
    events : concatenated DataFrame with `article` column
    """
    sub = f"{subj_id:02d}"
    surf: list[np.ndarray] = []
    events_all: list[pd.DataFrame] = []

    for article in range(1, N_ARTICLES + 1):
        matched_run = None
        for run in range(1, 6):
            ev_path = EVENTS_TEMPLATE.format(sub=sub, run=run)
            if not Path(ev_path).exists():
                continue
            df = pd.read_csv(ev_path, delimiter="\t")
            df = df[df["onset"].notna()].reset_index(drop=True)
            if _first_valid_article(df) == article:
                matched_run = run
                df["article"] = article
                events_all.append(df)
                break
        if matched_run is None:
            print(f"  sub-{sub}: article {article} not found — subject skipped")
            return None

        gii_path = GIFTI_TEMPLATE.format(sub=sub, run=matched_run, hemi=hemi)
        if not Path(gii_path).exists():
            print(f"  sub-{sub}: GIFTI not found: {gii_path} — subject skipped")
            return None

        fmri_gii = nib.load(gii_path)
        fmri_data = np.column_stack(
            [arr.data for arr in fmri_gii.darrays]
        ).astype(np.float32)                           # (N_VERTICES, n_trs)
        fmri_data = np.nan_to_num(
            zscore(fmri_data, axis=0, nan_policy="omit").astype(np.float32)
        )
        surf.append(fmri_data)

    events = pd.concat(events_all, ignore_index=True)
    return surf, events


# ---------------------------------------------------------------------------
# BOLD matrix construction 
# ---------------------------------------------------------------------------


def _snt_id(article: int, sent_i: int) -> str:
    return f"t.{article:02d}.{sent_i:02d}"


def build_bold_vectors(
    surf: list[np.ndarray],
    events: pd.DataFrame,
    words_list: list[int],
) -> list[np.ndarray]:
    """Build per-article BOLD targets in article-sequential order.

    Returns
    -------
    y_by_article : list of 5 arrays
        Each element has shape (total_pairs_in_article, N_VERTICES).

    The BOLD matrix construction 
        scan = ceil((next_fixation_onset_ms / 1000 + HRF_DELAY) / TR)
        fmri_snt[row, col] = surface_data[vertex, scan]
    """
    y_by_article: list[np.ndarray] = []

    for article in range(1, N_ARTICLES + 1):
        article_rows: list[np.ndarray] = []
        n_snts = SENTS_PER_ARTICLE[article - 1]
        surf_art = surf[article - 1]          # (N_VERTICES, n_trs)
        n_trs = surf_art.shape[1]

        for sent_i in range(1, n_snts + 1):
            sid = (sent_i - 1) if article == 1 else CUMS[article - 2] + (sent_i - 1)
            sent_len = words_list[sid]
            tril_r, tril_c = np.tril_indices(sent_len, k=-1)

            # fmri_snt: (sent_len, sent_len, N_VERTICES)
            fmri_snt = np.zeros((sent_len, sent_len, N_VERTICES), dtype=np.float32)

            event = events[
                events["SentenceID"].astype(str) == _snt_id(article, sent_i)
            ].copy()
            aoi = pd.to_numeric(event["CURRENT_FIX_INTEREST_AREA_ID"], errors="coerce")
            event = event[aoi.notna()].reset_index(drop=True)

            for ind in range(len(event) - 1):
                row = int(event.iloc[ind]["CURRENT_FIX_INTEREST_AREA_ID"]) - 1
                col = int(event.iloc[ind + 1]["CURRENT_FIX_INTEREST_AREA_ID"]) - 1
                scan = int(
                    np.ceil(
                        (event.iloc[ind + 1]["onset"] / 1000.0 + HRF_DELAY) / TR
                    )
                )
                while scan >= n_trs:
                    scan -= 1
                if 0 <= row < sent_len and 0 <= col < sent_len and scan >= 0:
                    fmri_snt[row, col] = surf_art[:, scan]

            # Lower triangle: (n_pairs, N_VERTICES)
            lt_block = fmri_snt[tril_r, tril_c]        # (n_pairs, N_VERTICES)
            article_rows.append(lt_block)

        y_by_article.append(np.concatenate(article_rows, axis=0))
    return y_by_article


def _y_cache_path(cache_dir: Path, subj_id: int, hemi: str) -> Path:
    return cache_dir / f"sub{subj_id:02d}_hemi{hemi}_y_by_article.npz"


def load_or_build_y_by_article(
    subj_id: int,
    hemi: str,
    words_list: list[int],
    cache_dir: Path,
    *,
    use_cache: bool,
) -> tuple[list[np.ndarray], str] | None:
    """Load per-subject/hemi Y cache or build it once from raw data.

    Returns
    -------
    (y_by_article, cache_status) where cache_status is "hit", "miss", or
    "disabled". Returns None when subject data is unavailable.
    """
    cache_path = _y_cache_path(cache_dir, subj_id, hemi)
    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        y_by_article = [data[f"art{i}"].astype(np.float32) for i in range(N_ARTICLES)]
        return y_by_article, "hit"

    result = load_surface_and_events(subj_id, hemi)
    if result is None:
        return None
    surf, events = result
    y_by_article = build_bold_vectors(surf, events, words_list)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{f"art{i}": y_by_article[i] for i in range(N_ARTICLES)})
        return y_by_article, "miss"

    return y_by_article, "disabled"


# ---------------------------------------------------------------------------
# Attention flattening  (same as 2_encode_saccade.py)
# ---------------------------------------------------------------------------


def build_X(
    layer_attn: np.ndarray, words_list: list[int]
) -> list[np.ndarray]:
    """Flatten attention to per-article lower-triangle feature matrices.
    """
    article_parts: list[np.ndarray] = []
    sent_idx = 0
    for art_idx, n_sents in enumerate(SENTS_PER_ARTICLE):
        art_rows: list[np.ndarray] = []
        for s in range(n_sents):
            w = words_list[sent_idx]
            tril_r, tril_c = np.tril_indices(w, k=-1)
            feat = layer_attn[art_idx, s, :, :w, :w][:, tril_r, tril_c].T  # (n_pairs, n_heads)
            art_rows.append(feat)
            sent_idx += 1
        article_parts.append(np.concatenate(art_rows, axis=0))

    X_full = np.concatenate(article_parts, axis=0)
    X_full = np.nan_to_num(zscore(X_full.flatten(), nan_policy="omit")).reshape(
        X_full.shape
    )

    sizes = [a.shape[0] for a in article_parts]
    offsets = np.cumsum([0] + sizes)
    X_by_article = [
        X_full[offsets[i]:offsets[i + 1]] for i in range(N_ARTICLES)
    ]
    return X_by_article


def build_sentence_offsets_by_article(words_list: list[int]) -> list[np.ndarray]:
    """Return per-article cumulative row offsets for sentence lower-triangle blocks.

    For each article, offsets length is ``n_sents + 1`` and offsets[-1] equals the
    number of lower-triangle rows in that article after flattening with ``k=-1``.
    """
    offsets_by_article: list[np.ndarray] = []
    sent_idx = 0
    for n_sents in SENTS_PER_ARTICLE:
        offsets = [0]
        for _ in range(n_sents):
            w = int(words_list[sent_idx])
            n_pairs = (w * (w - 1)) // 2
            offsets.append(offsets[-1] + n_pairs)
            sent_idx += 1
        offsets_by_article.append(np.asarray(offsets, dtype=np.int32))
    return offsets_by_article


def _validate_article_row_counts(
    X_by_article: list[np.ndarray],
    y_by_article: list[np.ndarray],
    sent_offsets_by_article: list[np.ndarray],
) -> tuple[bool, str]:
    """Verify X/Y rows match expected sentence-block lengths for every article."""
    for art_idx in range(N_ARTICLES):
        x_rows = int(X_by_article[art_idx].shape[0])
        y_rows = int(y_by_article[art_idx].shape[0])
        expected = int(sent_offsets_by_article[art_idx][-1])
        if x_rows != y_rows:
            return False, (
                f"article {art_idx + 1}: X/Y row mismatch "
                f"(X={x_rows}, y={y_rows})"
            )
        if x_rows != expected:
            return False, (
                f"article {art_idx + 1}: row count does not match sentence offsets "
                f"(rows={x_rows}, expected={expected})"
            )
    return True, ""


def _sentence_blocks_from_offsets(offsets: np.ndarray) -> list[np.ndarray]:
    """Build row-index blocks for each sentence in one article."""
    return [
        np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int32)
        for i in range(len(offsets) - 1)
    ]


def _group_sentences_by_block_size(sent_blocks: list[np.ndarray]) -> list[np.ndarray]:
    """Group sentence indices by lower-triangle row count."""
    groups: dict[int, list[int]] = {}
    for sent_idx, block in enumerate(sent_blocks):
        groups.setdefault(int(block.size), []).append(sent_idx)
    return [np.asarray(groups[k], dtype=np.int32) for k in sorted(groups)]


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
# Per-vertex regression worker
# ---------------------------------------------------------------------------


def _process_vertex(
    v: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Ridge regression for one vertex. Returns Pearson r on test set.

    Returns NaN when the test target or prediction is constant, avoiding
    scipy's ConstantInputWarning while preserving a valid-vertex mask for QC.
    """
    yt = y_train[:, v]
    yv = y_test[:, v]
    clf = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, yt)
    pred = clf.predict(X_test)
    if np.nanstd(yv) == 0 or np.nanstd(pred) == 0:
        return float("nan")
    r, _ = pearsonr(pred, yv)
    return float(r)


def _vertex_chunk_bounds(n_vertices: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split [0, n_vertices) into contiguous half-open ranges for parallel workers."""
    n_chunks = max(1, min(n_vertices, n_chunks))
    bounds: list[tuple[int, int]] = []
    base = n_vertices // n_chunks
    rem = n_vertices % n_chunks
    start = 0
    for i in range(n_chunks):
        w = base + (1 if i < rem else 0)
        end = start + w
        bounds.append((start, end))
        start = end
    return bounds


def _process_vertex_chunk(
    v_start: int,
    v_end: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Same math as ``_process_vertex`` for vertices ``v_start .. v_end-1`` (sequential)."""
    out = np.empty(v_end - v_start, dtype=np.float32)
    for j, v in enumerate(range(v_start, v_end)):
        out[j] = np.float32(_process_vertex(v, X_train, X_test, y_train, y_test))
    return out


def _corr_from_block_predictions(
    y_true_block: np.ndarray,
    pred_block: np.ndarray,
) -> np.ndarray:
    """Compute Pearson r per column with the same constant-signal handling as `_process_vertex`."""
    out = np.empty(y_true_block.shape[1], dtype=np.float32)
    for j in range(y_true_block.shape[1]):
        yv = y_true_block[:, j]
        pred = pred_block[:, j]
        if np.nanstd(yv) == 0 or np.nanstd(pred) == 0:
            out[j] = np.float32(np.nan)
            continue
        r, _ = pearsonr(pred, yv)
        out[j] = np.float32(r)
    return out


def _fit_multi_target_ridge_chunk(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one multi-output RidgeCV block while preserving per-target alpha selection."""
    clf = RidgeCV(
        alphas=RIDGE_ALPHAS,
        fit_intercept=True,
        alpha_per_target=True,
    ).fit(X_train, y_train_block)
    pred_block = np.asarray(clf.predict(X_test), dtype=np.float64)
    alpha_arr = np.asarray(clf.alpha_, dtype=np.float64)
    return pred_block, alpha_arr


def _predict_vertex_chunk_multitarget(
    v_start: int,
    v_end: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Predict one vertex chunk via one multi-target RidgeCV fit."""
    pred_block, _ = _fit_multi_target_ridge_chunk(
        X_train,
        X_test,
        y_train[:, v_start:v_end],
    )
    return np.asarray(pred_block, dtype=np.float32)


def _process_vertex_chunk_multitarget(
    v_start: int,
    v_end: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    """Process a vertex chunk via one multi-target RidgeCV fit with per-target alpha selection."""
    pred_block, _ = _fit_multi_target_ridge_chunk(
        X_train,
        X_test,
        y_train[:, v_start:v_end],
    )
    return _corr_from_block_predictions(y_test[:, v_start:v_end], pred_block)


def _prepare_corr_terms(
    pred_block: np.ndarray,
    y_true_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare centered correlation terms and observed r for permutation tests."""
    pred_center = np.asarray(pred_block, dtype=np.float32)
    y_center = np.asarray(y_true_block, dtype=np.float32)

    pred_mean = pred_center.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = y_center.mean(axis=0, dtype=np.float64).astype(np.float32)
    pred_center = pred_center - pred_mean[None, :]
    y_center = y_center - y_mean[None, :]

    pred_ss = np.sum(pred_center * pred_center, axis=0, dtype=np.float64)
    y_ss = np.sum(y_center * y_center, axis=0, dtype=np.float64)
    denom = np.sqrt(pred_ss * y_ss)
    valid_mask = denom > 0

    num_obs = np.sum(pred_center * y_center, axis=0, dtype=np.float64)
    r_obs = np.full(pred_center.shape[1], np.nan, dtype=np.float32)
    r_obs[valid_mask] = (num_obs[valid_mask] / denom[valid_mask]).astype(np.float32)
    return pred_center, y_center, denom, valid_mask, r_obs


def _build_sentence_contrib_tensors(
    pred_center: np.ndarray,
    y_center: np.ndarray,
    sent_blocks: list[np.ndarray],
) -> list[np.ndarray]:
    """Precompute sentence-to-sentence dot-product tensors for valid block sizes.

    Returns a list of tensors. Each tensor corresponds to one block-size group
    and has shape (n_group_sents, n_group_sents, n_vertices), where:
    tensor[src, dst, v] = sum_r pred_center[src_rows, v] * y_center[dst_rows, v]
    """
    contrib_tensors: list[np.ndarray] = []
    groups = _group_sentences_by_block_size(sent_blocks)
    for sent_group in groups:
        n_group = int(sent_group.size)
        first_rows = sent_blocks[int(sent_group[0])]
        block_len = int(first_rows.size)
        pred_stack = np.empty((n_group, block_len, pred_center.shape[1]), dtype=np.float32)
        y_stack = np.empty_like(pred_stack)
        for i, sent_idx in enumerate(sent_group):
            rows = sent_blocks[int(sent_idx)]
            pred_stack[i] = pred_center[rows]
            y_stack[i] = y_center[rows]
        tensor = np.einsum("irv,jrv->ijv", pred_stack, y_stack, optimize=True).astype(np.float32)
        contrib_tensors.append(tensor)
    return contrib_tensors


def _permuted_r_batch_from_contrib(
    contrib_tensors: list[np.ndarray],
    denom: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Compute a batch of permutation r vectors from precomputed contribution tensors."""
    n_vertices = int(denom.size)
    numer = np.zeros((batch_size, n_vertices), dtype=np.float32)
    for group_idx, tensor in enumerate(contrib_tensors):
        m = int(tensor.shape[0])
        if m == 1:
            numer += tensor[0, 0][None, :]
            continue

        rng = np.random.default_rng(seed + group_idx * 2_147_483_647)
        perms = np.empty((batch_size, m), dtype=np.int32)
        for b in range(batch_size):
            perms[b] = rng.permutation(m)
        dst = np.arange(m, dtype=np.int32)[None, :]
        picked = tensor[perms, dst, :]  # (batch, m, vertices)
        numer += picked.sum(axis=1, dtype=np.float32)

    r_batch = np.full((batch_size, n_vertices), np.nan, dtype=np.float32)
    r_batch[:, valid_mask] = numer[:, valid_mask] / denom[None, valid_mask]
    return r_batch


def _compute_perm_batch_agg(
    batch_start: int,
    batch_size: int,
    fold_perm_states: list[tuple[list[np.ndarray], np.ndarray, np.ndarray, int]],
) -> tuple[int, np.ndarray]:
    """Compute one aggregated permutation batch across folds."""
    n_vertices = int(fold_perm_states[0][1].size)
    agg_sum = np.zeros((batch_size, n_vertices), dtype=np.float32)
    agg_n = np.zeros((batch_size, n_vertices), dtype=np.int16)

    for contrib_tensors, denom, valid_mask, fold_seed in fold_perm_states:
        fold_batch_seed = int(fold_seed + batch_start * 104_729)
        r_fold_batch = _permuted_r_batch_from_contrib(
            contrib_tensors,
            denom,
            valid_mask,
            batch_size,
            seed=fold_batch_seed,
        )
        valid = np.isfinite(r_fold_batch)
        agg_sum[valid] += r_fold_batch[valid]
        agg_n[valid] += 1

    r_agg_batch = np.divide(
        agg_sum,
        agg_n,
        out=np.full((batch_size, n_vertices), np.nan, dtype=np.float32),
        where=agg_n > 0,
    )
    return batch_start, r_agg_batch


def summarize_corr_qc(corr_raw: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Compute QC metrics for a subject-level surface map."""
    nonconstant_test_mask = np.nanstd(y_test, axis=0) > 0
    valid_corr_mask = np.isfinite(corr_raw)
    valid_corr = corr_raw[valid_corr_mask]

    hist_edges = np.linspace(-1.0, 1.0, 41)
    hist_counts, hist_edges = np.histogram(valid_corr, bins=hist_edges)

    return {
        "n_vertices": int(corr_raw.size),
        "n_nonconstant_test_vertices": int(nonconstant_test_mask.sum()),
        "n_constant_test_vertices": int((~nonconstant_test_mask).sum()),
        "n_valid_corr_vertices": int(valid_corr_mask.sum()),
        "n_invalid_corr_vertices": int((~valid_corr_mask).sum()),
        "mean_valid_r": float(np.nanmean(valid_corr)) if valid_corr.size else float("nan"),
        "max_valid_r": float(np.nanmax(valid_corr)) if valid_corr.size else float("nan"),
        "min_valid_r": float(np.nanmin(valid_corr)) if valid_corr.size else float("nan"),
        "pct_positive_valid_r": float((valid_corr > 0).mean()) if valid_corr.size else float("nan"),
        "hist_counts": hist_counts.astype(np.int32),
        "hist_edges": hist_edges.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Optional OOF export for Phase 5 visualness analysis
# ---------------------------------------------------------------------------


def _summarize_oof_fold_by_sentence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sent_offsets: np.ndarray,
    *,
    article_id: int,
) -> dict[str, np.ndarray]:
    """Aggregate one held-out article into per-sentence per-vertex sufficient stats."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"OOF y_pred/y_true shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"OOF arrays must be 2D (rows x vertices), got shape={y_true.shape}")
    if sent_offsets.ndim != 1 or len(sent_offsets) < 2:
        raise ValueError("sent_offsets must be a 1D array with length >= 2")
    if int(sent_offsets[0]) != 0 or int(sent_offsets[-1]) != int(y_true.shape[0]):
        raise ValueError(
            f"OOF offsets/rows mismatch: offsets[0]={int(sent_offsets[0])}, "
            f"offsets[-1]={int(sent_offsets[-1])}, rows={int(y_true.shape[0])}"
        )

    n_sents = int(len(sent_offsets) - 1)
    n_vertices = int(y_true.shape[1])
    sentence_ids = np.asarray([_snt_id(article_id, i + 1) for i in range(n_sents)])
    article_arr = np.full(n_sents, article_id, dtype=np.int16)
    fold_arr = np.full(n_sents, article_id, dtype=np.int16)
    n_rows = np.diff(sent_offsets).astype(np.int32)

    sum_y = np.empty((n_sents, n_vertices), dtype=np.float32)
    sum_pred = np.empty((n_sents, n_vertices), dtype=np.float32)
    sum_y2 = np.empty((n_sents, n_vertices), dtype=np.float32)
    sum_pred2 = np.empty((n_sents, n_vertices), dtype=np.float32)
    sum_yp = np.empty((n_sents, n_vertices), dtype=np.float32)

    for sent_i in range(n_sents):
        lo = int(sent_offsets[sent_i])
        hi = int(sent_offsets[sent_i + 1])
        yt = y_true[lo:hi].astype(np.float32, copy=False)
        yp = y_pred[lo:hi].astype(np.float32, copy=False)
        sum_y[sent_i] = np.sum(yt, axis=0, dtype=np.float64).astype(np.float32)
        sum_pred[sent_i] = np.sum(yp, axis=0, dtype=np.float64).astype(np.float32)
        sum_y2[sent_i] = np.einsum("ij,ij->j", yt, yt, dtype=np.float64).astype(np.float32)
        sum_pred2[sent_i] = np.einsum("ij,ij->j", yp, yp, dtype=np.float64).astype(np.float32)
        sum_yp[sent_i] = np.einsum("ij,ij->j", yt, yp, dtype=np.float64).astype(np.float32)

    return {
        "sentence_id": sentence_ids,
        "article_id": article_arr,
        "fold": fold_arr,
        "n_rows": n_rows,
        "sum_y": sum_y,
        "sum_pred": sum_pred,
        "sum_y2": sum_y2,
        "sum_pred2": sum_pred2,
        "sum_y_pred": sum_yp,
    }


def _concat_oof_sentence_payloads(
    payloads: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate per-fold sentence stats into one subject/hemi/layer payload."""
    if not payloads:
        raise ValueError("empty OOF payload list")
    keys = payloads[0].keys()
    return {k: np.concatenate([p[k] for p in payloads], axis=0) for k in keys}


def export_oof_sentence_stats_npz(
    out_path: Path,
    payload: dict[str, np.ndarray],
    *,
    model_key: str,
    layer: int,
    hemi: str,
    subj_id: int,
) -> None:
    """Save compact per-sentence OOF sufficient stats for one subject/hemi/layer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    }
    if not req.issubset(set(payload.keys())):
        raise ValueError(f"OOF payload missing keys: {sorted(req - set(payload.keys()))}")

    n_sents = int(payload["sentence_id"].shape[0])
    unit_id = np.arange(N_VERTICES, dtype=np.int32) + (0 if hemi == "L" else N_VERTICES)
    for key in ("sum_y", "sum_pred", "sum_y2", "sum_pred2", "sum_y_pred"):
        arr = payload[key]
        if arr.shape != (n_sents, N_VERTICES):
            raise ValueError(
                f"invalid shape for '{key}': expected {(n_sents, N_VERTICES)}, got {arr.shape}"
            )

    np.savez_compressed(
        out_path,
        sentence_id=payload["sentence_id"],
        article_id=payload["article_id"].astype(np.int16),
        fold=payload["fold"].astype(np.int16),
        n_rows=payload["n_rows"].astype(np.int32),
        sum_y=payload["sum_y"].astype(np.float32),
        sum_pred=payload["sum_pred"].astype(np.float32),
        sum_y2=payload["sum_y2"].astype(np.float32),
        sum_pred2=payload["sum_pred2"].astype(np.float32),
        sum_y_pred=payload["sum_y_pred"].astype(np.float32),
        unit_id=unit_id,
        subject=np.asarray([int(subj_id)], dtype=np.int16),
        layer=np.asarray([int(layer)], dtype=np.int16),
        hemi=np.asarray([hemi], dtype="<U1"),
        model_key=np.asarray([str(model_key)]),
    )


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------


def process_subject_with_y(
    subj_id: int,
    layer: int,
    hemi: str,
    model_key: str,
    X_by_article: list[np.ndarray],
    y_by_article: list[np.ndarray],
    sent_offsets_by_article: list[np.ndarray],
    cache_status: str,
    n_jobs: int,
    *,
    single_fold: bool,
    test_article: int,
    export_oof: bool,
    oof_path: Path | None,
    oof_overwrite: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]] | None:
    """Run fMRI ridge regression using preloaded ``y_by_article`` (one load per subject/hemi)."""
    t0 = time.time()
    sub = f"{subj_id:02d}"

    ok_rows, row_msg = _validate_article_row_counts(
        X_by_article, y_by_article, sent_offsets_by_article
    )
    if not ok_rows:
        print(f"  sub-{sub}: {row_msg} — skipped")
        return None

    n_parallel = effective_n_jobs(n_jobs)
    chunk_bounds = _vertex_chunk_bounds(N_VERTICES, n_parallel)

    fold_corrs: list[np.ndarray] = []
    fold_indices = [test_article - 1] if single_fold else list(range(N_ARTICLES))
    if export_oof:
        if oof_path is None:
            raise ValueError("oof_path is required when export_oof=True")
        if oof_path.exists() and oof_overwrite:
            oof_path.unlink()
        fold_payloads: list[dict[str, np.ndarray]] = []
    for test_art in fold_indices:
        X_test = X_by_article[test_art]
        y_test = y_by_article[test_art]
        X_train = np.concatenate(
            [X_by_article[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )
        y_train = np.concatenate(
            [y_by_article[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )

        if export_oof:
            pred_chunks = Parallel(n_jobs=n_jobs)(
                delayed(_predict_vertex_chunk_multitarget)(
                    lo, hi, X_train, X_test, y_train
                )
                for lo, hi in chunk_bounds
            )
            pred_fold = np.concatenate(pred_chunks, axis=1).astype(np.float32)
            corr_fold = _corr_from_block_predictions(y_test, pred_fold).astype(np.float32)
            fold_payloads.append(
                _summarize_oof_fold_by_sentence(
                    y_test,
                    pred_fold,
                    sent_offsets_by_article[test_art],
                    article_id=(test_art + 1),
                )
            )
        else:
            chunk_out = Parallel(n_jobs=n_jobs)(
                delayed(_process_vertex_chunk_multitarget)(
                    lo, hi, X_train, X_test, y_train, y_test
                )
                for lo, hi in chunk_bounds
            )
            corr_fold = np.concatenate(chunk_out).astype(np.float32)
        fold_corrs.append(corr_fold)

    corr_stack = np.stack(fold_corrs, axis=0)  # (n_folds, N_VERTICES)
    valid_counts = np.isfinite(corr_stack).sum(axis=0)
    corr_sum = np.nansum(corr_stack, axis=0)
    corr_raw = np.divide(
        corr_sum,
        valid_counts,
        out=np.full_like(corr_sum, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    ).astype(np.float32)

    if single_fold:
        y_test_all = y_by_article[test_article - 1]
    else:
        y_test_all = np.concatenate(y_by_article, axis=0)
    qc = summarize_corr_qc(corr_raw, y_test_all)
    qc["mean_valid_folds_per_vertex"] = float(np.mean(valid_counts))
    qc["min_valid_folds_per_vertex"] = int(valid_counts.min())
    qc["max_valid_folds_per_vertex"] = int(valid_counts.max())
    qc["n_eval_folds"] = int(len(fold_indices))
    if export_oof:
        if not fold_payloads:
            raise RuntimeError("export_oof=True but no fold stats were collected")
        export_oof_sentence_stats_npz(
            oof_path,
            _concat_oof_sentence_payloads(fold_payloads),
            model_key=model_key,
            layer=layer,
            hemi=hemi,
            subj_id=subj_id,
        )

    corr_arr = np.nan_to_num(corr_raw, nan=0.0)
    print(
        f"  sub-{sub} layer{layer:02d} hemi-{hemi}: "
        f"mean_valid_r={qc['mean_valid_r']:.4f}  "
        f"max_valid_r={qc['max_valid_r']:.4f}  "
        f"valid={qc['n_valid_corr_vertices']}/{qc['n_vertices']}  "
        f"nonconstant_test={qc['n_nonconstant_test_vertices']}  "
        f"mean_valid_folds={qc['mean_valid_folds_per_vertex']:.2f}  "
        f"oof={'on' if export_oof else 'off'}  "
        f"cache={cache_status}  "
        f"({time.time()-t0:.1f}s)"
    )
    return corr_arr, qc


def process_subject_with_y_null(
    subj_id: int,
    layer: int,
    hemi: str,
    X_by_article: list[np.ndarray],
    y_by_article: list[np.ndarray],
    sent_offsets_by_article: list[np.ndarray],
    cache_status: str,
    n_jobs: int,
    *,
    single_fold: bool,
    test_article: int,
    n_perm: int,
    null_seed: int,
    null_batch: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int], dict[str, np.ndarray]] | None:
    """Run ridge once per fold and optional sentence-block null test on held-out folds."""
    t0 = time.time()
    sub = f"{subj_id:02d}"

    ok_rows, row_msg = _validate_article_row_counts(
        X_by_article, y_by_article, sent_offsets_by_article
    )
    if not ok_rows:
        print(f"  sub-{sub}: {row_msg} — skipped")
        return None

    n_parallel = effective_n_jobs(n_jobs)
    chunk_bounds = _vertex_chunk_bounds(N_VERTICES, n_parallel)
    fold_indices = [test_article - 1] if single_fold else list(range(N_ARTICLES))

    fold_corrs: list[np.ndarray] = []
    fold_perm_states: list[tuple[list[np.ndarray], np.ndarray, np.ndarray, int]] = []
    fold_seeds: list[int] = []
    checked_obs_equivalence = False

    base_seed = (
        int(null_seed)
        + int(subj_id) * 10_000_019
        + int(layer) * 1_000_003
        + (0 if hemi == "L" else 50_000_017)
    ) % (2**32 - 1)

    for fold_pos, test_art in enumerate(fold_indices):
        X_test = X_by_article[test_art]
        y_test = y_by_article[test_art]
        X_train = np.concatenate(
            [X_by_article[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )
        y_train = np.concatenate(
            [y_by_article[i] for i in range(N_ARTICLES) if i != test_art], axis=0
        )

        pred_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_predict_vertex_chunk_multitarget)(
                lo, hi, X_train, X_test, y_train
            )
            for lo, hi in chunk_bounds
        )
        pred_fold = np.concatenate(pred_chunks, axis=1).astype(np.float32)
        corr_fold = _corr_from_block_predictions(y_test, pred_fold).astype(np.float32)

        pred_center, y_center, denom, valid_mask, r_obs_vec = _prepare_corr_terms(
            pred_fold, y_test
        )
        if not checked_obs_equivalence:
            finite = np.isfinite(corr_fold) & np.isfinite(r_obs_vec)
            if finite.any():
                max_diff = float(np.max(np.abs(corr_fold[finite] - r_obs_vec[finite])))
                if max_diff > 1e-5:
                    raise ValueError(
                        f"Observed-r mismatch between legacy and vectorized paths: max diff={max_diff:.6g}"
                    )
            checked_obs_equivalence = True

        sent_blocks = _sentence_blocks_from_offsets(sent_offsets_by_article[test_art])
        contrib_tensors = _build_sentence_contrib_tensors(
            pred_center,
            y_center,
            sent_blocks,
        )
        fold_corrs.append(corr_fold)
        fold_seed = (base_seed + fold_pos * 7919) % (2**32 - 1)
        fold_seeds.append(int(fold_seed))
        fold_perm_states.append((contrib_tensors, denom.astype(np.float32), valid_mask, int(fold_seed)))

    corr_stack = np.stack(fold_corrs, axis=0)  # (n_folds, N_VERTICES)
    valid_counts = np.isfinite(corr_stack).sum(axis=0)
    corr_sum = np.nansum(corr_stack, axis=0)
    corr_raw = np.divide(
        corr_sum,
        valid_counts,
        out=np.full_like(corr_sum, np.nan, dtype=np.float32),
        where=valid_counts > 0,
    ).astype(np.float32)

    if single_fold:
        y_test_all = y_by_article[test_article - 1]
    else:
        y_test_all = np.concatenate(y_by_article, axis=0)
    qc = summarize_corr_qc(corr_raw, y_test_all)
    qc["mean_valid_folds_per_vertex"] = float(np.mean(valid_counts))
    qc["min_valid_folds_per_vertex"] = int(valid_counts.min())
    qc["max_valid_folds_per_vertex"] = int(valid_counts.max())
    qc["n_eval_folds"] = int(len(fold_indices))

    obs_valid = np.isfinite(corr_raw)
    exceed_counts = np.zeros(N_VERTICES, dtype=np.int32)
    null_max_r = np.full(n_perm, np.nan, dtype=np.float32)
    null_sum_r = np.zeros(N_VERTICES, dtype=np.float64)
    null_sumsq_r = np.zeros(N_VERTICES, dtype=np.float64)
    null_count_r = np.zeros(N_VERTICES, dtype=np.int32)

    batch_specs: list[tuple[int, int]] = []
    for batch_start in range(0, n_perm, null_batch):
        batch_end = min(batch_start + null_batch, n_perm)
        batch_specs.append((batch_start, batch_end - batch_start))

    batch_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_perm_batch_agg)(
            batch_start,
            batch_size,
            fold_perm_states,
        )
        for batch_start, batch_size in batch_specs
    )

    for batch_start, r_perm_agg_batch in batch_results:
        batch_size = int(r_perm_agg_batch.shape[0])
        batch_end = batch_start + batch_size
        finite_batch = np.isfinite(r_perm_agg_batch)
        valid_cmp = obs_valid[None, :] & finite_batch
        exceed_counts += np.sum(
            (r_perm_agg_batch >= corr_raw[None, :]) & valid_cmp,
            axis=0,
            dtype=np.int32,
        )
        batch_vals = np.where(finite_batch, r_perm_agg_batch, 0.0).astype(np.float64)
        null_sum_r += np.sum(batch_vals, axis=0, dtype=np.float64)
        null_sumsq_r += np.sum(batch_vals * batch_vals, axis=0, dtype=np.float64)
        null_count_r += np.sum(finite_batch, axis=0, dtype=np.int32)
        if np.isfinite(r_perm_agg_batch).any():
            null_max_r[batch_start:batch_end] = np.nanmax(
                r_perm_agg_batch, axis=1
            ).astype(np.float32)

    p_value_pos = np.full(N_VERTICES, np.nan, dtype=np.float32)
    p_value_pos[obs_valid] = (exceed_counts[obs_valid] + 1.0) / (n_perm + 1.0)

    null_mean_r = np.full(N_VERTICES, np.nan, dtype=np.float32)
    null_std_r = np.full(N_VERTICES, np.nan, dtype=np.float32)
    mean_valid = null_count_r > 0
    null_mean_r[mean_valid] = (null_sum_r[mean_valid] / null_count_r[mean_valid]).astype(
        np.float32
    )
    std_valid = null_count_r > 1
    if std_valid.any():
        mean_sq = np.asarray(null_mean_r[std_valid], dtype=np.float64) ** 2
        var = (
            null_sumsq_r[std_valid] / null_count_r[std_valid]
            - mean_sq
        )
        var = np.maximum(var, 0.0)
        null_std_r[std_valid] = np.sqrt(var).astype(np.float32)

    valid_null_mask = obs_valid & np.isfinite(null_mean_r) & np.isfinite(null_std_r) & (null_std_r > 0)
    z_null_r = np.full(N_VERTICES, np.nan, dtype=np.float32)
    z_null_r[valid_null_mask] = (
        (corr_raw[valid_null_mask] - null_mean_r[valid_null_mask]) / null_std_r[valid_null_mask]
    ).astype(np.float32)

    null_payload = {
        "obs_r": corr_raw.astype(np.float32),
        "p_value_pos": p_value_pos.astype(np.float32),
        "null_mean_r": null_mean_r.astype(np.float32),
        "null_std_r": null_std_r.astype(np.float32),
        "z_null_r": z_null_r.astype(np.float32),
        "valid_null_mask": valid_null_mask.astype(np.bool_),
        "null_count_r": null_count_r.astype(np.int32),
        "null_max_r_per_perm": null_max_r.astype(np.float32),
        "fold_articles": np.asarray([idx + 1 for idx in fold_indices], dtype=np.int32),
        "fold_seeds": np.asarray(fold_seeds, dtype=np.uint32),
        "n_perm": np.asarray([n_perm], dtype=np.int32),
        "null_seed": np.asarray([null_seed], dtype=np.int64),
        "null_batch": np.asarray([null_batch], dtype=np.int32),
        "single_fold": np.asarray([1 if single_fold else 0], dtype=np.int8),
        "test_article": np.asarray([test_article], dtype=np.int32),
    }

    corr_arr = np.nan_to_num(corr_raw, nan=0.0)
    print(
        f"  sub-{sub} layer{layer:02d} hemi-{hemi}: "
        f"mean_valid_r={qc['mean_valid_r']:.4f}  "
        f"max_valid_r={qc['max_valid_r']:.4f}  "
        f"valid={qc['n_valid_corr_vertices']}/{qc['n_vertices']}  "
        f"nonconstant_test={qc['n_nonconstant_test_vertices']}  "
        f"mean_valid_folds={qc['mean_valid_folds_per_vertex']:.2f}  "
        f"null_perm={n_perm}  "
        f"cache={cache_status}  "
        f"({time.time()-t0:.1f}s)"
    )
    return corr_arr, qc, null_payload


def process_subject(
    subj_id: int,
    layer: int,
    hemi: str,
    model_key: str,
    X_by_article: list[np.ndarray],
    words_list: list[int],
    n_jobs: int,
    *,
    single_fold: bool,
    test_article: int,
    cache_dir: Path,
    use_cache: bool,
    sent_offsets_by_article: list[np.ndarray],
    export_oof: bool,
    oof_path: Path | None,
    oof_overwrite: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]] | None:
    """Load Y from cache/disk then run regression (convenience wrapper)."""
    result = load_or_build_y_by_article(
        subj_id, hemi, words_list, cache_dir, use_cache=use_cache
    )
    if result is None:
        return None
    y_by_article, cache_status = result
    return process_subject_with_y(
        subj_id,
        layer,
        hemi,
        model_key,
        X_by_article,
        y_by_article,
        sent_offsets_by_article,
        cache_status,
        n_jobs,
        single_fold=single_fold,
        test_article=test_article,
        export_oof=export_oof,
        oof_path=oof_path,
        oof_overwrite=oof_overwrite,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_comma_separated_ints(s: str) -> list[int]:
    """Parse a comma/whitespace-separated list of non-negative layer indices (order kept)."""
    s = s.strip()
    if not s:
        raise ValueError("empty list")
    out: list[int] = []
    for tok in s.replace(",", " ").split():
        t = tok.strip()
        if not t:
            continue
        v = int(t, 10)
        if v < 0:
            raise ValueError(f"layer index must be >= 0, got {v}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("no valid integers in list")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: per-vertex fMRI ridge regression."
    )
    parser.add_argument("--model", required=True, help="Model key from models.py")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default 0)")
    parser.add_argument(
        "--layers",
        default=None,
        help=(
            "Comma-separated layer indices, e.g. 1,2,3,6,15. "
            "Not valid together with --all_layers or with --layer other than 0 (default)."
        ),
    )
    parser.add_argument("--all_layers", action="store_true", help="Run all layers")
    parser.add_argument("--hemi", choices=["L", "R"], default="L", help="Hemisphere")
    parser.add_argument(
        "--subjects", default=None, help="Comma-separated subject IDs"
    )
    parser.add_argument(
        "--all_subjects", action="store_true", help="Run all fMRI subjects"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for vertices (default -1 = all cores)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if subject-level files already exist",
    )
    parser.add_argument(
        "--single_fold",
        action="store_true",
        help="Run only one fold: train on 4 articles, test on --test_article.",
    )
    parser.add_argument(
        "--test_article",
        type=int,
        default=5,
        help="Held-out article ID for --single_fold (1-5, default: 5).",
    )
    parser.add_argument(
        "--skip_nan_layers",
        action="store_true",
        help="Skip layers whose attention file contains NaN values (and log them).",
    )
    parser.add_argument(
        "--no_y_cache",
        action="store_true",
        help="Disable per-subject/hemi y cache in results/fmri_cache.",
    )
    parser.add_argument(
        "--null_test",
        action="store_true",
        help=(
            "Run optional held-out-article sentence-block permutation null test "
            "(ridge fit once per fold; null outputs saved to *_null.npz)."
        ),
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=10000,
        help="Permutation count for --null_test (default: 10000).",
    )
    parser.add_argument(
        "--null_seed",
        type=int,
        default=1234,
        help="Base RNG seed for --null_test.",
    )
    parser.add_argument(
        "--null_batch",
        type=int,
        default=200,
        help="Permutation progress/report batch size for --null_test.",
    )
    parser.add_argument(
        "--null_overwrite",
        action="store_true",
        help="Recompute *_null.npz even if it already exists.",
    )
    parser.add_argument(
        "--export_oof",
        action="store_true",
        help=(
            "Export compact per-sentence OOF sufficient stats for Phase 5. "
            "Adds *_oof_stats.npz files and leaves corr/qc outputs unchanged."
        ),
    )
    parser.add_argument(
        "--oof_overwrite",
        action="store_true",
        help="Recompute *_oof_stats.npz even if it already exists.",
    )
    args = parser.parse_args()

    if args.export_oof and args.null_test:
        raise ValueError("--export_oof is currently supported for the normal Phase-3 path, not --null_test")

    if args.all_subjects:
        subjects = FMRI_SUBJECTS
    elif args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        subjects = FMRI_SUBJECTS[:2]

    attn_dir = ATTENTION_DIR / args.model
    all_npy = sorted(attn_dir.glob("rb_layer*.npy"))
    n_layers = len(all_npy)
    if n_layers == 0:
        print(f"No .npy files found in {attn_dir}. Run Phase 1 first.")
        sys.exit(1)

    if args.layers is not None:
        if args.all_layers:
            raise ValueError("use either --all_layers or --layers, not both")
        if args.layer != 0:
            raise ValueError("use either --layer or --layers, not both")
        layers = _parse_comma_separated_ints(args.layers)
    elif args.all_layers:
        layers = list(range(n_layers))
    else:
        layers = [args.layer]

    bad = [L for L in layers if L >= n_layers]
    if bad:
        max_idx = n_layers - 1
        raise ValueError(
            f"layer index out of range for {args.model} (n_layers={n_layers}, valid 0..{max_idx}): {bad}"
        )

    if not (1 <= args.test_article <= N_ARTICLES):
        raise ValueError(f"--test_article must be in [1, {N_ARTICLES}]")
    if args.n_perm <= 0:
        raise ValueError("--n_perm must be > 0")
    if args.null_batch <= 0:
        raise ValueError("--null_batch must be > 0")
    words_list = load_words_list()
    sent_offsets_by_article = build_sentence_offsets_by_article(words_list)
    use_cache = not args.no_y_cache
    cv_desc = (
        f"single-fold (train=all except article {args.test_article}, "
        f"test=article {args.test_article})"
        if args.single_fold else
        "leave-one-article-out (5 folds)"
    )

    print("=" * 64)
    print(f"Phase 3: fMRI Regression  [{args.model}]")
    print(f"  Subjects   : {subjects}")
    print(f"  Hemisphere : {args.hemi}")
    print(f"  Layers     : {layers}")
    print(f"  CV         : {cv_desc}")
    print(f"  n_jobs     : {args.n_jobs}")
    print(f"  y_cache    : {'on' if use_cache else 'off'} ({FMRI_CACHE_DIR})")
    print(f"  skip_nan   : {args.skip_nan_layers}")
    print(f"  null_test  : {args.null_test}")
    print(f"  export_oof : {args.export_oof}")
    if args.export_oof:
        print(f"  oof_ovw    : {args.oof_overwrite}")
    if args.null_test:
        print(f"  n_perm     : {args.n_perm}")
        print(f"  null_seed  : {args.null_seed}")
        print(f"  null_batch : {args.null_batch}")
        print(f"  null_ovw   : {args.null_overwrite}")
    print("=" * 64)

    skipped_layers: list[str] = []
    # One attention load + build_X per layer; Y loaded once per subject (when work pending).
    layer_data: list[tuple[int, list[np.ndarray]]] = []
    for layer in layers:
        attn_path = attn_dir / f"rb_layer{layer}.npy"
        if not attn_path.exists():
            msg = f"layer{layer:02d}: attention file not found"
            print(f"  {msg} — skipped")
            skipped_layers.append(msg)
            continue

        layer_attn = np.load(attn_path)
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

        X_by_article = build_X(layer_attn, words_list)
        layer_dir = FMRI_DIR / args.model / f"layer{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_data.append((layer, X_by_article))

    for subj_id in subjects:
        pending: list[tuple[int, list[np.ndarray], bool, bool, bool]] = []
        for layer, X_by_article in layer_data:
            layer_dir = FMRI_DIR / args.model / f"layer{layer:02d}"
            out_path = layer_dir / f"sub{subj_id:02d}_hemi{args.hemi}_corr.npy"
            null_path = layer_dir / f"sub{subj_id:02d}_hemi{args.hemi}_null.npz"
            oof_path = (
                FMRI_OOF_DIR
                / args.model
                / f"layer{layer:02d}"
                / f"sub{subj_id:02d}_hemi{args.hemi}_oof_stats.npz"
            )
            need_oof = args.export_oof and (args.oof_overwrite or (not oof_path.exists()))

            if args.null_test:
                need_corr = args.overwrite or (not out_path.exists())
                need_null = args.null_overwrite or (not null_path.exists())
                if not need_corr and not need_null:
                    print(
                        f"  sub-{subj_id:02d} layer{layer:02d} hemi-{args.hemi}: "
                        "corr/null already exist, skipping"
                    )
                    continue
                if (not need_corr) and need_null:
                    print(
                        f"  sub-{subj_id:02d} layer{layer:02d} hemi-{args.hemi}: "
                        "corr exists; recomputing fold fits for null test only"
                    )
                pending.append((layer, X_by_article, need_corr, need_null, False))
            else:
                need_corr = args.overwrite or (not out_path.exists())
                if not need_corr and not need_oof:
                    print(
                        f"  sub-{subj_id:02d} layer{layer:02d} hemi-{args.hemi}: "
                        "corr/oof already exist, skipping"
                    )
                    continue
                if (not need_corr) and need_oof:
                    print(
                        f"  sub-{subj_id:02d} layer{layer:02d} hemi-{args.hemi}: "
                        "corr exists; recomputing fold fits for OOF export only"
                    )
                pending.append((layer, X_by_article, need_corr, False, need_oof))

        if not pending:
            continue

        y_result = load_or_build_y_by_article(
            subj_id,
            args.hemi,
            words_list,
            FMRI_CACHE_DIR,
            use_cache=use_cache,
        )
        if y_result is None:
            continue
        y_by_article, cache_status = y_result

        for layer, X_by_article, need_corr, need_null, need_oof in pending:
            layer_dir = FMRI_DIR / args.model / f"layer{layer:02d}"
            out_path = layer_dir / f"sub{subj_id:02d}_hemi{args.hemi}_corr.npy"
            qc_path = layer_dir / f"sub{subj_id:02d}_hemi{args.hemi}_qc.npz"
            null_path = layer_dir / f"sub{subj_id:02d}_hemi{args.hemi}_null.npz"
            oof_path = (
                FMRI_OOF_DIR
                / args.model
                / f"layer{layer:02d}"
                / f"sub{subj_id:02d}_hemi{args.hemi}_oof_stats.npz"
            )
            if args.null_test:
                result = process_subject_with_y_null(
                    subj_id,
                    layer,
                    args.hemi,
                    X_by_article,
                    y_by_article,
                    sent_offsets_by_article,
                    cache_status,
                    args.n_jobs,
                    single_fold=args.single_fold,
                    test_article=args.test_article,
                    n_perm=args.n_perm,
                    null_seed=args.null_seed,
                    null_batch=args.null_batch,
                )
                if result is None:
                    continue
                corr, qc, null_payload = result
                if need_corr:
                    np.save(out_path, corr)
                    np.savez(qc_path, **qc)
                if need_null:
                    np.savez(null_path, **null_payload)
            else:
                result = process_subject_with_y(
                    subj_id,
                    layer,
                    args.hemi,
                    args.model,
                    X_by_article,
                    y_by_article,
                    sent_offsets_by_article,
                    cache_status,
                    args.n_jobs,
                    single_fold=args.single_fold,
                    test_article=args.test_article,
                    export_oof=need_oof,
                    oof_path=oof_path if need_oof else None,
                    oof_overwrite=args.oof_overwrite,
                )
                if result is not None:
                    corr, qc = result
                    if need_corr:
                        np.save(out_path, corr)
                        np.savez(qc_path, **qc)

    if skipped_layers:
        model_dir = FMRI_DIR / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / f"skipped_layers_hemi{args.hemi}.log"
        _append_skip_log(log_path, skipped_layers)
        print("\nSkipped layers:")
        for ln in skipped_layers:
            print(f"  - {ln}")
        print(f"Skip log saved to: {log_path}")

    print(f"\nDone  →  {FMRI_DIR / args.model}/")
    if args.export_oof:
        print(f"OOF stats  →  {FMRI_OOF_DIR / args.model}/")


if __name__ == "__main__":
    main()
