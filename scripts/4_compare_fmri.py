#!/usr/bin/env python3
"""
fMRI comparison: VLM vs LLM (whole-cortex encoding model).

Pipeline
--------
1) Subject-level global best-layer summary (Fig. 4a style)
2) Subject x vertex best-layer maps for each model
3) Vertex-wise two-sided paired statistics on best-r maps
   - uncorrected map (p < alpha)
   - BH-FDR map (q < alpha, primary)
   - optional max-T FWE map (supplementary robustness)

Main outputs
------------
results/comparison/fmri_stats_mean.csv
    Layer-profile diagnostics (per-layer global mean Pearson r).
results/comparison/fmri_global_best_layer.csv
    Subject-level best layer per model and hemisphere/domain.
results/comparison/fmri_global_best_stats.csv
    Two-sided paired tests on subject-level global-best scores.
results/comparison/fmri_best_layer.csv
    Backward-compatible alias of fmri_global_best_stats.csv.
results/comparison/fmri_surface_summary.csv
    Hemisphere-level summary of vertex-wise maps/significance.
results/comparison/fmri_vertex_stats_hemiL.npz
results/comparison/fmri_vertex_stats_hemiR.npz
    Precomputed maps: mean_delta, t, p, q, dz, masks, etc.
results/comparison/fmri_best_maps/*
    Subject x vertex best-r maps and best-layer-index maps for both models.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, ttest_rel

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import FMRI_SUBJECTS, N_VERTICES, RESULTS_DIR

FMRI_DIR = RESULTS_DIR / "fmri"
COMPARE_DIR = RESULTS_DIR / "comparison"

LLM_KEY = "qwen2-7b"
VLM_KEY = "qwen2-vl-7b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=np.float64)
    q[order] = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    q = np.clip(q, 0.0, 1.0)
    return q <= alpha, q


def paired_summary(llm: np.ndarray, vlm: np.ndarray) -> dict[str, float]:
    """Two-sided paired summary with Cohen's d_z and 95% CI on mean delta."""
    llm = np.asarray(llm, dtype=np.float64)
    vlm = np.asarray(vlm, dtype=np.float64)
    n = int(min(llm.size, vlm.size))
    llm = llm[:n]
    vlm = vlm[:n]
    diff = vlm - llm
    diff_mean = float(diff.mean()) if n else float("nan")
    diff_std = float(diff.std(ddof=1)) if n > 1 else float("nan")
    d_z = diff_mean / diff_std if np.isfinite(diff_std) and diff_std > 0 else 0.0
    llm_mean = float(llm.mean()) if n else float("nan")
    vlm_mean = float(vlm.mean()) if n else float("nan")
    llm_std = float(llm.std(ddof=1)) if n > 1 else float("nan")
    vlm_std = float(vlm.std(ddof=1)) if n > 1 else float("nan")

    if n >= 3:
        t_stat, p_val = ttest_rel(vlm, llm, alternative="two-sided")
        se = diff_std / math.sqrt(n) if np.isfinite(diff_std) else float("nan")
        t_crit = float(t_dist.ppf(0.975, df=n - 1))
        ci_low = diff_mean - t_crit * se
        ci_high = diff_mean + t_crit * se
    else:
        t_stat, p_val = float("nan"), float("nan")
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "n": n,
        "llm_mean": llm_mean,
        "llm_std": llm_std,
        "vlm_mean": vlm_mean,
        "vlm_std": vlm_std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "t_stat": float(t_stat),
        "p": float(p_val),
        "cohens_d": float(d_z),
        "cohens_dz": float(d_z),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _safe_key(model_key: str) -> str:
    return model_key.replace("/", "_")


def _available_layers(model_key: str) -> list[int]:
    model_dir = FMRI_DIR / model_key
    if not model_dir.exists():
        return []
    layers: list[int] = []
    for d in sorted(model_dir.iterdir()):
        if d.is_dir() and d.name.startswith("layer"):
            try:
                layers.append(int(d.name.replace("layer", "")))
            except ValueError:
                continue
    return layers


def _parse_subjects(subjects_arg: str | None) -> list[int]:
    if not subjects_arg:
        return FMRI_SUBJECTS
    return [int(x.strip()) for x in subjects_arg.split(",") if x.strip()]


def _parse_layers(layers_arg: str | None) -> list[int] | None:
    if layers_arg is None:
        return None
    layers = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    return sorted(set(layers))


def _resolve_layers(requested: list[int] | None, llm_key: str, vlm_key: str) -> list[int]:
    llm_layers = set(_available_layers(llm_key))
    vlm_layers = set(_available_layers(vlm_key))
    common = sorted(llm_layers & vlm_layers)
    if not common:
        return []
    if requested is None:
        return common
    return sorted(set(requested) & set(common))


def _best_selection_layers(layers: list[int], *, label: str = "best-layer selection") -> list[int]:
    """Layers used for best-layer/best-vertex selection (exclude layer 0)."""
    best_layers = [int(layer) for layer in layers if int(layer) > 0]
    if not best_layers:
        raise ValueError(
            f"No layers > 0 available for {label}. "
            f"Resolved layers were: {layers}"
        )
    return best_layers


# ---------------------------------------------------------------------------
# Phase 4.2b: global best-layer summary
# ---------------------------------------------------------------------------


def load_subject_mean_r(model_key: str, layer: int, hemi: str, subjects: list[int]) -> np.ndarray:
    """Load per-subject global mean Pearson r from Phase 3 QC files."""
    out = np.full(len(subjects), np.nan, dtype=np.float64)
    layer_dir = FMRI_DIR / model_key / f"layer{layer:02d}"
    for i, sid in enumerate(subjects):
        qc_path = layer_dir / f"sub{sid:02d}_hemi{hemi}_qc.npz"
        if not qc_path.exists():
            continue
        qc = np.load(qc_path, allow_pickle=True)
        out[i] = float(qc["mean_valid_r"])
    return out


def build_global_layer_matrix(
    model_key: str,
    layers: list[int],
    hemi: str,
    subjects: list[int],
) -> np.ndarray:
    """Return matrix (n_subjects, n_layers) of global mean Pearson r."""
    cols = [load_subject_mean_r(model_key, layer, hemi, subjects) for layer in layers]
    if not cols:
        return np.empty((len(subjects), 0), dtype=np.float64)
    return np.column_stack(cols)


def select_subject_best(layer_matrix: np.ndarray, layers: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select per-subject best layer from matrix (n_subjects, n_layers)."""
    n_subj = layer_matrix.shape[0]
    if layer_matrix.size == 0:
        return (
            np.full(n_subj, np.nan, dtype=np.float64),
            np.full(n_subj, -1, dtype=np.int32),
            np.zeros(n_subj, dtype=np.int32),
        )

    finite = np.isfinite(layer_matrix)
    n_avail = finite.sum(axis=1).astype(np.int32)
    work = np.where(finite, layer_matrix, -np.inf)
    best_idx = np.argmax(work, axis=1).astype(np.int32)
    best_r = work[np.arange(n_subj), best_idx].astype(np.float64)
    best_r[n_avail == 0] = np.nan
    layer_arr = np.asarray(layers, dtype=np.int32)
    best_layer = layer_arr[best_idx]
    best_layer[n_avail == 0] = -1
    return best_r, best_layer, n_avail


def build_layer_profile_table(
    layers: list[int],
    subjects: list[int],
    llm_key: str,
    vlm_key: str,
) -> pd.DataFrame:
    """Diagnostic per-layer global profile (L/R)."""
    rows: list[dict] = []
    for hemi in ("L", "R"):
        for layer in layers:
            llm = load_subject_mean_r(llm_key, layer, hemi, subjects)
            vlm = load_subject_mean_r(vlm_key, layer, hemi, subjects)
            valid = np.isfinite(llm) & np.isfinite(vlm)
            s = paired_summary(llm[valid], vlm[valid])
            rows.append({
                "hemi": hemi,
                "layer": layer,
                "n_subjects": s["n"],
                "llm_mean": s["llm_mean"],
                "llm_std": s["llm_std"],
                "vlm_mean": s["vlm_mean"],
                "vlm_std": s["vlm_std"],
                "diff_mean": s["diff_mean"],
                "diff_std": s["diff_std"],
                "t_stat": s["t_stat"],
                "p_raw": s["p"],
                "cohens_d": s["cohens_d"],
                "cohens_dz": s["cohens_dz"],
                "ci95_low": s["ci95_low"],
                "ci95_high": s["ci95_high"],
            })
    return pd.DataFrame(rows)


def compute_global_best_outputs(
    layers: list[int],
    subjects: list[int],
    llm_key: str,
    vlm_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute subject-level best-layer table + paired summary stats."""
    subject_rows: list[dict] = []
    stat_rows: list[dict] = []

    llm_mat_L = build_global_layer_matrix(llm_key, layers, "L", subjects)
    vlm_mat_L = build_global_layer_matrix(vlm_key, layers, "L", subjects)
    llm_mat_R = build_global_layer_matrix(llm_key, layers, "R", subjects)
    vlm_mat_R = build_global_layer_matrix(vlm_key, layers, "R", subjects)

    domain_specs = [
        ("L", llm_mat_L, vlm_mat_L),
        ("R", llm_mat_R, vlm_mat_R),
    ]

    # Optional pooled domain: require both hemispheres finite for each layer.
    llm_pooled = np.where(np.isfinite(llm_mat_L) & np.isfinite(llm_mat_R), (llm_mat_L + llm_mat_R) / 2.0, np.nan)
    vlm_pooled = np.where(np.isfinite(vlm_mat_L) & np.isfinite(vlm_mat_R), (vlm_mat_L + vlm_mat_R) / 2.0, np.nan)
    domain_specs.append(("LR_pooled", llm_pooled, vlm_pooled))

    for domain, llm_mat, vlm_mat in domain_specs:
        llm_best_r, llm_best_layer, llm_n_avail = select_subject_best(llm_mat, layers)
        vlm_best_r, vlm_best_layer, vlm_n_avail = select_subject_best(vlm_mat, layers)

        for i, sid in enumerate(subjects):
            subject_rows.append({
                "subject_id": int(sid),
                "domain": domain,
                "model": llm_key,
                "best_layer": int(llm_best_layer[i]),
                "best_r": float(llm_best_r[i]),
                "n_layers_available": int(llm_n_avail[i]),
            })
            subject_rows.append({
                "subject_id": int(sid),
                "domain": domain,
                "model": vlm_key,
                "best_layer": int(vlm_best_layer[i]),
                "best_r": float(vlm_best_r[i]),
                "n_layers_available": int(vlm_n_avail[i]),
            })

        valid = np.isfinite(llm_best_r) & np.isfinite(vlm_best_r)
        s = paired_summary(llm_best_r[valid], vlm_best_r[valid])
        stat_rows.append({
            "analysis": "global_per_subject_best",
            "domain": domain,
            "n_subjects": s["n"],
            "llm_mean": s["llm_mean"],
            "llm_std": s["llm_std"],
            "vlm_mean": s["vlm_mean"],
            "vlm_std": s["vlm_std"],
            "diff_mean": s["diff_mean"],
            "diff_std": s["diff_std"],
            "t_stat": s["t_stat"],
            "p_raw": s["p"],
            "cohens_d": s["cohens_d"],
            "cohens_dz": s["cohens_dz"],
            "ci95_low": s["ci95_low"],
            "ci95_high": s["ci95_high"],
        })

    return pd.DataFrame(subject_rows), pd.DataFrame(stat_rows)


def plot_layer_profile(layer_df: pd.DataFrame, out_path: Path) -> None:
    """Diagnostic layer profile plot (L/R)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, hemi in zip(axes, ("L", "R")):
        df = layer_df[layer_df["hemi"] == hemi].sort_values("layer")
        if df.empty:
            continue
        layers = df["layer"].to_numpy(dtype=np.int32)
        n = int(np.nanmax(df["n_subjects"])) if "n_subjects" in df.columns else 0
        sem_scale = 1.0 / np.sqrt(max(n, 1))
        ax.plot(layers, df["llm_mean"], "o-", color="#2196F3", label="LLM", linewidth=1.3, markersize=4)
        ax.fill_between(
            layers,
            df["llm_mean"] - df["llm_std"] * sem_scale,
            df["llm_mean"] + df["llm_std"] * sem_scale,
            color="#2196F3",
            alpha=0.15,
        )
        ax.plot(layers, df["vlm_mean"], "s-", color="#F44336", label="VLM", linewidth=1.3, markersize=4)
        ax.fill_between(
            layers,
            df["vlm_mean"] - df["vlm_std"] * sem_scale,
            df["vlm_mean"] + df["vlm_std"] * sem_scale,
            color="#F44336",
            alpha=0.15,
        )
        ax.set_title(f"Hemi {hemi} (diagnostic)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Global mean Pearson r")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("fMRI Layer Profile (diagnostic, not primary inference)", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_global_best_bar(
    subject_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_path: Path,
    *,
    llm_key: str,
    vlm_key: str,
) -> None:
    """Main-result bar plot for subject-level global best-layer summary."""
    domains = ["L", "R", "LR_pooled"]
    fig, axes = plt.subplots(1, len(domains), figsize=(15, 4.8), sharey=True)

    for ax, domain in zip(axes, domains):
        llm_rows = subject_df[(subject_df["domain"] == domain) & (subject_df["model"] == llm_key)]
        vlm_rows = subject_df[(subject_df["domain"] == domain) & (subject_df["model"] == vlm_key)]
        merged = llm_rows.merge(vlm_rows, on="subject_id", suffixes=("_llm", "_vlm"))
        llm = merged["best_r_llm"].to_numpy(dtype=np.float64)
        vlm = merged["best_r_vlm"].to_numpy(dtype=np.float64)
        valid = np.isfinite(llm) & np.isfinite(vlm)
        llm = llm[valid]
        vlm = vlm[valid]
        n = llm.size

        if n == 0:
            ax.set_title(f"{domain}: no paired data")
            continue

        llm_mean = float(llm.mean())
        vlm_mean = float(vlm.mean())
        llm_sem = float(llm.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        vlm_sem = float(vlm.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        x = np.array([0, 1], dtype=float)
        ax.bar(
            x,
            [llm_mean, vlm_mean],
            yerr=[llm_sem, vlm_sem],
            capsize=5,
            width=0.62,
            color=["#2196F3", "#F44336"],
            alpha=0.86,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{llm_key}", f"{vlm_key}"])

        ax.set_title(f"{domain}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(0, color="k", linewidth=0.6)

        s = stats_df[(stats_df["analysis"] == "global_per_subject_best") & (stats_df["domain"] == domain)]
        if not s.empty:
            r = s.iloc[0]
            txt = (
                f"t={r['t_stat']:.3f}, p={r['p_raw']:.4g}\n"
                f"d_z={r['cohens_dz']:.3f}"
            )
            ax.text(
                0.5,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 3},
            )

    axes[0].set_ylabel("Best-layer global mean Pearson r")
    fig.suptitle(
        "Global Best-Layer Summary — same per-subject layers as fmri_global_best_layer.csv (Step 4.2b)",
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Phase 4.2c: subject x vertex best-layer maps
# ---------------------------------------------------------------------------


def load_vertex_corr(model_key: str, layer: int, hemi: str, subjects: list[int]) -> np.ndarray:
    """Load per-vertex correlation maps: (n_subjects, N_VERTICES)."""
    layer_dir = FMRI_DIR / model_key / f"layer{layer:02d}"
    maps = np.full((len(subjects), N_VERTICES), np.nan, dtype=np.float64)
    for i, sid in enumerate(subjects):
        p = layer_dir / f"sub{sid:02d}_hemi{hemi}_corr.npy"
        if p.exists():
            arr = np.load(p)
            maps[i] = arr.astype(np.float64, copy=False)
    return maps


def build_model_vertex_best_maps(
    model_key: str,
    layers: list[int],
    hemi: str,
    subjects: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Best-r and best-layer-index maps per subject/vertex for one model."""
    n_subj = len(subjects)
    best_r = np.full((n_subj, N_VERTICES), -np.inf, dtype=np.float64)
    best_layer_idx = np.full((n_subj, N_VERTICES), -1, dtype=np.int32)
    valid_any = np.zeros((n_subj, N_VERTICES), dtype=bool)

    for layer in layers:
        maps = load_vertex_corr(model_key, layer, hemi, subjects)
        finite = np.isfinite(maps)
        better = finite & ((~valid_any) | (maps > best_r))
        best_r[better] = maps[better]
        best_layer_idx[better] = int(layer)
        valid_any |= finite

    best_r[~valid_any] = np.nan
    best_layer_idx[~valid_any] = -1
    return best_r, best_layer_idx


def save_best_maps(
    out_dir: Path,
    model_key: str,
    hemi: str,
    subjects: list[int],
    layers: list[int],
    best_r: np.ndarray,
    best_layer_idx: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_key(model_key)}_hemi{hemi}"
    np.save(out_dir / f"{stem}_subjects.npy", np.asarray(subjects, dtype=np.int32))
    np.save(out_dir / f"{stem}_layers_used.npy", np.asarray(layers, dtype=np.int32))
    np.save(out_dir / f"{stem}_best_r.npy", best_r.astype(np.float32))
    np.save(out_dir / f"{stem}_best_layer_idx.npy", best_layer_idx.astype(np.int16))


# ---------------------------------------------------------------------------
# Phase 4.2d: vertex-wise paired statistics
# ---------------------------------------------------------------------------


def _paired_t_from_diff(diff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_delta, t_map, p_map, dz_map) for diff = VLM-LLM."""
    n = diff.shape[0]
    mean_delta = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=1)
    se = std / np.sqrt(n)
    se[se == 0] = np.inf
    t_map = mean_delta / se
    p_map = 2.0 * t_dist.sf(np.abs(t_map), df=n - 1)
    dz = np.divide(mean_delta, std, out=np.zeros_like(mean_delta), where=std > 0)
    return mean_delta, t_map, p_map, dz


def max_t_fwe_pvalues(
    diff: np.ndarray,
    t_obs: np.ndarray,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """Sign-flip max-T FWE p-values (two-sided)."""
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
        perm = diff * signs[:, None]
        _, t_perm, _, _ = _paired_t_from_diff(perm)
        null_max[i] = float(np.nanmax(np.abs(t_perm)))

    sorted_null = np.sort(null_max)
    abs_obs = np.abs(t_obs)
    idx = np.searchsorted(sorted_null, abs_obs, side="left")
    count_ge = n_perm - idx
    return (count_ge + 1.0) / (n_perm + 1.0)


def compute_vertex_stats(
    llm_best: np.ndarray,
    vlm_best: np.ndarray,
    subjects: list[int],
    *,
    alpha: float,
    fwe: str,
    n_perm: int,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    """Compute vertex-wise paired stats from best maps."""
    valid_subj = np.isfinite(llm_best).all(axis=1) & np.isfinite(vlm_best).all(axis=1)
    used_subject_ids = np.asarray(subjects, dtype=np.int32)[valid_subj]
    llm = llm_best[valid_subj]
    vlm = vlm_best[valid_subj]
    n_subj = int(llm.shape[0])
    if n_subj < 3:
        raise ValueError("Need at least 3 paired subjects for vertex-wise tests.")

    diff = vlm - llm
    mean_delta, t_map, p_map, dz_map = _paired_t_from_diff(diff)
    uncorrected_mask = p_map < alpha

    reject_fdr, q_map = fdr_bh(p_map, alpha=alpha)
    fdr_mask = reject_fdr.astype(bool)

    p_fwe = None
    fwe_mask = None
    if fwe == "max_t":
        p_fwe = max_t_fwe_pvalues(diff, t_map, n_perm=n_perm, seed=seed)
        fwe_mask = p_fwe < alpha

    out: dict[str, np.ndarray | float | int] = {
        "subjects_used": used_subject_ids,
        "n_subjects": n_subj,
        "mean_delta": mean_delta.astype(np.float64),
        "t": t_map.astype(np.float64),
        "p": p_map.astype(np.float64),
        "q": q_map.astype(np.float64),
        "dz": dz_map.astype(np.float64),
        "uncorrected_mask": uncorrected_mask.astype(bool),
        "fdr_mask": fdr_mask.astype(bool),
        "llm_best_mean": float(llm.mean()),
        "vlm_best_mean": float(vlm.mean()),
    }
    if p_fwe is not None and fwe_mask is not None:
        out["p_fwe"] = p_fwe.astype(np.float64)
        out["fwe_mask"] = fwe_mask.astype(bool)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4b revised: global best-layer + vertex-wise paired statistics."
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer list. Default: all layers available for both models.",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject IDs. Default: FMRI_SUBJECTS from config.",
    )
    parser.add_argument(
        "--llm",
        default=LLM_KEY,
        help="LLM key under results/fmri/ (default: qwen2-7b)",
    )
    parser.add_argument(
        "--vlm",
        default=VLM_KEY,
        help="VLM key under results/fmri/ (default: qwen2-vl-7b)",
    )
    parser.add_argument(
        "--compare_dir",
        type=Path,
        default=None,
        help="Output directory for CSV/NPZ/PNG (default: results/comparison)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance alpha (default: 0.05)")
    parser.add_argument(
        "--fwe",
        choices=["none", "max_t"],
        default="none",
        help="Optional supplementary FWE correction.",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=10000,
        help="Permutation count for --fwe max_t.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for permutations.")
    args = parser.parse_args()

    requested_layers = _parse_layers(args.layers)
    subjects = _parse_subjects(args.subjects)
    llm_key, vlm_key = args.llm, args.vlm
    compare_dir = (args.compare_dir if args.compare_dir is not None else COMPARE_DIR).resolve()
    compare_dir.mkdir(parents=True, exist_ok=True)

    layers = _resolve_layers(requested_layers, llm_key, vlm_key)
    if not layers:
        raise ValueError("No common layers available for both models.")
    best_layers = _best_selection_layers(layers, label="Phase 4 best-layer outputs")

    print("=" * 72)
    print("Phase 4b revised: fMRI VLM vs LLM comparison")
    print(f"  Models    : {llm_key} vs {vlm_key}")
    print(f"  Layers    : {layers}")
    print(f"  BestLayers: {best_layers} (layer > 0)")
    print(f"  Subjects  : n={len(subjects)}")
    print(f"  Alpha     : {args.alpha}")
    print(f"  FWE mode  : {args.fwe}")
    if args.fwe == "max_t":
        print(f"  Perms     : {args.n_perm}")
    print("=" * 72)

    # Step 4.2b diagnostics: per-layer global profile.
    print("\n--- Step 4.2b diagnostic: layer profile table ---")
    layer_df = build_layer_profile_table(layers, subjects, llm_key, vlm_key)
    layer_csv = compare_dir / "fmri_stats_mean.csv"
    layer_df.to_csv(layer_csv, index=False, float_format="%.6f")
    print(f"  Saved: {layer_csv}")
    plot_layer_profile(layer_df, compare_dir / "fmri_layer_profile.png")

    # Step 4.2b primary: subject-level global best summary.
    print("\n--- Step 4.2b primary: subject-level global best-layer summary ---")
    subj_best_df, global_stats_df = compute_global_best_outputs(best_layers, subjects, llm_key, vlm_key)
    global_subj_csv = compare_dir / "fmri_global_best_layer.csv"
    global_stats_csv = compare_dir / "fmri_global_best_stats.csv"
    subj_best_df.to_csv(global_subj_csv, index=False, float_format="%.6f")
    global_stats_df.to_csv(global_stats_csv, index=False, float_format="%.6f")
    # Backward-compatible alias.
    global_stats_df.to_csv(compare_dir / "fmri_best_layer.csv", index=False, float_format="%.6f")
    print(f"  Saved: {global_subj_csv}")
    print(f"  Saved: {global_stats_csv}")
    print(f"  Saved: {compare_dir / 'fmri_best_layer.csv'}")
    plot_global_best_bar(
        subj_best_df,
        global_stats_df,
        compare_dir / "fmri_global_best_bar.png",
        llm_key=llm_key,
        vlm_key=vlm_key,
    )

    # Step 4.2c + 4.2d: best maps and vertex-wise paired statistics.
    print("\n--- Step 4.2c/4.2d: subject x vertex best maps + paired stats ---")
    best_map_dir = compare_dir / "fmri_best_maps"
    summary_rows: list[dict] = []
    for hemi in ("L", "R"):
        print(f"  Building best maps for hemi-{hemi} ...")
        llm_best_r, llm_best_idx = build_model_vertex_best_maps(llm_key, best_layers, hemi, subjects)
        vlm_best_r, vlm_best_idx = build_model_vertex_best_maps(vlm_key, best_layers, hemi, subjects)

        save_best_maps(best_map_dir, llm_key, hemi, subjects, best_layers, llm_best_r, llm_best_idx)
        save_best_maps(best_map_dir, vlm_key, hemi, subjects, best_layers, vlm_best_r, vlm_best_idx)

        print(f"  Vertex-wise paired tests for hemi-{hemi} ...")
        stats = compute_vertex_stats(
            llm_best_r,
            vlm_best_r,
            subjects,
            alpha=args.alpha,
            fwe=args.fwe,
            n_perm=args.n_perm,
            seed=args.seed + (0 if hemi == "L" else 1),
        )

        out_npz = compare_dir / f"fmri_vertex_stats_hemi{hemi}.npz"
        np.savez(
            out_npz,
            mean_delta=stats["mean_delta"],
            t=stats["t"],
            p=stats["p"],
            q=stats["q"],
            dz=stats["dz"],
            uncorrected_mask=stats["uncorrected_mask"],
            fdr_mask=stats["fdr_mask"],
            p_fwe=stats.get("p_fwe", np.array([])),
            fwe_mask=stats.get("fwe_mask", np.array([], dtype=bool)),
            subjects=stats["subjects_used"],
            n_subjects=np.array([stats["n_subjects"]], dtype=np.int32),
            llm_best_mean=np.array([stats["llm_best_mean"]], dtype=np.float64),
            vlm_best_mean=np.array([stats["vlm_best_mean"]], dtype=np.float64),
            layers_used=np.asarray(best_layers, dtype=np.int32),
        )
        print(f"  Saved: {out_npz}")

        mean_delta = np.asarray(stats["mean_delta"], dtype=np.float64)
        summary_rows.append({
            "hemi": hemi,
            "n_subjects_used": int(stats["n_subjects"]),
            "mean_delta_global": float(np.nanmean(mean_delta)),
            "max_delta": float(np.nanmax(mean_delta)),
            "min_delta": float(np.nanmin(mean_delta)),
            "n_vertices_vlm_gt_llm": int((mean_delta > 0).sum()),
            "pct_vertices_vlm_gt_llm": float((mean_delta > 0).mean() * 100.0),
            "n_uncorrected_sig": int(np.asarray(stats["uncorrected_mask"], dtype=bool).sum()),
            "n_fdr_sig": int(np.asarray(stats["fdr_mask"], dtype=bool).sum()),
            "n_fwe_sig": int(np.asarray(stats.get("fwe_mask", np.array([], dtype=bool)), dtype=bool).sum())
            if args.fwe != "none"
            else 0,
        })

    surface_df = pd.DataFrame(summary_rows)
    surface_csv = compare_dir / "fmri_surface_summary.csv"
    surface_df.to_csv(surface_csv, index=False, float_format="%.6f")
    # Convenience alias with a clearer name.
    surface_df.to_csv(compare_dir / "fmri_vertex_stats_summary.csv", index=False, float_format="%.6f")
    print(f"  Saved: {surface_csv}")
    print(f"  Saved: {compare_dir / 'fmri_vertex_stats_summary.csv'}")

    print("\n" + "=" * 72)
    print("Phase 4b revised comparison complete.")
    print(f"All outputs saved to: {compare_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

