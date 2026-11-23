#!/usr/bin/env python3
"""
Phase 4a: Statistical comparison of VLM vs LLM saccade alignment.

Loads per-subject Pearson r arrays saved by 2_encode_saccade.py for each
layer / model, runs paired t-tests, FDR correction, Cohen's d, and
produces summary tables + publication-ready figures.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, ttest_rel

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import RESULTS_DIR

SACCADE_DIR = RESULTS_DIR / "saccade"
COMPARE_DIR = RESULTS_DIR / "comparison"
NC_CSV = RESULTS_DIR / "noise_ceiling" / "saccade_noise_ceiling.csv"

LLM_KEY = "qwen2-7b"
VLM_KEY = "qwen2-vl-7b"

# Metric name → measure string in saccade_noise_ceiling.csv
_METRIC_TO_MEASURE = {"num": "saccade_count", "dur": "saccade_duration"}


def _best_selection_layers(layers: list[int], *, label: str = "best-layer selection") -> list[int]:
    """Layers used for best-layer/per-subject-best aggregation (exclude layer 0)."""
    best_layers = [int(layer) for layer in layers if int(layer) > 0]
    if not best_layers:
        raise ValueError(
            f"No layers > 0 available for {label}. "
            f"Resolved layers were: {layers}"
        )
    return best_layers


def load_saccade_nc() -> dict[str, float]:
    """Load saccade noise-ceiling r values from Phase 5 CSV.

    Returns {metric: nc_r} e.g. {"num": 0.3259, "dur": 0.3228}.
    Returns {} if file not found.
    """
    if not NC_CSV.exists():
        return {}
    try:
        df = pd.read_csv(NC_CSV)
        result = {}
        for metric, measure in _METRIC_TO_MEASURE.items():
            row = df[df["measure"] == measure]
            if not row.empty:
                result[metric] = float(row["noise_ceiling_r"].values[0])
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns (rejected, q_values) where rejected is a boolean mask and
    q_values are adjusted p-values.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    q = np.empty(n)
    q[sorted_idx] = sorted_p * n / (np.arange(n) + 1)
    # Enforce monotonicity (right to left)
    for i in range(n - 2, -1, -1):
        q[sorted_idx[i]] = min(q[sorted_idx[i]], q[sorted_idx[i + 1]])
    q = np.clip(q, 0, 1)
    return q <= alpha, q


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_saccade_results(
    model_key: str, metric: str
) -> dict[int, dict[str, np.ndarray | None]]:
    """Load per-layer results for a given model and metric.

    Returns
    -------
    {layer_index: {"values": (n_subjects,), "subjects": (n_subjects,) or None}}
    """
    model_dir = SACCADE_DIR / model_key
    results: dict[int, dict[str, np.ndarray | None]] = {}
    for f in sorted(model_dir.glob(f"layer*_{metric}.npy")):
        layer_str = f.stem.split("_")[0].replace("layer", "")
        layer = int(layer_str)
        values = np.load(f)
        subj_path = model_dir / f"layer{layer:02d}_subjects.npy"
        subjects = np.load(subj_path) if subj_path.exists() else None
        results[layer] = {"values": values, "subjects": subjects}
    return results


def _align_by_subject_ids(
    llm_vals: np.ndarray,
    llm_subj: np.ndarray | None,
    vlm_vals: np.ndarray,
    vlm_subj: np.ndarray | None,
    *,
    layer: int,
    allow_legacy_positional_pairing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return subject-aligned vectors and aligned subject IDs.

    If subject IDs are available for both models, alignment is done by ID
    intersection. If subject IDs are missing for both models, positional
    pairing can be optionally enabled for legacy outputs.
    """
    has_llm_ids = llm_subj is not None
    has_vlm_ids = vlm_subj is not None

    if has_llm_ids and has_vlm_ids:
        if len(llm_vals) != len(llm_subj):
            raise ValueError(
                f"Layer {layer}: LLM values/subjects length mismatch "
                f"({len(llm_vals)} vs {len(llm_subj)})."
            )
        if len(vlm_vals) != len(vlm_subj):
            raise ValueError(
                f"Layer {layer}: VLM values/subjects length mismatch "
                f"({len(vlm_vals)} vs {len(vlm_subj)})."
            )

        llm_subj_i = llm_subj.astype(np.int32)
        vlm_subj_i = vlm_subj.astype(np.int32)
        common = np.intersect1d(llm_subj_i, vlm_subj_i)
        if common.size == 0:
            return np.array([]), np.array([]), common

        llm_idx = {int(s): i for i, s in enumerate(llm_subj_i)}
        vlm_idx = {int(s): i for i, s in enumerate(vlm_subj_i)}
        llm_aligned = np.array([llm_vals[llm_idx[int(s)]] for s in common], dtype=np.float32)
        vlm_aligned = np.array([vlm_vals[vlm_idx[int(s)]] for s in common], dtype=np.float32)
        return llm_aligned, vlm_aligned, common

    if has_llm_ids != has_vlm_ids:
        raise ValueError(
            f"Layer {layer}: subject-id sidecar exists for only one model. "
            f"Re-run Phase 2 for both models so both have layerXX_subjects.npy."
        )

    # Legacy mode: no subject IDs on either side.
    if not allow_legacy_positional_pairing:
        raise ValueError(
            f"Layer {layer}: missing layerXX_subjects.npy for both models. "
            "Cannot guarantee correct pairing. Re-run Phase 2 for both models "
            "or pass --allow_legacy_positional_pairing (not recommended)."
        )

    if len(llm_vals) != len(vlm_vals):
        raise ValueError(
            f"Layer {layer}: legacy positional pairing requested but lengths differ "
            f"({len(llm_vals)} vs {len(vlm_vals)})."
        )
    pseudo = np.arange(len(llm_vals), dtype=np.int32)
    return llm_vals.astype(np.float32), vlm_vals.astype(np.float32), pseudo


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def _paired_summary(llm: np.ndarray, vlm: np.ndarray) -> dict[str, float]:
    """Two-sided paired test summary for VLM vs LLM."""
    n = int(min(len(llm), len(vlm)))
    llm = llm[:n].astype(np.float64)
    vlm = vlm[:n].astype(np.float64)
    diff = vlm - llm
    t_stat, p_val = ttest_rel(vlm, llm, alternative="two-sided")
    diff_mean = float(diff.mean())
    diff_std = float(diff.std(ddof=1)) if n > 1 else float("nan")
    d_z = diff_mean / diff_std if np.isfinite(diff_std) and diff_std > 0 else 0.0
    if n > 1 and np.isfinite(diff_std):
        se = diff_std / np.sqrt(n)
        t_crit = float(t_dist.ppf(0.975, df=n - 1))
        ci_low = diff_mean - t_crit * se
        ci_high = diff_mean + t_crit * se
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "n": n,
        "llm_mean": float(llm.mean()),
        "llm_std": float(llm.std(ddof=1)) if n > 1 else float("nan"),
        "vlm_mean": float(vlm.mean()),
        "vlm_std": float(vlm.std(ddof=1)) if n > 1 else float("nan"),
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "t_stat": float(t_stat),
        "p": float(p_val),
        "cohens_d": float(d_z),
        "cohens_dz": float(d_z),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def compare_layers(
    llm_data: dict[int, dict[str, np.ndarray | None]],
    vlm_data: dict[int, dict[str, np.ndarray | None]],
    metric: str,
    *,
    allow_legacy_positional_pairing: bool = False,
) -> pd.DataFrame:
    """Paired t-test at each layer; FDR across layers."""
    common_layers = sorted(set(llm_data.keys()) & set(vlm_data.keys()))
    rows: list[dict] = []

    p_values_raw: list[float] = []
    valid_row_indices: list[int] = []
    for layer in common_layers:
        llm_vals = llm_data[layer]["values"]
        llm_subj = llm_data[layer]["subjects"]
        vlm_vals = vlm_data[layer]["values"]
        vlm_subj = vlm_data[layer]["subjects"]
        llm, vlm, common_subj = _align_by_subject_ids(
            llm_vals,
            llm_subj,
            vlm_vals,
            vlm_subj,
            layer=layer,
            allow_legacy_positional_pairing=allow_legacy_positional_pairing,
        )
        n = len(common_subj)
        if n < 3:
            rows.append({
                "layer": layer,
                "metric": metric,
                "n_subjects": n,
                "note": "too few aligned subjects",
            })
            continue

        summary = _paired_summary(llm, vlm)
        p_values_raw.append(summary["p"])

        rows.append({
            "layer": layer,
            "metric": metric,
            "n_subjects": summary["n"],
            "llm_mean": summary["llm_mean"],
            "llm_std": summary["llm_std"],
            "vlm_mean": summary["vlm_mean"],
            "vlm_std": summary["vlm_std"],
            "diff_mean": summary["diff_mean"],
            "diff_std": summary["diff_std"],
            "t_stat": summary["t_stat"],
            "p_raw": summary["p"],
            "cohens_d": summary["cohens_d"],
            "cohens_dz": summary["cohens_dz"],
            "ci95_low": summary["ci95_low"],
            "ci95_high": summary["ci95_high"],
        })
        valid_row_indices.append(len(rows) - 1)

    df = pd.DataFrame(rows)
    if len(valid_row_indices) > 0:
        rejected, q_values = fdr_bh(np.array(p_values_raw))
        df["p_fdr"] = np.nan
        df["sig_fdr05"] = False
        for idx, q, rej in zip(valid_row_indices, q_values, rejected):
            df.at[idx, "p_fdr"] = q
            df.at[idx, "sig_fdr05"] = bool(rej)
    return df


# ---------------------------------------------------------------------------
# Best-layer analysis
# ---------------------------------------------------------------------------


def compute_global_best_outputs(
    llm_data: dict[int, dict[str, np.ndarray | None]],
    vlm_data: dict[int, dict[str, np.ndarray | None]],
    metric: str,
    *,
    llm_key: str,
    vlm_key: str,
    allow_legacy_positional_pairing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-level global best-layer table + paired summary stats."""
    all_common_layers = sorted(set(llm_data.keys()) & set(vlm_data.keys()))
    common_layers = _best_selection_layers(all_common_layers, label=f"saccade best-layer analysis ({metric})")
    subject_rows: list[dict] = []

    def _best_by_subject(
        model_data: dict[int, dict[str, np.ndarray | None]],
        model_name: str,
    ) -> tuple[dict[int, float], dict[int, int]]:
        best_r: dict[int, float] = {}
        best_layer: dict[int, int] = {}
        n_layers_available: dict[int, int] = {}

        for layer in common_layers:
            vals = np.asarray(model_data[layer]["values"], dtype=np.float32)
            subj = model_data[layer]["subjects"]
            if subj is None:
                if not allow_legacy_positional_pairing:
                    raise ValueError(
                        f"Missing layerXX_subjects.npy for {model_name} at layer {layer}. "
                        "Re-run Phase 2 or pass --allow_legacy_positional_pairing."
                    )
                subj_i = np.arange(len(vals), dtype=np.int32)
            else:
                subj_i = np.asarray(subj, dtype=np.int32)
                if len(subj_i) != len(vals):
                    raise ValueError(
                        f"Layer {layer}: {model_name} values/subjects length mismatch "
                        f"({len(vals)} vs {len(subj_i)})."
                    )

            for sid, val in zip(subj_i, vals):
                sid_i = int(sid)
                v = float(val)
                if not np.isfinite(v):
                    continue
                n_layers_available[sid_i] = n_layers_available.get(sid_i, 0) + 1
                if sid_i not in best_r or v > best_r[sid_i]:
                    best_r[sid_i] = v
                    best_layer[sid_i] = int(layer)

        for sid_i in sorted(n_layers_available):
            subject_rows.append({
                "subject_id": sid_i,
                "metric": metric,
                "model": model_name,
                "best_layer": int(best_layer.get(sid_i, -1)),
                "best_r": float(best_r.get(sid_i, np.nan)),
                "n_layers_available": int(n_layers_available[sid_i]),
            })
        return best_r, best_layer

    llm_best_r_map, _ = _best_by_subject(llm_data, llm_key)
    vlm_best_r_map, _ = _best_by_subject(vlm_data, vlm_key)

    common_subj = sorted(set(llm_best_r_map.keys()) & set(vlm_best_r_map.keys()))
    llm_best = np.asarray([llm_best_r_map[s] for s in common_subj], dtype=np.float64)
    vlm_best = np.asarray([vlm_best_r_map[s] for s in common_subj], dtype=np.float64)
    summary = _paired_summary(llm_best, vlm_best) if len(common_subj) >= 3 else {
        "n": len(common_subj),
        "llm_mean": float(np.nanmean(llm_best)) if len(common_subj) else float("nan"),
        "llm_std": float(np.nanstd(llm_best, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "vlm_mean": float(np.nanmean(vlm_best)) if len(common_subj) else float("nan"),
        "vlm_std": float(np.nanstd(vlm_best, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "diff_mean": float(np.nanmean(vlm_best - llm_best)) if len(common_subj) else float("nan"),
        "diff_std": float(np.nanstd(vlm_best - llm_best, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "t_stat": float("nan"),
        "p": float("nan"),
        "cohens_d": float("nan"),
        "cohens_dz": float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }

    stats_df = pd.DataFrame([{
        "analysis": "global_per_subject_best",
        "metric": metric,
        "n_subjects": summary["n"],
        "llm_mean": summary["llm_mean"],
        "llm_std": summary["llm_std"],
        "vlm_mean": summary["vlm_mean"],
        "vlm_std": summary["vlm_std"],
        "diff_mean": summary["diff_mean"],
        "diff_std": summary["diff_std"],
        "t_stat": summary["t_stat"],
        "p": summary["p"],
        "cohens_d": summary["cohens_d"],
        "cohens_dz": summary["cohens_dz"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
    }])

    return pd.DataFrame(subject_rows), stats_df


def compute_fixed_layer_outputs(
    llm_data: dict[int, dict[str, np.ndarray | None]],
    vlm_data: dict[int, dict[str, np.ndarray | None]],
    metric: str,
    *,
    layer: int,
    llm_key: str,
    vlm_key: str,
    allow_legacy_positional_pairing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-level fixed-layer table + paired summary stats."""
    all_common_layers = sorted(set(llm_data.keys()) & set(vlm_data.keys()))
    if layer not in all_common_layers:
        raise ValueError(
            f"Layer {layer} is unavailable for fixed-layer analysis ({metric}). "
            f"Common layers are: {all_common_layers}"
        )

    llm_vals = np.asarray(llm_data[layer]["values"], dtype=np.float32)
    llm_subj = llm_data[layer]["subjects"]
    vlm_vals = np.asarray(vlm_data[layer]["values"], dtype=np.float32)
    vlm_subj = vlm_data[layer]["subjects"]

    if llm_subj is None:
        if not allow_legacy_positional_pairing:
            raise ValueError(
                f"Missing layerXX_subjects.npy for {llm_key} at layer {layer}. "
                "Re-run Phase 2 or pass --allow_legacy_positional_pairing."
            )
        llm_subj_i = np.arange(len(llm_vals), dtype=np.int32)
    else:
        llm_subj_i = np.asarray(llm_subj, dtype=np.int32)
    if vlm_subj is None:
        if not allow_legacy_positional_pairing:
            raise ValueError(
                f"Missing layerXX_subjects.npy for {vlm_key} at layer {layer}. "
                "Re-run Phase 2 or pass --allow_legacy_positional_pairing."
            )
        vlm_subj_i = np.arange(len(vlm_vals), dtype=np.int32)
    else:
        vlm_subj_i = np.asarray(vlm_subj, dtype=np.int32)

    subject_rows: list[dict] = []
    for sid, val in zip(llm_subj_i, llm_vals):
        subject_rows.append({
            "subject_id": int(sid),
            "metric": metric,
            "model": llm_key,
            "best_layer": int(layer),
            "best_r": float(val),
            "n_layers_available": 1 if np.isfinite(float(val)) else 0,
        })
    for sid, val in zip(vlm_subj_i, vlm_vals):
        subject_rows.append({
            "subject_id": int(sid),
            "metric": metric,
            "model": vlm_key,
            "best_layer": int(layer),
            "best_r": float(val),
            "n_layers_available": 1 if np.isfinite(float(val)) else 0,
        })

    llm, vlm, common_subj = _align_by_subject_ids(
        llm_vals,
        llm_subj,
        vlm_vals,
        vlm_subj,
        layer=layer,
        allow_legacy_positional_pairing=allow_legacy_positional_pairing,
    )
    summary = _paired_summary(llm, vlm) if len(common_subj) >= 3 else {
        "n": len(common_subj),
        "llm_mean": float(np.nanmean(llm)) if len(common_subj) else float("nan"),
        "llm_std": float(np.nanstd(llm, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "vlm_mean": float(np.nanmean(vlm)) if len(common_subj) else float("nan"),
        "vlm_std": float(np.nanstd(vlm, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "diff_mean": float(np.nanmean(vlm - llm)) if len(common_subj) else float("nan"),
        "diff_std": float(np.nanstd(vlm - llm, ddof=1)) if len(common_subj) > 1 else float("nan"),
        "t_stat": float("nan"),
        "p": float("nan"),
        "cohens_d": float("nan"),
        "cohens_dz": float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }
    stats_df = pd.DataFrame([{
        "analysis": "fixed_layer",
        "layer": int(layer),
        "metric": metric,
        "n_subjects": summary["n"],
        "llm_mean": summary["llm_mean"],
        "llm_std": summary["llm_std"],
        "vlm_mean": summary["vlm_mean"],
        "vlm_std": summary["vlm_std"],
        "diff_mean": summary["diff_mean"],
        "diff_std": summary["diff_std"],
        "t_stat": summary["t_stat"],
        "p": summary["p"],
        "cohens_d": summary["cohens_d"],
        "cohens_dz": summary["cohens_dz"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
    }])
    return pd.DataFrame(subject_rows), stats_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_layer_profile(
    stats_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    use_nc: bool = False,
    *,
    llm_key: str,
    vlm_key: str,
) -> None:
    """Layer-profile curve: mean +/- SEM for both models, with significance markers.

    Parameters
    ----------
    use_nc : if True and NC columns are present, plot sign(r)·r²/NC_r²;
             if False, always plot raw Pearson r.
    """
    df = stats_df[stats_df["metric"] == metric].sort_values("layer")
    layers = df["layer"].values
    n_subj = df["n_subjects"].iloc[0]
    sem_scale = 1.0 / np.sqrt(n_subj)

    nc_available = "llm_mean_nc" in df.columns and df["llm_mean_nc"].notna().any()
    use_nc = use_nc and nc_available
    if use_nc:
        y_llm = df["llm_mean_nc"]
        y_vlm = df["vlm_mean_nc"]
        nc_r = float(df["noise_ceiling_r"].iloc[0]) if "noise_ceiling_r" in df.columns else 1.0
        nc_r2 = nc_r ** 2
        # Delta-method SEM for sign(r)·r²/NC_r²:
        #   std(f(r)) ≈ 2|r̄| · std_r / NC_r²
        y_std_llm = 2 * df["llm_mean"].abs() * df["llm_std"] / nc_r2
        y_std_vlm = 2 * df["vlm_mean"].abs() * df["vlm_std"] / nc_r2
        y_diff = df["diff_mean_nc"]
        y_label = f"sign(r) x r2 / NC_r2 ({metric})"
        nc_note = " [NC-normalised]"
    else:
        y_llm = df["llm_mean"]
        y_vlm = df["vlm_mean"]
        y_std_llm = df["llm_std"]
        y_std_vlm = df["vlm_std"]
        y_diff = df["diff_mean"]
        y_label = f"Pearson r ({metric})"
        nc_note = ""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(layers, y_llm, "o-", color="#2196F3", label=f"LLM ({llm_key})", markersize=3, linewidth=1.2)
    ax.fill_between(
        layers,
        y_llm - y_std_llm * sem_scale,
        y_llm + y_std_llm * sem_scale,
        color="#2196F3", alpha=0.15,
    )
    ax.plot(layers, y_vlm, "s-", color="#F44336", label=f"VLM ({vlm_key})", markersize=3, linewidth=1.2)
    ax.fill_between(
        layers,
        y_vlm - y_std_vlm * sem_scale,
        y_vlm + y_std_vlm * sem_scale,
        color="#F44336", alpha=0.15,
    )

    sig_rows = df[df["sig_fdr05"]]
    if len(sig_rows) > 0:
        y_sig = float(np.nanmax(np.concatenate([y_llm.values, y_vlm.values]))) * 1.02
        ax.scatter(sig_rows["layer"], [y_sig] * len(sig_rows), marker="*", c="gold", s=60, zorder=5, label="FDR q<0.05")

    ax.set_ylabel(y_label)
    ax.set_title(f"Saccade Alignment: LLM vs VLM — Layer Profile ({metric}){nc_note}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Difference subplot
    ax2 = axes[1]
    colors = ["#4CAF50" if d > 0 else "#FF9800" for d in y_diff]
    ax2.bar(layers, y_diff, color=colors, width=0.8, alpha=0.7)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("VLM - LLM" if not use_nc else "VLM - LLM (NC)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_best_layer_bar(
    best_stats_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    use_nc: bool = False,
    *,
    llm_key: str,
    vlm_key: str,
    analysis: str = "global_per_subject_best",
) -> None:
    """Main-result bar plot based on selected summary analysis."""
    df = best_stats_df[best_stats_df["metric"] == metric]
    if df.empty:
        return
    primary = df[df["analysis"] == analysis]
    if primary.empty:
        return
    row = primary.iloc[0]

    if use_nc and "llm_mean_nc" in row and pd.notna(row["llm_mean_nc"]):
        llm_mean = float(row["llm_mean_nc"])
        vlm_mean = float(row["vlm_mean_nc"])
        diff_mean = float(row["diff_mean_nc"])
        llm_std = float(2 * abs(row["llm_mean"]) * row["llm_std"] / (row["noise_ceiling_r2"]))
        vlm_std = float(2 * abs(row["vlm_mean"]) * row["vlm_std"] / (row["noise_ceiling_r2"]))
        y_label = "sign(r) x r2 / NC_r2"
        title_note = " [NC-normalised]"
    else:
        llm_mean = float(row["llm_mean"])
        vlm_mean = float(row["vlm_mean"])
        diff_mean = float(row["diff_mean"])
        llm_std = float(row["llm_std"])
        vlm_std = float(row["vlm_std"])
        y_label = "Pearson r"
        title_note = ""

    n = int(row["n_subjects"])
    llm_sem = llm_std / np.sqrt(n) if n > 0 and np.isfinite(llm_std) else 0.0
    vlm_sem = vlm_std / np.sqrt(n) if n > 0 and np.isfinite(vlm_std) else 0.0

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = np.array([0, 1], dtype=float)
    means = np.array([llm_mean, vlm_mean], dtype=float)
    sems = np.array([llm_sem, vlm_sem], dtype=float)
    colors = ["#2196F3", "#F44336"]
    ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{llm_key}", f"{vlm_key}"], fontsize=9)
    ax.set_ylabel(y_label)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.25)
    if analysis == "fixed_layer":
        layer_note = f", layer {int(row['layer'])}"
        ax.set_title(f"Saccade Fixed-Layer Result ({metric}{layer_note}){title_note}")
    else:
        ax.set_title(f"Saccade Best-Layer Main Result ({metric}){title_note}")

    p_val = float(row["p"])
    t_stat = float(row["t_stat"])
    d_z = float(row["cohens_dz"])
    ci_low = float(row["ci95_low"])
    ci_high = float(row["ci95_high"])
    text = (
        f"t={t_stat:.3f}, p={p_val:.4g},mean delta={diff_mean:+.4f}"
    )
    ax.text(
        0.5, 0.98, text,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4a: saccade VLM vs LLM comparison.")
    parser.add_argument("--metric", default="num", choices=["num", "dur", "both"])
    parser.add_argument(
        "--llm", "--LLM", default=LLM_KEY,
        help="LLM key matching results/saccade/<llm>/ (default: qwen2-7b)",
    )
    parser.add_argument(
        "--vlm", "--VLM", default=VLM_KEY,
        help="VLM key matching results/saccade/<vlm>/ (default: qwen2-vl-7b)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "Fixed layer for main bar/summary. "
            "If set, outputs fixed-layer summary instead of global-best summary."
        ),
    )
    parser.add_argument(
        "--compare_dir",
        type=Path,
        default=None,
        help="Output directory for CSV/PNG (default: results/comparison)",
    )
    parser.add_argument(
        "--allow_legacy_positional_pairing",
        action="store_true",
        help=(
            "Allow positional pairing when layerXX_subjects.npy is missing on both "
            "sides. Not recommended because subject mis-pairing cannot be ruled out."
        ),
    )
    args = parser.parse_args()

    compare_dir = (args.compare_dir if args.compare_dir is not None else COMPARE_DIR).resolve()
    llm_key, vlm_key = args.llm, args.vlm

    metrics = ["num", "dur"] if args.metric == "both" else [args.metric]
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Load noise ceiling (adds NC columns to output; t/p values are unaffected)
    nc_map = load_saccade_nc()
    if nc_map:
        print("Noise ceiling loaded: " +
              "  ".join(f"{m}={v:.4f}" for m, v in nc_map.items()))
        print("  (NC columns added to CSV; t-statistics are NC-invariant)")
    else:
        print("Noise ceiling not found — NC columns will be omitted from CSV output.")

    all_stats: list[pd.DataFrame] = []
    all_best_subjects: list[pd.DataFrame] = []
    all_best_stats: list[pd.DataFrame] = []

    for metric in metrics:
        print(f"\n{'='*64}")
        print(f"Saccade comparison: metric = {metric}")
        print(f"{'='*64}")

        llm_data = load_saccade_results(llm_key, metric)
        vlm_data = load_saccade_results(vlm_key, metric)
        print(f"  LLM layers loaded: {len(llm_data)}  VLM layers loaded: {len(vlm_data)}")
        common_layers_all = sorted(set(llm_data.keys()) & set(vlm_data.keys()))
        if common_layers_all:
            best_layers = _best_selection_layers(common_layers_all, label=f"saccade best-layer analysis ({metric})")
            print(f"  Best-layer selection uses layers > 0: {best_layers}")
            if args.layer is not None:
                print(f"  Fixed-layer mode: layer {args.layer}")

        if not llm_data or not vlm_data:
            print("  No data found — skipping.")
            continue

        stats = compare_layers(
            llm_data,
            vlm_data,
            metric,
            allow_legacy_positional_pairing=args.allow_legacy_positional_pairing,
        )
        if "diff_mean" not in stats.columns:
            print("  No layer has enough aligned subjects for paired testing — skipping metric.")
            continue

        # Append noise-ceiling-normalised columns following Gao et al. (2025):
        #   llm_mean_nc  = sign(r) × r²  / NC_r²
        #   vlm_mean_nc  = sign(r) × r²  / NC_r²
        #   diff_mean_nc = vlm_mean_nc - llm_mean_nc
        # sign(r) × r² preserves direction for negative r (e.g. after residue removal).
        # The t-statistic and p-value are computed on raw r and are NC-invariant.
        if metric in nc_map:
            nc_r = nc_map[metric]
            nc_r2 = nc_r ** 2
            stats["noise_ceiling_r"] = nc_r
            stats["noise_ceiling_r2"] = nc_r2
            stats["llm_mean_nc"] = stats["llm_mean"].apply(
                lambda v: float(np.sign(v) * v ** 2 / nc_r2)
            )
            stats["vlm_mean_nc"] = stats["vlm_mean"].apply(
                lambda v: float(np.sign(v) * v ** 2 / nc_r2)
            )
            stats["diff_mean_nc"] = stats["vlm_mean_nc"] - stats["llm_mean_nc"]

        all_stats.append(stats)

        n_sig = int(stats["sig_fdr05"].sum()) if "sig_fdr05" in stats.columns else 0
        n_for_print = int(np.nanmax(stats["n_subjects"])) if "n_subjects" in stats.columns and len(stats) else 0
        print(f"\n  Layer-wise paired t-tests (n={n_for_print} aligned subjects max):")
        print(f"  FDR-significant layers (q<0.05): {n_sig} / {len(stats)}")

        if n_sig > 0:
            sig_rows = stats[stats["sig_fdr05"]]
            for _, row in sig_rows.iterrows():
                direction = "VLM>LLM" if row["diff_mean"] > 0 else "LLM>VLM"
                print(
                    f"    Layer {int(row['layer']):2d}: "
                    f"diff={row['diff_mean']:+.4f}  t={row['t_stat']:.3f}  "
                    f"q={row['p_fdr']:.4f}  d_z={row['cohens_dz']:.3f}  ({direction})"
                )

        # Top 5 layers by absolute difference
        top5 = stats.reindex(stats["diff_mean"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n  Top 5 layers by |VLM-LLM|:")
        for _, row in top5.iterrows():
            marker = "*" if row["sig_fdr05"] else " "
            print(
                f"   {marker} Layer {int(row['layer']):2d}: "
                f"LLM={row['llm_mean']:.4f}  VLM={row['vlm_mean']:.4f}  "
                f"diff={row['diff_mean']:+.4f}  p_raw={row['p_raw']:.4f}  "
                f"q={row['p_fdr']:.4f}  d_z={row['cohens_dz']:.3f}"
            )

        if args.layer is None:
            best_subjects, best_stats = compute_global_best_outputs(
                llm_data,
                vlm_data,
                metric,
                llm_key=llm_key,
                vlm_key=vlm_key,
                allow_legacy_positional_pairing=args.allow_legacy_positional_pairing,
            )
            best_analysis = "global_per_subject_best"
            print("\n  Global per-subject best-layer analysis:")
        else:
            best_subjects, best_stats = compute_fixed_layer_outputs(
                llm_data,
                vlm_data,
                metric,
                layer=args.layer,
                llm_key=llm_key,
                vlm_key=vlm_key,
                allow_legacy_positional_pairing=args.allow_legacy_positional_pairing,
            )
            best_analysis = "fixed_layer"
            print(f"\n  Fixed-layer analysis (layer {args.layer}):")
        if metric in nc_map:
            nc_r = nc_map[metric]
            nc_r2 = nc_r ** 2
            best_stats["noise_ceiling_r"] = nc_r
            best_stats["noise_ceiling_r2"] = nc_r2
            best_stats["llm_mean_nc"] = best_stats["llm_mean"].apply(
                lambda v: float(np.sign(v) * v ** 2 / nc_r2)
            )
            best_stats["vlm_mean_nc"] = best_stats["vlm_mean"].apply(
                lambda v: float(np.sign(v) * v ** 2 / nc_r2)
            )
            best_stats["diff_mean_nc"] = best_stats["vlm_mean_nc"] - best_stats["llm_mean_nc"]

        all_best_subjects.append(best_subjects)
        all_best_stats.append(best_stats)
        row = best_stats.iloc[0]
        print(
            f"    n={int(row['n_subjects'])}  "
            f"LLM={row['llm_mean']:.4f}  VLM={row['vlm_mean']:.4f}  "
            f"diff={row['diff_mean']:+.4f}  t={row['t_stat']:.3f}  "
            f"p={row['p']:.4f}  d_z={row['cohens_dz']:.3f}"
        )

        plot_best_layer_bar(
            best_stats, metric,
            compare_dir / f"saccade_best_bar_{metric}.png",
            use_nc=False,
            llm_key=llm_key,
            vlm_key=vlm_key,
            analysis=best_analysis,
        )
        if metric in nc_map:
            plot_best_layer_bar(
                best_stats, metric,
                compare_dir / f"saccade_best_bar_{metric}_nc.png",
                use_nc=True,
                llm_key=llm_key,
                vlm_key=vlm_key,
                analysis=best_analysis,
            )

        plot_layer_profile(
            stats, metric,
            compare_dir / f"saccade_layer_profile_{metric}.png",
            use_nc=False,
            llm_key=llm_key,
            vlm_key=vlm_key,
        )
        if metric in nc_map:
            plot_layer_profile(
                stats, metric,
                compare_dir / f"saccade_layer_profile_{metric}_nc.png",
                use_nc=True,
                llm_key=llm_key,
                vlm_key=vlm_key,
            )

    if all_stats:
        stats_all = pd.concat(all_stats, ignore_index=True)
        stats_all.to_csv(compare_dir / "saccade_stats.csv", index=False, float_format="%.6f")
        print(f"\n  Saved: {compare_dir / 'saccade_stats.csv'}")

    if all_best_subjects and args.layer is None:
        best_subjects_all = pd.concat(all_best_subjects, ignore_index=True)
        best_subjects_all.to_csv(
            compare_dir / "saccade_global_best_layer.csv",
            index=False,
            float_format="%.6f",
        )
        print(f"  Saved: {compare_dir / 'saccade_global_best_layer.csv'}")

    if all_best_stats and args.layer is None:
        best_stats_all = pd.concat(all_best_stats, ignore_index=True)
        best_stats_all.to_csv(
            compare_dir / "saccade_global_best_stats.csv",
            index=False,
            float_format="%.6f",
        )
        # Backward-compatible alias.
        best_stats_all.to_csv(
            compare_dir / "saccade_best_layer.csv",
            index=False,
            float_format="%.6f",
        )
        print(f"  Saved: {compare_dir / 'saccade_global_best_stats.csv'}")
        print(f"  Saved: {compare_dir / 'saccade_best_layer.csv'}")
    elif all_best_subjects and all_best_stats:
        fixed_subjects_all = pd.concat(all_best_subjects, ignore_index=True)
        fixed_stats_all = pd.concat(all_best_stats, ignore_index=True)
        fixed_subjects_all.to_csv(
            compare_dir / "saccade_fixed_layer_values.csv",
            index=False,
            float_format="%.6f",
        )
        fixed_stats_all.to_csv(
            compare_dir / "saccade_fixed_layer_stats.csv",
            index=False,
            float_format="%.6f",
        )
        # Backward-compatible alias.
        fixed_stats_all.to_csv(
            compare_dir / "saccade_best_layer.csv",
            index=False,
            float_format="%.6f",
        )
        print(f"  Saved: {compare_dir / 'saccade_fixed_layer_values.csv'}")
        print(f"  Saved: {compare_dir / 'saccade_fixed_layer_stats.csv'}")
        print(f"  Saved: {compare_dir / 'saccade_best_layer.csv'}")

    print("\nPhase 4a (saccade comparison) complete.")


if __name__ == "__main__":
    main()
