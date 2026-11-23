#!/usr/bin/env python3
"""
Continuous visualness moderation.

Per-sentence Fisher-z from existing OOF sufficient stats; WLS slope of
D = z_VLM - z_LLM on z-scored Lancaster visualness_mean, with log(n_s) and
article FE. 
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import RESULTS_DIR
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "visualness_analysis", _SCRIPT_DIR / "5_visualness_analysis.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot import 5_visualness_analysis.py")
_v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v)
_corr_from_sufficient_stats = _v._corr_from_sufficient_stats
_onesided_signflip = _v._onesided_signflip
_npz_scalar = _v._npz_scalar

_EPS = 1e-6

PAIRS = [
    ("qwen2_pair", "qwen2-7b", "qwen2-vl-7b"),
    ("deepseek_pair", "deepseek-llm-7b-base", "deepseek-vl-7b-base"),
    ("mistral_pair", "mistral-7b-v0.1", "idefics2-8b-base"),
]
PAIR_DIFFS = [
    ("pair_difference_qwen_minus_mistral", "qwen2_pair", "mistral_pair"),
    ("pair_difference_deepseek_minus_mistral", "deepseek_pair", "mistral_pair"),
]


def _fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -1.0 + _EPS, 1.0 - _EPS))


def _holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    out = [np.nan] * n
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * float(pvals[idx]))
        running = max(running, adj)
        out[idx] = running
    return out


def _effect_summary(x: np.ndarray, *, n_perm: int, seed: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    mean = float(np.mean(x)) if n else float("nan")
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    d_z = mean / sd if np.isfinite(sd) and sd > 0 else float("nan")
    if n >= 3 and np.isfinite(sd):
        se = sd / math.sqrt(n)
        tcrit = float(student_t.ppf(0.975, df=n - 1))
        ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
    else:
        ci_low = ci_high = float("nan")
    sf = _onesided_signflip(x, n_perm=n_perm, seed=seed)
    return {
        "n_subjects": n,
        "mean_beta": mean,
        "sd_beta": sd,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "cohens_dz": float(d_z),
        "p_one_sided_signflip": float(sf["p_one_sided"]),
    }


def _design(V: np.ndarray, logn: np.ndarray, articles: np.ndarray) -> np.ndarray:
    arts = np.sort(np.unique(articles))
    cols = [np.ones(V.shape[0], dtype=np.float64), V, logn]
    for a in arts[1:]:
        cols.append((articles == a).astype(np.float64))
    return np.column_stack(cols)


def _wls_beta_V(D: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """D (S,U), X (S,K), w (S,) -> beta_V (U,). Rows must be finite in D."""
    Ww = w[:, None]
    xtwx = X.T @ (Ww * X)
    rhs = X.T @ (Ww * D)
    coef = np.linalg.solve(xtwx, rhs)
    return coef[1]


def _load_eligible_meta(
    path: Path, coverage_threshold: float, min_matched_words: int = 3
) -> pd.DataFrame:
    meta = pd.read_csv(path)
    need = {"sentence_id", "article_id", "coverage", "n_matched_words", "visualness_mean"}
    if not need.issubset(meta.columns):
        raise ValueError(f"{path} missing columns: {sorted(need - set(meta.columns))}")
    m = meta.copy()
    m["sentence_id"] = m["sentence_id"].astype(str)
    m = m[(m["coverage"] >= coverage_threshold)
          & (m["n_matched_words"] >= min_matched_words)].copy()
    m = m[np.isfinite(m["visualness_mean"].to_numpy(dtype=np.float64))].copy()
    if m.empty:
        raise ValueError("no eligible sentences after coverage filter")
    v = m["visualness_mean"].to_numpy(dtype=np.float64)
    m["V"] = (v - v.mean()) / v.std(ddof=0)
    return m.reset_index(drop=True)


def _index_oof(oof_root: Path, model_key: str) -> dict[tuple[int, str], Path]:
    model_dir = oof_root / model_key
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    files = sorted(model_dir.rglob("*_oof_stats.npz"))
    if not files:
        raise FileNotFoundError(f"no OOF npz under {model_dir}")
    out: dict[tuple[int, str], Path] = {}
    layers: set[int] = set()
    for fp in files:
        with np.load(fp, allow_pickle=False) as npz:
            subj = int(_npz_scalar(npz, "subject"))
            hemi = str(_npz_scalar(npz, "hemi"))
            layers.add(int(_npz_scalar(npz, "layer")))
            key = (subj, hemi)
            if key in out:
                raise ValueError(f"duplicate OOF for {model_key} {key}: {out[key]} vs {fp}")
            out[key] = fp
    if len(layers) != 1:
        raise ValueError(f"{model_key} has multiple OOF layers: {sorted(layers)}")
    return out


def _sentence_z_from_npz(npz: np.lib.npyio.NpzFile, sent_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z[S,U], n_rows[S], unit_id[U]) aligned to sent_ids order."""
    sid = np.asarray(npz["sentence_id"]).astype(str)
    lookup = {s: i for i, s in enumerate(sid)}
    idx = np.array([lookup[s] for s in sent_ids], dtype=np.int32)
    n_rows = np.asarray(npz["n_rows"], dtype=np.int64)[idx]
    unit_id = np.asarray(npz["unit_id"], dtype=np.int32)
    sum_y = np.asarray(npz["sum_y"], dtype=np.float64)[idx]
    sum_pred = np.asarray(npz["sum_pred"], dtype=np.float64)[idx]
    sum_y2 = np.asarray(npz["sum_y2"], dtype=np.float64)[idx]
    sum_pred2 = np.asarray(npz["sum_pred2"], dtype=np.float64)[idx]
    sum_yp = np.asarray(npz["sum_y_pred"], dtype=np.float64)[idx]
    z = np.empty(sum_y.shape, dtype=np.float64)
    for i in range(sum_y.shape[0]):
        r = _corr_from_sufficient_stats(
            int(n_rows[i]), sum_y[i], sum_pred[i], sum_y2[i], sum_pred2[i], sum_yp[i]
        )
        z[i] = _fisher_z(r)
    return z, n_rows, unit_id


def _fit_subject_hemi(
    llm_fp: Path,
    vlm_fp: Path,
    sent_ids: np.ndarray,
    V: np.ndarray,
    articles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    with np.load(llm_fp, allow_pickle=False) as llm_npz, np.load(vlm_fp, allow_pickle=False) as vlm_npz:
        z_llm, n_rows, unit_id = _sentence_z_from_npz(llm_npz, sent_ids)
        z_vlm, n_rows_v, unit_id_v = _sentence_z_from_npz(vlm_npz, sent_ids)
        layer_llm = int(_npz_scalar(llm_npz, "layer"))
        layer_vlm = int(_npz_scalar(vlm_npz, "layer"))
    if not np.array_equal(n_rows, n_rows_v) or not np.array_equal(unit_id, unit_id_v):
        raise ValueError(f"LLM/VLM OOF mismatch: {llm_fp.name} vs {vlm_fp.name}")
    D = z_vlm - z_llm
    row_ok = np.isfinite(D).all(axis=1) & (n_rows >= 2)
    if int(row_ok.sum()) < 8:
        raise RuntimeError(f"too few finite sentences for {llm_fp}: {int(row_ok.sum())}")
    logn = np.log(n_rows.astype(np.float64))
    X = _design(V[row_ok], logn[row_ok], articles[row_ok])
    beta = _wls_beta_V(D[row_ok], X, n_rows[row_ok].astype(np.float64))
    return beta, unit_id, layer_llm, layer_vlm, D, n_rows.astype(np.float64)


def run_pair(
    oof_root: Path,
    meta: pd.DataFrame,
    *,
    pair_tag: str,
    llm_key: str,
    vlm_key: str,
    out_dir: Path,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    llm_idx = _index_oof(oof_root, llm_key)
    vlm_idx = _index_oof(oof_root, vlm_key)
    keys = sorted(set(llm_idx) & set(vlm_idx))
    if not keys:
        raise RuntimeError(f"no overlapping OOF keys for {llm_key}/{vlm_key}")

    sent_ids = meta["sentence_id"].to_numpy(dtype=str)
    V = meta["V"].to_numpy(dtype=np.float64)
    articles = meta["article_id"].to_numpy(dtype=np.int16)

    unit_rows: list[pd.DataFrame] = []
    layers_llm: set[int] = set()
    layers_vlm: set[int] = set()
    by_subj: dict[int, list[np.ndarray]] = {}
    cortex_sum: dict[int, np.ndarray] = {}
    cortex_nhemi: dict[int, int] = {}
    cortex_n_cache: dict[int, np.ndarray] = {}

    for i_key, (subj, hemi) in enumerate(keys):
        if (i_key + 1) % 20 == 0 or i_key == 0:
            print(f"  [{pair_tag}] {i_key + 1}/{len(keys)} subject-hemi fits")
        beta, unit_id, lay_l, lay_v, D, n_rows = _fit_subject_hemi(
            llm_idx[(subj, hemi)], vlm_idx[(subj, hemi)], sent_ids, V, articles
        )
        layers_llm.add(lay_l)
        layers_vlm.add(lay_v)
        unit_rows.append(pd.DataFrame({
            "subject": np.full(unit_id.shape[0], subj, dtype=np.int16),
            "unit_id": unit_id,
            "beta": beta,
        }))
        by_subj.setdefault(subj, []).append(beta)
        cmean = np.nanmean(D, axis=1)
        cortex_sum[subj] = cortex_sum.get(subj, 0.0) + cmean
        cortex_nhemi[subj] = cortex_nhemi.get(subj, 0) + 1
        cortex_n_cache[subj] = n_rows

    unit_df = pd.concat(unit_rows, ignore_index=True)
    subj_means = []
    for subj in sorted(by_subj):
        b = np.concatenate(by_subj[subj])
        subj_means.append({"subject": subj, "n_units": int(b.size), "beta_mean": float(np.nanmean(b))})
    subj_df = pd.DataFrame(subj_means)

    stats = _effect_summary(subj_df["beta_mean"].to_numpy(), n_perm=n_perm, seed=seed)
    stats.update({
        "pair_tag": pair_tag,
        "llm_key": llm_key,
        "vlm_key": vlm_key,
        "layer_llm": sorted(layers_llm),
        "layer_vlm": sorted(layers_vlm),
        "n_eligible_sentences": int(len(meta)),
        "eligible_sentence_ids": meta["sentence_id"].tolist(),
        "formula": "D ~ 1 + V + log(n_rows) + C(article); WLS weight=n_rows; D=z_VLM-z_LLM",
        "V_definition": "zscore(visualness_mean) over coverage-eligible sentences",
    })

    # Article LOO on cortex-mean D (stability of whole-cortex beta_i)
    loo_rows = []
    cortex_D_cache = {s: cortex_sum[s] / cortex_nhemi[s] for s in cortex_sum}

    for drop_a in sorted(np.unique(articles).tolist()):
        keep = articles != drop_a
        betas = []
        for subj, cD in cortex_D_cache.items():
            n_rows = cortex_n_cache[subj]
            row_ok = keep & np.isfinite(cD) & (n_rows >= 2)
            X = _design(V[row_ok], np.log(n_rows[row_ok]), articles[row_ok])
            y = cD[row_ok][:, None]
            b = _wls_beta_V(y, X, n_rows[row_ok])
            betas.append(float(b[0]))
        betas_a = np.asarray(betas, dtype=np.float64)
        loo_rows.append({
            "dropped_article": int(drop_a),
            "n_subjects": int(betas_a.size),
            "mean_beta": float(np.mean(betas_a)),
            "n_positive_subjects": int(np.sum(betas_a > 0)),
            "sign_mean": int(np.sign(np.mean(betas_a))),
        })
    loo_df = pd.DataFrame(loo_rows)
    stats["article_loo_sign_range"] = sorted(set(int(x) for x in loo_df["sign_mean"].tolist()))
    stats["article_loo_mean_beta_min"] = float(loo_df["mean_beta"].min())
    stats["article_loo_mean_beta_max"] = float(loo_df["mean_beta"].max())

    out_dir.mkdir(parents=True, exist_ok=True)
    unit_df.to_csv(out_dir / "visualness_unit_slope.csv", index=False, float_format="%.8g")
    subj_df.to_csv(out_dir / "visualness_subject_slope_summary.csv", index=False, float_format="%.8g")
    loo_df.to_csv(out_dir / "article_loo_stability.csv", index=False, float_format="%.8g")
    cfg = {
        "pair_tag": pair_tag,
        "llm_key": llm_key,
        "vlm_key": vlm_key,
        "layer_llm": sorted(layers_llm),
        "layer_vlm": sorted(layers_vlm),
        "n_eligible_sentences": int(len(meta)),
        "eligible_sentence_ids": meta["sentence_id"].tolist(),
        "n_perm": n_perm,
        "seed": seed,
        "formula": stats["formula"],
        "V_definition": stats["V_definition"],
        "oof_policy": "accept_layers_on_disk",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "continuous_slope_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"[{pair_tag}] n={stats['n_subjects']} mean_beta={stats['mean_beta']:+.6g} "
        f"d_z={stats['cohens_dz']:+.3f} p={stats['p_one_sided_signflip']:.4g} "
        f"layers llm={sorted(layers_llm)} vlm={sorted(layers_vlm)}"
    )
    return unit_df, subj_df, stats


def _pair_difference(
    unit_a: pd.DataFrame,
    unit_b: pd.DataFrame,
    subj_a: pd.DataFrame,
    subj_b: pd.DataFrame,
    *,
    out_dir: Path,
    tag: str,
    n_perm: int,
    seed: int,
) -> dict:
    a = unit_a.rename(columns={"beta": "beta_a"})
    b = unit_b.rename(columns={"beta": "beta_b"})
    m = a.merge(b, on=["subject", "unit_id"], how="inner")
    if m.empty:
        raise RuntimeError(f"no overlapping units for {tag}")
    m["beta"] = m["beta_a"] - m["beta_b"]
    unit_out = m[["subject", "unit_id", "beta"]]
    sa = subj_a.rename(columns={"beta_mean": "beta_a"})
    sb = subj_b.rename(columns={"beta_mean": "beta_b"})
    sm = sa.merge(sb, on="subject", how="inner")
    sm["beta_mean"] = sm["beta_a"] - sm["beta_b"]
    stats = _effect_summary(sm["beta_mean"].to_numpy(), n_perm=n_perm, seed=seed)
    stats["pair_tag"] = tag
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_out.to_csv(out_dir / "visualness_unit_slope.csv", index=False, float_format="%.8g")
    sm[["subject", "beta_mean"]].to_csv(
        out_dir / "visualness_subject_slope_summary.csv", index=False, float_format="%.8g"
    )
    (out_dir / "continuous_slope_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"[{tag}] n={stats['n_subjects']} mean_delta_beta={stats['mean_beta']:+.6g} "
        f"p={stats['p_one_sided_signflip']:.4g}"
    )
    return stats


def run_smoke() -> None:
    rng = np.random.default_rng(0)
    n_s, n_u, n_subj = 40, 32, 12
    articles = np.repeat(np.arange(1, 5), 10)
    V = rng.normal(size=n_s)
    V = (V - V.mean()) / V.std()
    logn = np.log(rng.integers(20, 80, size=n_s).astype(np.float64))
    X = _design(V, logn, articles)
    w = np.exp(logn)

    # known positive slope
    true_beta = 0.05
    betas_pos = []
    for _ in range(n_subj):
        D = true_beta * V[:, None] + 0.01 * rng.normal(size=(n_s, n_u))
        betas_pos.append(float(np.mean(_wls_beta_V(D, X, w))))
    stats_pos = _effect_summary(np.asarray(betas_pos), n_perm=2000, seed=1)
    assert stats_pos["mean_beta"] > 0 and stats_pos["p_one_sided_signflip"] < 0.05, stats_pos

    # zero slope
    betas_zero = []
    for _ in range(n_subj):
        D = 0.01 * rng.normal(size=(n_s, n_u))
        betas_zero.append(float(np.mean(_wls_beta_V(D, X, w))))
    stats_zero = _effect_summary(np.asarray(betas_zero), n_perm=2000, seed=2)
    assert stats_zero["p_one_sided_signflip"] > 0.1, stats_zero

    # single-article driven: only article 1 carries the slope
    betas_full = []
    betas_drop1 = []
    for _ in range(n_subj):
        D = 0.01 * rng.normal(size=(n_s, n_u))
        D[articles == 1] += 0.2 * V[articles == 1, None]
        betas_full.append(float(np.mean(_wls_beta_V(D, X, w))))
        keep = articles != 1
        Xk = _design(V[keep], logn[keep], articles[keep])
        betas_drop1.append(float(np.mean(_wls_beta_V(D[keep], Xk, w[keep]))))
    assert np.mean(betas_full) > 0
    assert np.sign(np.mean(betas_full)) != np.sign(np.mean(betas_drop1)) or abs(np.mean(betas_drop1)) < 0.25 * abs(np.mean(betas_full))
    print("[smoke] positive / zero / article-driven checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous visualness moderation.")
    parser.add_argument("--oof_root", type=Path, default=RESULTS_DIR / "fmri" / "oof")
    parser.add_argument(
        "--sent_meta",
        type=Path,
        default=RESULTS_DIR / "visualness" / "sentence_visualness_groups.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "visualness_continuous")
    parser.add_argument("--coverage_threshold", type=float, default=0.75)
    parser.add_argument("--min_matched_words", type=int, default=3)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke()
        return
    if not (0 <= args.coverage_threshold <= 1):
        raise ValueError("--coverage_threshold must be in [0, 1]")
    if args.min_matched_words < 0:
        raise ValueError("--min_matched_words must be >= 0")
    if args.n_perm <= 0:
        raise ValueError("--n_perm must be > 0")

    t0 = time.time()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    meta = _load_eligible_meta(args.sent_meta, args.coverage_threshold, args.min_matched_words)
    print("=" * 72)
    print("Continuous visualness moderation")
    print(f"  oof_root : {args.oof_root}")
    print(f"  sent_meta: {args.sent_meta}")
    print(f"  out_dir  : {out_root}")
    print(f"  eligible : {len(meta)} sentences "
          f"(coverage>={args.coverage_threshold}, matched words>={args.min_matched_words})")
    print("=" * 72)

    pair_units: dict[str, pd.DataFrame] = {}
    pair_subj: dict[str, pd.DataFrame] = {}
    all_stats: list[dict] = []
    for pair_tag, llm, vlm in PAIRS:
        unit_df, subj_df, stats = run_pair(
            args.oof_root.resolve(),
            meta,
            pair_tag=pair_tag,
            llm_key=llm,
            vlm_key=vlm,
            out_dir=out_root / pair_tag,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        pair_units[pair_tag] = unit_df
        pair_subj[pair_tag] = subj_df
        all_stats.append(stats)

    pvals = [s["p_one_sided_signflip"] for s in all_stats]
    holm = _holm(pvals)
    for s, p_h in zip(all_stats, holm):
        s["p_holm"] = float(p_h)

    for tag, a, b in PAIR_DIFFS:
        st = _pair_difference(
            pair_units[a],
            pair_units[b],
            pair_subj[a],
            pair_subj[b],
            out_dir=out_root / tag,
            tag=tag,
            n_perm=args.n_perm,
            seed=args.seed + 17,
        )
        all_stats.append(st)

    rows = []
    for s in all_stats:
        rows.append({
            "pair_tag": s.get("pair_tag"),
            "llm_key": s.get("llm_key"),
            "vlm_key": s.get("vlm_key"),
            "n_subjects": s["n_subjects"],
            "mean_beta": s["mean_beta"],
            "ci95_low": s["ci95_low"],
            "ci95_high": s["ci95_high"],
            "cohens_dz": s["cohens_dz"],
            "p_one_sided_signflip": s["p_one_sided_signflip"],
            "p_holm": s.get("p_holm"),
            "layer_llm": s.get("layer_llm"),
            "layer_vlm": s.get("layer_vlm"),
        })
        tag = s.get("pair_tag")
        if tag in {p[0] for p in PAIRS}:
            (out_root / tag / "continuous_slope_stats.json").write_text(
                json.dumps(s, indent=2), encoding="utf-8"
            )

    pd.DataFrame(rows).to_csv(
        out_root / "continuous_slope_stats_all_pairs.csv", index=False, float_format="%.8g"
    )
    root_cfg = {
        "coverage_threshold": args.coverage_threshold,
        "min_matched_words": args.min_matched_words,
        "n_eligible_sentences": int(len(meta)),
        "sent_meta": str(args.sent_meta.resolve()),
        "oof_root": str(args.oof_root.resolve()),
        "oof_policy": "accept_layers_on_disk",
        "n_perm": args.n_perm,
        "seed": args.seed,
        "pairs": [{"tag": t, "llm": a, "vlm": b} for t, a, b in PAIRS],
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_root / "analysis_config.json").write_text(json.dumps(root_cfg, indent=2), encoding="utf-8")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    print(f"Saved: {out_root / 'continuous_slope_stats_all_pairs.csv'}")


if __name__ == "__main__":
    main()
