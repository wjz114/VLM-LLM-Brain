#!/usr/bin/env python3
"""
Group-level fMRI null threshold from Phase-3 corr maps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (  # noqa: E402
    FMRI_SUBJECTS,
    N_VERTICES,
    RESULTS_DIR,
)

FMRI_GROUP_NULL_DIR = RESULTS_DIR / "fmri_group_null"
FMRI_DIR = RESULTS_DIR / "fmri"


def _parse_subjects(subjects_arg: str | None, all_subjects: bool) -> list[int]:
    if all_subjects:
        return FMRI_SUBJECTS
    if subjects_arg:
        return [int(x.strip()) for x in subjects_arg.split(",") if x.strip()]
    return FMRI_SUBJECTS[:2]


def _parse_hemis(hemis_arg: str) -> list[str]:
    hemis = [h.strip().upper() for h in hemis_arg.split(",") if h.strip()]
    bad = [h for h in hemis if h not in {"L", "R"}]
    if bad:
        raise ValueError(f"Invalid hemisphere(s): {bad}. Use L and/or R.")
    if not hemis:
        raise ValueError("At least one hemisphere is required.")
    return hemis


def fdr_bh(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return np.zeros(p.shape, dtype=bool), q

    vals = p[finite]
    n = vals.size
    order = np.argsort(vals)
    ranked = vals[order]
    q_vals = np.empty(n, dtype=np.float64)
    q_vals[order] = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q_vals[order[i]] = min(q_vals[order[i]], q_vals[order[i + 1]])
    q_vals = np.clip(q_vals, 0.0, 1.0)
    q[finite] = q_vals
    return q <= alpha, q


def _batch_specs(n_perm: int, null_batch: int) -> list[tuple[int, int]]:
    return [
        (start, min(null_batch, n_perm - start))
        for start in range(0, n_perm, null_batch)
    ]


def _nanpercentile_axis0_chunks(
    arr: np.ndarray,
    percentile: float,
    vertex_batch: int,
) -> np.ndarray:
    out = np.full(N_VERTICES, np.nan, dtype=np.float32)
    for lo in range(0, N_VERTICES, vertex_batch):
        hi = min(lo + vertex_batch, N_VERTICES)
        out[lo:hi] = np.nanpercentile(arr[:, lo:hi], percentile, axis=0).astype(np.float32)
    return out


def _surface_adjacency(hemi: str) -> list[np.ndarray]:
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    mesh = fs["pial_left"] if hemi == "L" else fs["pial_right"]
    coords, faces = load_surf_mesh(mesh)
    if coords.shape[0] != N_VERTICES:
        raise ValueError(f"Mesh vertices {coords.shape[0]} != N_VERTICES {N_VERTICES}")

    neighbors = [set() for _ in range(N_VERTICES)]
    for a, b, c in np.asarray(faces, dtype=np.int32):
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return [np.fromiter(sorted(nb), dtype=np.int32) for nb in neighbors]


def _positive_clusters(
    mask: np.ndarray,
    mass_values: np.ndarray,
    adjacency: list[np.ndarray],
    min_vertices: int,
) -> list[tuple[np.ndarray, float]]:
    visited = np.zeros(mask.size, dtype=bool)
    clusters: list[tuple[np.ndarray, float]] = []
    for seed in np.flatnonzero(mask):
        seed = int(seed)
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        verts: list[int] = []
        while stack:
            v = stack.pop()
            verts.append(v)
            for nb in adjacency[v]:
                nb = int(nb)
                if mask[nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(verts) >= min_vertices:
            idx = np.asarray(verts, dtype=np.int32)
            clusters.append((idx, float(np.nansum(mass_values[idx]))))
    return clusters


def _empty_cluster_result() -> dict[str, np.ndarray]:
    return {
        "cluster_mask": np.zeros(N_VERTICES, dtype=bool),
        "cluster_id": np.zeros(N_VERTICES, dtype=np.int32),
        "cluster_forming_threshold": np.full(N_VERTICES, np.nan, dtype=np.float32),
        "cluster_mass": np.array([], dtype=np.float32),
        "cluster_size": np.array([], dtype=np.int32),
        "cluster_p": np.array([], dtype=np.float32),
        "cluster_null_max_mass": np.array([], dtype=np.float32),
    }


def _centered_cluster_mass_correction(
    observed: np.ndarray,
    null_maps: np.ndarray,
    threshold: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    alpha: float,
    min_vertices: int,
) -> dict[str, np.ndarray]:
    observed_excess = observed - threshold
    observed_clusters = _positive_clusters(
        np.isfinite(observed_excess) & (observed_excess > 0),
        observed_excess,
        adjacency,
        min_vertices,
    )

    null_max_mass = np.zeros(null_maps.shape[0], dtype=np.float32)
    for i in range(null_maps.shape[0]):
        perm_excess = np.asarray(null_maps[i], dtype=np.float32) - threshold
        perm_clusters = _positive_clusters(
            np.isfinite(perm_excess) & (perm_excess > 0),
            perm_excess,
            adjacency,
            min_vertices,
        )
        if perm_clusters:
            null_max_mass[i] = max(mass for _, mass in perm_clusters)

    cluster_mask = np.zeros(N_VERTICES, dtype=bool)
    cluster_id = np.zeros(N_VERTICES, dtype=np.int32)
    masses: list[float] = []
    sizes: list[int] = []
    p_values: list[float] = []
    next_id = 1
    for idx, mass in observed_clusters:
        p = float((np.sum(null_max_mass >= mass) + 1.0) / (null_max_mass.size + 1.0))
        masses.append(mass)
        sizes.append(int(idx.size))
        p_values.append(p)
        if p < alpha:
            cluster_mask[idx] = True
            cluster_id[idx] = next_id
            next_id += 1

    return {
        "cluster_mask": cluster_mask,
        "cluster_id": cluster_id,
        "cluster_forming_threshold": threshold.astype(np.float32),
        "cluster_mass": np.asarray(masses, dtype=np.float32),
        "cluster_size": np.asarray(sizes, dtype=np.int32),
        "cluster_p": np.asarray(p_values, dtype=np.float32),
        "cluster_null_max_mass": null_max_mass,
    }


def _print_null_diagnostics(label: str, observed: np.ndarray, p975: np.ndarray, p_values: np.ndarray) -> dict[str, float]:
    diff = observed - p975
    finite_diff = diff[np.isfinite(diff)]
    finite_p = p_values[np.isfinite(p_values)]
    if finite_diff.size:
        diff_min = float(np.nanmin(finite_diff))
        diff_median = float(np.nanmedian(finite_diff))
        diff_max = float(np.nanmax(finite_diff))
    else:
        diff_min = diff_median = diff_max = float("nan")
    if finite_p.size:
        p_q = np.quantile(finite_p, [0.0, 0.01, 0.05, 0.5, 0.95, 1.0])
    else:
        p_q = np.full(6, np.nan, dtype=np.float64)
    print(
        f"  {label}: obs-p975 min/median/max="
        f"{diff_min:.6g}/{diff_median:.6g}/{diff_max:.6g}; "
        f"p quantiles min/1%/5%/50%/95%/max="
        f"{p_q[0]:.6g}/{p_q[1]:.6g}/{p_q[2]:.6g}/{p_q[3]:.6g}/{p_q[4]:.6g}/{p_q[5]:.6g}"
    )
    print(f"  {label}: obs-p975 max/median={diff_max:.6g}/{diff_median:.6g}")
    return {
        f"{label}_diff_min": diff_min,
        f"{label}_diff_median": diff_median,
        f"{label}_diff_max": diff_max,
        f"{label}_p_min": float(p_q[0]),
        f"{label}_p_q01": float(p_q[1]),
        f"{label}_p_q05": float(p_q[2]),
        f"{label}_p_median": float(p_q[3]),
        f"{label}_p_q95": float(p_q[4]),
        f"{label}_p_max": float(p_q[5]),
    }


def _parse_layers_arg(layers_arg: str | None, model_key: str) -> list[int]:
    if layers_arg:
        layers: list[int] = []
        for tok in layers_arg.replace(",", " ").split():
            layer = int(tok)
            if layer < 0:
                raise ValueError(f"Layer index must be >= 0: {layer}")
            if layer not in layers:
                layers.append(layer)
        if not layers:
            raise ValueError("--null_layers did not contain any layer index")
        return layers

    model_dir = FMRI_DIR / model_key
    layers = []
    for path in model_dir.glob("layer*"):
        suffix = path.name[5:]
        if path.is_dir() and suffix.isdigit():
            layer = int(suffix)
            if layer > 0:
                layers.append(layer)
    if not layers:
        raise FileNotFoundError(f"No layerXX directories with layer > 0 found under {model_dir}")
    return sorted(set(layers))


def _corr_path(model_key: str, layer: int, hemi: str, subj_id: int) -> Path:
    return FMRI_DIR / model_key / f"layer{layer:02d}" / f"sub{subj_id:02d}_hemi{hemi}_corr.npy"


def _load_corr_map(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.shape != (N_VERTICES,):
        raise ValueError(f"Unexpected corr shape for {path}: {arr.shape}")
    return arr


def _load_corr_stacks(
    *,
    model_key: str,
    observed_layer: int | None,
    null_layers: list[int],
    hemi: str,
    subjects: list[int],
) -> tuple[np.ndarray | None, np.ndarray, list[int]]:
    obs_rows: list[np.ndarray] = []
    null_rows: list[np.ndarray] = []
    loaded_subjects: list[int] = []
    missing = 0

    for sid in subjects:
        obs = None
        if observed_layer is not None:
            obs = _load_corr_map(_corr_path(model_key, observed_layer, hemi, sid))
            if obs is None:
                missing += 1
                continue

        layer_rows: list[np.ndarray] = []
        complete = True
        for layer in null_layers:
            arr = _load_corr_map(_corr_path(model_key, layer, hemi, sid))
            if arr is None:
                complete = False
                break
            layer_rows.append(arr)
        if not complete:
            missing += 1
            continue

        if obs is not None:
            obs_rows.append(obs)
        null_rows.append(np.stack(layer_rows, axis=0))
        loaded_subjects.append(int(sid))

    if missing:
        print(f"  hemi-{hemi}: skipped {missing} subject(s) with incomplete corr maps")
    if len(loaded_subjects) < 3:
        raise ValueError(f"Need at least 3 complete subjects for hemi-{hemi}; got {len(loaded_subjects)}")

    return (
        np.stack(obs_rows, axis=0).astype(np.float32) if obs_rows else None,
        np.stack(null_rows, axis=0).astype(np.float32),
        loaded_subjects,
    )


def _spatial_center_subject_layer(stack: np.ndarray) -> np.ndarray:
    mean = np.nanmean(stack, axis=-1, keepdims=True).astype(np.float32)
    return (stack - mean).astype(np.float32)


def _best_layer_observed_stack(
    select_stack: np.ndarray,
    value_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layer_score = np.nanmean(select_stack, axis=2).astype(np.float32)
    layer_score = np.nan_to_num(layer_score, nan=-np.inf)
    best_idx = np.argmax(layer_score, axis=1).astype(np.int32)
    rows = [value_stack[subj_i, int(best_idx[subj_i])] for subj_i in range(value_stack.shape[0])]
    return np.stack(rows, axis=0).astype(np.float32), best_idx


def _output_stem(args: argparse.Namespace) -> str:
    stem = f"group_null_layer{args.layer:02d}"
    if args.observed_mode == "best_layer":
        stem += "_observed_best_layer"
    if args.spatial_center == "subject_layer":
        stem += "_spatial_center"
    return stem


def _subject_sign_flip_best_layer_null(
    select_stack: np.ndarray,
    value_stack: np.ndarray,
    observed_group_mean_r: np.ndarray,
    *,
    n_perm: int,
    null_batch: int,
    seed: int,
    vertex_batch: int,
    tmp_path: Path,
    save_full: bool,
    cluster_adjacency: list[np.ndarray] | None,
    cluster_forming_percentile: float,
    cluster_alpha: float,
    cluster_min_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if select_stack.shape != value_stack.shape:
        raise ValueError(f"select/value stack shape mismatch: {select_stack.shape} vs {value_stack.shape}")
    n_subjects, n_layers, n_vertices = value_stack.shape
    if n_vertices != N_VERTICES:
        raise ValueError(f"Expected {N_VERTICES} vertices, got {n_vertices}")

    layer_score = np.nanmean(select_stack, axis=2).astype(np.float32)
    layer_score = np.nan_to_num(layer_score, nan=-np.inf)
    if tmp_path.exists():
        tmp_path.unlink()
    tmp = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(n_perm, N_VERTICES))
    exceed_counts = np.zeros(N_VERTICES, dtype=np.int32)
    null_sum = np.zeros(N_VERTICES, dtype=np.float64)
    null_count = np.zeros(N_VERTICES, dtype=np.int32)
    best_layer_counts = np.zeros((n_subjects, n_layers), dtype=np.int32)
    batch_specs = _batch_specs(n_perm, null_batch)

    for i, (batch_start, batch_size) in enumerate(batch_specs, start=1):
        rng = np.random.default_rng(seed + batch_start * 104_729)
        signs = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=(batch_size, n_subjects, n_layers),
        )
        best_idx = np.argmax(signs * layer_score[None, :, :], axis=2)

        group_sum = np.zeros((batch_size, N_VERTICES), dtype=np.float32)
        group_n = np.zeros((batch_size, N_VERTICES), dtype=np.int16)
        for subj_i in range(n_subjects):
            chosen = best_idx[:, subj_i]
            chosen_sign = signs[np.arange(batch_size), subj_i, chosen]
            maps = value_stack[subj_i, chosen, :] * chosen_sign[:, None]
            finite = np.isfinite(maps)
            group_sum[finite] += maps[finite]
            group_n[finite] += 1
            best_layer_counts[subj_i] += np.bincount(chosen, minlength=n_layers).astype(np.int32)

        r_group_batch = np.divide(
            group_sum,
            group_n,
            out=np.full((batch_size, N_VERTICES), np.nan, dtype=np.float32),
            where=group_n > 0,
        )
        batch_end = batch_start + batch_size
        tmp[batch_start:batch_end] = r_group_batch
        finite = np.isfinite(r_group_batch)
        valid_cmp = np.isfinite(observed_group_mean_r)[None, :] & finite
        exceed_counts += np.sum(
            (r_group_batch >= observed_group_mean_r[None, :]) & valid_cmp,
            axis=0,
            dtype=np.int32,
        )
        null_sum += np.sum(np.where(finite, r_group_batch, 0.0), axis=0, dtype=np.float64)
        null_count += np.sum(finite, axis=0, dtype=np.int32)
        print(f"  best-layer sign-flip batch {i}/{len(batch_specs)} done")

    tmp.flush()
    p975 = _nanpercentile_axis0_chunks(tmp, 97.5, vertex_batch)
    cluster_result = _empty_cluster_result()
    if cluster_adjacency is not None:
        print(
            "  cluster correction: "
            f"forming=null p{cluster_forming_percentile:g}, "
            f"alpha={cluster_alpha:g}, min_vertices={cluster_min_vertices}"
        )
        cluster_threshold = _nanpercentile_axis0_chunks(tmp, cluster_forming_percentile, vertex_batch)
        cluster_result = _centered_cluster_mass_correction(
            observed_group_mean_r,
            tmp,
            cluster_threshold,
            cluster_adjacency,
            alpha=cluster_alpha,
            min_vertices=cluster_min_vertices,
        )
    mean = np.divide(
        null_sum,
        null_count,
        out=np.full(N_VERTICES, np.nan, dtype=np.float64),
        where=null_count > 0,
    ).astype(np.float32)
    p = np.full(N_VERTICES, np.nan, dtype=np.float32)
    obs_valid = np.isfinite(observed_group_mean_r)
    p[obs_valid] = (exceed_counts[obs_valid] + 1.0) / (n_perm + 1.0)
    null_dist = np.asarray(tmp, dtype=np.float32).copy() if save_full else np.array([], dtype=np.float32)
    del tmp
    if tmp_path.exists():
        tmp_path.unlink()
    return mean, p975, p, exceed_counts, null_dist, best_layer_counts, cluster_result


def _run_hemi(
    *,
    args: argparse.Namespace,
    hemi: str,
    null_layers: list[int],
    subjects: list[int],
    out_dir: Path,
    cluster_adjacency: list[np.ndarray] | None,
) -> dict:
    out_stem = _output_stem(args)
    out_path = out_dir / f"{out_stem}_hemi{hemi}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"hemi-{hemi}: {out_path.name} exists; use --overwrite to recompute")
        with np.load(out_path) as data:
            return {
                "hemi": hemi,
                "n_subjects": int(np.asarray(data["n_subjects"]).ravel()[0]),
                "mean_observed_r": float(np.nanmean(data["observed_group_mean_r"])),
                "mean_null_p975_r": float(np.nanmean(data["null_group_p975_r"])),
                "n_vertices_threshold_sig": int(np.asarray(data["threshold_mask"], dtype=bool).sum()),
                "n_vertices_fdr_sig": int(np.asarray(data["fdr_mask"], dtype=bool).sum()),
            }

    print(f"\n=== hemi-{hemi}: load Phase-3 corr maps ===")
    obs_stack, null_stack, subject_ids = _load_corr_stacks(
        model_key=args.model,
        observed_layer=args.layer if args.observed_mode == "fixed_layer" else None,
        null_layers=null_layers,
        hemi=hemi,
        subjects=subjects,
    )
    raw_null_stack = null_stack
    value_null_stack = (
        _spatial_center_subject_layer(raw_null_stack)
        if args.spatial_center == "subject_layer"
        else raw_null_stack
    )
    if args.observed_mode == "best_layer":
        obs_stack, observed_best_layer_idx = _best_layer_observed_stack(raw_null_stack, value_null_stack)
    else:
        if obs_stack is None:
            raise RuntimeError("fixed_layer observed mode requires observed corr maps")
        if args.spatial_center == "subject_layer":
            obs_stack = _spatial_center_subject_layer(obs_stack)
        observed_best_layer_idx = np.full(len(subject_ids), args.layer, dtype=np.int32)
    observed_group_mean_r = np.nanmean(obs_stack, axis=0).astype(np.float32)
    observed_label = "best_layer" if args.observed_mode == "best_layer" else f"layer{args.layer:02d}"
    print(
        f"  loaded subjects={len(subject_ids)} layers={len(null_layers)} "
        f"observed={observed_label} spatial_center={args.spatial_center} "
        f"obs_mean={np.nanmean(observed_group_mean_r):.6f}"
    )

    print(f"=== hemi-{hemi}: null-aware best-layer sign-flip ({args.n_perm}) ===")
    null_group_mean_r, null_group_p975_r, p_value_pos, _, null_dist, best_layer_counts, cluster_result = (
        _subject_sign_flip_best_layer_null(
            raw_null_stack,
            value_null_stack,
            observed_group_mean_r,
            n_perm=args.n_perm,
            null_batch=args.null_batch,
            seed=args.seed + (0 if hemi == "L" else 50_000_017) + args.layer * 1_000_003,
            vertex_batch=args.vertex_batch,
            tmp_path=out_dir / f"group_null_layer{args.layer:02d}_hemi{hemi}_best_layer_signflip.tmp.dat",
            save_full=args.save_group_null,
            cluster_adjacency=cluster_adjacency,
            cluster_forming_percentile=args.cluster_forming_percentile,
            cluster_alpha=args.cluster_alpha,
            cluster_min_vertices=args.cluster_min_vertices,
        )
    )
    fdr_mask, q = fdr_bh(p_value_pos, args.alpha)
    uncorrected_mask = p_value_pos < args.alpha
    threshold_mask = observed_group_mean_r > null_group_p975_r
    subject_diag = _print_null_diagnostics(
        "best_layer_signflip",
        observed_group_mean_r,
        null_group_p975_r,
        p_value_pos,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observed_group_mean_r=observed_group_mean_r.astype(np.float32),
        null_group_mean_r=null_group_mean_r.astype(np.float32),
        null_group_p975_r=null_group_p975_r.astype(np.float32),
        p_value_pos=p_value_pos.astype(np.float32),
        threshold_mask=threshold_mask.astype(bool),
        fdr_mask=np.asarray(fdr_mask, dtype=bool),
        mean_r=observed_group_mean_r.astype(np.float32),
        p=p_value_pos.astype(np.float32),
        q=np.asarray(q, dtype=np.float32),
        uncorrected_mask=np.asarray(uncorrected_mask, dtype=bool),
        null_group_r=null_dist,
        null_source=np.asarray(["null_aware_best_layer_sign_flip"]),
        cluster_method=np.asarray([args.cluster_method]),
        cluster_alpha=np.asarray([args.cluster_alpha], dtype=np.float32),
        cluster_forming_percentile=np.asarray([args.cluster_forming_percentile], dtype=np.float32),
        cluster_min_vertices=np.asarray([args.cluster_min_vertices], dtype=np.int32),
        cluster_mask=cluster_result["cluster_mask"].astype(bool),
        cluster_id=cluster_result["cluster_id"].astype(np.int32),
        cluster_forming_threshold=cluster_result["cluster_forming_threshold"].astype(np.float32),
        cluster_mass=cluster_result["cluster_mass"].astype(np.float32),
        cluster_size=cluster_result["cluster_size"].astype(np.int32),
        cluster_p=cluster_result["cluster_p"].astype(np.float32),
        cluster_null_max_mass=cluster_result["cluster_null_max_mass"].astype(np.float32),
        null_layers=np.asarray(null_layers, dtype=np.int32),
        null_best_layer_mode=np.asarray(["subject_global_mean"]),
        best_layer_counts=best_layer_counts.astype(np.int32),
        observed_mode=np.asarray([args.observed_mode]),
        observed_best_layer_idx=observed_best_layer_idx.astype(np.int32),
        spatial_center=np.asarray([args.spatial_center]),
        subjects=np.asarray(subject_ids, dtype=np.int32),
        n_subjects=np.asarray([len(subject_ids)], dtype=np.int32),
        model=np.asarray([args.model]),
        layer=np.asarray([args.layer], dtype=np.int32),
        hemi=np.asarray([hemi], dtype="<U1"),
        n_perm=np.asarray([args.n_perm], dtype=np.int32),
        seed=np.asarray([args.seed], dtype=np.int64),
        alpha=np.asarray([args.alpha], dtype=np.float32),
    )

    summary = {
        "hemi": hemi,
        "n_subjects": len(subject_ids),
        "mean_observed_r": float(np.nanmean(observed_group_mean_r)),
        "mean_null_mean_r": float(np.nanmean(null_group_mean_r)),
        "mean_null_p975_r": float(np.nanmean(null_group_p975_r)),
        "mean_null_p975_minus_observed": float(np.nanmean(null_group_p975_r - observed_group_mean_r)),
        "mean_best_layer_signflip_p975_r": float(np.nanmean(null_group_p975_r)),
        "n_vertices_threshold_sig": int(np.asarray(threshold_mask, dtype=bool).sum()),
        "n_vertices_uncorrected_sig": int(np.asarray(uncorrected_mask, dtype=bool).sum()),
        "n_vertices_fdr_sig": int(np.asarray(fdr_mask, dtype=bool).sum()),
        "n_vertices_cluster_sig": int(np.asarray(cluster_result["cluster_mask"], dtype=bool).sum()),
        "n_clusters": int(cluster_result["cluster_p"].size),
        "n_clusters_sig": int(np.sum(cluster_result["cluster_p"] < args.cluster_alpha)),
        "pct_vertices_threshold_sig": float(np.asarray(threshold_mask, dtype=bool).mean() * 100.0),
        "n_null_layers": len(null_layers),
        "observed_mode": args.observed_mode,
        "spatial_center": args.spatial_center,
        "out_path": str(out_path),
    }
    summary.update(subject_diag)
    print(
        f"hemi-{hemi}: threshold_sig={summary['n_vertices_threshold_sig']}/{N_VERTICES}  "
        f"fdr_sig={summary['n_vertices_fdr_sig']}/{N_VERTICES}  "
        f"cluster_sig={summary['n_vertices_cluster_sig']}/{N_VERTICES}  "
        f"mean_obs={summary['mean_observed_r']:.6f}  "
        f"best_layer_p975={summary['mean_null_p975_r']:.6f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group-level fMRI threshold with null-aware best-layer sign flip."
    )
    parser.add_argument("--model", required=True, help="Model key under results/fmri.")
    parser.add_argument("--layer", type=int, required=True, help="Observed fixed-layer index.")
    parser.add_argument(
        "--observed_mode",
        choices=["fixed_layer", "best_layer"],
        default="fixed_layer",
        help="Observed map source: fixed --layer, or per-subject global best layer from --null_layers.",
    )
    parser.add_argument(
        "--null_layers",
        default=None,
        help="Comma/space-separated candidate layers for null best-layer selection. Default: all layerXX dirs excluding layer 0.",
    )
    parser.add_argument(
        "--spatial_center",
        choices=["none", "subject_layer"],
        default="none",
        help="Subtract each subject-layer map's spatial mean before group thresholding.",
    )
    parser.add_argument("--hemis", default="L,R", help="Comma-separated hemispheres, default L,R.")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject IDs.")
    parser.add_argument("--all_subjects", action="store_true", help="Use all configured fMRI subjects.")
    parser.add_argument("--n_perm", type=int, default=1000, help="Number of sign-flip permutations.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--null_batch", type=int, default=50, help="Permutation batch size.")
    parser.add_argument("--vertex_batch", type=int, default=256, help="Vertex chunk size for percentile calculation.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for uncorrected/FDR masks.")
    parser.add_argument(
        "--cluster_method",
        choices=["none", "centered_mass"],
        default="none",
        help="Optional surface cluster correction using null max cluster mass.",
    )
    parser.add_argument(
        "--cluster_forming_percentile",
        type=float,
        default=95.0,
        help="Vertex-wise null percentile used to form clusters.",
    )
    parser.add_argument("--cluster_alpha", type=float, default=0.05, help="Cluster-level corrected alpha.")
    parser.add_argument("--cluster_min_vertices", type=int, default=50, help="Minimum cluster size in vertices.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing group null output.")
    parser.add_argument("--save_group_null", action="store_true", help="Save full n_perm x vertex group null array.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output root, default results/fmri_group_null/<model>/layerXX.",
    )
    args = parser.parse_args()

    if args.n_perm <= 0:
        raise ValueError("--n_perm must be > 0")
    if args.null_batch <= 0:
        raise ValueError("--null_batch must be > 0")
    if args.vertex_batch <= 0:
        raise ValueError("--vertex_batch must be > 0")
    if not (0.0 < args.cluster_forming_percentile < 100.0):
        raise ValueError("--cluster_forming_percentile must be between 0 and 100")
    if args.cluster_min_vertices <= 0:
        raise ValueError("--cluster_min_vertices must be > 0")

    hemis = _parse_hemis(args.hemis)
    subjects = _parse_subjects(args.subjects, args.all_subjects)
    null_layers = _parse_layers_arg(args.null_layers, args.model)

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else FMRI_GROUP_NULL_DIR / args.model / f"layer{args.layer:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Group fMRI null threshold")
    print(f"  Model      : {args.model}")
    print(f"  Obs layer  : {args.layer}")
    print(f"  Obs mode   : {args.observed_mode}")
    print(f"  Spat center: {args.spatial_center}")
    print(f"  Null layers: {null_layers}")
    print(f"  Hemis      : {hemis}")
    print(f"  Subjects   : n={len(subjects)}")
    print("  Main null  : null-aware subject-global best-layer sign flip")
    print(f"  n_perm     : {args.n_perm}")
    print(f"  null_batch : {args.null_batch}")
    print(f"  vertex_bat : {args.vertex_batch}")
    print(f"  Cluster    : {args.cluster_method}")
    if args.cluster_method != "none":
        print(
            "  ClusterCfg : "
            f"forming=p{args.cluster_forming_percentile:g}, "
            f"alpha={args.cluster_alpha:g}, min_vertices={args.cluster_min_vertices}"
        )
    print(f"  Output dir : {out_dir}")
    print("=" * 72)

    adjacency_by_hemi: dict[str, list[np.ndarray]] = {}
    if args.cluster_method == "centered_mass":
        if args.spatial_center == "none":
            print("  warning: cluster_method=centered_mass is intended for centered maps; consider --spatial_center subject_layer")
        for hemi in hemis:
            adjacency_by_hemi[hemi] = _surface_adjacency(hemi)

    summaries = []
    for hemi in hemis:
        summaries.append(
            _run_hemi(
                args=args,
                hemi=hemi,
                null_layers=null_layers,
                subjects=subjects,
                out_dir=out_dir,
                cluster_adjacency=adjacency_by_hemi.get(hemi),
            )
        )

    summary_csv = out_dir / f"{_output_stem(args)}_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False, float_format="%.6f")
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
