"""Recompute camera-ready Table 5 from archived participant slope estimates.

The four statistical/loading functions are from
05_plot_continuous_visualness_subject_panel.py. Source checksums are recorded
in expected_results/table5/sources.csv. Reproduction starts from saved slopes.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = [
    {"tag": "qwen2_pair"},
    {"tag": "deepseek_pair"},
    {"tag": "mistral_pair"},
]
CONTRASTS = [
    ("qwen2_pair", "mistral_pair", "Qwen - Mistral"),
    ("deepseek_pair", "mistral_pair", "DeepSeek - Mistral"),
]


def _onesided_signflip(vec: np.ndarray, n_perm: int, seed: int) -> float:
    x = np.asarray(vec, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    obs = float(np.mean(x))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, x.size))
    perm_means = np.mean(signs * x[None, :], axis=1)
    return float((1 + np.sum(perm_means >= obs)) / (n_perm + 1))



def _holm(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    out = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * float(pvals[idx]))
        running = max(running, adj)
        out[idx] = running
    return out



def _summarize(x: np.ndarray, *, n_perm: int, seed: int) -> dict:
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
    p = _onesided_signflip(x, n_perm=n_perm, seed=seed)
    return {
        "n": n,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "d_z": d_z,
        "p": p,
        "n_pos": int(np.sum(x > 0)),
        "n_neg": int(np.sum(x < 0)),
        "vals": x,
    }



def _load_betas(root: Path) -> dict[str, pd.Series]:
    betas: dict[str, pd.Series] = {}
    for spec in PRIMARY:
        path = root / spec["tag"] / "visualness_subject_slope_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "subject" not in df.columns or "beta_mean" not in df.columns:
            raise ValueError(f"{path} needs columns subject, beta_mean")
        s = df.set_index("subject")["beta_mean"].astype(np.float64)
        s = s[np.isfinite(s)].sort_index()
        betas[spec["tag"]] = s
    # Strict subject alignment
    ref = betas[PRIMARY[0]["tag"]].index
    for tag, s in betas.items():
        if not s.index.equals(ref):
            raise ValueError(f"subject IDs differ: {tag} vs {PRIMARY[0]['tag']}")
        if len(s) != 50:
            raise ValueError(f"{tag}: expected n=50, got {len(s)}")
    return betas



def compute_table(root: Path, n_perm: int, seed: int) -> pd.DataFrame:
    betas = _load_betas(root)
    for tag, values in betas.items():
        if values.index.has_duplicates or set(values.index) != set(range(1, 53)) - {21, 52}:
            raise ValueError(f"{tag}: expected distinct subjects 1..52 excluding 21 and 52")
    primary = [(spec["tag"], _summarize(betas[spec["tag"]].to_numpy(),
                                      n_perm=n_perm, seed=seed)) for spec in PRIMARY]
    contrasts = [(label, _summarize((betas[a] - betas[b]).to_numpy(dtype=np.float64),
                                    n_perm=n_perm, seed=seed + 17))
                 for a, b, label in CONTRASTS]
    rows = []
    for family in (primary, contrasts):
        adjusted = _holm([stats["p"] for _, stats in family])
        for (label, stats), p_holm in zip(family, adjusted):
            rows.append({"group": label, "n": stats["n"], "n_pos": stats["n_pos"],
                         "n_neg": stats["n_neg"], "mean": stats["mean"],
                         "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                         "d_z": stats["d_z"], "p_signflip": stats["p"], "p_holm": p_holm})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "expected_results/table5")
    parser.add_argument("--out", type=Path, default=ROOT / "results/table5/summary.csv")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.n_perm <= 0:
        raise ValueError("--n_perm must be positive")
    result = compute_table(args.root, args.n_perm, args.seed)
    if args.n_perm == 10000 and args.seed == 1234:
        expected = pd.read_csv(args.root / "summary.csv")
        pd.testing.assert_frame_equal(result, expected, check_dtype=False,
                                      check_exact=False, rtol=5.1e-8, atol=5.1e-12)
        print("Verified all five rows against the archived Table 5 summary.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.8g")
    print(result.to_string(index=False))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
