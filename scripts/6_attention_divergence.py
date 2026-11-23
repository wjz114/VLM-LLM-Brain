#!/usr/bin/env python3
"""Jensen-Shannon divergence between the attention matrices of matched LLM/VLM pairs.

Usage
-----
    python scripts/6_attention_divergence.py                 # everything
    python scripts/6_attention_divergence.py --pairs qwen2   # one pair
    python scripts/6_attention_divergence.py --no_plot
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import ttest_1samp, ttest_rel

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import ATTENTION_DIR, RESULTS_DIR, SENTS_PER_ARTICLE, load_words_list

OUTPUT_DIR = RESULTS_DIR / "attention_divergence"
VARIANTS = ("full", "tril", "nosink")

# Pair names match 7_representational_similarity.PAIR_SPECS so the two
# channels join on `pair` without a lookup table.
PAIR_SPECS: tuple[dict, ...] = (
    {
        "name": "qwen2",
        "label": "Qwen2 / Qwen2-VL",
        "llm": "qwen2-7b",
        "vlm": "qwen2-vl-7b",
        "relation": "within_family",
    },
    {
        "name": "deepseek",
        "label": "DeepSeek-LLM / DeepSeek-VL",
        "llm": "deepseek-llm-7b-base",
        "vlm": "deepseek-vl-7b-base",
        "relation": "within_family",
    },
    {
        "name": "mistral_idefics2",
        "label": "Mistral / Idefics2",
        "llm": "mistral-7b-v0.1",
        "vlm": "idefics2-8b-base",
        "relation": "cross_architecture",
    },
    {
        # Two language models with no shared training lineage: an upper
        # reference. Layer and head counts differ, so this pair appears in
        # the quarter table only.
        "name": "floor_qwen2_mistral",
        "label": "Qwen2 / Mistral (unrelated reference)",
        "llm": "qwen2-7b",
        "vlm": "mistral-7b-v0.1",
        "relation": "unrelated_reference",
    },
)


def _load_numeric_module(stem: str):
    """Import a sibling script whose module name starts with a digit."""
    spec = importlib.util.spec_from_file_location(
        stem.replace("-", "_"), _SCRIPT_DIR / f"{stem}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fdr_bh = _load_numeric_module("4_compare_saccade").fdr_bh


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------


def js_div(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in nats.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sum_p = p.sum(axis=axis, keepdims=True)
    sum_q = q.sum(axis=axis, keepdims=True)
    # A masked row can lose all its mass to float32 underflow (see
    # divergence_by_sentence). Such a row has no distribution, so it is NaN
    # rather than 0 and gets excluded from the average, not counted as agreement.
    valid = (sum_p > 0) & (sum_q > 0)
    p = np.divide(p, sum_p, out=np.zeros_like(p), where=valid)
    q = np.divide(q, sum_q, out=np.zeros_like(q), where=valid)
    m = (p + q) / 2.0
    left = rel_entr(p, m).sum(axis=axis)
    right = rel_entr(q, m).sum(axis=axis)
    return np.where(np.squeeze(valid, axis=axis), (left + right) / 2.0, np.nan)


def apply_variant(block: np.ndarray, variant: str) -> np.ndarray:
    """Mask a causal attention block ``(..., n, n)`` for one masking variant.
    """
    if variant == "full":
        return block
    if variant == "tril":  # strict lower triangle: the 7,421-pair space
        out = block[..., 1:, :].copy()
        n = out.shape[-2]
        out[..., np.arange(n), np.arange(1, n + 1)] = 0.0
        return out
    if variant == "nosink":  # causal row minus the sentence-initial sink
        return block[..., 1:, 1:]
    raise ValueError(f"Unknown variant {variant!r}")


def sentence_layout() -> list[tuple[int, int, int]]:
    """[(article_0based, sentence, n_words), ...] in article-then-sentence order."""
    words = load_words_list()
    layout, cursor = [], 0
    for article, n_sents in enumerate(SENTS_PER_ARTICLE):
        for sentence in range(n_sents):
            layout.append((article, sentence, int(words[cursor])))
            cursor += 1
    if cursor != len(words):
        raise ValueError(f"words_list has {len(words)} entries, layout consumed {cursor}")
    return layout


def load_layer(model: str, layer: int) -> np.ndarray:
    path = ATTENTION_DIR / model / f"rb_layer{layer}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No attention cache: {path}")
    return np.load(path, mmap_mode="r")


def n_layers_of(model: str) -> int:
    directory = ATTENTION_DIR / model
    count = len(list(directory.glob("rb_layer*.npy")))
    if count == 0:
        raise FileNotFoundError(
            f"No rb_layer*.npy under {directory}. Run:\n"
            f"  python scripts/1_extract_attention.py --model {model}"
        )
    return count


def divergence_by_sentence(
    array_a: np.ndarray,
    array_b: np.ndarray,
    layout: list[tuple[int, int, int]],
    variant: str,
) -> tuple[np.ndarray, float]:
    """One D_JS per sentence: per head, mean over rows, then mean over heads.
    """
    out = np.empty(len(layout))
    dropped = total = 0
    for i, (article, sentence, n_words) in enumerate(layout):
        block_a = apply_variant(
            np.asarray(array_a[article, sentence, ..., :n_words, :n_words], np.float64), variant
        )
        block_b = apply_variant(
            np.asarray(array_b[article, sentence, ..., :n_words, :n_words], np.float64), variant
        )
        rows = js_div(block_a, block_b)
        dropped += int(np.isnan(rows).sum())
        total += rows.size
        out[i] = np.nanmean(rows)
    return out, dropped / total


def sink_mass(array: np.ndarray, layout: list[tuple[int, int, int]]) -> float:
    """Mean normalized attention to word 0, over rows i>=1, heads and sentences."""
    values = []
    for article, sentence, n_words in layout:
        block = np.asarray(array[article, sentence, ..., 1:n_words, :n_words], np.float64)
        values.append((block[..., 0] / block.sum(-1)).mean())
    return float(np.mean(values))


def quarter_attention(model: str) -> list[np.ndarray]:
    """Per quarter of depth: attention averaged over heads and over that quarter's layers.
    """
    total = n_layers_of(model)
    blocks = []
    for chunk in np.array_split(np.arange(total), 4):
        stack = np.mean(
            [np.asarray(load_layer(model, int(layer)), np.float64).mean(axis=2) for layer in chunk],
            axis=0,
        )
        blocks.append(stack)
    return blocks


# --------------------------------------------------------------------------
# Per-pair analysis
# --------------------------------------------------------------------------


def per_layer_table(
    spec: dict, layout: list[tuple[int, int, int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-layer divergence for a pair with matching layer and head counts.

    Returns (per-sentence long table, per-layer summary table).
    """
    llm, vlm = spec["llm"], spec["vlm"]
    n_layers = n_layers_of(llm)
    if n_layers != n_layers_of(vlm):
        raise ValueError(f"{spec['name']}: layer counts differ; use the quarter estimand")

    long_rows, summary_rows = [], []
    for layer in range(n_layers):
        array_llm = load_layer(llm, layer)
        array_vlm = load_layer(vlm, layer)
        if array_llm.shape[2] != array_vlm.shape[2]:
            raise ValueError(f"{spec['name']} layer {layer}: head counts differ")

        computed = {v: divergence_by_sentence(array_llm, array_vlm, layout, v) for v in VARIANTS}
        values = {v: computed[v][0] for v in VARIANTS}

        # Within-model adjacent-layer reference, same recipe as the cross-model
        # value. The last layer has no successor and is left undefined rather
        # than carried forward.
        if layer + 1 < n_layers:
            adjacent_llm = divergence_by_sentence(
                array_llm, load_layer(llm, layer + 1), layout, "full"
            )[0].mean()
            adjacent_vlm = divergence_by_sentence(
                array_vlm, load_layer(vlm, layer + 1), layout, "full"
            )[0].mean()
        else:
            adjacent_llm = adjacent_vlm = np.nan
        adjacent = float(np.mean([adjacent_llm, adjacent_vlm]))

        for i, (article, sentence, n_words) in enumerate(layout):
            long_rows.append(
                {
                    "pair": spec["name"],
                    "layer": layer,
                    "article": article + 1,
                    "sentence": sentence,
                    "n_words": n_words,
                    **{f"d_js_{v}": values[v][i] for v in VARIANTS},
                }
            )

        summary_rows.append(
            {
                "pair": spec["name"],
                "pair_label": spec["label"],
                "relation": spec["relation"],
                "llm": llm,
                "vlm": vlm,
                "layer": layer,
                "relative_depth": layer / max(n_layers - 1, 1),
                **{f"d_js_{v}": float(values[v].mean()) for v in VARIANTS},
                **{f"d_js_{v}_sd": float(values[v].std(ddof=1)) for v in VARIANTS},
                "adjacent_llm": adjacent_llm,
                "adjacent_vlm": adjacent_vlm,
                "layer_steps_js": float(values["full"].mean()) / adjacent
                if adjacent > 0
                else np.nan,
                "sink_mass_llm": sink_mass(array_llm, layout),
                "sink_mass_vlm": sink_mass(array_vlm, layout),
                **{f"dropped_frac_{v}": computed[v][1] for v in VARIANTS},
            }
        )

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def quarter_table(spec: dict, layout: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Head-averaged quarter-block divergence; defined for every pair."""
    blocks_llm = quarter_attention(spec["llm"])
    blocks_vlm = quarter_attention(spec["vlm"])
    rows = []
    for quarter, (block_llm, block_vlm) in enumerate(zip(blocks_llm, blocks_vlm)):
        values = {
            v: divergence_by_sentence(block_llm, block_vlm, layout, v)[0] for v in VARIANTS
        }
        for i, (article, sentence, n_words) in enumerate(layout):
            rows.append(
                {
                    "pair": spec["name"],
                    "pair_label": spec["label"],
                    "relation": spec["relation"],
                    "quarter": quarter,
                    "article": article + 1,
                    "sentence": sentence,
                    "n_words": n_words,
                    **{f"d_js_{v}": values[v][i] for v in VARIANTS},
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def between_pair_contrasts(quarter_long: pd.DataFrame) -> pd.DataFrame:
    """Paired contrasts between pairs, per quarter, over the 148 sentences.
    """
    matched = [s["name"] for s in PAIR_SPECS if s["relation"] != "unrelated_reference"]
    present = [p for p in matched if p in set(quarter_long["pair"])]
    rows = []
    for quarter in sorted(quarter_long["quarter"].unique()):
        block = quarter_long[quarter_long["quarter"] == quarter]
        for i, name_a in enumerate(present):
            for name_b in present[i + 1 :]:
                side_a = block[block["pair"] == name_a].sort_values(["article", "sentence"])
                side_b = block[block["pair"] == name_b].sort_values(["article", "sentence"])
                values_a = side_a["d_js_full"].to_numpy()
                values_b = side_b["d_js_full"].to_numpy()
                diff = values_a - values_b
                t_stat, p_value = ttest_rel(values_a, values_b)
                by_article = (
                    pd.DataFrame({"article": side_a["article"].to_numpy(), "diff": diff})
                    .groupby("article")["diff"]
                    .mean()
                    .to_numpy()
                )
                t_article, p_article = ttest_1samp(by_article, 0.0)
                rows.append(
                    {
                        "quarter": quarter,
                        "pair_a": name_a,
                        "pair_b": name_b,
                        "mean_a": values_a.mean(),
                        "mean_b": values_b.mean(),
                        "mean_diff": diff.mean(),
                        "t": t_stat,
                        "p": p_value,
                        "cohens_dz": diff.mean() / diff.std(ddof=1),
                        "t_article": t_article,
                        "p_article": p_article,
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        rejected, q_values = fdr_bh(table["p"].to_numpy())
        table["p_fdr"] = q_values
        table["sig_fdr05"] = rejected
    return table


def pair_js_summary(layer_summary: pd.DataFrame) -> pd.DataFrame:
    """Per-pair median sink-corrected D_JS (descriptive; n = 3 matched pairs)."""
    table = (
        layer_summary.groupby(["pair", "pair_label", "relation"], sort=False)
        .agg(n_layers=("layer", "nunique"), d_js_nosink=("d_js_nosink", "median"))
        .reset_index()
    )
    return table


# --------------------------------------------------------------------------
# Checks and plotting
# --------------------------------------------------------------------------


def self_check(model: str, layout: list[tuple[int, int, int]]) -> None:
    """Identity, boundedness, and mask safety."""
    array = load_layer(model, 1)
    identical, _ = divergence_by_sentence(array, array, layout, "full")
    if not np.allclose(identical, 0.0, atol=1e-12):
        raise AssertionError(f"D_JS({model}, itself) = {identical.max()!r}, expected 0")

    other = load_layer(model, n_layers_of(model) - 1)
    for variant in VARIANTS:
        values, dropped = divergence_by_sentence(array, other, layout, variant)
        if not np.isfinite(values).all():
            raise AssertionError(f"Non-finite sentence D_JS in variant {variant!r}")
        if values.max() > np.log(2) + 1e-9:
            raise AssertionError(f"D_JS {values.max()} exceeds the log 2 bound")
        if dropped > 0.01:
            raise AssertionError(f"{variant!r} dropped {dropped:.2%} of rows; investigate")
    print("  Self-check passed: D_JS(x, x) = 0, all variants finite and below log 2")


def plot_divergence(
    layer_summary: pd.DataFrame, quarter_long: pd.DataFrame, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for pair in layer_summary["pair"].unique():
        subset = layer_summary[layer_summary["pair"] == pair].sort_values("layer")
        axes[0].plot(
            subset["relative_depth"],
            subset["d_js_nosink"],
            marker="o",
            ms=3,
            label=subset["pair_label"].iloc[0],
        )
    axes[0].set_xlabel("relative depth")
    axes[0].set_ylabel(r"$D_{JS}$ nosink (nats)")
    axes[0].set_title("Per-layer attention divergence, sink column removed")

    quarter_means = quarter_long.groupby(["pair", "pair_label", "quarter"])["d_js_full"].mean()
    for (pair, label), series in quarter_means.groupby(level=[0, 1]):
        style = {"linestyle": "--", "alpha": 0.6} if "floor" in pair else {}
        axes[1].plot(
            series.index.get_level_values("quarter"), series.to_numpy(), marker="s", ms=4,
            label=label, **style,
        )
    axes[1].set_xlabel("quarter of depth")
    axes[1].set_ylabel(r"$D_{JS}$ (nats)")
    axes[1].set_xticks(range(4))
    axes[1].set_title("Quarter-layer divergence, heads averaged")

    for axis in axes:
        axis.set_yscale("log")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair names.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    requested = [name.strip() for name in args.pairs.split(",")] if args.pairs else None
    specs = [s for s in PAIR_SPECS if requested is None or s["name"] in requested]
    if not specs:
        raise SystemExit(f"No pair matched {args.pairs!r}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = sentence_layout()
    print(f"{len(layout)} sentences, {sum(n for _, _, n in layout)} words")

    sentence_tables, layer_tables, quarter_tables, skipped = [], [], [], []
    checked = False

    for spec in specs:
        print(f"\n[{spec['name']}] {spec['label']}")
        try:
            if not checked:
                self_check(spec["llm"], layout)
                checked = True

            if spec["relation"] != "unrelated_reference":
                long_table, summary = per_layer_table(spec, layout)
                sentence_tables.append(long_table)
                layer_tables.append(summary)
                print(
                    f"  per-layer D_JS nosink: min={summary['d_js_nosink'].min():.5f} "
                    f"median={summary['d_js_nosink'].median():.5f} "
                    f"max={summary['d_js_nosink'].max():.5f}"
                )

            quarters = quarter_table(spec, layout)
            quarter_tables.append(quarters)
            print(
                "  quarter D_JS: "
                + "  ".join(
                    f"Q{q}={v:.5f}"
                    for q, v in quarters.groupby("quarter")["d_js_full"].mean().items()
                )
            )
        except FileNotFoundError as error:
            print(f"  SKIPPED: {error}")
            skipped.append(spec["name"])

    if not quarter_tables:
        raise SystemExit("No pair could be analyzed; extract attention first.")

    layer_summary = (
        pd.concat(layer_tables, ignore_index=True) if layer_tables else pd.DataFrame()
    )
    quarter_long = pd.concat(quarter_tables, ignore_index=True)

    if sentence_tables:
        pd.concat(sentence_tables, ignore_index=True).to_csv(
            output_dir / "js_by_sentence.csv", index=False, float_format="%.9g"
        )
        layer_summary.to_csv(
            output_dir / "js_by_layer.csv", index=False, float_format="%.9g"
        )
        layer_summary[
            ["pair", "pair_label", "layer", "relative_depth", "layer_steps_js",
             "adjacent_llm", "adjacent_vlm"]
        ].to_csv(output_dir / "js_layer_steps.csv", index=False, float_format="%.9g")
    quarter_long.to_csv(output_dir / "js_by_quarter.csv", index=False, float_format="%.9g")

    contrasts = between_pair_contrasts(quarter_long)
    contrasts.to_csv(output_dir / "pair_contrasts.csv", index=False, float_format="%.9g")

    summary = pair_js_summary(layer_summary) if not layer_summary.empty else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(output_dir / "js_pair_summary.csv", index=False, float_format="%.9g")

    if not args.no_plot and not layer_summary.empty:
        plot_divergence(layer_summary, quarter_long, output_dir / "divergence_curves.png")

    print("\n=== Attention divergence (median D_JS nosink by pair) ===")
    if not summary.empty:
        print(summary.round(5).to_string(index=False))

    if not contrasts.empty:
        print("\n=== Between-pair contrasts (quarter, paired over 148 sentences) ===")
        print(
            contrasts[
                ["quarter", "pair_a", "pair_b", "mean_diff", "cohens_dz", "p_fdr", "sig_fdr05"]
            ]
            .round(5)
            .to_string(index=False)
        )

    if skipped:
        print(f"Skipped (no attention cache): {', '.join(skipped)}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
