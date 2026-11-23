#!/usr/bin/env python3
"""Layer-wise RSA between matched LLM/VLM pairs.

Purpose
-------
The alignment analyses ask whether a VLM predicts human fMRI and reading
behaviour better than its matched LLM. 

Representation unit
-------------------
Word-level hidden states h in R^D (subword mean-pooled), cached by
``extract_hidden_states.py``. Not the encoding-pipeline pair feature
``phi_lm``. Word-level pooling makes cross-tokenizer comparison legitimate.

Usage
-----
    python scripts/7_representational_similarity.py
    python scripts/7_representational_similarity.py --pairs qwen2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import HIDDEN_DIR, RESULTS_DIR, SENTS_PER_ARTICLE, load_words_list

OUTPUT_DIR = RESULTS_DIR / "representational_similarity"

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
        "name": "floor_qwen2_mistral",
        "label": "Qwen2 / Mistral (unrelated floor)",
        "llm": "qwen2-7b",
        "vlm": "mistral-7b-v0.1",
        "relation": "unrelated_floor",
    },
)


def load_sentence_layout(model: str) -> list[tuple[int, int, int]]:
    """[(article, sentence, n_valid_words), ...] from words_list.p (1-based article)."""
    _ = model
    words = load_words_list()
    layout, cursor = [], 0
    for article, n_sents in enumerate(SENTS_PER_ARTICLE, start=1):
        for sentence in range(n_sents):
            layout.append((article, sentence, int(words[cursor])))
            cursor += 1
    if cursor != len(words):
        raise ValueError(f"words_list has {len(words)} entries, layout consumed {cursor}")
    return layout


def available_layers(model: str) -> int:
    directory = HIDDEN_DIR / model
    if not directory.exists():
        raise FileNotFoundError(f"No hidden-state cache for {model!r}: {directory}")
    layers = list(directory.glob("rb_layer*.npy"))
    if not layers:
        raise FileNotFoundError(
            f"No rb_layer*.npy under {directory}. Run:\n"
            f"  python scripts/extract_hidden_states.py --model {model}"
        )
    return len(layers)


def build_word_matrix(
    model: str, layer: int, layout: list[tuple[int, int, int]], n_words: int
) -> np.ndarray:
    array = np.load(HIDDEN_DIR / model / f"rb_layer{layer}.npy", mmap_mode="r")
    out = np.empty((n_words, array.shape[-1]), dtype=np.float64)
    row = 0
    for article, sentence, count in layout:
        out[row : row + count] = array[article - 1, sentence, :count, :]
        row += count
    if row != n_words:
        raise ValueError(f"{model} layer {layer}: filled {row} rows, expected {n_words}")
    if not np.isfinite(out).all():
        raise ValueError(f"{model} layer {layer}: non-finite hidden states")
    return out


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    sd = matrix.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (matrix - matrix.mean(axis=0, keepdims=True)) / sd


def rsa_spearman(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Spearman correlation between the two RDMs (1 − Pearson r, upper triangle)."""
    upper = np.triu_indices(matrix_a.shape[0], k=1)
    rdm_a = 1.0 - np.corrcoef(matrix_a)[upper]
    rdm_b = 1.0 - np.corrcoef(matrix_b)[upper]
    return float(spearmanr(rdm_a, rdm_b).statistic)


def analyze_pair(spec: dict, *, verbose: bool) -> pd.DataFrame:
    llm_key, vlm_key = spec["llm"], spec["vlm"]
    layout = load_sentence_layout(llm_key)
    n_words = sum(count for _, _, count in layout)

    n_layers_llm = available_layers(llm_key)
    n_layers_vlm = available_layers(vlm_key)
    n_layers = min(n_layers_llm, n_layers_vlm)
    if n_layers_llm != n_layers_vlm:
        print(
            f"  ! layer counts differ ({llm_key}: {n_layers_llm}, "
            f"{vlm_key}: {n_layers_vlm}); comparing the first {n_layers}"
        )
    if verbose:
        print(f"  {n_words} words, {len(layout)} sentences, {n_layers} layers")

    rows: list[dict] = []
    prev_llm: np.ndarray | None = None
    prev_vlm: np.ndarray | None = None

    for layer in range(n_layers):
        llm_z = zscore_columns(build_word_matrix(llm_key, layer, layout, n_words))
        vlm_z = zscore_columns(build_word_matrix(vlm_key, layer, layout, n_words))
        rsa = rsa_spearman(llm_z, vlm_z)
        within_llm = rsa_spearman(prev_llm, llm_z) if prev_llm is not None else np.nan
        within_vlm = rsa_spearman(prev_vlm, vlm_z) if prev_vlm is not None else np.nan

        rows.append(
            {
                "pair": spec["name"],
                "pair_label": spec["label"],
                "relation": spec["relation"],
                "llm": llm_key,
                "vlm": vlm_key,
                "layer": layer,
                "relative_depth": layer / max(n_layers - 1, 1),
                "n_words": n_words,
                "rsa": rsa,
                "within_llm_adjacent_rsa": within_llm,
                "within_vlm_adjacent_rsa": within_vlm,
            }
        )
        if verbose:
            extra = (
                f"  adj={np.nanmean([within_llm, within_vlm]):.4f}"
                if np.isfinite(within_llm)
                else ""
            )
            print(f"    layer {layer:>2}  rsa={rsa:.4f}{extra}")
        prev_llm, prev_vlm = llm_z, vlm_z

    return pd.DataFrame(rows)


def self_check(model: str) -> None:
    layout = load_sentence_layout(model)
    n_words = sum(count for _, _, count in layout)
    matrix = zscore_columns(build_word_matrix(model, 1, layout, n_words))
    value = rsa_spearman(matrix, matrix)
    if not np.isclose(value, 1.0, atol=1e-9):
        raise AssertionError(f"Self-RSA for {model} is {value!r}, expected 1.0")
    print(f"  Self-check passed: RSA({model}, itself) = {value:.10f}")


def plot_depth_curves(table: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pair in table["pair"].unique():
        subset = table[table["pair"] == pair].sort_values("layer")
        style = {"linestyle": "--", "alpha": 0.6} if "floor" in pair else {}
        ax.plot(
            subset["relative_depth"], subset["rsa"],
            marker="o", ms=3, label=subset["pair_label"].iloc[0], **style,
        )
    ax.set_xlabel("relative depth")
    ax.set_ylabel("RSA (Spearman on RDMs)")
    ax.set_title("Representational similarity between matched models")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair names.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    requested = [name.strip() for name in args.pairs.split(",")] if args.pairs else None
    specs = [s for s in PAIR_SPECS if requested is None or s["name"] in requested]
    if not specs:
        raise SystemExit(f"No pair matched {args.pairs!r}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []
    skipped: list[str] = []
    checked_self = False

    for spec in specs:
        print(f"\n[{spec['name']}] {spec['label']}")
        try:
            if not checked_self:
                self_check(spec["llm"])
                checked_self = True
            table = analyze_pair(spec, verbose=not args.quiet)
        except FileNotFoundError as error:
            print(f"  SKIPPED: {error}")
            skipped.append(spec["name"])
            continue

        pair_dir = output_dir / spec["name"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(pair_dir / "rsa_by_layer.csv", index=False, float_format="%.9g")
        tables.append(table)
        print(
            f"  rsa: min={table['rsa'].min():.4f} "
            f"median={table['rsa'].median():.4f} max={table['rsa'].max():.4f}"
        )

    if not tables:
        raise SystemExit("No pair could be analyzed; extract hidden states first.")

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output_dir / "rsa_by_layer.csv", index=False, float_format="%.9g")
    summary = (
        combined.groupby(["pair", "pair_label", "relation"], sort=False)
        .agg(n_layers=("layer", "nunique"), rsa_median=("rsa", "median"))
        .reset_index()
    )
    summary_path = output_dir / "rsa_all_pairs.csv"
    summary.to_csv(summary_path, index=False, float_format="%.9g")
    figure_path = output_dir / "depth_curves.png"
    plot_depth_curves(combined, figure_path)

    print("\n=== Median RSA by pair (floor = unrelated Qwen2 / Mistral) ===")
    print(summary.to_string(index=False))
    if skipped:
        print(f"\nSkipped (no hidden-state cache): {', '.join(skipped)}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {output_dir / 'rsa_by_layer.csv'} (per-layer diagnostic)")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
