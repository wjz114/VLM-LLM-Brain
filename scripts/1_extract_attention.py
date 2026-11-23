#!/usr/bin/env python3
"""
Phase 1: Extract per-layer, per-head, word-level attention matrices.

For each model in the registry, processes all 148 sentences (5 articles)
and saves one .npy file per layer.

Output shape per file:
    (N_ARTICLES=5, MAX_N_SENTS=31, n_heads, MAX_SENT_LEN=16, MAX_SENT_LEN=16)

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    ATTENTION_DIR,
    HF_CACHE_DIR,
    MAX_N_SENTS,
    MAX_SENT_LEN,
    N_ARTICLES,
    SENTS_PER_ARTICLE,
    SENTENCES_FILE,
    TOTAL_SENTS,
    load_sentences,
)
from models import MODELS

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_key: str, *, quantize: bool = True):
    """Instantiate model + tokenizer according to the model registry.

    Returns ``(model, tokenizer)`` with the model already in eval mode.
    """
    from transformers import AutoTokenizer

    cfg = MODELS[model_key]
    hf_name: str = cfg["hf_name"]
    cache_dir: str = cfg.get("cache_dir", HF_CACHE_DIR)
    trust_remote_code: bool = cfg.get("trust_remote_code", True)
    use_fast_tokenizer: bool = cfg.get("use_fast_tokenizer", True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        # Some multimodal checkpoints register tokenizer via processor.
        if cfg.get("allow_processor_tokenizer_fallback", False):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                hf_name, trust_remote_code=trust_remote_code, cache_dir=cache_dir,
            )
            if not hasattr(processor, "tokenizer"):
                raise RuntimeError(
                    f"Processor for {hf_name} does not expose .tokenizer"
                ) from exc
            tokenizer = processor.tokenizer
        else:
            raise

    # Resolve load kwargs (torch_dtype is stored as string for serialisability)
    load_kwargs: dict = {}
    for k, v in cfg.get("load_kwargs", {}).items():
        load_kwargs[k] = getattr(torch, v) if k == "torch_dtype" else v

    if quantize and cfg.get("quantize_8bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    cls_name: str = cfg["model_class"]
    if cls_name == "AutoModelForCausalLM":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **load_kwargs,
        )
    elif cls_name == "Qwen2_5_VLTextModel":
        from transformers import Qwen2_5_VLTextModel

        # Loads only the language backbone from the VL checkpoint.
        # Visual encoder weights appear as UNEXPECTED keys and are skipped — expected.
        model = Qwen2_5_VLTextModel.from_pretrained(
            hf_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **load_kwargs,
        )
    elif cls_name == "Qwen2VLTextModel":
        try:
            from transformers import Qwen2VLTextModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2VLTextModel is unavailable in this transformers build. "
                "Upgrade transformers to a version that supports Qwen2-VL."
            ) from exc

        model = Qwen2VLTextModel.from_pretrained(
            hf_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **load_kwargs,
        )
    elif cls_name == "DeepseekVLTextBackbone":
        try:
            from deepseek_vl.models import MultiModalityCausalLM  # noqa: F401  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek-VL support requires the `deepseek_vl` package. "
                "Install it in this environment, e.g.:\n"
                "  pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git"
            ) from exc

        from transformers import AutoModelForCausalLM

        try:
            vl_model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
                **load_kwargs,
            )
        except ValueError as exc:
            if "model type `multi_modality`" in str(exc):
                raise RuntimeError(
                    "DeepSeek-VL model loading failed because `multi_modality` "
                    "is still unregistered. Ensure `deepseek_vl` imports in the "
                    "same Python environment used to run this script."
                ) from exc
            raise

        if not hasattr(vl_model, "language_model"):
            raise RuntimeError(
                "DeepSeek-VL checkpoint loaded but missing `.language_model`; "
                "cannot extract text attentions."
            )
        # We extract language-only attentions from the text backbone.
        model = vl_model.language_model
    elif cls_name == "Idefics2ForConditionalGeneration":
        from transformers import Idefics2ForConditionalGeneration

        model = Idefics2ForConditionalGeneration.from_pretrained(
            hf_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **load_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_class in registry: {cls_name!r}")

    model.eval()
    return model, tokenizer


def get_model_info(model) -> tuple[int, int]:
    """Return ``(n_layers, n_heads)`` handling VL config nesting."""
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    elif hasattr(cfg, "language_config"):
        cfg = cfg.language_config
    return cfg.num_hidden_layers, cfg.num_attention_heads


def _get_device(model) -> torch.device:
    """Determine where to place input tensors."""
    try:
        return model.device
    except AttributeError:
        pass
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Subword → word merging  
# ---------------------------------------------------------------------------


def build_word_token_mapping(
    tokenizer, words: list[str],
) -> tuple[dict, dict[int, list[int]]]:
    """Tokenize pre-split *words* and return ``(encoding, word2tok)``.

    Uses ``is_split_into_words=True`` with ``word_ids()`` for robust
    subword alignment that works across all fast-tokenizer backends
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    if not hasattr(enc, "word_ids"):
        raise RuntimeError(
            "Tokenizer output does not provide word_ids(); this pipeline "
            "requires a fast tokenizer that supports is_split_into_words=True."
        )

    try:
        word_ids = enc.word_ids()
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer does not support word_ids() for split words. "
            "Use a fast tokenizer for this model."
        ) from exc

    word2tok: dict[int, list[int]] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None:
            word2tok.setdefault(wid, []).append(tok_idx)
    return enc, word2tok


def merge_attentions_per_head(
    attn_head: np.ndarray,
    word2tok: dict[int, list[int]],
    n_words: int,
) -> np.ndarray:
    """Merge token-level attention to word-level for a single head.

    """
    T = attn_head.shape[0]

    # Step 1: collapse "to" dimension (columns → sum)
    col_merged = np.zeros((T, n_words), dtype=np.float64)
    for w in range(n_words):
        toks = word2tok[w]
        col_merged[:, w] = attn_head[:, toks].sum(axis=1)

    # Step 2: collapse "from" dimension (rows → mean)
    result = np.zeros((n_words, n_words), dtype=np.float64)
    for w in range(n_words):
        toks = word2tok[w]
        result[w, :] = col_merged[toks, :].mean(axis=0)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-sentence extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_sentence_attention(
    model, tokenizer, sentence: str, device: torch.device,
) -> tuple[np.ndarray, int]:
    """Run *sentence* through *model* and return word-level attention.

    Returns
    -------
    word_attn : ndarray, shape ``(n_layers, n_heads, n_words, n_words)``
    n_words : int
    """
    words = sentence.split()
    n_words = len(words)

    enc, word2tok = build_word_token_mapping(tokenizer, words)

    mapped = set(word2tok.keys())
    expected = set(range(n_words))
    if mapped != expected:
        raise RuntimeError(
            f"Incomplete word→token mapping (missing {expected - mapped}) "
            f"for: {sentence!r}"
        )

    inputs = {
        k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)
    }
    out = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if getattr(out, "attentions", None) is None:
        raise RuntimeError(
            "Model forward pass did not return attentions. "
            "Check attn_implementation, model_class loader, and compatibility "
            "with output_attentions=True."
        )

    # out.attentions: tuple of (1, n_heads, T, T) tensors — one per layer
    attn_np = np.stack(
        [a.squeeze(0).float().cpu().numpy() for a in out.attentions], axis=0,
    )  # (n_layers, n_heads, T, T)

    n_layers, n_heads = attn_np.shape[:2]

    word_attn = np.zeros(
        (n_layers, n_heads, n_words, n_words), dtype=np.float32,
    )
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            word_attn[layer_idx, head_idx] = merge_attentions_per_head(
                attn_np[layer_idx, head_idx], word2tok, n_words,
            )

    return word_attn, n_words


# ---------------------------------------------------------------------------
# Post-extraction validation
# ---------------------------------------------------------------------------


def _run_validation(
    save_dir: Path, n_layers: int, n_heads: int,
    article_sentences: list[list[str]],
) -> None:
    """Quick sanity checks on the saved .npy files."""
    print("\n=== Post-extraction validation ===")

    path0 = save_dir / "rb_layer0.npy"
    layer0 = np.load(path0)
    print(f"  Layer 0 shape: {layer0.shape}")

    neg = int((layer0 < -1e-6).sum())
    print(f"  Negative attention values: {neg}")

    ok, bad = 0, 0
    for art_idx, art_sents in enumerate(article_sentences):
        for s_idx, sent in enumerate(art_sents):
            n_w = len(sent.split())
            for h in range(n_heads):
                row_sums = layer0[art_idx, s_idx, h, :n_w, :n_w].sum(axis=1)
                if np.allclose(row_sums, 1.0, atol=0.05):
                    ok += 1
                else:
                    bad += 1

    print(f"  Row-sum ~1.0: {ok} OK, {bad} violations (across all heads/sentences)")

    total_lt = sum(
        len(s.split()) * (len(s.split()) - 1) // 2
        for art in article_sentences
        for s in art
    )
    print(f"  Total lower-triangle elements (concatenated): {total_lt}")
    print(f"  Total .npy files saved: {n_layers}")
    print("  Validation complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: extract word-level attention matrices from LLM / VLM.",
    )
    parser.add_argument(
        "--model", required=True, choices=list(MODELS.keys()),
        help="Model key from models.py",
    )
    parser.add_argument(
        "--sentences", type=str, default=None,
        help="Path to sentences file (default: data/reading_brain_sentences.txt)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/attentions/<model>)",
    )
    parser.add_argument(
        "--no_quantize", action="store_true",
        help="Disable 8-bit quantization (use fp16 instead)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("=" * 64)
    print(f"Phase 1: Attention Extraction  [{args.model}]")
    print("=" * 64)

    # Load model
    print("\nLoading model …")
    t_load = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.model, quantize=not args.no_quantize,
    )
    n_layers, n_heads = get_model_info(model)
    device = _get_device(model)
    print(f"  Loaded in {time.time() - t_load:.1f}s")
    print(f"  Layers: {n_layers}  |  Heads: {n_heads}  |  Device: {device}")

    # Load sentences
    sentences_path = Path(args.sentences) if args.sentences else SENTENCES_FILE
    article_sentences = load_sentences(sentences_path)

    total_sentences = sum(len(a) for a in article_sentences)
    print(
        f"\nArticles: {len(article_sentences)}  |  "
        f"Sentences: {total_sentences}  |  "
        f"Max words/sent: {max(len(s.split()) for a in article_sentences for s in a)}"
    )

    save_dir = Path(args.output_dir) if args.output_dir else ATTENTION_DIR / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract attention
    # ------------------------------------------------------------------
    #   tensor_per_article[layer] is a list of (MAX_N_SENTS, n_heads, W, W)
    tensor_per_article: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    processed = 0
    t_start = time.time()

    for art_idx, article in enumerate(article_sentences):
        attn_per_sent: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

        for sent_idx, sentence in enumerate(article):
            processed += 1
            word_attn, n_words = extract_sentence_attention(
                model, tokenizer, sentence, device,
            )

            if n_words > MAX_SENT_LEN:
                raise RuntimeError(
                    f"Sentence has {n_words} words (limit {MAX_SENT_LEN}): "
                    f"{sentence!r}"
                )

            pad = MAX_SENT_LEN - n_words
            for layer in range(n_layers):
                t = torch.from_numpy(word_attn[layer])   # (n_heads, n_w, n_w)
                if pad > 0:
                    t = F.pad(t, (0, pad, 0, pad))
                attn_per_sent[layer].append(t)

            # Progress
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_sentences - processed) / rate if rate > 0 else 0
            print(
                f"\r  [{processed:>3}/{total_sentences}] "
                f"art {art_idx + 1} sent {sent_idx + 1:>2}/{len(article)} "
                f"({n_words:>2}w)  {rate:.1f} sent/s  ETA {eta:.0f}s",
                end="", flush=True,
            )

        print()  # newline after each article

        # Stack sentences → (n_sent, n_heads, W, W) and pad to MAX_N_SENTS
        for layer in range(n_layers):
            stacked = torch.stack(attn_per_sent[layer])
            pad_n = MAX_N_SENTS - stacked.shape[0]
            if pad_n > 0:
                stacked = F.pad(stacked, (0, 0, 0, 0, 0, 0, 0, pad_n))
            tensor_per_article[layer].append(stacked)

    # ------------------------------------------------------------------
    # Save per-layer .npy files
    # ------------------------------------------------------------------
    expected_shape = (N_ARTICLES, MAX_N_SENTS, n_heads, MAX_SENT_LEN, MAX_SENT_LEN)

    print(f"\nSaving {n_layers} layer files to {save_dir}/")
    for layer in range(n_layers):
        arr = torch.stack(tensor_per_article[layer]).numpy()
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"Layer {layer}: shape {arr.shape} != expected {expected_shape}"
            )
        nan_count = int(np.isnan(arr).sum())
        if nan_count > 0:
            raise RuntimeError(
                f"Layer {layer} contains {nan_count} NaN values in extracted attention. "
                "Aborting save to avoid propagating corrupted features. "
                "Try bf16 model loading for Qwen2-family checkpoints."
            )
        np.save(save_dir / f"rb_layer{layer}.npy", arr)

        if layer == 0 or layer == n_layers - 1 or (layer + 1) % 10 == 0:
            print(f"  rb_layer{layer}.npy  {arr.shape}")

    total_time = time.time() - t_start
    print(f"\nDone — {processed} sentences in {total_time:.1f}s")
    print(f"Output directory: {save_dir}")

    # ------------------------------------------------------------------
    # Post-extraction validation
    # ------------------------------------------------------------------
    _run_validation(save_dir, n_layers, n_heads, article_sentences)


if __name__ == "__main__":
    main()
