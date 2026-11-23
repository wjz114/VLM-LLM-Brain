#!/usr/bin/env python3
"""Extract word-level hidden states for the RSA manipulation check.

Saves per-layer tensors shaped (5, 31, 16, D) under data/hidden_states/<model>/.
These are word vectors h, not the encoding-pipeline pair feature phi_lm.

Usage
-----
    python scripts/extract_hidden_states.py --model qwen2-7b
    python scripts/extract_hidden_states.py --model qwen2-7b --output_dir scratch
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
    HIDDEN_DIR,
    MAX_N_SENTS,
    MAX_SENT_LEN,
    N_ARTICLES,
    SENTENCES_FILE,
    load_sentences,
)
from models import MODELS

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "extract_attention", _SCRIPT_DIR / "1_extract_attention.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot import 1_extract_attention.py")
_attn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attn)
load_model_and_tokenizer = _attn.load_model_and_tokenizer
build_word_token_mapping = _attn.build_word_token_mapping
_get_device = _attn._get_device


def _text_cfg(model):
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    if hasattr(cfg, "language_config"):
        return cfg.language_config
    return cfg


def mean_pool_word_states(token_hs, word2tok, n_words):
    d = token_hs.shape[-1]
    out = np.zeros((n_words, d), dtype=np.float32)
    for w in range(n_words):
        out[w] = token_hs[word2tok[w]].mean(axis=0)
    return out


def pad_word_states(word_hs, max_len=MAX_SENT_LEN):
    n, d = word_hs.shape
    if n > max_len:
        raise ValueError(f"n_words={n} > max_len={max_len}")
    out = np.zeros((max_len, d), dtype=np.float32)
    out[:n] = word_hs
    return out


@torch.no_grad()
def extract_sentence_hidden(model, tokenizer, sentence, device):
    words = sentence.split()
    n_words = len(words)
    enc, word2tok = build_word_token_mapping(tokenizer, words)
    if set(word2tok) != set(range(n_words)):
        raise RuntimeError(f"Incomplete word→token mapping for: {sentence!r}")

    inputs = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
    out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    if getattr(out, "hidden_states", None) is None:
        raise RuntimeError("Forward pass did not return hidden_states.")

    hs_tuple = out.hidden_states[1:]  # drop embeddings; index matches attention layers
    n_layers = len(hs_tuple)
    d = hs_tuple[0].shape[-1]
    word_hs = np.zeros((n_layers, n_words, d), dtype=np.float32)
    for li, h in enumerate(hs_tuple):
        tok = h.squeeze(0).float().cpu().numpy()
        word_hs[li] = mean_pool_word_states(tok, word2tok, n_words)
    return word_hs, n_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract word-level hidden states for RSA.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--sentences", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="First article only. Pass --output_dir so the real cache is not overwritten.",
    )
    args = parser.parse_args()
    if args.smoke_test and args.output_dir is None:
        raise SystemExit("--smoke_test requires --output_dir (refuses to write the real cache).")

    print("=" * 64)
    print(f"Hidden-state extraction  [{args.model}]")
    print("=" * 64)

    t_load = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, quantize=not args.no_quantize)
    cfg = _text_cfg(model)
    n_layers = int(cfg.num_hidden_layers)
    hidden_size = int(cfg.hidden_size)
    device = _get_device(model)
    print(f"  Loaded in {time.time() - t_load:.1f}s")
    print(f"  Layers: {n_layers}  |  hidden_size: {hidden_size}  |  Device: {device}")

    sentences_path = Path(args.sentences) if args.sentences else SENTENCES_FILE
    article_sentences = load_sentences(sentences_path)
    if args.smoke_test:
        print("\n*** SMOKE TEST — first article only ***")
        article_sentences = article_sentences[:1]

    total_sentences = sum(len(a) for a in article_sentences)
    save_dir = Path(args.output_dir) if args.output_dir else HIDDEN_DIR / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    tensor_per_article: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    processed = 0
    t_start = time.time()

    for art_idx, article in enumerate(article_sentences):
        hs_per_sent: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        for sent_idx, sentence in enumerate(article):
            processed += 1
            word_hs, n_words = extract_sentence_hidden(model, tokenizer, sentence, device)
            if n_words > MAX_SENT_LEN:
                raise RuntimeError(f"Sentence has {n_words} words (limit {MAX_SENT_LEN})")
            for layer in range(n_layers):
                hs_per_sent[layer].append(torch.from_numpy(pad_word_states(word_hs[layer])))
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_sentences - processed) / rate if rate > 0 else 0
            print(
                f"\r  [{processed:>3}/{total_sentences}] "
                f"art {art_idx + 1} sent {sent_idx + 1:>2}/{len(article)} "
                f"({n_words:>2}w)  {rate:.1f} sent/s  ETA {eta:.0f}s",
                end="", flush=True,
            )
        print()
        for layer in range(n_layers):
            stacked = torch.stack(hs_per_sent[layer])
            pad_n = MAX_N_SENTS - stacked.shape[0]
            if pad_n > 0:
                stacked = F.pad(stacked, (0, 0, 0, 0, 0, pad_n))
            tensor_per_article[layer].append(stacked)

    if len(article_sentences) < N_ARTICLES:
        dummy = torch.zeros(MAX_N_SENTS, MAX_SENT_LEN, hidden_size)
        for layer in range(n_layers):
            while len(tensor_per_article[layer]) < N_ARTICLES:
                tensor_per_article[layer].append(dummy)

    expected_shape = (N_ARTICLES, MAX_N_SENTS, MAX_SENT_LEN, hidden_size)
    print(f"\nSaving {n_layers} layer files to {save_dir}/")
    for layer in range(n_layers):
        arr = torch.stack(tensor_per_article[layer]).numpy().astype(np.float32)
        if arr.shape != expected_shape:
            raise RuntimeError(f"Layer {layer}: shape {arr.shape} != {expected_shape}")
        if np.isnan(arr).any():
            raise RuntimeError(f"Layer {layer} contains NaNs")
        np.save(save_dir / f"rb_layer{layer}.npy", arr)
        if layer == 0 or layer == n_layers - 1 or (layer + 1) % 10 == 0:
            print(f"  rb_layer{layer}.npy  {arr.shape}")
    print(f"\nDone — {processed} sentences in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
