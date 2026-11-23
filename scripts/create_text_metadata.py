#!/usr/bin/env python3
"""
scripts/create_text_metadata.py

Current generation logic:
  - `data/reading_brain_sentences.txt` and `data/words.csv` are the corrected
    source files.
  - This script cross-validates that they match exactly.
  - If validation passes, it regenerates `data/words_list.p`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"

SENTENCES_FILE = DATA_DIR / "reading_brain_sentences.txt"
WORDS_CSV = DATA_DIR / "words.csv"
WORDS_LIST_PKL = DATA_DIR / "words_list.p"

EXPECTED_SENTS_PER_ART = [31, 31, 28, 28, 30]
EXPECTED_TOTAL_SENTS = 148
EXPECTED_MAX_WORDS = 16
REQUIRED_CSV_COLUMNS = ["SentenceID", "WordPosition", "Word"]


def sentence_id(article_idx: int, sentence_idx: int) -> str:
    return f"t.{article_idx:02d}.{sentence_idx:02d}"


def parse_sentences_txt(path: Path) -> list[tuple[str, list[str]]]:
    """Return `(sentence_id, words)` pairs in article/sentence order."""
    text = path.read_text(encoding="utf-8").strip()
    article_blocks = text.split("\n\n")

    if len(article_blocks) != len(EXPECTED_SENTS_PER_ART):
        raise RuntimeError(
            f"{path.name}: expected {len(EXPECTED_SENTS_PER_ART)} article blocks, "
            f"got {len(article_blocks)}"
        )

    sentences: list[tuple[str, list[str]]] = []
    for article_idx, article_block in enumerate(article_blocks, start=1):
        article_sentences = [line.strip() for line in article_block.splitlines() if line.strip()]
        expected_count = EXPECTED_SENTS_PER_ART[article_idx - 1]
        if len(article_sentences) != expected_count:
            raise RuntimeError(
                f"Article {article_idx}: expected {expected_count} sentences in "
                f"{path.name}, got {len(article_sentences)}"
            )

        for sentence_idx, sentence in enumerate(article_sentences, start=1):
            words = sentence.split()
            sentences.append((sentence_id(article_idx, sentence_idx), words))

    if len(sentences) != EXPECTED_TOTAL_SENTS:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_TOTAL_SENTS} sentences, got {len(sentences)}"
        )

    return sentences


def parse_words_csv(path: Path) -> list[tuple[str, list[str]]]:
    """Return `(sentence_id, words)` pairs from `words.csv` in sentence order."""
    df = pd.read_csv(path, dtype={"SentenceID": str, "WordPosition": int, "Word": str})

    missing_columns = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_columns:
        raise RuntimeError(f"{path.name}: missing required columns: {missing_columns}")

    sentences: list[tuple[str, list[str]]] = []
    grouped = df.groupby("SentenceID", sort=False)

    for sid, group in grouped:
        group = group.sort_values("WordPosition")
        positions = group["WordPosition"].tolist()
        expected_positions = list(range(1, len(group) + 1))
        if positions != expected_positions:
            raise RuntimeError(
                f"{path.name}: {sid} has non-contiguous WordPosition values: "
                f"{positions[:10]}{'...' if len(positions) > 10 else ''}"
            )

        words = group["Word"].fillna("").astype(str).tolist()
        if any(word == "" for word in words):
            raise RuntimeError(f"{path.name}: {sid} contains empty Word entries")

        sentences.append((sid, words))

    if len(sentences) != EXPECTED_TOTAL_SENTS:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_TOTAL_SENTS} unique SentenceID values, "
            f"got {len(sentences)}"
        )

    return sentences


def validate_alignment(
    txt_sentences: list[tuple[str, list[str]]],
    csv_sentences: list[tuple[str, list[str]]],
) -> list[str]:
    """Return a list of validation errors between the two source files."""
    errors: list[str] = []

    if len(txt_sentences) != len(csv_sentences):
        errors.append(
            f"Sentence count mismatch: txt={len(txt_sentences)} csv={len(csv_sentences)}"
        )
        return errors

    for index, ((txt_sid, txt_words), (csv_sid, csv_words)) in enumerate(
        zip(txt_sentences, csv_sentences),
        start=1,
    ):
        if txt_sid != csv_sid:
            errors.append(
                f"Sentence order mismatch at index {index}: txt={txt_sid} csv={csv_sid}"
            )
            continue

        if txt_words != csv_words:
            errors.append(
                f"{txt_sid}: txt/csv mismatch "
                f"(txt_len={len(txt_words)}, csv_len={len(csv_words)})"
            )
            for word_index, (txt_word, csv_word) in enumerate(zip(txt_words, csv_words), start=1):
                if txt_word != csv_word:
                    errors.append(
                        f"  pos {word_index}: txt={txt_word!r} csv={csv_word!r}"
                    )
                    break
            if len(txt_words) != len(csv_words):
                errors.append(
                    f"  extra words: txt_tail={txt_words[len(csv_words):]!r} "
                    f"csv_tail={csv_words[len(txt_words):]!r}"
                )

    return errors


def validate_lengths(sentences: list[tuple[str, list[str]]]) -> list[int]:
    """Validate global sentence statistics and return the sentence lengths."""
    lengths = [len(words) for _, words in sentences]

    if len(lengths) != EXPECTED_TOTAL_SENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SENTS} sentence lengths, got {len(lengths)}"
        )

    max_words = max(lengths)
    if max_words > EXPECTED_MAX_WORDS:
        raise RuntimeError(
            f"Max sentence length {max_words} exceeds Gao limit {EXPECTED_MAX_WORDS}"
        )

    return lengths


def write_words_list(lengths: list[int], out_path: Path) -> None:
    with out_path.open("wb") as handle:
        pickle.dump(lengths, handle)


def main() -> None:
    print("=" * 68)
    print("create_text_metadata.py")
    print("Validate finalized text metadata and regenerate words_list.p")
    print("=" * 68)

    print("\n[1/4] Parsing reading_brain_sentences.txt ...")
    txt_sentences = parse_sentences_txt(SENTENCES_FILE)
    print(f"  Parsed {len(txt_sentences)} sentences from {SENTENCES_FILE.name}")

    print("\n[2/4] Parsing words.csv ...")
    csv_sentences = parse_words_csv(WORDS_CSV)
    print(f"  Parsed {len(csv_sentences)} sentence groups from {WORDS_CSV.name}")

    print("\n[3/4] Cross-validating sources ...")
    alignment_errors = validate_alignment(txt_sentences, csv_sentences)
    if alignment_errors:
        print("  Validation failed. First mismatches:")
        for error in alignment_errors[:20]:
            print(f"    {error}")
        raise RuntimeError(
            f"Found {len(alignment_errors)} txt/csv validation issue(s); "
            f"refusing to regenerate {WORDS_LIST_PKL.name}"
        )
    print("  reading_brain_sentences.txt and words.csv match exactly")

    lengths = validate_lengths(txt_sentences)
    lt_total = sum(length * (length - 1) // 2 for length in lengths)

    print("\n[4/4] Writing words_list.p ...")
    write_words_list(lengths, WORDS_LIST_PKL)
    print(f"  Wrote {WORDS_LIST_PKL} ({len(lengths)} entries)")
    print(f"  Per-article counts: {EXPECTED_SENTS_PER_ART}")
    print(f"  Word counts: min={min(lengths)} max={max(lengths)} total={sum(lengths)}")
    print(f"  Lower-triangle elements: {lt_total}")
    print(f"  First 10 sentence lengths: {lengths[:10]}")


if __name__ == "__main__":
    main()
