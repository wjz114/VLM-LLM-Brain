"""
Central path and constant configuration for the LLM-VLM brain alignment project.

Environment variable overrides
-------------------------------
READING_BRAIN_DIR : path to the Reading Brain dataset root (ds003974-main/).
                    Defaults to <project_root>/L1-dataset/ds003974-main/
HF_CACHE_DIR      : path for HuggingFace model cache.
                    Defaults to HuggingFace's own default (~/.cache/huggingface/).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_ROOT = Path(
    os.environ.get("READING_BRAIN_DIR", PROJECT_ROOT / "L1-dataset" / "ds003974-main")
)

_hf_cache_env = os.environ.get("HF_CACHE_DIR")
HF_CACHE_DIR = _hf_cache_env if _hf_cache_env else None  # None → HuggingFace default

SENTENCES_FILE = DATA_DIR / "reading_brain_sentences.txt"
WORDS_CSV = DATA_DIR / "words.csv"
WORDS_LIST_PKL = DATA_DIR / "words_list.p"
ATTENTION_DIR = DATA_DIR / "attentions"
HIDDEN_DIR = DATA_DIR / "hidden_states"

# Events TSVs and raw sub-XX directories live directly under L1-dataset/,
# NOT inside ds003974-main/.  GIFTIs live under ds003974-main/fsaverage5_processed/.
L1_ROOT = DATASET_ROOT.parent

EVENTS_TEMPLATE = str(
    L1_ROOT / "sub-{sub}" / "func"
    / "sub-{sub}_task-read_run-{run}_events.tsv"
)
GIFTI_TEMPLATE = str(
    DATASET_ROOT / "fsaverage5_processed"
    / "sub-{sub}_task-read_run-{run}_hemi-{hemi}_space-fsaverage5_bold.func.gii"
)

# ── Dataset constants (from Gao et al.) ────────────────────────────────────────

N_ARTICLES = 5
SENTS_PER_ARTICLE = [31, 31, 28, 28, 30]
TOTAL_SENTS = sum(SENTS_PER_ARTICLE)              # 148
MAX_N_SENTS = max(SENTS_PER_ARTICLE)               # 31
MAX_SENT_LEN = 16

# ── Subject handling ───────────────────────────────────────────────────────────

EXCLUDE_EYETRACKING = {21}
EXCLUDE_FMRI = {21, 52}
ALL_SUBJECTS = list(range(1, 53))
EYETRACK_SUBJECTS = [s for s in ALL_SUBJECTS if s not in EXCLUDE_EYETRACKING]
FMRI_SUBJECTS = [s for s in ALL_SUBJECTS if s not in EXCLUDE_FMRI]

# ── Regression constants ───────────────────────────────────────────────────────

SPLIT_IDX = int(TOTAL_SENTS * 0.9)                 # 133 train / 15 test
TR = 0.4                                            # seconds
HRF_DELAY = 5.0                                     # seconds
N_VERTICES = 10_242                                  # fsaverage5 per hemisphere
RIDGE_ALPHAS = np.logspace(1, 3, 20)


def _compute_gao_rm_id() -> list[int]:
    rm: list[int] = []
    for s in range(MAX_N_SENTS):
        for a in range(N_ARTICLES):
            if s >= SENTS_PER_ARTICLE[a]:
                rm.append(s * N_ARTICLES + a)
    return rm


GAO_RM_ID = _compute_gao_rm_id()                    # [142,143,147,148,152,153,154]

# ── Helper functions ───────────────────────────────────────────────────────────


def load_sentences(path: Path | str | None = None) -> list[list[str]]:
    """Load sentences from the canonical text file, grouped by article."""
    p = Path(path) if path else SENTENCES_FILE
    text = p.read_text(encoding="utf-8").strip()
    return [block.strip().split("\n") for block in text.split("\n\n")]


def load_words_list(path: Path | str | None = None) -> list[int]:
    """Load the pickled list of per-sentence word counts (148 integers)."""
    import pickle

    p = Path(path) if path else WORDS_LIST_PKL
    with p.open("rb") as fh:
        return pickle.load(fh)
