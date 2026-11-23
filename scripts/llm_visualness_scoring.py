"""
LLM visualness scoring: 3-judge calibrated pipeline.

Usage:
    from scripts.llm_visualness_scoring import run_fallback_pipeline_one_fold

    calibrated_df = run_fallback_pipeline_one_fold(
        fold_id=0,
        train_df=train_df,       # must have columns listed in fit_calibrators_for_fold
        test_lowcov_df=test_df,  # sentences to score; must have sentence_id + sentence
    )

Environment variables (or .env):
    OPENAI_API_KEY
    GEMINI_API_KEY
    MOONSHOT_API_KEY
"""

import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.isotonic import IsotonicRegression
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

load_dotenv()

# ── Models ─────────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-5.4-2026-03-05"
GEMINI_MODEL = "gemini-2.5-flash"
KIMI_MODEL   = "kimi-k2.6"

# ── Clients ────────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

kimi_client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a strict rater of sentence visualness for cognitive neuroscience research.

Your task is to rate ONLY the visual sensory salience conveyed by a single English sentence.

Important rules:
1. Use only the sentence itself.
2. Ignore article context, discourse context, and task context.
3. Do NOT rate general concreteness, emotional intensity, familiarity, importance, or linguistic complexity.
4. Use ordinary lexical semantics only; do not enrich the sentence with elaborate world knowledge.
5. Return JSON only, matching the required schema exactly.
6. Use a 0.0 to 5.0 scale, with at most one decimal place.

Definitions:
- scene_content: how much the sentence explicitly evokes visible entities, objects, scenes, or spatial layout.
- visual_detail: how much the sentence contains visually discriminable attributes such as shape, color, appearance, motion, or spatial relations.
- imageability: how easy it is to form a stable visual mental image from the sentence alone.

Scale anchors:
0.0 = essentially no visual content
1.0 = very weak visual cue
2.0 = limited visual content
3.0 = moderate visual content
4.0 = strong visual content
5.0 = extremely vivid visual content"""

USER_TEMPLATE = """\
Rate the following sentence.

Sentence:
"{sentence}"

Return JSON with:
- scene_content
- visual_detail
- imageability
- confidence
- rationale_short

Constraints:
- scene_content, visual_detail, imageability: float in [0.0, 5.0]
- confidence: float in [0.0, 1.0]
- rationale_short: <= 20 English words
- No text outside JSON"""

# ── Schema ─────────────────────────────────────────────────────────────────────
class VisualnessRating(BaseModel):
    scene_content:   float = Field(ge=0.0, le=5.0)
    visual_detail:   float = Field(ge=0.0, le=5.0)
    imageability:    float = Field(ge=0.0, le=5.0)
    confidence:      float = Field(ge=0.0, le=1.0)
    rationale_short: str   = Field(max_length=80)

VISUALNESS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scene_content":   {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "visual_detail":   {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "imageability":    {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "confidence":      {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale_short": {"type": "string", "maxLength": 80},
    },
    "required": ["scene_content", "visual_detail", "imageability", "confidence", "rationale_short"],
}

def _finalize(d: dict, provider: str, model: str) -> dict:
    d["raw_visualness"] = round((d["scene_content"] + d["visual_detail"] + d["imageability"]) / 3.0, 4)
    d["provider"] = provider
    d["model"]    = model
    return d

# ── Per-judge scorers with retry ───────────────────────────────────────────────
_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, Exception)),
    reraise=True,
)

@_retry
def score_with_openai(sentence: str) -> dict:
    resp = openai_client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
        text_format=VisualnessRating,
        reasoning={"effort": "low"},
    )
    d = resp.output_parsed.model_dump()
    d["response_id"] = resp.id
    return _finalize(d, "openai", OPENAI_MODEL)

@_retry
def score_with_gemini(sentence: str) -> dict:
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=USER_TEMPLATE.format(sentence=sentence),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VisualnessRating,
            temperature=0,
            top_p=1.0,
            max_output_tokens=200,
        ),
    )
    d = json.loads(resp.text)
    return _finalize(d, "gemini", GEMINI_MODEL)

@_retry
def score_with_kimi(sentence: str) -> dict:
    resp = kimi_client.chat.completions.create(
        model=KIMI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "visualness_rating", "schema": VISUALNESS_JSON_SCHEMA, "strict": True},
        },
        max_completion_tokens=200,
        extra_body={"thinking": {"type": "disabled"}},
        # Do NOT set temperature=0 for kimi-k2.6 (Moonshot-specific constraint)
    )
    d = json.loads(resp.choices[0].message.content)
    d["response_id"] = resp.id
    return _finalize(d, "kimi", KIMI_MODEL)

def score_all_three(sentence: str) -> dict:
    return {
        "sentence": sentence,
        "openai":   score_with_openai(sentence),
        "gemini":   score_with_gemini(sentence),
        "kimi":     score_with_kimi(sentence),
    }

# ── Calibration ────────────────────────────────────────────────────────────────
class FoldJudgeCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=5.0, increasing=True, out_of_bounds="clip")

    def fit(self, raw_scores, gold_scores):
        self.model.fit(raw_scores, gold_scores)
        return self

    def predict(self, raw_scores):
        return self.model.predict(raw_scores)

def fit_calibrators_for_fold(train_df: pd.DataFrame) -> dict:
    """Fit per-judge isotonic calibrator on high-coverage training sentences.

    train_df must have columns: is_high_coverage, gold_lancaster_vision_sentence_score,
    openai_raw_visualness, gemini_raw_visualness, kimi_raw_visualness.
    """
    hc = train_df.loc[train_df["is_high_coverage"]].copy()
    return {
        judge: FoldJudgeCalibrator().fit(
            hc[f"{judge}_raw_visualness"].to_numpy(),
            hc["gold_lancaster_vision_sentence_score"].to_numpy(),
        )
        for judge in ("openai", "gemini", "kimi")
    }

def apply_calibrators(scored_df: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    out = scored_df.copy()
    for judge in ("openai", "gemini", "kimi"):
        out[f"{judge}_calibrated_visualness"] = calibrators[judge].predict(
            out[f"{judge}_raw_visualness"].to_numpy()
        )
    out["fusion_calibrated_visualness"] = out[[
        "openai_calibrated_visualness",
        "gemini_calibrated_visualness",
        "kimi_calibrated_visualness",
    ]].median(axis=1)
    return out

# ── Full fold pipeline ─────────────────────────────────────────────────────────
def run_fallback_pipeline_one_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    test_lowcov_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score low-coverage sentences with three judges, calibrate, and fuse.

    Args:
        fold_id: identifier (used only for logging).
        train_df: training articles; needs is_high_coverage + gold_lancaster_vision_sentence_score
                  + {judge}_raw_visualness columns.
        test_lowcov_df: held-out low-coverage sentences; needs sentence_id + sentence.

    Returns:
        DataFrame with raw, calibrated, and fused visualness columns.
    """
    print(f"[Fold {fold_id}] Scoring {len(test_lowcov_df)} sentences with 3 LLM judges...")
    rows = []
    for _, row in test_lowcov_df.iterrows():
        scored = score_all_three(row["sentence"])
        rows.append({
            "sentence_id":           row["sentence_id"],
            "sentence":              row["sentence"],
            "openai_raw_visualness": scored["openai"]["raw_visualness"],
            "gemini_raw_visualness": scored["gemini"]["raw_visualness"],
            "kimi_raw_visualness":   scored["kimi"]["raw_visualness"],
            "openai_confidence":     scored["openai"]["confidence"],
            "gemini_confidence":     scored["gemini"]["confidence"],
            "kimi_confidence":       scored["kimi"]["confidence"],
        })
    scored_df = pd.DataFrame(rows)

    print(f"[Fold {fold_id}] Fitting calibrators on training high-coverage sentences...")
    calibrators = fit_calibrators_for_fold(train_df)

    print(f"[Fold {fold_id}] Applying calibration and fusing scores...")
    return apply_calibrators(scored_df, calibrators)
