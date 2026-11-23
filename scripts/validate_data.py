#!/usr/bin/env python3
"""
Phase 0 dataset validation for the LLM-VLM project.

Checks:
  1. `fsaverage5_processed` contains complete left/right hemisphere GIFTI files
     with valid vertex/TR shapes.
  2. Available fsaverage5 subjects line up with the raw dataset subject folders.
  3. Required event TSV columns exist for every available `(subject, run)` pair.
  4. Each run contains exactly one article, while each subject covers articles
     1-5 exactly once across the five runs. The article order may be
     counterbalanced, so this script does NOT assume `run == article`.
  5. Valid sentence-fixation rows stay within BOLD coverage after a +5s HRF delay.
  6. Finalized text metadata passes regeneration validation.

This script is intended to answer the project question:
  "Can we proceed without downloading raw data or rerunning preprocessing?"
and to guard against introducing a false `run == article` assumption into
downstream code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd

from config import DATASET_ROOT, L1_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT
DATASET = DATASET_ROOT
FSAVG_DIR = DATASET / "fsaverage5_processed"
TEXT_METADATA_SCRIPT = ROOT / "scripts" / "create_text_metadata.py"

TR_SECONDS = 0.4
HRF_DELAY_SECONDS = 5.0
EXPECTED_VERTEXS = 10242
EXPECTED_RUNS = {1, 2, 3, 4, 5}
REQUIRED_EVENT_COLUMNS = {"onset", "duration", "CURRENT_FIX_INTEREST_AREA_ID", "SentenceID"}
VALID_SENTENCE_ID_RE = re.compile(r"^t\.(\d+)\.(\d+)$")


def collect_subject_dirs() -> list[Path]:
    return sorted(
        path for path in L1_ROOT.iterdir() if path.is_dir() and path.name.startswith("sub-")
    )


def collect_fsavg_files() -> list[Path]:
    return sorted(FSAVG_DIR.glob("*.func.gii"))


def validate_fsaverage5() -> dict:
    subject_dirs = collect_subject_dirs()
    fsavg_files = collect_fsavg_files()

    shape_errors: list[str] = []
    hemi_tr_mismatches: list[str] = []
    tr_counts_by_run: dict[tuple[str, int], dict[str, int]] = {}

    for gii_path in fsavg_files:
        parts = gii_path.name.split("_")
        sub = parts[0].replace("sub-", "")
        run = int(parts[2].replace("run-", ""))
        hemi = parts[3].replace("hemi-", "")

        gii = nib.load(str(gii_path))
        n_trs = len(gii.darrays)
        n_vertices = gii.darrays[0].data.shape[0] if n_trs > 0 else -1
        if n_vertices != EXPECTED_VERTEXS or n_trs <= 0:
            shape_errors.append(
                f"{gii_path.name}: vertices={n_vertices}, trs={n_trs}"
            )

        tr_counts_by_run.setdefault((sub, run), {})[hemi] = n_trs

    for (sub, run), hemis in sorted(tr_counts_by_run.items()):
        if set(hemis) != {"L", "R"}:
            hemi_tr_mismatches.append(f"sub-{sub} run-{run}: hemis={sorted(hemis)}")
        elif hemis["L"] != hemis["R"]:
            hemi_tr_mismatches.append(
                f"sub-{sub} run-{run}: L={hemis['L']} vs R={hemis['R']}"
            )

    dataset_subjects = {path.name.replace("sub-", "") for path in subject_dirs}
    fsavg_subjects = {sub for (sub, _run) in tr_counts_by_run}
    missing_fsavg_subjects = sorted(dataset_subjects - fsavg_subjects)

    return {
        "subject_dirs": subject_dirs,
        "fsavg_files": fsavg_files,
        "fsavg_subjects": fsavg_subjects,
        "missing_fsavg_subjects": missing_fsavg_subjects,
        "shape_errors": shape_errors,
        "hemi_tr_mismatches": hemi_tr_mismatches,
        "tr_counts_by_run": tr_counts_by_run,
    }


def iter_valid_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    sid = df["SentenceID"].fillna("").astype(str).str.strip()
    onset = pd.to_numeric(df["onset"], errors="coerce")
    valid = df[sid.str.match(VALID_SENTENCE_ID_RE) & onset.notna()].copy()
    valid["SentenceID"] = sid[sid.str.match(VALID_SENTENCE_ID_RE) & onset.notna()]
    valid["onset_seconds"] = onset[sid.str.match(VALID_SENTENCE_ID_RE) & onset.notna()] / 1000.0
    return valid


def validate_events(tr_counts_by_run: dict[tuple[str, int], dict[str, int]]) -> dict:
    missing_events: list[str] = []
    missing_columns: list[str] = []
    article_mapping_errors: list[str] = []
    subject_article_coverage_errors: list[str] = []
    timing_errors: list[str] = []

    valid_sentence_rows_total = 0
    runs_checked = 0
    run_article_maps: dict[str, dict[int, int]] = {}

    fsavg_subjects = sorted({sub for (sub, _run) in tr_counts_by_run})
    for sub in fsavg_subjects:
        run_article_maps[sub] = {}
        for run in sorted(EXPECTED_RUNS):
            ev_path = L1_ROOT / f"sub-{sub}" / "func" / f"sub-{sub}_task-read_run-{run}_events.tsv"
            if not ev_path.exists():
                missing_events.append(ev_path.name)
                continue

            df = pd.read_csv(ev_path, sep="\t")
            missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
            if missing:
                missing_columns.append(f"{ev_path.name}: missing={sorted(missing)}")
                continue

            valid_rows = iter_valid_sentence_rows(df)
            if valid_rows.empty:
                article_mapping_errors.append(f"{ev_path.name}: no valid sentence rows")
                continue

            articles = sorted(
                {
                    int(VALID_SENTENCE_ID_RE.match(sid).group(1))
                    for sid in valid_rows["SentenceID"]
                }
            )
            if len(articles) != 1:
                article_mapping_errors.append(
                    f"{ev_path.name}: articles_seen={articles}, expected exactly one article"
                )
            else:
                run_article_maps[sub][run] = articles[0]

            max_onset = float(valid_rows["onset_seconds"].max())
            n_trs = tr_counts_by_run[(sub, run)]["L"]
            bold_seconds = n_trs * TR_SECONDS
            if max_onset + HRF_DELAY_SECONDS > bold_seconds + 0.5:
                timing_errors.append(
                    f"{ev_path.name}: max_onset+HRF={max_onset + HRF_DELAY_SECONDS:.2f}s "
                    f"> bold={bold_seconds:.2f}s"
                )

            valid_sentence_rows_total += len(valid_rows)
            runs_checked += 1

        mapped_articles = sorted(run_article_maps[sub].values())
        if mapped_articles != sorted(EXPECTED_RUNS):
            subject_article_coverage_errors.append(
                f"sub-{sub}: run_to_article={run_article_maps[sub]}, "
                f"articles_seen={mapped_articles}, expected={[1, 2, 3, 4, 5]}"
            )

    return {
        "missing_events": missing_events,
        "missing_columns": missing_columns,
        "article_mapping_errors": article_mapping_errors,
        "subject_article_coverage_errors": subject_article_coverage_errors,
        "timing_errors": timing_errors,
        "valid_sentence_rows_total": valid_sentence_rows_total,
        "runs_checked": runs_checked,
        "run_article_maps": run_article_maps,
    }


def validate_text_metadata() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TEXT_METADATA_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    print("=" * 72)
    print("Phase 0 Validation")
    print("=" * 72)

    fsavg = validate_fsaverage5()
    events = validate_events(fsavg["tr_counts_by_run"])
    text_metadata_result = validate_text_metadata()

    dataset_subject_count = len(fsavg["subject_dirs"])
    fsavg_subject_count = len(fsavg["fsavg_subjects"])
    fsavg_file_count = len(fsavg["fsavg_files"])
    run_pair_count = len(fsavg["tr_counts_by_run"])

    expected_missing_fsavg = ["21", "52"]
    text_metadata_passed = text_metadata_result.returncode == 0

    checks = {
        "52 dataset subjects present": dataset_subject_count == 52,
        "500 fsaverage5 GIFTI files present": fsavg_file_count == 500,
        "50 fsaverage5 subjects present": fsavg_subject_count == 50,
        "missing fsaverage5 subjects match Gao exclusions": (
            fsavg["missing_fsavg_subjects"] == expected_missing_fsavg
        ),
        "250 unique (subject, run) fsaverage5 pairs": run_pair_count == 250,
        "0 GIFTI shape errors": not fsavg["shape_errors"],
        "0 hemisphere TR mismatches": not fsavg["hemi_tr_mismatches"],
        "0 missing events files for available fsaverage5 subjects": not events["missing_events"],
        "0 missing required event columns": not events["missing_columns"],
        "0 within-run mixed-article errors": not events["article_mapping_errors"],
        "all subjects cover articles 1-5 exactly once across runs": (
            not events["subject_article_coverage_errors"]
        ),
        "0 timing coverage errors on valid sentence rows": not events["timing_errors"],
        "text metadata validation/regeneration passes": text_metadata_passed,
    }

    all_pass = all(checks.values())

    print(f"Dataset subjects: {dataset_subject_count}")
    print(f"fsaverage5 subjects: {fsavg_subject_count}")
    print(f"Missing fsaverage5 subjects: {fsavg['missing_fsavg_subjects']}")
    print(f"fsaverage5 GIFTI files: {fsavg_file_count}")
    print(f"Unique (subject, run) pairs: {run_pair_count}")
    print(f"Valid sentence fixation rows checked: {events['valid_sentence_rows_total']}")
    sample_maps = {
        sub: events["run_article_maps"][sub]
        for sub in sorted(events["run_article_maps"])[:3]
    }
    print(f"Example run->article mappings: {sample_maps}")

    print("\nChecks:")
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    if not text_metadata_passed:
        print("\ncreate_text_metadata.py output:")
        print(text_metadata_result.stdout)
        print(text_metadata_result.stderr)

    detail_groups = {
        "shape_errors": fsavg["shape_errors"],
        "hemi_tr_mismatches": fsavg["hemi_tr_mismatches"],
        "missing_events": events["missing_events"],
        "missing_columns": events["missing_columns"],
        "article_mapping_errors": events["article_mapping_errors"],
        "subject_article_coverage_errors": events["subject_article_coverage_errors"],
        "timing_errors": events["timing_errors"],
    }
    for label, items in detail_groups.items():
        if items:
            print(f"\n{label} (showing up to 10):")
            for item in items[:10]:
                print(f"  {item}")

    print("\nSummary:")
    if all_pass:
        print("  PASS")
        print("  Conclusion: existing events.tsv files plus fsaverage5_processed data are sufficient.")
        print("  Note: article order is counterbalanced across runs/subjects, so downstream code")
        print("  should infer run->article from SentenceID rather than assuming run == article.")
        print("  This matches Gao's original regression scripts.")
        print("  Raw data download is NOT needed.")
        print("  Full preprocessing rerun is NOT needed.")
    else:
        print("  FAIL")
        print("  Conclusion: Phase 0 found issues that should be reviewed before proceeding.")


if __name__ == "__main__":
    main()
