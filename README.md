# Human-Model Alignment: Does Visual Learning Affect Natural Reading?

Code for: *[Do VLMs Align Better with Humans than LLMs Do during Natural Reading?]*.

## Overview

We compare three matched LLM/VLM pairs under **text-only** input during natural reading. The pipeline tests behavioral alignment (regressive saccades), neural alignment (whole-cortex fMRI), and content-conditioned neural differences (visualness). 

![framework](img/overall_framework.png)


## Requirements

- **Hardware:** GPU with ≥ 24 GB VRAM (Phase 1); 64 GB RAM recommended (Phase 3)
- **Software tested:** Python 3.10.20, PyTorch 2.10.0+cu126, CUDA 12.6

## Installation

```bash
pip install -r requirements.txt
```

Core analysis dependencies are pinned to the tested versions in `requirements.txt`.
The PyTorch requirement accepts platform-specific builds of version 2.10.0.
The GPU experiments used the CUDA 12.6 build.


For DeepSeek-VL, also install the vendor package:

```bash
pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git
```

## Data Setup

### 1. Reading Brain dataset (fMRI + eye-tracking)

Download from OpenNeuro: <https://openneuro.org/datasets/ds003974>

Set `READING_BRAIN_DIR` to the `ds003974-main` directory when using an
external data location. Validation and encoding use this same setting.
The sibling `sub-XX/func/` directories contain event TSV files.
For example, in PowerShell run

```powershell
$env:READING_BRAIN_DIR = 'D:/ReadingBrain/ds003974-main'
```


```
L1-dataset/
├── ds003974-main/
│   └── fsaverage5_processed/   ← GIFTIs
└── sub-01/
    └── func/
    ...
```

### 2. Lancaster Sensorimotor Norms

Included as `Lancaster_Sensorimotor_norms.csv` (Vision / Visual.mean dimension).
Cite: Lynott et al. (2019), *Behavior Research Methods*. doi:10.3758/s13428-019-01316-z

### 3. Model weights

Downloaded from HuggingFace on first run. 

## Reproduction Steps

Run the scripts in numerical order. 

BOLD and eye-movement files must pass `validate_data.py` before encoding.

```bash
# 0. Validate dataset integrity
python scripts/validate_data.py

# 1. Word-level attention
python scripts/1_extract_attention.py --model qwen2-7b

# 2. Saccade encoding
python scripts/2_encode_saccade.py --model qwen2-7b --all_layers --all_subjects

# 3. fMRI encoding across layers (Qwen2 example)
python scripts/3_encode_fmri.py --model qwen2-7b --all_layers --all_subjects --hemi L
python scripts/3_encode_fmri.py --model qwen2-7b --all_layers --all_subjects --hemi R

# 4a–b. Paired comparison 
python scripts/4_compare_saccade.py --llm qwen2-7b --vlm qwen2-vl-7b
python scripts/4_compare_fmri.py --llm qwen2-7b --vlm qwen2-vl-7b

# 4c. Single-model group permutation test (example fixed layer 1)
python scripts/4_permutation_test_fmri.py --model qwen2-7b --layer 1 --all_subjects

# 5. Sentence visualness metadata (Lancaster) + binary high/low analysis
python scripts/5_visualness_analysis.py --llm qwen2-7b --vlm qwen2-vl-7b

# 5b. Continuous visualness from fixed-layer out-of-fold (OOF) statistics.
# Run step 1 for all six models first. The fixed layers used here are
# 21 for the Qwen2 pair and 1 for the DeepSeek and Mistral/Idefics2 pairs.
(
  set -e
  mkdir -p results/fmri
  oof_input=$(mktemp -d results/fmri/oof-fixed.XXXXXX)
  for model in qwen2-7b qwen2-vl-7b deepseek-llm-7b-base deepseek-vl-7b-base \
               mistral-7b-v0.1 idefics2-8b-base; do
    case "$model" in
      qwen2-*) layer=21 ;;
      *)       layer=1 ;;
    esac
    for hemi in L R; do
      python scripts/3_encode_fmri.py --model "$model" --layer "$layer" \
        --hemi "$hemi" --all_subjects --export_oof
    done
    # Link just the selected layer, including when other layers are already on disk.
    layer_dir=$(printf 'results/fmri/oof/%s/layer%02d' "$model" "$layer")
    mkdir "$oof_input/$model"
    ln "$layer_dir/"*_oof_stats.npz "$oof_input/$model/"
  done
  python scripts/5_continuous_visualness.py --oof_root "$oof_input" \
    --sent_meta results/visualness/sentence_visualness_groups.csv
)

# 6. Attention Jensen–Shannon divergence 
python scripts/6_attention_divergence.py

# 7. Hidden-state RSA 
python scripts/extract_hidden_states.py --model qwen2-7b
python scripts/7_representational_similarity.py
```

## Lancaster sentence eligibility

The main visualness analysis requires Lancaster coverage >= 0.75 and at
least three matched content words. This retains 143 of 148 sentences;
the other five are marked `excluded` in
`expected_results/sentence_visualness_groups.csv`. Both visualness
scripts use these defaults. Within each article, the high and low
groups each contain the floor of 20% of eligible sentences, ranked by
Lancaster visual strength. Article-wise ranks and standardized scores
are calculated over the eligible sentences.

## Table 5 from participant slopes

```bash
python scripts/5_summarize_visualness.py
```

This CPU-only command starts from the three archived participant slope tables
under `expected_results/table5/`. Each file contains 50 participants and the
columns `subject`, `n_units`, and `beta_mean`. `beta_mean` is the participant's
whole-cortex visual-strength moderation slope for VLM minus LLM. Participants
21 and 52 are excluded.


## Acknowledgments

- Dataset: Reading Brain (OpenNeuro ds003974). Cite: Li et al. (2022). doi:10.18112/openneuro.ds003974.v3.0.0

## License

MIT — see `LICENSE`.
