# From Fuzzy to Formal: AI-for-QI Pipeline

Code accompanying:

> Vossler, P., Feng, J., et al. *From Fuzzy to Formal: Scaling Hospital Quality Improvement with AI.*

This repository implements the LLM-based, human-in-the-loop pipeline for **QI factor discovery** described in the paper, and the two case studies run at Zuckerberg San Francisco General Hospital (ZSFG):

1. **Length of Stay** (`exp_los/`) — factors extending hospital stay beyond medical necessity.
2. **30-day Unplanned Readmissions** (`exp_readmission/`) — factors contributing to unplanned readmissions.

The paper's Supplementary Information contains the full prompt text, per-iteration spec sheets, and learning-process details for both case studies. This repository complements that material so that QI teams at other hospitals can clone, install, plug in their own EHR data, and re-run the pipeline.

## Repository to paper section map

| Paper section | Code |
|---|---|
| Methods → AI/ML stages of QI factor discovery (Problem Formalization, Model Learning, Model Validation) | `exp_los/sconscript`, `exp_readmission/sconscript` (per-version `PROMPT_VERSIONS_DICT` entries) and `exp_los/prompts/`, `exp_readmission/prompts/` (one template per stage, per version) |
| Methods → Two-stage extraction (Gantt + Reasons) | `src/extract_gantt_chart.py`, `src/extract_reasons.py` |
| Methods → Confidence scoring | `src/score_confidences.py` |
| Methods → Human-AI Spec-Solution Co-Optimization (web review interface) | `ui/start_validation_app.py`, `ui/validation_app.py`, `ui/templates/validation_interface.html` |
| Methods → Statistical Inference (Bayesian calibration of human/AI agreement) | `src/bayesian_inference.py`, `src/bayesian_*model.stan`, `src/analyze_annotations.py`, `src/do_inference.ipynb` |
| Methods → Conformal filtering | `src/conformal_inference.py`, `src/filter_reasons_conformal.py`, `src/eval_conformal_inference.py` |
| Methods → BERTopic clustering and theme tagging | `src/cluster_reasons.py`, `src/tag_themes.py`, `src/plot_themes.py` |
| Methods → Deduplication of bloated notes | `bloatectomy/` (vendored, GPL-3.0-or-later) |

## Pipeline overview

The final configurations published in the paper are **LOS v9** and **readmission v8**, both using Claude Opus 4.5 (`us.anthropic.claude-opus-4-5-20251101-v1:0`) via AWS Bedrock for extraction and scoring, with Claude Haiku 4.5 for theme tagging. Earlier iterations used GPT-5 Mini and GPT-5; the full version-by-version trajectory is in the paper Supplementary.

For each case study, the pipeline runs:
1. **Assemble** — load and filter clinical notes from the source EHR data.
2. **Extract Gantt chart** — first LLM call: timeline of events for each encounter.
3. **Extract reasons** — second LLM call: structured contributing factors with quoted evidence.
4. **Score confidences** — third LLM call: 0–100% confidence on each factor.
5. **Validate** — Flask web UI for expert annotation; results stored in DuckDB.
6. **Calibrate** — Bayesian models in Stan estimate the human/AI agreement curve.
7. **Cluster & tag** — BERTopic over extracted reasons → manual cluster naming → theme tagging across the full cohort.

## Setup

### 1. Install

The core install (extraction, validation UI, Bayesian calibration) is Mac/Linux/CPU-friendly:

```bash
pip install -r requirements.txt
```

The clustering and theme-tagging stage (`cluster_reasons.py`, `tag_themes.py`, `plot_themes.py`) pulls in BERTopic, transformers, and HuggingFace tooling. Install it on top of the core requirements only if you need that stage:

```bash
pip install -r requirements-clustering.txt
```

The GPU wheels (`torch`, `triton`, `cuda-bindings`, `nvidia-*`) in `requirements-clustering.txt` are gated to `sys_platform == "linux"` and pip will skip them on macOS — clustering will still run on CPU, just slower.

`lab_llm` is installed directly from <https://github.com/jjfenglab/llm-api> and provides a unified interface to OpenAI and AWS Bedrock (it also has an Azure OpenAI / UCSF Versa path, which is the deployment used in the paper but not required by this repository).

### 2. Configure credentials

The pipeline expects the following environment variables (a `.env` file at the repo root is loaded automatically by `python-dotenv`):

```bash
# LLM provider — set credentials for any provider whose models you intend to
# call. lab_llm picks the provider based on the model string in the sconscripts.

# OpenAI (gpt-* models, including the GPT-5 family used in early iterations).
# The repo accepts the standard OPENAI_API_KEY. lab_llm's native name is
# OPENAI_ACCESS_TOKEN; either works (see src/common.py).
OPENAI_API_KEY=sk-...

# AWS Bedrock (Claude Opus / Haiku — the models in the published v8/v9 runs).
BEDROCK_ACCESS_KEY=...
BEDROCK_ACCESS_KEY_SECRET=...
# Optional: BEDROCK_ENDPOINT_URL=https://...

# Path to your local readmission EHR database (required by exp_readmission).
READMISSION_DB=/absolute/path/to/your_readmission.db
```

The LOS pipeline reads CSVs at `exp_los/data/<icd>_all_timelines.csv` (configured in `exp_los/sconscript`).

## Input data contract

**This repository ships no patient data.** ZSFG EHR data is protected health information and cannot be released (paper Ethics & Data Availability statements).

To run the pipeline against your own institution's data, prepare:

**For `exp_readmission/`** — a DuckDB or SQLite database with at minimum `account_detail` (encounters), `notes` (clinical notes), and `adt` (admission-discharge-transfer events) tables. The SQL in `src/assemble_readmission_notes.py` is the authoritative reference for the expected columns; it selects discharge summaries, ED notes, H&Ps, optionally consult and outpatient notes, for each readmission encounter.

**For `exp_los/`** — one or more CSVs at `exp_los/data/` with columns `encounter_id`, `note_text`, `los_days`, and `Y`. `src/assemble_los_notes.py` filters by LOS range (default 4–30 days) and shuffles. Update `SOURCE_DATA` in `exp_los/sconscript` to point at your files.

The pipeline does not depend on patient identifiers beyond an opaque `encounter_id`. Surrogate IDs from de-identified datasets work as-is.

## Reproducing the case studies

```bash
# Length of stay
scons exp_los
# Results under: exp_los/_output/<icd>/prompt_v9/default/

# Readmissions
scons exp_readmission
# Results under: exp_readmission/_output/prompt_v8/default/
```

Each run produces (per version): `assembled_notes.csv`, `gantt_charts.json`, `extracted_reasons.csv`, `scored_reasons.csv`, and a `validation.html` if a reviewer is configured.

### Web validation interface

Once extraction has finished, start the Flask annotation app:

```bash
python ui/start_validation_app.py \
    --db-path exp_los/annotations.db \
    --assembled-notes-csv exp_los/_output/all/prompt_v9/default/assembled_notes.csv \
    --extracted-reasons-csv exp_los/_output/all/prompt_v9/default/scored_reasons.csv \
    --gantt-json exp_los/_output/all/prompt_v9/default/gantt_charts.json \
    --prompt-id exp_los_v9 \
    --calibration monotone_ordinal \
    --reviewer-id <your-name>
```

Reviewers accept/reject each AI-extracted factor against highlighted evidence in the note. Annotations are written to a DuckDB database that downstream scripts read.

### Calibration / Bayesian analysis

```bash
python src/analyze_annotations.py \
    --prompt-id exp_los_v9 \
    --db-path exp_los/annotations.db \
    --calibration-model monotone_ordinal
```

Models live in `src/bayesian_*model.stan`; `src/do_inference.ipynb` walks through posterior plots. The monotone-ordinal model is the one reported in the paper.

### Cleaning BERTopic cluster names

After `scons exp_<case>` finishes clustering, `<outdir>/cluster_members.json` contains BERTopic's raw clusters. The published pipeline cleans these into human-interpretable themes by prompting an LLM. The cleaned `cluster_members_gemini_clean.json` files are reproducible on an indepdendent cohort by running the pipeline through `cluster_reasons.py` and then prompting an LLM with the prompt below:

> Read `<outdir>/cluster_members.json` produced by `src/cluster_reasons.py`. Create a new JSON `cluster_members_gemini_clean.json` with the same structure but human-interpretable topic names. If needed, divide any BERTopic theme further into smaller categories. Each topic should be cohesive and motivate a single hospital QI initiative. Fields per topic: `topic_name` (full name), `topic_description` (detailed), `qi_initiative` (example QI initiatives), `members` (list of cluster members).

Save the result as `cluster_members_gemini_clean.json` in the same `<outdir>`. `tag_themes` and `plot_themes` consume this file.
