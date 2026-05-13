# `src/` — Core pipeline modules

The top-level [README](../README.md) describes the full pipeline, how to install and configure, and how to reproduce the LOS and readmission case studies end-to-end. This file is a short index of what lives in `src/`.

## Data assembly
- `assemble_los_notes.py` — load and LOS-filter timeline CSVs for `exp_los`.
- `assemble_readmission_notes.py` — pull notes for readmission encounters from the source EHR database (`READMISSION_DB`).
- `filter_notes.py` — drop notes that don't meet the per-version filter criteria.

## LLM extraction (uses `lab_llm`)
- `extract_gantt_chart.py` — first LLM call: per-encounter Gantt chart of events.
- `extract_reasons.py` — second LLM call: structured contributing factors with quoted evidence.
- `score_confidences.py` — third LLM call: confidence scoring on each factor.
- `common.py` — shared LLM setup, prompt loading, and bloatectomy-based note deduplication.
- `prompts/`, `models/` — Pydantic schemas and prompt templates consumed by the extraction scripts.

## Annotation, calibration, and inference
- `upload_annotations.py` — load CSV annotations into the per-experiment DuckDB.
- `analyze_annotations.py` — compute calibration curves and run the Bayesian models.
- `bayesian_inference.py`, `bayesian_*model.stan` — Stan models used in the paper (monotone-ordinal is the reported one).
- `do_inference.ipynb` — notebook walkthrough of the calibration analysis and plots.
- `conformal_inference.py`, `filter_reasons_conformal.py`, `eval_conformal_inference.py` — split-conformal filter for downstream theme tagging.

## Clustering and themes
- `cluster_reasons.py` — BERTopic clustering over filtered reasons.
- `tag_themes.py` — tag each reason with the cleaned cluster name (LLM).
- `plot_themes.py` — theme distribution figures used in the paper.

## Utilities
- `utils/db_utils.py` — DuckDB helpers.
- `plot_themes.py`, `plot_themes_*` — figure helpers.
- `test_note_deduplication.py` — smoke test for the bloatectomy wrapper.

## Required environment

LLM calls require OpenAI or AWS Bedrock credentials (set via `OPENAI_ACCESS_TOKEN` or `BEDROCK_ACCESS_KEY` / `BEDROCK_ACCESS_KEY_SECRET`); see the top-level README. Stan models compile on first use via CmdStanPy.
