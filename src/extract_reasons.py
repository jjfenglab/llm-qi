"""Phase 2: Extract structured reasons using LLM.

This script calls an LLM (via lab_llm) to extract structured reasons from
clinical notes with caching and batch processing. It's generalized to work
with any prompt template.

Supports two modes:
1. Single-call mode (v6): Extracts Gantt chart + reasons in one LLM call
2. Two-stage mode (v7+): Uses existing Gantt charts (--input-gantt-json) and extracts only reasons

Usage:
    # Single-call mode (v6)
    python src/extract_reasons.py --input-csv <path> --output-csv <path> --prompt-template <path>

    # Two-stage mode (v7+)
    python src/extract_reasons.py --input-csv <path> --input-gantt-json <path> --prompt-template <path>
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lab_llm import TextDataset

from common import (
    list_available_models,
    load_prompt_template,
    create_prompt_from_template,
    deduplicate_note,
    parse_model_string,
    setup_llm_api,
)
from models import GenericReasons, GenericReasonsBase, ReasonsOnly


def extract_reasons(
    input_csv_path: str,
    output_csv_path: str,
    output_gantt_json_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "data/extraction.txt",
    model_str: str = "gpt-4o-mini-2024-07-18",
    batch_size: int = 10,
    max_retries: int = 2,
    temperature: float = 0,
    limit: int = None,
    filter_csv_path: str = None,
    random_seed: int = None,
    max_char: int = None,
    deduplicate: bool = True,
    no_confidence: bool = False,
    input_gantt_json_path: str = None,
) -> pd.DataFrame:
    """Extract structured reasons from notes using LLM with a custom prompt template.

    Supports two modes:
    1. Single-call mode: Extracts Gantt chart + reasons together (when input_gantt_json_path is None)
    2. Two-stage mode: Uses existing Gantt charts and extracts only reasons (when input_gantt_json_path is provided)

    Args:
        input_csv_path: Path to assembled notes CSV file from Phase 1
        output_csv_path: Path to output CSV file for extracted reasons
        output_gantt_json_path: Path to output JSON file for Gantt charts (single-call mode only)
        prompt_template_path: Path to prompt template file
        cache_db_path: Path to cache database
        log_file_path: Path to log file (txt format)
        model_str: Model string (e.g., 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18')
        batch_size: Batch size for LLM processing
        max_retries: Maximum number of retries for failed extractions
        limit: Optional limit on number of notes to process (for testing)
        filter_csv_path: Optional path to filter CSV (only process encounters with keep=1)
        random_seed: Random seed for shuffling data before applying limit (for reproducibility)
        max_char: Optional maximum character length for note_text (filter after limit)
        deduplicate: Whether to apply bloatectomy deduplication to notes (default: True)
        no_confidence: If True, use GenericReasonsBase (no inline confidence) instead of GenericReasons
        input_gantt_json_path: Path to input Gantt charts JSON (two-stage mode). When provided,
            uses ReasonsOnly model and prompt must include {gantt_chart_json} placeholder.

    Returns:
        DataFrame with extracted reasons
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"

    # Determine mode
    two_stage_mode = input_gantt_json_path is not None
    if two_stage_mode:
        assert Path(input_gantt_json_path).exists(), f"Input Gantt JSON file not found: {input_gantt_json_path}"

    # Setup logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load assembled notes
    logger.info(f"Loading assembled notes from CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Apply filter if provided
    if filter_csv_path:
        assert Path(filter_csv_path).exists(), f"Filter CSV file not found: {filter_csv_path}"
        logger.info(f"Loading filter results from CSV: {filter_csv_path}")
        filter_df = pd.read_csv(filter_csv_path)
        assert 'encounter_id' in filter_df.columns, "Missing encounter_id column in filter CSV"
        assert 'keep' in filter_df.columns, "Missing keep column in filter CSV"

        # Get encounter IDs to keep (keep=1)
        keep_ids = filter_df[filter_df['keep'] == 1]['encounter_id'].tolist()
        original_count = len(df)
        df = df[df['encounter_id'].isin(keep_ids)]
        logger.info(f"Filtered from {original_count} to {len(df)} encounters (keep=1)")

    if max_char is not None:
        pre_filter_count = len(df)
        df = df[df['note_text'].str.len() <= max_char]
        logger.info(f"Filtered by max_char={max_char}: {pre_filter_count} -> {len(df)} notes")

    if limit:
        print("MY ENCOUNTERS", df.encounter_id.unique().size)
        if random_seed is not None:
            logger.info(f"Shuffling data with random_seed={random_seed}, then limiting to {limit} notes")
            df = df.sample(frac=1, random_state=random_seed).head(limit)
            print(df.encounter_id.unique())
        else:
            logger.info(f"Limiting to {limit} notes (no shuffle)")
            df = df.head(limit)

    assert len(df) > 0, "No assembled notes found"
    assert 'encounter_id' in df.columns, "Missing encounter_id column"
    assert 'note_text' in df.columns, "Missing note_text column"

    # Setup LLM API (shorter timeout for two-stage mode since output is simpler)
    timeout = 60 * 5 if two_stage_mode else 60 * 10
    logger.info(f"Setting up LLM API (timeout={timeout}s)")
    api = setup_llm_api(cache_db_path, log_file_path, model_str, timeout=timeout)

    # Load prompt template
    logger.info(f"Loading prompt template from {prompt_template_path}")
    prompt_template = load_prompt_template(prompt_template_path)

    # Validate prompt template for two-stage mode
    if two_stage_mode:
        assert "{gantt_chart_json}" in prompt_template, \
            "Two-stage mode requires prompt template with {gantt_chart_json} placeholder"
        logger.info("Two-stage mode: using existing Gantt charts from input JSON")

        # Load input Gantt charts
        with open(input_gantt_json_path, 'r') as f:
            input_gantt_charts = json.load(f)
        logger.info(f"Loaded {len(input_gantt_charts)} Gantt charts from {input_gantt_json_path}")

    # Apply deduplication if enabled
    if deduplicate:
        logger.info("Applying bloatectomy deduplication to notes")
        original_chars = df['note_text'].str.len().mean()
        notes = [deduplicate_note(note) for note in df['note_text']]
        deduped_chars = np.mean([len(n) for n in notes])
        reduction_pct = (1 - deduped_chars / original_chars) * 100 if original_chars > 0 else 0
        logger.info(f"Deduplication: {original_chars:,} -> {deduped_chars:,} chars ({reduction_pct:.1f}% reduction)")
    else:
        notes = df['note_text'].tolist()

    # Create prompts
    logger.info("Creating prompts")
    if two_stage_mode:
        # Two-stage mode: include Gantt chart in prompt
        prompts = []
        valid_indices = []
        for i, (encounter_id, note) in enumerate(zip(df['encounter_id'], notes)):
            gantt_chart = input_gantt_charts.get(str(encounter_id))
            if gantt_chart is None:
                logger.warning(f"No Gantt chart found for encounter {encounter_id}, skipping")
                continue
            gantt_chart_json = json.dumps(gantt_chart, indent=2)
            prompt = prompt_template.format(
                note_text=note,
                gantt_chart_json=gantt_chart_json,
            )
            prompts.append(prompt)
            valid_indices.append(i)

        # Filter df to only include encounters with Gantt charts
        df = df.iloc[valid_indices].reset_index(drop=True)
        notes = [notes[i] for i in valid_indices]
        logger.info(f"Processing {len(prompts)} encounters with valid Gantt charts")
    else:
        # Single-call mode: just note_text
        prompts = [
            prompt_template.format(note_text=note)
            for note in notes
        ]
    assert len(prompts) == len(df), "Mismatch between prompts and notes"

    # Create dataset
    dataset = TextDataset(prompts)

    # Select response model based on mode and no_confidence flag
    if two_stage_mode:
        response_model = ReasonsOnly
    else:
        response_model = GenericReasonsBase if no_confidence else GenericReasons
    logger.info(f"Using response model: {response_model.__name__}")

    # Run batch extraction
    # Use fewer tokens for two-stage mode (no Gantt chart in output)
    max_tokens = 5000 if two_stage_mode else 10000
    logger.info(f"Extracting reasons (batch_size={batch_size}, max_tokens={max_tokens})")
    results = asyncio.run(
        api.get_outputs(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            response_model=response_model,
        )
    )

    # Validate results
    assert len(results) == len(df), f"Expected {len(df)} results, got {len(results)}"

    # Convert results to DataFrame
    logger.info("Processing results")
    extracted_data = []
    gantt_charts = {}
    none_count = 0

    for i, (encounter_id, result) in enumerate(zip(df['encounter_id'], results)):
        if result is None:
            # LLM call failed after retries
            none_count += 1
            extracted_data.append({
                'encounter_id': encounter_id,
                'reason_text': None,
                'category': None,
                'confidence': None,
                'explanation_support': None,
                'explanation_contrary': None,
                'process_improvement': None,
                'relevant_quotes': None,
                'extraction_failed': True,
            })
            # Add empty Gantt chart entry for failed extractions (single-call mode only)
            if not two_stage_mode:
                gantt_charts[encounter_id] = None
        else:
            # Successful extraction
            assert isinstance(result, (GenericReasons, GenericReasonsBase, ReasonsOnly)), \
                f"Expected GenericReasons, GenericReasonsBase, or ReasonsOnly, got {type(result)}"

            # Extract Gantt chart data to separate JSON structure (single-call mode only)
            if not two_stage_mode:
                if hasattr(result, 'gantt_chart') and result.gantt_chart:
                    gantt_charts[encounter_id] = {
                        'index_admission_summary': result.gantt_chart.index_admission_summary,
                        'readmission_summary': result.gantt_chart.readmission_summary,
                        'events': [
                            {
                                'event_id': event.event_id,
                                'label': event.label,
                                'category': getattr(event, 'category', None),
                                'description': event.description,
                                'start_time': event.start_time,
                                'end_time': event.end_time,
                                'time_uncertainty': getattr(event, 'time_uncertainty', None),
                                'relevant_quotes': getattr(event, 'relevant_quotes', None),
                            }
                            for event in result.gantt_chart.events
                        ]
                    }
                else:
                    gantt_charts[encounter_id] = None

            if result.reasons:
                # Add one row per reason (confidence may be inline or None if using base model)
                for reason in result.reasons:
                    extracted_data.append({
                        'encounter_id': encounter_id,
                        'reason_text': reason.reason,
                        'category': reason.category,
                        'confidence': getattr(reason, 'confidence', None),
                        'explanation_support': reason.explanation_support,
                        'explanation_contrary': reason.explanation_contrary,
                        'process_improvement': reason.process_improvement,
                        'relevant_quotes': reason.relevant_quotes,
                        'extraction_failed': False,
                    })
            else:
                # No reasons found - add a single row indicating this
                extracted_data.append({
                    'encounter_id': encounter_id,
                    'reason_text': None,
                    'category': None,
                    'confidence': None,
                    'explanation_support': None,
                    'explanation_contrary': None,
                    'process_improvement': None,
                    'relevant_quotes': None,
                    'extraction_failed': False,
                })

    results_df = pd.DataFrame(extracted_data)

    # Add derived field for compatibility with downstream analysis
    # For each encounter, determine if any reasons were mentioned
    encounter_has_reasons = results_df.groupby('encounter_id')['reason_text'].apply(
        lambda x: x.notna().any()
    ).reset_index()
    encounter_has_reasons.columns = ['encounter_id', 'has_reasons_mentioned']
    results_df = results_df.merge(encounter_has_reasons, on='encounter_id', how='left')

    # Report statistics
    logger.info("Extraction Statistics")
    logger.info(f"Total encounters processed: {len(df)}")
    logger.info(f"Failed extractions: {none_count}")

    # Count encounters with and without reasons
    successful_df = results_df[~results_df['extraction_failed']]
    encounters_with_reasons = successful_df[successful_df['reason_text'].notna()]['encounter_id'].nunique()
    total_encounters = successful_df['encounter_id'].nunique()

    if len(successful_df) > 0:
        logger.info(f"Successful extractions: {total_encounters} encounters")
        logger.info(f"Encounters with reasons: {encounters_with_reasons}")
        logger.info(f"Detection rate: {encounters_with_reasons/total_encounters:.1%}")

        # Reason statistics
        reasons_df = successful_df[successful_df['reason_text'].notna()]
        if len(reasons_df) > 0:
            logger.info(f"Total reasons extracted: {len(reasons_df)}")
            logger.info("Category distribution:")
            for category, count in reasons_df['category'].value_counts().items():
                logger.info(f"  {category}: {count}")
            logger.info("Confidence distribution (mean ± std):")
            logger.info(f"Overall: {reasons_df['confidence'].mean():.3f} ± {reasons_df['confidence'].std():.3f}")
            for category in reasons_df['category'].unique():
                cat_df = reasons_df[reasons_df['category'] == category]
                logger.info(f"{category.title()}: {cat_df['confidence'].mean():.3f} ± {cat_df['confidence'].std():.3f}")

    # Save results to CSV
    logger.info(f"Saving results to CSV: {output_csv_path}")
    results_df.to_csv(output_csv_path, index=False)

    # Save Gantt charts to JSON file (single-call mode only)
    if not two_stage_mode:
        logger.info(f"Saving Gantt charts to JSON: {output_gantt_json_path}")
        with open(output_gantt_json_path, 'w') as f:
            json.dump(gantt_charts, f, indent=2)
        logger.info(f"Saved {len([g for g in gantt_charts.values() if g is not None])} Gantt charts out of {len(gantt_charts)} encounters")
    else:
        logger.info("Two-stage mode: Gantt charts already exist in input JSON, not saving output")

    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured reasons from notes using LLM with custom prompt template"
    )
    parser.add_argument(
        "--input-csv",
        default="output/assembled_notes.csv",
        help="Path to assembled notes CSV file from Phase 1"
    )
    parser.add_argument(
        "--output-csv",
        default="output/extracted_reasons.csv",
        help="Path to output CSV file for extracted reasons"
    )
    parser.add_argument(
        "--output-gantt-json",
        "--gantt-json",  # backwards compatibility alias
        default="output/gantt_charts.json",
        help="Path to output JSON file for Gantt charts (single-call mode only)"
    )
    parser.add_argument(
        "--input-gantt-json",
        default=None,
        help="Path to input Gantt charts JSON (two-stage mode). When provided, uses ReasonsOnly model and prompt must include {gantt_chart_json} placeholder."
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Path to prompt template file (must contain {note_text} placeholder)"
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database"
    )
    parser.add_argument(
        "--log-file",
        default="data/extraction.txt",
        help="Path to log file (txt format)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for LLM processing"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum tries for failed extractions"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="LLM model to use (e.g., 'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of notes to process (for testing)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for shuffling data before applying --limit (for reproducibility). If not provided, data is not shuffled."
    )
    parser.add_argument(
        "--max-char",
        type=int,
        default=None,
        help="Maximum character length for note_text (filter applied after --limit)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--filter-csv",
        default=None,
        help="Path to filter CSV file (only process encounters with keep=1)"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable bloatectomy deduplication of clinical notes (deduplication is on by default)"
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Use GenericReasonsBase (no inline confidence) instead of GenericReasons. Use this when confidence scoring is done in a separate step."
    )

    args = parser.parse_args()

    # Handle list-models command
    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return

    # Validate model string early
    try:
        model = parse_model_string(args.model)
        print(f"✓ Model validation passed: {model.name}")
    except ValueError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # Create data directories if needed
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.input_gantt_json is None:
        Path(args.output_gantt_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Run extraction
    df = extract_reasons(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        output_gantt_json_path=args.output_gantt_json,
        prompt_template_path=args.prompt_template,
        cache_db_path=args.cache_db,
        log_file_path=args.log_file,
        model_str=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        temperature=args.temperature,
        limit=args.limit,
        filter_csv_path=args.filter_csv,
        random_seed=args.random_seed,
        max_char=args.max_char,
        deduplicate=not args.no_deduplicate,
        no_confidence=args.no_confidence,
        input_gantt_json_path=args.input_gantt_json,
    )

    # Get logger for final summary
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully extracted reasons from {len(df)} notes")
    logger.info(f"Model used: {args.model}")
    logger.info(f"Reasons CSV: {args.output_csv}")
    if args.input_gantt_json is None:
        logger.info(f"Gantt JSON: {args.output_gantt_json}")
    else:
        logger.info(f"Input Gantt JSON: {args.input_gantt_json}")


if __name__ == "__main__":
    main()
