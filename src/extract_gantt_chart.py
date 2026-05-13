"""Phase 2a: Extract Gantt charts using LLM.

This script extracts Gantt charts (value stream mapping) from clinical notes.
It's the first step of the two-stage extraction pipeline (v7+).

Usage:
    python src/extract_gantt_chart.py --input-csv <path> --output-json <path> --prompt-template <path>
    python src/extract_gantt_chart.py --model gpt-4o-2024-08-06 --limit 10
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
    deduplicate_note,
    parse_model_string,
    setup_llm_api,
)
from models import GanttChartBase


def extract_gantt_charts(
    input_csv_path: str,
    output_json_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "data/gantt_extraction.txt",
    model_str: str = "gpt-4o-mini-2024-07-18",
    batch_size: int = 10,
    max_retries: int = 2,
    temperature: float = 0,
    limit: int = None,
    filter_csv_path: str = None,
    random_seed: int = None,
    max_char: int = None,
    deduplicate: bool = True,
) -> dict:
    """Extract Gantt charts from notes using LLM.

    Args:
        input_csv_path: Path to assembled notes CSV file
        output_json_path: Path to output JSON file for Gantt charts
        prompt_template_path: Path to Gantt extraction prompt template
        cache_db_path: Path to cache database
        log_file_path: Path to log file
        model_str: Model string (e.g., 'gpt-4o-2024-08-06')
        batch_size: Batch size for LLM processing
        max_retries: Maximum number of retries for failed extractions
        temperature: Temperature for generation
        limit: Optional limit on number of notes to process
        filter_csv_path: Optional path to filter CSV (only process encounters with keep=1)
        random_seed: Random seed for shuffling data before applying limit
        max_char: Optional maximum character length for note_text
        deduplicate: Whether to apply bloatectomy deduplication to notes

    Returns:
        Dictionary mapping encounter_id to GanttChartBase data
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"

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

        keep_ids = filter_df[filter_df['keep'] == 1]['encounter_id'].tolist()
        original_count = len(df)
        df = df[df['encounter_id'].isin(keep_ids)]
        logger.info(f"Filtered from {original_count} to {len(df)} encounters (keep=1)")

    if max_char is not None:
        pre_filter_count = len(df)
        df = df[df['note_text'].str.len() <= max_char]
        logger.info(f"Filtered by max_char={max_char}: {pre_filter_count} -> {len(df)} notes")

    if limit:
        if random_seed is not None:
            logger.info(f"Shuffling data with random_seed={random_seed}, then limiting to {limit} notes")
            df = df.sample(frac=1, random_state=random_seed).head(limit)
        else:
            logger.info(f"Limiting to {limit} notes (no shuffle)")
            df = df.head(limit)

    assert len(df) > 0, "No assembled notes found"
    assert 'encounter_id' in df.columns, "Missing encounter_id column"
    assert 'note_text' in df.columns, "Missing note_text column"

    # Setup LLM API
    logger.info("Setting up LLM API")
    api = setup_llm_api(cache_db_path, log_file_path, model_str, timeout=60 * 5)

    # Load prompt template
    logger.info(f"Loading prompt template from {prompt_template_path}")
    prompt_template = load_prompt_template(prompt_template_path)

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
    prompts = [
        prompt_template.format(note_text=note)
        for note in notes
    ]
    assert len(prompts) == len(df), "Mismatch between prompts and notes"

    # Create dataset
    dataset = TextDataset(prompts)

    # Run batch extraction
    logger.info(f"Extracting Gantt charts (batch_size={batch_size})")
    results = asyncio.run(
        api.get_outputs(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=4000,
            temperature=temperature,
            max_retries=max_retries,
            response_model=GanttChartBase,
        )
    )

    # Validate results
    assert len(results) == len(df), f"Expected {len(df)} results, got {len(results)}"

    # Convert results to dictionary
    logger.info("Processing results")
    gantt_charts = {}
    success_count = 0
    fail_count = 0

    for encounter_id, result in zip(df['encounter_id'], results):
        if result is None:
            fail_count += 1
            gantt_charts[str(encounter_id)] = None
        else:
            success_count += 1
            assert isinstance(result, GanttChartBase), f"Expected GanttChartBase, got {type(result)}"
            gantt_charts[str(encounter_id)] = {
                'index_admission_summary': result.index_admission_summary,
                'readmission_summary': result.readmission_summary,
                'events': [
                    {
                        'event_id': event.event_id,
                        'label': event.label,
                        'description': event.description,
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'relevant_quotes': event.relevant_quotes,
                    }
                    for event in result.events
                ]
            }

    # Report statistics
    logger.info("Gantt Extraction Statistics")
    logger.info(f"Total encounters processed: {len(df)}")
    logger.info(f"Successful extractions: {success_count}")
    logger.info(f"Failed extractions: {fail_count}")

    # Calculate average events per chart
    event_counts = [len(g['events']) for g in gantt_charts.values() if g is not None]
    if event_counts:
        logger.info(f"Average events per Gantt chart: {np.mean(event_counts):.1f}")

    # Save results to JSON
    logger.info(f"Saving Gantt charts to JSON: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(gantt_charts, f, indent=2)

    return gantt_charts


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Gantt charts from notes using LLM"
    )
    parser.add_argument(
        "--input-csv",
        default="output/assembled_notes.csv",
        help="Path to assembled notes CSV file"
    )
    parser.add_argument(
        "--output-json",
        default="output/gantt_charts.json",
        help="Path to output JSON file for Gantt charts"
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Path to Gantt extraction prompt template file"
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database"
    )
    parser.add_argument(
        "--log-file",
        default="data/gantt_extraction.txt",
        help="Path to log file"
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
        default=2,
        help="Maximum retries for failed extractions"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="LLM model to use"
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
        help="Random seed for shuffling data before applying --limit"
    )
    parser.add_argument(
        "--max-char",
        type=int,
        default=None,
        help="Maximum character length for note_text"
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
        help="Disable bloatectomy deduplication of clinical notes"
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
        print(f"Model validation passed: {model.name}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create directories if needed
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Run extraction
    gantt_charts = extract_gantt_charts(
        input_csv_path=args.input_csv,
        output_json_path=args.output_json,
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
    )

    # Get logger for final summary
    logger = logging.getLogger(__name__)
    success_count = len([g for g in gantt_charts.values() if g is not None])
    logger.info(f"Successfully extracted {success_count} Gantt charts")
    logger.info(f"Model used: {args.model}")
    logger.info(f"Output JSON: {args.output_json}")


if __name__ == "__main__":
    main()
