"""Filter notes using LLM assessment.

This script calls an LLM (via lab_llm) to filter clinical notes based on a
prompt (e.g., whether the patient truly had an unplanned readmission). It
outputs a CSV with encounter_id, keep (1/0), explanation, and quote.

This should be run prior to extract_reasons.py to filter out irrelevant observations.

Usage:
    python src/filter_notes.py --input-csv <path> --output-csv <path> --prompt-template <path>
    python src/filter_notes.py --model gpt-4o-2024-08-06 --limit 10
"""

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd


def filter_notes_passthrough(
    input_csv_path: str,
    output_csv_path: str,
    limit: int = None
) -> pd.DataFrame:
    """Passthrough filter that keeps all notes (no actual filtering).

    Args:
        input_csv_path: Path to assembled notes CSV file
        output_csv_path: Path to output CSV file with filter results
        limit: Optional limit on number of notes to process

    Returns:
        DataFrame with all rows set to keep=1
    """
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading notes from CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    if limit:
        logger.info(f"Limiting to {limit} notes for testing")
        df = df.head(limit)

    assert len(df) > 0, "No notes found"
    assert 'encounter_id' in df.columns, "Missing encounter_id column"

    logger.info("Running passthrough filter (no-filter mode, all rows keep=1)")

    results_df = pd.DataFrame({
        'encounter_id': df['encounter_id'],
        'keep': 1,
        'explanation': 'Passthrough filter - no filtering applied',
        'quote': None,
    })

    logger.info(f"Total encounters: {len(results_df)} (all kept)")
    logger.info(f"Saving results to CSV: {output_csv_path}")
    results_df.to_csv(output_csv_path, index=False)

    return results_df


def filter_notes(
    input_csv_path: str,
    output_csv_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "data/filter.txt",
    model_str: str = "gpt-4o-mini-2024-07-18",
    batch_size: int = 10,
    max_retries: int = 2,
    temperature: float = 0,
    limit: int = None
) -> pd.DataFrame:
    """Filter notes using LLM assessment.

    Args:
        input_csv_path: Path to assembled notes CSV file
        output_csv_path: Path to output CSV file with filter results
        prompt_template_path: Path to prompt template file
        cache_db_path: Path to cache database
        log_file_path: Path to log file
        model_str: LLM model to use
        batch_size: Batch size for LLM processing
        max_retries: Maximum retries for failed calls
        temperature: LLM temperature
        limit: Optional limit on number of notes to process

    Returns:
        DataFrame with filter results
    """
    from lab_llm import TextDataset
    
    from common import (
        list_available_models,
        load_prompt_template,
        create_prompt_from_template,
        parse_model_string,
        setup_llm_api,
    )
    from models import FilterResult

    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"

    logger = logging.getLogger(__name__)

    logger.info(f"Loading notes from CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    if limit:
        logger.info(f"Limiting to {limit} notes for testing")
        df = df.head(limit)

    assert len(df) > 0, "No notes found"
    assert 'encounter_id' in df.columns, "Missing encounter_id column"
    assert 'note_text' in df.columns, "Missing note_text column"

    logger.info("Setting up LLM API")
    api = setup_llm_api(cache_db_path, log_file_path, model_str)

    logger.info(f"Loading prompt template from {prompt_template_path}")
    prompt_template = load_prompt_template(prompt_template_path)

    logger.info("Creating prompts")
    prompts = [create_prompt_from_template(prompt_template, note) for note in df['note_text']]
    assert len(prompts) == len(df), "Mismatch between prompts and notes"

    dataset = TextDataset(prompts)

    logger.info(f"Running filter assessment (batch_size={batch_size})")
    results = asyncio.run(
        api.get_outputs(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=1000,
            temperature=temperature,
            max_retries=max_retries,
            response_model=FilterResult,
        )
    )

    assert len(results) == len(df), f"Expected {len(df)} results, got {len(results)}"

    logger.info("Processing results")
    filter_data = []
    none_count = 0

    for encounter_id, result in zip(df['encounter_id'], results):
        if result is None:
            none_count += 1
            filter_data.append({
                'encounter_id': encounter_id,
                'keep': None,
                'explanation': None,
                'quote': None,
            })
        else:
            assert isinstance(result, FilterResult), f"Expected FilterResult, got {type(result)}"
            filter_data.append({
                'encounter_id': encounter_id,
                'keep': result.keep,
                'explanation': result.explanation,
                'quote': result.quote,
            })

    results_df = pd.DataFrame(filter_data)

    # Report statistics
    logger.info("Filter Statistics")
    logger.info(f"Total encounters processed: {len(df)}")
    logger.info(f"Failed assessments (keep=None): {none_count}")

    successful_df = results_df[results_df['keep'].notna()]
    if len(successful_df) > 0:
        keep_count = (successful_df['keep'] == 1).sum()
        exclude_count = (successful_df['keep'] == 0).sum()
        logger.info(f"Successful assessments: {len(successful_df)}")
        logger.info(f"Keep (1): {keep_count} ({keep_count/len(successful_df):.1%})")
        logger.info(f"Exclude (0): {exclude_count} ({exclude_count/len(successful_df):.1%})")

    logger.info(f"Saving results to CSV: {output_csv_path}")
    results_df.to_csv(output_csv_path, index=False)

    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter notes using LLM assessment"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to assembled notes CSV file"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file with filter results"
    )
    parser.add_argument(
        "--prompt-template",
        help="Path to prompt template file (must contain {note_text} placeholder). Required unless --no-filter is set."
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip LLM filtering and keep all rows (sets keep=1 for all)"
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database"
    )
    parser.add_argument(
        "--log-file",
        default="data/filter.txt",
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
        default=1,
        help="Maximum retries for failed calls"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of notes to process (for testing)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.no_filter:
        # Passthrough mode - no LLM filtering
        df = filter_notes_passthrough(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            limit=args.limit
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Successfully processed {len(df)} notes (no-filter mode)")
        logger.info(f"Output CSV: {args.output_csv}")
    else:
        # LLM filtering mode
        assert args.prompt_template is not None, "--prompt-template is required when not using --no-filter"

        try:
            model = parse_model_string(args.model)
            print(f"Model validation passed: {model.name}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

        df = filter_notes(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            prompt_template_path=args.prompt_template,
            cache_db_path=args.cache_db,
            log_file_path=args.log_file,
            model_str=args.model,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            temperature=args.temperature,
            limit=args.limit
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Successfully filtered {len(df)} notes")
        logger.info(f"Model used: {args.model}")
        logger.info(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
