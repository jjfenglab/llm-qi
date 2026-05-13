"""Score confidence for extracted reasons using LLM.

This script takes the output from src/extract_reasons.py and rescores the confidence
values using a dedicated confidence scoring prompt template.

Usage:
    python src/score_confidences.py --input-csv <path> --output-csv <path> --prompt-template <path>
    python src/score_confidences.py --model gpt-4o-2024-08-06 --limit 10
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
from models.confidence_scoring import ConfidenceScores


def create_confidence_prompt(
    template: str,
    note_text: str,
    gantt_chart_json: str,
    extracted_reasons_text: str
) -> str:
    """Create a confidence scoring prompt from template by substituting all placeholders.

    Args:
        template: Prompt template with placeholders
        note_text: The clinical note text
        gantt_chart_json: JSON representation of the Gantt chart
        extracted_reasons_text: Text representation of extracted reasons (without confidence)

    Returns:
        Formatted prompt string for LLM confidence scoring
    """
    return template.format(
        note_text=note_text,
        gantt_chart_json=gantt_chart_json,
        extracted_reasons=extracted_reasons_text
    )


def format_reasons_for_prompt(reasons_data: list) -> str:
    """Format extracted reasons for inclusion in the confidence scoring prompt.

    Args:
        reasons_data: List of reason dictionaries from the CSV

    Returns:
        Formatted string representation of reasons (excluding confidence scores)
    """
    if not reasons_data:
        return "No reasons were identified."

    formatted_reasons = []
    for i, reason in enumerate(reasons_data, 1):
        formatted_reason = f"{i}. Reason Name: {reason['reason_text']}\n"
        formatted_reason += f"   Explanation Support: {reason['explanation_support']}\n"
        formatted_reason += f"   Explanation Contrary: {reason['explanation_contrary']}\n"
        formatted_reason += f"   Relevant Quotes: {reason['relevant_quotes']}\n"
        if reason.get('process_improvement'):
            formatted_reason += f"   Process Improvement: {reason['process_improvement']}\n"
        formatted_reasons.append(formatted_reason)

    return "\n".join(formatted_reasons)


def score_confidences(
    input_csv_path: str,
    output_csv_path: str,
    gantt_json_path: str,
    notes_csv_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "data/confidence_scoring.txt",
    model_str: str = "gpt-4o-mini-2024-07-18",
    batch_size: int = 10,
    max_retries: int = 2,
    temperature: float = 0,
    limit: int = None,
    random_seed: int = None,
    deduplicate: bool = True
) -> pd.DataFrame:
    """Score confidence for extracted reasons using LLM.

    Args:
        input_csv_path: Path to CSV file with extracted reasons (output from extract_reasons.py)
        output_csv_path: Path to output CSV file with updated confidence scores
        gantt_json_path: Path to JSON file with Gantt charts (from extract_reasons.py)
        notes_csv_path: Path to assembled notes CSV file (original note text)
        prompt_template_path: Path to confidence scoring prompt template
        cache_db_path: Path to cache database
        log_file_path: Path to log file
        model_str: Model string for LLM
        batch_size: Batch size for LLM processing
        max_retries: Maximum retries for failed calls
        temperature: Temperature for LLM generation
        limit: Optional limit on encounters to process
        random_seed: Random seed for sampling
        deduplicate: Whether to deduplicate notes

    Returns:
        DataFrame with updated confidence scores
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"
    assert Path(gantt_json_path).exists(), f"Gantt JSON file not found: {gantt_json_path}"
    assert Path(notes_csv_path).exists(), f"Notes CSV file not found: {notes_csv_path}"

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

    # Load extracted reasons
    logger.info(f"Loading extracted reasons from CSV: {input_csv_path}")
    reasons_df = pd.read_csv(input_csv_path)

    # Shuffle rows within each encounter to mitigate position bias
    logger.info(f"Shuffling rows within encounter with seed")
    rng = np.random.default_rng(random_seed)
    reasons_df = reasons_df.groupby('encounter_id', group_keys=False).apply(
        lambda g: g.iloc[rng.permutation(len(g))]
    ).reset_index(drop=True)

    # Load original notes
    logger.info(f"Loading original notes from CSV: {notes_csv_path}")
    notes_df = pd.read_csv(notes_csv_path)
    assert 'encounter_id' in notes_df.columns, "Missing encounter_id column in notes CSV"
    assert 'note_text' in notes_df.columns, "Missing note_text column in notes CSV"

    # Load Gantt charts
    logger.info(f"Loading Gantt charts from JSON: {gantt_json_path}")
    with open(gantt_json_path, 'r') as f:
        gantt_charts = json.load(f)

    # Filter to encounters that have reasons and successful extractions
    valid_encounters = reasons_df[
        (~reasons_df['extraction_failed']) &
        (reasons_df['reason_text'].notna())
    ]['encounter_id'].unique()

    logger.info(f"Found {len(valid_encounters)} encounters with valid extracted reasons")

    if limit:
        if random_seed is not None:
            logger.info(f"Shuffling encounters with random_seed={random_seed}, then limiting to {limit}")
            import random
            random.seed(random_seed)
            valid_encounters = random.sample(list(valid_encounters), min(limit, len(valid_encounters)))
        else:
            logger.info(f"Limiting to {limit} encounters (no shuffle)")
            valid_encounters = valid_encounters[:limit]

    # Setup LLM API
    logger.info("Setting up LLM API")
    api = setup_llm_api(cache_db_path, log_file_path, model_str, timeout=60 * 5)

    # Load prompt template
    logger.info(f"Loading confidence scoring template from {prompt_template_path}")
    prompt_template = load_prompt_template(prompt_template_path)

    # Validate template has all required placeholders
    required_placeholders = ["{note_text}", "{gantt_chart_json}", "{extracted_reasons}"]
    for placeholder in required_placeholders:
        assert placeholder in prompt_template, f"Prompt template must contain {placeholder} placeholder"

    # Prepare data for each encounter
    logger.info("Preparing prompts for confidence scoring")
    encounter_data = []
    prompts = []

    for encounter_id in valid_encounters:
        # Get original note text
        encounter_notes = notes_df[notes_df['encounter_id'] == encounter_id]
        assert len(encounter_notes) > 0, f"No note found for encounter {encounter_id}"

        note_text = encounter_notes.iloc[0]['note_text']
        if deduplicate:
            note_text = deduplicate_note(note_text)

        # Get Gantt chart
        gantt_chart = gantt_charts.get(str(encounter_id))
        if gantt_chart is None:
            logger.warning(f"No Gantt chart found for encounter {encounter_id}, skipping")
            continue

        gantt_chart_json = json.dumps(gantt_chart, indent=2)

        # Get extracted reasons for this encounter
        encounter_reasons = reasons_df[
            (reasons_df['encounter_id'] == encounter_id) &
            (reasons_df['reason_text'].notna())
        ]

        if len(encounter_reasons) == 0:
            logger.warning(f"No valid reasons found for encounter {encounter_id}, skipping")
            continue

        # Format reasons for prompt (exclude confidence)
        reasons_data = encounter_reasons.to_dict('records')
        extracted_reasons_text = format_reasons_for_prompt(reasons_data)

        # Create prompt
        prompt = create_confidence_prompt(
            prompt_template,
            note_text,
            gantt_chart_json,
            extracted_reasons_text
        )

        encounter_data.append({
            'encounter_id': encounter_id,
            'reasons_data': reasons_data,
            'original_indices': encounter_reasons.index.tolist()
        })
        prompts.append(prompt)

    logger.info(f"Created {len(prompts)} confidence scoring prompts")

    if len(prompts) == 0:
        logger.warning("No valid encounters to process")
        return reasons_df

    # Create dataset and run batch scoring
    dataset = TextDataset(prompts)

    logger.info(f"Scoring confidences (batch_size={batch_size})")
    results = asyncio.run(
        api.get_outputs(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=5000,
            temperature=temperature,
            max_retries=max_retries,
            response_model=ConfidenceScores,
        )
    )

    assert len(results) == len(encounter_data), f"Expected {len(encounter_data)} results, got {len(results)}"

    # Update confidence scores in the original DataFrame
    logger.info("Updating confidence scores in DataFrame")
    updated_df = reasons_df.copy()
    confidence_update_count = 0
    failed_count = 0

    for i, (encounter_info, result) in enumerate(zip(encounter_data, results)):
        encounter_id = encounter_info['encounter_id']
        original_indices = encounter_info['original_indices']
        reasons_data = encounter_info['reasons_data']

        if result is None:
            failed_count += 1
            logger.warning(f"Failed to score confidence for encounter {encounter_id}")
            continue

        assert isinstance(result, ConfidenceScores), f"Expected ConfidenceScores, got {type(result)}"

        # Match scored confidences to original reasons
        if len(result.confidences) != len(reasons_data):
            logger.warning(
                f"Encounter {encounter_id}: Expected {len(reasons_data)} confidence scores, "
                f"got {len(result.confidences)}. Skipping this encounter."
            )
            failed_count += 1
            continue

        # Update confidence scores and add confidence reasons
        for j, (original_idx, confidence_score) in enumerate(zip(original_indices, result.confidences)):
            updated_df.loc[original_idx, 'confidence'] = confidence_score.confidence
            updated_df.loc[original_idx, 'confidence_reason'] = confidence_score.confidence_reason
            confidence_update_count += 1

    # Report statistics
    logger.info("Confidence Scoring Statistics")
    logger.info(f"Total encounters processed: {len(encounter_data)}")
    logger.info(f"Failed confidence scoring: {failed_count}")
    logger.info(f"Successfully updated confidence scores: {confidence_update_count}")

    # Add confidence_reason column if it doesn't exist
    if 'confidence_reason' not in updated_df.columns:
        updated_df['confidence_reason'] = None

    # Save updated results
    logger.info(f"Saving updated results to CSV: {output_csv_path}")
    updated_df.to_csv(output_csv_path, index=False)

    # Log confidence distribution
    scored_df = updated_df[updated_df['confidence'].notna()]
    if len(scored_df) > 0:
        logger.info("Updated confidence distribution:")
        logger.info(f"Mean confidence: {scored_df['confidence'].mean():.1f}")
        logger.info(f"Std confidence: {scored_df['confidence'].std():.1f}")
        logger.info("Confidence by category:")
        for category in scored_df['category'].unique():
            cat_df = scored_df[scored_df['category'] == category]
            logger.info(f"  {category}: {cat_df['confidence'].mean():.1f} ± {cat_df['confidence'].std():.1f}")

    return updated_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Score confidence for extracted reasons using LLM"
    )
    parser.add_argument(
        "--input-csv",
        default="output/extracted_reasons.csv",
        help="Path to CSV with extracted reasons (output from extract_reasons.py)"
    )
    parser.add_argument(
        "--output-csv",
        default="output/scored_reasons.csv",
        help="Path to output CSV file with updated confidence scores"
    )
    parser.add_argument(
        "--gantt-json",
        default="output/gantt_charts.json",
        help="Path to Gantt charts JSON file (from extract_reasons.py)"
    )
    parser.add_argument(
        "--notes-csv",
        default="output/assembled_notes.csv",
        help="Path to assembled notes CSV file (original note text)"
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Path to confidence scoring prompt template file"
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database"
    )
    parser.add_argument(
        "--log-file",
        default="data/confidence_scoring.txt",
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
        help="Maximum retries for failed scoring"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for LLM generation"
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
        help="Limit number of encounters to process (for testing)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for encounter sampling"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
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
        print(f"✓ Model validation passed: {model.name}")
    except ValueError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # Create data directories if needed
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Run confidence scoring
    df = score_confidences(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        gantt_json_path=args.gantt_json,
        notes_csv_path=args.notes_csv,
        prompt_template_path=args.prompt_template,
        cache_db_path=args.cache_db,
        log_file_path=args.log_file,
        model_str=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        temperature=args.temperature,
        limit=args.limit,
        random_seed=args.random_seed,
        deduplicate=not args.no_deduplicate
    )

    # Get logger for final summary
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully scored confidence for {len(df)} reason entries")
    logger.info(f"Model used: {args.model}")
    logger.info(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
