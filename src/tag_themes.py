#!/usr/bin/env python3
"""Tag extracted reasons with themes using LLM classification.

This script takes filtered reasons and a themes JSON file, then uses an LLM to
classify each reason into one or more themes based on semantic similarity to
the theme members.

Usage:
    python src/tag_themes.py \
        --input-csv output/filtered_reasons.csv \
        --themes-json exp_los/_output/all/prompt_v9/default/cluster_members_claude_clean.json \
        --output-csv output/tagged_reasons.csv \
        --prompt-template exp_los/prompts/theme_tagging_template.txt
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from lab_llm import TextDataset

from common import (
    list_available_models,
    parse_model_string,
    setup_llm_api,
)


class ThemeTagging(BaseModel):
    """Response model for theme tagging."""
    themes: List[str] = Field(
        default_factory=list,
        description="List of theme IDs that match reason for extended LOS. Empty if no themes match."
    )


def load_themes_json(themes_json_path: str) -> dict:
    """Load themes from JSON file.

    Args:
        themes_json_path: Path to themes JSON file

    Returns:
        Dictionary with theme_id as keys
    """
    assert Path(themes_json_path).exists(), f"Themes JSON not found: {themes_json_path}"

    with open(themes_json_path, 'r', encoding='utf-8') as f:
        themes = json.load(f)

    assert isinstance(themes, dict), "Themes JSON must be a dictionary"
    assert len(themes) > 0, "Themes JSON must not be empty"

    return themes


def format_themes_text(themes: dict) -> str:
    """Format themes dictionary into text for prompt template.

    Args:
        themes: Dictionary of themes with their members

    Returns:
        Formatted string describing all themes
    """
    themes_desc = []
    for theme_id, theme_data in themes.items():
        topic_name = theme_data.get("topic_name", theme_id)
        theme_entry = f"  Theme ID: {theme_id}\n    Full name: {topic_name}"
        themes_desc.append(theme_entry)

    return "\n".join(themes_desc)


def create_theme_tagging_prompt(
    template: str,
    reason_text: str,
    reason_explanation: str,
    themes_text: str,
) -> str:
    """Create a prompt for theme tagging from template.

    Args:
        template: Prompt template with {reason_text} and {themes_text} placeholders
        reason_text: The reason text to classify
        themes_text: Formatted themes description

    Returns:
        Formatted prompt string
    """
    return template.format(
        reason_text=reason_text,
        reason_explanation=reason_explanation,
        themes_text=themes_text,
    )


def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up logger to write to file and console."""
    logger = logging.getLogger("tag_themes")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


def tag_themes(
    input_csv_path: str,
    themes_json_path: str,
    output_csv_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "output/theme_tagging.txt",
    model_str: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    batch_size: int = 10,
    max_retries: int = 2,
    temperature: float = 0,
    limit: int = None,
    random_seed: int = None,
    reason_column: str = "reason_text",
    min_confidence: float = None,
) -> pd.DataFrame:
    """Tag extracted reasons with themes using LLM classification.

    Args:
        input_csv_path: Path to filtered reasons CSV
        themes_json_path: Path to themes JSON file
        output_csv_path: Path to output tagged CSV
        prompt_template_path: Path to prompt template file
        cache_db_path: Path to cache database
        log_file_path: Path to log file
        model_str: LLM model to use
        batch_size: Batch size for LLM processing
        max_retries: Maximum retries for failed calls
        temperature: Temperature for LLM sampling
        limit: Optional limit on number of reasons to process
        random_seed: Random seed for shuffling before limit
        reason_column: Column name containing the reason text
        min_confidence: Minimum LLM score to include a reason (filters on 'confidence' column)

    Returns:
        DataFrame with tagged reasons
    """
    logger = setup_logger(log_file_path)

    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV not found: {input_csv_path}"
    assert Path(themes_json_path).exists(), f"Themes JSON not found: {themes_json_path}"
    assert Path(prompt_template_path).exists(), f"Prompt template not found: {prompt_template_path}"

    logger.info(f"Tagging reasons with themes")
    logger.info(f"Input CSV: {input_csv_path}")
    logger.info(f"Themes JSON: {themes_json_path}")
    logger.info(f"Prompt template: {prompt_template_path}")
    logger.info(f"Model: {model_str}")

    # Load themes
    logger.info("Loading themes...")
    themes = load_themes_json(themes_json_path)
    logger.info(f"Loaded {len(themes)} themes")

    # Load prompt template
    logger.info(f"Loading prompt template from {prompt_template_path}")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()
    assert "{reason_text}" in prompt_template, "Prompt template must contain {reason_text} placeholder"
    assert "{themes_text}" in prompt_template, "Prompt template must contain {themes_text} placeholder"

    # Format themes text once (same for all prompts)
    themes_text = format_themes_text(themes)

    # Load filtered reasons
    logger.info("Loading filtered reasons...")
    df = pd.read_csv(input_csv_path)
    assert reason_column in df.columns, f"Missing '{reason_column}' column in CSV"
    logger.info(f"Loaded {len(df)} reasons from {df['encounter_id'].nunique()} encounters")

    # Filter by minimum LLM score if specified
    if min_confidence is not None:
        assert 'confidence' in df.columns, f"Column 'confidence' not found in input CSV. Available columns: {list(df.columns)}"
        original_count = len(df)
        df = df[df['confidence'] >= min_confidence].copy()
        logger.info(f"Filtered by min_confidence >= {min_confidence}: {original_count} -> {len(df)} reasons")
        assert len(df) > 0, f"No reasons remaining after filtering by min_confidence >= {min_confidence}"

    # Apply limit with optional shuffle
    if limit:
        if random_seed is not None:
            logger.info(f"Shuffling with seed={random_seed}, then limiting to {limit}")
            df = df.sample(frac=1, random_state=random_seed).head(limit)
        else:
            logger.info(f"Limiting to {limit} reasons")
            df = df.head(limit)

    assert len(df) > 0, "No reasons to process after filtering"

    # Setup LLM API
    logger.info("Setting up LLM API...")
    api = setup_llm_api(cache_db_path, log_file_path, model_str)

    # Create prompts
    logger.info("Creating prompts...")
    prompts = [
        create_theme_tagging_prompt(prompt_template, df[reason_column].iloc[idx],  df["explanation_support"].iloc[idx], themes_text)
        for idx in range(df.shape[0])
    ]
    print(prompts[0])

    # Create dataset
    dataset = TextDataset(prompts)

    # Run batch tagging
    logger.info(f"Running theme tagging (batch_size={batch_size})...")
    results = asyncio.run(
        api.get_outputs(
            dataset=dataset,
            batch_size=batch_size,
            max_new_tokens=1000,
            temperature=temperature,
            max_retries=max_retries,
            response_model=ThemeTagging,
        )
    )

    # Validate results
    assert len(results) == len(df), f"Expected {len(df)} results, got {len(results)}"

    # Process results
    logger.info("Processing results...")
    tagged_data = []
    failed_count = 0
    total_themes_assigned = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        result = results[idx]

        if result is None:
            failed_count += 1
            tagged_data.append({
                **row.to_dict(),
                'theme_names': None,
                'tagging_failed': True,
            })
        else:
            assert isinstance(result, ThemeTagging), f"Expected ThemeTagging, got {type(result)}"

            # Extract theme assignments
            theme_names = [themes.get(t, {}).get('topic_name', t) for t in result.themes]
            total_themes_assigned += len(theme_names)

            tagged_data.append({
                **row.to_dict(),
                'theme_names': json.dumps(theme_names) if theme_names else None,
                'tagging_failed': False,
            })

    results_df = pd.DataFrame(tagged_data)

    # Log statistics
    logger.info("=" * 60)
    logger.info("TAGGING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total reasons processed: {len(df)}")
    logger.info(f"Failed tagging: {failed_count}")
    logger.info(f"Total theme assignments: {total_themes_assigned}")
    logger.info(f"Mean themes per reason: {total_themes_assigned / len(df):.2f}")

    # Count reasons with at least one theme
    has_theme = results_df['theme_names'].notna()
    logger.info(f"Reasons with at least one theme: {has_theme.sum()} ({has_theme.mean():.1%})")

    # Theme distribution
    theme_counts = {}
    for theme_names_json in results_df[results_df['theme_names'].notna()]['theme_names']:
        for theme_name in json.loads(theme_names_json):
            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1

    logger.info("\nTheme distribution:")
    for theme_id, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        topic_name = themes.get(theme_id, {}).get('topic_name', theme_id)
        logger.info(f"  {topic_name}: {count}")

    # Save output
    logger.info(f"\nSaving tagged reasons to: {output_csv_path}")
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)

    logger.info("Tagging complete")
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tag extracted reasons with themes using LLM classification"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to filtered reasons CSV",
    )
    parser.add_argument(
        "--themes-json",
        required=True,
        help="Path to themes JSON file (cluster_members_claude_clean.json)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output tagged CSV",
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Path to prompt template file (must contain {reason_text} and {themes_text} placeholders)",
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database",
    )
    parser.add_argument(
        "--log-file",
        default="output/theme_tagging.txt",
        help="Path to log file",
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="LLM model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for LLM processing",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries for failed LLM calls",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for LLM sampling",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of reasons to process (for testing)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for shuffling before applying --limit",
    )
    parser.add_argument(
        "--reason-column",
        default="reason_text",
        help="Column name containing the reason text",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum LLM score to include a reason (filters on 'confidence' column)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
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
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Run tagging
    tag_themes(
        input_csv_path=args.input_csv,
        themes_json_path=args.themes_json,
        output_csv_path=args.output_csv,
        prompt_template_path=args.prompt_template,
        cache_db_path=args.cache_db,
        log_file_path=args.log_file,
        model_str=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        temperature=args.temperature,
        limit=args.limit,
        random_seed=args.random_seed,
        reason_column=args.reason_column,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
