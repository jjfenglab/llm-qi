#!/usr/bin/env python3
"""
Filter extracted reasons using two-level conformal inference.

This script takes scored reasons and filters them using the conformal inference
procedure to:
1. Select only reasons in the prediction set (Level 1)
2. Keep only encounters whose score upper bound meets the minimum threshold (Level 2)

Usage:
    python src/filter_reasons_conformal.py \
        --input-csv output/scored_reasons.csv \
        --output-csv output/filtered_reasons.csv \
        --db-path exp_los/annotations.db \
        --prompt-id exp_los_v8 \
        --min-score 4 \
        --alpha 0.1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_inference import ConformalInference
from eval_conformal_inference import load_annotations_for_conformal, map_llm_confidence_to_score


def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up logger to write to file and console."""
    logger = logging.getLogger("filter_conformal")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


def filter_reasons_conformal(
    input_csv_path: str,
    output_csv_path: str,
    db_path: str,
    prompt_id: str,
    min_score: float,
    alpha: float,
    log_file: str = None,
    bin_cutpoints: list = None,
) -> pd.DataFrame:
    """Filter extracted reasons using two-level conformal inference.

    Args:
        input_csv_path: Path to scored reasons CSV (output from score_confidences.py)
        output_csv_path: Path to output filtered reasons CSV
        db_path: Path to DuckDB database with calibration annotations
        prompt_id: Prompt ID used for calibration annotations
        min_score: Minimum score upper bound threshold (e.g., 4)
        alpha: Miscoverage rate for conformal inference (e.g., 0.1)
        log_file: Optional path to log file
        bin_cutpoints: List of 4 cutpoints for binning LLM confidence (default: [50, 60, 70, 80])

    Returns:
        DataFrame with filtered reasons
    """
    if bin_cutpoints is None:
        bin_cutpoints = [50, 60, 70, 80]
    logger = setup_logger(log_file)

    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV not found: {input_csv_path}"
    assert Path(db_path).exists(), f"Database not found: {db_path}"

    logger.info(f"Filtering reasons using two-level conformal inference")
    logger.info(f"Input CSV: {input_csv_path}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Calibration prompt ID: {prompt_id}")
    logger.info(f"Min score threshold: {min_score}")
    logger.info(f"Alpha (joint miscoverage): {alpha}")
    logger.info(f"Bin cutpoints: {bin_cutpoints}")

    # Compute per-level alpha for two-level method
    # (1 - per_level_alpha)^2 = 1 - alpha => per_level_alpha = 1 - sqrt(1 - alpha)
    per_level_alpha = 1 - np.sqrt(1 - alpha)
    logger.info(f"Per-level alpha: {per_level_alpha:.4f}")
    logger.info(f"Joint coverage target: {(1 - per_level_alpha) ** 2:.1%}")

    # Load and fit conformal inference model on calibration data
    logger.info("Loading calibration annotations...")
    calib_df = load_annotations_for_conformal(db_path, prompt_id)
    assert len(calib_df) > 0, f"No annotations found for prompt_id: {prompt_id}"
    logger.info(f"Loaded {len(calib_df)} calibration annotations")

    # Map LLM confidence (0-100) to score scale (1-5) for calibration data
    logger.info("Mapping calibration LLM confidence (0-100) to score scale (1-5)")
    calib_df["LLM_confidence"] = map_llm_confidence_to_score(calib_df["LLM_confidence"].values, bin_cutpoints=bin_cutpoints)

    # Fit conformal model
    logger.info("Fitting conformal inference model...")
    ci = ConformalInference()
    fit_results = ci.fit(calib_df, alpha=per_level_alpha)
    logger.info(f"Fitted on {fit_results['n_observations']} observations")
    logger.info(f"q1 (prediction set threshold): {fit_results['q1']:.2f}")

    q2 = ci.get_quantile_q2(per_level_alpha)
    logger.info(f"q2 (score bound addition): {q2:.2f}")

    # Load scored reasons
    logger.info(f"Loading scored reasons from: {input_csv_path}")
    reasons_df = pd.read_csv(input_csv_path)
    assert "encounter_id" in reasons_df.columns, "Missing encounter_id column"
    assert "confidence" in reasons_df.columns, "Missing confidence column"

    logger.info(f"Loaded {len(reasons_df)} reasons from {reasons_df['encounter_id'].nunique()} encounters")

    # Filter to valid reasons (non-null confidence, not extraction_failed)
    valid_mask = (
        reasons_df["confidence"].notna() &
        (~reasons_df.get("extraction_failed", False))
    )
    valid_reasons = reasons_df[valid_mask].copy()
    logger.info(f"Valid reasons: {len(valid_reasons)} from {valid_reasons['encounter_id'].nunique()} encounters")

    # Map confidence (0-100) to score scale (1-5) for conformal inference
    valid_reasons["LLM_confidence"] = map_llm_confidence_to_score(valid_reasons["confidence"].values, bin_cutpoints=bin_cutpoints)

    # Apply conformal inference to get prediction sets and upper bounds
    logger.info("Applying conformal inference to scored reasons...")
    filtered_df, metadata = ci.get_prediction_sets_batch(valid_reasons, alpha=per_level_alpha)

    logger.info(f"Conformal inference results:")
    logger.info(f"  Mean prediction set size: {metadata['mean_set_size']:.2f}")
    logger.info(f"  Mean set fraction: {metadata['mean_set_fraction']:.1%}")
    logger.info(f"  Mean score upper bound: {metadata['mean_score_upper_bound']:.2f}")

    # Filter to reasons in prediction set
    in_set_df = filtered_df[filtered_df["in_prediction_set"]].copy()
    logger.info(f"Reasons in prediction sets: {len(in_set_df)}")

    # Filter encounters by score upper bound threshold
    encounter_bounds = in_set_df.groupby("encounter_id")["score_upper_bound"].first()
    passing_encounters = encounter_bounds[encounter_bounds >= min_score].index
    logger.info(f"Encounters with upper bound >= {min_score}: {len(passing_encounters)}")

    # Final filtered dataset
    final_df = in_set_df[in_set_df["encounter_id"].isin(passing_encounters)].copy()

    # Clean up columns - drop temporary conformal inference columns
    cols_to_drop = ["LLM_confidence", "in_prediction_set", "score_upper_bound"]
    for col in cols_to_drop:
        if col in final_df.columns:
            final_df = final_df.drop(columns=[col])

    # Log statistics
    logger.info("=" * 60)
    logger.info("FILTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original reasons: {len(reasons_df)}")
    logger.info(f"Original encounters: {reasons_df['encounter_id'].nunique()}")
    logger.info(f"Valid reasons: {len(valid_reasons)}")
    logger.info(f"Reasons in prediction sets: {len(in_set_df)}")
    logger.info(f"Passing encounters (upper bound >= {min_score}): {len(passing_encounters)}")
    logger.info(f"Final filtered reasons: {len(final_df)}")
    logger.info(f"Final filtered encounters: {final_df['encounter_id'].nunique()}")
    logger.info(f"Reason retention rate: {len(final_df) / len(reasons_df):.1%}")
    logger.info(f"Encounter retention rate: {final_df['encounter_id'].nunique() / reasons_df['encounter_id'].nunique():.1%}")

    # Score upper bound distribution for passing encounters
    passing_bounds = encounter_bounds[encounter_bounds >= min_score]
    logger.info(f"\nScore upper bounds for passing encounters:")
    logger.info(f"  Mean: {passing_bounds.mean():.2f}")
    logger.info(f"  Median: {passing_bounds.median():.2f}")
    logger.info(f"  Min: {passing_bounds.min():.2f}")
    logger.info(f"  Max: {passing_bounds.max():.2f}")

    # Save output
    logger.info(f"\nSaving filtered reasons to: {output_csv_path}")
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)

    logger.info("Filtering complete")
    return final_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter extracted reasons using two-level conformal inference"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to scored reasons CSV (output from score_confidences.py)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output filtered reasons CSV",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to DuckDB database with calibration annotations",
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        help="Prompt ID used for calibration annotations",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=4.0,
        help="Minimum score upper bound threshold (default: 4.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage rate for conformal inference (default: 0.1)",
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (optional)",
    )
    parser.add_argument(
        "--bin-range",
        type=int,
        nargs=4,
        default=[50, 60, 70, 80],
        metavar=("C1", "C2", "C3", "C4"),
        help="Cutpoints for binning LLM confidence (0-100) to scores (1-5). Default: 50 60 70 80",
    )

    args = parser.parse_args()

    filter_reasons_conformal(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        db_path=args.db_path,
        prompt_id=args.prompt_id,
        min_score=args.min_score,
        alpha=args.alpha,
        log_file=args.log_file,
        bin_cutpoints=args.bin_range,
    )


if __name__ == "__main__":
    main()
