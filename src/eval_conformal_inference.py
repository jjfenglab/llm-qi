#!/usr/bin/env python3
"""
Evaluation script for two-level conformal inference on LLM reason annotations.

This script loads annotations from a database, fits the conformal inference model,
and reports coverage statistics and prediction set sizes.
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from conformal_inference import ConformalInference, StandardConformalInference


def map_llm_confidence_to_score(confidence: np.ndarray, bin_cutpoints: list = None) -> np.ndarray:
    """Map LLM confidence (0-100) to score scale (1-5).

    Default mapping (cutpoints=[50, 60, 70, 80]):
        ≤50: 1
        51-60: 2
        61-70: 3
        71-80: 4
        >80: 5

    Args:
        confidence: Array of LLM confidence values (0-100)
        bin_cutpoints: List of 4 cutpoints for binning (default: [50, 60, 70, 80])

    Returns:
        Array of mapped scores (1-5)
    """
    if bin_cutpoints is None:
        bin_cutpoints = [50, 60, 70, 80]

    assert len(bin_cutpoints) == 4, f"bin_cutpoints must have exactly 4 values, got {len(bin_cutpoints)}"

    scores = np.ones_like(confidence, dtype=float)
    scores = np.where(confidence > bin_cutpoints[0], 2, scores)
    scores = np.where(confidence > bin_cutpoints[1], 3, scores)
    scores = np.where(confidence > bin_cutpoints[2], 4, scores)
    scores = np.where(confidence > bin_cutpoints[3], 5, scores)
    return scores


def load_annotations_for_conformal(db_path: str, prompt_id: str) -> pd.DataFrame:
    """Load annotations from database in format suitable for conformal inference.

    Args:
        db_path: Path to DuckDB database
        prompt_id: Prompt/experiment identifier

    Returns:
        DataFrame with encounter_id, reason_id, LLM_confidence, annotation
    """
    assert Path(db_path).exists(), f"Database file not found: {db_path}"

    conn = duckdb.connect(db_path)

    df = conn.execute(
        """
        SELECT encounter_id, reason_id, reason_text, LLM_confidence, annotation, reviewer_id
        FROM validations
        WHERE prompt_id = ?
          AND feedback_type = 'reason'
          AND annotation IS NOT NULL
        ORDER BY encounter_id, reason_id, reviewer_id
        """,
        [prompt_id],
    ).df()

    conn.close()

    return df


def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up logger to write to file and console."""
    logger = logging.getLogger("conformal_inference")
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


def main():
    """Main entry point for conformal inference evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate two-level conformal inference on LLM reason annotations"
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        help="Unique identifier for the prompt/experiment",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the DuckDB database file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage rate (default: 0.1 for 90%% coverage per level)",
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

    logger = setup_logger(args.log_file)

    logger.info(f"Running two-level conformal inference for prompt_id: {args.prompt_id}")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Alpha (joint miscoverage rate): {args.alpha}")
    logger.info(f"Joint coverage target: {1-args.alpha:.1%}")
    logger.info(f"Bin cutpoints: {args.bin_range}")

    # Compute per-level alpha for two-level method to achieve joint coverage of 1-alpha
    # (1 - per_level_alpha)^2 = 1 - alpha  =>  per_level_alpha = 1 - sqrt(1 - alpha)
    per_level_alpha = 1 - np.sqrt(1 - args.alpha)
    logger.info(f"Per-level alpha (for two-level method): {per_level_alpha:.4f}")
    logger.info(f"Per-level coverage target: {1-per_level_alpha:.1%}")

    # Load annotations
    df = load_annotations_for_conformal(args.db_path, args.prompt_id)
    logger.info(f"Loaded {len(df)} annotations")

    assert len(df) > 0, f"No annotations found for prompt_id: {args.prompt_id}"

    # Map LLM confidence (0-100) to score scale (1-5) to match human annotations
    logger.info("Mapping LLM confidence (0-100) to score scale (1-5)")
    df["LLM_confidence"] = map_llm_confidence_to_score(df["LLM_confidence"].values, bin_cutpoints=args.bin_range)

    logger.info(f"Number of unique encounters: {df['encounter_id'].nunique()}")
    logger.info(f"Number of unique reviewers: {df['reviewer_id'].nunique()}")

    # Fit conformal inference model
    logger.info("=" * 60)
    logger.info("FITTING TWO-LEVEL CONFORMAL INFERENCE MODEL")
    logger.info("=" * 60)

    ci = ConformalInference()
    fit_results = ci.fit(df, alpha=per_level_alpha)

    logger.info(f"Fitted on {fit_results['n_observations']} observations")

    logger.info(f"\nΔ₁ statistics (prediction set residuals):")
    logger.info(f"  Mean:   {fit_results['delta1_mean']:.2f}")
    logger.info(f"  Std:    {fit_results['delta1_std']:.2f}")
    logger.info(f"  Median: {fit_results['delta1_median']:.2f}")
    logger.info(f"  Min:    {fit_results['delta1_min']:.2f}")
    logger.info(f"  Max:    {fit_results['delta1_max']:.2f}")
    logger.info(f"  q₁:     {fit_results['q1']:.2f}")

    logger.info(f"\nSelected set (Δ₁ ≤ q₁):")
    logger.info(f"  N selected: {fit_results['n_selected']} / {fit_results['n_observations']}")
    logger.info(f"  Selection rate: {fit_results['selection_rate']:.1%}")

    if fit_results['delta2_mean'] is not None:
        logger.info(f"\nΔ₂ statistics (score bound residuals, from selected set):")
        logger.info(f"  Mean:   {fit_results['delta2_mean']:.2f}")
        logger.info(f"  Std:    {fit_results['delta2_std']:.2f}")
        logger.info(f"  Median: {fit_results['delta2_median']:.2f}")
        logger.info(f"  Min:    {fit_results['delta2_min']:.2f}")
        logger.info(f"  Max:    {fit_results['delta2_max']:.2f}")

    # Report quantiles at different alpha levels
    logger.info("=" * 60)
    logger.info("CONFORMAL QUANTILES")
    logger.info("=" * 60)

    for alpha_val in [0.05, 0.10, 0.15, 0.20]:
        q1 = ci.get_quantile_q1(alpha_val)
        q2 = ci.get_quantile_q2(alpha_val)
        joint = (1 - alpha_val) ** 2
        logger.info(f"  α={alpha_val:.2f} (joint {joint:.1%}): q₁={q1:.2f}, q₂={q2:.2f}")

    # Evaluate coverage
    logger.info("=" * 60)
    logger.info(f"COVERAGE EVALUATION (joint α={args.alpha}, per-level α={per_level_alpha:.4f})")
    logger.info("=" * 60)

    coverage_results = ci.evaluate_coverage(df, alpha=per_level_alpha)

    logger.info(f"\nLevel 1 (prediction set contains true best reason):")
    logger.info(f"  Target:    {coverage_results['level1_target']:.1%}")
    logger.info(f"  Empirical: {coverage_results['level1_empirical']:.1%}")
    logger.info(f"  Covered:   {coverage_results['level1_n_covered']} / {coverage_results['n_total']}")

    logger.info(f"\nLevel 2 (true score ≤ upper bound):")
    logger.info(f"  Target:    {coverage_results['level2_target']:.1%}")
    logger.info(f"  Empirical: {coverage_results['level2_empirical']:.1%}")
    logger.info(f"  Covered:   {coverage_results['level2_n_covered']} / {coverage_results['n_total']}")

    logger.info(f"\nJoint coverage (both levels satisfied):")
    logger.info(f"  Target:    {coverage_results['joint_target']:.1%}")
    logger.info(f"  Empirical: {coverage_results['joint_empirical']:.1%}")
    logger.info(f"  Covered:   {coverage_results['joint_n_covered']} / {coverage_results['n_total']}")

    # Show prediction set size distribution
    details = coverage_results["details"]
    logger.info(f"\nPrediction set sizes:")
    logger.info(f"  Mean:   {details['n_in_pred_set'].mean():.2f}")
    logger.info(f"  Median: {details['n_in_pred_set'].median():.1f}")
    logger.info(f"  Min:    {details['n_in_pred_set'].min()}")
    logger.info(f"  Max:    {details['n_in_pred_set'].max()}")

    # Show set size reduction compared to original LLM output
    logger.info(f"\nSet size reduction (prediction set vs full LLM output):")
    logger.info(f"  Original set sizes:")
    logger.info(f"    Mean:   {details['n_reasons'].mean():.2f}")
    logger.info(f"    Median: {details['n_reasons'].median():.1f}")
    set_size_ratio = details['n_in_pred_set'] / details['n_reasons']
    logger.info(f"  Prediction set as fraction of original:")
    logger.info(f"    Mean:   {set_size_ratio.mean():.1%}")
    logger.info(f"    Median: {set_size_ratio.median():.1%}")
    reduction = 1 - set_size_ratio
    logger.info(f"  Reduction in set size:")
    logger.info(f"    Mean:   {reduction.mean():.1%}")
    logger.info(f"    Median: {reduction.median():.1%}")

    # Max LLM assigned score in set
    print(details.keys())
    logger.info(f"\nScore upper bounds:")
    logger.info(f"  Mean:   {details['max_llm_in_set'].mean():.2f}")
    logger.info(f"  Median: {details['max_llm_in_set'].median():.2f}")
    logger.info(f"  Min:    {details['max_llm_in_set'].min():.2f}")
    logger.info(f"  Max:    {details['max_llm_in_set'].max():.2f}")
    logger.info(f"  Dist:    {details['max_llm_in_set'].value_counts()}")

    # Show score upper bound statistics
    logger.info(f"\nScore upper bounds:")
    logger.info(f"  Mean:   {details['score_upper_bound'].mean():.2f}")
    logger.info(f"  Median: {details['score_upper_bound'].median():.2f}")
    logger.info(f"  Min:    {details['score_upper_bound'].min():.2f}")
    logger.info(f"  Max:    {details['score_upper_bound'].max():.2f}")
    logger.info(f"  Dist:    {details['score_upper_bound'].value_counts()}")

    logger.info("=" * 60)
    logger.info("Analysis complete")

    # ======================================================================
    # BASELINE COMPARISON: Standard Conformal Inference
    # ======================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE: STANDARD CONFORMAL INFERENCE")
    logger.info("=" * 60)
    logger.info("Standard conformal treats each reason independently,")
    logger.info("ignoring the hierarchical structure of reason generation.")
    logger.info("")

    std_ci = StandardConformalInference()
    std_fit_results = std_ci.fit(df)

    logger.info(f"Fitted on {std_fit_results['n_calibration_points']} reasons")
    logger.info(f"  (from {std_fit_results['n_observations']} observations)")

    logger.info(f"\nΔ statistics (residuals = Y* - Ŷ for each reason):")
    logger.info(f"  Mean:   {std_fit_results['delta_mean']:.2f}")
    logger.info(f"  Std:    {std_fit_results['delta_std']:.2f}")
    logger.info(f"  Median: {std_fit_results['delta_median']:.2f}")
    logger.info(f"  Min:    {std_fit_results['delta_min']:.2f}")
    logger.info(f"  Max:    {std_fit_results['delta_max']:.2f}")

    # Report quantiles at different alpha levels
    logger.info(f"\nStandard conformal quantiles:")
    for alpha_val in [0.05, 0.10, 0.15, 0.20]:
        q = std_ci.get_quantile(alpha_val)
        logger.info(f"  α={alpha_val:.2f} ({1-alpha_val:.0%} coverage): q={q:.2f}")

    # Evaluate coverage
    logger.info(f"\nCoverage evaluation (α={args.alpha}):")
    std_coverage = std_ci.evaluate_coverage(df, alpha=args.alpha)

    logger.info(f"  Target:    {std_coverage['coverage_target']:.1%}")
    logger.info(f"  Empirical: {std_coverage['coverage_empirical']:.1%}")
    logger.info(f"  Covered:   {std_coverage['n_covered']} / {std_coverage['n_total']}")

    logger.info(f"\nUpper bound statistics:")
    logger.info(f"  Quantile (width added to LLM score): {std_coverage['quantile']:.2f}")
    logger.info(f"  Mean upper bound:   {std_coverage['mean_upper_bound']:.2f}")
    logger.info(f"  Median upper bound: {std_coverage['median_upper_bound']:.2f}")
    logger.info(f"  Min upper bound:    {std_coverage['min_upper_bound']:.2f}")
    logger.info(f"  Max upper bound:    {std_coverage['max_upper_bound']:.2f}")

    # ======================================================================
    # COMPARISON SUMMARY
    # ======================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON: TWO-LEVEL vs STANDARD CONFORMAL")
    logger.info("=" * 60)

    q2_twolevel = ci.get_quantile_q2(per_level_alpha)
    q_standard = std_ci.get_quantile(args.alpha)

    logger.info(f"\nMethod comparison (both targeting {1-args.alpha:.0%} overall coverage):")
    logger.info(f"                              Two-Level    Standard")
    logger.info(f"  Per-level/reason alpha:     {per_level_alpha:.4f}       {args.alpha:.2f}")
    logger.info(f"  Quantile for upper bound:   {q2_twolevel:.2f}          {q_standard:.2f}")

    # Compare upper bounds on the "best reason" per observation
    # For two-level: max LLM in prediction set + q2
    # For standard: just apply q to each reason's LLM score
    twolevel_details = coverage_results["details"]
    std_details = std_coverage["details"]

    logger.info(f"\nUpper bound on best reason's true score (per observation):")
    logger.info(f"  Two-level method:")
    logger.info(f"    Mean:   {twolevel_details['score_upper_bound'].mean():.2f}")
    logger.info(f"    Median: {twolevel_details['score_upper_bound'].median():.2f}")

    # For standard: compute upper bound on max LLM score per observation
    std_per_obs = (
        std_details.groupby("encounter_id")
        .agg({
            "LLM_confidence": "max",
            "upper_bound": "max",
        })
    )
    logger.info(f"  Standard method (upper bound on max-scoring reason):")
    logger.info(f"    Mean:   {std_per_obs['upper_bound'].mean():.2f}")
    logger.info(f"    Median: {std_per_obs['upper_bound'].median():.2f}")

    logger.info(f"\nKey insight:")
    logger.info(f"  Two-level uses q₂={q2_twolevel:.2f} (at α={per_level_alpha:.4f} from selected set)")
    logger.info(f"  Standard uses q={q_standard:.2f} (at α={args.alpha:.2f} from all reasons)")
    if q2_twolevel < q_standard:
        logger.info(f"  → Two-level produces tighter bounds by {q_standard - q2_twolevel:.2f}")
    else:
        logger.info(f"  → Standard produces tighter bounds by {q2_twolevel - q_standard:.2f}")

    logger.info("=" * 60)
    logger.info("Full analysis complete")


if __name__ == "__main__":
    main()
