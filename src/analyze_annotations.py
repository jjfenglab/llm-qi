#!/usr/bin/env python3
"""
Analyze annotations for a given prompt-id.

This script:
1. Runs Bayesian inference on annotations (directly, without UI)
2. Computes average interrater agreement for shared observations
3. Outputs descriptive statistics by LLM confidence bins
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bayesian_inference import (
    BayesianCalibration,
    BayesianMonotoneCalibration,
    BayesianMonotoneOrderedLogisticCalibration,
    BayesianNonparamCalibration,
    BayesianNonparamProbCalibration,
)


def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up logger to write to file and console."""
    logger = logging.getLogger("analyze_annotations")
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


def load_annotations(db_path: str, prompt_id: str) -> pd.DataFrame:
    """Load annotations from the database for a given prompt_id.

    Args:
        db_path: Path to the DuckDB database file
        prompt_id: Unique identifier for the prompt/experiment

    Returns:
        DataFrame with columns: encounter_id, reason_id, reason_text,
        LLM_confidence, annotation, reviewer_id
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


def bin_confidence_to_score(confidence: float, cutpoints: List[int]) -> int:
    """Map LLM confidence (0-100) to a 1-N score based on bins.

    Args:
        confidence: LLM confidence score (0-100)
        cutpoints: List of cutpoints defining bin boundaries (e.g., [60, 70, 80, 90])

    Returns:
        Mapped score (1 to len(cutpoints)+1)
    """
    for i, cutpoint in enumerate(cutpoints):
        if confidence < cutpoint:
            return i + 1
    return len(cutpoints) + 1


def _compute_interrater_stats_from_observations(shared_observations: List[Dict]) -> Dict:
    """Compute interrater agreement statistics from a list of shared observations.

    Args:
        shared_observations: List of dicts with 'annotations' key containing list of
            {'reviewer_id', 'annotation'} dicts

    Returns:
        Dict with exact_agreement_rate, within_1_agreement_rate, mean_absolute_difference
    """
    if not shared_observations:
        return {
            "exact_agreement_rate": 0.0,
            "within_1_agreement_rate": 0.0,
            "mean_absolute_difference": 0.0,
        }

    exact_matches = 0
    within_1_matches = 0
    total_pairs = 0
    all_differences = []

    for obs in shared_observations:
        annotations_list = obs["annotations"]
        for i in range(len(annotations_list)):
            for j in range(i + 1, len(annotations_list)):
                a1 = annotations_list[i]["annotation"]
                a2 = annotations_list[j]["annotation"]
                diff = abs(a1 - a2)
                all_differences.append(diff)
                total_pairs += 1
                if a1 == a2:
                    exact_matches += 1
                if diff <= 1:
                    within_1_matches += 1

    return {
        "exact_agreement_rate": exact_matches / total_pairs if total_pairs > 0 else 0.0,
        "within_1_agreement_rate": within_1_matches / total_pairs if total_pairs > 0 else 0.0,
        "mean_absolute_difference": float(np.mean(all_differences)) if all_differences else 0.0,
    }


def compute_interrater_agreement(
    df: pd.DataFrame, n_bootstrap: int = 1000, ci_level: float = 0.95
) -> Dict:
    """Compute interrater agreement for observations reviewed by multiple reviewers.

    An observation is identified by (encounter_id, reason_id). If multiple reviewers
    rated the same observation, we compute pairwise agreement.

    Args:
        df: DataFrame with columns encounter_id, reason_id, annotation, reviewer_id
        n_bootstrap: Number of bootstrap samples for confidence intervals
        ci_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dict with agreement statistics and bootstrap confidence intervals
    """
    # Group by (encounter_id, reason_id) and find observations with multiple reviewers
    grouped = df.groupby(["encounter_id", "reason_id"])

    shared_observations = []
    for (enc_id, reason_id), group in grouped:
        if len(group) > 1:
            reviewers = group["reviewer_id"].unique()
            if len(reviewers) > 1:
                shared_observations.append(
                    {
                        "encounter_id": enc_id,
                        "reason_id": reason_id,
                        "annotations": group[["reviewer_id", "annotation"]].to_dict(
                            "records"
                        ),
                        "n_reviewers": len(reviewers),
                    }
                )

    if not shared_observations:
        return {
            "n_shared_observations": 0,
            "message": "No shared observations found (same encounter_id and reason_id with different reviewers)",
        }

    # Compute point estimates
    point_estimates = _compute_interrater_stats_from_observations(shared_observations)

    # Compute total pairwise comparisons for reporting
    total_pairs = 0
    for obs in shared_observations:
        n = len(obs["annotations"])
        total_pairs += n * (n - 1) // 2

    # Bootstrap at reason-level
    n_obs = len(shared_observations)
    bootstrap_stats = {
        "exact_agreement_rate": [],
        "within_1_agreement_rate": [],
        "mean_absolute_difference": [],
    }

    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        # Resample observations (reason-level)
        indices = rng.choice(n_obs, size=n_obs, replace=True)
        resampled_obs = [shared_observations[i] for i in indices]
        boot_stats = _compute_interrater_stats_from_observations(resampled_obs)
        for key in bootstrap_stats:
            bootstrap_stats[key].append(boot_stats[key])

    # Compute confidence intervals
    alpha = 1 - ci_level
    ci_lower_pct = alpha / 2 * 100
    ci_upper_pct = (1 - alpha / 2) * 100

    results = {
        "n_shared_observations": len(shared_observations),
        "n_pairwise_comparisons": total_pairs,
    }

    for key in bootstrap_stats:
        samples = np.array(bootstrap_stats[key])
        results[key] = point_estimates[key]
        results[f"{key}_ci_lower"] = float(np.percentile(samples, ci_lower_pct))
        results[f"{key}_ci_upper"] = float(np.percentile(samples, ci_upper_pct))

    return results


def _compute_llm_human_stats(llm_mapped_scores: np.ndarray, human_scores: np.ndarray) -> Dict:
    """Compute LLM-human agreement statistics from score arrays.

    Args:
        llm_mapped_scores: Array of binned LLM confidence scores
        human_scores: Array of human annotation scores

    Returns:
        Dict with exact_agreement_rate, within_1_agreement_rate, mean_absolute_difference
    """
    if len(llm_mapped_scores) == 0:
        return {
            "exact_agreement_rate": 0.0,
            "within_1_agreement_rate": 0.0,
            "mean_absolute_difference": 0.0,
        }

    differences = np.abs(llm_mapped_scores - human_scores)
    exact_matches = np.sum(differences == 0)
    within_1_matches = np.sum(differences <= 1)
    total = len(differences)

    return {
        "exact_agreement_rate": exact_matches / total if total > 0 else 0.0,
        "within_1_agreement_rate": within_1_matches / total if total > 0 else 0.0,
        "mean_absolute_difference": float(np.mean(differences)),
    }


def compute_llm_human_agreement(
    df: pd.DataFrame, cutpoints: List[int], n_bootstrap: int = 1000, ci_level: float = 0.95
) -> Dict:
    """Compute agreement between binned LLM confidence scores and human annotations.

    LLM confidence is binned and mapped to scores based on provided cutpoints.

    Args:
        df: DataFrame with columns LLM_confidence, annotation, encounter_id, reason_id
        cutpoints: List of cutpoints defining bin boundaries (e.g., [60, 70, 80, 90])
        n_bootstrap: Number of bootstrap samples for confidence intervals
        ci_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dict with agreement statistics and bootstrap confidence intervals
    """
    # Filter for valid data
    df_valid = df[df["LLM_confidence"].notna() & df["annotation"].notna()].copy()

    if len(df_valid) == 0:
        return {
            "n_comparisons": 0,
            "message": "No valid observations with both LLM confidence and human annotation",
        }

    # Map LLM confidence to scores based on cutpoints
    df_valid["llm_mapped_score"] = df_valid["LLM_confidence"].apply(
        lambda x: bin_confidence_to_score(x, cutpoints)
    )

    llm_mapped_scores = df_valid["llm_mapped_score"].to_numpy()
    human_scores = df_valid["annotation"].to_numpy().astype(int)

    # Compute point estimates
    point_estimates = _compute_llm_human_stats(llm_mapped_scores, human_scores)

    # Get unique reason-level identifiers for bootstrap resampling
    reason_ids = df_valid.groupby(["encounter_id", "reason_id"]).ngroup().to_numpy()
    unique_reason_ids = np.unique(reason_ids)
    n_reasons = len(unique_reason_ids)

    # Bootstrap at reason-level
    bootstrap_stats = {
        "exact_agreement_rate": [],
        "within_1_agreement_rate": [],
        "mean_absolute_difference": [],
    }

    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        # Resample reason IDs
        resampled_reason_ids = rng.choice(unique_reason_ids, size=n_reasons, replace=True)

        # Get indices for resampled reasons
        resampled_indices = []
        for rid in resampled_reason_ids:
            resampled_indices.extend(np.where(reason_ids == rid)[0].tolist())

        boot_llm = llm_mapped_scores[resampled_indices]
        boot_human = human_scores[resampled_indices]
        boot_stats = _compute_llm_human_stats(boot_llm, boot_human)

        for key in bootstrap_stats:
            bootstrap_stats[key].append(boot_stats[key])

    # Compute confidence intervals
    alpha = 1 - ci_level
    ci_lower_pct = alpha / 2 * 100
    ci_upper_pct = (1 - alpha / 2) * 100

    results = {
        "n_comparisons": len(df_valid),
        "n_reasons": n_reasons,
    }

    for key in bootstrap_stats:
        samples = np.array(bootstrap_stats[key])
        results[key] = point_estimates[key]
        results[f"{key}_ci_lower"] = float(np.percentile(samples, ci_lower_pct))
        results[f"{key}_ci_upper"] = float(np.percentile(samples, ci_upper_pct))

    return results


def compute_descriptive_stats_by_confidence_bin(df: pd.DataFrame, cutpoints: List[int]) -> pd.DataFrame:
    """Compute descriptive statistics of human scores by LLM confidence bins.

    Args:
        df: DataFrame with columns LLM_confidence and annotation
        cutpoints: List of cutpoints defining bin boundaries (e.g., [60, 70, 80, 90])

    Returns:
        DataFrame with columns: bin, n, mean, std, median, ci_lower, ci_upper
    """
    # Filter out rows with missing confidence
    df_valid = df[df["LLM_confidence"].notna()].copy()

    # Create bins from cutpoints: [0, cutpoint1, cutpoint2, ..., 101]
    bins = [0] + cutpoints + [101]
    # Create labels: "0-{c1-1}", "{c1}-{c2-1}", ..., "{cn}-100"
    bin_labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            bin_labels.append(f"{bins[i]}-100")
        else:
            bin_labels.append(f"{bins[i]}-{bins[i+1]-1}")

    df_valid["confidence_bin"] = pd.cut(
        df_valid["LLM_confidence"],
        bins=bins,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )

    # Group by bin and compute statistics
    bin_stats = (
        df_valid.groupby("confidence_bin", observed=True)["annotation"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    bin_stats.columns = ["bin", "n", "mean", "std", "median"]

    # Compute 95% confidence intervals for the mean using t-distribution
    ci_lower = []
    ci_upper = []
    for _, row in bin_stats.iterrows():
        n = row["n"]
        mean = row["mean"]
        std = row["std"]
        if n > 1 and pd.notna(std):
            sem = std / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_lower.append(mean - t_crit * sem)
            ci_upper.append(mean + t_crit * sem)
        else:
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)

    bin_stats["ci_lower"] = ci_lower
    bin_stats["ci_upper"] = ci_upper

    return bin_stats


def plot_confidence_intervals_by_bin(
    bin_stats: pd.DataFrame,
    output_file: str,
    logger: logging.Logger,
) -> None:
    """Plot mean annotated scores with 95% confidence intervals by LLM-assigned bin.

    Args:
        bin_stats: DataFrame with columns bin, n, mean, std, ci_lower, ci_upper
        output_file: Path to save the plot
        logger: Logger instance
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    x_positions = np.arange(len(bin_stats))
    means = bin_stats["mean"].values
    ci_lower = bin_stats["ci_lower"].values
    ci_upper = bin_stats["ci_upper"].values

    # Compute error bars (distance from mean to CI bounds)
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means
    yerr = np.array([yerr_lower, yerr_upper])

    # Plot with error bars
    ax.errorbar(
        x_positions,
        means,
        yerr=yerr,
        fmt="o",
        markersize=10,
        capsize=6,
        capthick=2,
        elinewidth=2,
        color="steelblue",
        ecolor="steelblue",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_stats["bin"].values, fontsize=14)
    ax.set_xlabel("LLM Confidence Bin", fontsize=16)
    ax.set_ylabel("Mean Human Annotation", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(1, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Despine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confidence interval plot to: {output_file}")


def plot_boxplot_by_bin(
    df: pd.DataFrame,
    cutpoints: List[int],
    output_file: str,
    logger: logging.Logger,
) -> None:
    """Plot boxplot of annotated scores by LLM-assigned confidence bin.

    Args:
        df: DataFrame with columns LLM_confidence and annotation
        cutpoints: List of cutpoints defining bin boundaries
        output_file: Path to save the plot
        logger: Logger instance
    """
    # Filter out rows with missing values
    df_valid = df[df["LLM_confidence"].notna() & df["annotation"].notna()].copy()

    # Create bins from cutpoints
    bins = [0] + cutpoints + [101]
    bin_labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            bin_labels.append(f"{bins[i]}-100")
        else:
            bin_labels.append(f"{bins[i]}-{bins[i+1]-1}")

    df_valid["confidence_bin"] = pd.cut(
        df_valid["LLM_confidence"],
        bins=bins,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    # Group data by bin for boxplot
    grouped_data = [
        group["annotation"].values
        for _, group in df_valid.groupby("confidence_bin", observed=True)
    ]
    bin_labels_present = [
        name for name, _ in df_valid.groupby("confidence_bin", observed=True)
    ]

    # Create boxplot
    bp = ax.boxplot(
        grouped_data,
        labels=bin_labels_present,
        patch_artist=True,
    )

    # Style the boxplot
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    # Add sample size annotations at fixed y position near top
    for i, (label, data) in enumerate(zip(bin_labels_present, grouped_data)):
        ax.annotate(
            f"n={len(data)}",
            (i + 1, 4.8),
            ha="center",
            fontsize=12,
            color="gray",
        )

    ax.set_xlabel("LLM Confidence Bin", fontsize=16)
    ax.set_ylabel("Human Annotation", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(1, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Despine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved boxplot to: {output_file}")


def run_bayesian_inference(
    df: pd.DataFrame, calibration_model: str, logger: logging.Logger
) -> Dict:
    """Run Bayesian inference on annotations.

    Args:
        df: DataFrame with columns LLM_confidence and annotation
        calibration_model: Which calibration model to use
        logger: Logger instance

    Returns:
        Dict with inference results
    """
    # Filter for valid data
    df_valid = df[df["LLM_confidence"].notna() & df["annotation"].notna()].copy()

    if len(df_valid) < 2:
        return {
            "status": "insufficient_data",
            "n_observations": len(df_valid),
            "message": "Need at least 2 annotations for inference",
        }

    llm_scores = df_valid["LLM_confidence"].to_numpy().astype(float)
    human_scores = df_valid["annotation"].to_numpy().astype(float)

    # Initialize the appropriate calibration model
    if calibration_model == "nonparam":
        calibration = BayesianNonparamCalibration("src/bayesian_delta_model.stan")
        logger.info("Using nonparametric delta calibration model (discrete 0-3)")
    elif calibration_model == "nonparam_prob":
        calibration = BayesianNonparamProbCalibration("src/bayesian_delta_cts_model.stan")
        logger.info(
            "Using nonparametric piecewise constant calibration model (0-100)"
        )
    elif calibration_model == "monotone":
        calibration = BayesianMonotoneCalibration("src/bayesian_monotone_model.stan")
        logger.info(
            "Using monotone calibration model (0-100, enforces monotonicity)"
        )
    elif calibration_model == "monotone_ordinal":
        calibration = BayesianMonotoneOrderedLogisticCalibration(
            "src/bayesian_monotone_ordered_logistic.stan"
        )
        logger.info(
            "Using monotone ordered logistic calibration model (0-100 → ordinal 1-5)"
        )
    else:  # default to linear
        calibration = BayesianCalibration("src/bayesian_linear_regression.stan")
        logger.info("Using linear regression calibration model")

    # Compile and fit
    calibration.compile_model()
    logger.info("Bayesian calibration model compiled successfully")

    logger.info(f"Running MCMC with {len(df_valid)} observations")
    results = calibration.fit(
        confidences=llm_scores,
        annotations=human_scores,
        chains=4,
        iter_sampling=1000,
        iter_warmup=1000,
        show_progress=True,
    )

    if results["status"] == "success":
        # Get expected scores at each level
        expected_scores = calibration.get_expected_scores_at_levels()
        results["expected_scores"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in expected_scores.items()
        }

    return results


def _log_agreement_results(results: Dict, logger: logging.Logger) -> None:
    """Log agreement results with confidence intervals formatted nicely.

    Args:
        results: Dict with agreement statistics (point estimates and CI bounds)
        logger: Logger instance
    """
    # Metrics that have CI bounds
    ci_metrics = ["exact_agreement_rate", "within_1_agreement_rate", "mean_absolute_difference"]

    for key, value in results.items():
        # Skip CI bound keys (they'll be logged with their main metric)
        if key.endswith("_ci_lower") or key.endswith("_ci_upper"):
            continue

        if key in ci_metrics:
            ci_lower = results.get(f"{key}_ci_lower")
            ci_upper = results.get(f"{key}_ci_upper")
            if ci_lower is not None and ci_upper is not None:
                logger.info(f"  {key}: {value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            else:
                logger.info(f"  {key}: {value:.3f}")
        elif isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        else:
            logger.info(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze annotations for a given prompt-id"
    )
    parser.add_argument("--prompt-id", required=True, help="Unique identifier for the prompt/experiment")
    parser.add_argument("--db-path", required=True, help="Path to the DuckDB database file")
    parser.add_argument(
        "--calibration-model",
        choices=["linear", "nonparam", "nonparam_prob", "monotone", "monotone_ordinal"],
        default="monotone_ordinal",
        help="Calibration model to use (default: monotone_ordinal)",
    )
    parser.add_argument(
        "--bin-range",
        nargs="+",
        type=int,
        default=[60, 70, 80, 90],
        help="Cutpoints for binning LLM confidence (default: 60 70 80 90)",
    )
    parser.add_argument("--log-file", help="Path to log file (optional)")
    parser.add_argument(
        "--output-plot-file",
        help="Path to save the confidence interval plot (optional)",
    )
    parser.add_argument(
        "--output-boxplot-file",
        help="Path to save the boxplot of annotation distributions (optional)",
    )

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    logger.info(f"Analyzing annotations for prompt_id: {args.prompt_id}")
    logger.info(f"Database path: {args.db_path}")

    # Load annotations
    df = load_annotations(args.db_path, args.prompt_id)
    logger.info(f"Loaded {len(df)} annotations")

    assert len(df) > 0, f"No annotations found for prompt_id: {args.prompt_id}"

    # Print basic summary
    logger.info(f"Number of unique encounters: {df['encounter_id'].nunique()}")
    logger.info(f"Number of unique reviewers: {df['reviewer_id'].nunique()}")
    logger.info(f"Reviewers: {df['reviewer_id'].unique().tolist()}")

    # Compute interrater agreement (human-human)
    logger.info("=" * 60)
    logger.info("HUMAN-HUMAN INTERRATER AGREEMENT (bootstrap 95% CI at reason-level)")
    logger.info("=" * 60)
    agreement = compute_interrater_agreement(df)
    _log_agreement_results(agreement, logger)

    # Compute LLM-human agreement
    logger.info("=" * 60)
    cutpoints = args.bin_range
    n_bins = len(cutpoints) + 1
    logger.info(f"LLM-HUMAN AGREEMENT (bootstrap 95% CI at reason-level, binned to 1-{n_bins})")
    # Build mapping string from cutpoints
    mapping_parts = [f"0-{cutpoints[0]-1}->1"]
    for i in range(len(cutpoints) - 1):
        mapping_parts.append(f"{cutpoints[i]}-{cutpoints[i+1]-1}->{i+2}")
    mapping_parts.append(f"{cutpoints[-1]}-100->{n_bins}")
    logger.info(f"  Mapping: {', '.join(mapping_parts)}")
    logger.info("=" * 60)
    llm_human_agreement = compute_llm_human_agreement(df, cutpoints)
    _log_agreement_results(llm_human_agreement, logger)

    # Compute descriptive statistics by confidence bin
    logger.info("=" * 60)
    logger.info("DESCRIPTIVE STATISTICS BY LLM CONFIDENCE BIN")
    logger.info("=" * 60)
    bin_stats = compute_descriptive_stats_by_confidence_bin(df, cutpoints)
    logger.info("\n" + bin_stats.to_string(index=False))

    # Generate confidence interval plot if output file specified
    if args.output_plot_file:
        plot_confidence_intervals_by_bin(bin_stats, args.output_plot_file, logger)

    # Generate boxplot if output file specified
    if args.output_boxplot_file:
        plot_boxplot_by_bin(df, cutpoints, args.output_boxplot_file, logger)

    # Run Bayesian inference
    #logger.info("=" * 60)
    #logger.info(f"BAYESIAN INFERENCE (model: {args.calibration_model})")
    #logger.info("=" * 60)
    #inference_results = run_bayesian_inference(df, args.calibration_model, logger)
    #if inference_results["status"] == "success":
    #    logger.info(f"Inference successful with {inference_results['n_observations']} observations")
    #    logger.info(f"Number of posterior samples: {inference_results['n_samples']}")

    #    logger.info("\nSummary statistics:")
    #    for key, value in inference_results["summary"].items():
    #        logger.info(f"  {key}: {value:.3f}")

    #    logger.info("\nExpected human scores by LLM confidence level:")
    #    for key, value in inference_results["expected_scores"].items():
    #        logger.info(f"  {key}: mean={value['mean']:.3f}, std={value['std']:.3f}")
    #else:
    #    logger.warning(f"Inference failed: {inference_results.get('message', 'Unknown error')}")

    logger.info("=" * 60)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
