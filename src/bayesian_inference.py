#!/usr/bin/env python3
"""
Core Bayesian inference functionality for Human-AI Co-discovery calibration.
Separated from the Flask API for direct use in notebooks and testing.

Uses Bayesian linear regression to model the relationship between
LLM confidence scores (0-3 Likert) and human annotation scores (0-3 Likert).
"""

import os
import numpy as np
import pandas as pd
import cmdstanpy
from typing import List, Dict, Any, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_annotations_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and filter annotation data from CSV file.

    Applies standard filtering:
    - Only includes rows where feedback_type == "reason" (excludes gantt charts)
    - Excludes rows with missing LLM_confidence or annotation
    - Only includes valid Likert scale annotations (0-3)

    Args:
        csv_path: Path to annotations CSV file

    Returns:
        Tuple of (llm_scores, human_scores) as numpy arrays
    """
    annot_df = pd.read_csv(csv_path)

    # Filter for reason feedback type only (not gantt charts)
    annot_df = annot_df[annot_df.feedback_type == "reason"]

    # Filter for valid annotations (0-3 Likert scale)
    annot_df = annot_df[annot_df.annotation.isin([0, 1, 2, 3])]

    # Filter for valid confidence values
    annot_df = annot_df[annot_df.LLM_confidence.notna()]

    # Convert to numpy arrays with proper types
    llm_scores = annot_df.LLM_confidence.to_numpy().astype(float)
    human_scores = annot_df.annotation.to_numpy().astype(float)

    logger.info(f"Loaded {len(llm_scores)} reason annotations from {csv_path}")

    return llm_scores, human_scores


def preprocess_data_for_stan(llm_scores: np.ndarray, human_scores: np.ndarray) -> Dict[str, Any]:
    """
    Preprocess LLM and human scores for Stan linear regression model.

    Args:
        llm_scores: Array of LLM confidence scores
        human_scores: Array of human annotation scores

    Returns:
        Dict with processed data ready for Stan model
    """
    assert len(llm_scores) == len(human_scores), "LLM scores and human scores must have same length"

    return {
        'N': len(llm_scores),
        'x': llm_scores.tolist(), 
        'y': [int(h) for h in human_scores.tolist()],
    }


def compile_stan_model(stan_model_path: str = "src/bayesian_linear_regression.stan") -> cmdstanpy.CmdStanModel:
    """
    Compile the Stan model.

    Args:
        stan_model_path: Path to the Stan model file

    Returns:
        Compiled CmdStanModel object
    """
    try:
        model = cmdstanpy.CmdStanModel(stan_file=stan_model_path)
        logger.info("Stan model compiled successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to compile Stan model: {e}")
        raise


def run_mcmc_inference(
    stan_model: cmdstanpy.CmdStanModel,
    llm_scores: np.ndarray,
    human_scores: np.ndarray,
    chains: int = 4,
    iter_sampling: int = 1000,
    iter_warmup: int = 1000,
    show_progress: bool = False
) -> Dict[str, Any]:
    """
    Run MCMC inference on LLM-human score data using linear regression.

    Args:
        stan_model: Compiled Stan model
        llm_scores: Array of LLM confidence scores (0-3 Likert)
        human_scores: Array of human annotation scores (0-3 Likert)
        chains: Number of MCMC chains
        iter_sampling: Number of sampling iterations per chain
        iter_warmup: Number of warmup iterations per chain
        show_progress: Whether to show sampling progress

    Returns:
        Dict containing posterior samples and summary statistics
    """
    if len(llm_scores) < 2:
        return {
            "status": "insufficient_data",
            "n_observations": len(llm_scores),
            "message": "Need at least 2 data points for inference"
        }

    # Preprocess data
    stan_data = preprocess_data_for_stan(llm_scores, human_scores)

    # Run MCMC sampling
    logger.info(f"Running MCMC with {len(llm_scores)} observations")
    fit = stan_model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        show_progress=show_progress
    )

    # Extract posterior samples
    beta_0_samples = fit.stan_variable("beta_0")
    beta_1_samples = fit.stan_variable("beta_1")
    sigma_samples = fit.stan_variable("sigma")

    # Calculate summary statistics
    beta_0_mean = float(np.mean(beta_0_samples))
    beta_1_mean = float(np.mean(beta_1_samples))
    beta_0_std = float(np.std(beta_0_samples))
    beta_1_std = float(np.std(beta_1_samples))
    sigma_mean = float(np.mean(sigma_samples))
    sigma_std = float(np.std(sigma_samples))

    return {
        "status": "success",
        "n_observations": len(llm_scores),
        "n_samples": len(beta_0_samples),
        "posterior_samples": {
            "beta_0": beta_0_samples,
            "beta_1": beta_1_samples,
            "sigma": sigma_samples
        },
        "summary": {
            "beta_0_mean": beta_0_mean,
            "beta_0_std": beta_0_std,
            "beta_1_mean": beta_1_mean,
            "beta_1_std": beta_1_std,
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std
        },
        "fit_object": fit  # For advanced diagnostics
    }


def generate_regression_lines(
    beta_0_samples: np.ndarray,
    beta_1_samples: np.ndarray,
    x_grid: Optional[np.ndarray] = None,
    max_lines: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate regression lines from posterior samples.

    Args:
        beta_0_samples: Posterior samples for intercept parameter
        beta_1_samples: Posterior samples for slope parameter
        x_grid: X values (LLM confidence scores 0-100) to evaluate lines at
        max_lines: Maximum number of lines to return

    Returns:
        List of regression lines with line points
    """
    if x_grid is None:
        x_grid = np.array([0, 25, 50, 75, 100])  # LLM confidence scale points

    regression_lines = []
    n_lines = min(len(beta_0_samples), max_lines)

    for i in range(n_lines):
        b0, b1 = beta_0_samples[i], beta_1_samples[i]
        line_points = []

        for x in x_grid:
            predicted_y = b0 + b1 * x
            line_points.append({
                "x": float(x),
                "predicted_y": float(predicted_y)
            })

        regression_lines.append({
            "sample_id": i,
            "beta_0": float(b0),
            "beta_1": float(b1),
            "line_points": line_points
        })

    return regression_lines


def compute_regression_statistics(
    llm_scores: np.ndarray,
    human_scores: np.ndarray,
    beta_0_mean: float,
    beta_1_mean: float
) -> Dict[str, float]:
    """
    Compute regression statistics for the fitted model.

    Args:
        llm_scores: Array of LLM confidence scores (0-100)
        human_scores: Array of human annotation scores (1-3)
        beta_0_mean: Posterior mean of intercept
        beta_1_mean: Posterior mean of slope

    Returns:
        Dict with regression statistics
    """
    # Predicted values using posterior means
    predicted_scores = beta_0_mean + beta_1_mean * llm_scores

    # Mean squared error
    mse = np.mean((predicted_scores - human_scores) ** 2)

    # R-squared
    ss_res = np.sum((human_scores - predicted_scores) ** 2)
    ss_tot = np.sum((human_scores - np.mean(human_scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean absolute error
    mae = np.mean(np.abs(predicted_scores - human_scores))

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r_squared": float(r_squared),
        "mean_llm_score": float(np.mean(llm_scores)),
        "mean_human_score": float(np.mean(human_scores)),
        "n_observations": len(llm_scores)
    }


class BayesianCalibration:
    """
    Main class for Bayesian linear regression calibration.
    Models the relationship between LLM scores and human scores (0-3).
    """

    def __init__(self, stan_model_path: str = "src/bayesian_linear_regression.stan"):
        """
        Initialize the Bayesian calibration system.

        Args:
            stan_model_path: Path to the Stan model file
        """
        self.stan_model_path = stan_model_path
        self.stan_model = None
        self.last_inference_results = None

    def compile_model(self):
        """Compile the Stan model."""
        self.stan_model = compile_stan_model(self.stan_model_path)

    def fit(
        self,
        confidences: np.ndarray,
        annotations: np.ndarray,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian linear regression model to data.

        Args:
            confidences: Array of LLM confidence scores (0-3 Likert)
            annotations: Array of human annotation scores (0-3 Likert)
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        if self.stan_model is None:
            self.compile_model()

        results = run_mcmc_inference(
            self.stan_model,
            confidences,
            annotations,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

        self.last_inference_results = results
        return results

    def fit_from_csv(
        self,
        csv_path: str,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Load annotation data from CSV, filter for reasons only, and fit the model.

        Args:
            csv_path: Path to annotations CSV file
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        llm_scores, human_scores = load_annotations_from_csv(csv_path)

        return self.fit(
            llm_scores,
            human_scores,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

    def compute_statistics(
        self,
        llm_scores: np.ndarray,
        human_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression statistics using the fitted model.

        Args:
            llm_scores: Array of LLM confidence scores (0-3)
            human_scores: Array of human annotation scores (0-3)

        Returns:
            Dict with regression statistics
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        summary = self.last_inference_results["summary"]
        return compute_regression_statistics(
            llm_scores,
            human_scores,
            summary["beta_0_mean"],
            summary["beta_1_mean"]
        )

    def plot_calibration_curves(
        self,
        figsize: Tuple[float, float] = (10, 8),
        title: str = "Bayesian Calibration: LLM vs Human Scores"
    ):
        """
        Create a violinplot showing posterior distributions of expected human scores
        at each LLM confidence level (0-100 scale).

        Args:
            figsize: Figure size tuple (width, height)
            title: Plot title

        Returns:
            matplotlib.axes.Axes: The plot axes object
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        expected_scores = self.get_expected_scores_at_levels()

        # Prepare data for violinplot: list of arrays, one per LLM confidence level
        llm_levels = [0, 25, 50, 75, 100]
        data = [expected_scores[str(x)] for x in llm_levels]

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Create violinplot
        parts = ax.violinplot(data, positions=llm_levels, showmeans=True, showmedians=False, widths=15)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Plot perfect calibration line: LLM 0→Human 1, LLM 100→Human 3
        plt.plot([0, 100], [1, 3], color='black', linestyle='--', linewidth=2,
                label='Perfect Calibration', zorder=5)

        # Plot posterior means as points
        means = [np.mean(expected_scores[str(x)]) for x in llm_levels]
        plt.scatter(llm_levels, means, color='red', s=80, zorder=10, label='Posterior Mean')

        # Customize plot
        plt.xlabel('LLM Confidence Score (0-100)', fontsize=12)
        plt.ylabel('Expected Human Score (1-3)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)
        plt.xticks([0, 25, 50, 75, 100])
        plt.xlim(-10, 110)
        plt.ylim(0.5, 3.5)

        plt.tight_layout()
        return ax

    def get_expected_scores_at_levels(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples of expected human scores at each LLM confidence level (0-100 scale).

        Returns:
            Dict with keys "100", "75", "50", "25", "0" mapping to arrays of posterior samples
            for the expected human score at each LLM confidence level.
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        samples = self.last_inference_results["posterior_samples"]
        beta_0 = samples["beta_0"]
        beta_1 = samples["beta_1"]

        # For linear regression: E[y|x] = beta_0 + beta_1 * x
        # LLM confidence is on 0-100 scale
        return {
            "100": beta_0 + beta_1 * 100,
            "75": beta_0 + beta_1 * 75,
            "50": beta_0 + beta_1 * 50,
            "25": beta_0 + beta_1 * 25,
            "0": beta_0 + beta_1 * 0,  # = beta_0
        }


class BayesianNonparamCalibration(BayesianCalibration):
    """
    Bayesian calibration using a nonparametric delta model for discrete LLM scores (0-3).

    Instead of assuming a linear relationship, this model estimates separate
    expected values for each LLM rating level using a cumulative delta parameterization:
    - E[y|x=3] = beta_3
    - E[y|x=2] = beta_3 + delta_2
    - E[y|x=1] = beta_3 + delta_2 + delta_1
    - E[y|x=0] = beta_3 + delta_2 + delta_1 + delta_0
    """

    def __init__(self, stan_model_path: str = "src/bayesian_delta_model.stan"):
        """
        Initialize the Bayesian nonparametric calibration system.

        Args:
            stan_model_path: Path to the Stan delta model file
        """
        super().__init__(stan_model_path)

    def fit(
        self,
        confidences: np.ndarray,
        annotations: np.ndarray,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian delta model to data.

        Args:
            confidences: Array of LLM confidence scores (0-3 Likert)
            annotations: Array of human annotation scores (0-3 Likert)
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        if self.stan_model is None:
            self.compile_model()

        if len(confidences) < 2:
            self.last_inference_results = {
                "status": "insufficient_data",
                "n_observations": len(confidences),
                "message": "Need at least 2 data points for inference"
            }
            return self.last_inference_results

        # Preprocess data (same format as linear model)
        stan_data = preprocess_data_for_stan(confidences, annotations)

        # Run MCMC sampling
        logger.info(f"Running MCMC (delta model) with {len(confidences)} observations")
        fit = self.stan_model.sample(
            data=stan_data,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

        # Extract posterior samples for delta model parameters
        beta_3_samples = fit.stan_variable("beta_3")
        delta_2_samples = fit.stan_variable("delta_2")
        delta_1_samples = fit.stan_variable("delta_1")
        delta_0_samples = fit.stan_variable("delta_0")
        sigma_samples = fit.stan_variable("sigma")

        # Calculate summary statistics
        self.last_inference_results = {
            "status": "success",
            "n_observations": len(confidences),
            "n_samples": len(beta_3_samples),
            "posterior_samples": {
                "beta_3": beta_3_samples,
                "delta_2": delta_2_samples,
                "delta_1": delta_1_samples,
                "delta_0": delta_0_samples,
                "sigma": sigma_samples
            },
            "summary": {
                "beta_3_mean": float(np.mean(beta_3_samples)),
                "beta_3_std": float(np.std(beta_3_samples)),
                "delta_2_mean": float(np.mean(delta_2_samples)),
                "delta_2_std": float(np.std(delta_2_samples)),
                "delta_1_mean": float(np.mean(delta_1_samples)),
                "delta_1_std": float(np.std(delta_1_samples)),
                "delta_0_mean": float(np.mean(delta_0_samples)),
                "delta_0_std": float(np.std(delta_0_samples)),
                "sigma_mean": float(np.mean(sigma_samples)),
                "sigma_std": float(np.std(sigma_samples))
            },
            "fit_object": fit
        }

        return self.last_inference_results

    def get_expected_scores_at_levels(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples of expected human scores at each LLM rating level (3, 2, 1, 0).

        Returns:
            Dict with keys "3", "2", "1", "0" mapping to arrays of posterior samples
            for the expected human score at each LLM rating level.
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        samples = self.last_inference_results["posterior_samples"]
        beta_3 = samples["beta_3"]
        delta_2 = samples["delta_2"]
        delta_1 = samples["delta_1"]
        delta_0 = samples["delta_0"]

        # For delta model:
        # E[y|x=3] = beta_3
        # E[y|x=2] = beta_3 + delta_2
        # E[y|x=1] = beta_3 + delta_2 + delta_1
        # E[y|x=0] = beta_3 + delta_2 + delta_1 + delta_0
        return {
            "3": beta_3,
            "2": beta_3 + delta_2,
            "1": beta_3 + delta_2 + delta_1,
            "0": beta_3 + delta_2 + delta_1 + delta_0,
        }


class BayesianNonparamProbCalibration(BayesianCalibration):
    """
    Bayesian calibration using a nonparametric piecewise constant delta model
    for continuous LLM confidence scores (0-100 scale).

    This model estimates separate expected values for 6 LLM confidence bins:
    - E[y|x > 90] = beta_100
    - E[y|80 < x <= 90] = beta_100 + deltas[5]
    - E[y|70 < x <= 80] = beta_100 + deltas[5] + deltas[4]
    - E[y|60 < x <= 70] = beta_100 + deltas[5] + deltas[4] + deltas[3]
    - E[y|50 < x <= 60] = beta_100 + deltas[5] + deltas[4] + deltas[3] + deltas[2]
    - E[y|x <= 50] = beta_100 + deltas[5] + deltas[4] + deltas[3] + deltas[2] + deltas[1]
    """

    # Bin labels for display (bin center or range description)
    BIN_LABELS = ["≤50", "51-60", "61-70", "71-80", "81-90", ">90"]
    BIN_KEYS = ["le50", "51_60", "61_70", "71_80", "81_90", "gt90"]

    def __init__(self, stan_model_path: str = "src/bayesian_delta_cts_model.stan"):
        """
        Initialize the Bayesian nonparametric probability calibration system.

        Args:
            stan_model_path: Path to the Stan delta model file
        """
        super().__init__(stan_model_path)

    def fit(
        self,
        confidences: np.ndarray,
        annotations: np.ndarray,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian piecewise constant delta model to data.

        Args:
            confidences: Array of LLM confidence scores (0-100 scale)
            annotations: Array of human annotation scores (1-3 Likert)
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        if self.stan_model is None:
            self.compile_model()

        if len(confidences) < 2:
            self.last_inference_results = {
                "status": "insufficient_data",
                "n_observations": len(confidences),
                "message": "Need at least 2 data points for inference"
            }
            return self.last_inference_results

        # Preprocess data (same format as linear model)
        stan_data = preprocess_data_for_stan(confidences, annotations)

        # Run MCMC sampling
        logger.info(f"Running MCMC (piecewise constant delta model) with {len(confidences)} observations")
        fit = self.stan_model.sample(
            data=stan_data,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

        # Extract posterior samples for delta model parameters
        beta_100_samples = fit.stan_variable("beta_100")
        deltas_samples = fit.stan_variable("deltas")  # Shape: (n_samples, 5)
        sigma_samples = fit.stan_variable("sigma")

        # Calculate summary statistics
        self.last_inference_results = {
            "status": "success",
            "n_observations": len(confidences),
            "n_samples": len(beta_100_samples),
            "posterior_samples": {
                "beta_100": beta_100_samples,
                "deltas": deltas_samples,  # 2D array: (n_samples, 5)
                "sigma": sigma_samples
            },
            "summary": {
                "beta_100_mean": float(np.mean(beta_100_samples)),
                "beta_100_std": float(np.std(beta_100_samples)),
                "sigma_mean": float(np.mean(sigma_samples)),
                "sigma_std": float(np.std(sigma_samples))
            },
            "fit_object": fit
        }

        return self.last_inference_results

    def get_expected_scores_at_levels(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples of expected human scores at each LLM confidence bin.

        Returns:
            Dict with keys for each bin mapping to arrays of posterior samples
            for the expected human score in that bin.
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        samples = self.last_inference_results["posterior_samples"]
        beta_100 = samples["beta_100"]
        deltas = samples["deltas"]  # Shape: (n_samples, 5)

        # For piecewise constant delta model:
        # E[y|x > 90] = beta_100
        # E[y|80 < x <= 90] = beta_100 + deltas[:,4]  (deltas[5] in Stan, 0-indexed as 4)
        # E[y|70 < x <= 80] = beta_100 + deltas[:,4] + deltas[:,3]
        # E[y|60 < x <= 70] = beta_100 + deltas[:,4] + deltas[:,3] + deltas[:,2]
        # E[y|50 < x <= 60] = beta_100 + deltas[:,4] + deltas[:,3] + deltas[:,2] + deltas[:,1]
        # E[y|x <= 50] = beta_100 + deltas[:,4] + deltas[:,3] + deltas[:,2] + deltas[:,1] + deltas[:,0]

        return {
            "gt90": beta_100,
            "81_90": beta_100 + deltas[:, 4],
            "71_80": beta_100 + deltas[:, 4] + deltas[:, 3],
            "61_70": beta_100 + deltas[:, 4] + deltas[:, 3] + deltas[:, 2],
            "51_60": beta_100 + deltas[:, 4] + deltas[:, 3] + deltas[:, 2] + deltas[:, 1],
            "le50": beta_100 + deltas[:, 4] + deltas[:, 3] + deltas[:, 2] + deltas[:, 1] + deltas[:, 0],
        }

    def plot_calibration_curves(
        self,
        figsize: Tuple[float, float] = (10, 8),
        title: str = "Bayesian Calibration: LLM vs Human Scores (Piecewise Constant)"
    ):
        """
        Create a violinplot showing posterior distributions of expected human scores
        at each LLM confidence bin.

        Args:
            figsize: Figure size tuple (width, height)
            title: Plot title

        Returns:
            matplotlib.axes.Axes: The plot axes object
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        expected_scores = self.get_expected_scores_at_levels()

        # Prepare data for violinplot: list of arrays, one per bin
        # Order from low to high confidence
        bin_keys = self.BIN_KEYS  # ["le50", "51_60", "61_70", "71_80", "81_90", "gt90"]
        data = [expected_scores[k] for k in bin_keys]

        # X positions: use bin centers for plotting
        positions = [25, 55, 65, 75, 85, 95]

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Create violinplot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False, widths=8)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Plot perfect calibration line: LLM 0→Human 1, LLM 100→Human 3
        plt.plot([0, 100], [1, 3], color='black', linestyle='--', linewidth=2,
                label='Perfect Calibration', zorder=5)

        # Plot posterior means as points
        means = [np.mean(expected_scores[k]) for k in bin_keys]
        plt.scatter(positions, means, color='red', s=80, zorder=10, label='Posterior Mean')

        # Add bin boundary lines
        for boundary in [50, 60, 70, 80, 90]:
            plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)

        # Customize plot
        plt.xlabel('LLM Confidence Score (0-100)', fontsize=12)
        plt.ylabel('Expected Human Score (1-3)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)
        plt.xticks(positions, self.BIN_LABELS)
        plt.xlim(0, 100)
        plt.ylim(0.5, 3.5)

        plt.tight_layout()
        return ax


class BayesianMonotoneCalibration(BayesianCalibration):
    """
    Bayesian calibration using a monotonic effect model for continuous LLM confidence (0-100).

    This model enforces monotonicity: higher LLM confidence always leads to higher
    expected human scores. It uses a simplex parameterization to model how the total
    effect is distributed across confidence bins.

    Model structure:
    - E[y|x <= 50] = beta_50
    - E[y|50 < x <= 60] = beta_50 + delta_scale * mo(simplex, 1)
    - E[y|60 < x <= 70] = beta_50 + delta_scale * mo(simplex, 2)
    - E[y|70 < x <= 80] = beta_50 + delta_scale * mo(simplex, 3)
    - E[y|80 < x <= 90] = beta_50 + delta_scale * mo(simplex, 4)

    Where mo(simplex, i) = 5 * sum(simplex[1:i]) for i > 0, enforcing monotonicity.
    """

    # Bin labels for display
    BIN_LABELS = ["≤50", "51-60", "61-70", "71-80", "81-90"] #, ">90"]
    BIN_KEYS = ["50", "60", "70", "80", "90"] #, "gt90"]

    def __init__(self, stan_model_path: str = "src/bayesian_monotone_model.stan"):
        """
        Initialize the Bayesian monotone calibration system.

        Args:
            stan_model_path: Path to the Stan monotone model file
        """
        super().__init__(stan_model_path)

    def fit(
        self,
        confidences: np.ndarray,
        annotations: np.ndarray,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian monotone model to data.

        Args:
            confidences: Array of LLM confidence scores (0-100 scale)
            annotations: Array of human annotation scores (1-3 Likert)
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        if self.stan_model is None:
            self.compile_model()

        if len(confidences) < 2:
            self.last_inference_results = {
                "status": "insufficient_data",
                "n_observations": len(confidences),
                "message": "Need at least 2 data points for inference"
            }
            return self.last_inference_results

        # Preprocess data (same format as other models)
        stan_data = preprocess_data_for_stan(confidences, annotations)

        # Run MCMC sampling
        logger.info(f"Running MCMC (monotone model) with {len(confidences)} observations")
        fit = self.stan_model.sample(
            data=stan_data,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

        # Extract posterior samples
        beta_50_samples = fit.stan_variable("beta_50")
        delta_simplex_samples = fit.stan_variable("delta_simplex")  # Shape: (n_samples, 5)
        delta_scale_samples = fit.stan_variable("delta_scale")
        sigma_samples = fit.stan_variable("sigma")
        expected_by_bin_samples = fit.stan_variable("expected_by_bin")  # Shape: (n_samples, 6)

        # Calculate summary statistics
        self.last_inference_results = {
            "status": "success",
            "n_observations": len(confidences),
            "n_samples": len(beta_50_samples),
            "posterior_samples": {
                "beta_50": beta_50_samples,
                "delta_simplex": delta_simplex_samples,  # 2D array: (n_samples, 5)
                "delta_scale": delta_scale_samples,
                "sigma": sigma_samples,
                "expected_by_bin": expected_by_bin_samples  # 2D array: (n_samples, 6)
            },
            "summary": {
                "beta_50_mean": float(np.mean(beta_50_samples)),
                "beta_50_std": float(np.std(beta_50_samples)),
                "delta_scale_mean": float(np.mean(delta_scale_samples)),
                "delta_scale_std": float(np.std(delta_scale_samples)),
                "sigma_mean": float(np.mean(sigma_samples)),
                "sigma_std": float(np.std(sigma_samples))
            },
            "fit_object": fit
        }

        return self.last_inference_results

    def get_expected_scores_at_levels(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples of expected human scores at each LLM confidence bin.

        Returns:
            Dict with keys for each bin mapping to arrays of posterior samples
            for the expected human score in that bin.
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        samples = self.last_inference_results["posterior_samples"]
        expected_by_bin = samples["expected_by_bin"]  # Shape: (n_samples, 6)

        # Map bin indices to keys
        # Stan uses 1-based indexing for the array, Python 0-based
        return {
            "50": expected_by_bin[:, 0],
            "60": expected_by_bin[:, 1],
            "70": expected_by_bin[:, 2],
            "80": expected_by_bin[:, 3],
            "90": expected_by_bin[:, 4],
        }

    def plot_calibration_curves(
        self,
        figsize: Tuple[float, float] = (10, 8),
        title: str = "Bayesian Calibration: LLM vs Human Scores (Monotone)"
    ):
        """
        Create a violinplot showing posterior distributions of expected human scores
        at each LLM confidence bin.

        Args:
            figsize: Figure size tuple (width, height)
            title: Plot title

        Returns:
            matplotlib.axes.Axes: The plot axes object
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        expected_scores = self.get_expected_scores_at_levels()

        # Prepare data for violinplot: list of arrays, one per bin
        # Order from low to high confidence
        bin_keys = self.BIN_KEYS
        data = [expected_scores[k] for k in bin_keys]

        # X positions: use bin centers for plotting
        positions = [
                50,
                60,
                70,
                80,
                90,
            ]

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Create violinplot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False, widths=8)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Plot posterior means as points (connected by line to show monotonicity)
        means = [np.mean(expected_scores[k]) for k in bin_keys]
        plt.scatter(positions, means, color='red', s=80, zorder=10, label='Posterior Mean')
        plt.plot(positions, means, color='red', linewidth=2, alpha=0.7, zorder=9)

        # Customize plot
        plt.xlabel('LLM Confidence Score (0-100)', fontsize=12)
        plt.ylabel('Expected Human Score (1-3)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)
        plt.xticks(positions, positions)
        plt.xlim(40, 100)
        plt.ylim(0.5, 3.5)

        plt.tight_layout()
        return ax


class BayesianMonotoneOrderedLogisticCalibration(BayesianCalibration):
    """
    Bayesian calibration using a monotonic ordered logistic regression model for
    continuous LLM confidence (0-100) and ordinal human scores (1-5).

    This model enforces monotonicity through a simplex parameterization and uses
    ordered logistic regression for the likelihood. It outputs probabilities for
    each category (1-5) at each confidence bin rather than point estimates.

    Model structure:
    - Bins: 0-4 based on confidence_to_bin() function (≤50, 51-60, 61-70, 71-80, >80)
    - Monotonic effect: eta = delta_scale * mo(delta_simplex, bin_idx)
    - Ordered logistic: y ~ ordered_logistic(eta, c) where c are 4 cutpoints
    - Output: prob_by_bin[5][5] - probabilities for each category at each bin
    """

    # Bin labels for display (5 bins)
    BIN_LABELS = ["50", "60", "70", "80", "90"]
    BIN_KEYS = ["bin0", "bin1", "bin2", "bin3", "bin4"]

    def __init__(self, stan_model_path: str = "src/bayesian_monotone_ordered_logistic.stan"):
        """
        Initialize the Bayesian monotone ordered logistic calibration system.

        Args:
            stan_model_path: Path to the Stan monotone ordered logistic model file
        """
        super().__init__(stan_model_path)

    def fit(
        self,
        confidences: np.ndarray,
        annotations: np.ndarray,
        chains: int = 4,
        iter_sampling: int = 1000,
        iter_warmup: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian monotone ordered logistic model to data.

        Args:
            confidences: Array of LLM confidence scores (0-100 scale)
            annotations: Array of human annotation scores (1-5 ordinal)
            chains: Number of MCMC chains
            iter_sampling: Number of sampling iterations per chain
            iter_warmup: Number of warmup iterations per chain
            show_progress: Whether to show sampling progress

        Returns:
            Dict containing inference results
        """
        if self.stan_model is None:
            self.compile_model()

        if len(confidences) < 2:
            self.last_inference_results = {
                "status": "insufficient_data",
                "n_observations": len(confidences),
                "message": "Need at least 2 data points for inference"
            }
            return self.last_inference_results

        # Human scores should be 1-5 (5-point Likert scale)
        annotations_1indexed = annotations.copy()

        # Ensure annotations are in valid range 1-5
        assert np.all((annotations_1indexed >= 1) & (annotations_1indexed <= 5)), \
            "Annotations must be in range 1-5 for ordered logistic model"

        # Preprocess data
        stan_data = preprocess_data_for_stan(confidences, annotations_1indexed)

        # Run MCMC sampling
        logger.info(f"Running MCMC (monotone ordered logistic) with {len(confidences)} observations")
        fit = self.stan_model.sample(
            data=stan_data,
            chains=chains,
            iter_sampling=iter_sampling,
            iter_warmup=iter_warmup,
            show_progress=show_progress
        )

        # Extract posterior samples
        c_samples = fit.stan_variable("c")  # Shape: (n_samples, 4) - cutpoints
        delta_simplex_samples = fit.stan_variable("delta_simplex")  # Shape: (n_samples, 4)
        delta_scale_samples = fit.stan_variable("delta_scale")  # Shape: (n_samples,)
        prob_by_bin_samples = fit.stan_variable("prob_by_bin")  # Shape: (n_samples, 5, 5)

        # Calculate summary statistics
        self.last_inference_results = {
            "status": "success",
            "n_observations": len(confidences),
            "n_samples": len(delta_scale_samples),
            "posterior_samples": {
                "c": c_samples,  # 2D array: (n_samples, 4)
                "delta_simplex": delta_simplex_samples,  # 2D array: (n_samples, 4)
                "delta_scale": delta_scale_samples,  # 1D array: (n_samples,)
                "prob_by_bin": prob_by_bin_samples  # 3D array: (n_samples, 5, 5)
            },
            "summary": {
                "delta_scale_mean": float(np.mean(delta_scale_samples)),
                "delta_scale_std": float(np.std(delta_scale_samples)),
                "c1_mean": float(np.mean(c_samples[:, 0])),
                "c2_mean": float(np.mean(c_samples[:, 1])),
                "c3_mean": float(np.mean(c_samples[:, 2])),
                "c4_mean": float(np.mean(c_samples[:, 3])),
            },
            "fit_object": fit
        }

        return self.last_inference_results

    def get_expected_scores_at_levels(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples of expected human scores at each confidence bin.

        For ordered logistic, we compute the expected category value:
        E[y] = 1*P(y=1) + 2*P(y=2) + 3*P(y=3) + 4*P(y=4) + 5*P(y=5)

        Returns:
            Dict with keys for each bin mapping to arrays of posterior samples
            for the expected human score in that bin.
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        samples = self.last_inference_results["posterior_samples"]
        prob_by_bin = samples["prob_by_bin"]  # Shape: (n_samples, 5, 5)

        # Compute expected value for each bin and each posterior sample
        # Categories are 1-5, so use [1, 2, 3, 4, 5]
        category_values = np.array([1, 2, 3, 4, 5])
        expected_by_bin = np.sum(prob_by_bin * category_values[None, None, :], axis=2)  # Shape: (n_samples, 5)

        # Map bin indices to keys
        return {
            "bin0": expected_by_bin[:, 0],
            "bin1": expected_by_bin[:, 1],
            "bin2": expected_by_bin[:, 2],
            "bin3": expected_by_bin[:, 3],
            "bin4": expected_by_bin[:, 4],
        }

    def plot_calibration_curves(
        self,
        figsize: Tuple[float, float] = (10, 8),
        title: str = "Bayesian Calibration: LLM vs Human Scores (Monotone Ordered Logistic)"
    ):
        """
        Create a violinplot showing posterior distributions of expected human scores
        at each LLM confidence bin.

        Args:
            figsize: Figure size tuple (width, height)
            title: Plot title

        Returns:
            matplotlib.axes.Axes: The plot axes object
        """
        assert self.last_inference_results is not None, "Must run fit() first"
        assert self.last_inference_results["status"] == "success", "Last inference failed"

        expected_scores = self.get_expected_scores_at_levels()

        # Prepare data for violinplot: list of arrays, one per bin
        bin_keys = self.BIN_KEYS
        data = [expected_scores[k] for k in bin_keys]

        # X positions
        positions = [50, 60, 70, 80, 90]  # Midpoints of the bins

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Create violinplot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False, widths=8)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Plot posterior means as points (connected by line to show monotonicity)
        means = [np.mean(expected_scores[k]) for k in bin_keys]
        plt.scatter(positions, means, color='red', s=80, zorder=10, label='Posterior Mean')
        plt.plot(positions, means, color='red', linewidth=2, alpha=0.7, zorder=9)

        # Customize plot
        plt.xlabel('LLM Confidence Score (0-100)', fontsize=12)
        plt.ylabel('Expected Human Score (1-5)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11)
        plt.xticks(positions, self.BIN_LABELS)
        plt.xlim(40, 100)
        plt.ylim(0.5, 5.5)

        plt.tight_layout()
        return ax

