#!/usr/bin/env python3
"""Generate the calibration panels reported in the paper, with per-bin
sample sizes.

Reproduces plot_confidence_intervals_by_bin from src/analyze_annotations.py
(same style, data, and t-based CIs for the mean expert rating per confidence
bin) and adds an "n=..." annotation under each bin.

Reads the annotation databases through src/agreement_metrics.py (see that
module's docstring for the expected data layout).

Usage: python src/generate_calibration_figure.py
Outputs: exp_los/_output/plot_calibration_los.png,
         exp_readmission/_output/plot_calibration_readm.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agreement_metrics import REPO, STUDIES, load  # noqa: E402

OUTPUTS = {
    "los": REPO / "exp_los/_output/plot_calibration_los.png",
    "readmission": REPO / "exp_readmission/_output/plot_calibration_readm.png",
}


def bin_labels(cutpoints):
    bins = [0] + list(cutpoints) + [101]
    labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            labels.append(f"{bins[i]}-100")
        else:
            labels.append(f"{bins[i]}-{bins[i + 1] - 1}")
    return bins, labels


def bin_stats(df, cutpoints):
    bins, labels = bin_labels(cutpoints)
    df = df.copy()
    df["confidence_bin"] = pd.cut(
        df["confidence"], bins=bins, labels=labels, right=False, include_lowest=True
    )
    rows = []
    for label, grp in df.groupby("confidence_bin", observed=True):
        n = len(grp)
        mean = grp["annotation"].mean()
        std = grp["annotation"].std()
        if n > 1 and pd.notna(std):
            sem = std / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            lo, hi = mean - t_crit * sem, mean + t_crit * sem
        else:
            lo = hi = np.nan
        rows.append({"bin": label, "n": n, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def make_plot(stats_df, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(stats_df))
    means = stats_df["mean"].to_numpy()
    yerr = np.array(
        [means - stats_df["lo"].to_numpy(), stats_df["hi"].to_numpy() - means]
    )
    yerr = np.nan_to_num(yerr, nan=0.0)

    ax.errorbar(
        x,
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

    for i, n in enumerate(stats_df["n"]):
        ax.annotate(f"n={n}", (i, 1.10), ha="center", fontsize=12, color="gray")

    ax.set_xlim(-0.55, len(stats_df) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["bin"], fontsize=14)
    ax.set_xlabel("LLM Confidence Bin", fontsize=16)
    ax.set_ylabel("Mean Human Annotation", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(1, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main():
    for study, cfg in STUDIES.items():
        _, conf_df = load(cfg["db"], cfg["prompt_id"], cfg["bin_edges"])
        stats_df = bin_stats(conf_df, cfg["bin_edges"])
        print(
            f"{study}: "
            + ", ".join(
                f"{r['bin']} (n={r['n']}, mean={r['mean']:.2f})"
                for _, r in stats_df.iterrows()
            )
        )
        OUTPUTS[study].parent.mkdir(parents=True, exist_ok=True)
        make_plot(stats_df, OUTPUTS[study])


if __name__ == "__main__":
    main()
