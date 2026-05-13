#!/usr/bin/env python3
"""Plot theme distribution from tagged reasons.

This script takes the output from tag_themes.py and creates visualizations
showing how often each theme appears in the dataset.

Usage:
    python src/plot_themes.py \
        --input-csv output/tagged_reasons.csv \
        --themes-json exp_los/_output/all/prompt_v9/default/cluster_members_claude_clean.json \
        --theme-distribution-reasons-png output/theme_distribution_reasons.png \
        --theme-distribution-encounters-png output/theme_distribution_encounters.png \
        --theme-summary-csv output/theme_summary.csv
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up logger to write to file and console."""
    logger = logging.getLogger("plot_themes")
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


def load_themes_json(themes_json_path: str) -> dict:
    """Load themes from JSON file."""
    assert Path(themes_json_path).exists(), f"Themes JSON not found: {themes_json_path}"

    with open(themes_json_path, 'r', encoding='utf-8') as f:
        themes = json.load(f)

    return themes


def count_themes(df: pd.DataFrame, theme_column: str = "theme_names") -> dict:
    """Count occurrences of each theme.

    Args:
        df: DataFrame with tagged reasons
        theme_column: Column containing JSON-encoded theme names

    Returns:
        Dictionary mapping theme_name to count
    """
    theme_counts = {}

    for themes_json in df[df[theme_column].notna()][theme_column]:
        theme_names = json.loads(themes_json)
        for theme_name in theme_names:
            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1

    return theme_counts


def count_themes_by_encounter(df: pd.DataFrame, theme_column: str = "theme_names") -> dict:
    """Count unique encounters per theme (each encounter counted once per theme).

    Args:
        df: DataFrame with tagged reasons (must have 'encounter_id' column)
        theme_column: Column containing JSON-encoded theme names

    Returns:
        Dictionary mapping theme_name to unique encounter count
    """
    theme_encounters = {}

    for _, row in df[df[theme_column].notna()].iterrows():
        encounter_id = row['encounter_id']
        theme_names = json.loads(row[theme_column])
        for theme_name in theme_names:
            if theme_name not in theme_encounters:
                theme_encounters[theme_name] = set()
            theme_encounters[theme_name].add(encounter_id)

    return {k: len(v) for k, v in theme_encounters.items()}


def plot_theme_distribution(
    theme_counts: dict,
    output_path: str,
    title: str = "Theme Distribution",
    figsize: tuple = None,
    top_n: int = None,
    color: str = "#2E86AB",
) -> None:
    """Create horizontal bar chart of theme distribution.

    Args:
        theme_counts: Dictionary mapping theme_name to count
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height). If None, auto-calculated based on number of themes.
        top_n: If specified, only show top N themes
        color: Bar color
    """
    # Sort by count descending
    sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])

    if top_n:
        sorted_themes = sorted_themes[:top_n]

    # Theme names are already in the keys
    names = [t[0] for t in sorted_themes]
    counts = [t[1] for t in sorted_themes]

    # Auto-calculate figure size based on number of themes
    if figsize is None:
        height = max(8, len(names) * 0.4)  # ~0.4 inches per bar
        figsize = (12, height)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, counts, color=color, edgecolor='white', linewidth=0.5)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va='center',
            fontsize=10,
        )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Extend x-axis limit to make room for count labels
    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count * 1.15)

    # Adjust layout with extra left margin for long theme names
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_themes(
    input_csv_path: str,
    themes_json_path: str,
    theme_distribution_reasons_png: str,
    theme_distribution_encounters_png: str,
    theme_summary_csv: str,
    log_file_path: str = None,
    top_n: int = None,
) -> None:
    """Generate theme distribution plots.

    Args:
        input_csv_path: Path to tagged reasons CSV
        themes_json_path: Path to themes JSON file
        theme_distribution_reasons_png: Output path for reasons distribution plot
        theme_distribution_encounters_png: Output path for encounters distribution plot
        theme_summary_csv: Output path for theme summary CSV
        log_file_path: Optional path to log file
        top_n: If specified, only show top N themes
    """
    logger = setup_logger(log_file_path)

    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV not found: {input_csv_path}"
    assert Path(themes_json_path).exists(), f"Themes JSON not found: {themes_json_path}"

    logger.info(f"Plotting theme distribution")
    logger.info(f"Input CSV: {input_csv_path}")
    logger.info(f"Themes JSON: {themes_json_path}")

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(input_csv_path)
    themes = load_themes_json(themes_json_path)

    # Build reverse lookup from topic_name to theme data for qi_initiative
    theme_by_name = {
        data.get('topic_name', tid): data
        for tid, data in themes.items()
    }

    logger.info(f"Loaded {len(df)} tagged reasons")
    logger.info(f"Loaded {len(themes)} themes")

    # Count themes (by reason)
    logger.info("Counting themes by reason...")
    theme_counts = count_themes(df)
    logger.info(f"Found {len(theme_counts)} themes with at least one reason")

    # Count themes (by encounter)
    logger.info("Counting themes by encounter...")
    encounter_counts = count_themes_by_encounter(df)

    # Plot 1: Theme distribution by reason count
    logger.info("Creating theme distribution plot (by reasons)...")
    Path(theme_distribution_reasons_png).parent.mkdir(parents=True, exist_ok=True)
    plot_theme_distribution(
        theme_counts,
        output_path=theme_distribution_reasons_png,
        title="Theme Distribution (by Reason Count)",
        top_n=top_n,
    )

    # Plot 2: Theme distribution by encounter count
    logger.info("Creating theme distribution plot (by encounters)...")
    Path(theme_distribution_encounters_png).parent.mkdir(parents=True, exist_ok=True)
    plot_theme_distribution(
        encounter_counts,
        output_path=theme_distribution_encounters_png,
        title="Theme Distribution (by Unique Encounters)",
        top_n=top_n,
        color="#27AE60",
    )

    # Generate summary statistics
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total reasons: {len(df)}")
    logger.info(f"Reasons with at least one theme: {df['theme_names'].notna().sum()}")
    logger.info(f"Unique encounters: {df['encounter_id'].nunique()}")
    logger.info(f"Themes assigned: {len(theme_counts)}")
    logger.info(f"Total theme assignments: {sum(theme_counts.values())}")
    logger.info(f"Mean themes per reason: {sum(theme_counts.values()) / len(df):.2f}")

    # Top themes table
    logger.info("\nTop 10 themes by reason count:")
    sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])[:10]
    for i, (theme_name, count) in enumerate(sorted_themes, 1):
        enc_count = encounter_counts.get(theme_name, 0)
        logger.info(f"  {i}. {theme_name}: {count} reasons, {enc_count} encounters")

    # Save summary to CSV
    summary_data = []
    for theme_name, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        qi_initiative = theme_by_name.get(theme_name, {}).get('qi_initiative', '')
        enc_count = encounter_counts.get(theme_name, 0)
        summary_data.append({
            'theme_name': theme_name,
            'qi_initiative': qi_initiative,
            'reason_count': count,
            'encounter_count': enc_count,
        })

    summary_df = pd.DataFrame(summary_data)
    Path(theme_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(theme_summary_csv, index=False)
    logger.info(f"\nSaved theme summary to: {theme_summary_csv}")

    logger.info("Plotting complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot theme distribution from tagged reasons"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to tagged reasons CSV (output from tag_themes.py)",
    )
    parser.add_argument(
        "--themes-json",
        required=True,
        help="Path to themes JSON file",
    )
    parser.add_argument(
        "--theme-distribution-reasons-png",
        required=True,
        help="Output path for theme distribution by reasons plot",
    )
    parser.add_argument(
        "--theme-distribution-encounters-png",
        required=True,
        help="Output path for theme distribution by encounters plot",
    )
    parser.add_argument(
        "--theme-summary-csv",
        required=True,
        help="Output path for theme summary CSV",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (optional)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only show top N themes in plots",
    )

    args = parser.parse_args()

    plot_themes(
        input_csv_path=args.input_csv,
        themes_json_path=args.themes_json,
        theme_distribution_reasons_png=args.theme_distribution_reasons_png,
        theme_distribution_encounters_png=args.theme_distribution_encounters_png,
        theme_summary_csv=args.theme_summary_csv,
        log_file_path=args.log_file,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
