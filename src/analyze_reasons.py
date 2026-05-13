"""Phase 4: Analyze and visualize readmission reasons.

This script generates statistics and visualizations from the clustered reasons.

Usage:
    python src/analyze_reasons.py --extracted-csv <path> --clustered-csv <path> --topics-csv <path> --output-dir <path>
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def setup_plotting_style():
    """Setup matplotlib/seaborn plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def analyze_detection_rate(df: pd.DataFrame, output_file: str):
    """Analyze and plot detection rate.

    Args:
        df: DataFrame with extracted reasons
        output_file: Output file path for the plot
    """
    print("\n=== Detection Rate Analysis ===")

    # Filter out failed extractions
    df_valid = df[df['has_reasons_mentioned'].notna()]
    total = len(df_valid)
    with_reasons = df_valid['has_reasons_mentioned'].sum()
    detection_rate = with_reasons / total if total > 0 else 0

    print(f"Total notes: {len(df)}")
    print(f"Valid extractions: {total}")
    print(f"Notes with reasons mentioned: {with_reasons}")
    print(f"Detection rate: {detection_rate:.1%}")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['With Reasons', 'Without Reasons']
    counts = [with_reasons, total - with_reasons]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Notes')
    ax.set_title('Readmission Reason Detection Rate')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/total:.1%})',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_confidence(df: pd.DataFrame, output_file: str):
    """Analyze confidence distribution.

    Args:
        df: DataFrame with extracted reasons
        output_file: Output file path for the plot
    """
    print("\n=== Confidence Analysis ===")

    df_valid = df[df['confidence'].notna()]
    confidence_counts = df_valid['confidence'].value_counts()

    print("Confidence distribution:")
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} ({count/len(df_valid):.1%})")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    confidence_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Number of Notes')
    ax.set_title('Extraction Confidence Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Add count labels
    for i, (conf, count) in enumerate(confidence_counts.items()):
        ax.text(i, count, f'{count}\n({count/len(df_valid):.1%})',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_reasons_per_encounter(reasons_df: pd.DataFrame, output_file: str):
    """Analyze number of reasons per encounter.

    Args:
        reasons_df: DataFrame with clustered reasons (one row per reason)
        output_file: Output file path for the plot
    """
    print("\n=== Reasons per Encounter Analysis ===")

    reasons_per_encounter = reasons_df.groupby('encounter_id').size()

    print(f"Total encounters: {len(reasons_per_encounter)}")
    print(f"Mean reasons per encounter: {reasons_per_encounter.mean():.2f}")
    print(f"Median reasons per encounter: {reasons_per_encounter.median():.0f}")
    print(f"Max reasons per encounter: {reasons_per_encounter.max()}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    reasons_per_encounter.hist(bins=range(1, reasons_per_encounter.max()+2),
                                ax=ax, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Reasons')
    ax.set_ylabel('Number of Encounters')
    ax.set_title('Distribution of Reasons per Encounter')
    ax.axvline(reasons_per_encounter.mean(), color='red', linestyle='--',
               label=f'Mean: {reasons_per_encounter.mean():.2f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_top_reasons(reasons_df: pd.DataFrame, output_file: str, top_n: int = 20):
    """Analyze and plot top reasons.

    Args:
        reasons_df: DataFrame with clustered reasons
        output_file: Output file path for the plot
        top_n: Number of top reasons to show
    """
    print(f"\n=== Top {top_n} Reasons Analysis ===")

    reason_counts = reasons_df['reason_text'].value_counts().head(top_n)

    print(f"Top {top_n} reasons:")
    for i, (reason, count) in enumerate(reason_counts.items(), 1):
        print(f"{i:2d}. {reason}: {count}")

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 10))
    reason_counts.plot(kind='barh', ax=ax, color='teal', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Reason')
    ax.set_title(f'Top {top_n} Readmission Reasons')
    ax.invert_yaxis()

    # Add count labels
    for i, count in enumerate(reason_counts.values):
        ax.text(count, i, f' {count}', va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_topics(
    reasons_df: pd.DataFrame,
    topic_summaries_df: pd.DataFrame,
    output_file: str
):
    """Analyze topic distribution.

    Args:
        reasons_df: DataFrame with clustered reasons
        topic_summaries_df: DataFrame with topic summaries
        output_file: Output file path for the plot
    """
    print("\n=== Topic Analysis ===")

    # Remove outlier topic if present
    topic_summaries_clean = topic_summaries_df[topic_summaries_df['topic_id'] != -1]

    print(f"Number of topics (excluding outliers): {len(topic_summaries_clean)}")
    print(f"Outlier reasons (topic -1): {(reasons_df['topic_id'] == -1).sum()}")

    # Show top topics
    top_topics = topic_summaries_clean.nlargest(10, 'count')
    print("\nTop 10 topics by size:")
    for _, row in top_topics.iterrows():
        terms = json.loads(row['representative_terms'])[:5]
        print(f"  Topic {row['topic_id']}: {row['count']} reasons - {', '.join(terms)}")

    # Plot topic distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = topic_summaries_clean.nlargest(20, 'count')

    # Create labels with topic terms
    labels = []
    for _, row in top_20.iterrows():
        terms = json.loads(row['representative_terms'])[:3]
        label = f"T{row['topic_id']}: {', '.join(terms)}"
        labels.append(label)

    ax.barh(range(len(top_20)), top_20['count'], color='mediumpurple', alpha=0.7)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Reasons')
    ax.set_title('Top 20 Topics by Size')
    ax.invert_yaxis()

    # Add count labels
    for i, count in enumerate(top_20['count'].values):
        ax.text(count, i, f' {count}', va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_wordcloud(reasons_df: pd.DataFrame, output_file: str):
    """Create word cloud from all reasons.

    Args:
        reasons_df: DataFrame with clustered reasons
        output_file: Output file path for the plot
    """
    print("\n=== Creating Word Cloud ===")

    # Combine all reason texts
    all_text = ' '.join(reasons_df['reason_text'].tolist())

    # Create word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5
    ).generate(all_text)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Readmission Reasons Word Cloud', fontsize=20, pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_bertopic_visualizations(
    topic_model_path: str,
    output_dir: Path
):
    """Generate BERTopic built-in visualizations.

    Args:
        topic_model_path: Path to saved BERTopic model
        output_dir: Output directory for plots
    """
    print("\n=== Generating BERTopic Visualizations ===")

    assert Path(topic_model_path).exists(), f"Topic model not found: {topic_model_path}"

    # Load model
    with open(topic_model_path, 'rb') as f:
        topic_model = pickle.load(f)

    # Generate visualizations (these create interactive HTML plots)
    visualizations = {
        'topics': topic_model.visualize_topics,
        'barchart': topic_model.visualize_barchart,
        'hierarchy': topic_model.visualize_hierarchy,
        'heatmap': topic_model.visualize_heatmap,
    }

    for name, viz_func in visualizations.items():
        try:
            fig = viz_func()
            output_path = output_dir / f'topic_{name}.html'
            fig.write_html(str(output_path))
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Warning: Could not generate {name} visualization: {e}")


def generate_summary_report(
    extracted_df: pd.DataFrame,
    reasons_df: pd.DataFrame,
    topic_summaries_df: pd.DataFrame,
    output_file: str
):
    """Generate summary statistics report.

    Args:
        extracted_df: DataFrame with extracted reasons
        reasons_df: DataFrame with clustered reasons
        topic_summaries_df: DataFrame with topic summaries
        output_file: Output file path for the report
    """
    print("\n=== Generating Summary Report ===")

    report_lines = [
        "# Readmission Reasons Analysis Summary",
        "",
        "## Extraction Statistics",
        f"- Total notes analyzed: {len(extracted_df)}",
        f"- Notes with reasons: {extracted_df['has_reasons_mentioned'].sum()}",
        f"- Detection rate: {extracted_df['has_reasons_mentioned'].mean():.1%}",
        "",
        "## Reason Statistics",
        f"- Total reasons extracted: {len(reasons_df)}",
        f"- Unique reasons: {reasons_df['reason_text'].nunique()}",
        f"- Mean reasons per encounter: {reasons_df.groupby('encounter_id').size().mean():.2f}",
        "",
        "## Topic Statistics",
        f"- Number of topics (excluding outliers): {len(topic_summaries_df[topic_summaries_df['topic_id'] != -1])}",
        f"- Outlier reasons: {(reasons_df['topic_id'] == -1).sum()}",
        "",
        "## Top 10 Reasons",
    ]

    top_reasons = reasons_df['reason_text'].value_counts().head(10)
    for i, (reason, count) in enumerate(top_reasons.items(), 1):
        report_lines.append(f"{i}. {reason}: {count}")

    report_lines.extend([
        "",
        "## Top 5 Topics",
    ])

    top_topics = topic_summaries_df[topic_summaries_df['topic_id'] != -1].nlargest(5, 'count')
    for _, row in top_topics.iterrows():
        terms = json.loads(row['representative_terms'])[:5]
        report_lines.append(f"- Topic {row['topic_id']} ({row['count']} reasons): {', '.join(terms)}")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved: {output_file}")
    print("\n" + "\n".join(report_lines))


def analyze_results(
    extracted_csv_path: str,
    clustered_csv_path: str,
    topics_csv_path: str,
    topic_model_path: str,
    detection_rate_png: str,
    confidence_distribution_png: str,
    reasons_per_encounter_png: str,
    top_reasons_png: str,
    topic_distribution_png: str,
    wordcloud_png: str,
    summary_report_md: str
):
    """Analyze and visualize results.

    Args:
        extracted_csv_path: Path to extracted reasons CSV
        clustered_csv_path: Path to clustered reasons CSV
        topics_csv_path: Path to topic summaries CSV
        topic_model_path: Path to saved BERTopic model
        detection_rate_png: Path to save detection rate plot
        confidence_distribution_png: Path to save confidence distribution plot
        reasons_per_encounter_png: Path to save reasons per encounter plot
        top_reasons_png: Path to save top reasons plot
        topic_distribution_png: Path to save topic distribution plot
        wordcloud_png: Path to save wordcloud plot
        summary_report_md: Path to save summary report
    """
    # Validate inputs
    assert Path(extracted_csv_path).exists(), f"Extracted reasons CSV not found: {extracted_csv_path}"
    assert Path(clustered_csv_path).exists(), f"Clustered reasons CSV not found: {clustered_csv_path}"
    assert Path(topics_csv_path).exists(), f"Topic summaries CSV not found: {topics_csv_path}"

    # Create output directories for all files
    for file_path in [detection_rate_png, confidence_distribution_png, reasons_per_encounter_png,
                      top_reasons_png, topic_distribution_png, wordcloud_png, summary_report_md]:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Setup plotting
    setup_plotting_style()

    # Load data
    print("=== Loading data ===")
    extracted_df = pd.read_csv(extracted_csv_path)
    reasons_df = pd.read_csv(clustered_csv_path)
    topic_summaries_df = pd.read_csv(topics_csv_path)

    print(f"Loaded {len(extracted_df)} extracted reasons from {extracted_csv_path}")
    print(f"Loaded {len(reasons_df)} clustered reasons from {clustered_csv_path}")
    print(f"Loaded {len(topic_summaries_df)} topic summaries from {topics_csv_path}")

    # Run analyses
    analyze_detection_rate(extracted_df, detection_rate_png)
    analyze_confidence(extracted_df, confidence_distribution_png)
    analyze_reasons_per_encounter(reasons_df, reasons_per_encounter_png)
    analyze_top_reasons(reasons_df, top_reasons_png)
    analyze_topics(reasons_df, topic_summaries_df, topic_distribution_png)
    create_wordcloud(reasons_df, wordcloud_png)

    # Generate summary report
    generate_summary_report(extracted_df, reasons_df, topic_summaries_df, summary_report_md)

    print(f"\n✓ Analysis complete! Results saved to individual files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize readmission reasons"
    )
    parser.add_argument(
        "--extracted-csv",
        default="output/extracted_reasons.csv",
        help="Path to extracted reasons CSV file"
    )
    parser.add_argument(
        "--clustered-csv",
        default="output/clustered_reasons.csv",
        help="Path to clustered reasons CSV file"
    )
    parser.add_argument(
        "--topics-csv",
        default="output/topic_summaries.csv",
        help="Path to topic summaries CSV file"
    )
    parser.add_argument(
        "--topic-model",
        default="output/topic_model.pkl",
        help="Path to saved BERTopic model"
    )
    parser.add_argument(
        "--detection-rate-png",
        required=True,
        help="Path to save detection rate plot"
    )
    parser.add_argument(
        "--confidence-distribution-png",
        required=True,
        help="Path to save confidence distribution plot"
    )
    parser.add_argument(
        "--reasons-per-encounter-png",
        required=True,
        help="Path to save reasons per encounter plot"
    )
    parser.add_argument(
        "--top-reasons-png",
        required=True,
        help="Path to save top reasons plot"
    )
    parser.add_argument(
        "--topic-distribution-png",
        required=True,
        help="Path to save topic distribution plot"
    )
    parser.add_argument(
        "--wordcloud-png",
        required=True,
        help="Path to save wordcloud plot"
    )
    parser.add_argument(
        "--summary-report-md",
        required=True,
        help="Path to save summary report"
    )

    args = parser.parse_args()

    # Run analysis
    analyze_results(
        extracted_csv_path=args.extracted_csv,
        clustered_csv_path=args.clustered_csv,
        topics_csv_path=args.topics_csv,
        topic_model_path=args.topic_model,
        detection_rate_png=args.detection_rate_png,
        confidence_distribution_png=args.confidence_distribution_png,
        reasons_per_encounter_png=args.reasons_per_encounter_png,
        top_reasons_png=args.top_reasons_png,
        topic_distribution_png=args.topic_distribution_png,
        wordcloud_png=args.wordcloud_png,
        summary_report_md=args.summary_report_md
    )


if __name__ == "__main__":
    main()
