"""Phase 3: Cluster readmission reasons using BERTopic.

This script clusters extracted reasons using BERTopic with biomedical embeddings
and compares with k-means clustering.

Usage:
    python src/cluster_reasons.py --input-csv <path> --output-dir <path>
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_reason_corpus(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Prepare corpus of reasons from extracted data.

    Args:
        df: DataFrame with extracted reasons (already flattened, one row per reason)

    Returns:
        Tuple of (filtered_df, reasons_list)
    """
    # Filter to only rows with actual reasons (not null reason_text)
    df_with_reasons = df[df['reason_text'].notna()].copy()
    assert len(df_with_reasons) > 0, "No reasons found in the data"

    print(f"Encounters with reasons: {df_with_reasons['encounter_id'].nunique()}")
    print(f"Total individual reasons: {len(df_with_reasons)}")

    # Basic cleaning and filtering
    filtered_rows = []
    for _, row in df_with_reasons.iterrows():
        reason = str(row['reason_text']).strip().lower()

        # Filter very short reasons
        if len(reason.split()) >= 2:  # At least 2 words
            filtered_rows.append({
                'encounter_id': row['encounter_id'],
                'reason_text': reason,
                'category': row.get('category', 'unknown'),
                'confidence': row.get('confidence', 0.5),
                'explanation_support': row.get('explanation_support', ''),
                'explanation_contrary': row.get('explanation_contrary', ''),
                'relevant_quotes': row.get('relevant_quotes', '')
            })

    filtered_df = pd.DataFrame(filtered_rows)
    assert len(filtered_df) > 0, "No valid reasons after filtering"

    print(f"Reasons after filtering: {len(filtered_df)}")
    print(f"Unique reasons: {filtered_df['reason_text'].nunique()}")

    if 'category' in filtered_df.columns:
        print(f"Category distribution: {filtered_df['category'].value_counts().to_dict()}")

    reasons_list = filtered_df['reason_text'].tolist()
    return filtered_df, reasons_list


def generate_embeddings(
    reasons_list: list,
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Generate embeddings for reasons using biomedical model.

    Args:
        reasons_list: List of reason texts
        model_name: Name of sentence transformer model

    Returns:
        Numpy array of embeddings
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(reasons_list)} reasons...")
    embeddings = model.encode(reasons_list, show_progress_bar=True)

    assert embeddings.shape[0] == len(reasons_list), \
        f"Expected {len(reasons_list)} embeddings, got {embeddings.shape[0]}"
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def cluster_with_bertopic(
    reasons_list: list,
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    n_clusters: int = 10,
) -> Tuple[BERTopic, list, list]:
    """Cluster reasons using BERTopic.

    Args:
        reasons_list: List of reason texts
        embeddings: Precomputed embeddings
        min_cluster_size: Minimum size for a cluster

    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    print(f"\n=== Running BERTopic ===")

    # Configure UMAP
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Configure HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        prediction_data=True
    )

    # Create BERTopic model
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_clusters,
        top_n_words=10,
        n_gram_range=(1, 2),
        calculate_probabilities=True,
        verbose=True
    )

    # Fit model
    print("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(reasons_list, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info)} topics (including outliers)")
    print(f"Outlier topic (-1) has {(np.array(topics) == -1).sum()} reasons")

    # Show top topics
    print("\nTop 10 topics by size:")
    print(topic_info.head(10)[['Topic', 'Count', 'Name']])

    return topic_model, topics, probs


def cluster_with_kmeans(
    embeddings: np.ndarray,
    k_range: range = range(5, 30)
) -> Tuple[KMeans, int, list]:
    """Cluster reasons using k-means with optimal k selection.

    Args:
        embeddings: Precomputed embeddings
        k_range: Range of k values to test

    Returns:
        Tuple of (kmeans_model, optimal_k, cluster_labels)
    """
    print(f"\n=== Running k-means ===")
    print(f"Testing k from {k_range.start} to {k_range.stop-1}")

    # Find optimal k using silhouette score
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"k={k}: silhouette={score:.3f}")

    # Choose k with best silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    print(f"\nOptimal k: {optimal_k} (silhouette={best_score:.3f})")

    # Cluster with optimal k
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(embeddings)

    return kmeans_final, optimal_k, labels


def prepare_cluster_members_json(
    reasons_list: list,
    topics: list,
    topic_model: BERTopic,
    max_members: int = 20
) -> dict:
    """Prepare cluster members as JSON output.

    Args:
        reasons_list: List of reason texts
        topics: List of topic assignments
        topic_model: Fitted BERTopic model

    Returns:
        Dict mapping topic_id to cluster info including members
    """
    topic_info = topic_model.get_topic_info()
    topic_names = dict(zip(topic_info['Topic'], topic_info['Name']))

    # Group reasons by topic
    clusters = {}
    for reason, topic_id in zip(reasons_list, topics):
        if topic_id not in clusters:
            clusters[topic_id] = []
        if len(clusters[topic_id]) < max_members:
            clusters[topic_id].append(reason)

    # Format output
    result = {}
    for topic_id, members in sorted(clusters.items()):
        topic_name = topic_names.get(topic_id, f"Topic_{topic_id}")
        terms = topic_model.get_topic(topic_id)
        representative_terms = [term for term, _ in terms[:5]] if terms and topic_id != -1 else []

        result[str(topic_id)] = {
            "topic_name": topic_name,
            "members": sorted(set(members))  # Unique, sorted members
        }

    return result


def prepare_results_for_saving(
    flattened_df: pd.DataFrame,
    topics: list,
    probs: list,
    topic_model: BERTopic
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare BERTopic results for database storage.

    Args:
        flattened_df: DataFrame with encounter_id and reason_text
        topics: List of topic assignments
        probs: List of topic probabilities
        topic_model: Fitted BERTopic model

    Returns:
        Tuple of (reasons_df, topic_summaries_df)
    """
    # Create reasons dataframe with topic assignments
    reasons_df = flattened_df.copy()
    reasons_df['topic_id'] = topics
    reasons_df['topic_probability'] = [p.max() if hasattr(p, 'max') else p for p in probs]

    # Get topic names and representative terms
    topic_info = topic_model.get_topic_info()
    topic_names = dict(zip(topic_info['Topic'], topic_info['Name']))
    reasons_df['topic_name'] = reasons_df['topic_id'].map(topic_names)

    # Get representative terms for each reason's topic
    topic_terms_dict = {}
    for topic_id in reasons_df['topic_id'].unique():
        if topic_id == -1:
            topic_terms_dict[topic_id] = json.dumps([])
        else:
            terms = topic_model.get_topic(topic_id)
            if terms:
                topic_terms_dict[topic_id] = json.dumps([term for term, _ in terms[:5]])
            else:
                topic_terms_dict[topic_id] = json.dumps([])

    reasons_df['representative_terms'] = reasons_df['topic_id'].map(topic_terms_dict)

    # Create topic summaries dataframe
    topic_summaries = []
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        terms = topic_model.get_topic(topic_id)

        if terms and topic_id != -1:
            representative_terms = [term for term, _ in terms[:10]]
            representative_docs = topic_model.get_representative_docs(topic_id)[:5]
        else:
            representative_terms = []
            representative_docs = []

        topic_summaries.append({
            'topic_id': topic_id,
            'topic_name': row['Name'],
            'count': row['Count'],
            'representative_terms': json.dumps(representative_terms),
            'representative_docs': json.dumps(representative_docs),
            'topic_coherence': None  # Could compute if needed
        })

    topic_summaries_df = pd.DataFrame(topic_summaries)

    return reasons_df, topic_summaries_df


def cluster_reasons(
    input_csv_path: str,
    clustered_reasons_csv: str,
    topic_summaries_csv: str,
    topic_model_pkl: str,
    cluster_members_json: str,
    n_neighbors: int = 15,
    min_cluster_size: int = 5,
    n_clusters: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
    min_confidence: float = None,
):
    """Cluster readmission reasons using BERTopic.

    Args:
        input_csv_path: Path to extracted reasons CSV
        clustered_reasons_csv: Path to save clustered reasons CSV
        topic_summaries_csv: Path to save topic summaries CSV
        topic_model_pkl: Path to save BERTopic model pickle
        cluster_members_json: Path to save cluster members JSON
        n_neighbors: Number of neighbors for UMAP
        min_cluster_size: Minimum cluster size for HDBSCAN
        n_clusters: Target number of clusters
        embedding_model: Name of embedding model to use
        min_confidence: Minimum LLM score to include a reason (filters on 'confidence' column)
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV not found: {input_csv_path}"

    # Create output directories for all files
    for file_path in [clustered_reasons_csv, topic_summaries_csv, topic_model_pkl, cluster_members_json]:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Load extracted reasons
    print("=== Loading extracted reasons ===")
    df = pd.read_csv(input_csv_path)
    assert len(df) > 0, "No extracted reasons found"

    # Filter by minimum LLM score if specified
    if min_confidence is not None:
        assert 'confidence' in df.columns, f"Column 'confidence' not found in input CSV. Available columns: {list(df.columns)}"
        original_count = len(df)
        df = df[df['confidence'] >= min_confidence].copy()
        print(f"Filtered by min_confidence >= {min_confidence}: {original_count} -> {len(df)} reasons")
        assert len(df) > 0, f"No reasons remaining after filtering by min_confidence >= {min_confidence}"

    # Prepare corpus
    print("\n=== Preparing reason corpus ===")
    flattened_df, reasons_list = prepare_reason_corpus(df)

    # Generate embeddings
    print("\n=== Generating embeddings ===")
    embeddings = generate_embeddings(reasons_list, model_name=embedding_model)

    # Cluster with BERTopic
    topic_model, topics, probs = cluster_with_bertopic(
        reasons_list, embeddings, n_neighbors, min_cluster_size, n_clusters
    )

    # Prepare results for saving
    print("\n=== Preparing results ===")
    reasons_df, topic_summaries_df = prepare_results_for_saving(
        flattened_df, topics, probs, topic_model
    )

    # Prepare cluster members JSON
    cluster_members = prepare_cluster_members_json(reasons_list, topics, topic_model)

    # Save cluster members JSON
    print(f"\n=== Saving cluster members JSON to {cluster_members_json} ===")
    with open(cluster_members_json, 'w') as f:
        json.dump(cluster_members, f, indent=2)

    # Save results to CSV
    print(f"\n=== Saving results ===")
    reasons_df.to_csv(clustered_reasons_csv, index=False)
    topic_summaries_df.to_csv(topic_summaries_csv, index=False)
    print(f"Saved clustered reasons to {clustered_reasons_csv}")
    print(f"Saved topic summaries to {topic_summaries_csv}")

    # Save model
    print(f"\n=== Saving BERTopic model to {topic_model_pkl} ===")
    with open(topic_model_pkl, 'wb') as f:
        pickle.dump(topic_model, f)

    print(f"\n✓ Successfully clustered {len(reasons_df)} reasons into {len(topic_summaries_df)} topics")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cluster readmission reasons using BERTopic"
    )
    parser.add_argument(
        "--input-csv",
        default="output/extracted_reasons.csv",
        help="Path to extracted reasons CSV file"
    )
    parser.add_argument(
        "--clustered-reasons-csv",
        required=True,
        help="Path to save clustered reasons CSV file"
    )
    parser.add_argument(
        "--topic-summaries-csv",
        required=True,
        help="Path to save topic summaries CSV file"
    )
    parser.add_argument(
        "--topic-model-pkl",
        required=True,
        help="Path to save BERTopic model pickle file"
    )
    parser.add_argument(
        "--cluster-members-json",
        required=True,
        help="Path to save cluster members JSON file"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="target num neighbors"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="target num clusters"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum LLM score to include a reason (filters on 'confidence' column)"
    )

    args = parser.parse_args()

    # Run clustering
    cluster_reasons(
        input_csv_path=args.input_csv,
        clustered_reasons_csv=args.clustered_reasons_csv,
        topic_summaries_csv=args.topic_summaries_csv,
        topic_model_pkl=args.topic_model_pkl,
        cluster_members_json=args.cluster_members_json,
        n_neighbors=args.n_neighbors,
        min_cluster_size=args.min_cluster_size,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
