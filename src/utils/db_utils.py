"""Database utility functions for storing and loading analysis data."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional


def setup_analysis_database(db_path: str) -> duckdb.DuckDBPyConnection:
    """Setup the analysis database with required tables.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        DuckDB connection object
    """
    conn = duckdb.connect(db_path)

    # Create assembled_notes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assembled_notes (
            encounter_id VARCHAR PRIMARY KEY,
            note_text TEXT,
            note_timestamp TIMESTAMP,
            note_type VARCHAR,
            note_metadata VARCHAR
        )
    """)

    # Create extracted_reasons table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extracted_reasons (
            encounter_id VARCHAR PRIMARY KEY,
            has_reasons_mentioned BOOLEAN,
            reasons VARCHAR,  -- JSON array of reasons
            confidence VARCHAR,
            relevant_excerpt TEXT,
            extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create clustered_reasons table (flattened, one row per reason)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clustered_reasons (
            id INTEGER PRIMARY KEY,
            encounter_id VARCHAR,
            reason_text VARCHAR,
            topic_id INTEGER,
            topic_name VARCHAR,
            topic_probability FLOAT,
            representative_terms VARCHAR  -- JSON array
        )
    """)

    # Create topic_summaries table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_summaries (
            topic_id INTEGER PRIMARY KEY,
            topic_name VARCHAR,
            count INTEGER,
            representative_terms VARCHAR,  -- JSON array
            representative_docs VARCHAR,   -- JSON array
            topic_coherence FLOAT
        )
    """)

    return conn


def save_assembled_notes(df: pd.DataFrame, db_path: str):
    """Save assembled notes to database.

    Args:
        df: DataFrame with columns [encounter_id, note_text, note_timestamp, note_type, note_metadata]
        db_path: Path to the DuckDB database file
    """
    conn = duckdb.connect(db_path)
    conn.execute("DELETE FROM assembled_notes")  # Clear existing data
    conn.execute("""
        INSERT INTO assembled_notes
        SELECT * FROM df
    """)
    conn.close()
    print(f"Saved {len(df)} assembled notes to {db_path}")


def load_assembled_notes(db_path: str) -> pd.DataFrame:
    """Load assembled notes from database.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        DataFrame with assembled notes
    """
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM assembled_notes").fetchdf()
    conn.close()
    print(f"Loaded {len(df)} assembled notes from {db_path}")
    return df


def save_extracted_reasons(df: pd.DataFrame, db_path: str):
    """Save extracted reasons to database.

    Args:
        df: DataFrame with columns [encounter_id, has_reasons_mentioned, reasons, confidence, relevant_excerpt]
        db_path: Path to the DuckDB database file
    """
    conn = duckdb.connect(db_path)
    conn.execute("DELETE FROM extracted_reasons")  # Clear existing data
    conn.execute("""
        INSERT INTO extracted_reasons (encounter_id, has_reasons_mentioned, reasons, confidence, relevant_excerpt)
        SELECT * FROM df
    """)
    conn.close()
    print(f"Saved {len(df)} extracted reasons to {db_path}")


def load_extracted_reasons(db_path: str, only_with_reasons: bool = False) -> pd.DataFrame:
    """Load extracted reasons from database.

    Args:
        db_path: Path to the DuckDB database file
        only_with_reasons: If True, only return rows where has_reasons_mentioned=True

    Returns:
        DataFrame with extracted reasons
    """
    conn = duckdb.connect(db_path, read_only=True)
    query = "SELECT * FROM extracted_reasons"
    if only_with_reasons:
        query += " WHERE has_reasons_mentioned = TRUE"
    df = conn.execute(query).fetchdf()
    conn.close()
    print(f"Loaded {len(df)} extracted reasons from {db_path}")
    return df


def save_clustered_reasons(
    reasons_df: pd.DataFrame,
    topic_summaries_df: pd.DataFrame,
    db_path: str
):
    """Save clustered reasons and topic summaries to database.

    Args:
        reasons_df: DataFrame with columns [encounter_id, reason_text, topic_id, topic_name, topic_probability, representative_terms]
        topic_summaries_df: DataFrame with columns [topic_id, topic_name, count, representative_terms, representative_docs, topic_coherence]
        db_path: Path to the DuckDB database file
    """
    conn = duckdb.connect(db_path)

    # Save clustered reasons
    conn.execute("DELETE FROM clustered_reasons")
    reasons_df['id'] = range(len(reasons_df))
    conn.execute("""
        INSERT INTO clustered_reasons
        SELECT * FROM reasons_df
    """)

    # Save topic summaries
    conn.execute("DELETE FROM topic_summaries")
    conn.execute("""
        INSERT INTO topic_summaries
        SELECT * FROM topic_summaries_df
    """)

    conn.close()
    print(f"Saved {len(reasons_df)} clustered reasons and {len(topic_summaries_df)} topic summaries to {db_path}")


def load_clustered_reasons(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clustered reasons and topic summaries from database.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        Tuple of (reasons_df, topic_summaries_df)
    """
    conn = duckdb.connect(db_path, read_only=True)
    reasons_df = conn.execute("SELECT * FROM clustered_reasons").fetchdf()
    topic_summaries_df = conn.execute("SELECT * FROM topic_summaries").fetchdf()
    conn.close()
    print(f"Loaded {len(reasons_df)} clustered reasons and {len(topic_summaries_df)} topics from {db_path}")
    return reasons_df, topic_summaries_df
