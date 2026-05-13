"""Filter notes given in a csv to include only those whose LOS are in a specified time range.

Usage:
    python src/assemble_los_notes.py --source-csv file1.csv file2.csv --output-csv out.csv --min-los 1 --max-los 30
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def assemble_notes(
    input_csv_paths: List[str],
    output_csv_path: str,
    min_los: float,
    max_los: float,
    limit: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    # Read and concatenate all CSVs
    dfs = [pd.read_csv(path) for path in input_csv_paths]
    assert all(["los_days" in df.columns for df in dfs]), f"Expected 'los' column, got columns: {df.columns.tolist()}"
    assert all(["note_text" in df.columns for df in dfs]), f"Expected 'los' column, got columns: {df.columns.tolist()}"
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(input_csv_paths)} file(s)")

    # Filter by LOS range
    df_filtered = df[(df["los_days"] >= min_los) & (df["los_days"] <= max_los)]
    print(f"Filtered to {len(df_filtered)} rows with LOS in [{min_los}, {max_los}]")

    # Shuffle
    df_shuffled = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Apply limit if requested
    if limit is not None:
        df_final = df_shuffled.head(limit)
        print(f"Applied limit: {len(df_final)} rows")
    else:
        df_final = df_shuffled

    # Add Y column (unified outcome variable = los_days for LOS analysis)
    df_final["Y"] = df_final["los_days"]

    # Save to CSV
    print(f"Saving to: {output_csv_path}")
    df_final.to_csv(output_csv_path, index=False)

    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Assemble clinical notes for LOS analysis"
    )
    parser.add_argument(
        "--source-csv",
        nargs="+",
        required=True,
        help="Path(s) to source csv(s) with clinical notes"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--min-los",
        type=float,
        required=True,
        help="Minimum LOS (inclusive)"
    )
    parser.add_argument(
        "--max-los",
        type=float,
        required=True,
        help="Maximum LOS (inclusive)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    df = assemble_notes(
        input_csv_paths=args.source_csv,
        output_csv_path=args.output_csv,
        min_los=args.min_los,
        max_los=args.max_los,
        limit=args.limit,
        seed=args.seed,
    )

    print(f"Successfully assembled {len(df)} notes to {args.output_csv}")


if __name__ == "__main__":
    main()
