# Given a csv file that was exported from the web interface,
# adds the annotations to the annotations database.

import argparse
import csv
import sys
from pathlib import Path

import duckdb


def upload_annotations(
    prompt_id: str,
    input_csv_path: str,
    db_path: str,
) -> int:
    """Upload annotations from a CSV file to the annotations database.

    Args:
        input_csv_path: Path to CSV file exported from web interface
        db_path: Path to the DuckDB database file

    Returns:
        Number of annotations uploaded
    """
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"
    assert Path(db_path).exists(), f"Database file not found: {db_path}"

    # Read CSV file
    csv.field_size_limit(sys.maxsize)
    validations = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            validations.append(dict(row))

    assert len(validations) > 0, "CSV file is empty"
    assert 'reviewer_id' in validations[0], "CSV must contain reviewer_id column"

    # Connect to database
    conn = duckdb.connect(db_path)

    # Group validations by (reviewer_id) to delete existing ones
    groups = {}
    for v in validations:
        key = v['reviewer_id']
        if key not in groups:
            groups[key] = []
        groups[key].append(v)

    # Check for existing validations for each group
    for reviewer_id, group_validations in groups.items():
        encounter_ids = list(set(v['encounter_id'] for v in group_validations))
        if encounter_ids:
            placeholders = ','.join(['?' for _ in encounter_ids])
            result = conn.execute(
                f"SELECT * FROM validations WHERE encounter_id IN ({placeholders}) AND prompt_id = ? AND reviewer_id = ?",
                #f"DELETE FROM validations WHERE encounter_id IN ({placeholders}) AND prompt_id = ? AND reviewer_id = ?",
                encounter_ids + [prompt_id, reviewer_id]
            )
            assert len(result.df()) == 0

    # Get next ID
    next_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM validations").fetchone()
    next_id = next_id_result[0] if next_id_result else 1

    # Insert validations
    for i, validation in enumerate(validations):
        llm_confidence = validation.get('LLM_confidence')
        if llm_confidence in ['N/A', '', None]:
            llm_confidence = None
        else:
            llm_confidence = float(llm_confidence)

        annotation = validation.get('annotation')
        if annotation in ['N/A', '', None]:
            annotation = None
        else:
            annotation = int(annotation)

        conn.execute("""
            INSERT INTO validations
            (id, encounter_id, reason_text, LLM_confidence, annotation, reviewer_notes,
             feedback_type, reason_id, prompt_id, reviewer_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id + i,
            validation['encounter_id'],
            validation.get('reason_text'),
            llm_confidence,
            annotation,
            validation.get('reviewer_notes'),
            validation.get('feedback_type'),
            validation.get('reason_id'),
            prompt_id,
            validation['reviewer_id'],
        ])

    conn.close()

    return len(validations)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload annotations from CSV to the annotations database"
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to CSV file exported from web interface"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the DuckDB database file"
    )

    args = parser.parse_args()

    count = upload_annotations(
        prompt_id=args.prompt_id,
        input_csv_path=args.input_csv,
        db_path=args.db_path,
    )

    print(f"Uploaded {count} annotations to database")


if __name__ == "__main__":
    main()
