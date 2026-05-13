"""Phase 1: Assemble notes from the database for readmission encounters.

This script queries the DuckDB database for readmission encounters and extracts
specific note types from both index admission and readmission encounters:
- Discharge summary from the index admission
- Consult notes from the index admission (optional, via --include-consults)
- Care plan notes from the index admission (optional, via --include-care-plans)
- Discharge instructions from the index admission (optional, via --include-discharge-instructions)
- Patient instructions from the index admission (optional, via --include-patient-instructions)
- Outpatient encounter notes between discharge and readmission (optional, via --include-outpatient)
- Care plan notes from outpatient visits (optional, via --include-care-plans)
- ED Provider Note from the readmission
- H&P note from the readmission
- Consult notes from the readmission (optional, via --include-consults)
- Discharge summary from the readmission

Care plan note types include: Care Plan, Assessment & Plan Note, Care and Service Plan,
Discharge Plans, and Treatment Plan.

These notes are concatenated with bracketed headers for analysis.

Usage:
    python src/assemble_readmission_notes.py --source-db <path> --output-csv <path>
    python src/assemble_readmission_notes.py --source-db <path> --output-csv <path> --include-outpatient --include-consults --include-care-plans
"""

import argparse
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# Database utilities not needed for CSV output

def assemble_notes(
    source_db_path: str,
    output_csv_path: str,
    limit: int = None,
    min_note_length: int = 50,
    seed: int = 42,
    include_outpatient: bool = False,
    include_consults: bool = False,
    include_care_plans: bool = False,
    include_discharge_instructions: bool = False,
    include_patient_instructions: bool = False,
    diagnosis_cohorts: list = None,
) -> pd.DataFrame:
    """Assemble specific notes from source database.

    Extracts and concatenates the following notes per readmission encounter:
    - Discharge summary from the index admission
    - Consult notes from the index admission (if include_consults=True)
    - Care plan notes from index admission (if include_care_plans=True)
    - Discharge instructions from index admission (if include_discharge_instructions=True)
    - Patient instructions from index admission (if include_patient_instructions=True)
    - Outpatient encounter notes between discharge and readmission (if include_outpatient=True)
    - Care plan notes from outpatient visits (if include_care_plans=True)
    - ED Provider Note from the readmission
    - H&P note from the readmission
    - Consult notes from the readmission (if include_consults=True)
    - Discharge summary from the readmission

    Args:
        source_db_path: Path to the source DuckDB database with clinical notes
        output_csv_path: Path to the output CSV file
        limit: Optional limit on number of encounters to process (for testing)
        min_note_length: Minimum note length to include (default 50)
        seed: Random seed for shuffling (default 42)
        include_outpatient: Whether to include outpatient notes between discharge and readmission
        include_consults: Whether to include consult notes from index admission and readmission
        include_care_plans: Whether to include care plan notes from index admission and outpatient visits
        include_discharge_instructions: Whether to include discharge instructions from index admission
        include_patient_instructions: Whether to include patient instructions from index admission
        diagnosis_cohorts: Optional list of diagnosis cohorts to filter by (e.g., ['COPD', 'Heart Failure', 'AMI']).
            When specified, only includes readmissions where BOTH index_diagnosis_cohort AND
            readmission_diagnosis_cohort are in this list.

    Returns:
        DataFrame with assembled and concatenated notes
    """
    # Validate inputs
    assert Path(source_db_path).exists(), f"Source database not found: {source_db_path}"

    print(f"Connecting to source database: {source_db_path}")
    print(f"Include outpatient notes: {include_outpatient}")
    print(f"Include consult notes: {include_consults}")
    print(f"Include care plan notes: {include_care_plans}")
    print(f"Include discharge instructions: {include_discharge_instructions}")
    print(f"Include patient instructions: {include_patient_instructions}")
    print(f"Diagnosis cohort filter: {diagnosis_cohorts}")
    conn = duckdb.connect(source_db_path, read_only=True)

    # Build diagnosis cohort filter clause if specified
    diagnosis_cohort_filter = ""
    if diagnosis_cohorts:
        cohort_list = ", ".join(f"'{c}'" for c in diagnosis_cohorts)
        diagnosis_cohort_filter = f"""
                AND r.index_diagnosis_cohort IN ({cohort_list})"""

    # Build query to extract specific notes from both index admission and readmission encounters
    # Base inpatient notes: discharge summary from index, ED provider note, H&P, discharge summary from readmission
    # Optionally includes outpatient notes between discharge and readmission
    # Optionally includes consult notes from index admission and readmission

    # Consult notes from index admission (only included if include_consults=True)
    consults_index_query = f"""
            UNION ALL

            -- Consult notes from index admission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'consult_index' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.index_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.index_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'Consults'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
    """ if include_consults else ""

    ## Consult notes from readmission (only included if include_consults=True)
    #consults_readmission_query = f"""
    #        UNION ALL

    #        -- Consult notes from readmission
    #        SELECT
    #            r.readmission_csn_surrogate as encounter_id,
    #            DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
    #            'consult_readmission' as note_category,
    #            STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
    #            MIN(n.note_entry_dttm) as note_timestamp
    #        FROM readmissions r
    #        JOIN notes n ON r.readmission_csn_surrogate = n.pat_enc_csn_id_surrogate
    #        WHERE r.readmission_csn_surrogate IS NOT NULL
    #            AND r.zsfg_hospital_wide_in_numerator = '1'
    #            AND r.zsfg_hospital_wide_in_denominator = '1'
    #            AND n.note_text IS NOT NULL
    #            AND n.note_type = 'Consults'
    #            AND LENGTH(n.note_text) >= {min_note_length}
    #        GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
    #""" if include_consults else ""

    # Care plan note types to include
    CARE_PLAN_TYPES = (
        "'Care Plan'",
        "'Assessment & Plan Note'",
        "'Care and Service Plan'",
        "'Discharge Plans'",
        "'Treatment Plan'",
    )
    care_plan_types_sql = ", ".join(CARE_PLAN_TYPES)

    # Care plan notes from index admission (only included if include_care_plans=True)
    care_plans_index_query = f"""
            UNION ALL

            -- Care plan notes from index admission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'care_plan_index' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.index_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.index_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type IN ({care_plan_types_sql})
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
    """ if include_care_plans else ""

    # Discharge instructions from index admission (only included if include_discharge_instructions=True)
    discharge_instructions_query = f"""
            UNION ALL

            -- Discharge instructions from index admission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'discharge_instructions_index' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.index_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.index_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'Discharge Instructions'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
    """ if include_discharge_instructions else ""

    # Patient instructions from index admission (only included if include_patient_instructions=True)
    patient_instructions_query = f"""
            UNION ALL

            -- Patient instructions from index admission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'patient_instructions_index' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.index_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.index_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'Patient Instructions'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
    """ if include_patient_instructions else ""

    # Care plan notes from outpatient visits between discharge and readmission
    # Note: We don't include e.contact_date in the output since it can be unreliable
    care_plans_outpatient_query = f"""
            UNION ALL

            -- Care plan notes from outpatient visits between discharge and readmission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'care_plan_outpatient' as note_category,
                -- Include encounter metadata in the note text (excluding contact_date which can be unreliable)
                'Encounter Type: ' || COALESCE(e.encounter_type, 'Unknown') ||
                ' | Note Type: ' || COALESCE(n.note_type, 'Unknown') ||
                chr(10) || STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN encounters e ON r.pat_id_surrogate = e.pat_id_surrogate
            JOIN notes n ON e.pat_enc_csn_id_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND e.enc_patient_class = 'OP'
                AND e.contact_date > r.index_discharge_date::DATE
                AND e.contact_date < r.readmission_date::DATE
                AND n.note_text IS NOT NULL
                AND n.note_type IN ({care_plan_types_sql})
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, e.pat_enc_csn_id_surrogate, e.contact_date, e.encounter_type, n.note_id, n.note_type
    """ if include_care_plans else ""

    # Outpatient notes subquery (only included if include_outpatient=True)
    # Note: We don't include e.contact_date in the output since it can be unreliable
    # (encounter linkage issues). The note_entry_dttm is more accurate for the actual service date.
    outpatient_query = f"""
            UNION ALL

            -- Outpatient encounter notes between discharge and readmission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'outpatient_note' as note_category,
                -- Include encounter metadata in the note text (excluding contact_date which can be unreliable)
                'Encounter Type: ' || COALESCE(e.encounter_type, 'Unknown') ||
                ' | Note Type: ' || COALESCE(n.note_type, 'Unknown') ||
                chr(10) || STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN encounters e ON r.pat_id_surrogate = e.pat_id_surrogate
            JOIN notes n ON e.pat_enc_csn_id_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND e.enc_patient_class = 'OP'
                AND e.contact_date > r.index_discharge_date::DATE
                AND e.contact_date < r.readmission_date::DATE
                AND n.note_text IS NOT NULL
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, e.pat_enc_csn_id_surrogate, e.contact_date, e.encounter_type, n.note_id, n.note_type
    """ if include_outpatient else ""

    # Build list of note categories that should keep all notes (not just the first)
    multi_note_categories = []
    if include_outpatient:
        multi_note_categories.append('outpatient_note')
    if include_consults:
        multi_note_categories.extend(['consult_index', 'consult_readmission'])
    if include_care_plans:
        multi_note_categories.extend(['care_plan_index', 'care_plan_outpatient'])

    # Final select - handle multi-note categories separately (keep all, not ranked)
    if multi_note_categories:
        exclude_list = ", ".join(f"'{cat}'" for cat in multi_note_categories)
        include_list = ", ".join(f"'{cat}'" for cat in multi_note_categories)
        final_select = f"""
        ranked_notes AS (
            SELECT
                encounter_id,
                days_to_readmission,
                note_category,
                note_text,
                note_timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY encounter_id, note_category
                    ORDER BY note_timestamp
                ) as rn
            FROM note_extracts
            WHERE note_category NOT IN ({exclude_list})
        )
        -- Take the first (earliest) note of each category for single-note categories
        SELECT encounter_id, days_to_readmission, note_category, note_text, note_timestamp
        FROM ranked_notes
        WHERE rn = 1

        UNION ALL

        -- Keep all notes for multi-note categories (not ranked)
        SELECT encounter_id, days_to_readmission, note_category, note_text, note_timestamp
        FROM note_extracts
        WHERE note_category IN ({include_list})

        ORDER BY encounter_id, note_timestamp
        """
    else:
        final_select = """
        ranked_notes AS (
            SELECT
                encounter_id,
                days_to_readmission,
                note_category,
                note_text,
                note_timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY encounter_id, note_category
                    ORDER BY note_timestamp
                ) as rn
            FROM note_extracts
        )
        SELECT encounter_id, days_to_readmission, note_category, note_text, note_timestamp
        FROM ranked_notes
        WHERE rn = 1
        ORDER BY encounter_id, note_timestamp
        """

    query = f"""
        WITH note_extracts AS (
            -- Discharge summary from index admission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'discharge_summary_index' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.index_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.index_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'  -- Unplanned readmissions only
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'Discharge Summary'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
            {consults_index_query}
            {care_plans_index_query}
            {discharge_instructions_query}
            {patient_instructions_query}

            UNION ALL

            -- ED Provider Note from readmission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'ed_provider_note' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.readmission_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'ED Provider Notes'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id

            UNION ALL

            -- H&P note from readmission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'hp_note' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.readmission_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'H&P'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id

            UNION ALL

            -- Discharge summary from readmission
            SELECT
                r.readmission_csn_surrogate as encounter_id,
                DATE_DIFF('day', r.index_discharge_date::DATE, r.readmission_date::DATE) as days_to_readmission,
                'discharge_summary_readmission' as note_category,
                STRING_AGG(n.note_text, '' ORDER BY n.line) as note_text,
                MIN(n.note_entry_dttm) as note_timestamp
            FROM readmissions r
            JOIN notes n ON r.readmission_csn_surrogate = n.pat_enc_csn_id_surrogate
            WHERE r.readmission_csn_surrogate IS NOT NULL
                AND r.zsfg_hospital_wide_in_numerator = '1'
                AND r.zsfg_hospital_wide_in_denominator = '1'{diagnosis_cohort_filter}
                AND n.note_text IS NOT NULL
                AND n.note_type = 'Discharge Summary'
                AND LENGTH(n.note_text) >= {min_note_length}
            GROUP BY r.readmission_csn_surrogate, r.index_discharge_date, r.readmission_date, n.note_id
            {outpatient_query}
            {care_plans_outpatient_query}
        ),
        {final_select}
    """

    print("Executing query to fetch readmission notes...")
    df = conn.execute(query).fetchdf()
    conn.close()

    # Validate output
    assert len(df) > 0, "No notes found in source database"
    assert 'encounter_id' in df.columns, "Missing encounter_id column"
    assert 'note_category' in df.columns, "Missing note_category column"
    assert 'note_text' in df.columns, "Missing note_text column"
    assert df['encounter_id'].notna().all(), "Found null encounter_ids"
    assert df['note_text'].notna().all(), "Found null note_text"

    print(f"Found {len(df)} individual notes across {df['encounter_id'].nunique()} encounters")
    print("Note categories found:")
    print(df['note_category'].value_counts().to_string())

    # Concatenate notes per encounter with bracketed headers
    def format_note_header(timestamp, note_type_label):
        """Format note header as [<day of week> <date> <time> <note type>]."""
        if pd.notna(timestamp):
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            day_of_week = timestamp.strftime('%A')
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M')
            return f"[{day_of_week} {date_str} {time_str} {note_type_label}]"
        else:
            return f"[Unknown Date {note_type_label}]"

    def concatenate_notes_for_encounter(encounter_df):
        """Concatenate notes for a single encounter with bracketed headers, ordered by timestamp."""
        concatenated_parts = []

        # Define labels for note types
        note_type_labels = {
            'discharge_summary_index': 'Discharge Summary - Index Admission',
            'consult_index': 'Consult - Index Admission',
            'care_plan_index': 'Care Plan - Index Admission',
            'discharge_instructions_index': 'Discharge Instructions - Index Admission',
            'patient_instructions_index': 'Patient Instructions - Index Admission',
            'outpatient_note': 'Outpatient Note',
            'care_plan_outpatient': 'Care Plan - Outpatient',
            'ed_provider_note': 'ED Provider Note - Readmission',
            'hp_note': 'H&P - Readmission',
            'consult_readmission': 'Consult - Readmission',
            'discharge_summary_readmission': 'Discharge Summary - Readmission'
        }

        # Sort all notes by timestamp
        sorted_notes = encounter_df.sort_values('note_timestamp')

        for _, row in sorted_notes.iterrows():
            note_text = row['note_text']
            note_timestamp = row['note_timestamp']
            note_category = row['note_category']
            label = note_type_labels[note_category]

            header = format_note_header(note_timestamp, label)
            formatted_note = f"{header}\n{note_text}"
            concatenated_parts.append(formatted_note)

        return "\n\n".join(concatenated_parts)

    # Group by encounter and concatenate notes
    print("Concatenating notes per encounter...")
    assembled_notes = []
    for encounter_id, encounter_df in df.groupby('encounter_id'):
        concatenated_text = concatenate_notes_for_encounter(encounter_df)
        # days_to_readmission is same for all rows of an encounter
        days_to_readmission = encounter_df['days_to_readmission'].iloc[0]
        assembled_notes.append({
            'encounter_id': encounter_id,
            'note_text': concatenated_text,
            'note_count': len(encounter_df),
            'note_categories': ', '.join(sorted(encounter_df['note_category'].unique())),
            'Y': days_to_readmission,
        })

    # Convert to DataFrame
    df_final = pd.DataFrame(assembled_notes)

    print(f"\nAssembled {len(df_final)} encounter note sets")
    print(f"Average concatenated note length: {df_final['note_text'].str.len().mean():.0f} characters")
    print(f"Average notes per encounter: {df_final['note_count'].mean():.1f}")

    # Validate final output
    assert len(df_final) > 0, "No assembled notes created"
    duplicate_count = df_final['encounter_id'].duplicated().sum()
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate encounter_ids in final output"

    # Shuffle
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Shuffled with seed={seed}")

    # Apply limit if requested
    if limit is not None:
        df_final = df_final.head(limit)
        print(f"Applied limit: {len(df_final)} rows")

    # Save to CSV
    print(f"\nSaving to CSV file: {output_csv_path}")
    df_final.to_csv(output_csv_path, index=False)

    return df_final


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Assemble clinical notes for readmission analysis"
    )
    parser.add_argument(
        "--source-db",
        required=True,
        help="Path to source DuckDB database with clinical notes"
    )
    parser.add_argument(
        "--output-csv",
        default="data/assembled_notes.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of encounters (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--include-outpatient",
        action="store_true",
        help="Include outpatient encounter notes between discharge and readmission"
    )
    parser.add_argument(
        "--include-consults",
        action="store_true",
        help="Include consult notes from index admission and readmission"
    )
    parser.add_argument(
        "--include-care-plans",
        action="store_true",
        help="Include care plan notes from index admission, readmission, and outpatient visits"
    )
    parser.add_argument(
        "--include-instructions",
        action="store_true",
        help="Include discharge and patient instructions from index admission"
    )
    parser.add_argument(
        "--diagnosis-cohorts",
        nargs="+",
        default=None,
        help="Filter to specific diagnosis cohorts (e.g., 'COPD' 'Heart Failure' 'AMI'). "
             "Only includes readmissions where BOTH index and readmission diagnoses are in this list."
    )

    args = parser.parse_args()

    # Validate arguments
    assert args.source_db, "Source database path is required"

    # Create data directory if needed
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Run assembly
    df = assemble_notes(
        source_db_path=args.source_db,
        output_csv_path=args.output_csv,
        limit=args.limit,
        seed=args.seed,
        include_outpatient=args.include_outpatient,
        include_consults=args.include_consults,
        include_care_plans=args.include_care_plans,
        include_discharge_instructions=args.include_instructions,
        include_patient_instructions=args.include_instructions,
        diagnosis_cohorts=args.diagnosis_cohorts,
    )

    print(f"\n✓ Successfully assembled {len(df)} notes")
    print(f"  Output: {args.output_csv}")


if __name__ == "__main__":
    main()
