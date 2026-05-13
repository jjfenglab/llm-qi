#!/usr/bin/env python3
"""
Startup script for the dynamic validation app.
Automatically detects available CSV files and starts the Flask server.

Supports two modes:
1. Server mode (default): Starts Flask server for interactive validation
2. Export mode (--export): Generates standalone HTML with embedded data
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Import ValidationDataManager to reuse encounter assignment logic
sys.path.append(str(Path(__file__).parent))
from validation_app import ValidationDataManager

def validate_file_exists(file_path, file_type):
    """Validate that a file exists and is readable."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ {file_type} not found: {file_path}")
        return False
    if not path.is_file():
        print(f"❌ {file_type} is not a file: {file_path}")
        return False
    print(f"✓ {file_type}: {file_path}")
    return True


def read_csv_to_dict(file_path: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    csv.field_size_limit(sys.maxsize)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


def load_data_for_export(assembled_notes_csv: str, extracted_reasons_csv: str,
                         gantt_json: str = None) -> Dict[str, Any]:
    """Load all data needed for export."""
    # Load assembled notes
    assembled_notes = read_csv_to_dict(assembled_notes_csv)
    notes_by_id = {n['encounter_id']: n for n in assembled_notes}

    # Load extracted reasons and group by encounter
    raw_extracted = read_csv_to_dict(extracted_reasons_csv)
    y_lookup = {n['encounter_id']: n.get('Y') for n in assembled_notes}

    encounters = {}
    for row in raw_extracted:
        encounter_id = row['encounter_id']

        if encounter_id not in encounters:
            encounters[encounter_id] = {
                'encounter_id': encounter_id,
                'has_reasons_mentioned': 'False',
                'extraction_failed': row.get('extraction_failed', 'False'),
                'confidence': 0.0,
                'relevant_excerpt': '',
                'reasons': [],
                'reason_details': [],
                'Y': y_lookup.get(encounter_id),
            }

        # If this row has a reason, add it
        if row.get('reason_text') and str(row['reason_text']).strip() not in ['', 'None']:
            encounters[encounter_id]['has_reasons_mentioned'] = 'True'

            # Update confidence to highest reason confidence
            reason_confidence = float(row.get('confidence', 0.0)) if row.get('confidence') else 0.0
            encounters[encounter_id]['confidence'] = max(encounters[encounter_id]['confidence'], reason_confidence)

            # Build relevant excerpt
            quote = row.get('relevant_quotes', '')
            if quote and quote.strip():
                current_excerpt = encounters[encounter_id]['relevant_excerpt']
                if current_excerpt:
                    encounters[encounter_id]['relevant_excerpt'] += ' ... ' + quote.strip()
                else:
                    encounters[encounter_id]['relevant_excerpt'] = quote.strip()

            # Add reason
            encounters[encounter_id]['reasons'].append(row['reason_text'])
            encounters[encounter_id]['reason_details'].append({
                'reason': row['reason_text'],
                'category': row.get('category', ''),
                'confidence': row.get('confidence', ''),
                'confidence_reason': row.get('confidence_reason', ''),
                'explanation_support': row.get('explanation_support', ''),
                'explanation_contrary': row.get('explanation_contrary', ''),
                'process_improvement': row.get('process_improvement', ''),
                'relevant_quotes': row.get('relevant_quotes', '')
            })

    extracted_reasons = list(encounters.values())

    # Load gantt charts
    gantt_charts = {}
    if gantt_json and Path(gantt_json).exists():
        with open(gantt_json, 'r', encoding='utf-8') as f:
            gantt_data = json.load(f)
            if isinstance(gantt_data, dict):
                gantt_charts = gantt_data
            elif isinstance(gantt_data, list):
                gantt_charts = {chart['encounter_id']: chart for chart in gantt_data if 'encounter_id' in chart}

    # Build full encounter details (pre-fetched)
    encounter_details = {}
    for encounter in extracted_reasons:
        encounter_id = encounter['encounter_id']
        note = notes_by_id.get(encounter_id, {})

        # Process note text: convert 4 contiguous spaces to newlines
        if note and note.get('note_text'):
            note = dict(note)
            note['note_text'] = note['note_text'].replace('    ', '\n')

        gantt = gantt_charts.get(encounter_id)

        encounter_details[encounter_id] = {
            'encounter': encounter,
            'note': note,
            'gantt': gantt,
            'validations': {}
        }

    return {
        'encounters': extracted_reasons,
        'encounter_details': encounter_details,
        'total': len(extracted_reasons)
    }


def generate_export_html(data: Dict[str, Any], prompt_id: str, output_path: str, reviewer_id: str = None, rating_statement_completion: str = None):
    """Generate standalone HTML file with embedded data."""
    template_path = Path(__file__).parent / 'templates' / 'validation_interface.html'
    assert template_path.exists(), f"Template not found: {template_path}"

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Create the embedded data script
    reviewer_info = f" (Reviewer: {reviewer_id})" if reviewer_id else ""
    embedded_data_script = f"""
    <script>
        // Embedded data for standalone export mode{reviewer_info}
        const EMBEDDED_MODE = true;
        const EMBEDDED_DATA = {json.dumps(data, indent=2)};
        const EMBEDDED_REVIEWER_ID = '{reviewer_id or 'embedded'}';
    </script>
    """

    # Insert embedded data before the main script section
    # Find the <script> tag that contains "// Global state"
    insert_marker = '<script>\n        // Global state'
    assert insert_marker in template, "Could not find insertion point in template"

    template = template.replace(insert_marker, embedded_data_script + '\n    ' + insert_marker)

    # Replace the prompt_id template variable
    template = template.replace("{{ prompt_id }}", prompt_id)

    # Replace the rating statement completion placeholder
    assert rating_statement_completion is not None, "rating_statement_completion is required"
    template = template.replace("{{ rating_statement_completion }}", rating_statement_completion)

    # Write the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"✓ Generated standalone HTML: {output_path}")

def check_stan_model(calibration_model: str):
    """Check if Stan model file exists for the specified calibration model."""
    model_paths = {
        'linear': 'src/bayesian_linear_regression.stan',
        'nonparam': 'src/bayesian_delta_model.stan',
        'nonparam_prob': 'src/bayesian_delta_cts_model.stan',
        'monotone': 'src/bayesian_monotone_model.stan',
        'monotone_ordinal': 'src/bayesian_monotone_ordered_logistic.stan',
    }
    stan_model_path = Path(model_paths.get(calibration_model, model_paths['nonparam_prob']))
    assert stan_model_path.exists(), f"Stan model not found: {stan_model_path}"
    print(f"✓ Stan model ({calibration_model}): {stan_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Start the dynamic validation app")
    parser.add_argument('--assembled-notes-csv', required=True,
                       help='Path to assembled notes CSV file')
    parser.add_argument('--extracted-reasons-csv', required=True,
                       help='Path to extracted reasons CSV file')
    parser.add_argument('--clustered-reasons-csv',
                       help='Path to clustered reasons CSV file (optional)')
    parser.add_argument('--gantt-json',
                       help='Path to gantt charts JSON file (optional)')
    parser.add_argument('--port', type=int, default=5123,
                       help='Port to run the validation app on (default: 5123)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--prompt-id', required=True,
                       help='Unique identifier for this prompt/experiment (used to isolate validation data)')
    parser.add_argument('--reviewer-ids', nargs='+',
                       help='List of reviewer IDs for random encounter assignment (required for server mode)')
    parser.add_argument('--add-reviewer-ids', nargs='+',
            help='If adding: List of new reviewer IDs for random encounter assignment')
    parser.add_argument('--exclude-encounters', nargs='+',
            help='If adding: which encounters to exclude')
    parser.add_argument('--fix-shared-encounters', nargs='+',
            help='If adding: which encounters are shared')
    parser.add_argument('--num-read-all', type=int,
                       help='Number of observations to be reviewed by ALL reviewers (required for server mode)')
    parser.add_argument('--calibration-model', type=str,
                       choices=['linear', 'nonparam', 'nonparam_prob', 'monotone', 'monotone_ordinal'],
                       default='monotone',
                       help='Calibration model: "linear", "nonparam" (discrete 0-3), "nonparam_prob" (piecewise 0-100), "monotone" (monotonic effect), or "monotone_ordinal" (monotone ordered logistic 1-4)')
    parser.add_argument('--db-path',
                       help='Path to the DuckDB database file for storing validations (required for server mode)')
    parser.add_argument('--export',
                       help='Export standalone HTML file with embedded data (skips server mode)')
    parser.add_argument('--rating-statement-completion', required=True,
                       help='Text to complete the rating statement "This contributing factor is a modifiable gap that if improved would ___."')

    args = parser.parse_args()

    # Handle export mode
    if args.export:
        # Validate required arguments for reviewer-specific export
        if args.reviewer_ids is None or len(args.reviewer_ids) == 0:
            print("❌ Error: --reviewer-ids is required when using --export")
            print("   Please specify the reviewer IDs to generate separate HTML files for each reviewer.")
            sys.exit(1)

        if args.num_read_all is None:
            print("❌ Error: --num-read-all is required when using --export")
            print("   Please specify the number of encounters to be reviewed by ALL reviewers.")
            sys.exit(1)

        print("📦 Export Mode: Generating reviewer-specific HTML files")
        print("=" * 50)

        # Validate input files
        print("Validating input files:")
        files_valid = True
        files_valid &= validate_file_exists(args.assembled_notes_csv, "Assembled notes CSV")
        files_valid &= validate_file_exists(args.extracted_reasons_csv, "Extracted reasons CSV")

        if args.gantt_json:
            files_valid &= validate_file_exists(args.gantt_json, "Gantt charts JSON")

        if not files_valid:
            print("\n❌ Some required files are missing or invalid.")
            sys.exit(1)

        print()
        print("Setting up encounter assignments...")

        # Create ValidationDataManager to handle encounter assignment
        data_manager = ValidationDataManager(calibration_model=args.calibration_model)

        # Set required attributes (following the pattern from configure_app)
        data_manager.prompt_id = args.prompt_id
        data_manager.reviewer_ids = args.reviewer_ids if len(args.add_reviewer_ids) == 0 else args.add_reviewer_ids
        data_manager.add_reviewer_ids = args.add_reviewer_ids
        data_manager.exclude_encounters = args.exclude_encounters
        data_manager.fix_shared_encounters = args.fix_shared_encounters
        data_manager.num_read_all = args.num_read_all
        # Skip database initialization for export mode

        # Load data from CSV files
        data_manager.load_data_from_csvs(
            assembled_notes_csv=args.assembled_notes_csv,
            extracted_reasons_csv=args.extracted_reasons_csv,
            clustered_reasons_csv=args.clustered_reasons_csv,
            gantt_json=args.gantt_json
        )
        full_data = load_data_for_export(
            assembled_notes_csv=args.assembled_notes_csv,
            extracted_reasons_csv=args.extracted_reasons_csv,
            gantt_json=args.gantt_json
        )

        # Assign encounters to reviewers using existing logic
        data_manager.assign_encounters_to_reviewers()

        print(f"  - {len(data_manager.extracted_reasons)} total encounters")
        print(f"  - {args.num_read_all} encounters assigned to ALL reviewers")
        print(f"  - {len(data_manager.reviewer_ids)} reviewers: {', '.join(data_manager.reviewer_ids)}")

        # Generate HTML file for each reviewer
        print()
        print("Generating reviewer-specific HTML files...")
        base_path = Path(args.export)
        base_name = base_path.stem
        base_dir = base_path.parent
        extension = base_path.suffix or '.html'

        generated_files = []
        for reviewer_id in data_manager.reviewer_ids:
            print(f"  - Processing reviewer: {reviewer_id}")

            # Get encounter IDs assigned to this reviewer (bypass validation status lookup)
            if reviewer_id not in data_manager.reviewer_encounters:
                print(f"    ⚠️  No encounters found for reviewer {reviewer_id}")
                continue

            reviewer_encounter_ids = data_manager.reviewer_encounters[reviewer_id]
            if not reviewer_encounter_ids:
                print(f"    ⚠️  No encounters assigned to reviewer {reviewer_id}")
                continue

            # Filter data to include only this reviewer's encounters
            # Iterate in order of reviewer_encounter_ids to preserve read_all first, then individual
            encounters_by_id = {e['encounter_id']: e for e in full_data['encounters']}
            filtered_encounters = []
            filtered_encounter_details = {}

            for encounter_id in reviewer_encounter_ids:
                if encounter_id in encounters_by_id:
                    filtered_encounters.append(encounters_by_id[encounter_id])
                    if encounter_id in full_data['encounter_details']:
                        filtered_encounter_details[encounter_id] = full_data['encounter_details'][encounter_id]

            reviewer_data = {
                'encounters': filtered_encounters,
                'encounter_details': filtered_encounter_details,
                'total': len(filtered_encounters)
            }

            # Generate filename for this reviewer
            reviewer_filename = f"{base_name}_reviewer_{reviewer_id}{extension}"
            reviewer_path = base_dir / reviewer_filename

            # Generate HTML file
            generate_export_html(reviewer_data, args.prompt_id, str(reviewer_path), reviewer_id, args.rating_statement_completion)
            generated_files.append(str(reviewer_path))

            print(f"    ✓ {len(filtered_encounters)} encounters → {reviewer_filename}")

        print()
        print(f"📋 Prompt ID: {args.prompt_id}")
        print(f"📂 Generated {len(generated_files)} HTML files:")
        for file_path in generated_files:
            print(f"   📄 {file_path}")
        print()
        print("Open any HTML file in a browser to use the validation interface.")
        print("Validations will be stored in browser localStorage and can be exported as JSON.")
        return

    # Server mode - validate required arguments
    assert args.reviewer_ids is not None and len(args.reviewer_ids) > 0, \
        "--reviewer-ids is required for server mode"
    assert args.num_read_all is not None, \
        "--num-read-all is required for server mode"
    assert args.db_path is not None, \
        "--db-path is required for server mode"

    print("🚀 Starting Dynamic Validation App")
    print("=" * 50)

    # Check Stan model for Bayesian inference
    check_stan_model(args.calibration_model)
    print()

    # Validate required CSV files
    print("Validating input files:")
    files_valid = True

    files_valid &= validate_file_exists(args.assembled_notes_csv, "Assembled notes CSV")
    files_valid &= validate_file_exists(args.extracted_reasons_csv, "Extracted reasons CSV")

    # Check optional files
    if args.clustered_reasons_csv:
        files_valid &= validate_file_exists(args.clustered_reasons_csv, "Clustered reasons CSV")

    if args.gantt_json:
        files_valid &= validate_file_exists(args.gantt_json, "Gantt charts JSON")

    if not files_valid:
        print("\n❌ Some required files are missing or invalid.")
        print("Please check the file paths and try again.")
        sys.exit(1)

    print()

    # Build command arguments
    cmd = [
        sys.executable,
        'ui/validation_app.py',
        '--assembled-notes-csv', args.assembled_notes_csv,
        '--extracted-reasons-csv', args.extracted_reasons_csv,
        '--port', str(args.port),
        '--host', args.host
    ]

    if args.clustered_reasons_csv:
        cmd.extend(['--clustered-reasons-csv', args.clustered_reasons_csv])

    if args.gantt_json:
        cmd.extend(['--gantt-json', args.gantt_json])

    if args.debug:
        cmd.append('--debug')

    cmd.extend(['--prompt-id', args.prompt_id])
    cmd.extend(['--reviewer-ids'] + list(sorted(args.reviewer_ids)))
    cmd.extend(['--num-read-all', str(args.num_read_all)])
    cmd.extend(['--db-path', args.db_path])
    cmd.extend(['--calibration-model', args.calibration_model])
    cmd.extend(['--rating-statement-completion', args.rating_statement_completion])

    print(f"📋 Prompt ID: {args.prompt_id}")
    print(f"👥 Reviewer IDs: {', '.join(args.reviewer_ids)}")
    print(f"📖 Num read-all: {args.num_read_all}")
    print(f"💾 Database: {args.db_path}")
    print(f"📈 Calibration model: {args.calibration_model}")
    assert len(args.reviewer_ids) >= 1
    print("Starting validation app with command:")
    print(" ".join(cmd))
    print()
    print(f"📡 Validation app (with integrated Bayesian inference) will run on: http://{args.host}:{args.port}")
    print(f"🔗 For port tunneling: ssh -L {args.port}:localhost:{args.port} user@aws-instance")
    print(f"💻 Then visit: http://localhost:{args.port}")
    print(f"📊 Calibration results: http://localhost:{args.port}/api/calibration")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        # Change to the project root directory
        os.chdir(Path(__file__).parent.parent)
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Validation app stopped")
    except Exception as e:
        print(f"❌ Error starting validation app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
