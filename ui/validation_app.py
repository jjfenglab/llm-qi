#!/usr/bin/env python3
"""
Dynamic Flask validation app for readmission reasons.
Replaces static HTML generation with real-time API-driven interface.
"""

import os
import io
import json
import csv
import sys
import base64
import duckdb
import logging
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import Bayesian inference functionality directly
from common import deduplicate_note
from bayesian_inference import BayesianNonparamCalibration, BayesianCalibration, BayesianNonparamProbCalibration, BayesianMonotoneCalibration, BayesianMonotoneOrderedLogisticCalibration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ValidationDataManager:
    """Manages encounter data and validation results."""

    def __init__(self, calibration_model: str = "linear"):
        """
        Initialize ValidationDataManager.

        Args:
            calibration_model: Which calibration model to use.
                - "linear": Bayesian linear regression (beta_0 + beta_1 * x)
                - "nonparam": Nonparametric delta model (separate expected value per level)
        """
        self.db_path = None  # Set later via configure()
        self.assembled_notes = []
        self.extracted_reasons = []
        self.clustered_reasons = []
        self.gantt_charts = {}
        self.reviewer_ids = []
        self.exclude_encounters = []
        self.fix_shared_encounters = []
        self.num_read_all = 0
        self.reviewer_encounters = {}  # Maps reviewer_id -> list of encounter_ids
        self.calibration_model = calibration_model

        # Initialize Bayesian calibration based on model choice
        if calibration_model == "nonparam":
            self.bayesian_calibration = BayesianNonparamCalibration("src/bayesian_delta_model.stan")
            logger.info("Using nonparametric delta calibration model (discrete 0-3)")
        elif calibration_model == "nonparam_prob":
            self.bayesian_calibration = BayesianNonparamProbCalibration("src/bayesian_delta_cts_model.stan")
            logger.info("Using nonparametric piecewise constant calibration model (0-100)")
        elif calibration_model == "monotone":
            self.bayesian_calibration = BayesianMonotoneCalibration("src/bayesian_monotone_model.stan")
            logger.info("Using monotone calibration model (0-100, enforces monotonicity)")
        elif calibration_model == "monotone_ordinal":
            self.bayesian_calibration = BayesianMonotoneOrderedLogisticCalibration("src/bayesian_monotone_ordered_logistic.stan")
            logger.info("Using monotone ordered logistic calibration model (0-100 → ordinal 1-5)")
        else:  # default to linear
            self.bayesian_calibration = BayesianCalibration("src/bayesian_linear_regression.stan")
            logger.info("Using linear regression calibration model")

        self.bayesian_calibration.compile_model()
        logger.info("Bayesian calibration model compiled successfully")
        # Note: Database is initialized later via configure() once db_path is known

    def _init_database(self):
        """Initialize DuckDB database with validation schema."""
        assert self.db_path is not None, "db_path must be set before initializing database"
        conn = duckdb.connect(self.db_path)

        # Create validation table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validations (
                id BIGINT PRIMARY KEY,
                encounter_id VARCHAR NOT NULL,
                reason_text TEXT,
                LLM_confidence FLOAT,
                annotation INTEGER,  -- 1 for accepted, 0 for rejected, NULL for pending
                reviewer_notes TEXT,
                feedback_type VARCHAR,  -- 'reason' or 'gantt'
                reason_id VARCHAR,      -- For linking to specific reasons (encounter_id_index)
                prompt_id VARCHAR,      -- Unique identifier for this prompt/experiment
                reviewer_id VARCHAR,    -- Identifier for the reviewer
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_encounter_id ON validations(encounter_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reason_id ON validations(reason_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_id ON validations(prompt_id)")

        conn.close()
        logger.info(f"Initialized validation database: {self.db_path}")

    def load_data_from_csvs(self, assembled_notes_csv: str, extracted_reasons_csv: str,
            clustered_reasons_csv: str = None, gantt_json: str = None):
        """Load data from CSV files."""
        # Load assembled notes
        assert Path(assembled_notes_csv).exists(), f"Assembled notes CSV not found: {assembled_notes_csv}"
        self.assembled_notes = self._read_csv_to_dict(assembled_notes_csv)
        assert len(self.assembled_notes) > 0, "Assembled notes CSV is empty"
        assert 'Y' in self.assembled_notes[0], f"Missing 'Y' column in assembled notes. Available columns: {list(self.assembled_notes[0].keys())}"
        logger.info(f"Loaded {len(self.assembled_notes)} assembled notes")

        # Load extracted reasons
        assert Path(extracted_reasons_csv).exists(), f"Extracted reasons CSV not found: {extracted_reasons_csv}"
        raw_extracted = self._read_csv_to_dict(extracted_reasons_csv)
        assert len(raw_extracted) > 0, "Extracted reasons CSV is empty"
        self.extracted_reasons = self._group_reasons_by_encounter(raw_extracted)
        logger.info(f"Loaded {len(self.extracted_reasons)} extracted reason encounters")

        # Load clustered reasons (optional)
        if clustered_reasons_csv and Path(clustered_reasons_csv).exists():
            self.clustered_reasons = self._read_csv_to_dict(clustered_reasons_csv)
            logger.info(f"Loaded {len(self.clustered_reasons)} clustered reasons")

        # Load gantt charts (optional)
        if gantt_json and Path(gantt_json).exists():
            with open(gantt_json, 'r', encoding='utf-8') as f:
                gantt_data = json.load(f)
                if isinstance(gantt_data, dict):
                    self.gantt_charts = gantt_data
                elif isinstance(gantt_data, list):
                    self.gantt_charts = {chart['encounter_id']: chart for chart in gantt_data if 'encounter_id' in chart and chart['encounter_id'] not in self.exclude_encounters}
            logger.info(f"Loaded {len(self.gantt_charts)} gantt charts")

    def _read_csv_to_dict(self, file_path: str) -> List[Dict]:
        """Read CSV file and return list of dictionaries."""
        csv.field_size_limit(sys.maxsize)
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))

        return data

    def _group_reasons_by_encounter(self, extracted_reasons: List[Dict]) -> List[Dict]:
        """Group flattened reason rows back into encounter-level data."""
        # Build lookup for Y values from assembled notes
        y_lookup = {n['encounter_id']: n.get('Y') for n in self.assembled_notes}

        encounters = {}

        for row in extracted_reasons:
            encounter_id = row['encounter_id']
            if encounter_id in self.exclude_encounters:
                continue

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

        return list(encounters.values())

    def assign_encounters_to_reviewers(self):
        """Randomly assign encounters to reviewers evenly using deterministic seed.

        First selects num_read_all encounters for ALL reviewers to review,
        then distributes remaining encounters evenly across reviewers.
        """
        import random

        assert len(self.reviewer_ids) > 0, "No reviewer IDs configured"
        assert len(self.extracted_reasons) > 0, "No encounters loaded"

        # Get all encounter IDs
        encounter_ids = [e['encounter_id'] for e in self.extracted_reasons if e not in self.exclude_encounters]

        assert self.num_read_all <= len(encounter_ids), \
            f"num_read_all ({self.num_read_all}) exceeds total encounters ({len(encounter_ids)})"

        # First num_read_all encounters are assigned to ALL reviewers
        shuffled_ids = encounter_ids.copy()
        if self.fix_shared_encounters:
            read_all_ids = self.fix_shared_encounters
        else:
            # Use deterministic seed based on prompt_id for reproducibility
            # Used this for readmission
            seed = int(hashlib.md5(self.prompt_id.encode()).hexdigest(), 16) % (2**32)
            # Used this for LOS
            #seed = int(hashlib.md5(("exp_los_v7").encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
    
            # Shuffle encounter IDs
            rng.shuffle(shuffled_ids)

            read_all_ids = shuffled_ids[:self.num_read_all]

        remaining_ids = [enc_id for enc_id in shuffled_ids if enc_id not in read_all_ids]

        # Initialize reviewer encounters with read-all encounters
        if self.add_reviewer_ids:
            self.reviewer_encounters = {reviewer_id: list(read_all_ids) for reviewer_id in self.add_reviewer_ids}
            num_to_review = len(remaining_ids) // len(self.add_reviewer_ids)
            assert len(remaining_ids) % len(self.add_reviewer_ids) == 0
            for idx, reviewer_id in enumerate(self.add_reviewer_ids):
                self.reviewer_encounters[reviewer_id] += remaining_ids[idx * num_to_review: (idx + 1) * num_to_review]
            logger.info(f"Assigned {len(encounter_ids)} encounters to {self.add_reviewer_ids} reviewers")
        else:
            self.reviewer_encounters = {reviewer_id: list(read_all_ids) for reviewer_id in self.reviewer_ids}

            # Distribute remaining encounters evenly across reviewers
            for i, encounter_id in enumerate(remaining_ids):
                reviewer_id = self.reviewer_ids[i % len(self.reviewer_ids)]
                self.reviewer_encounters[reviewer_id].append(encounter_id)

            logger.info(f"Assigned {len(encounter_ids)} encounters to {len(self.reviewer_ids)} reviewers")
        logger.info(f"  - {self.num_read_all} encounters assigned to ALL reviewers: {read_all_ids}")
        logger.info(f"  - {len(remaining_ids)} encounters distributed individually")

    def get_encounters_for_reviewer(self, reviewer_id: str) -> Optional[List[Dict]]:
        """Get encounters assigned to a specific reviewer."""
        if reviewer_id not in self.reviewer_encounters:
            return None

        encounter_ids = self.reviewer_encounters[reviewer_id]
        encounters = []
        for encounter_id in encounter_ids:
            encounter = next((e for e in self.extracted_reasons if e['encounter_id'] == encounter_id), None)
            if encounter:
                encounter_copy = encounter.copy()
                encounter_copy['validation_status'] = self.get_encounter_validation_status(encounter_id)
                encounters.append(encounter_copy)

        return encounters

    def get_encounters(self, limit: int = 12, offset: int = 0, has_reasons: Optional[bool] = None) -> Dict:
        """Get encounters with optional filtering."""
        encounters = self.extracted_reasons

        # Filter by whether encounters have reasons
        if has_reasons is not None:
            encounters = [e for e in encounters if (e.get('has_reasons_mentioned') == 'True') == has_reasons]

        # Get total count
        total = len(encounters)

        # Apply pagination
        paginated = encounters[offset:offset + limit] if limit > 0 else encounters[offset:]

        # Enrich with validation status
        for encounter in paginated:
            encounter['validation_status'] = self.get_encounter_validation_status(encounter['encounter_id'])

        return {
            'encounters': paginated,
            'total': total,
            'offset': offset,
            'limit': limit,
            'has_more': offset + len(paginated) < total
        }

    def get_encounter_details(self, encounter_id: str) -> Optional[Dict]:
        """Get full details for a specific encounter."""
        # Get extracted reason data
        extracted = next((e for e in self.extracted_reasons if e['encounter_id'] == encounter_id), None)
        if not extracted:
            return None

        # Get assembled note data
        note = next((n for n in self.assembled_notes if n['encounter_id'] == encounter_id), None)

        # Process note text: convert 4 contiguous spaces to newlines for readability
        if note and note.get('note_text'):
            note = dict(note)  # Make a copy to avoid modifying original
            note['note_text'] = note['note_text'].replace('    ', '\n')
            #note['note_text'] = deduplicate_note(note['note_text'])

        # Get gantt chart data
        gantt = self.gantt_charts.get(encounter_id)

        # Get existing validations
        validations = self.get_encounter_validations(encounter_id)

        return {
            'encounter': extracted,
            'note': note,
            'gantt': gantt,
            'validations': validations
        }

    def get_encounter_validations(self, encounter_id: str) -> Dict:
        """Get existing validations for an encounter."""
        conn = duckdb.connect(self.db_path)

        results = conn.execute("""
            SELECT reason_id, annotation, reviewer_notes, created_at, feedback_type
            FROM validations
            WHERE encounter_id = ?
        """, [encounter_id]).fetchall()

        conn.close()

        validations = {}
        for row in results:
            reason_id, annotation, notes, created_at, feedback_type = row
            validations[reason_id or feedback_type] = {
                'status': 'accepted' if annotation == 1 else 'rejected' if annotation == 0 else None,
                'notes': notes or '',
                'timestamp': created_at
            }

        return validations

    def get_encounter_validation_status(self, encounter_id: str) -> Dict:
        """Get validation completion status for an encounter."""
        extracted = next((e for e in self.extracted_reasons if e['encounter_id'] == encounter_id), None)
        if not extracted or extracted.get('has_reasons_mentioned') != 'True':
            return {'completed': 0, 'total': 0}

        total_reasons = len(extracted.get('reasons', []))

        conn = duckdb.connect(self.db_path)
        completed = conn.execute("""
            SELECT COUNT(*) FROM validations
            WHERE encounter_id = ? AND feedback_type = 'reason' AND annotation IS NOT NULL
        """, [encounter_id]).fetchone()[0]
        conn.close()

        return {'completed': completed, 'total': total_reasons}

    def export_validations(self) -> List[Dict]:
        """Export all validations."""
        conn = duckdb.connect(self.db_path)

        results = conn.execute("""
            SELECT encounter_id, reason_text, LLM_confidence, annotation, reviewer_notes,
                   feedback_type, reason_id, prompt_id, reviewer_id, created_at
            FROM validations
            WHERE annotation IS NOT NULL
            ORDER BY encounter_id, created_at
        """).fetchall()

        conn.close()

        export_data = []
        for row in results:
            export_data.append({
                'encounter_id': row[0],
                'reason_text': row[1],
                'LLM_confidence': row[2],
                'annotation': row[3],
                'reviewer_notes': row[4],
                'feedback_type': row[5],
                'reason_id': row[6],
                'prompt_id': row[7],
                'reviewer_id': row[8],
                'created_at': row[9]
            })

        return export_data

    def run_bayesian_inference(self, prompt_id: str = None) -> Dict:
        """Run Bayesian inference on validated annotations for a specific prompt_id."""
        if prompt_id is None:
            prompt_id = getattr(self, 'prompt_id', None)

        if prompt_id is None:
            return {
                "status": "error",
                "message": "No prompt_id provided"
            }

        conn = duckdb.connect(self.db_path)

        # Get all annotations for this prompt_id - filter for Likert scale annotations (1-3)
        results = conn.execute("""
            SELECT LLM_confidence, annotation
            FROM validations
            WHERE feedback_type = 'reason'
              AND LLM_confidence IS NOT NULL
              AND prompt_id = ?
        """, [prompt_id]).fetchall()

        conn.close()

        if len(results) < 2:
            return {
                "status": "insufficient_data",
                "n_observations": len(results),
                "message": "Need at least 2 annotations for inference"
            }

        # Extract data arrays with proper numpy type conversion
        llm_scores = np.array([row[0] for row in results]).astype(float)
        human_scores = np.array([row[1] for row in results]).astype(float)

        # Run inference using the BayesianCalibration class
        logger.info(f"Running MCMC with {len(results)} reason annotations for prompt {prompt_id}")
        inference_results = self.bayesian_calibration.fit(
            confidences=llm_scores,
            annotations=human_scores,
            chains=4,
            iter_sampling=1000,
            iter_warmup=1000,
            show_progress=False
        )

        if inference_results["status"] != "success":
            return inference_results

        # Get expected scores at each LLM level (posterior samples)
        expected_scores = self.bayesian_calibration.get_expected_scores_at_levels()

        # Convert numpy arrays to lists for JSON serialization
        expected_scores_serializable = {
            k: v.tolist() for k, v in expected_scores.items()
        }

        return {
            "status": "success",
            "n_observations": inference_results["n_observations"],
            "n_samples": inference_results["n_samples"],
            "summary": inference_results["summary"],
            "expected_scores": expected_scores_serializable
        }

    def is_encounter_fully_annotated(self, encounter_id: str) -> bool:
        """Check if all reasons in an encounter have been validated."""
        status = self.get_encounter_validation_status(encounter_id)
        return status['total'] > 0 and status['completed'] == status['total']


# Initialize data manager (will be re-initialized if CLI args specify different model)
data_manager = None

# API Routes
@app.route('/api/retrieve', methods=['POST'])
def retrieve_encounters():
    """Retrieve encounters for a specific reviewer."""
    data = request.get_json()
    reviewer_id = data.get('reviewer_id')

    if not reviewer_id:
        return jsonify({'success': False, 'error': 'No reviewer_id provided'}), 400

    encounters = data_manager.get_encounters_for_reviewer(reviewer_id)

    if encounters is None:
        return jsonify({'success': False, 'error': f'Invalid reviewer ID: {reviewer_id}'}), 400

    return jsonify({
        'success': True,
        'encounters': encounters,
        'total': len(encounters)
    })

@app.route('/api/encounters')
def get_encounters():
    """Get encounters with pagination."""
    limit = int(request.args.get('limit', 20))
    offset = int(request.args.get('offset', 0))
    has_reasons = request.args.get('has_reasons')

    # Convert has_reasons to boolean if provided
    if has_reasons is not None:
        has_reasons = has_reasons.lower() == 'true'

    result = data_manager.get_encounters(limit=limit, offset=offset, has_reasons=has_reasons)
    return jsonify(result)

@app.route('/api/encounters/<encounter_id>')
def get_encounter_details(encounter_id):
    """Get full details for a specific encounter."""
    details = data_manager.get_encounter_details(encounter_id)
    if not details:
        return jsonify({'error': 'Encounter not found'}), 404

    return jsonify(details)

@app.route('/api/export')
def export_validations():
    """Export all validations as CSV data."""
    validations = data_manager.export_validations()
    return jsonify({
        'data': validations,
        'count': len(validations),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def get_validation_stats():
    """Get overall validation statistics."""
    conn = duckdb.connect(data_manager.db_path)

    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT encounter_id) as total_encounters,
            COUNT(*) as total_validations,
            COUNT(CASE WHEN annotation = 1 THEN 1 END) as accepted_reasons,
            COUNT(CASE WHEN annotation = 0 THEN 1 END) as rejected_reasons,
            COUNT(CASE WHEN feedback_type = 'gantt' THEN 1 END) as gantt_comments
        FROM validations
        WHERE annotation IS NOT NULL
    """).fetchone()

    conn.close()

    return jsonify({
        'total_encounters': stats[0],
        'total_validations': stats[1],
        'accepted_reasons': stats[2],
        'rejected_reasons': stats[3],
        'gantt_comments': stats[4]
    })

@app.route('/api/calibration')
def get_calibration_results():
    """Get Bayesian calibration results with violinplot as base64 PNG."""
    inference_result = data_manager.run_bayesian_inference()

    # Generate plot
    if inference_result['status'] != 'success':
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, inference_result.get('message', 'No calibration data available'),
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel('LLM Confidence Score (0-3)', fontsize=12)
        ax.set_ylabel('Expected Human Score (posterior)', fontsize=12)
        ax.set_title('Bayesian Calibration', fontsize=14, fontweight='bold')
    else:
        data_manager.bayesian_calibration.plot_calibration_curves(
            title="Bayesian Calibration: LLM vs Human Scores"
        )

    # Save plot to bytes buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close('all')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Add plot to response
    inference_result['plot_base64'] = plot_base64

    return jsonify(inference_result)

@app.route('/api/submit_validations', methods=['POST'])
def submit_validations():
    """Submit all validations from frontend and trigger Bayesian inference."""
    data = request.get_json()
    validations = data.get('validations', [])
    prompt_id = data.get('prompt_id')
    reviewer_id = data.get('reviewer_id')

    if not validations:
        return jsonify({'success': False, 'error': 'No validations provided'}), 400

    if not prompt_id:
        return jsonify({'success': False, 'error': 'No prompt_id provided'}), 400

    if not reviewer_id:
        return jsonify({'success': False, 'error': 'No reviewer_id provided'}), 400

    # Process and store validations
    conn = duckdb.connect(data_manager.db_path)

    # Delete existing validations for encounters being exported with this prompt_id and reviewer_id
    encounter_ids = list(set(v['encounter_id'] for v in validations))
    if encounter_ids:
        placeholders = ','.join(['?' for _ in encounter_ids])
        conn.execute(
            f"DELETE FROM validations WHERE encounter_id IN ({placeholders}) AND prompt_id = ? AND reviewer_id = ?",
            encounter_ids + [prompt_id, reviewer_id]
        )

    # Get next ID
    next_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM validations").fetchone()
    next_id = next_id_result[0] if next_id_result else 1

    for i, validation in enumerate(validations):
        conn.execute("""
            INSERT INTO validations
            (id, encounter_id, reason_text, LLM_confidence, annotation, reviewer_notes,
             feedback_type, reason_id, prompt_id, reviewer_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id + i,
            validation['encounter_id'],
            validation['reason_text'],
            validation['LLM_confidence'] if validation['LLM_confidence'] != 'N/A' else None,
            int(validation['annotation']) if validation['annotation'] not in ['N/A', None] else None,
            validation['reviewer_notes'],
            validation['feedback_type'],
            validation.get('reason_id'),
            prompt_id,
            reviewer_id
        ])

    conn.close()

    # Run Bayesian inference for this prompt_id
    inference_result = data_manager.run_bayesian_inference(prompt_id=prompt_id)

    return jsonify({
        'success': True,
        'message': f'Submitted {len(validations)} validations successfully for prompt_id={prompt_id}',
        'inference_result': inference_result
    })

# Web Routes
@app.route('/')
def index():
    """Serve the main validation interface."""
    return render_template('validation_interface.html', prompt_id=data_manager.prompt_id, rating_statement_completion=data_manager.rating_statement_completion)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

# Configuration and startup
def configure_app(assembled_notes_csv: str, extracted_reasons_csv: str,
                 clustered_reasons_csv: str = None, gantt_json: str = None,
                 prompt_id: str = None, reviewer_ids: List[str] = None, 
                 num_read_all: int = 0, db_path: str = None,
                 calibration_model: str = "linear",
                 rating_statement_completion: str = None):
    """Configure the app with data sources.

    Args:
        calibration_model: Which calibration model to use ("linear" or "nonparam")
        rating_statement_completion: Text to complete the rating statement
    """
    global data_manager

    assert prompt_id is not None, "prompt_id is required"
    assert reviewer_ids is not None and len(reviewer_ids) > 0, "reviewer_ids is required"
    assert num_read_all >= 0, "num_read_all must be non-negative"
    assert db_path is not None, "db_path is required"
    assert calibration_model in ["linear", "nonparam", "nonparam_prob", "monotone", "monotone_ordinal"], f"calibration_model must be 'linear', 'nonparam', 'nonparam_prob', 'monotone', or 'monotone_ordinal', got '{calibration_model}'"
    assert rating_statement_completion is not None, "rating_statement_completion is required"

    # Initialize data manager with the specified calibration model
    data_manager = ValidationDataManager(calibration_model=calibration_model)

    # Store prompt_id, reviewer_ids, num_read_all, and db_path in data_manager
    data_manager.prompt_id = prompt_id
    data_manager.reviewer_ids = reviewer_ids
    data_manager.num_read_all = num_read_all
    data_manager.db_path = db_path
    data_manager.rating_statement_completion = rating_statement_completion

    # Initialize the database now that we have the path
    data_manager._init_database()

    logger.info(f"Loading validation data from CSV files for prompt_id={prompt_id}...")
    data_manager.load_data_from_csvs(
        assembled_notes_csv=assembled_notes_csv,
        extracted_reasons_csv=extracted_reasons_csv,
        clustered_reasons_csv=clustered_reasons_csv,
        gantt_json=gantt_json
    )

    # Assign encounters to reviewers
    data_manager.assign_encounters_to_reviewers()

    logger.info(f"Validation app configured successfully")
    logger.info(f"- Prompt ID: {prompt_id}")
    logger.info(f"- Calibration model: {calibration_model}")
    logger.info(f"- Reviewer IDs: {', '.join(reviewer_ids)}")
    logger.info(f"- Num read-all: {num_read_all}")
    logger.info(f"- Database: {db_path}")
    logger.info(f"- {len(data_manager.assembled_notes)} assembled notes")
    logger.info(f"- {len(data_manager.extracted_reasons)} extracted reason encounters")
    logger.info(f"- {len(data_manager.clustered_reasons)} clustered reasons")
    logger.info(f"- {len(data_manager.gantt_charts)} gantt charts")
    for reviewer_id in reviewer_ids:
        count = len(data_manager.reviewer_encounters.get(reviewer_id, []))
        logger.info(f"- Reviewer '{reviewer_id}': {count} encounters")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Flask validation app")
    parser.add_argument('--assembled-notes-csv', required=True, help='Path to assembled notes CSV')
    parser.add_argument('--extracted-reasons-csv', required=True, help='Path to extracted reasons CSV')
    parser.add_argument('--clustered-reasons-csv', help='Path to clustered reasons CSV')
    parser.add_argument('--gantt-json', help='Path to gantt charts JSON')
    parser.add_argument('--port', type=int, default=5123, help='Port to run on (default: 5123)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--prompt-id', required=True, help='Unique identifier for this prompt/experiment')
    parser.add_argument('--reviewer-ids', nargs='+', required=True, help='List of reviewer IDs for random encounter assignment')
    parser.add_argument('--num-read-all', type=int, default=0, help='Number of observations to be reviewed by ALL reviewers')
    parser.add_argument('--db-path', required=True, help='Path to the DuckDB database file for storing validations')
    parser.add_argument('--calibration-model', choices=['linear', 'nonparam', 'nonparam_prob', 'monotone', 'monotone_ordinal'], default='monotone',
                        help='Calibration model: "linear" (beta_0 + beta_1*x), "nonparam" (discrete delta model 0-3), "nonparam_prob" (piecewise constant 0-100), "monotone" (monotonic effect model), or "monotone_ordinal" (monotone ordered logistic 1-4)')
    parser.add_argument('--rating-statement-completion', required=True,
                        help='Text to complete the rating statement "This contributing factor is a modifiable gap that if improved would ___."')

    args = parser.parse_args()

    # Configure the app with data (will raise AssertionError on failure)
    configure_app(
        assembled_notes_csv=args.assembled_notes_csv,
        extracted_reasons_csv=args.extracted_reasons_csv,
        clustered_reasons_csv=args.clustered_reasons_csv,
        gantt_json=args.gantt_json,
        prompt_id=args.prompt_id,
        reviewer_ids=args.reviewer_ids,
        num_read_all=args.num_read_all,
        db_path=args.db_path,
        calibration_model=args.calibration_model,
        rating_statement_completion=args.rating_statement_completion
    )

    logger.info(f"Starting validation app on {args.host}:{args.port} with prompt_id={args.prompt_id}, calibration_model={args.calibration_model}")
    app.run(host=args.host, port=args.port, debug=args.debug)
