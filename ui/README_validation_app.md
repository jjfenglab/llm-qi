# Dynamic Validation App

A Flask-based web application for validating readmission reasons with real-time Bayesian inference integration.

## Overview

This app replaces the static HTML validation interface (`generate_validation_page.py`) with a dynamic, API-driven interface that:

- **Loads encounters dynamically** via API calls (batches of 10)
- **Stores validation results** in DuckDB database using the same schema as CSV exports
- **Runs integrated Bayesian inference** directly in the same Flask server
- **Automatically triggers inference** after completing encounter annotations
- **Provides real-time updates** of validation statistics and calibration curves

## Architecture

### Components
- **`validation_app.py`**: Main Flask application with API endpoints
- **`templates/validation_interface.html`**: Single-page application frontend
- **`start_validation_app.py`**: Startup script with auto-detection of CSV files
- **`validation.db`**: DuckDB database for storing validation results

### API Endpoints
- `GET /api/encounters` - Get encounters with pagination and filtering
- `GET /api/encounters/{id}` - Get full details for specific encounter
- `POST /api/encounters/{id}/validate` - Submit validation for reasons
- `GET /api/export` - Export validation results as CSV
- `GET /api/stats` - Get validation statistics
- `GET /api/calibration` - Get latest Bayesian calibration results and curves

### Database Schema

The validation database uses the same schema as the original CSV export:

```sql
CREATE TABLE validations (
    id INTEGER PRIMARY KEY,
    encounter_id VARCHAR NOT NULL,
    reason_text TEXT,
    LLM_confidence FLOAT,
    relevant_excerpt TEXT,
    topic_name VARCHAR,
    topic_probability FLOAT,
    topic_terms TEXT,
    annotation INTEGER,  -- 1 for accepted, 0 for rejected, NULL for pending
    reviewer_notes TEXT,
    validation_timestamp TIMESTAMP,
    note_type VARCHAR,
    note_timestamp TIMESTAMP,
    feedback_type VARCHAR,  -- 'reason' or 'gantt'
    reason_id VARCHAR,      -- encounter_id_index format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Setup

### 1. Install Dependencies

The app uses Flask and DuckDB (should already be installed in your environment):

```bash
pip install flask duckdb
```

### 2. Start Validation App

Use the startup script with explicit file paths:

```bash
cd <repo-root>

# Find your CSV files first (example paths)
find . -name "*assembled*notes*.csv" -type f
find . -name "*extracted*reasons*.csv" -type f
find . -name "*clustered*reasons*.csv" -type f
find . -name "*gantt*.json" -type f

# Then start the app with actual paths
python ui/start_validation_app.py \
    --assembled-notes-csv exp_readmission/assembled_notes.csv \
    --extracted-reasons-csv exp_readmission/extracted_reasons.csv \
    --clustered-reasons-csv exp_readmission/clustered_reasons.csv \
    --gantt-json exp_readmission/gantt_charts.json
```

This will:
- Validate that all specified files exist
- Start the validation app on port 5123
- Show tunnel commands for accessing from your laptop

### Manual Startup

You can also start the app manually:

```bash
python ui/validation_app.py \
    --assembled-notes-csv path/to/assembled_notes.csv \
    --extracted-reasons-csv path/to/extracted_reasons.csv \
    --clustered-reasons-csv path/to/clustered_reasons.csv \
    --gantt-json path/to/gantt_charts.json \
    --port 5123
```

**Required arguments:**
- `--assembled-notes-csv`: Path to assembled clinical notes CSV
- `--extracted-reasons-csv`: Path to extracted reasons CSV

**Optional arguments:**
- `--clustered-reasons-csv`: Path to clustered reasons CSV (for topic info)
- `--gantt-json`: Path to gantt charts JSON (for timelines)
- `--port`: Port to run on (default: 5123)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--debug`: Enable debug mode

## Usage

### Access the Interface

1. **Local (on AWS machine)**: http://localhost:5123
2. **Via SSH tunnel**:
   ```bash
   ssh -L 5123:localhost:5123 user@your-aws-instance
   ```
   Then visit: http://localhost:5123

### Validation Workflow

1. **Filter encounters** by "All", "With Reasons", or "No Reasons"
2. **Navigate through pages** using pagination controls
3. **Select an encounter** to view details
4. **Validate reasons** by clicking Accept/Reject buttons
5. **Add notes** for individual reasons or timeline comments
6. **Automatic inference**: When all reasons in an encounter are validated, Bayesian inference runs automatically
7. **Export results** using the export button

### Features

- **Real-time statistics** showing validation progress
- **Batch loading** for efficient handling of large datasets
- **Validation persistence** in database (survives page refreshes)
- **CSV export** compatible with existing analysis pipelines
- **Automatic inference triggering** when encounters are completed
- **Responsive design** works on different screen sizes

## Integration with Existing Workflow

### Data Sources
The app reads the same CSV files as the static validation page:
- `assembled_notes.csv` - Clinical notes and metadata
- `extracted_reasons.csv` - Flattened reason extraction results
- `clustered_reasons.csv` - Topic clustering results (optional)
- `gantt_charts.json` - Patient timeline visualizations (optional)

### Export Compatibility
The validation export format is identical to the static page exports, ensuring compatibility with existing analysis scripts.

### Bayesian Inference
The app includes integrated Bayesian inference using the `BayesianCalibration` class from `src/bayesian_inference.py`. When encounters are fully annotated, inference runs automatically to provide real-time calibration updates.

## Development

### Adding Features

To add new features:

1. **Backend**: Add API endpoints in `validation_app.py`
2. **Frontend**: Update `templates/validation_interface.html`
3. **Database**: Add columns to the `validations` table schema

### Debugging

Run with debug mode:
```bash
python ui/validation_app.py --debug [other args]
```

This enables:
- Flask debug mode with auto-reload
- Detailed error messages
- Request logging

## Migration from Static Workflow

The app is designed to eventually replace the static HTML generation:

1. **Phase 1**: Run alongside existing workflow for testing
2. **Phase 2**: Migrate validation workflows to dynamic app
3. **Phase 3**: Deprecate static HTML generation

The database and export format ensure seamless migration of existing validation results.

## Troubleshooting

### Common Issues

**"File not found" errors**
- Double-check all file paths are correct and absolute
- Ensure files exist and are readable
- Use tab completion or `ls` to verify paths

**"Bayesian inference model not found"**
- Ensure `src/bayesian_logistic_regression.stan` exists
- Check that PyStan/CmdStanPy dependencies are installed

**"Can't connect from laptop"**
- Verify SSH tunnel: `ssh -L 5123:localhost:5123 user@aws`
- Try different ports if 5123 is in use
- Check firewall settings

**Database errors**
- Delete `ui/validation.db` to reset database
- Check file permissions on the database

### Performance

For large datasets (>1000 encounters):
- The app loads encounters in batches of 10
- Database queries are indexed for performance
- Consider increasing batch size if needed

## Future Enhancements

Potential improvements:
- Real-time calibration curve visualization
- Multi-user support with user authentication
- Advanced filtering and search capabilities
- Integration with additional inference models
- Export to additional formats (Excel, JSON)
- Validation workflow templates