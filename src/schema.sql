-- Simple DuckDB schema for Bayesian calibration system
CREATE TABLE annotations (
    model_id VARCHAR NOT NULL,
    observation_id VARCHAR NOT NULL,
    reason_id VARCHAR NOT NULL,
    llm_confidence DOUBLE NOT NULL,
    annotation BOOLEAN, -- NULL = not yet annotated, TRUE/FALSE = human annotation
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, observation_id, reason_id)
);

-- Index for fast lookups by model_id
CREATE INDEX idx_annotations_model_id ON annotations(model_id);