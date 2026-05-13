"""Pydantic model for readmission reason extraction."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ReadmissionReasons(BaseModel):
    """Structured model for extracting readmission reasons from clinical notes.

    Attributes:
        has_reasons_mentioned: Whether the note explicitly mentions reasons for readmission
        reasons: List of distinct clinical reasons for readmission
        confidence: Confidence level in the extraction (high/medium/low)
        relevant_quotes: Brief quote from note supporting the extraction
    """

    has_reasons_mentioned: bool = Field(
        description="Whether the note explicitly mentions reason(s) for readmission"
    )

    reasons: List[str] = Field(
        default_factory=list,
        description="List of distinct clinical reasons for readmission mentioned in the note. "
                   "Be specific but concise (e.g., 'respiratory failure' not 'breathing problems')"
    )

    confidence: str = Field(
        description="Confidence in extraction: 'high' (explicit mention), "
                   "'medium' (implied/inferred), or 'low' (unclear/ambiguous)"
    )

    relevant_quotes: Optional[str] = Field(
        default=None,
        description="Exact quote from the note supporting the extracted reasons. "
                   "Keep under 200 characters."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "has_reasons_mentioned": True,
                    "reasons": ["respiratory failure", "septic shock"],
                    "confidence": "high",
                    "relevant_quotes": "Readmitted to ICU for respiratory failure secondary to pneumonia and septic shock"
                },
                {
                    "has_reasons_mentioned": False,
                    "reasons": [],
                    "confidence": "high",
                    "relevant_quotes": None
                }
            ]
        }
