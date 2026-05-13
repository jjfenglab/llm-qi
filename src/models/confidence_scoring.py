"""Pydantic model for confidence scoring of extracted reasons."""

from pydantic import BaseModel, Field
from typing import List


class ConfidenceScore(BaseModel):
    """Individual confidence score for a reason."""

    reason: str = Field(
        description="The reason text being scored (should match exactly with extracted reason)"
    )

    confidence: int = Field(
        description="Confidence probability (0-100) rounded to closest decile"
    )

    confidence_reason: str = Field(
        description="Detailed explanation for why this confidence score was assigned"
    )


class ConfidenceScores(BaseModel):
    """Collection of confidence scores for all reasons for a patient encounter."""

    confidences: List[ConfidenceScore] = Field(
        description="List of confidence scores for each reason, in the same order as presented"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "confidences": [
                        {
                            "reason": "late initiation of outpatient imaging scheduling",
                            "confidence": 90,
                            "confidence_reason": "Clear evidence of avoidable delay. Patient was medically ready for discharge but had to wait 3 days for outpatient MRI scheduling. This represents a clear process improvement opportunity that could reduce LOS."
                        },
                        {
                            "reason": "delayed specialist consultation",
                            "confidence": 80,
                            "confidence_reason": "Specialist consult was delayed by 2 days, though some clinical complexity may have justified the delay. Overall represents a moderate process improvement opportunity."
                        }
                    ]
                }
            ]
        }