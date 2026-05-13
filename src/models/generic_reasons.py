"""Pydantic model for generic reason extraction."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from .gantt_models import GanttChart, GanttChartBase


class ReasonBase(BaseModel):
    """Individual reason identified from clinical notes (without confidence score)."""

    reason: str = Field(
        description="Concise description of the identified reason using standard medical terminology"
    )

    category: Literal["clinical", "operational", "social"] = Field(
        description="Category of reason: 'clinical' for medical/physiological factors, 'operational' for system/process/coordination factors, 'social' for social determinants of health"
    )

    explanation_support: str = Field(
        description="Detailed step-by-step reasoning for why this was identified as a contributing factor that lengthened LOS, referencing both the Gantt chart timeline and clinical notes"
    )

    explanation_contrary: str = Field(
        description="Explanations for why this factor may NOT have lengthened LOS (contrary evidence or alternative interpretations)"
    )

    relevant_quotes: str = Field(
        description="Exact quote from the note supporting this reason. Word-for-word, under 200 characters."
    )

    process_improvement: Optional[str] = Field(
        default=None,
        description="Specific process change that could have shortened LOS. Focus on timing (when could action have started earlier?) and workflow changes within the hospital's control."
    )


class Reason(ReasonBase):
    """Individual reason identified from clinical notes with inline confidence."""

    confidence: int = Field(
        description="Confidence level on probability scale (0-100)"
    )


class ReasonsOnly(BaseModel):
    """Structured model for extracting reasons when Gantt chart is provided separately.

    Use this in the two-stage extraction pipeline where Gantt chart is extracted first,
    then reasons are extracted given the Gantt chart as context.
    """

    reasons: List[ReasonBase] = Field(
        default_factory=list,
        description="List of distinct reasons identified by analyzing both the Gantt chart timeline and clinical notes, with categories and supporting/contrary explanations"
    )


class GenericReasonsBase(BaseModel):
    """Structured model for extracting reasons from clinical notes (without inline confidence).

    Use this when confidence scoring is done in a separate step.
    Uses simplified GanttChartBase (no category/time_uncertainty on events).
    """

    gantt_chart: GanttChartBase = Field(
        description="Step 1: Gantt chart mapping the patient journey with key events and timeline relevant to the QI metric"
    )

    reasons: List[ReasonBase] = Field(
        default_factory=list,
        description="Step 2: List of distinct reasons identified by analyzing both the Gantt chart timeline and clinical notes, with categories and supporting/contrary explanations"
    )


class GenericReasons(BaseModel):
    """Structured model for extracting reasons from clinical notes using two-step process.

    This model supports a two-step QI analysis approach:
    1. First step: Generate Gantt chart representing patient journey
    2. Second step: Use Gantt chart + clinical notes to identify contributing reasons with confidence

    This can work for any type of reason extraction (readmission, length of stay, etc.).
    """

    gantt_chart: GanttChart = Field(
        description="Step 1: Gantt chart mapping the patient journey with key events and timeline relevant to the QI metric"
    )

    reasons: List[Reason] = Field(
        default_factory=list,
        description="Step 2: List of distinct reasons identified by analyzing both the Gantt chart timeline and clinical notes, with categories, supporting/contrary explanations, and confidence"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "gantt_chart": {
                        "index_admission_summary": "65-year-old admitted with heart failure exacerbation and volume overload",
                        "readmission_summary": "Returned with worsening dyspnea and lower extremity edema",
                        "events": [
                            {
                                "event_id": 1,
                                "label": "Emergency Department",
                                "category": "admission",
                                "description": "Initial evaluation for shortness of breath",
                                "start_time": "2024-01-10 09:15",
                                "end_time": "2024-01-10 15:30",
                                "relevant_quotes": "Patient presents with acute dyspnea and lower extremity edema"
                            },
                            {
                                "event_id": 2,
                                "label": "IV Diuretic Treatment",
                                "category": "treatment",
                                "description": "Furosemide administration for fluid overload",
                                "start_time": "2024-01-10 16:00",
                                "end_time": "2024-01-12 08:00",
                                "relevant_quotes": "Started on IV furosemide 40mg BID"
                            },
                        ],
                    },
                    "reasons": [
                        {
                            "reason": "late initiation of outpatient imaging scheduling",
                            "category": "operational",
                            "explanation_support": "Based on the Gantt chart timeline, patient was medically ready for discharge but had to wait 3 days for outpatient imaging. The MRI was not scheduled until after medical readiness was established.",
                            "explanation_contrary": "A possible contrary reason is that outpatient imaging slots may have been limited regardless of when the order was placed.",
                            "relevant_quotes": "Patient ready for discharge but pending MRI... MRI completed",
                            "process_improvement": "Outpatient MRI could have been scheduled 2-3 days earlier when discharge trajectory became clear, reducing wait time.",
                            "confidence": 90,
                        },
                    ],
                }
            ]
        }
