"""Simplified Pydantic models for QI-focused Gantt chart generation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import re


class GanttEventBase(BaseModel):
    """Individual event/phase in the patient journey (simplified, without category/time_uncertainty)."""

    event_id: int = Field(
        description="Unique identifier for this event"
    )

    label: str = Field(
        description="Brief descriptive label for the event (e.g., 'Emergency Department', 'Surgery', 'Discharge Planning')"
    )

    description: str = Field(
        description="Detailed description of what happened during this phase"
    )

    start_time: str = Field(
        description="Start time of the event in format YYYY-MM-DD HH:MM (use best estimate if exact time unavailable)"
    )

    end_time: str = Field(
        description="End time of the event in format YYYY-MM-DD HH:MM (use best estimate if exact time unavailable)"
    )

    relevant_quotes: Optional[str] = Field(
        default=None,
        description="Supporting quote from clinical notes for this event, under 200 characters"
    )

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Validate timestamp matches YYYY-MM-DD HH:MM format."""
        pattern = r'^\d{4}-\d{2}-\d{2}\s+(?:24:00|\d{2}:\d{2})$'
        if not re.match(pattern, v.strip()):
            raise ValueError(f"Timestamp must be 'YYYY-MM-DD HH:MM', got: {v}")
        return v.strip()


class GanttEvent(GanttEventBase):
    """Individual event/phase in the patient journey for QI analysis (with category/time_uncertainty)."""

    category: str = Field(
        description="Category of event: 'admission' for initial care, 'treatment' for active medical care, 'procedure' for specific interventions, 'waiting' for delays, 'coordination' for care transitions, 'discharge' for final phases, 'out of hospital' for home care, SNF, etc."
    )

    time_uncertainty: Optional[str] = Field(
        default=None,
        description="Optional indicator if timestamps are estimated (e.g., 'estimated', 'approximate')"
    )


class GanttChartBase(BaseModel):
    """Complete Gantt chart representing the patient journey (simplified events)."""

    index_admission_summary: str = Field(
        description="Brief summary of the stated reason for patient's original (index) admission. Summarize only, no interpretation."
    )

    readmission_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the stated reason for patient's readmission. Summarize only, no interpretation. Optional - only used for readmission analysis."
    )

    events: List[GanttEventBase] = Field(
        description="List of key events/phases in chronological order. Focus on essential care phases and potential bottlenecks relevant to the QI metric."
    )


class GanttChart(BaseModel):
    """Complete Gantt chart representing the patient journey for QI analysis."""

    index_admission_summary: str = Field(
        description="Brief summary of the stated reason for patient's original (index) admission. Summarize only, no interpretation."
    )

    readmission_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of the stated reason for patient's readmission. Summarize only, no interpretation. Optional - only used for readmission analysis."
    )

    events: List[GanttEvent] = Field(
        description="List of key events/phases in chronological order. Focus on essential care phases and potential bottlenecks relevant to the QI metric."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "index_admission_summary": "72-year-old admitted for community-acquired pneumonia with COPD exacerbation",
                    "readmission_summary": "Returned with worsening dyspnea and hypoxia, found to have recurrent pneumonia",
                    "events": [
                        {
                            "event_id": 1,
                            "label": "Emergency Department",
                            "category": "admission",
                            "description": "Initial evaluation and stabilization",
                            "start_time": "2024-01-15 08:30",
                            "end_time": "2024-01-15 14:20",
                            "relevant_quotes": "Patient arrived via ambulance with acute dyspnea"
                        },
                        {
                            "event_id": 2,
                            "label": "IV Antibiotic Treatment",
                            "category": "treatment",
                            "description": "Intravenous ceftriaxone course for pneumonia",
                            "start_time": "2024-01-15 16:00",
                            "end_time": "2024-01-17 12:00",
                            "relevant_quotes": "Started on IV ceftriaxone for community-acquired pneumonia"
                        },
                        {
                            "event_id": 3,
                            "label": "Discharge Planning",
                            "category": "discharge",
                            "description": "Medication reconciliation and discharge coordination",
                            "start_time": "2024-01-17 10:00",
                            "end_time": "2024-01-17 16:30",
                            "relevant_quotes": "Patient ready for discharge pending medication reconciliation"
                        }
                    ],
                }
            ]
        }
