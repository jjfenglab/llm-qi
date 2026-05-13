"""Pydantic model for filtering notes based on LLM assessment."""

from pydantic import BaseModel, Field
from typing import Literal


class FilterResult(BaseModel):
    """Result of filtering assessment for a clinical note."""

    keep: Literal[0, 1] = Field(
        description="Whether to keep this observation: 1 = keep (meets criteria), 0 = exclude (does not meet criteria)"
    )

    explanation: str = Field(
        description="Brief explanation for the filtering decision"
    )

    quote: str = Field(
        description="Exact quote from the note supporting this decision. Word-for-word, under 200 characters."
    )
