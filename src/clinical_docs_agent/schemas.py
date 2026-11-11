"""Structured output the model must return. The same Pydantic schema is sent to every provider."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    section: str = Field(description="Exact section heading from the label, e.g. 'Dosage and administration'")
    quote: str = Field(description="A short verbatim quote (under 40 words) from that section that supports the answer")


class LabelAnswer(BaseModel):
    answerable: bool = Field(description="False if the provided label sections do not answer the question")
    answer: str = Field(description="Concise answer based only on the label text, or a short explanation of why it can't be answered")
    citations: list[Citation] = Field(description="Supporting quotes; empty when answerable is false")
