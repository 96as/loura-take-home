from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["pending", "classified", "failed"]
Category = Literal["billing", "technical", "account", "other"]
Priority = Literal["low", "medium", "high"]

MAX_SUMMARY_CHARS = 200
# A sentence boundary: terminal punctuation, whitespace, then more text.
_SENTENCE_BREAK = re.compile(r"[.!?]\s+\S")


class TicketIn(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    subject: str = Field(default="", max_length=200)
    body: str = Field(max_length=10_000)

    @field_validator("body")
    @classmethod
    def _body_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("body must contain a non-whitespace character")
        return v


class Classification(BaseModel):
    """What the LLM must return. Unknown extra fields are ignored."""

    category: Category
    priority: Priority
    summary: str

    @field_validator("summary")
    @classmethod
    def _summary_rules(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("summary is empty")
        if len(v) > MAX_SUMMARY_CHARS:
            raise ValueError(f"summary longer than {MAX_SUMMARY_CHARS} characters")
        if _SENTENCE_BREAK.search(v):
            raise ValueError("summary is more than one sentence")
        return v


class TicketOut(BaseModel):
    id: str
    subject: str
    body: str
    status: Status
    category: Category | None = None
    priority: Priority | None = None
    summary: str | None = None
    attempts: int
    last_error: str | None = None
    suspected_injection: bool
    created_at: str
    updated_at: str


class TicketPage(BaseModel):
    items: list[TicketOut]
    next_cursor: str | None = None


class ReclassifyIn(BaseModel):
    target: Literal["failed", "stale_prompt"]


class ReclassifyOut(BaseModel):
    queued: int
