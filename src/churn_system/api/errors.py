"""Structured API error payloads (stable contracts for clients)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """JSON body for 4xx/5xx responses."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable message")
    detail: str | None = Field(None, description="Extra context (e.g. validation)")
