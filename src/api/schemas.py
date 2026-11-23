"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="What the agent should do in the workspace.")
    session_id: str | None = Field(None, description="Continue an earlier session instead of starting fresh.")


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class RunResponse(BaseModel):
    status: str
    final_text: str
    steps: int
    session_id: str
    usage: UsageOut


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    workspace: str
