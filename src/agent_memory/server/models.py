"""Pydantic request/response models for the agent-memory HTTP server."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateMemoryRequest(BaseModel):
    """Request body for POST /memories."""

    content: str
    layer: str = "episodic"
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "agent_inference"
    safety_critical: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response body representing a single memory entry."""

    memory_id: str
    content: str
    layer: str
    importance_score: float
    freshness_score: float
    source: str
    safety_critical: bool
    composite_score: float
    importance_level: str
    created_at: str
    last_accessed: str
    access_count: int
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response body for GET /search."""

    query: str
    results: list[MemoryResponse] = Field(default_factory=list)
    total: int = 0


class DeleteMemoryResponse(BaseModel):
    """Response body for DELETE /memories/{id}."""

    memory_id: str
    deleted: bool


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    service: str = "agent-memory"
    version: str = "0.1.0"
    memory_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str = ""


__all__ = [
    "CreateMemoryRequest",
    "MemoryResponse",
    "SearchResponse",
    "DeleteMemoryResponse",
    "HealthResponse",
    "ErrorResponse",
]
