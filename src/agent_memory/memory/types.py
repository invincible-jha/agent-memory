"""Core memory types: enumerations and the MemoryEntry dataclass."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class MemoryLayer(str, Enum):
    """The four cognitive memory layers."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemorySource(str, Enum):
    """Origin of a memory entry."""

    USER_INPUT = "user_input"
    TOOL_OUTPUT = "tool_output"
    AGENT_INFERENCE = "agent_inference"
    DOCUMENT = "document"
    EXTERNAL_API = "external_api"


class ImportanceLevel(str, Enum):
    """Qualitative importance tier."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"

    @staticmethod
    def from_score(score: float) -> "ImportanceLevel":
        """Map a numeric score [0, 1] to an ImportanceLevel."""
        if score >= 0.9:
            return ImportanceLevel.CRITICAL
        if score >= 0.7:
            return ImportanceLevel.HIGH
        if score >= 0.5:
            return ImportanceLevel.MEDIUM
        if score >= 0.3:
            return ImportanceLevel.LOW
        return ImportanceLevel.TRIVIAL


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class MemoryEntry(BaseModel):
    """A single memory record shared across all storage layers."""

    memory_id: str = Field(default_factory=_new_id)
    content: str
    layer: MemoryLayer
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.AGENT_INFERENCE
    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed: datetime = Field(default_factory=_utcnow)
    access_count: int = Field(default=0, ge=0)
    safety_critical: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def composite_score(self) -> float:
        """Combined relevance proxy: importance * freshness."""
        return round(self.importance_score * self.freshness_score, 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def importance_level(self) -> ImportanceLevel:
        """Qualitative tier for this entry's importance score."""
        return ImportanceLevel.from_score(self.importance_score)

    def touch(self) -> "MemoryEntry":
        """Return a new entry with updated access timestamp and count."""
        return self.model_copy(
            update={
                "last_accessed": _utcnow(),
                "access_count": self.access_count + 1,
            }
        )

    model_config = {"frozen": False}


__all__ = [
    "MemoryEntry",
    "MemoryLayer",
    "MemorySource",
    "ImportanceLevel",
]
