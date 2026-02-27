"""Contradiction detection data models.

Standalone frozen dataclasses used by the TF-IDF-based contradiction engine.
These are distinct from the Pydantic-based MemoryEntry in memory.types — they
are self-contained and carry an optional pre-computed embedding field to allow
callers to supply external vectors without a hard dependency on any embedding
library.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ContradictionType(str, Enum):
    """Semantic category of a detected contradiction between two memory entries.

    Values
    ------
    DIRECT_NEGATION
        One entry explicitly negates a claim made in the other (e.g. "X is Y"
        vs "X is not Y").
    TEMPORAL_INCONSISTENCY
        The entries agree topically but are separated by enough time that one
        is likely stale (e.g. a fact that changes over time).
    FACTUAL_CONFLICT
        The entries make incompatible factual claims about the same entity and
        attribute without explicit negation (e.g. "Alice lives in London" vs
        "Alice lives in Berlin").
    PREFERENCE_CHANGE
        A previously stated preference or stance has been superseded by a more
        recent one — not necessarily a factual error, but worth flagging.
    """

    DIRECT_NEGATION = "direct_negation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    FACTUAL_CONFLICT = "factual_conflict"
    PREFERENCE_CHANGE = "preference_change"


# ---------------------------------------------------------------------------
# Core data models (frozen dataclasses — immutable value objects)
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MemoryEntry:
    """A lightweight, immutable memory record used by the contradiction engine.

    This is intentionally separate from ``agent_memory.memory.types.MemoryEntry``
    to allow the contradiction subsystem to be used standalone without pulling
    in the full Pydantic model hierarchy.

    Parameters
    ----------
    id:
        Unique identifier for this memory record.  Auto-generated UUID if not
        supplied.
    content:
        The raw text content of the memory.
    timestamp:
        When the memory was created.  Defaults to the current UTC time.
    source:
        Opaque string identifying the origin of the memory (e.g. "user_input",
        "tool_output").  No validation is applied — any string is accepted.
    embedding:
        Optional pre-computed dense vector representation of the content.
        When supplied, the contradiction detector will use cosine similarity
        on this vector instead of computing TF-IDF on the fly.
    """

    content: str
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=_utcnow)
    source: str = "agent_inference"
    embedding: Optional[list[float]] = field(default=None, compare=False)


@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction between two memory entries.

    Parameters
    ----------
    entry_a:
        The first entry in the contradicting pair.
    entry_b:
        The second entry in the contradicting pair.
    similarity_score:
        Semantic similarity between the two entries in [0, 1].  Higher means
        the entries address the same topic more closely.
    contradiction_type:
        The category of contradiction detected.
    confidence:
        Model confidence in the contradiction classification, in [0, 1].
    explanation:
        Human-readable description of why the entries contradict each other.
    """

    entry_a: MemoryEntry
    entry_b: MemoryEntry
    similarity_score: float
    contradiction_type: ContradictionType
    confidence: float
    explanation: str


@dataclass(frozen=True)
class Resolution:
    """The outcome of applying a resolution strategy to a contradiction.

    Parameters
    ----------
    strategy_used:
        The name of the strategy that produced this resolution.
    kept_entries:
        The entries that should be retained after resolution.
    archived_entries:
        The entries that should be archived or soft-deleted.
    notes:
        Optional free-text notes describing the resolution rationale.
    """

    strategy_used: str
    kept_entries: tuple[MemoryEntry, ...]
    archived_entries: tuple[MemoryEntry, ...]
    notes: str = ""


__all__ = [
    "ContradictionType",
    "MemoryEntry",
    "Contradiction",
    "Resolution",
]
