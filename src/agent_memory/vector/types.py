"""Pydantic models for vector search entries and results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VectorEntry(BaseModel):
    """A stored embedding alongside its key and metadata.

    Parameters
    ----------
    key:
        Unique identifier for this vector (typically a ``memory_id``).
    vector:
        The embedding as a list of floats.
    metadata:
        Arbitrary key-value data associated with this vector.
    """

    key: str
    vector: list[float]
    metadata: dict[str, object] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """A single result from a vector similarity search.

    Parameters
    ----------
    key:
        The identifier of the matching vector entry.
    score:
        Cosine similarity in ``[0.0, 1.0]`` (or ``[-1.0, 1.0]`` for raw
        dot-product; implementations should clamp to ``[0.0, 1.0]``).
    metadata:
        Metadata copied from the matching :class:`VectorEntry`.
    """

    key: str
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


__all__ = ["VectorEntry", "VectorSearchResult"]
