"""Result ranker — multi-signal weighted scoring for memory search results."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from agent_memory.memory.types import MemoryEntry


@dataclass
class RankingWeights:
    """Weights for the four ranking signals.

    All weights must be non-negative.  They do not need to sum to 1;
    the final score is the weighted average normalised to [0, 1].
    """

    text_relevance: float = 0.4
    importance: float = 0.3
    freshness: float = 0.2
    provenance: float = 0.1

    def __post_init__(self) -> None:
        for name in ("text_relevance", "importance", "freshness", "provenance"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"Weight {name!r} must be >= 0, got {value}")

    @property
    def total(self) -> float:
        return (
            self.text_relevance
            + self.importance
            + self.freshness
            + self.provenance
        )


class ResultRanker:
    """Score memory entries using a weighted combination of signals.

    Signals
    -------
    text_relevance:
        TF-style token overlap between query and entry content, scaled to
        [0, 1] via a sigmoid-like mapping.
    importance:
        The entry's ``importance_score`` field [0, 1].
    freshness:
        The entry's ``freshness_score`` field [0, 1].
    provenance:
        Derived from ``provenance_reliability`` in the entry's metadata
        (stored by ProvenanceTracker), or from the source reliability
        heuristic if not present.

    Parameters
    ----------
    weights:
        Custom weights for each signal.  A ``RankingWeights`` with the
        standard defaults is used if not provided.
    """

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self._weights = weights or RankingWeights()

    @property
    def weights(self) -> RankingWeights:
        return self._weights

    def score(self, entry: MemoryEntry, query: str) -> float:
        """Compute a combined relevance score in [0, 1].

        Parameters
        ----------
        entry:
            The memory entry to score.
        query:
            The original search query.

        Returns
        -------
        float
            Weighted score in [0.0, 1.0].
        """
        weights = self._weights
        total_weight = weights.total
        if total_weight == 0.0:
            return 0.0

        text_score = self._text_relevance(entry.content, query) * weights.text_relevance
        importance_score = entry.importance_score * weights.importance
        freshness_score = entry.freshness_score * weights.freshness
        provenance_score = self._provenance_score(entry) * weights.provenance

        raw = (text_score + importance_score + freshness_score + provenance_score) / total_weight
        return round(min(1.0, max(0.0, raw)), 6)

    def rank(
        self,
        entries: list[tuple[MemoryEntry, str]],
    ) -> list[tuple[float, MemoryEntry]]:
        """Rank a list of (entry, query) pairs.

        Parameters
        ----------
        entries:
            Each element is ``(entry, query)`` where query is the original
            search string used to retrieve this entry.

        Returns
        -------
        list[tuple[float, MemoryEntry]]
            ``(score, entry)`` pairs sorted by score descending.
        """
        scored = [(self.score(entry, query), entry) for entry, query in entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Private signal implementations
    # ------------------------------------------------------------------

    def _text_relevance(self, content: str, query: str) -> float:
        """TF-style overlap score between query tokens and content."""
        query_tokens = _tokenise(query)
        if not query_tokens:
            return 0.0
        content_tokens = _tokenise(content)
        if not content_tokens:
            return 0.0

        total_tf = 0.0
        for token in query_tokens:
            tf = content_tokens.count(token) / len(content_tokens)
            total_tf += tf

        # Average TF across query tokens, then apply a scaling function
        avg_tf = total_tf / len(query_tokens)
        # Sigmoid-style saturation: map small TF values to meaningful scores
        return round(_sigmoid_scale(avg_tf), 6)

    def _provenance_score(self, entry: MemoryEntry) -> float:
        """Extract provenance reliability from metadata, or estimate from source."""
        raw = entry.metadata.get("provenance_reliability")
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
        # Fallback: heuristic from source type
        return _SOURCE_PROVENANCE.get(entry.source.value, 0.5)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_SOURCE_PROVENANCE: dict[str, float] = {
    "tool_output": 0.95,
    "document": 0.85,
    "external_api": 0.80,
    "user_input": 0.65,
    "agent_inference": 0.45,
}


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _sigmoid_scale(value: float, steepness: float = 10.0) -> float:
    """Map a small float [0, 1] to a sharper [0, 1] via a sigmoid."""
    if value <= 0.0:
        return 0.0
    # Standard sigmoid shifted so that value=0.5 maps near 0.99
    return 1.0 / (1.0 + math.exp(-steepness * (value - 0.5)))


__all__ = ["ResultRanker", "RankingWeights"]
