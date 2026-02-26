"""Heuristic importance scorer for memory entries."""

from __future__ import annotations

import re

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Keyword sets used for heuristic scoring
# ---------------------------------------------------------------------------

_CRITICAL_KEYWORDS: frozenset[str] = frozenset(
    """critical emergency danger urgent fatal error failure critical
    alert warning must never always forbidden prohibited stop halt
    safety secure confidential secret private restrict""".split()
)

_HIGH_KEYWORDS: frozenset[str] = frozenset(
    """important key primary main essential core required deadline
    constraint limit threshold maximum minimum required mandatory""".split()
)

_MEDIUM_KEYWORDS: frozenset[str] = frozenset(
    """should prefer recommended default typical normal standard
    common often usually generally""".split()
)

# Source reliability weights (higher = more important origin)
_SOURCE_WEIGHTS: dict[MemorySource, float] = {
    MemorySource.TOOL_OUTPUT: 0.9,
    MemorySource.DOCUMENT: 0.8,
    MemorySource.EXTERNAL_API: 0.75,
    MemorySource.USER_INPUT: 0.65,
    MemorySource.AGENT_INFERENCE: 0.5,
}

# Layer base importance
_LAYER_BASE: dict[MemoryLayer, float] = {
    MemoryLayer.WORKING: 0.6,
    MemoryLayer.EPISODIC: 0.5,
    MemoryLayer.SEMANTIC: 0.55,
    MemoryLayer.PROCEDURAL: 0.7,
}


class ImportanceScorer:
    """Score memory entries using content heuristics, source reliability, and layer.

    Safety-critical entries always receive a score of 1.0 regardless of
    other heuristics — the ``safety_critical`` flag acts as an override.
    """

    def score(self, entry: MemoryEntry) -> float:
        """Return an importance score in [0.0, 1.0] for the given entry."""
        if entry.safety_critical:
            return 1.0

        keyword_score = self._keyword_score(entry.content)
        source_weight = _SOURCE_WEIGHTS.get(entry.source, 0.5)
        layer_base = _LAYER_BASE.get(entry.layer, 0.5)

        # Weighted blend: keywords carry most weight, source and layer add context
        raw = keyword_score * 0.5 + source_weight * 0.3 + layer_base * 0.2
        return min(1.0, max(0.0, raw))

    def score_and_update(self, entry: MemoryEntry) -> MemoryEntry:
        """Return a new MemoryEntry with the importance_score field updated."""
        new_score = self.score(entry)
        return entry.model_copy(update={"importance_score": new_score})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _keyword_score(self, content: str) -> float:
        """Scan content for importance-indicating keywords."""
        words = set(re.findall(r"[a-z0-9]+", content.lower()))
        if words & _CRITICAL_KEYWORDS:
            return 1.0
        if words & _HIGH_KEYWORDS:
            return 0.75
        if words & _MEDIUM_KEYWORDS:
            return 0.55
        return 0.4  # neutral / no signal


__all__ = ["ImportanceScorer"]
