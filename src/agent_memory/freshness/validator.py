"""Stale memory detector — flag entries below a freshness threshold."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from agent_memory.freshness.scorer import FreshnessScorer
from agent_memory.memory.types import MemoryEntry


class StaleMemoryDetector:
    """Identify memories that have dropped below an acceptable freshness level.

    Parameters
    ----------
    scorer:
        The freshness scorer to use.
    threshold:
        Entries with a freshness score below this value are considered stale.
        Defaults to 0.3.
    """

    def __init__(
        self,
        scorer: FreshnessScorer | None = None,
        threshold: float = 0.3,
    ) -> None:
        self._scorer = scorer or FreshnessScorer()
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def is_stale(self, entry: MemoryEntry, now: Optional[datetime] = None) -> bool:
        """Return True if the entry's freshness score is below the threshold."""
        return self._scorer.score(entry, now) < self._threshold

    def find_stale(
        self,
        entries: Sequence[MemoryEntry],
        now: Optional[datetime] = None,
    ) -> list[MemoryEntry]:
        """Return all entries in the sequence that are stale."""
        return [e for e in entries if self.is_stale(e, now)]

    def freshness_report(
        self,
        entries: Sequence[MemoryEntry],
        now: Optional[datetime] = None,
    ) -> list[dict[str, object]]:
        """Return a summary dict for each entry with ID, score, and stale flag."""
        return [
            {
                "memory_id": e.memory_id,
                "freshness_score": self._scorer.score(e, now),
                "is_stale": self.is_stale(e, now),
                "content_preview": e.content[:80],
            }
            for e in entries
        ]


__all__ = ["StaleMemoryDetector"]
