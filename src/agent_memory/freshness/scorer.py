"""Freshness scorer — score entries based on last-verified timestamp and decay."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_memory.freshness.decay import FreshnessDecay
from agent_memory.memory.types import MemoryEntry


def _age_hours(reference: datetime, now: datetime | None = None) -> float:
    actual_now = now or datetime.now(timezone.utc)
    ref_utc = reference.replace(tzinfo=timezone.utc) if reference.tzinfo is None else reference
    now_utc = actual_now.replace(tzinfo=timezone.utc) if actual_now.tzinfo is None else actual_now
    return max(0.0, (now_utc - ref_utc).total_seconds() / 3600.0)


class FreshnessScorer:
    """Compute freshness scores for memory entries.

    The reference timestamp used for scoring is:
    1. The ``last_verified`` key in ``entry.metadata`` if present (ISO format).
    2. Otherwise, ``entry.last_accessed``.
    3. Finally, ``entry.created_at`` as a fallback.

    Parameters
    ----------
    decay:
        The decay curve to use.  Defaults to exponential with a 48-hour half-life.
    """

    def __init__(self, decay: FreshnessDecay | None = None) -> None:
        self._decay = decay or FreshnessDecay(mode="exponential", half_life_hours=48.0)

    def score(self, entry: MemoryEntry, now: datetime | None = None) -> float:
        """Return freshness in [0, 1] for the given entry."""
        reference = self._reference_time(entry)
        age = _age_hours(reference, now)
        return round(self._decay.compute(age), 6)

    def score_and_update(
        self, entry: MemoryEntry, now: datetime | None = None
    ) -> MemoryEntry:
        """Return a new MemoryEntry with the freshness_score field updated."""
        new_score = self.score(entry, now)
        return entry.model_copy(update={"freshness_score": new_score})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reference_time(self, entry: MemoryEntry) -> datetime:
        raw = entry.metadata.get("last_verified")
        if raw:
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                pass
        return entry.last_accessed


__all__ = ["FreshnessScorer"]
