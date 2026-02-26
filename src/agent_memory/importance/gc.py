"""Memory garbage collector — removes entries below importance threshold after decay."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from agent_memory.importance.decay import DecayCurve, ExponentialDecay
from agent_memory.importance.safety import SafetyMemoryProtector
from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry


def _age_hours(entry: MemoryEntry) -> float:
    """Return the age of an entry in hours from now."""
    now = datetime.now(timezone.utc)
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = now - created
    return delta.total_seconds() / 3600.0


class MemoryGarbageCollector:
    """Collect and remove memories whose effective importance falls below a threshold.

    Effective importance = ``entry.importance_score * decay(age_hours)``.
    Safety-critical entries are never collected.

    Parameters
    ----------
    store:
        The memory store to operate on.
    decay_curve:
        The decay function to apply to importance scores.  Defaults to
        exponential decay with a 48-hour half-life.
    threshold:
        Entries with effective importance below this value are candidates
        for removal.  Defaults to 0.05.
    """

    def __init__(
        self,
        store: MemoryStore,
        decay_curve: DecayCurve | None = None,
        threshold: float = 0.05,
    ) -> None:
        self._store = store
        self._decay = decay_curve or ExponentialDecay(half_life_hours=48.0)
        self._threshold = threshold
        self._protector = SafetyMemoryProtector()

    @property
    def threshold(self) -> float:
        return self._threshold

    def effective_importance(self, entry: MemoryEntry) -> float:
        """Return decayed importance for the given entry."""
        age = _age_hours(entry)
        return entry.importance_score * self._decay.decay(age)

    def candidates(self, entries: Sequence[MemoryEntry] | None = None) -> list[MemoryEntry]:
        """Return entries eligible for garbage collection.

        Parameters
        ----------
        entries:
            Explicit list of entries to evaluate; if None, all entries in
            the store are evaluated.
        """
        pool = list(entries) if entries is not None else list(self._store.all())
        unprotected = self._protector.filter_unprotected(pool)
        return [e for e in unprotected if self.effective_importance(e) < self._threshold]

    def collect(self, dry_run: bool = False) -> list[str]:
        """Remove eligible entries from the store.

        Parameters
        ----------
        dry_run:
            If True, identify candidates but do not actually delete them.

        Returns
        -------
        list[str]
            Memory IDs that were (or would be) removed.
        """
        evicted_ids = [e.memory_id for e in self.candidates()]
        if not dry_run:
            for mid in evicted_ids:
                self._store.delete(mid)
        return evicted_ids

    def stats(self) -> dict[str, int]:
        """Return a summary of current store state relative to GC threshold."""
        all_entries = list(self._store.all())
        unprotected = self._protector.filter_unprotected(all_entries)
        protected = self._protector.filter_protected(all_entries)
        eligible = [e for e in unprotected if self.effective_importance(e) < self._threshold]
        return {
            "total": len(all_entries),
            "protected": len(protected),
            "unprotected": len(unprotected),
            "eligible_for_gc": len(eligible),
        }


__all__ = ["MemoryGarbageCollector"]
