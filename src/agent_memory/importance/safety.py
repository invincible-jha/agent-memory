"""Safety memory protector — prevents decay and deletion of safety-critical entries."""

from __future__ import annotations

from typing import Sequence

from agent_memory.memory.types import MemoryEntry


class SafetyMemoryProtector:
    """Enforce that safety-critical memories are never degraded or removed.

    This class acts as a guard layer that other components should call before
    applying decay or deleting entries.  It does not store any memories
    itself — it only evaluates ``safety_critical`` flags.
    """

    def is_protected(self, entry: MemoryEntry) -> bool:
        """Return True if the entry must not be decayed or deleted."""
        return entry.safety_critical

    def filter_unprotected(
        self, entries: Sequence[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Return only entries that are *not* safety-critical."""
        return [e for e in entries if not e.safety_critical]

    def filter_protected(
        self, entries: Sequence[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Return only entries that are safety-critical."""
        return [e for e in entries if e.safety_critical]

    def apply_protected_importance(self, entry: MemoryEntry) -> MemoryEntry:
        """If safety-critical, ensure importance_score is always 1.0."""
        if entry.safety_critical and entry.importance_score < 1.0:
            return entry.model_copy(update={"importance_score": 1.0})
        return entry

    def enforce_on_batch(
        self, entries: Sequence[MemoryEntry]
    ) -> list[MemoryEntry]:
        """Return entries with safety-critical ones having importance_score = 1.0."""
        return [self.apply_protected_importance(e) for e in entries]


__all__ = ["SafetyMemoryProtector"]
