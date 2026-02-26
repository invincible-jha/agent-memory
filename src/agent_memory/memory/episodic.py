"""Episodic memory — temporal events indexed by timestamp with range queries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer


class EpisodicMemory(MemoryStore):
    """Stores events ordered by creation time; supports time-range retrieval.

    All entries are stored in insertion order.  Range queries use the
    ``created_at`` field on each MemoryEntry.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        self._entries[entry.memory_id] = entry

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(memory_id)

    def all(self, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        entries = sorted(self._entries.values(), key=lambda e: e.created_at)
        if layer is not None:
            entries = [e for e in entries if e.layer == layer]
        return entries

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            return len(self._entries)
        return sum(1 for e in self._entries.values() if e.layer == layer)

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._entries:
            return False
        del self._entries[memory_id]
        return True

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        to_delete = [mid for mid, e in self._entries.items() if e.layer == layer]
        for mid in to_delete:
            del self._entries[mid]
        return len(to_delete)

    # ------------------------------------------------------------------
    # Episodic-specific helpers
    # ------------------------------------------------------------------

    def query_range(
        self,
        start: datetime,
        end: datetime,
        layer: Optional[MemoryLayer] = None,
    ) -> Sequence[MemoryEntry]:
        """Return entries whose ``created_at`` falls within [start, end]."""
        # Normalise to UTC-aware for safe comparison
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        results = [
            e
            for e in self.all(layer=layer)
            if start_utc <= _to_utc(e.created_at) <= end_utc
        ]
        return results

    def most_recent(self, n: int = 10, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        """Return the *n* most-recently created entries."""
        ordered = list(self.all(layer=layer))
        return ordered[-n:][::-1]

    def oldest(self, n: int = 10, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        """Return the *n* oldest entries."""
        ordered = list(self.all(layer=layer))
        return ordered[:n]


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = ["EpisodicMemory"]
