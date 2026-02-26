"""Abstract base class for all memory store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer


class MemoryStore(ABC):
    """Contract that every storage backend must satisfy."""

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Persist a memory entry, replacing any existing entry with the same ID."""

    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Return the entry with the given ID, or None if not found."""

    @abstractmethod
    def all(self, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        """Return all entries, optionally filtered to a single layer."""

    @abstractmethod
    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        """Return the number of stored entries, optionally filtered by layer."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Remove an entry by ID. Returns True if it existed."""

    @abstractmethod
    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove all entries (optionally within a layer). Returns number deleted."""

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Full-text search; default implementation does substring matching."""
        tokens = query.lower().split()
        results: list[MemoryEntry] = []
        for entry in self.all(layer=layer):
            content_lower = entry.content.lower()
            if all(token in content_lower for token in tokens):
                results.append(entry)
        return results[:limit]


__all__ = ["MemoryStore"]
