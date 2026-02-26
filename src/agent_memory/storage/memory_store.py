"""In-memory storage backend — dict-based with substring search."""

from __future__ import annotations

from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.storage.base import StorageBackend


class InMemoryStorage(StorageBackend):
    """Volatile dict-based storage backend.

    All data lives in process memory and is lost when the object is
    garbage-collected.  Suitable for unit tests, prototyping, and
    short-lived agents that do not require persistence.

    Search uses case-insensitive substring matching across all stored
    content fields.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> None:
        """Store an entry, replacing any existing entry with the same ID."""
        self._store[entry.memory_id] = entry

    def load(self, key: str) -> Optional[MemoryEntry]:
        """Return the entry with the given memory_id, or None."""
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Remove an entry by memory_id. Returns True if it existed."""
        if key not in self._store:
            return False
        del self._store[key]
        return True

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Case-insensitive substring search across all stored content.

        All query tokens (split on whitespace) must be present in the
        content for an entry to match.
        """
        tokens = query.lower().split()
        results: list[MemoryEntry] = []
        for entry in self._store.values():
            if layer is not None and entry.layer != layer:
                continue
            content_lower = entry.content.lower()
            if all(token in content_lower for token in tokens):
                results.append(entry)
            if len(results) >= limit:
                break
        return results

    def list_keys(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Return stored memory_id values, optionally filtered by layer."""
        if layer is None:
            keys = list(self._store.keys())
        else:
            keys = [k for k, e in self._store.items() if e.layer == layer]
        return keys[:limit]

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove all entries, optionally filtered by layer."""
        if layer is None:
            count = len(self._store)
            self._store.clear()
            return count
        to_delete = [k for k, e in self._store.items() if e.layer == layer]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        """Return the number of stored entries."""
        if layer is None:
            return len(self._store)
        return sum(1 for e in self._store.values() if e.layer == layer)

    def load_all(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[MemoryEntry]:
        """Return all stored entries, optionally filtered by layer."""
        if layer is None:
            entries = list(self._store.values())
        else:
            entries = [e for e in self._store.values() if e.layer == layer]
        return entries[:limit]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        return f"InMemoryStorage(count={len(self._store)})"


__all__ = ["InMemoryStorage"]
