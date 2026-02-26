"""Working memory — FIFO queue with configurable capacity and auto-eviction."""

from __future__ import annotations

import collections
from typing import Optional, Sequence

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer


class WorkingMemory(MemoryStore):
    """Short-term FIFO buffer; evicts oldest entries when capacity is exceeded.

    Parameters
    ----------
    capacity:
        Maximum number of entries kept at any time. When a new entry is
        stored and the buffer is full, the oldest entry is silently dropped.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        # deque maxlen enforces FIFO eviction automatically
        self._queue: collections.deque[MemoryEntry] = collections.deque(maxlen=capacity)
        # index for O(1) retrieval by ID
        self._index: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """Add or replace an entry. Replacement does not change queue order."""
        if entry.memory_id in self._index:
            # Update in-place within the deque
            for i, existing in enumerate(self._queue):
                if existing.memory_id == entry.memory_id:
                    self._queue[i] = entry
                    self._index[entry.memory_id] = entry
                    return
        # New entry: if queue is at capacity, deque drops leftmost automatically
        # We must remove the evicted entry from the index first
        if len(self._queue) == self._capacity:
            evicted = self._queue[0]
            self._index.pop(evicted.memory_id, None)
        self._queue.append(entry)
        self._index[entry.memory_id] = entry

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._index.get(memory_id)

    def all(self, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        entries = list(self._queue)
        if layer is not None:
            entries = [e for e in entries if e.layer == layer]
        return entries

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            return len(self._queue)
        return sum(1 for e in self._queue if e.layer == layer)

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._index:
            return False
        new_queue: collections.deque[MemoryEntry] = collections.deque(maxlen=self._capacity)
        for entry in self._queue:
            if entry.memory_id != memory_id:
                new_queue.append(entry)
        self._queue = new_queue
        del self._index[memory_id]
        return True

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            count = len(self._queue)
            self._queue.clear()
            self._index.clear()
            return count
        to_delete = [e.memory_id for e in self._queue if e.layer == layer]
        for mid in to_delete:
            self.delete(mid)
        return len(to_delete)

    # ------------------------------------------------------------------
    # WorkingMemory-specific helpers
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return len(self._queue) == self._capacity

    def peek_oldest(self) -> Optional[MemoryEntry]:
        """Return the oldest entry without removing or touching it."""
        return self._queue[0] if self._queue else None

    def peek_newest(self) -> Optional[MemoryEntry]:
        """Return the most-recently added entry."""
        return self._queue[-1] if self._queue else None


__all__ = ["WorkingMemory"]
