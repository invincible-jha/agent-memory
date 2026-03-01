"""Abstract base class for async storage backends.

This module provides a parallel async API that mirrors the synchronous
``StorageBackend`` ABC.  Existing sync backends are left untouched.

Classes
-------
- AsyncStorageBackend  — abstract base for all async storage backends
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer


class AsyncStorageBackend(ABC):
    """Contract that all async persistent storage backends must satisfy.

    All methods are coroutines (``async def``).  Implementations should use
    ``asyncio.Lock`` for in-process thread safety where needed.

    Unlike the synchronous ``StorageBackend``, this ABC is designed for use
    in async-first applications (e.g. FastAPI, AIOHTTP, asyncio services).
    """

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> None:
        """Persist a memory entry, replacing any existing entry with the same key.

        Parameters
        ----------
        entry:
            The entry to persist. ``entry.memory_id`` is used as the key.
        """

    @abstractmethod
    async def load(self, key: str) -> Optional[MemoryEntry]:
        """Load an entry by its key.

        Parameters
        ----------
        key:
            The ``memory_id`` of the entry to load.

        Returns
        -------
        MemoryEntry | None
            The entry, or None if no entry with that key exists.
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove an entry by key.

        Parameters
        ----------
        key:
            The ``memory_id`` to remove.

        Returns
        -------
        bool
            True if the entry existed and was deleted, False otherwise.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Search stored entries by content.

        Parameters
        ----------
        query:
            The search query string.
        layer:
            Optional layer filter.
        limit:
            Maximum number of results to return.

        Returns
        -------
        Sequence[MemoryEntry]
            Matching entries, ordered by relevance (implementation-defined).
        """

    @abstractmethod
    async def list_keys(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[str]:
        """List stored entry keys.

        Parameters
        ----------
        layer:
            Optional layer filter.
        limit:
            Maximum number of keys to return.

        Returns
        -------
        list[str]
            Entry keys (``memory_id`` values).
        """

    @abstractmethod
    async def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove all stored entries, optionally filtered by layer.

        Parameters
        ----------
        layer:
            If provided, only entries in this layer are removed.
            If None, all entries are removed.

        Returns
        -------
        int
            Number of entries deleted.
        """

    # ------------------------------------------------------------------
    # Default implementations (backends may override for efficiency)
    # ------------------------------------------------------------------

    async def count(self, layer: Optional[MemoryLayer] = None) -> int:
        """Return the number of stored entries.

        The default implementation uses ``list_keys``; backends with
        native count operations should override this for efficiency.
        """
        keys = await self.list_keys(layer=layer)
        return len(keys)

    async def load_all(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[MemoryEntry]:
        """Load multiple entries at once.

        The default implementation calls ``load`` for each key from
        ``list_keys``. Backends may override this for batch efficiency.

        Parameters
        ----------
        layer:
            Optional layer filter.
        limit:
            Maximum number of entries to return.

        Returns
        -------
        list[MemoryEntry]
            Loaded entries (None results are omitted).
        """
        keys = await self.list_keys(layer=layer, limit=limit)
        results: list[MemoryEntry] = []
        for key in keys:
            entry = await self.load(key)
            if entry is not None:
                results.append(entry)
        return results


__all__ = ["AsyncStorageBackend"]
