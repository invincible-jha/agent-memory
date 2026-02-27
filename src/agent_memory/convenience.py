"""Convenience API for agent-memory — 3-line quickstart.

Example
-------
::

    from agent_memory import Memory
    m = Memory()
    m.add("The sky is blue", user_id="user-1")
    results = m.search("sky")

"""
from __future__ import annotations

from typing import Any


class Memory:
    """Zero-config memory manager for the 80% use case.

    Wraps UnifiedMemory with in-memory storage (no database required)
    and provides simplified add/search/clear methods.

    Example
    -------
    ::

        from agent_memory import Memory
        m = Memory()
        m.add("Paris is the capital of France", user_id="user-1")
        results = m.search("capital France")
        print(results[0].entry.content if results else "no results")
    """

    def __init__(self) -> None:
        from agent_memory.unified.memory import UnifiedMemory
        from agent_memory.unified.config import MemoryConfig

        config = MemoryConfig(storage_backend="memory")
        self._memory = UnifiedMemory(config=config)

    def add(
        self,
        text: str,
        user_id: str = "default-user",
        layer: str = "semantic",
    ) -> Any:
        """Add a text memory.

        Parameters
        ----------
        text:
            Text content to memorize.
        user_id:
            User identifier for multi-user isolation.
        layer:
            Memory layer — one of ``"working"``, ``"episodic"``,
            ``"semantic"`` (default), or ``"procedural"``.

        Returns
        -------
        MemoryEntry
            The stored memory entry with its assigned ID.

        Example
        -------
        ::

            m = Memory()
            entry = m.add("The Eiffel Tower is in Paris", user_id="alice")
            print(entry.entry_id)
        """
        from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource

        layer_map: dict[str, MemoryLayer] = {
            "working": MemoryLayer.WORKING,
            "episodic": MemoryLayer.EPISODIC,
            "semantic": MemoryLayer.SEMANTIC,
            "procedural": MemoryLayer.PROCEDURAL,
        }
        memory_layer = layer_map.get(layer, MemoryLayer.SEMANTIC)

        entry = MemoryEntry(
            content=text,
            layer=memory_layer,
            source=MemorySource.USER_INPUT,
            agent_id=user_id,
        )
        return self._memory.store(entry)

    def search(self, query: str, limit: int = 10) -> list[Any]:
        """Search stored memories by text query.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[MemoryEntry]
            List of matching memory entries ordered by relevance.

        Example
        -------
        ::

            m = Memory()
            m.add("Python is a programming language")
            results = m.search("programming")
            print(results[0].entry.content if results else "empty")
        """
        return self._memory.search(query, limit=limit)

    def clear(self) -> None:
        """Remove all stored memories."""
        self._memory.storage.clear()
        for layer_store in self._memory._all_layer_stores():
            layer_store.clear()

    @property
    def underlying(self) -> Any:
        """The underlying UnifiedMemory instance."""
        return self._memory

    def __repr__(self) -> str:
        return "Memory(backend=in-memory)"
