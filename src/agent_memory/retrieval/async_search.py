"""Async memory search engine — searches across all cognitive layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.retrieval.search import SearchResult
from agent_memory.storage.async_base import AsyncStorageBackend


class AsyncMemorySearchEngine:
    """Search across multiple async storage backends and cognitive layers.

    Combines results from all registered backends, deduplicates by
    ``memory_id``, scores them, and returns a ranked list.

    Parameters
    ----------
    backends:
        Mapping of ``MemoryLayer`` to ``AsyncStorageBackend`` instances.
    ranker:
        An optional ``ResultRanker`` instance for scoring.  If not provided,
        a default ``ResultRanker`` is constructed on first use.
    default_limit:
        Default maximum number of results returned per search.
    """

    def __init__(
        self,
        backends: Optional[dict[MemoryLayer, AsyncStorageBackend]] = None,
        ranker: Optional[object] = None,
        default_limit: int = 20,
    ) -> None:
        self._backends: dict[MemoryLayer, AsyncStorageBackend] = backends or {}
        self._ranker = ranker
        self._default_limit = default_limit

    def register_backend(self, layer: MemoryLayer, backend: AsyncStorageBackend) -> None:
        """Register an AsyncStorageBackend for a specific layer."""
        self._backends[layer] = backend

    async def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0,
        min_freshness: float = 0.0,
    ) -> list[SearchResult]:
        """Search for entries matching the query across registered backends.

        Parameters
        ----------
        query:
            The free-text search query.
        layer:
            If provided, only the backend for that layer is searched.
        limit:
            Maximum results to return; falls back to ``default_limit``.
        min_importance:
            Exclude entries with ``importance_score`` below this value.
        min_freshness:
            Exclude entries with ``freshness_score`` below this value.

        Returns
        -------
        list[SearchResult]
            Results sorted by score descending.
        """
        effective_limit = limit if limit is not None else self._default_limit
        raw_entries: dict[str, tuple[MemoryEntry, MemoryLayer]] = {}

        search_layers = [layer] if layer is not None else list(self._backends.keys())
        for search_layer in search_layers:
            backend = self._backends.get(search_layer)
            if backend is None:
                continue
            results = await backend.search(
                query, layer=search_layer, limit=effective_limit * 2
            )
            for entry in results:
                if entry.memory_id not in raw_entries:
                    raw_entries[entry.memory_id] = (entry, search_layer)

        # Apply filters
        filtered = [
            (entry, lyr)
            for entry, lyr in raw_entries.values()
            if entry.importance_score >= min_importance
            and entry.freshness_score >= min_freshness
        ]

        # Score and rank
        search_results = self._rank_results(filtered, query)
        search_results.sort()  # uses SearchResult.__lt__ for descending score

        # Re-assign ranks after sorting
        for rank_idx, result in enumerate(search_results[:effective_limit]):
            result.rank = rank_idx + 1

        return search_results[:effective_limit]

    async def search_all(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> list[SearchResult]:
        """Convenience method: search all layers without filters."""
        return await self.search(query, layer=None, limit=limit)

    async def get_by_layer(
        self,
        layer: MemoryLayer,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return all entries from a given layer, up to ``limit``."""
        backend = self._backends.get(layer)
        if backend is None:
            return []
        return (await backend.load_all(layer=layer, limit=limit))[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rank_results(
        self,
        entries: list[tuple[MemoryEntry, MemoryLayer]],
        query: str,
    ) -> list[SearchResult]:
        """Score and wrap entries as SearchResult objects."""
        from agent_memory.retrieval.ranking import ResultRanker

        ranker: ResultRanker = self._ranker or ResultRanker()  # type: ignore[assignment]
        results: list[SearchResult] = []
        for rank_idx, (entry, matched_layer) in enumerate(entries):
            score = ranker.score(entry, query)
            results.append(
                SearchResult(
                    entry=entry,
                    score=score,
                    rank=rank_idx + 1,
                    matched_layer=matched_layer,
                    query=query,
                )
            )
        return results


__all__ = ["AsyncMemorySearchEngine"]
