"""Memory search engine — searches across all cognitive layers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer


@dataclass
class SearchResult:
    """A single result returned from a memory search."""

    entry: MemoryEntry
    score: float
    rank: int
    matched_layer: MemoryLayer
    query: str
    retrieved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __lt__(self, other: SearchResult) -> bool:
        """Support sorting by score descending."""
        return self.score > other.score


class MemorySearchEngine:
    """Search across multiple memory stores and cognitive layers.

    Combines results from all registered stores, deduplicates by
    ``memory_id``, and hands them to a ``ResultRanker`` for final
    ordering.

    Parameters
    ----------
    stores:
        Mapping of ``MemoryLayer`` to ``MemoryStore`` instances.  All
        registered stores are searched in parallel (sequentially in this
        implementation) and results are merged.
    ranker:
        A ``ResultRanker`` instance used to score results before returning.
        If not provided, a default ``ResultRanker`` with equal weights is
        used.
    default_limit:
        Default maximum number of results returned per search.
    """

    def __init__(
        self,
        stores: Optional[dict[MemoryLayer, MemoryStore]] = None,
        ranker: Optional[object] = None,
        default_limit: int = 20,
    ) -> None:
        self._stores: dict[MemoryLayer, MemoryStore] = stores or {}
        self._ranker = ranker
        self._default_limit = default_limit

    def register_store(self, layer: MemoryLayer, store: MemoryStore) -> None:
        """Register a MemoryStore for a specific layer."""
        self._stores[layer] = store

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0,
        min_freshness: float = 0.0,
    ) -> list[SearchResult]:
        """Search for entries matching the query.

        Parameters
        ----------
        query:
            The free-text search query.
        layer:
            If provided, only the store for that layer is searched.
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

        search_layers = [layer] if layer is not None else list(self._stores.keys())
        for search_layer in search_layers:
            store = self._stores.get(search_layer)
            if store is None:
                continue
            results = store.search(query, layer=search_layer, limit=effective_limit * 2)
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
        search_results.sort()  # uses __lt__ for descending score order

        # Re-assign ranks after sorting
        for rank_idx, result in enumerate(search_results[:effective_limit]):
            result.rank = rank_idx + 1

        return search_results[:effective_limit]

    def search_all(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> list[SearchResult]:
        """Convenience method: search all layers without filters."""
        return self.search(query, layer=None, limit=limit)

    def get_by_layer(
        self,
        layer: MemoryLayer,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return all entries from a given layer, up to ``limit``."""
        store = self._stores.get(layer)
        if store is None:
            return []
        return list(store.all(layer=layer))[:limit]

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


def _tokenise(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


__all__ = ["MemorySearchEngine", "SearchResult"]
