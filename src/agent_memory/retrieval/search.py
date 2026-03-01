"""Memory search engine — searches across all cognitive layers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Sequence

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer

if TYPE_CHECKING:
    from agent_memory.vector.protocol import EmbedderProtocol, VectorStoreProtocol
    from agent_memory.vector.types import VectorSearchResult


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

    def vector_search(
        self,
        query_text: str,
        embedder: "EmbedderProtocol",
        vector_store: "VectorStoreProtocol",
        top_k: int = 10,
        min_score: float = 0.0,
        text_weight: float = 0.5,
        layer: Optional[MemoryLayer] = None,
    ) -> list[SearchResult]:
        """Search memory using vector similarity, optionally fused with text search.

        The query text is embedded and used to search the vector store.
        Optionally, results are fused with a text-based search from the
        registered memory stores.

        When ``text_weight`` is ``0.0``, only vector scores are used.
        When ``text_weight`` is ``1.0``, only text scores are used.
        Any value in between blends both signals proportionally.

        The returned :class:`SearchResult` objects are constructed by
        matching vector keys (assumed to be ``memory_id`` values) against
        entries in the registered memory stores, or by synthesising a
        minimal :class:`~agent_memory.memory.types.MemoryEntry` when no
        matching store entry is found.

        Parameters
        ----------
        query_text:
            Free-text query to embed and search.
        embedder:
            An :class:`~agent_memory.vector.protocol.EmbedderProtocol`
            implementation used to embed ``query_text``.
        vector_store:
            A :class:`~agent_memory.vector.protocol.VectorStoreProtocol`
            implementation to search against.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum vector similarity threshold.
        text_weight:
            Weight for the text search signal in ``[0.0, 1.0]``.  The
            vector signal weight is ``1.0 - text_weight``.  Defaults to
            ``0.5`` (equal blend).
        layer:
            If provided, text search is limited to this layer.  Vector
            results are returned regardless of layer.

        Returns
        -------
        list[SearchResult]
            Results sorted by fused score descending.
        """
        vector_weight: float = max(0.0, min(1.0, 1.0 - text_weight))
        clamped_text_weight: float = max(0.0, min(1.0, text_weight))

        # Embed the query and search the vector store
        query_vector: list[float] = embedder.embed(query_text)
        vector_results: list[VectorSearchResult] = vector_store.search(
            query_vector, top_k=top_k, min_score=min_score
        )

        # Build a map of memory_id -> vector score for fast lookup
        vector_score_map: dict[str, float] = {
            vr.key: vr.score for vr in vector_results
        }

        # Optionally run text search to blend scores
        text_score_map: dict[str, float] = {}
        if clamped_text_weight > 0.0 and self._stores:
            text_results = self.search(
                query_text,
                layer=layer,
                limit=top_k,
            )
            for text_result in text_results:
                text_score_map[text_result.entry.memory_id] = text_result.score

        # Build a set of all candidate memory IDs
        all_ids: set[str] = set(vector_score_map.keys()) | set(text_score_map.keys())

        # Build a lookup of memory_id -> (entry, layer) from registered stores
        entry_map: dict[str, tuple[MemoryEntry, MemoryLayer]] = {}
        for store_layer, store in self._stores.items():
            if layer is not None and store_layer is not layer:
                continue
            for stored_entry in store.all(layer=store_layer):
                if stored_entry.memory_id in all_ids:
                    entry_map[stored_entry.memory_id] = (stored_entry, store_layer)

        # Compute fused scores and build results
        fused: list[SearchResult] = []
        for memory_id in all_ids:
            v_score: float = vector_score_map.get(memory_id, 0.0)
            t_score: float = text_score_map.get(memory_id, 0.0)

            if clamped_text_weight == 0.0:
                fused_score = v_score
            elif vector_weight == 0.0:
                fused_score = t_score
            else:
                fused_score = vector_weight * v_score + clamped_text_weight * t_score

            if memory_id in entry_map:
                entry, matched_layer = entry_map[memory_id]
            else:
                # No matching store entry; synthesise a minimal entry from vector metadata
                matched_layer = layer or MemoryLayer.SEMANTIC
                # Look up metadata from vector results
                vector_metadata: dict[str, object] = {}
                for vr in vector_results:
                    if vr.key == memory_id:
                        vector_metadata = vr.metadata
                        break
                entry = MemoryEntry(
                    memory_id=memory_id,
                    content=str(vector_metadata.get("content", "")),
                    layer=matched_layer,
                )

            fused.append(
                SearchResult(
                    entry=entry,
                    score=round(fused_score, 6),
                    rank=0,
                    matched_layer=matched_layer,
                    query=query_text,
                )
            )

        fused.sort()  # descending by score via __lt__

        for rank_idx, result in enumerate(fused[:top_k]):
            result.rank = rank_idx + 1

        return fused[:top_k]

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
