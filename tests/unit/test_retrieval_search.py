"""Unit tests for agent_memory.retrieval.search — MemorySearchEngine."""

from __future__ import annotations

import pytest

from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.semantic import SemanticMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.memory.working import WorkingMemory
from agent_memory.retrieval.ranking import RankingWeights, ResultRanker
from agent_memory.retrieval.search import MemorySearchEngine, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
    importance: float = 0.5,
    freshness: float = 1.0,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        importance_score=importance,
        freshness_score=freshness,
    )


def _build_engine(*entries: tuple[MemoryLayer, str]) -> MemorySearchEngine:
    """Build a search engine pre-populated with (layer, content) pairs."""
    stores: dict[MemoryLayer, object] = {
        MemoryLayer.WORKING: WorkingMemory(capacity=64),
        MemoryLayer.EPISODIC: EpisodicMemory(),
        MemoryLayer.SEMANTIC: SemanticMemory(),
    }
    for layer, content in entries:
        entry = _make_entry(content, layer=layer)
        stores[layer].store(entry)  # type: ignore[attr-defined]
    engine = MemorySearchEngine(stores=stores)  # type: ignore[arg-type]
    return engine


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_lt_for_descending_sort(self) -> None:
        """__lt__ should order by score descending (higher score is 'less than')."""
        e = _make_entry("x")
        r_high = SearchResult(
            entry=e, score=0.9, rank=1, matched_layer=MemoryLayer.SEMANTIC, query="x"
        )
        r_low = SearchResult(
            entry=e, score=0.1, rank=2, matched_layer=MemoryLayer.SEMANTIC, query="x"
        )
        # r_high < r_low means r_high sorts before r_low (descending)
        assert r_high < r_low

    def test_retrieved_at_set_on_creation(self) -> None:
        e = _make_entry("x")
        result = SearchResult(
            entry=e, score=0.5, rank=1, matched_layer=MemoryLayer.SEMANTIC, query="q"
        )
        assert result.retrieved_at is not None


# ---------------------------------------------------------------------------
# MemorySearchEngine construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty_stores_by_default(self) -> None:
        engine = MemorySearchEngine()
        results = engine.search("anything")
        assert results == []

    def test_default_limit(self) -> None:
        engine = MemorySearchEngine(default_limit=5)
        assert engine._default_limit == 5

    def test_custom_ranker_stored(self) -> None:
        ranker = ResultRanker()
        engine = MemorySearchEngine(ranker=ranker)
        assert engine._ranker is ranker


# ---------------------------------------------------------------------------
# register_store
# ---------------------------------------------------------------------------


class TestRegisterStore:
    def test_register_adds_store(self) -> None:
        engine = MemorySearchEngine()
        store = SemanticMemory()
        engine.register_store(MemoryLayer.SEMANTIC, store)
        assert engine._stores[MemoryLayer.SEMANTIC] is store

    def test_register_overwrites_existing(self) -> None:
        engine = MemorySearchEngine()
        s1 = SemanticMemory()
        s2 = SemanticMemory()
        engine.register_store(MemoryLayer.SEMANTIC, s1)
        engine.register_store(MemoryLayer.SEMANTIC, s2)
        assert engine._stores[MemoryLayer.SEMANTIC] is s2


# ---------------------------------------------------------------------------
# search — basic behaviour
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_list_of_search_results(self) -> None:
        engine = _build_engine((MemoryLayer.SEMANTIC, "python programming"))
        results = engine.search("python")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_finds_matching_entry(self) -> None:
        engine = _build_engine((MemoryLayer.SEMANTIC, "the quick brown fox"))
        results = engine.search("quick")
        assert len(results) >= 1

    def test_search_empty_store_returns_empty(self) -> None:
        engine = MemorySearchEngine()
        results = engine.search("anything")
        assert results == []

    def test_search_sorted_descending_by_score(self) -> None:
        engine = _build_engine(
            (MemoryLayer.SEMANTIC, "python python python"),
            (MemoryLayer.SEMANTIC, "python once"),
        )
        results = engine.search("python")
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_search_respects_limit(self) -> None:
        stores: dict[MemoryLayer, SemanticMemory] = {
            MemoryLayer.SEMANTIC: SemanticMemory()
        }
        for i in range(10):
            stores[MemoryLayer.SEMANTIC].store(_make_entry(f"common term item {i}"))
        engine = MemorySearchEngine(stores=stores)  # type: ignore[arg-type]
        results = engine.search("common term", limit=3)
        assert len(results) <= 3

    def test_search_uses_default_limit_when_none_given(self) -> None:
        stores: dict[MemoryLayer, SemanticMemory] = {
            MemoryLayer.SEMANTIC: SemanticMemory()
        }
        for i in range(30):
            stores[MemoryLayer.SEMANTIC].store(_make_entry(f"common keyword entry {i}"))
        engine = MemorySearchEngine(stores=stores, default_limit=5)  # type: ignore[arg-type]
        results = engine.search("common keyword")
        assert len(results) <= 5

    def test_search_filters_by_layer(self) -> None:
        engine = _build_engine(
            (MemoryLayer.SEMANTIC, "shared content term"),
            (MemoryLayer.EPISODIC, "shared content term"),
        )
        results = engine.search("shared content term", layer=MemoryLayer.SEMANTIC)
        assert all(r.matched_layer is MemoryLayer.SEMANTIC for r in results)

    def test_search_deduplicates_by_memory_id(self) -> None:
        engine = _build_engine((MemoryLayer.SEMANTIC, "unique content here"))
        results = engine.search("unique content here")
        ids = [r.entry.memory_id for r in results]
        assert len(ids) == len(set(ids))

    def test_search_min_importance_filter(self) -> None:
        stores: dict[MemoryLayer, SemanticMemory] = {
            MemoryLayer.SEMANTIC: SemanticMemory()
        }
        stores[MemoryLayer.SEMANTIC].store(
            _make_entry("keyword content low", importance=0.1)
        )
        stores[MemoryLayer.SEMANTIC].store(
            _make_entry("keyword content high", importance=0.9)
        )
        engine = MemorySearchEngine(stores=stores)  # type: ignore[arg-type]
        results = engine.search("keyword content", min_importance=0.5)
        assert all(r.entry.importance_score >= 0.5 for r in results)

    def test_search_min_freshness_filter(self) -> None:
        stores: dict[MemoryLayer, SemanticMemory] = {
            MemoryLayer.SEMANTIC: SemanticMemory()
        }
        stores[MemoryLayer.SEMANTIC].store(
            _make_entry("keyword content fresh", freshness=0.9)
        )
        stores[MemoryLayer.SEMANTIC].store(
            _make_entry("keyword content stale", freshness=0.05)
        )
        engine = MemorySearchEngine(stores=stores)  # type: ignore[arg-type]
        results = engine.search("keyword content", min_freshness=0.5)
        assert all(r.entry.freshness_score >= 0.5 for r in results)

    def test_search_assigns_sequential_ranks(self) -> None:
        engine = _build_engine(
            (MemoryLayer.SEMANTIC, "alpha content term"),
            (MemoryLayer.SEMANTIC, "beta content term"),
        )
        results = engine.search("content term")
        for i, result in enumerate(results):
            assert result.rank == i + 1

    def test_search_missing_layer_store_skipped(self) -> None:
        engine = MemorySearchEngine()
        # Search for a layer that has no registered store
        results = engine.search("anything", layer=MemoryLayer.PROCEDURAL)
        assert results == []


# ---------------------------------------------------------------------------
# search_all
# ---------------------------------------------------------------------------


class TestSearchAll:
    def test_search_all_across_all_layers(self) -> None:
        engine = _build_engine(
            (MemoryLayer.SEMANTIC, "common search term entry"),
            (MemoryLayer.EPISODIC, "common search term entry"),
        )
        results = engine.search_all("common search term entry")
        assert len(results) >= 1

    def test_search_all_returns_list(self) -> None:
        engine = MemorySearchEngine()
        results = engine.search_all("anything")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# get_by_layer
# ---------------------------------------------------------------------------


class TestGetByLayer:
    def test_get_by_layer_returns_entries_for_layer(self) -> None:
        engine = _build_engine(
            (MemoryLayer.SEMANTIC, "semantic entry one"),
            (MemoryLayer.SEMANTIC, "semantic entry two"),
        )
        entries = engine.get_by_layer(MemoryLayer.SEMANTIC)
        assert len(entries) == 2

    def test_get_by_layer_missing_store_returns_empty(self) -> None:
        engine = MemorySearchEngine()
        entries = engine.get_by_layer(MemoryLayer.PROCEDURAL)
        assert entries == []

    def test_get_by_layer_respects_limit(self) -> None:
        stores: dict[MemoryLayer, SemanticMemory] = {
            MemoryLayer.SEMANTIC: SemanticMemory()
        }
        for i in range(20):
            stores[MemoryLayer.SEMANTIC].store(_make_entry(f"item {i}"))
        engine = MemorySearchEngine(stores=stores)  # type: ignore[arg-type]
        entries = engine.get_by_layer(MemoryLayer.SEMANTIC, limit=5)
        assert len(entries) <= 5

    def test_get_by_layer_returns_memory_entries(self) -> None:
        engine = _build_engine((MemoryLayer.SEMANTIC, "test"))
        entries = engine.get_by_layer(MemoryLayer.SEMANTIC)
        assert all(isinstance(e, MemoryEntry) for e in entries)
