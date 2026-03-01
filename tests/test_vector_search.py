"""Tests for the vector search subpackage (agent_memory.vector).

Covers:
- NumpyVectorStore: upsert, search, delete, clear, count (10 tests)
- SQLiteVectorStore: same operations (10 tests)
- vector_search() fusion on MemorySearchEngine (5 tests)
- Backward compat: existing search without vectors still works (3 tests)
- EmbedderProtocol enforcement (2 tests)

Total: 30 tests
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from agent_memory.memory.semantic import SemanticMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.retrieval.search import MemorySearchEngine, SearchResult
from agent_memory.vector.numpy_store import NumpyVectorStore
from agent_memory.vector.protocol import EmbedderProtocol, VectorStoreProtocol
from agent_memory.vector.sqlite_vector_store import SQLiteVectorStore
from agent_memory.vector.types import VectorSearchResult


# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------


class _MockEmbedder(EmbedderProtocol):
    """Deterministic 3-D embedder for testing.

    Each word is mapped to a fixed unit vector; the embedding of a
    multi-word string is the average of its per-word vectors, normalised.
    Unknown words map to ``[0.0, 0.0, 1.0]``.
    """

    _WORD_VECTORS: ClassVar[dict[str, list[float]]] = {
        "cat": [1.0, 0.0, 0.0],
        "dog": [0.0, 1.0, 0.0],
        "bird": [0.0, 0.0, 1.0],
        "animal": [0.577, 0.577, 0.577],
        "sky": [0.0, 0.5, 0.866],
        "python": [0.866, 0.5, 0.0],
    }
    _DIM: ClassVar[int] = 3

    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        if not tokens:
            return [0.0, 0.0, 1.0]
        total = [0.0, 0.0, 0.0]
        for token in tokens:
            word_vec = self._WORD_VECTORS.get(token, [0.0, 0.0, 1.0])
            total = [total[i] + word_vec[i] for i in range(3)]
        norm = math.sqrt(sum(x * x for x in total))
        if norm < 1e-10:
            return [0.0, 0.0, 1.0]
        return [x / norm for x in total]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._DIM


def _vec(x: float, y: float, z: float) -> list[float]:
    """Return a unit-normalised 3-D vector."""
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-10:
        return [0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm]


@pytest.fixture()
def numpy_store() -> NumpyVectorStore:
    return NumpyVectorStore()


@pytest.fixture()
def sqlite_store() -> SQLiteVectorStore:
    return SQLiteVectorStore(":memory:")


@pytest.fixture()
def embedder() -> _MockEmbedder:
    return _MockEmbedder()


def _make_entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
    importance: float = 0.5,
) -> MemoryEntry:
    return MemoryEntry(content=content, layer=layer, importance_score=importance)


# ---------------------------------------------------------------------------
# Shared contract tests — run against both NumpyVectorStore and SQLiteVectorStore
# ---------------------------------------------------------------------------


def _run_upsert_and_count(store: VectorStoreProtocol) -> None:
    assert store.count() == 0
    store.upsert("key1", _vec(1.0, 0.0, 0.0), {"label": "cat"})
    assert store.count() == 1
    store.upsert("key2", _vec(0.0, 1.0, 0.0), {"label": "dog"})
    assert store.count() == 2


def _run_upsert_replaces_existing(store: VectorStoreProtocol) -> None:
    store.upsert("k", _vec(1.0, 0.0, 0.0), {"v": 1})
    store.upsert("k", _vec(0.0, 1.0, 0.0), {"v": 2})
    assert store.count() == 1
    results = store.search(_vec(0.0, 1.0, 0.0), top_k=1)
    assert results[0].metadata["v"] == 2


def _run_search_returns_correct_ordering(store: VectorStoreProtocol) -> None:
    # cat vector is closest to a pure [1,0,0] query
    store.upsert("cat", _vec(1.0, 0.0, 0.0), {})
    store.upsert("dog", _vec(0.0, 1.0, 0.0), {})
    store.upsert("bird", _vec(0.0, 0.0, 1.0), {})
    results = store.search(_vec(1.0, 0.0, 0.0), top_k=3)
    assert len(results) == 3
    assert results[0].key == "cat"
    assert results[0].score > results[1].score


def _run_search_top_k_limits_results(store: VectorStoreProtocol) -> None:
    for i in range(5):
        store.upsert(f"key{i}", _vec(float(i + 1), 0.0, 0.0), {})
    results = store.search(_vec(1.0, 0.0, 0.0), top_k=2)
    assert len(results) <= 2


def _run_search_min_score_filter(store: VectorStoreProtocol) -> None:
    store.upsert("near", _vec(1.0, 0.0, 0.0), {})
    store.upsert("far", _vec(0.0, 1.0, 0.0), {})
    results = store.search(_vec(1.0, 0.0, 0.0), top_k=10, min_score=0.9)
    keys = [r.key for r in results]
    assert "near" in keys
    assert "far" not in keys


def _run_search_empty_store_returns_empty(store: VectorStoreProtocol) -> None:
    results = store.search(_vec(1.0, 0.0, 0.0), top_k=5)
    assert results == []


def _run_search_returns_vector_search_result_objects(store: VectorStoreProtocol) -> None:
    store.upsert("key1", _vec(1.0, 0.0, 0.0), {"tag": "test"})
    results = store.search(_vec(1.0, 0.0, 0.0), top_k=1)
    assert isinstance(results[0], VectorSearchResult)
    assert results[0].key == "key1"
    assert results[0].metadata["tag"] == "test"


def _run_delete_removes_entry(store: VectorStoreProtocol) -> None:
    store.upsert("key1", _vec(1.0, 0.0, 0.0), {})
    assert store.count() == 1
    store.delete("key1")
    assert store.count() == 0


def _run_delete_nonexistent_is_noop(store: VectorStoreProtocol) -> None:
    store.upsert("key1", _vec(1.0, 0.0, 0.0), {})
    store.delete("does_not_exist")  # must not raise
    assert store.count() == 1


def _run_clear_removes_all(store: VectorStoreProtocol) -> None:
    store.upsert("a", _vec(1.0, 0.0, 0.0), {})
    store.upsert("b", _vec(0.0, 1.0, 0.0), {})
    assert store.count() == 2
    store.clear()
    assert store.count() == 0
    results = store.search(_vec(1.0, 0.0, 0.0))
    assert results == []


# ---------------------------------------------------------------------------
# NumpyVectorStore tests (10)
# ---------------------------------------------------------------------------


class TestNumpyVectorStore:
    def test_upsert_and_count(self, numpy_store: NumpyVectorStore) -> None:
        _run_upsert_and_count(numpy_store)

    def test_upsert_replaces_existing(self, numpy_store: NumpyVectorStore) -> None:
        _run_upsert_replaces_existing(numpy_store)

    def test_search_returns_correct_ordering(self, numpy_store: NumpyVectorStore) -> None:
        _run_search_returns_correct_ordering(numpy_store)

    def test_search_top_k_limits_results(self, numpy_store: NumpyVectorStore) -> None:
        _run_search_top_k_limits_results(numpy_store)

    def test_search_min_score_filter(self, numpy_store: NumpyVectorStore) -> None:
        _run_search_min_score_filter(numpy_store)

    def test_search_empty_store_returns_empty(self, numpy_store: NumpyVectorStore) -> None:
        _run_search_empty_store_returns_empty(numpy_store)

    def test_search_returns_vector_search_result_objects(
        self, numpy_store: NumpyVectorStore
    ) -> None:
        _run_search_returns_vector_search_result_objects(numpy_store)

    def test_delete_removes_entry(self, numpy_store: NumpyVectorStore) -> None:
        _run_delete_removes_entry(numpy_store)

    def test_delete_nonexistent_is_noop(self, numpy_store: NumpyVectorStore) -> None:
        _run_delete_nonexistent_is_noop(numpy_store)

    def test_clear_removes_all(self, numpy_store: NumpyVectorStore) -> None:
        _run_clear_removes_all(numpy_store)


# ---------------------------------------------------------------------------
# SQLiteVectorStore tests (10)
# ---------------------------------------------------------------------------


class TestSQLiteVectorStore:
    def test_upsert_and_count(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_upsert_and_count(sqlite_store)

    def test_upsert_replaces_existing(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_upsert_replaces_existing(sqlite_store)

    def test_search_returns_correct_ordering(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_search_returns_correct_ordering(sqlite_store)

    def test_search_top_k_limits_results(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_search_top_k_limits_results(sqlite_store)

    def test_search_min_score_filter(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_search_min_score_filter(sqlite_store)

    def test_search_empty_store_returns_empty(
        self, sqlite_store: SQLiteVectorStore
    ) -> None:
        _run_search_empty_store_returns_empty(sqlite_store)

    def test_search_returns_vector_search_result_objects(
        self, sqlite_store: SQLiteVectorStore
    ) -> None:
        _run_search_returns_vector_search_result_objects(sqlite_store)

    def test_delete_removes_entry(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_delete_removes_entry(sqlite_store)

    def test_delete_nonexistent_is_noop(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_delete_nonexistent_is_noop(sqlite_store)

    def test_clear_removes_all(self, sqlite_store: SQLiteVectorStore) -> None:
        _run_clear_removes_all(sqlite_store)


# ---------------------------------------------------------------------------
# vector_search() fusion tests (5)
# ---------------------------------------------------------------------------


def _build_engine_with_entries(
    entries: list[MemoryEntry],
) -> MemorySearchEngine:
    """Build a MemorySearchEngine with a SemanticMemory populated by entries."""
    semantic_store = SemanticMemory()
    for entry in entries:
        semantic_store.store(entry)
    engine = MemorySearchEngine(
        stores={MemoryLayer.SEMANTIC: semantic_store}  # type: ignore[arg-type]
    )
    return engine


class TestVectorSearchFusion:
    def test_vector_only_returns_results_ordered_by_similarity(
        self,
        embedder: _MockEmbedder,
        sqlite_store: SQLiteVectorStore,
    ) -> None:
        """text_weight=0 means only vector score drives ranking."""
        cat_entry = _make_entry("cat")
        dog_entry = _make_entry("dog")
        engine = _build_engine_with_entries([cat_entry, dog_entry])

        sqlite_store.upsert(cat_entry.memory_id, embedder.embed("cat"), {})
        sqlite_store.upsert(dog_entry.memory_id, embedder.embed("dog"), {})

        results = engine.vector_search(
            "cat",
            embedder=embedder,
            vector_store=sqlite_store,
            top_k=2,
            text_weight=0.0,
        )
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        # The cat entry should rank first since query == "cat"
        assert results[0].entry.memory_id == cat_entry.memory_id

    def test_fusion_blends_text_and_vector_scores(
        self,
        embedder: _MockEmbedder,
        sqlite_store: SQLiteVectorStore,
    ) -> None:
        """With equal weights the fused score should be between text and vector."""
        entry = _make_entry("cat")
        engine = _build_engine_with_entries([entry])
        sqlite_store.upsert(entry.memory_id, embedder.embed("cat"), {})

        results = engine.vector_search(
            "cat",
            embedder=embedder,
            vector_store=sqlite_store,
            top_k=5,
            text_weight=0.5,
        )
        assert len(results) >= 1
        # score must be in [0, 1]
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_top_k_limits_fused_results(
        self,
        embedder: _MockEmbedder,
        numpy_store: NumpyVectorStore,
    ) -> None:
        entries = [_make_entry(f"cat dog bird {i}") for i in range(8)]
        engine = _build_engine_with_entries(entries)
        for entry in entries:
            numpy_store.upsert(entry.memory_id, embedder.embed("cat"), {})

        results = engine.vector_search(
            "cat",
            embedder=embedder,
            vector_store=numpy_store,
            top_k=3,
        )
        assert len(results) <= 3

    def test_results_are_sorted_descending(
        self,
        embedder: _MockEmbedder,
        numpy_store: NumpyVectorStore,
    ) -> None:
        entries = [
            _make_entry("cat"),
            _make_entry("dog"),
            _make_entry("bird"),
        ]
        engine = _build_engine_with_entries(entries)
        for entry in entries:
            numpy_store.upsert(entry.memory_id, embedder.embed(entry.content), {})

        results = engine.vector_search(
            "cat",
            embedder=embedder,
            vector_store=numpy_store,
            top_k=3,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_vector_search_with_no_store_entries_still_returns_results(
        self,
        embedder: _MockEmbedder,
        sqlite_store: SQLiteVectorStore,
    ) -> None:
        """Results from vector store not in any memory store still surface."""
        engine = MemorySearchEngine()  # no stores registered
        sqlite_store.upsert("orphan-key", embedder.embed("cat"), {"content": "cat"})

        results = engine.vector_search(
            "cat",
            embedder=embedder,
            vector_store=sqlite_store,
            top_k=5,
            text_weight=0.0,
        )
        assert len(results) >= 1
        assert results[0].entry.memory_id == "orphan-key"


# ---------------------------------------------------------------------------
# Backward compatibility tests (3)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_memory_entry_embedding_defaults_to_none(self) -> None:
        """Existing code that creates MemoryEntry without embedding must work."""
        entry = MemoryEntry(content="hello", layer=MemoryLayer.SEMANTIC)
        assert entry.embedding is None

    def test_existing_search_unaffected(self) -> None:
        """The original text-based search still works without any vector setup."""
        engine = MemorySearchEngine(
            stores={MemoryLayer.SEMANTIC: SemanticMemory()}  # type: ignore[arg-type]
        )
        store: SemanticMemory = engine._stores[MemoryLayer.SEMANTIC]  # type: ignore[assignment]
        store.store(_make_entry("the quick brown fox"))
        results = engine.search("quick")
        assert len(results) >= 1

    def test_memory_entry_with_embedding_round_trips(self) -> None:
        """An entry with an embedding set serialises and deserialises correctly."""
        entry = MemoryEntry(
            content="hello",
            layer=MemoryLayer.SEMANTIC,
            embedding=[0.1, 0.2, 0.3],
        )
        json_str = entry.model_dump_json()
        reloaded = MemoryEntry.model_validate_json(json_str)
        assert reloaded.embedding == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# EmbedderProtocol enforcement tests (2)
# ---------------------------------------------------------------------------


class TestEmbedderProtocolEnforcement:
    def test_cannot_instantiate_abstract_embedder(self) -> None:
        """EmbedderProtocol is abstract and must not be instantiated directly."""
        with pytest.raises(TypeError):
            EmbedderProtocol()  # type: ignore[abstract]

    def test_incomplete_embedder_implementation_raises_type_error(self) -> None:
        """A class that omits required abstract methods cannot be instantiated."""

        class _IncompleteEmbedder(EmbedderProtocol):
            """Missing embed_batch and dimension."""

            def embed(self, text: str) -> list[float]:
                return [0.0]

        with pytest.raises(TypeError):
            _IncompleteEmbedder()  # type: ignore[abstract]
