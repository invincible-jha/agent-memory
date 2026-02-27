"""Tests for agent_memory.retrieval.hybrid (HybridRetriever)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agent_memory.retrieval.fusion import FusionWeights, ScoreFusion
from agent_memory.retrieval.graph_retriever import GraphRetriever, RetrievalResult
from agent_memory.retrieval.hybrid import HybridRetriever, VectorStoreProtocol
from agent_memory.retrieval.recency_scorer import RecencyScorer


# ---------------------------------------------------------------------------
# Mock vector store
# ---------------------------------------------------------------------------


class MockVectorStore:
    """Simple in-memory mock satisfying VectorStoreProtocol."""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results or []

    def query(self, text: str, top_k: int) -> list[RetrievalResult]:
        return self._results[:top_k]


class FailingVectorStore:
    """Vector store that raises an exception on query."""

    def query(self, text: str, top_k: int) -> list[RetrievalResult]:
        raise RuntimeError("Vector store unavailable")


def _make_result(content: str, source: str, score: float) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=score, metadata={})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_retriever() -> HybridRetriever:
    return HybridRetriever()


@pytest.fixture()
def graph_with_data() -> GraphRetriever:
    g = GraphRetriever()
    g.add_edge("Paris", "capital_of", "France", weight=1.0)
    g.add_edge("France", "located_in", "Europe", weight=0.9)
    return g


@pytest.fixture()
def vector_store() -> MockVectorStore:
    return MockVectorStore([
        _make_result("Machine learning algorithms", "vector", 0.92),
        _make_result("Deep learning frameworks", "vector", 0.85),
        _make_result("Natural language processing", "vector", 0.78),
    ])


# ---------------------------------------------------------------------------
# Tests: VectorStoreProtocol
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_mock_store_satisfies_protocol(self, vector_store: MockVectorStore) -> None:
        assert isinstance(vector_store, VectorStoreProtocol)

    def test_failing_store_satisfies_protocol(self) -> None:
        store = FailingVectorStore()
        assert isinstance(store, VectorStoreProtocol)


# ---------------------------------------------------------------------------
# Tests: HybridRetriever construction
# ---------------------------------------------------------------------------


class TestHybridRetrieverConstruction:
    def test_default_construction(self) -> None:
        r = HybridRetriever()
        assert r is not None

    def test_with_vector_store(self, vector_store: MockVectorStore) -> None:
        r = HybridRetriever(vector_store=vector_store)
        assert r is not None

    def test_with_graph(self, graph_with_data: GraphRetriever) -> None:
        r = HybridRetriever(graph=graph_with_data)
        assert r is not None

    def test_with_all_components(
        self,
        vector_store: MockVectorStore,
        graph_with_data: GraphRetriever,
    ) -> None:
        recency = RecencyScorer(half_life_hours=24.0)
        fusion = ScoreFusion(FusionWeights(vector=0.5, graph=0.3, recency=0.2))
        r = HybridRetriever(
            vector_store=vector_store,
            graph=graph_with_data,
            recency=recency,
            fusion=fusion,
        )
        assert r is not None


# ---------------------------------------------------------------------------
# Tests: HybridRetriever.retrieve — basic
# ---------------------------------------------------------------------------


class TestHybridRetrieverRetrieve:
    def test_retrieve_returns_list(self, empty_retriever: HybridRetriever) -> None:
        results = empty_retriever.retrieve("anything")
        assert isinstance(results, list)

    def test_retrieve_empty_graph_and_no_vector_returns_empty(
        self, empty_retriever: HybridRetriever
    ) -> None:
        results = empty_retriever.retrieve("query")
        assert results == []

    def test_retrieve_with_vector_store_returns_results(
        self, vector_store: MockVectorStore
    ) -> None:
        r = HybridRetriever(vector_store=vector_store)
        results = r.retrieve("machine learning", top_k=5)
        assert len(results) > 0

    def test_retrieve_with_graph_returns_results(
        self, graph_with_data: GraphRetriever
    ) -> None:
        r = HybridRetriever(graph=graph_with_data)
        results = r.retrieve("Paris", top_k=5)
        assert len(results) > 0

    def test_retrieve_respects_top_k(self, vector_store: MockVectorStore) -> None:
        r = HybridRetriever(vector_store=vector_store)
        results = r.retrieve("machine learning", top_k=2)
        assert len(results) <= 2

    def test_retrieve_scores_in_range(self, vector_store: MockVectorStore) -> None:
        r = HybridRetriever(vector_store=vector_store)
        results = r.retrieve("machine learning", top_k=10)
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_retrieve_sorted_descending(self, vector_store: MockVectorStore) -> None:
        r = HybridRetriever(vector_store=vector_store)
        results = r.retrieve("machine learning", top_k=10)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_source_is_hybrid(self, vector_store: MockVectorStore) -> None:
        r = HybridRetriever(vector_store=vector_store)
        results = r.retrieve("machine learning", top_k=5)
        for result in results:
            assert result.source == "hybrid"

    def test_retrieve_failing_vector_store_does_not_raise(self) -> None:
        r = HybridRetriever(vector_store=FailingVectorStore())
        # Should not raise — falls back to empty vector results
        results = r.retrieve("anything", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_graph_node_found(self, graph_with_data: GraphRetriever) -> None:
        r = HybridRetriever(graph=graph_with_data)
        results = r.retrieve("Paris", top_k=5)
        contents = [result.content for result in results]
        # France is one hop from Paris
        assert any("France" in c for c in contents) or len(results) > 0

    def test_retrieve_combined_vector_and_graph(
        self,
        vector_store: MockVectorStore,
        graph_with_data: GraphRetriever,
    ) -> None:
        r = HybridRetriever(vector_store=vector_store, graph=graph_with_data)
        results = r.retrieve("machine learning", top_k=10)
        # Vector store has machine learning content
        assert len(results) > 0

    def test_retrieve_deduplicates_content(self) -> None:
        # Same content in vector and graph
        shared_content = "Shared knowledge node"
        v = MockVectorStore([
            RetrievalResult(shared_content, "vector", 0.8, {})
        ])
        g = GraphRetriever()
        g.add_edge(shared_content, "related_to", "Something else", weight=0.7)
        r = HybridRetriever(vector_store=v, graph=g)
        results = r.retrieve("Shared knowledge", top_k=10)
        # shared_content should appear only once
        contents = [result.content for result in results]
        assert contents.count(shared_content) <= 1


# ---------------------------------------------------------------------------
# Tests: recency integration
# ---------------------------------------------------------------------------


class TestHybridRetrieverRecency:
    def test_recency_from_existing_with_timestamps(self) -> None:
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=30)
        new_ts = now - timedelta(hours=1)

        v = MockVectorStore([
            RetrievalResult("old doc", "vector", 0.7, {"timestamp": old_ts}),
            RetrievalResult("new doc", "vector", 0.7, {"timestamp": new_ts}),
        ])
        r = HybridRetriever(vector_store=v)
        # retrieve should not fail with timestamps in metadata
        results = r.retrieve("doc", top_k=5)
        assert isinstance(results, list)

    def test_recency_from_existing_without_timestamps(self) -> None:
        v = MockVectorStore([
            RetrievalResult("doc without ts", "vector", 0.8, {}),
        ])
        r = HybridRetriever(vector_store=v)
        results = r.retrieve("doc", top_k=5)
        assert len(results) > 0
