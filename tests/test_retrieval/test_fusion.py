"""Tests for agent_memory.retrieval.fusion."""
from __future__ import annotations

import pytest

from agent_memory.retrieval.fusion import FusionWeights, ScoreFusion
from agent_memory.retrieval.graph_retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(content: str, source: str, score: float) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=score, metadata={})


# ---------------------------------------------------------------------------
# Tests: FusionWeights
# ---------------------------------------------------------------------------


class TestFusionWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = FusionWeights()
        assert abs(w.vector + w.graph + w.recency - 1.0) < 0.01

    def test_default_values(self) -> None:
        w = FusionWeights()
        assert w.vector == 0.5
        assert w.graph == 0.3
        assert w.recency == 0.2

    def test_custom_weights_valid(self) -> None:
        w = FusionWeights(vector=0.6, graph=0.2, recency=0.2)
        assert w.vector == 0.6

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            FusionWeights(vector=0.5, graph=0.5, recency=0.5)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            FusionWeights(vector=-0.1, graph=0.6, recency=0.5)

    def test_weights_frozen(self) -> None:
        w = FusionWeights()
        with pytest.raises((AttributeError, TypeError)):
            w.vector = 0.9  # type: ignore[misc]

    def test_weights_within_tolerance(self) -> None:
        # Floating-point tolerance test (within 0.01)
        w = FusionWeights(vector=0.334, graph=0.333, recency=0.333)
        assert abs(w.vector + w.graph + w.recency - 1.0) <= 0.01


# ---------------------------------------------------------------------------
# Tests: ScoreFusion
# ---------------------------------------------------------------------------


class TestScoreFusion:
    def test_empty_inputs_returns_empty(self) -> None:
        fusion = ScoreFusion()
        result = fusion.fuse([], [], [], top_k=5)
        assert result == []

    def test_single_vector_result(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("doc1", "vector", 1.0)]
        results = fusion.fuse(v, [], [], top_k=5)
        assert len(results) == 1
        assert results[0].content == "doc1"

    def test_single_graph_result(self) -> None:
        fusion = ScoreFusion()
        g = [_make_result("node1", "graph", 0.8)]
        results = fusion.fuse([], g, [], top_k=5)
        assert len(results) == 1
        assert results[0].content == "node1"

    def test_fused_source_is_hybrid(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("doc1", "vector", 1.0)]
        results = fusion.fuse(v, [], [], top_k=5)
        assert results[0].source == "hybrid"

    def test_overlapping_content_fused_once(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("doc1", "vector", 0.9)]
        g = [_make_result("doc1", "graph", 0.7)]
        results = fusion.fuse(v, g, [], top_k=5)
        # "doc1" appears in both — should be deduped
        assert len(results) == 1

    def test_overlapping_content_fused_score_weighted_sum(self) -> None:
        weights = FusionWeights(vector=0.5, graph=0.3, recency=0.2)
        fusion = ScoreFusion(weights=weights)
        v = [_make_result("doc1", "vector", 1.0)]
        g = [_make_result("doc1", "graph", 1.0)]
        r = [_make_result("doc1", "recency", 1.0)]
        results = fusion.fuse(v, g, r, top_k=5)
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_sorted_by_score_descending(self) -> None:
        fusion = ScoreFusion()
        v = [
            _make_result("a", "vector", 0.9),
            _make_result("b", "vector", 0.3),
            _make_result("c", "vector", 0.6),
        ]
        results = fusion.fuse(v, [], [], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result(f"doc{i}", "vector", float(i) / 10) for i in range(1, 11)]
        results = fusion.fuse(v, [], [], top_k=3)
        assert len(results) == 3

    def test_top_k_zero_returns_empty(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("doc1", "vector", 0.9)]
        results = fusion.fuse(v, [], [], top_k=0)
        assert results == []

    def test_scores_in_range(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("a", "vector", 0.8)]
        g = [_make_result("b", "graph", 0.6)]
        r = [_make_result("c", "recency", 0.4)]
        results = fusion.fuse(v, g, r, top_k=10)
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_non_overlapping_results_all_present(self) -> None:
        fusion = ScoreFusion()
        v = [_make_result("doc_v", "vector", 0.9)]
        g = [_make_result("doc_g", "graph", 0.7)]
        r = [_make_result("doc_r", "recency", 0.5)]
        results = fusion.fuse(v, g, r, top_k=10)
        contents = {result.content for result in results}
        assert "doc_v" in contents
        assert "doc_g" in contents
        assert "doc_r" in contents

    def test_weights_property(self) -> None:
        weights = FusionWeights(vector=0.5, graph=0.3, recency=0.2)
        fusion = ScoreFusion(weights=weights)
        assert fusion.weights is weights

    def test_vector_heavy_weights_prioritise_vector_results(self) -> None:
        weights = FusionWeights(vector=0.9, graph=0.05, recency=0.05)
        fusion = ScoreFusion(weights=weights)
        v = [_make_result("by_vector", "vector", 1.0)]
        g = [_make_result("by_graph", "graph", 0.1)]
        results = fusion.fuse(v, g, [], top_k=2)
        assert results[0].content == "by_vector"

    def test_recency_only_present_in_one_result(self) -> None:
        fusion = ScoreFusion()
        r = [_make_result("fresh", "recency", 0.95)]
        results = fusion.fuse([], [], r, top_k=5)
        assert len(results) == 1
        assert results[0].content == "fresh"
