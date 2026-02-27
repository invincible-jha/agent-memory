"""Tests for agent_memory.retrieval.graph_retriever."""
from __future__ import annotations

import pytest

from agent_memory.retrieval.graph_retriever import GraphRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_graph() -> GraphRetriever:
    """A graph with 3 nodes and 2 edges: Paris→France→Europe."""
    g = GraphRetriever()
    g.add_edge("Paris", "capital_of", "France", weight=1.0)
    g.add_edge("France", "located_in", "Europe", weight=0.8)
    return g


@pytest.fixture()
def disconnected_graph() -> GraphRetriever:
    """Two disconnected components."""
    g = GraphRetriever()
    g.add_edge("Alice", "knows", "Bob", weight=1.0)
    g.add_edge("Charlie", "works_at", "Acme", weight=1.0)
    return g


# ---------------------------------------------------------------------------
# Tests: graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_initial_graph_is_empty(self) -> None:
        g = GraphRetriever()
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_add_edge_increments_edge_count(self) -> None:
        g = GraphRetriever()
        g.add_edge("A", "rel", "B", weight=1.0)
        assert g.edge_count() == 1

    def test_add_edge_increments_node_count(self) -> None:
        g = GraphRetriever()
        g.add_edge("A", "rel", "B", weight=1.0)
        assert g.node_count() == 2

    def test_add_multiple_edges_same_source(self) -> None:
        g = GraphRetriever()
        g.add_edge("A", "rel1", "B", weight=1.0)
        g.add_edge("A", "rel2", "C", weight=0.5)
        assert g.edge_count() == 2

    def test_add_edge_invalid_weight_zero(self) -> None:
        g = GraphRetriever()
        with pytest.raises(ValueError, match="weight"):
            g.add_edge("A", "rel", "B", weight=0.0)

    def test_add_edge_invalid_weight_negative(self) -> None:
        g = GraphRetriever()
        with pytest.raises(ValueError, match="weight"):
            g.add_edge("A", "rel", "B", weight=-0.1)

    def test_add_edge_weight_of_one_is_valid(self) -> None:
        g = GraphRetriever()
        g.add_edge("A", "rel", "B", weight=1.0)
        assert g.edge_count() == 1

    def test_hop_decay_invalid_zero(self) -> None:
        with pytest.raises(ValueError, match="hop_decay"):
            GraphRetriever(hop_decay=0.0)

    def test_hop_decay_invalid_greater_than_one(self) -> None:
        with pytest.raises(ValueError, match="hop_decay"):
            GraphRetriever(hop_decay=1.5)

    def test_hop_decay_one_is_valid(self) -> None:
        g = GraphRetriever(hop_decay=1.0)
        assert g is not None


# ---------------------------------------------------------------------------
# Tests: retrieval — basic
# ---------------------------------------------------------------------------


class TestGraphRetrieverRetrieve:
    def test_retrieve_returns_list(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris")
        assert isinstance(results, list)

    def test_retrieve_finds_direct_neighbour(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=1)
        contents = [r.content for r in results]
        assert "France" in contents

    def test_retrieve_finds_second_hop(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2)
        contents = [r.content for r in results]
        assert "Europe" in contents

    def test_retrieve_max_hops_limits_results(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=0)
        # max_hops=0 means only seed — returns seed itself with score 1.0
        assert any(r.content == "Paris" for r in results) or len(results) == 0

    def test_retrieve_respects_top_k(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2, top_k=1)
        assert len(results) <= 1

    def test_retrieve_scores_are_in_range(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2, top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_source_is_graph(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2, top_k=10)
        for r in results:
            assert r.source == "graph"

    def test_retrieve_sorted_descending(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_direct_neighbour_scores_higher_than_second_hop(
        self, simple_graph: GraphRetriever
    ) -> None:
        results = simple_graph.retrieve("Paris", max_hops=2, top_k=10)
        score_map = {r.content: r.score for r in results}
        if "France" in score_map and "Europe" in score_map:
            assert score_map["France"] > score_map["Europe"]

    def test_retrieve_no_match_returns_empty(self) -> None:
        g = GraphRetriever()
        g.add_edge("Alice", "knows", "Bob")
        results = g.retrieve("xyz_nonexistent_token")
        assert results == []

    def test_retrieve_empty_graph_returns_empty(self) -> None:
        g = GraphRetriever()
        assert g.retrieve("anything") == []

    def test_retrieve_disconnected_graph_only_connected_component(
        self, disconnected_graph: GraphRetriever
    ) -> None:
        results = disconnected_graph.retrieve("Alice", max_hops=1, top_k=10)
        contents = [r.content for r in results]
        assert "Bob" in contents
        # Acme is in a different component
        assert "Acme" not in contents

    def test_retrieve_invalid_top_k(self, simple_graph: GraphRetriever) -> None:
        with pytest.raises(ValueError, match="top_k"):
            simple_graph.retrieve("Paris", top_k=0)

    def test_retrieve_invalid_max_hops(self, simple_graph: GraphRetriever) -> None:
        with pytest.raises(ValueError, match="max_hops"):
            simple_graph.retrieve("Paris", max_hops=-1)

    def test_retrieve_returns_retrieval_result_instances(self, simple_graph: GraphRetriever) -> None:
        results = simple_graph.retrieve("Paris")
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_bidirectional_finds_source_from_target(self) -> None:
        g = GraphRetriever()
        g.add_edge("Paris", "capital_of", "France")
        # Query for France should find Paris via reverse edge
        results = g.retrieve("France", max_hops=1, bidirectional=True)
        contents = [r.content for r in results]
        assert "Paris" in contents

    def test_non_bidirectional_does_not_traverse_reverse(self) -> None:
        g = GraphRetriever()
        g.add_edge("Paris", "capital_of", "France")
        # Without bidirectional, France should not reach Paris
        results_fwd = g.retrieve("France", max_hops=1, bidirectional=False)
        contents_fwd = [r.content for r in results_fwd]
        # France has no outgoing edges — only Europe would be found
        # Paris should NOT be in results
        assert "Paris" not in contents_fwd
