"""Tests for EpisodicConsolidator — E14.3."""

from __future__ import annotations

import pytest

from agent_memory.consolidation.consolidator import (
    ConsolidationResult,
    EpisodicConsolidator,
    _build_term_vector,
    _cosine_similarity,
    _tokenise,
)
from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(content: str, importance: float = 0.5) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.EPISODIC,
        importance_score=importance,
        source=MemorySource.USER_INPUT,
    )


def _make_semantic(content: str) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.SEMANTIC,
        importance_score=0.5,
    )


# ---------------------------------------------------------------------------
# Unit tests — utility functions
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_removes_stop_words(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_lowercases_input(self) -> None:
        tokens = _tokenise("Python Programming Language")
        assert "python" in tokens
        assert "programming" in tokens

    def test_removes_single_char_tokens(self) -> None:
        tokens = _tokenise("a b c def")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "def" in tokens

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenise("") == []

    def test_numbers_included(self) -> None:
        tokens = _tokenise("version 42 released")
        assert "42" in tokens
        assert "version" in tokens


class TestBuildTermVector:
    def test_normalised_to_term_frequency(self) -> None:
        tokens = ["cat", "cat", "dog"]
        vec = _build_term_vector(tokens)
        assert abs(vec["cat"] - 2 / 3) < 1e-9
        assert abs(vec["dog"] - 1 / 3) < 1e-9

    def test_empty_tokens_returns_empty(self) -> None:
        assert _build_term_vector([]) == {}


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        vec = {"cat": 0.5, "dog": 0.5}
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self) -> None:
        vec_a = {"cat": 1.0}
        vec_b = {"dog": 1.0}
        assert _cosine_similarity(vec_a, vec_b) == 0.0

    def test_empty_vectors_return_zero(self) -> None:
        assert _cosine_similarity({}, {"cat": 1.0}) == 0.0
        assert _cosine_similarity({"cat": 1.0}, {}) == 0.0

    def test_partial_overlap(self) -> None:
        vec_a = {"cat": 0.5, "dog": 0.5}
        vec_b = {"cat": 0.5, "fish": 0.5}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# EpisodicConsolidator tests
# ---------------------------------------------------------------------------


class TestEpisodicConsolidatorInit:
    def test_default_parameters(self) -> None:
        consolidator = EpisodicConsolidator()
        assert consolidator._similarity_threshold == 0.35
        assert consolidator._min_cluster_size == 2
        assert consolidator._max_centroid_terms == 10

    def test_custom_parameters(self) -> None:
        consolidator = EpisodicConsolidator(
            similarity_threshold=0.7,
            min_cluster_size=3,
            max_centroid_terms=5,
        )
        assert consolidator._similarity_threshold == 0.7
        assert consolidator._min_cluster_size == 3

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold"):
            EpisodicConsolidator(similarity_threshold=1.5)

    def test_invalid_threshold_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold"):
            EpisodicConsolidator(similarity_threshold=-0.1)

    def test_invalid_min_cluster_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_cluster_size"):
            EpisodicConsolidator(min_cluster_size=0)


class TestConsolidateEmpty:
    def test_empty_list_returns_empty_result(self) -> None:
        consolidator = EpisodicConsolidator()
        result = consolidator.consolidate([])
        assert isinstance(result, ConsolidationResult)
        assert result.episodes_processed == 0
        assert result.clusters == []
        assert result.semantic_entries == []

    def test_non_episodic_entries_skipped(self) -> None:
        consolidator = EpisodicConsolidator()
        semantic = _make_semantic("stable fact about Python")
        result = consolidator.consolidate([semantic])
        assert result.episodes_processed == 0


class TestConsolidateSingle:
    def test_single_episode_forms_no_cluster(self) -> None:
        consolidator = EpisodicConsolidator(min_cluster_size=2)
        episode = _make_episode("Agent processed user authentication request successfully.")
        result = consolidator.consolidate([episode])
        assert result.episodes_processed == 1
        assert len(result.clusters) == 1
        assert result.clusters_formed == 0
        assert result.semantic_entries == []


class TestConsolidateClustering:
    def test_similar_episodes_form_cluster(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.3, min_cluster_size=2)
        episodes = [
            _make_episode("Agent executed database query for user account records."),
            _make_episode("Agent ran database query returning user account data."),
            _make_episode("Agent performed database lookup for user account details."),
        ]
        result = consolidator.consolidate(episodes)
        assert result.episodes_processed == 3
        # Expect at least one cluster to form
        assert result.clusters_formed >= 1
        assert len(result.semantic_entries) >= 1

    def test_dissimilar_episodes_form_separate_clusters(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.9, min_cluster_size=2)
        episodes = [
            _make_episode("Astronomy telescope discovered new exoplanet orbit."),
            _make_episode("Recipe cook pasta boil water salt garlic tomato sauce."),
        ]
        result = consolidator.consolidate(episodes)
        # With high threshold, dissimilar episodes go to separate clusters
        assert len(result.clusters) >= 2
        assert result.clusters_formed == 0  # Each cluster has only 1 member

    def test_semantic_entries_have_correct_layer(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.2, min_cluster_size=2)
        episodes = [
            _make_episode("Python python programming language code script function."),
            _make_episode("Python language programming code module script execution."),
        ]
        result = consolidator.consolidate(episodes)
        for entry in result.semantic_entries:
            assert entry.layer == MemoryLayer.SEMANTIC
            assert entry.source == MemorySource.AGENT_INFERENCE

    def test_semantic_entry_metadata_contains_cluster_info(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.2, min_cluster_size=2)
        episodes = [
            _make_episode("Machine learning model training dataset features labels."),
            _make_episode("Machine learning model training data features prediction."),
        ]
        result = consolidator.consolidate(episodes)
        assert result.semantic_entries
        entry = result.semantic_entries[0]
        assert "cluster_id" in entry.metadata
        assert "episode_count" in entry.metadata

    def test_importance_propagated_to_semantic_entry(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.2, min_cluster_size=2)
        episodes = [
            _make_episode("Security audit log access control violation detected.", importance=0.8),
            _make_episode("Security audit access control log policy violation found.", importance=0.8),
        ]
        result = consolidator.consolidate(episodes)
        if result.semantic_entries:
            entry = result.semantic_entries[0]
            # Importance should be near 0.8 * 1.1 = 0.88, capped at 1.0
            assert 0.7 <= entry.importance_score <= 1.0


class TestSimilarityMethod:
    def test_identical_content_returns_high_similarity(self) -> None:
        consolidator = EpisodicConsolidator()
        entry_a = _make_episode("Python programming language features modules packages.")
        entry_b = _make_episode("Python programming language features modules packages.")
        sim = consolidator.similarity(entry_a, entry_b)
        assert sim > 0.9

    def test_different_content_returns_low_similarity(self) -> None:
        consolidator = EpisodicConsolidator()
        entry_a = _make_episode("Cooking pasta water boil garlic basil tomato sauce.")
        entry_b = _make_episode("Quantum physics electron spin orbital momentum energy.")
        sim = consolidator.similarity(entry_a, entry_b)
        assert sim < 0.3

    def test_similarity_symmetric(self) -> None:
        consolidator = EpisodicConsolidator()
        entry_a = _make_episode("Database query optimization index performance tuning.")
        entry_b = _make_episode("Database index performance query optimization plan.")
        sim_ab = consolidator.similarity(entry_a, entry_b)
        sim_ba = consolidator.similarity(entry_b, entry_a)
        assert abs(sim_ab - sim_ba) < 1e-9


class TestConsolidateResultFields:
    def test_result_cluster_ids_are_sequential(self) -> None:
        consolidator = EpisodicConsolidator(similarity_threshold=0.9)
        episodes = [
            _make_episode("Alpha beta gamma delta epsilon zeta eta theta."),
            _make_episode("One two three four five six seven eight nine ten."),
            _make_episode("Red blue green yellow orange purple violet indigo."),
        ]
        result = consolidator.consolidate(episodes)
        cluster_ids = [c.cluster_id for c in result.clusters]
        assert cluster_ids == list(range(len(cluster_ids)))

    def test_all_episodes_assigned_to_clusters(self) -> None:
        consolidator = EpisodicConsolidator()
        episodes = [_make_episode(f"Unique episode content number {i}.") for i in range(5)]
        result = consolidator.consolidate(episodes)
        assigned_ids = {eid for cluster in result.clusters for eid in cluster.episode_ids}
        all_ids = {ep.memory_id for ep in episodes}
        assert assigned_ids == all_ids
