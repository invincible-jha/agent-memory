"""Unit tests for agent_memory.retrieval.ranking — ResultRanker and RankingWeights."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.retrieval.ranking import (
    RankingWeights,
    ResultRanker,
    _sigmoid_scale,
    _tokenise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    importance: float = 0.5,
    freshness: float = 1.0,
    source: MemorySource = MemorySource.AGENT_INFERENCE,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.SEMANTIC,
        importance_score=importance,
        freshness_score=freshness,
        source=source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RankingWeights
# ---------------------------------------------------------------------------


class TestRankingWeights:
    def test_default_weights(self) -> None:
        weights = RankingWeights()
        assert weights.text_relevance == pytest.approx(0.4)
        assert weights.importance == pytest.approx(0.3)
        assert weights.freshness == pytest.approx(0.2)
        assert weights.provenance == pytest.approx(0.1)

    def test_total_property(self) -> None:
        weights = RankingWeights(
            text_relevance=0.4, importance=0.3, freshness=0.2, provenance=0.1
        )
        assert weights.total == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        weights = RankingWeights(
            text_relevance=1.0, importance=0.0, freshness=0.0, provenance=0.0
        )
        assert weights.total == pytest.approx(1.0)

    def test_negative_text_relevance_raises(self) -> None:
        with pytest.raises(ValueError, match="text_relevance"):
            RankingWeights(text_relevance=-0.1)

    def test_negative_importance_raises(self) -> None:
        with pytest.raises(ValueError, match="importance"):
            RankingWeights(importance=-0.1)

    def test_negative_freshness_raises(self) -> None:
        with pytest.raises(ValueError, match="freshness"):
            RankingWeights(freshness=-0.1)

    def test_negative_provenance_raises(self) -> None:
        with pytest.raises(ValueError, match="provenance"):
            RankingWeights(provenance=-0.1)

    def test_all_zero_weights_valid(self) -> None:
        weights = RankingWeights(
            text_relevance=0.0, importance=0.0, freshness=0.0, provenance=0.0
        )
        assert weights.total == 0.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_extracts_alphanumeric_tokens(self) -> None:
        tokens = _tokenise("Hello World 123")
        assert tokens == ["hello", "world", "123"]

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenise("") == []

    def test_punctuation_stripped(self) -> None:
        tokens = _tokenise("hello, world! python.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase_normalised(self) -> None:
        tokens = _tokenise("Python PYTHON python")
        assert all(t == "python" for t in tokens)


class TestSigmoidScale:
    def test_zero_input_returns_zero(self) -> None:
        assert _sigmoid_scale(0.0) == 0.0

    def test_small_positive_returns_small_positive(self) -> None:
        result = _sigmoid_scale(0.1)
        assert 0.0 < result < 1.0

    def test_half_maps_to_near_half(self) -> None:
        result = _sigmoid_scale(0.5)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_large_value_approaches_one(self) -> None:
        result = _sigmoid_scale(1.0)
        assert result > 0.9


# ---------------------------------------------------------------------------
# ResultRanker.score
# ---------------------------------------------------------------------------


class TestResultRankerScore:
    def test_score_in_zero_to_one_range(self) -> None:
        ranker = ResultRanker()
        entry = _make_entry("hello world")
        score = ranker.score(entry, "hello")
        assert 0.0 <= score <= 1.0

    def test_all_zero_weights_returns_zero(self) -> None:
        weights = RankingWeights(
            text_relevance=0.0, importance=0.0, freshness=0.0, provenance=0.0
        )
        ranker = ResultRanker(weights=weights)
        entry = _make_entry("hello world")
        assert ranker.score(entry, "hello") == 0.0

    def test_exact_content_match_scores_higher_than_no_match(self) -> None:
        ranker = ResultRanker()
        e_match = _make_entry("python is a great language")
        e_no_match = _make_entry("the weather is sunny today")
        assert ranker.score(e_match, "python") > ranker.score(e_no_match, "python")

    def test_higher_importance_gives_higher_score(self) -> None:
        ranker = ResultRanker()
        e_high = _make_entry("neutral content", importance=0.9)
        e_low = _make_entry("neutral content", importance=0.1)
        assert ranker.score(e_high, "neutral") > ranker.score(e_low, "neutral")

    def test_higher_freshness_gives_higher_score(self) -> None:
        ranker = ResultRanker()
        e_fresh = _make_entry("neutral content", freshness=1.0)
        e_stale = _make_entry("neutral content", freshness=0.1)
        assert ranker.score(e_fresh, "neutral") > ranker.score(e_stale, "neutral")

    def test_tool_output_source_scores_higher_than_inference(self) -> None:
        weights = RankingWeights(
            text_relevance=0.0, importance=0.0, freshness=0.0, provenance=1.0
        )
        ranker = ResultRanker(weights=weights)
        e_tool = _make_entry("x", source=MemorySource.TOOL_OUTPUT)
        e_agent = _make_entry("x", source=MemorySource.AGENT_INFERENCE)
        assert ranker.score(e_tool, "") > ranker.score(e_agent, "")

    def test_provenance_reliability_from_metadata_used(self) -> None:
        weights = RankingWeights(
            text_relevance=0.0, importance=0.0, freshness=0.0, provenance=1.0
        )
        ranker = ResultRanker(weights=weights)
        e_high = _make_entry("x", metadata={"provenance_reliability": "0.99"})
        e_low = _make_entry("x", metadata={"provenance_reliability": "0.01"})
        assert ranker.score(e_high, "") > ranker.score(e_low, "")

    def test_invalid_provenance_metadata_falls_back_to_source(self) -> None:
        ranker = ResultRanker()
        entry = _make_entry("x", metadata={"provenance_reliability": "not-a-number"})
        score = ranker.score(entry, "x")
        assert 0.0 <= score <= 1.0

    def test_empty_query_still_returns_valid_score(self) -> None:
        ranker = ResultRanker()
        entry = _make_entry("some content here")
        score = ranker.score(entry, "")
        assert 0.0 <= score <= 1.0

    def test_score_rounded_to_six_decimal_places(self) -> None:
        ranker = ResultRanker()
        entry = _make_entry("test content")
        score = ranker.score(entry, "test")
        assert len(str(score).rstrip("0").split(".")[-1]) <= 6


# ---------------------------------------------------------------------------
# ResultRanker.rank
# ---------------------------------------------------------------------------


class TestResultRankerRank:
    def test_rank_returns_list_of_tuples(self) -> None:
        ranker = ResultRanker()
        entries = [
            (_make_entry("hello world"), "hello"),
            (_make_entry("another entry"), "hello"),
        ]
        ranked = ranker.rank(entries)
        assert isinstance(ranked, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)

    def test_rank_sorted_descending_by_score(self) -> None:
        ranker = ResultRanker()
        e_high = _make_entry("python is the best language", importance=0.9)
        e_low = _make_entry("random unrelated content here", importance=0.1)
        ranked = ranker.rank([(e_high, "python"), (e_low, "python")])
        assert ranked[0][1].memory_id == e_high.memory_id

    def test_rank_empty_list_returns_empty(self) -> None:
        ranker = ResultRanker()
        assert ranker.rank([]) == []

    def test_rank_scores_in_zero_to_one_range(self) -> None:
        ranker = ResultRanker()
        entries = [(_make_entry(f"content {i}"), "content") for i in range(5)]
        ranked = ranker.rank(entries)
        for score, _ in ranked:
            assert 0.0 <= score <= 1.0

    def test_rank_uses_per_entry_query(self) -> None:
        """Each entry is ranked against its own query string."""
        ranker = ResultRanker()
        e1 = _make_entry("python code execution")
        e2 = _make_entry("database query optimization")
        ranked = ranker.rank([(e1, "python"), (e2, "database")])
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# ResultRanker.weights property
# ---------------------------------------------------------------------------


class TestWeightsProperty:
    def test_weights_property_returns_ranking_weights(self) -> None:
        weights = RankingWeights(text_relevance=0.6, importance=0.4, freshness=0.0, provenance=0.0)
        ranker = ResultRanker(weights=weights)
        assert ranker.weights is weights

    def test_default_weights_property(self) -> None:
        ranker = ResultRanker()
        assert isinstance(ranker.weights, RankingWeights)
