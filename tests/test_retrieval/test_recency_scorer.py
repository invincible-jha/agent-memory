"""Tests for agent_memory.retrieval.recency_scorer."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.retrieval.graph_retriever import RetrievalResult
from agent_memory.retrieval.recency_scorer import RecencyScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> RecencyScorer:
    return RecencyScorer(half_life_hours=168.0)  # 7 days


@pytest.fixture()
def reference() -> datetime:
    return datetime(2026, 1, 8, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestRecencyScorerConstruction:
    def test_default_half_life(self) -> None:
        s = RecencyScorer()
        assert s.half_life_hours == 168.0

    def test_custom_half_life(self) -> None:
        s = RecencyScorer(half_life_hours=24.0)
        assert s.half_life_hours == 24.0

    def test_invalid_half_life_zero(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            RecencyScorer(half_life_hours=0.0)

    def test_invalid_half_life_negative(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            RecencyScorer(half_life_hours=-10.0)


# ---------------------------------------------------------------------------
# Tests: score method
# ---------------------------------------------------------------------------


class TestRecencyScorerScore:
    def test_same_time_scores_one(self, scorer: RecencyScorer, reference: datetime) -> None:
        score = scorer.score(reference, reference=reference)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_future_scores_one(self, scorer: RecencyScorer, reference: datetime) -> None:
        future = reference + timedelta(hours=24)
        score = scorer.score(future, reference=reference)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_one_half_life_scores_half(self, reference: datetime) -> None:
        scorer = RecencyScorer(half_life_hours=168.0)
        old = reference - timedelta(hours=168)
        score = scorer.score(old, reference=reference)
        assert score == pytest.approx(0.5, abs=1e-4)

    def test_two_half_lives_scores_quarter(self, reference: datetime) -> None:
        scorer = RecencyScorer(half_life_hours=24.0)
        old = reference - timedelta(hours=48)
        score = scorer.score(old, reference=reference)
        assert score == pytest.approx(0.25, abs=1e-4)

    def test_score_decreases_with_age(self, scorer: RecencyScorer, reference: datetime) -> None:
        young = reference - timedelta(hours=1)
        old = reference - timedelta(hours=1000)
        assert scorer.score(young, reference=reference) > scorer.score(old, reference=reference)

    def test_score_in_range(self, scorer: RecencyScorer, reference: datetime) -> None:
        for hours in [0, 24, 168, 720, 8760]:
            ts = reference - timedelta(hours=hours)
            score = scorer.score(ts, reference=reference)
            assert 0.0 <= score <= 1.0

    def test_score_uses_now_as_default_reference(self) -> None:
        scorer = RecencyScorer()
        # An item one week old should score ~0.5
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        score = scorer.score(one_week_ago)
        assert 0.4 < score < 0.6  # Allow some tolerance for test execution time

    def test_score_naive_datetime_treated_as_utc(self, reference: datetime) -> None:
        scorer = RecencyScorer(half_life_hours=168.0)
        naive_old = datetime(2026, 1, 1, 12, 0, 0)  # naive
        # Should not raise
        score = scorer.score(naive_old, reference=reference)
        assert 0.0 <= score <= 1.0

    def test_exponential_decay_formula(self, reference: datetime) -> None:
        """Verify score = 2^(-age/half_life)."""
        half_life = 48.0
        scorer = RecencyScorer(half_life_hours=half_life)
        age_hours = 12.0
        ts = reference - timedelta(hours=age_hours)
        expected = math.pow(2.0, -age_hours / half_life)
        score = scorer.score(ts, reference=reference)
        assert score == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: rank method
# ---------------------------------------------------------------------------


class TestRecencyScorerRank:
    def test_rank_returns_list(self, scorer: RecencyScorer, reference: datetime) -> None:
        items = [("content A", reference - timedelta(hours=10))]
        results = scorer.rank(items, reference=reference)
        assert isinstance(results, list)

    def test_rank_returns_retrieval_results(self, scorer: RecencyScorer, reference: datetime) -> None:
        items = [("content A", reference - timedelta(hours=1))]
        results = scorer.rank(items, reference=reference)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_rank_source_is_recency(self, scorer: RecencyScorer, reference: datetime) -> None:
        items = [("content A", reference - timedelta(hours=1))]
        results = scorer.rank(items, reference=reference)
        assert all(r.source == "recency" for r in results)

    def test_rank_sorted_descending(self, scorer: RecencyScorer, reference: datetime) -> None:
        items = [
            ("old content", reference - timedelta(days=7)),
            ("new content", reference - timedelta(hours=1)),
            ("medium content", reference - timedelta(days=2)),
        ]
        results = scorer.rank(items, reference=reference)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_newer_item_ranked_higher(self, scorer: RecencyScorer, reference: datetime) -> None:
        items = [
            ("old", reference - timedelta(days=30)),
            ("new", reference - timedelta(hours=1)),
        ]
        results = scorer.rank(items, reference=reference)
        assert results[0].content == "new"

    def test_rank_empty_input(self, scorer: RecencyScorer) -> None:
        results = scorer.rank([])
        assert results == []

    def test_rank_metadata_contains_timestamp(self, scorer: RecencyScorer, reference: datetime) -> None:
        ts = reference - timedelta(hours=5)
        items = [("content", ts)]
        results = scorer.rank(items, reference=reference)
        assert "timestamp" in results[0].metadata
