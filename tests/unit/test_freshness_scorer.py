"""Unit tests for agent_memory.freshness.scorer — FreshnessScorer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.freshness.decay import FreshnessDecay
from agent_memory.freshness.scorer import FreshnessScorer, _age_hours
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_entry(
    content: str = "test",
    last_accessed_offset_hours: float = 0.0,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create an entry with last_accessed shifted backwards by offset_hours."""
    now = _utcnow()
    last_accessed = now - timedelta(hours=last_accessed_offset_hours)
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.EPISODIC,
        last_accessed=last_accessed,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# _age_hours helper
# ---------------------------------------------------------------------------


class TestAgeHoursHelper:
    def test_zero_age_for_now(self) -> None:
        now = _utcnow()
        age = _age_hours(now, now)
        assert age == pytest.approx(0.0, abs=1e-3)

    def test_one_hour_ago(self) -> None:
        now = _utcnow()
        one_hour_ago = now - timedelta(hours=1)
        age = _age_hours(one_hour_ago, now)
        assert age == pytest.approx(1.0, rel=1e-3)

    def test_negative_age_clamped_to_zero(self) -> None:
        now = _utcnow()
        future = now + timedelta(hours=5)
        age = _age_hours(future, now)
        assert age == 0.0

    def test_naive_reference_treated_as_utc(self) -> None:
        naive_now = datetime.utcnow()  # naive
        aware_now = datetime.now(timezone.utc)
        age = _age_hours(naive_now, aware_now)
        assert age == pytest.approx(0.0, abs=0.1)

    def test_now_defaults_to_current_time(self) -> None:
        one_hour_ago = _utcnow() - timedelta(hours=1)
        age = _age_hours(one_hour_ago)  # no 'now' argument
        assert age == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# FreshnessScorer.score
# ---------------------------------------------------------------------------


class TestFreshnessScore:
    def test_fresh_entry_scores_near_one(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=0.0)
        score = scorer.score(entry)
        assert score >= 0.9

    def test_old_entry_scores_lower_than_fresh(self) -> None:
        scorer = FreshnessScorer()
        fresh = _make_entry(last_accessed_offset_hours=0.0)
        old = _make_entry(last_accessed_offset_hours=200.0)
        assert scorer.score(fresh) > scorer.score(old)

    def test_score_in_zero_to_one_range(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=48.0)
        score = scorer.score(entry)
        assert 0.0 <= score <= 1.0

    def test_score_uses_last_verified_metadata_when_present(self) -> None:
        """Entry with very recent last_verified should score higher than last_accessed."""
        scorer = FreshnessScorer()
        now_iso = _utcnow().isoformat()
        entry_with_meta = _make_entry(
            last_accessed_offset_hours=500.0,
            metadata={"last_verified": now_iso},
        )
        entry_without = _make_entry(last_accessed_offset_hours=500.0)
        # last_verified=now should give a much higher score
        assert scorer.score(entry_with_meta) > scorer.score(entry_without)

    def test_invalid_last_verified_falls_back_to_last_accessed(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(
            last_accessed_offset_hours=0.0,
            metadata={"last_verified": "not-a-valid-datetime"},
        )
        # Should not raise; falls back gracefully
        score = scorer.score(entry)
        assert 0.0 <= score <= 1.0

    def test_custom_now_parameter(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=0.0)
        # Pretend 'now' is 24 hours in the future — entry will look 24h old
        future_now = _utcnow() + timedelta(hours=24)
        score_future = scorer.score(entry, now=future_now)
        score_present = scorer.score(entry)
        assert score_present > score_future

    def test_custom_decay_curve(self) -> None:
        linear_decay = FreshnessDecay(mode="linear", half_life_hours=24.0)
        scorer = FreshnessScorer(decay=linear_decay)
        entry = _make_entry(last_accessed_offset_hours=24.0)
        score = scorer.score(entry)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_score_rounded_to_six_decimal_places(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=1.0)
        score = scorer.score(entry)
        # Check it is not an excessively long float
        assert len(str(score).rstrip("0").split(".")[-1]) <= 6


# ---------------------------------------------------------------------------
# FreshnessScorer.score_and_update
# ---------------------------------------------------------------------------


class TestScoreAndUpdate:
    def test_returns_new_entry_not_same_object(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry()
        updated = scorer.score_and_update(entry)
        assert updated is not entry

    def test_updated_freshness_score_matches_score_call(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=12.0)
        expected = scorer.score(entry)
        updated = scorer.score_and_update(entry)
        assert updated.freshness_score == pytest.approx(expected, rel=1e-6)

    def test_original_entry_unchanged(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry()
        original_score = entry.freshness_score
        scorer.score_and_update(entry)
        assert entry.freshness_score == original_score

    def test_content_preserved_after_update(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(content="important memory content")
        updated = scorer.score_and_update(entry)
        assert updated.content == "important memory content"

    def test_updated_score_is_in_valid_range(self) -> None:
        scorer = FreshnessScorer()
        entry = _make_entry(last_accessed_offset_hours=100.0)
        updated = scorer.score_and_update(entry)
        assert 0.0 <= updated.freshness_score <= 1.0
