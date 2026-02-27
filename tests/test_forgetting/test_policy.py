"""Tests for ForgettingPolicy and ForgettingScheduler — E14.5."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.forgetting.policy import (
    DecayFunction,
    DecayRecord,
    ForgettingPolicy,
    ForgettingScheduler,
    PolicyConfig,
    SweepResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(offset_hours: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=offset_hours)


def _make_policy(
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL,
    half_life_hours: float = 24.0,
    forget_threshold: float = 0.1,
    reinforce_on_access: bool = True,
    safety_critical_exempt: bool = True,
) -> ForgettingPolicy:
    config = PolicyConfig(
        decay_function=decay_function,
        half_life_hours=half_life_hours,
        forget_threshold=forget_threshold,
        reinforce_on_access=reinforce_on_access,
        safety_critical_exempt=safety_critical_exempt,
    )
    return ForgettingPolicy(config)


# ---------------------------------------------------------------------------
# PolicyConfig validation
# ---------------------------------------------------------------------------


class TestPolicyConfig:
    def test_default_config_valid(self) -> None:
        config = PolicyConfig()
        assert config.decay_function == DecayFunction.EXPONENTIAL
        assert config.half_life_hours == 24.0
        assert config.forget_threshold == 0.1

    def test_invalid_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            PolicyConfig(half_life_hours=0.0)

    def test_invalid_half_life_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours"):
            PolicyConfig(half_life_hours=-1.0)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="forget_threshold"):
            PolicyConfig(forget_threshold=1.5)

    def test_frozen(self) -> None:
        config = PolicyConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.half_life_hours = 48.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DecayRecord
# ---------------------------------------------------------------------------


class TestDecayRecord:
    def test_elapsed_hours_zero_when_just_created(self) -> None:
        record = DecayRecord(entry_id="test")
        elapsed = record.elapsed_hours()
        assert elapsed < 0.01  # Should be nearly instant

    def test_elapsed_hours_with_past_reinforcement(self) -> None:
        past = _utc(-2.0)  # 2 hours ago
        record = DecayRecord(entry_id="test", last_reinforced=past)
        elapsed = record.elapsed_hours(_utc())
        assert 1.9 < elapsed < 2.1


# ---------------------------------------------------------------------------
# ForgettingPolicy — registration
# ---------------------------------------------------------------------------


class TestForgettingPolicyRegistration:
    def test_register_entry(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        assert policy.tracked_count() == 1

    def test_register_duplicate_raises(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        with pytest.raises(ValueError, match="already registered"):
            policy.register("e1")

    def test_deregister_entry(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        assert policy.deregister("e1") is True
        assert policy.tracked_count() == 0

    def test_deregister_nonexistent_returns_false(self) -> None:
        policy = _make_policy()
        assert policy.deregister("nonexistent") is False


# ---------------------------------------------------------------------------
# Score computation — exponential
# ---------------------------------------------------------------------------


class TestExponentialDecay:
    def test_score_at_time_zero_is_initial(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.EXPONENTIAL, half_life_hours=24.0)
        policy.register("e1", initial_score=1.0)
        now = _utc()
        # Entry just registered, elapsed is ~0
        score = policy.compute_score("e1", now=now)
        assert score is not None
        assert 0.99 < score <= 1.0

    def test_score_at_half_life_is_approximately_half(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.EXPONENTIAL, half_life_hours=10.0)
        past = _utc(-10.0)  # 10 hours ago
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score is not None
        assert 0.45 < score < 0.55

    def test_score_never_below_zero(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.EXPONENTIAL, half_life_hours=1.0)
        past = _utc(-1000.0)  # Far in the past
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score is not None
        assert score >= 0.0

    def test_nonexistent_entry_returns_none(self) -> None:
        policy = _make_policy()
        assert policy.compute_score("nonexistent") is None


# ---------------------------------------------------------------------------
# Score computation — linear
# ---------------------------------------------------------------------------


class TestLinearDecay:
    def test_score_decreases_linearly(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.LINEAR, half_life_hours=100.0)
        past = _utc(-50.0)  # 50 hours ago
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score is not None
        # After half the half-life, score should be near 0.5
        assert 0.45 < score < 0.56

    def test_score_bottoms_out_at_zero(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.LINEAR, half_life_hours=10.0)
        past = _utc(-50.0)  # Well past half_life
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score == 0.0


# ---------------------------------------------------------------------------
# Score computation — step
# ---------------------------------------------------------------------------


class TestStepDecay:
    def test_score_unchanged_before_threshold(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.STEP, half_life_hours=24.0)
        past = _utc(-12.0)  # 12 hours ago, under 24h threshold
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score == 1.0

    def test_score_drops_to_zero_at_threshold(self) -> None:
        policy = _make_policy(decay_function=DecayFunction.STEP, half_life_hours=24.0)
        past = _utc(-25.0)  # 25 hours ago, past 24h threshold
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score == 0.0


# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------


class TestReinforcement:
    def test_reinforce_resets_score_to_one(self) -> None:
        policy = _make_policy(half_life_hours=1.0)
        past = _utc(-5.0)  # 5 hours ago — heavily decayed
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        score_before = policy.compute_score("e1", now=_utc())
        assert score_before is not None and score_before < 0.5
        policy.reinforce("e1")
        score_after = policy.compute_score("e1", now=_utc())
        assert score_after is not None and score_after > 0.95

    def test_reinforce_nonexistent_returns_false(self) -> None:
        policy = _make_policy()
        assert policy.reinforce("nonexistent") is False

    def test_reinforce_disabled_does_not_reset(self) -> None:
        policy = _make_policy(reinforce_on_access=False, half_life_hours=1.0)
        past = _utc(-5.0)
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        policy.reinforce("e1")
        # Score should not be reset since reinforce_on_access=False
        score = policy.compute_score("e1", now=_utc())
        assert score is not None and score < 0.5


# ---------------------------------------------------------------------------
# Safety critical exemption
# ---------------------------------------------------------------------------


class TestSafetyCriticalExempt:
    def test_safety_critical_never_forgotten(self) -> None:
        policy = _make_policy(half_life_hours=1.0, forget_threshold=0.5)
        past = _utc(-100.0)
        policy.register("e1", safety_critical=True)
        policy._records["e1"].last_reinforced = past
        score = policy.compute_score("e1", now=_utc())
        assert score == 1.0  # Exempt from decay

    def test_non_safety_critical_forgotten(self) -> None:
        policy = _make_policy(
            decay_function=DecayFunction.STEP,
            half_life_hours=1.0,
            forget_threshold=0.5,
        )
        past = _utc(-2.0)
        policy.register("e1", safety_critical=False)
        policy._records["e1"].last_reinforced = past
        forgotten = policy.forgotten_entries(now=_utc())
        assert "e1" in forgotten


# ---------------------------------------------------------------------------
# Purge forgotten
# ---------------------------------------------------------------------------


class TestPurgeForgotten:
    def test_purge_removes_forgotten_entries(self) -> None:
        policy = _make_policy(
            decay_function=DecayFunction.STEP,
            half_life_hours=1.0,
            forget_threshold=0.5,
        )
        past = _utc(-2.0)
        policy.register("e1")
        policy.register("e2")  # Not decayed
        policy._records["e1"].last_reinforced = past
        purged = policy.purge_forgotten(now=_utc())
        assert "e1" in purged
        assert policy.tracked_count() == 1

    def test_all_scores_returns_dict(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        policy.register("e2")
        scores = policy.all_scores()
        assert "e1" in scores
        assert "e2" in scores
        assert all(0.0 <= s <= 1.0 for s in scores.values())


# ---------------------------------------------------------------------------
# ForgettingScheduler
# ---------------------------------------------------------------------------


class TestForgettingScheduler:
    def test_sweep_returns_result(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        scheduler = ForgettingScheduler(policy)
        result = scheduler.sweep()
        assert isinstance(result, SweepResult)
        assert result.entries_before == 1

    def test_sweep_purges_forgotten_entries(self) -> None:
        policy = _make_policy(
            decay_function=DecayFunction.STEP,
            half_life_hours=1.0,
            forget_threshold=0.5,
        )
        past = _utc(-2.0)
        policy.register("e1")
        policy._records["e1"].last_reinforced = past
        scheduler = ForgettingScheduler(policy)
        result = scheduler.sweep(now=_utc())
        assert "e1" in result.forgotten_ids
        assert result.purged_count == 1
        assert result.entries_after == 0

    def test_sweep_history_accumulated(self) -> None:
        policy = _make_policy()
        scheduler = ForgettingScheduler(policy)
        scheduler.sweep()
        scheduler.sweep()
        assert len(scheduler.sweep_history()) == 2

    def test_total_purged_accumulates(self) -> None:
        policy = _make_policy(
            decay_function=DecayFunction.STEP,
            half_life_hours=1.0,
            forget_threshold=0.5,
        )
        scheduler = ForgettingScheduler(policy)

        for i in range(3):
            entry_id = f"e{i}"
            policy.register(entry_id)
            policy._records[entry_id].last_reinforced = _utc(-2.0)

        scheduler.sweep(now=_utc())
        assert scheduler.total_purged() == 3

    def test_sweep_result_fields(self) -> None:
        policy = _make_policy()
        policy.register("e1")
        policy.register("e2")
        scheduler = ForgettingScheduler(policy)
        result = scheduler.sweep()
        assert result.entries_before == 2
        assert result.entries_after <= 2
        assert isinstance(result.sweep_time, datetime)
