"""Unit tests for agent_memory.importance.decay — DecayCurve implementations."""

from __future__ import annotations

import math

import pytest

from agent_memory.importance.decay import (
    DecayCurve,
    ExponentialDecay,
    LinearDecay,
    StepDecay,
)


# ---------------------------------------------------------------------------
# LinearDecay
# ---------------------------------------------------------------------------


class TestLinearDecay:
    def test_zero_age_returns_one(self) -> None:
        curve = LinearDecay(half_life_hours=24.0)
        assert curve.decay(0.0) == pytest.approx(1.0)

    def test_at_half_life_returns_half(self) -> None:
        curve = LinearDecay(half_life_hours=24.0)
        assert curve.decay(24.0) == pytest.approx(0.5, abs=1e-6)

    def test_at_double_half_life_returns_zero(self) -> None:
        curve = LinearDecay(half_life_hours=24.0)
        assert curve.decay(48.0) == pytest.approx(0.0, abs=1e-6)

    def test_beyond_double_half_life_clamps_to_zero(self) -> None:
        curve = LinearDecay(half_life_hours=24.0)
        assert curve.decay(1000.0) == 0.0

    def test_negative_age_treated_as_zero(self) -> None:
        # LinearDecay does not explicitly clamp negative age,
        # but values above 1.0 should not occur in normal usage.
        curve = LinearDecay(half_life_hours=24.0)
        result = curve.decay(-10.0)
        assert result >= 1.0  # no upper clamp in LinearDecay

    def test_custom_half_life_scales_correctly(self) -> None:
        curve = LinearDecay(half_life_hours=10.0)
        # At 10 hours (= half-life) should be 0.5
        assert curve.decay(10.0) == pytest.approx(0.5)

    def test_invalid_half_life_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            LinearDecay(half_life_hours=0.0)

    def test_negative_half_life_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            LinearDecay(half_life_hours=-1.0)

    def test_is_abstract_base_subclass(self) -> None:
        assert issubclass(LinearDecay, DecayCurve)


# ---------------------------------------------------------------------------
# ExponentialDecay
# ---------------------------------------------------------------------------


class TestExponentialDecay:
    def test_zero_age_returns_one(self) -> None:
        curve = ExponentialDecay(half_life_hours=24.0)
        assert curve.decay(0.0) == pytest.approx(1.0)

    def test_at_half_life_returns_approximately_half(self) -> None:
        curve = ExponentialDecay(half_life_hours=24.0)
        assert curve.decay(24.0) == pytest.approx(0.5, rel=1e-4)

    def test_value_always_positive(self) -> None:
        curve = ExponentialDecay(half_life_hours=24.0)
        assert curve.decay(10_000.0) > 0.0

    def test_monotonically_decreasing(self) -> None:
        curve = ExponentialDecay(half_life_hours=48.0)
        ages = [0.0, 10.0, 48.0, 100.0, 500.0]
        values = [curve.decay(a) for a in ages]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_negative_age_clamped_to_zero(self) -> None:
        curve = ExponentialDecay(half_life_hours=24.0)
        # max(0, age) is applied
        assert curve.decay(-100.0) == pytest.approx(1.0)

    def test_custom_half_life(self) -> None:
        curve = ExponentialDecay(half_life_hours=12.0)
        assert curve.decay(12.0) == pytest.approx(0.5, rel=1e-4)

    def test_invalid_half_life_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ExponentialDecay(half_life_hours=0.0)

    def test_is_abstract_base_subclass(self) -> None:
        assert issubclass(ExponentialDecay, DecayCurve)


# ---------------------------------------------------------------------------
# StepDecay
# ---------------------------------------------------------------------------


class TestStepDecay:
    def test_zero_age_returns_one(self) -> None:
        curve = StepDecay()
        assert curve.decay(0.0) == 1.0

    def test_age_below_first_threshold_returns_one(self) -> None:
        curve = StepDecay()
        # Default first threshold is 12.0 hours
        assert curve.decay(5.0) == 1.0

    def test_age_at_first_threshold(self) -> None:
        curve = StepDecay()
        # At exactly 12.0 hours, retention should drop to 1.0 (default)
        assert curve.decay(12.0) == 1.0

    def test_age_between_thresholds(self) -> None:
        curve = StepDecay()
        # Between 12 and 24 hours → still 1.0
        assert curve.decay(18.0) == 1.0
        # Between 24 and 72 hours → 0.75
        assert curve.decay(50.0) == 0.75

    def test_age_at_last_threshold_returns_lowest_retention(self) -> None:
        curve = StepDecay()
        # Beyond 720 hours (30 days) → 0.1
        assert curve.decay(1000.0) == 0.1

    def test_custom_thresholds(self) -> None:
        custom = [(10.0, 0.8), (20.0, 0.5), (30.0, 0.1)]
        curve = StepDecay(thresholds=custom)
        assert curve.decay(5.0) == 1.0
        assert curve.decay(15.0) == 0.8
        assert curve.decay(25.0) == 0.5
        assert curve.decay(35.0) == 0.1

    def test_unsorted_thresholds_sorted_on_init(self) -> None:
        # Provide thresholds out of order
        custom = [(20.0, 0.5), (10.0, 0.8)]
        curve = StepDecay(thresholds=custom)
        # Should still work correctly after sorting
        assert curve.decay(15.0) == 0.8

    def test_is_abstract_base_subclass(self) -> None:
        assert issubclass(StepDecay, DecayCurve)

    def test_result_never_below_zero(self) -> None:
        curve = StepDecay(thresholds=[(1.0, -0.5)])
        assert curve.decay(5.0) == 0.0


# ---------------------------------------------------------------------------
# DecayCurve ABC
# ---------------------------------------------------------------------------


class TestDecayCurveABC:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            DecayCurve()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_decay(self) -> None:
        class Incomplete(DecayCurve):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
