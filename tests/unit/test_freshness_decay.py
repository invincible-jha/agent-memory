"""Unit tests for agent_memory.freshness.decay — FreshnessDecay."""

from __future__ import annotations

import math

import pytest

from agent_memory.freshness.decay import FreshnessDecay


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestFreshnessDecayInit:
    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            FreshnessDecay(mode="invalid")

    def test_zero_half_life_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="half_life_hours must be positive"):
            FreshnessDecay(mode="exponential", half_life_hours=0.0)

    def test_negative_half_life_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            FreshnessDecay(mode="linear", half_life_hours=-1.0)

    def test_mode_property(self) -> None:
        decay = FreshnessDecay(mode="linear")
        assert decay.mode == "linear"

    def test_all_valid_modes(self) -> None:
        for mode in ("linear", "exponential", "step"):
            decay = FreshnessDecay(mode=mode)
            assert decay.mode == mode

    def test_modes_frozenset_contains_three(self) -> None:
        assert len(FreshnessDecay.MODES) == 3


# ---------------------------------------------------------------------------
# Linear mode
# ---------------------------------------------------------------------------


class TestLinearMode:
    def test_zero_age_returns_one(self) -> None:
        decay = FreshnessDecay(mode="linear", half_life_hours=48.0)
        assert decay.compute(0.0) == pytest.approx(1.0)

    def test_at_half_life_returns_half(self) -> None:
        decay = FreshnessDecay(mode="linear", half_life_hours=48.0)
        assert decay.compute(48.0) == pytest.approx(0.5)

    def test_at_double_half_life_returns_zero(self) -> None:
        decay = FreshnessDecay(mode="linear", half_life_hours=48.0)
        assert decay.compute(96.0) == pytest.approx(0.0, abs=1e-9)

    def test_beyond_double_half_life_clamped_to_zero(self) -> None:
        decay = FreshnessDecay(mode="linear", half_life_hours=48.0)
        assert decay.compute(1000.0) == 0.0

    def test_negative_age_treated_as_zero(self) -> None:
        decay = FreshnessDecay(mode="linear", half_life_hours=48.0)
        assert decay.compute(-10.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Exponential mode
# ---------------------------------------------------------------------------


class TestExponentialMode:
    def test_zero_age_returns_one(self) -> None:
        decay = FreshnessDecay(mode="exponential", half_life_hours=48.0)
        assert decay.compute(0.0) == pytest.approx(1.0)

    def test_at_half_life_returns_approximately_half(self) -> None:
        decay = FreshnessDecay(mode="exponential", half_life_hours=48.0)
        assert decay.compute(48.0) == pytest.approx(0.5, rel=1e-4)

    def test_value_decays_toward_zero(self) -> None:
        # Exponential decay approaches 0 for very large ages;
        # floating-point may round to 0.0 at extreme values — that is acceptable.
        decay = FreshnessDecay(mode="exponential", half_life_hours=48.0)
        result = decay.compute(1_000_000.0)
        assert result >= 0.0

    def test_monotonically_decreasing(self) -> None:
        decay = FreshnessDecay(mode="exponential", half_life_hours=48.0)
        ages = [0.0, 12.0, 48.0, 200.0]
        values = [decay.compute(a) for a in ages]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_negative_age_treated_as_zero(self) -> None:
        decay = FreshnessDecay(mode="exponential", half_life_hours=24.0)
        assert decay.compute(-50.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Step mode
# ---------------------------------------------------------------------------


class TestStepMode:
    def test_zero_age_returns_one(self) -> None:
        decay = FreshnessDecay(mode="step", half_life_hours=48.0)
        assert decay.compute(0.0) == 1.0

    def test_below_first_threshold_returns_one(self) -> None:
        # Default first threshold is 6 hours → 1.0
        decay = FreshnessDecay(mode="step")
        assert decay.compute(3.0) == 1.0

    def test_at_first_threshold_returns_its_retention(self) -> None:
        decay = FreshnessDecay(mode="step")
        # Threshold at 6 hours → retention 1.0
        assert decay.compute(6.0) == 1.0

    def test_between_24h_and_72h_threshold(self) -> None:
        decay = FreshnessDecay(mode="step")
        # Between 24h (0.8) and 72h (0.6) → 0.8
        assert decay.compute(48.0) == 0.8

    def test_very_old_entry_returns_lowest_retention(self) -> None:
        decay = FreshnessDecay(mode="step")
        # Beyond 2160h → 0.05
        assert decay.compute(3000.0) == 0.05

    def test_result_never_below_zero(self) -> None:
        decay = FreshnessDecay(mode="step")
        assert decay.compute(999_999.0) >= 0.0

    def test_negative_age_treated_as_zero(self) -> None:
        decay = FreshnessDecay(mode="step")
        assert decay.compute(-10.0) == 1.0


# ---------------------------------------------------------------------------
# compute — output range guarantee
# ---------------------------------------------------------------------------


class TestComputeRange:
    @pytest.mark.parametrize("mode", ["linear", "exponential", "step"])
    @pytest.mark.parametrize("age", [0.0, 1.0, 10.0, 100.0, 1000.0])
    def test_output_always_in_zero_to_one(self, mode: str, age: float) -> None:
        decay = FreshnessDecay(mode=mode, half_life_hours=24.0)
        result = decay.compute(age)
        assert 0.0 <= result <= 1.0, f"mode={mode}, age={age}, result={result}"
