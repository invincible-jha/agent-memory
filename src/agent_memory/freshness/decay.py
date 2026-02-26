"""Freshness-specific decay curves."""

from __future__ import annotations

import math


class FreshnessDecay:
    """Three freshness decay strategies on a single class, selectable by mode.

    Parameters
    ----------
    mode:
        ``"linear"``, ``"exponential"``, or ``"step"``.
    half_life_hours:
        Controls the rate of decay for linear and exponential modes.
    """

    MODES: frozenset[str] = frozenset({"linear", "exponential", "step"})

    _STEP_THRESHOLDS: list[tuple[float, float]] = [
        (6.0, 1.0),
        (24.0, 0.8),
        (72.0, 0.6),
        (168.0, 0.4),
        (720.0, 0.15),
        (2160.0, 0.05),
    ]

    def __init__(
        self,
        mode: str = "exponential",
        half_life_hours: float = 48.0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}")
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self._mode = mode
        self._half_life = half_life_hours
        self._lambda = math.log(2.0) / half_life_hours

    @property
    def mode(self) -> str:
        return self._mode

    def compute(self, age_hours: float) -> float:
        """Return freshness retention in [0, 1] for the given age."""
        age = max(0.0, age_hours)
        if self._mode == "linear":
            return max(0.0, 1.0 - age / (self._half_life * 2.0))
        if self._mode == "exponential":
            return math.exp(-self._lambda * age)
        # step
        result = 1.0
        for threshold, retention in self._STEP_THRESHOLDS:
            if age >= threshold:
                result = retention
            else:
                break
        return max(0.0, result)


__all__ = ["FreshnessDecay"]
