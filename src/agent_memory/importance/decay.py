"""Decay curve implementations for memory importance over time."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class DecayCurve(ABC):
    """Abstract base for all decay functions.

    A decay curve maps an age (in hours) to a retention factor in [0, 1].
    ``decay(0)`` should return 1.0 and the value should monotonically
    decrease towards 0 as age increases.
    """

    @abstractmethod
    def decay(self, age_hours: float) -> float:
        """Return retention factor for a memory of the given age.

        Parameters
        ----------
        age_hours:
            How many hours have elapsed since the memory was created/verified.

        Returns
        -------
        float
            Value in [0.0, 1.0]; 1.0 means fully fresh, 0.0 means fully decayed.
        """


class LinearDecay(DecayCurve):
    """Importance decreases linearly to zero over ``half_life_hours`` * 2.

    Parameters
    ----------
    half_life_hours:
        Age at which retention drops to 0.5.  Defaults to 24 hours.
    """

    def __init__(self, half_life_hours: float = 24.0) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self._half_life = half_life_hours

    def decay(self, age_hours: float) -> float:
        """Linear ramp from 1.0 at age=0 to 0.0 at age=2*half_life."""
        full_decay_age = self._half_life * 2.0
        return max(0.0, 1.0 - age_hours / full_decay_age)


class ExponentialDecay(DecayCurve):
    """Importance decays exponentially: retention = exp(-lambda * age).

    Parameters
    ----------
    half_life_hours:
        Age at which retention drops to 0.5.  Defaults to 24 hours.
    """

    def __init__(self, half_life_hours: float = 24.0) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self._lambda = math.log(2.0) / half_life_hours

    def decay(self, age_hours: float) -> float:
        """Exponential retention: exp(-lambda * age)."""
        return math.exp(-self._lambda * max(0.0, age_hours))


class StepDecay(DecayCurve):
    """Retention drops in discrete steps at configurable age thresholds.

    Parameters
    ----------
    thresholds:
        Ordered list of ``(age_hours, retention_factor)`` tuples.  The
        last threshold with ``age_hours <= age`` is used.  If no threshold
        applies, returns 1.0.
    """

    _DEFAULT_THRESHOLDS: list[tuple[float, float]] = [
        (12.0, 1.0),
        (24.0, 0.75),
        (72.0, 0.5),
        (168.0, 0.25),
        (720.0, 0.1),
    ]

    def __init__(
        self,
        thresholds: list[tuple[float, float]] | None = None,
    ) -> None:
        raw = thresholds if thresholds is not None else self._DEFAULT_THRESHOLDS
        # Sort ascending by age
        self._thresholds = sorted(raw, key=lambda t: t[0])

    def decay(self, age_hours: float) -> float:
        """Step function: use the retention of the last applicable threshold."""
        result = 1.0
        for age_limit, retention in self._thresholds:
            if age_hours >= age_limit:
                result = retention
            else:
                break
        return max(0.0, result)


__all__ = ["DecayCurve", "LinearDecay", "ExponentialDecay", "StepDecay"]
