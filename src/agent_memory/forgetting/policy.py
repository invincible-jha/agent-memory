"""ForgettingPolicy — configurable decay-based memory forgetting.

Supports three commodity decay functions:
- Exponential: score *= base^(elapsed_hours)   — steepest decay
- Linear:      score -= rate * elapsed_hours    — constant rate of forgetting
- Step:        score resets to 0 after a fixed number of hours

Access reinforcement resets the decay timer (restores to full score).
ForgettingScheduler runs periodic sweeps and identifies entries to forget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence


class DecayFunction(str, Enum):
    """Decay function to use in the forgetting policy."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for a ForgettingPolicy.

    Parameters
    ----------
    decay_function:
        Which decay function to apply.
    half_life_hours:
        For exponential: hours until score halves.
        For linear: hours until score reaches 0 (starting from 1.0).
        For step: hours until score drops to 0.
    forget_threshold:
        Score below which an entry is considered forgotten (0 – 1).
    reinforce_on_access:
        If True, accessing an entry resets its decay timer (returns score to 1.0).
    safety_critical_exempt:
        If True, entries marked safety_critical are never forgotten.
    """

    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    half_life_hours: float = 24.0
    forget_threshold: float = 0.1
    reinforce_on_access: bool = True
    safety_critical_exempt: bool = True

    def __post_init__(self) -> None:
        if self.half_life_hours <= 0:
            raise ValueError(
                f"half_life_hours must be > 0, got {self.half_life_hours}"
            )
        if not 0.0 <= self.forget_threshold <= 1.0:
            raise ValueError(
                f"forget_threshold must be in [0, 1], got {self.forget_threshold}"
            )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DecayRecord:
    """Tracks the decay state for a single memory entry.

    Parameters
    ----------
    entry_id:
        The memory entry being tracked.
    current_score:
        Current decay score in [0, 1]. Starts at 1.0.
    last_reinforced:
        UTC datetime when this entry was last accessed/reinforced.
    safety_critical:
        If True, this entry should never be forgotten.
    """

    entry_id: str
    current_score: float = 1.0
    last_reinforced: datetime = field(default_factory=_utcnow)
    safety_critical: bool = False

    def elapsed_hours(self, now: Optional[datetime] = None) -> float:
        """Return hours elapsed since last reinforcement."""
        reference = now or _utcnow()
        delta = reference - self.last_reinforced
        return max(0.0, delta.total_seconds() / 3600.0)


class ForgettingPolicy:
    """Manages decay scores for a collection of memory entries.

    Applies the configured decay function to compute current scores and
    identifies entries that have decayed past the forget threshold.

    Parameters
    ----------
    config:
        Policy configuration. Defaults to exponential decay with 24h half-life.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self._config = config or PolicyConfig()
        self._records: dict[str, DecayRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        entry_id: str,
        safety_critical: bool = False,
        initial_score: float = 1.0,
    ) -> None:
        """Register a memory entry for decay tracking.

        Parameters
        ----------
        entry_id:
            Unique identifier for the entry.
        safety_critical:
            Whether this entry is exempt from forgetting.
        initial_score:
            Starting decay score (default 1.0).

        Raises
        ------
        ValueError
            If the entry is already registered.
        """
        if entry_id in self._records:
            raise ValueError(f"Entry {entry_id!r} is already registered.")
        self._records[entry_id] = DecayRecord(
            entry_id=entry_id,
            current_score=min(1.0, max(0.0, initial_score)),
            safety_critical=safety_critical,
        )

    def deregister(self, entry_id: str) -> bool:
        """Remove an entry from decay tracking.

        Returns True if removed, False if not found.
        """
        if entry_id not in self._records:
            return False
        del self._records[entry_id]
        return True

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    def reinforce(self, entry_id: str) -> bool:
        """Record an access event, resetting the decay timer.

        Parameters
        ----------
        entry_id:
            The entry that was accessed.

        Returns
        -------
        bool
            True if reinforced, False if entry not found or
            ``reinforce_on_access`` is disabled.
        """
        record = self._records.get(entry_id)
        if record is None:
            return False
        if self._config.reinforce_on_access:
            record.current_score = 1.0
            record.last_reinforced = _utcnow()
        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_score(
        self,
        entry_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[float]:
        """Compute the current decay score for an entry.

        Parameters
        ----------
        entry_id:
            The entry to score.
        now:
            Reference time (defaults to UTC now).

        Returns
        -------
        float | None
            Current decay score in [0, 1], or None if entry not found.
        """
        record = self._records.get(entry_id)
        if record is None:
            return None
        if record.safety_critical and self._config.safety_critical_exempt:
            return 1.0
        elapsed = record.elapsed_hours(now)
        return self._apply_decay(record.current_score, elapsed)

    def update_scores(self, now: Optional[datetime] = None) -> None:
        """Update the stored score for all records using current time.

        This is called by the scheduler before sweeping for forgotten entries.
        """
        reference = now or _utcnow()
        for record in self._records.values():
            if record.safety_critical and self._config.safety_critical_exempt:
                continue
            elapsed = record.elapsed_hours(reference)
            new_score = self._apply_decay(record.current_score, elapsed)
            record.current_score = new_score
            record.last_reinforced = reference  # Reset timer after snapshot

    # ------------------------------------------------------------------
    # Forgetting
    # ------------------------------------------------------------------

    def forgotten_entries(self, now: Optional[datetime] = None) -> list[str]:
        """Return IDs of entries whose decay score is at or below the threshold.

        Parameters
        ----------
        now:
            Reference time (defaults to UTC now).

        Returns
        -------
        list[str]
            Entry IDs that should be forgotten.
        """
        result: list[str] = []
        reference = now or _utcnow()
        for entry_id, record in self._records.items():
            score = self.compute_score(entry_id, now=reference)
            if score is not None and score <= self._config.forget_threshold:
                result.append(entry_id)
        return result

    def purge_forgotten(self, now: Optional[datetime] = None) -> list[str]:
        """Remove forgotten entries from tracking and return their IDs.

        Parameters
        ----------
        now:
            Reference time (defaults to UTC now).

        Returns
        -------
        list[str]
            Entry IDs that were purged.
        """
        to_purge = self.forgotten_entries(now=now)
        for entry_id in to_purge:
            self._records.pop(entry_id, None)
        return to_purge

    def tracked_count(self) -> int:
        """Return the number of entries currently being tracked."""
        return len(self._records)

    def all_scores(self, now: Optional[datetime] = None) -> dict[str, float]:
        """Return current decay scores for all tracked entries.

        Parameters
        ----------
        now:
            Reference time (defaults to UTC now).

        Returns
        -------
        dict[str, float]
            Maps entry_id to current decay score.
        """
        reference = now or _utcnow()
        return {
            entry_id: self.compute_score(entry_id, now=reference) or 0.0
            for entry_id in self._records
        }

    # ------------------------------------------------------------------
    # Private decay functions
    # ------------------------------------------------------------------

    def _apply_decay(self, current_score: float, elapsed_hours: float) -> float:
        """Apply the configured decay function to a score."""
        if elapsed_hours <= 0.0:
            return current_score

        config = self._config
        half_life = config.half_life_hours

        if config.decay_function == DecayFunction.EXPONENTIAL:
            # score(t) = score_0 * (0.5)^(t / half_life)
            decay_factor = math.pow(0.5, elapsed_hours / half_life)
            new_score = current_score * decay_factor

        elif config.decay_function == DecayFunction.LINEAR:
            # score(t) = score_0 - (1/half_life) * elapsed_hours
            # half_life_hours represents time to go from 1.0 to 0.0
            rate = 1.0 / half_life
            new_score = current_score - rate * elapsed_hours

        else:  # STEP
            # Score drops to 0 after half_life_hours have elapsed
            new_score = 0.0 if elapsed_hours >= half_life else current_score

        return max(0.0, min(1.0, new_score))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Result of a forgetting scheduler sweep.

    Parameters
    ----------
    forgotten_ids:
        Entry IDs that were purged in this sweep.
    sweep_time:
        UTC time when the sweep ran.
    entries_before:
        Number of tracked entries before the sweep.
    entries_after:
        Number of tracked entries after the sweep.
    """

    forgotten_ids: list[str]
    sweep_time: datetime
    entries_before: int
    entries_after: int

    @property
    def purged_count(self) -> int:
        """Number of entries purged."""
        return len(self.forgotten_ids)


class ForgettingScheduler:
    """Runs periodic forgetting sweeps against a ForgettingPolicy.

    Callers are responsible for calling ``sweep()`` at appropriate intervals
    (e.g. on a background timer or before each memory retrieval).

    Parameters
    ----------
    policy:
        The ForgettingPolicy to sweep.
    """

    def __init__(self, policy: ForgettingPolicy) -> None:
        self._policy = policy
        self._sweep_history: list[SweepResult] = []

    def sweep(self, now: Optional[datetime] = None) -> SweepResult:
        """Run a single forgetting sweep.

        Updates all decay scores and purges entries below the forget threshold.

        Parameters
        ----------
        now:
            Reference time (defaults to UTC now).

        Returns
        -------
        SweepResult
            Details of what was purged.
        """
        reference = now or _utcnow()
        entries_before = self._policy.tracked_count()
        self._policy.update_scores(now=reference)
        forgotten_ids = self._policy.purge_forgotten(now=reference)
        entries_after = self._policy.tracked_count()

        result = SweepResult(
            forgotten_ids=forgotten_ids,
            sweep_time=reference,
            entries_before=entries_before,
            entries_after=entries_after,
        )
        self._sweep_history.append(result)
        return result

    def sweep_history(self) -> list[SweepResult]:
        """Return the list of past sweep results."""
        return list(self._sweep_history)

    def total_purged(self) -> int:
        """Return total entries purged across all sweeps."""
        return sum(r.purged_count for r in self._sweep_history)


__all__ = [
    "DecayFunction",
    "PolicyConfig",
    "ForgettingPolicy",
    "ForgettingScheduler",
    "DecayRecord",
    "SweepResult",
]
