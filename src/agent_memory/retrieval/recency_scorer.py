"""Time-weighted decay scoring for memory retrieval.

:class:`RecencyScorer` assigns a score in ``[0.0, 1.0]`` to a piece of
content based on its age relative to a reference time.  The score uses
exponential decay:

    score = 2^(-age_in_hours / half_life_hours)

where ``age_in_hours`` is the time elapsed since the item's timestamp
and ``half_life_hours`` is the configurable half-life (default: 168 h,
equivalent to one week).

This means:
- Items at exactly ``t = reference`` score 1.0 (freshest).
- Items one half-life old score 0.5.
- Items two half-lives old score 0.25.
- Items score monotonically approach 0 as age increases.

Usage
-----
>>> from datetime import datetime, timezone, timedelta
>>> scorer = RecencyScorer(half_life_hours=24.0)
>>> reference = datetime(2026, 1, 8, tzinfo=timezone.utc)
>>> old = datetime(2026, 1, 7, tzinfo=timezone.utc)  # 24 h old
>>> score = scorer.score(old, reference=reference)
>>> abs(score - 0.5) < 0.001
True
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from agent_memory.retrieval.graph_retriever import RetrievalResult


class RecencyScorer:
    """Time-weighted exponential decay scorer.

    Parameters
    ----------
    half_life_hours:
        The half-life of the exponential decay in hours.  An item
        ``half_life_hours`` old scores 0.5; an item twice as old scores
        0.25.  Defaults to ``168.0`` (7 days).

    Example
    -------
    >>> scorer = RecencyScorer(half_life_hours=168.0)
    >>> now = datetime.now(timezone.utc)
    >>> scorer.score(now) == pytest.approx(1.0, abs=0.001)  # type: ignore
    True
    """

    def __init__(self, half_life_hours: float = 168.0) -> None:
        if half_life_hours <= 0.0:
            raise ValueError(
                f"half_life_hours must be positive, got {half_life_hours}"
            )
        self._half_life_hours = half_life_hours

    @property
    def half_life_hours(self) -> float:
        """The configured half-life in hours."""
        return self._half_life_hours

    def score(
        self,
        timestamp: datetime,
        reference: datetime | None = None,
    ) -> float:
        """Compute the recency score for a timestamp.

        Parameters
        ----------
        timestamp:
            When the item was created or last updated.
        reference:
            The reference point for age computation.  Defaults to the
            current UTC time when ``None``.

        Returns
        -------
        float
            Score in ``[0.0, 1.0]``.  Items in the future receive 1.0.
        """
        ref = reference if reference is not None else datetime.now(timezone.utc)
        ts_utc = _to_utc(timestamp)
        ref_utc = _to_utc(ref)

        age_seconds = (ref_utc - ts_utc).total_seconds()
        if age_seconds <= 0.0:
            # Future or same-time item — maximum freshness
            return 1.0

        age_hours = age_seconds / 3600.0
        raw = math.pow(2.0, -age_hours / self._half_life_hours)
        return round(min(1.0, max(0.0, raw)), 6)

    def rank(
        self,
        items: list[tuple[str, datetime]],
        reference: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Score and rank a list of ``(content, timestamp)`` pairs.

        Parameters
        ----------
        items:
            Each element is ``(content, timestamp)`` where ``content``
            is the text of the memory item and ``timestamp`` is when it
            was created.
        reference:
            Reference time for age computation.  Defaults to now.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by recency score descending.  Each entry has
            ``source="recency"``.
        """
        ref = reference if reference is not None else datetime.now(timezone.utc)
        results: list[RetrievalResult] = [
            RetrievalResult(
                content=content,
                source="recency",
                score=self.score(ts, reference=ref),
                metadata={"timestamp": ts},
            )
            for content, ts in items
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime) -> datetime:
    """Normalise *dt* to UTC-aware.  Assumes UTC if no tzinfo is set."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
