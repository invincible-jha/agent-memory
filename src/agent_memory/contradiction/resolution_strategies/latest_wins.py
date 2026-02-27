"""Latest-wins resolution strategy.

:class:`LatestWinsStrategy` resolves a contradiction by always preferring
the more recently timestamped memory entry.  When both entries share the
same timestamp, ``existing`` is kept as a tie-breaker (stable behaviour).

This is the simplest possible resolution policy and is appropriate when the
agent's most recent information is assumed to supersede older data.

Usage
-----
>>> from agent_memory.contradiction.models import MemoryEntry
>>> from agent_memory.contradiction.resolution_strategies.latest_wins import LatestWinsStrategy
>>> from datetime import datetime, timezone, timedelta
>>> existing = MemoryEntry(content="Paris is in France.", source="user_input")
>>> newer = MemoryEntry(content="Paris is in Germany.", source="user_input")
>>> strategy = LatestWinsStrategy()
>>> result = strategy.resolve(existing, newer)
>>> result.action in ("replace", "keep_both")
True
"""
from __future__ import annotations

from datetime import timezone

from agent_memory.contradiction.models import MemoryEntry
from agent_memory.contradiction.resolution import ResolutionResult, ResolutionStrategyBase  # noqa: E501


class LatestWinsStrategy(ResolutionStrategyBase):
    """Always use the most recently timestamped memory entry.

    The winner is the entry with the later ``timestamp``.  If both entries
    have identical timestamps, *existing* is kept (deterministic tie-break).

    The action for this strategy is always ``"replace"`` — the older entry
    is discarded in favour of the newer one.

    Parameters
    ----------
    None

    Example
    -------
    >>> strategy = LatestWinsStrategy()
    >>> strategy.name
    'latest_wins'
    """

    @property
    def name(self) -> str:
        return "latest_wins"

    def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
        """Resolve by keeping the newer of the two entries.

        Parameters
        ----------
        existing:
            The entry already in memory.
        new:
            The incoming entry that contradicts *existing*.

        Returns
        -------
        ResolutionResult
            ``action="replace"`` if *new* is more recent; otherwise
            ``action="replace"`` is returned with *existing* as content
            (stable tie-break: keep existing when timestamps are equal).
        """
        ts_existing = _to_utc(existing.timestamp)
        ts_new = _to_utc(new.timestamp)

        if ts_new > ts_existing:
            # Incoming entry is newer — replace existing with new
            return ResolutionResult(
                strategy_used=self.name,
                action="replace",
                resolved_content=new.content,
                requires_user_input=False,
                explanation=(
                    f"Incoming entry (ts={new.timestamp.isoformat()}) is more recent than "
                    f"existing entry (ts={existing.timestamp.isoformat()}). "
                    f"Replaced with newer content."
                ),
            )
        else:
            # Existing entry is newer or equal — keep existing
            return ResolutionResult(
                strategy_used=self.name,
                action="replace",
                resolved_content=existing.content,
                requires_user_input=False,
                explanation=(
                    f"Existing entry (ts={existing.timestamp.isoformat()}) is not older "
                    f"than incoming entry (ts={new.timestamp.isoformat()}). "
                    f"Retained existing content."
                ),
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_utc(dt: object) -> object:
    """Normalise a datetime to UTC-aware.  Returns input unchanged if not datetime."""
    from datetime import datetime

    if not isinstance(dt, datetime):
        return dt
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
