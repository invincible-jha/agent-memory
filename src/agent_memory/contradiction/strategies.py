"""Resolution strategy implementations for the contradiction engine.

Each strategy is a plain class with a single ``apply`` method that accepts a
``Contradiction`` and returns a ``Resolution``.  Strategies are stateless and
can be shared freely across calls.

Available strategies
--------------------
- ``KeepNewerStrategy``   — keep the most-recently timestamped entry
- ``KeepBothStrategy``    — keep both entries, annotating both with a
                            contradiction marker
- ``FlagForReviewStrategy`` — mark both entries as pending human review without
                              archiving either
- ``MergeStrategy``       — synthesise a new merged entry combining both
                            content strings and archive the originals

All strategies implement the ``ResolutionStrategy`` protocol (structural
typing — no ABC import required by callers).
"""

from __future__ import annotations

from typing import Protocol

from agent_memory.contradiction.models import (
    Contradiction,
    MemoryEntry,
    Resolution,
)


# ---------------------------------------------------------------------------
# Protocol (structural interface)
# ---------------------------------------------------------------------------


class ResolutionStrategyProtocol(Protocol):
    """Structural protocol satisfied by all strategy classes."""

    def apply(self, contradiction: Contradiction) -> Resolution:
        ...


# ---------------------------------------------------------------------------
# KeepNewerStrategy
# ---------------------------------------------------------------------------


class KeepNewerStrategy:
    """Keep the most recently created entry and archive the older one.

    When both entries share the same timestamp the tie is broken by preferring
    ``entry_a`` (stable, deterministic behaviour).
    """

    def apply(self, contradiction: Contradiction) -> Resolution:
        """Apply recency-wins logic to the contradiction.

        Parameters
        ----------
        contradiction:
            The contradiction to resolve.

        Returns
        -------
        Resolution
            Resolution that retains the newer entry and archives the older.
        """
        entry_a = contradiction.entry_a
        entry_b = contradiction.entry_b

        # Normalise timestamps to UTC-aware for safe comparison
        ts_a = _to_utc(entry_a.timestamp)
        ts_b = _to_utc(entry_b.timestamp)

        if ts_a >= ts_b:
            kept = entry_a
            archived = entry_b
            rationale = (
                f"Kept entry '{entry_a.id}' (timestamp {entry_a.timestamp.isoformat()}) "
                f"over '{entry_b.id}' (timestamp {entry_b.timestamp.isoformat()}) "
                f"— recency wins."
            )
        else:
            kept = entry_b
            archived = entry_a
            rationale = (
                f"Kept entry '{entry_b.id}' (timestamp {entry_b.timestamp.isoformat()}) "
                f"over '{entry_a.id}' (timestamp {entry_a.timestamp.isoformat()}) "
                f"— recency wins."
            )

        return Resolution(
            strategy_used="keep_newer",
            kept_entries=(kept,),
            archived_entries=(archived,),
            notes=rationale,
        )


# ---------------------------------------------------------------------------
# KeepBothStrategy
# ---------------------------------------------------------------------------


class KeepBothStrategy:
    """Keep both entries in the memory store with a contradiction annotation.

    Neither entry is archived; instead the ``notes`` field carries a marker
    that both entries have been flagged as contradicting each other.  This is
    useful when the agent needs to retain the full information history.
    """

    def apply(self, contradiction: Contradiction) -> Resolution:
        """Apply keep-both logic to the contradiction.

        Parameters
        ----------
        contradiction:
            The contradiction to resolve.

        Returns
        -------
        Resolution
            Resolution that retains both entries with an annotation note.
        """
        entry_a = contradiction.entry_a
        entry_b = contradiction.entry_b

        annotation = (
            f"Both entries retained with contradiction annotation. "
            f"Type: {contradiction.contradiction_type.value}. "
            f"Similarity: {contradiction.similarity_score:.4f}. "
            f"Explanation: {contradiction.explanation}"
        )

        return Resolution(
            strategy_used="keep_both_with_context",
            kept_entries=(entry_a, entry_b),
            archived_entries=(),
            notes=annotation,
        )


# ---------------------------------------------------------------------------
# FlagForReviewStrategy
# ---------------------------------------------------------------------------


class FlagForReviewStrategy:
    """Mark both entries as pending human review without archiving either.

    This is a conservative strategy — no data is discarded and no automatic
    decision is made.  The ``notes`` field signals that downstream systems
    should surface the contradiction to a human reviewer.
    """

    def apply(self, contradiction: Contradiction) -> Resolution:
        """Flag both entries for manual review.

        Parameters
        ----------
        contradiction:
            The contradiction to resolve.

        Returns
        -------
        Resolution
            Resolution that keeps both entries and flags them for review.
        """
        entry_a = contradiction.entry_a
        entry_b = contradiction.entry_b

        review_note = (
            f"PENDING_REVIEW: entries '{entry_a.id}' and '{entry_b.id}' "
            f"contradict each other ({contradiction.contradiction_type.value}, "
            f"confidence={contradiction.confidence:.4f}). "
            f"Human review required. Explanation: {contradiction.explanation}"
        )

        return Resolution(
            strategy_used="flag_for_review",
            kept_entries=(entry_a, entry_b),
            archived_entries=(),
            notes=review_note,
        )


# ---------------------------------------------------------------------------
# MergeStrategy
# ---------------------------------------------------------------------------


class MergeStrategy:
    """Create a merged entry combining both content strings, archive the originals.

    The merged entry is a new ``MemoryEntry`` whose ``content`` is the
    concatenation of the two originals separated by a merge marker, and whose
    ``source`` is ``"merged"``.  The merged entry takes the timestamp of the
    newer original so it is naturally prioritised by recency-based downstream
    queries.

    The two source entries are moved to ``archived_entries``.
    """

    MERGE_SEPARATOR: str = " [MERGED WITH] "

    def apply(self, contradiction: Contradiction) -> Resolution:
        """Synthesise a merged entry from the two contradicting entries.

        Parameters
        ----------
        contradiction:
            The contradiction to resolve.

        Returns
        -------
        Resolution
            Resolution with a single new merged entry in ``kept_entries``
            and both originals in ``archived_entries``.
        """
        entry_a = contradiction.entry_a
        entry_b = contradiction.entry_b

        # Use the newer entry's timestamp for the merge
        ts_a = _to_utc(entry_a.timestamp)
        ts_b = _to_utc(entry_b.timestamp)
        merge_timestamp = entry_a.timestamp if ts_a >= ts_b else entry_b.timestamp

        merged_content = entry_a.content + self.MERGE_SEPARATOR + entry_b.content
        merged_source = f"merged:{entry_a.source}+{entry_b.source}"

        merged_entry = MemoryEntry(
            content=merged_content,
            timestamp=merge_timestamp,
            source=merged_source,
        )

        notes = (
            f"Merged entries '{entry_a.id}' and '{entry_b.id}' into '{merged_entry.id}'. "
            f"Contradiction type: {contradiction.contradiction_type.value}. "
            f"Explanation: {contradiction.explanation}"
        )

        return Resolution(
            strategy_used="merge",
            kept_entries=(merged_entry,),
            archived_entries=(entry_a, entry_b),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_utc(dt_value: object) -> object:
    """Normalise a datetime to UTC-aware.  Returns the input unchanged if not a datetime."""
    from datetime import datetime, timezone

    if not isinstance(dt_value, datetime):
        return dt_value
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


__all__ = [
    "ResolutionStrategyProtocol",
    "KeepNewerStrategy",
    "KeepBothStrategy",
    "FlagForReviewStrategy",
    "MergeStrategy",
]
