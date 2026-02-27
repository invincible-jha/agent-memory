"""Contradiction resolver — strategies for resolving detected contradictions.

Two resolver classes are exported:

ContradictionResolver (original, backward-compatible)
    Works with the Pydantic ``ContradictionPair`` / ``MemoryEntry`` types from
    the existing codebase.  Supports ``RECENCY_WINS``, ``SOURCE_PRIORITY``,
    ``CONFIDENCE_BASED``, and ``MANUAL`` strategies.

SimpleContradictionResolver
    Lightweight resolver that works with the frozen dataclass types from
    ``contradiction.models`` (``Contradiction``, ``Resolution``).  Dispatches
    to the strategy implementations in ``contradiction.strategies`` using a
    plain string strategy name.  Suitable for use with ``TFIDFContradictionDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from agent_memory.contradiction.models import Contradiction, Resolution
from agent_memory.contradiction.report import ContradictionPair
from agent_memory.contradiction.strategies import (
    FlagForReviewStrategy,
    KeepBothStrategy,
    KeepNewerStrategy,
    MergeStrategy,
)
from agent_memory.memory.types import MemoryEntry


# ---------------------------------------------------------------------------
# Original resolver (backward-compatible)
# ---------------------------------------------------------------------------


class ResolutionStrategy(str, Enum):
    """Available strategies for resolving contradictions."""

    RECENCY_WINS = "recency_wins"
    SOURCE_PRIORITY = "source_priority"
    CONFIDENCE_BASED = "confidence_based"
    MANUAL = "manual"


@dataclass
class ResolutionResult:
    """The outcome of resolving a contradiction pair."""

    contradiction_pair: ContradictionPair
    strategy_used: ResolutionStrategy
    winner_id: str
    loser_id: str
    rationale: str
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_manual_review: bool = False


# Source priority ordering (higher index = higher priority)
_SOURCE_PRIORITY: dict[str, int] = {
    "agent_inference": 0,
    "user_input": 1,
    "external_api": 2,
    "document": 3,
    "tool_output": 4,
}


class ContradictionResolver:
    """Resolve contradictions between memory entries using configurable strategies.

    Parameters
    ----------
    default_strategy:
        The strategy to apply when ``resolve()`` is called without an
        explicit override. Defaults to ``RECENCY_WINS``.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.RECENCY_WINS,
    ) -> None:
        self._default_strategy = default_strategy

    @property
    def default_strategy(self) -> ResolutionStrategy:
        return self._default_strategy

    def resolve(
        self,
        contradiction: ContradictionPair,
        entry_a: MemoryEntry,
        entry_b: MemoryEntry,
        strategy: Optional[ResolutionStrategy] = None,
        manual_winner_id: Optional[str] = None,
    ) -> ResolutionResult:
        """Resolve a contradiction between two memory entries.

        Parameters
        ----------
        contradiction:
            The detected contradiction pair to resolve.
        entry_a:
            The first memory entry in the pair.
        entry_b:
            The second memory entry in the pair.
        strategy:
            Override strategy; falls back to ``default_strategy`` if None.
        manual_winner_id:
            Required when ``strategy`` is ``MANUAL``; must be the ID of
            either entry_a or entry_b.

        Returns
        -------
        ResolutionResult
            Structured resolution outcome with winner, loser, and rationale.

        Raises
        ------
        ValueError
            If entry IDs don't match the contradiction pair, or if MANUAL
            strategy is used without a valid ``manual_winner_id``.
        """
        active_strategy = strategy or self._default_strategy

        # Validate entries correspond to the contradiction pair
        pair_ids = {contradiction.entry_a_id, contradiction.entry_b_id}
        if entry_a.memory_id not in pair_ids or entry_b.memory_id not in pair_ids:
            raise ValueError(
                f"Entry IDs {entry_a.memory_id!r} and {entry_b.memory_id!r} "
                f"do not match contradiction pair IDs {pair_ids!r}."
            )

        if active_strategy == ResolutionStrategy.RECENCY_WINS:
            return self._resolve_recency(contradiction, entry_a, entry_b)
        if active_strategy == ResolutionStrategy.SOURCE_PRIORITY:
            return self._resolve_source_priority(contradiction, entry_a, entry_b)
        if active_strategy == ResolutionStrategy.CONFIDENCE_BASED:
            return self._resolve_confidence(contradiction, entry_a, entry_b)
        if active_strategy == ResolutionStrategy.MANUAL:
            return self._resolve_manual(contradiction, entry_a, entry_b, manual_winner_id)

        # Fallback — should never be reached with the enum guard above
        return self._resolve_recency(contradiction, entry_a, entry_b)

    def resolve_batch(
        self,
        pairs: list[tuple[ContradictionPair, MemoryEntry, MemoryEntry]],
        strategy: Optional[ResolutionStrategy] = None,
    ) -> list[ResolutionResult]:
        """Resolve multiple contradiction pairs using the same strategy.

        Parameters
        ----------
        pairs:
            List of ``(ContradictionPair, entry_a, entry_b)`` tuples.
        strategy:
            Strategy to apply to all pairs; falls back to default if None.

        Returns
        -------
        list[ResolutionResult]
            One result per input pair, in the same order.
        """
        return [
            self.resolve(pair, entry_a, entry_b, strategy=strategy)
            for pair, entry_a, entry_b in pairs
        ]

    # ------------------------------------------------------------------
    # Private strategy implementations
    # ------------------------------------------------------------------

    def _resolve_recency(
        self,
        contradiction: ContradictionPair,
        entry_a: MemoryEntry,
        entry_b: MemoryEntry,
    ) -> ResolutionResult:
        """The more recently created entry wins."""
        created_a = _to_utc(entry_a.created_at)
        created_b = _to_utc(entry_b.created_at)

        if created_a >= created_b:
            winner, loser = entry_a, entry_b
            rationale = (
                f"Entry {entry_a.memory_id!r} is newer "
                f"({entry_a.created_at.isoformat()} >= {entry_b.created_at.isoformat()})."
            )
        else:
            winner, loser = entry_b, entry_a
            rationale = (
                f"Entry {entry_b.memory_id!r} is newer "
                f"({entry_b.created_at.isoformat()} > {entry_a.created_at.isoformat()})."
            )

        return ResolutionResult(
            contradiction_pair=contradiction,
            strategy_used=ResolutionStrategy.RECENCY_WINS,
            winner_id=winner.memory_id,
            loser_id=loser.memory_id,
            rationale=rationale,
        )

    def _resolve_source_priority(
        self,
        contradiction: ContradictionPair,
        entry_a: MemoryEntry,
        entry_b: MemoryEntry,
    ) -> ResolutionResult:
        """The entry from the more authoritative source wins."""
        priority_a = _SOURCE_PRIORITY.get(entry_a.source.value, 0)
        priority_b = _SOURCE_PRIORITY.get(entry_b.source.value, 0)

        if priority_a >= priority_b:
            winner, loser = entry_a, entry_b
            rationale = (
                f"Source {entry_a.source.value!r} (priority={priority_a}) >= "
                f"{entry_b.source.value!r} (priority={priority_b})."
            )
        else:
            winner, loser = entry_b, entry_a
            rationale = (
                f"Source {entry_b.source.value!r} (priority={priority_b}) > "
                f"{entry_a.source.value!r} (priority={priority_a})."
            )

        return ResolutionResult(
            contradiction_pair=contradiction,
            strategy_used=ResolutionStrategy.SOURCE_PRIORITY,
            winner_id=winner.memory_id,
            loser_id=loser.memory_id,
            rationale=rationale,
        )

    def _resolve_confidence(
        self,
        contradiction: ContradictionPair,
        entry_a: MemoryEntry,
        entry_b: MemoryEntry,
    ) -> ResolutionResult:
        """The entry with higher importance_score * freshness_score wins."""
        score_a = entry_a.composite_score
        score_b = entry_b.composite_score

        if score_a >= score_b:
            winner, loser = entry_a, entry_b
            rationale = (
                f"Entry {entry_a.memory_id!r} has higher composite score "
                f"({score_a:.4f} >= {score_b:.4f})."
            )
        else:
            winner, loser = entry_b, entry_a
            rationale = (
                f"Entry {entry_b.memory_id!r} has higher composite score "
                f"({score_b:.4f} > {score_a:.4f})."
            )

        return ResolutionResult(
            contradiction_pair=contradiction,
            strategy_used=ResolutionStrategy.CONFIDENCE_BASED,
            winner_id=winner.memory_id,
            loser_id=loser.memory_id,
            rationale=rationale,
        )

    def _resolve_manual(
        self,
        contradiction: ContradictionPair,
        entry_a: MemoryEntry,
        entry_b: MemoryEntry,
        manual_winner_id: Optional[str],
    ) -> ResolutionResult:
        """The caller explicitly designates the winner."""
        valid_ids = {entry_a.memory_id, entry_b.memory_id}
        if manual_winner_id not in valid_ids:
            return ResolutionResult(
                contradiction_pair=contradiction,
                strategy_used=ResolutionStrategy.MANUAL,
                winner_id="",
                loser_id="",
                rationale=(
                    f"Manual resolution requested but winner_id {manual_winner_id!r} "
                    f"is not one of {sorted(valid_ids)!r}. Requires human review."
                ),
                requires_manual_review=True,
            )

        winner_id = manual_winner_id
        loser_id = entry_b.memory_id if winner_id == entry_a.memory_id else entry_a.memory_id
        return ResolutionResult(
            contradiction_pair=contradiction,
            strategy_used=ResolutionStrategy.MANUAL,
            winner_id=winner_id,
            loser_id=loser_id,
            rationale=f"Manually designated winner: {winner_id!r}.",
        )


# ---------------------------------------------------------------------------
# SimpleContradictionResolver (works with new lightweight model types)
# ---------------------------------------------------------------------------

# Registry mapping strategy name strings to strategy class instances.
# This is intentionally built lazily (at first use) to keep module import cost low.
_STRATEGY_REGISTRY: dict[str, object] = {}


def _get_strategy_registry() -> dict[str, object]:
    global _STRATEGY_REGISTRY
    if not _STRATEGY_REGISTRY:
        _STRATEGY_REGISTRY = {
            "keep_newer": KeepNewerStrategy(),
            "keep_older": _KeepOlderStrategy(),
            "keep_both_with_context": KeepBothStrategy(),
            "flag_for_review": FlagForReviewStrategy(),
            "merge": MergeStrategy(),
        }
    return _STRATEGY_REGISTRY


class _KeepOlderStrategy:
    """Internal strategy: keeps the oldest entry and archives the newer one."""

    def apply(self, contradiction: Contradiction) -> Resolution:
        entry_a = contradiction.entry_a
        entry_b = contradiction.entry_b

        from datetime import timezone as tz

        def _utc(dt_val: object) -> object:
            from datetime import datetime

            if not isinstance(dt_val, datetime):
                return dt_val
            if dt_val.tzinfo is None:
                return dt_val.replace(tzinfo=tz.utc)
            return dt_val.astimezone(tz.utc)

        ts_a = _utc(entry_a.timestamp)
        ts_b = _utc(entry_b.timestamp)

        from datetime import datetime

        if isinstance(ts_a, datetime) and isinstance(ts_b, datetime):
            if ts_a <= ts_b:
                kept = entry_a
                archived = entry_b
            else:
                kept = entry_b
                archived = entry_a
        else:
            kept = entry_a
            archived = entry_b

        return Resolution(
            strategy_used="keep_older",
            kept_entries=(kept,),
            archived_entries=(archived,),
            notes=(
                f"Kept older entry '{kept.id}' and archived newer entry '{archived.id}'."
            ),
        )


_VALID_STRATEGY_NAMES: frozenset[str] = frozenset(
    ["keep_newer", "keep_older", "keep_both_with_context", "flag_for_review", "merge"]
)


class SimpleContradictionResolver:
    """Lightweight resolver for the frozen-dataclass contradiction model.

    Accepts a ``Contradiction`` (from ``contradiction.models``) and dispatches
    to the named strategy implementation from ``contradiction.strategies``.

    Parameters
    ----------
    default_strategy:
        Name of the default strategy to use when ``resolve()`` is called
        without an explicit ``strategy`` argument.  Must be one of:
        ``'keep_newer'``, ``'keep_older'``, ``'keep_both_with_context'``,
        ``'flag_for_review'``, ``'merge'``.

    Example
    -------
    >>> resolver = SimpleContradictionResolver()
    >>> resolution = resolver.resolve(contradiction, strategy="keep_newer")
    """

    def __init__(self, default_strategy: str = "keep_newer") -> None:
        if default_strategy not in _VALID_STRATEGY_NAMES:
            raise ValueError(
                f"Unknown strategy {default_strategy!r}. "
                f"Valid options: {sorted(_VALID_STRATEGY_NAMES)!r}"
            )
        self._default_strategy = default_strategy

    @property
    def default_strategy(self) -> str:
        return self._default_strategy

    def resolve(
        self,
        contradiction: Contradiction,
        strategy: Optional[str] = None,
    ) -> Resolution:
        """Resolve a contradiction using the specified strategy.

        Parameters
        ----------
        contradiction:
            The detected contradiction to resolve.
        strategy:
            Strategy name override.  Falls back to ``default_strategy`` when
            ``None``.  Must be one of ``'keep_newer'``, ``'keep_older'``,
            ``'keep_both_with_context'``, ``'flag_for_review'``, ``'merge'``.

        Returns
        -------
        Resolution
            The resolution outcome.

        Raises
        ------
        ValueError
            If the strategy name is not recognised.
        """
        active_strategy = strategy if strategy is not None else self._default_strategy

        registry = _get_strategy_registry()
        strategy_obj = registry.get(active_strategy)

        if strategy_obj is None:
            raise ValueError(
                f"Unknown strategy {active_strategy!r}. "
                f"Valid options: {sorted(_VALID_STRATEGY_NAMES)!r}"
            )

        # All strategy objects expose .apply(contradiction) -> Resolution
        return strategy_obj.apply(contradiction)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = [
    "ContradictionResolver",
    "ResolutionResult",
    "ResolutionStrategy",
    "SimpleContradictionResolver",
]
