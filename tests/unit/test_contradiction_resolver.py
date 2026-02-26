"""Unit tests for agent_memory.contradiction.resolver — ContradictionResolver."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.contradiction.report import ContradictionPair
from agent_memory.contradiction.resolver import (
    ContradictionResolver,
    ResolutionResult,
    ResolutionStrategy,
)
from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test",
    source: MemorySource = MemorySource.AGENT_INFERENCE,
    importance: float = 0.5,
    freshness: float = 1.0,
    age_hours: float = 0.0,
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(hours=age_hours)
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.SEMANTIC,
        source=source,
        importance_score=importance,
        freshness_score=freshness,
        created_at=created_at,
    )


def _make_pair(entry_a: MemoryEntry, entry_b: MemoryEntry) -> ContradictionPair:
    return ContradictionPair(
        entry_a_id=entry_a.memory_id,
        entry_b_id=entry_b.memory_id,
        entry_a_content=entry_a.content,
        entry_b_content=entry_b.content,
        conflict_description="test conflict",
        similarity_score=0.5,
    )


# ---------------------------------------------------------------------------
# ResolutionStrategy enum
# ---------------------------------------------------------------------------


class TestResolutionStrategy:
    def test_strategy_values(self) -> None:
        assert ResolutionStrategy.RECENCY_WINS.value == "recency_wins"
        assert ResolutionStrategy.SOURCE_PRIORITY.value == "source_priority"
        assert ResolutionStrategy.CONFIDENCE_BASED.value == "confidence_based"
        assert ResolutionStrategy.MANUAL.value == "manual"


# ---------------------------------------------------------------------------
# ContradictionResolver construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_strategy_is_recency_wins(self) -> None:
        resolver = ContradictionResolver()
        assert resolver.default_strategy == ResolutionStrategy.RECENCY_WINS

    def test_custom_default_strategy(self) -> None:
        resolver = ContradictionResolver(
            default_strategy=ResolutionStrategy.SOURCE_PRIORITY
        )
        assert resolver.default_strategy == ResolutionStrategy.SOURCE_PRIORITY


# ---------------------------------------------------------------------------
# validate — mismatched entry IDs
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mismatched_ids_raises_value_error(self) -> None:
        resolver = ContradictionResolver()
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        e3 = _make_entry("c")  # unrelated entry
        pair = _make_pair(e1, e2)
        with pytest.raises(ValueError, match="do not match"):
            resolver.resolve(pair, e1, e3)


# ---------------------------------------------------------------------------
# RECENCY_WINS strategy
# ---------------------------------------------------------------------------


class TestRecencyWins:
    def test_newer_entry_a_wins(self) -> None:
        e_old = _make_entry("old", age_hours=10.0)
        e_new = _make_entry("new", age_hours=0.0)
        pair = _make_pair(e_new, e_old)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e_new, e_old, strategy=ResolutionStrategy.RECENCY_WINS
        )
        assert result.winner_id == e_new.memory_id
        assert result.loser_id == e_old.memory_id

    def test_newer_entry_b_wins(self) -> None:
        e_old = _make_entry("old", age_hours=10.0)
        e_new = _make_entry("new", age_hours=0.0)
        pair = _make_pair(e_old, e_new)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e_old, e_new, strategy=ResolutionStrategy.RECENCY_WINS
        )
        assert result.winner_id == e_new.memory_id

    def test_strategy_recorded_in_result(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b", age_hours=5.0)
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(pair, e1, e2, strategy=ResolutionStrategy.RECENCY_WINS)
        assert result.strategy_used == ResolutionStrategy.RECENCY_WINS

    def test_rationale_mentions_newer_entry(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b", age_hours=100.0)
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(pair, e1, e2, strategy=ResolutionStrategy.RECENCY_WINS)
        assert e1.memory_id in result.rationale or "newer" in result.rationale.lower()


# ---------------------------------------------------------------------------
# SOURCE_PRIORITY strategy
# ---------------------------------------------------------------------------


class TestSourcePriority:
    def test_tool_output_beats_agent_inference(self) -> None:
        e_tool = _make_entry("tool data", source=MemorySource.TOOL_OUTPUT)
        e_agent = _make_entry("inferred", source=MemorySource.AGENT_INFERENCE)
        pair = _make_pair(e_tool, e_agent)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e_tool, e_agent, strategy=ResolutionStrategy.SOURCE_PRIORITY
        )
        assert result.winner_id == e_tool.memory_id

    def test_document_beats_user_input(self) -> None:
        e_doc = _make_entry("doc", source=MemorySource.DOCUMENT)
        e_user = _make_entry("user", source=MemorySource.USER_INPUT)
        pair = _make_pair(e_doc, e_user)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e_doc, e_user, strategy=ResolutionStrategy.SOURCE_PRIORITY
        )
        assert result.winner_id == e_doc.memory_id

    def test_strategy_recorded(self) -> None:
        e1 = _make_entry("a", source=MemorySource.DOCUMENT)
        e2 = _make_entry("b", source=MemorySource.USER_INPUT)
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(pair, e1, e2, strategy=ResolutionStrategy.SOURCE_PRIORITY)
        assert result.strategy_used == ResolutionStrategy.SOURCE_PRIORITY


# ---------------------------------------------------------------------------
# CONFIDENCE_BASED strategy
# ---------------------------------------------------------------------------


class TestConfidenceBased:
    def test_higher_composite_score_wins(self) -> None:
        e_high = _make_entry("high", importance=0.9, freshness=0.9)
        e_low = _make_entry("low", importance=0.1, freshness=0.1)
        pair = _make_pair(e_high, e_low)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e_high, e_low, strategy=ResolutionStrategy.CONFIDENCE_BASED
        )
        assert result.winner_id == e_high.memory_id

    def test_equal_scores_entry_a_wins(self) -> None:
        e1 = _make_entry("a", importance=0.5, freshness=0.5)
        e2 = _make_entry("b", importance=0.5, freshness=0.5)
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e1, e2, strategy=ResolutionStrategy.CONFIDENCE_BASED
        )
        # When scores are equal, entry_a wins (>= comparison)
        assert result.winner_id == e1.memory_id

    def test_strategy_recorded(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(pair, e1, e2, strategy=ResolutionStrategy.CONFIDENCE_BASED)
        assert result.strategy_used == ResolutionStrategy.CONFIDENCE_BASED


# ---------------------------------------------------------------------------
# MANUAL strategy
# ---------------------------------------------------------------------------


class TestManual:
    def test_manual_winner_a_designated(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e1, e2,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id=e1.memory_id,
        )
        assert result.winner_id == e1.memory_id
        assert result.loser_id == e2.memory_id

    def test_manual_winner_b_designated(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e1, e2,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id=e2.memory_id,
        )
        assert result.winner_id == e2.memory_id

    def test_invalid_manual_winner_id_sets_requires_review(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e1, e2,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id="totally-invalid-id",
        )
        assert result.requires_manual_review is True
        assert result.winner_id == ""

    def test_none_manual_winner_requires_review(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, e1, e2,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id=None,
        )
        assert result.requires_manual_review is True


# ---------------------------------------------------------------------------
# Default strategy fallback
# ---------------------------------------------------------------------------


class TestDefaultStrategy:
    def test_default_strategy_used_when_no_override(self) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b", age_hours=10.0)
        pair = _make_pair(e1, e2)
        resolver = ContradictionResolver(
            default_strategy=ResolutionStrategy.RECENCY_WINS
        )
        result = resolver.resolve(pair, e1, e2)
        assert result.strategy_used == ResolutionStrategy.RECENCY_WINS


# ---------------------------------------------------------------------------
# resolve_batch
# ---------------------------------------------------------------------------


class TestResolveBatch:
    def test_batch_returns_one_result_per_pair(self) -> None:
        resolver = ContradictionResolver()
        pairs: list[tuple[ContradictionPair, MemoryEntry, MemoryEntry]] = []
        for _ in range(3):
            e1 = _make_entry("a")
            e2 = _make_entry("b")
            pairs.append((_make_pair(e1, e2), e1, e2))
        results = resolver.resolve_batch(pairs)
        assert len(results) == 3

    def test_batch_applies_strategy_to_all(self) -> None:
        resolver = ContradictionResolver()
        e1 = _make_entry("a", source=MemorySource.TOOL_OUTPUT)
        e2 = _make_entry("b", source=MemorySource.AGENT_INFERENCE)
        pairs = [(_make_pair(e1, e2), e1, e2)]
        results = resolver.resolve_batch(
            pairs, strategy=ResolutionStrategy.SOURCE_PRIORITY
        )
        assert results[0].strategy_used == ResolutionStrategy.SOURCE_PRIORITY

    def test_empty_batch_returns_empty_list(self) -> None:
        resolver = ContradictionResolver()
        assert resolver.resolve_batch([]) == []

    def test_batch_results_are_resolution_result_instances(self) -> None:
        resolver = ContradictionResolver()
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        pairs = [(_make_pair(e1, e2), e1, e2)]
        results = resolver.resolve_batch(pairs)
        assert all(isinstance(r, ResolutionResult) for r in results)
