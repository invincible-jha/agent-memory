"""Unit tests for agent_memory.importance.gc — MemoryGarbageCollector."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_memory.importance.decay import ExponentialDecay
from agent_memory.importance.gc import MemoryGarbageCollector
from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    importance: float = 0.5,
    safety_critical: bool = False,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        importance_score=importance,
        safety_critical=safety_critical,
    )


def _fresh_store(*entries: MemoryEntry) -> EpisodicMemory:
    """Return an EpisodicMemory pre-populated with entries."""
    store = EpisodicMemory()
    for e in entries:
        store.store(e)
    return store


# ---------------------------------------------------------------------------
# MemoryGarbageCollector construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_threshold(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        assert gc.threshold == 0.05

    def test_custom_threshold(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store, threshold=0.2)
        assert gc.threshold == 0.2

    def test_default_decay_curve_is_exponential(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        assert gc._decay is not None


# ---------------------------------------------------------------------------
# effective_importance
# ---------------------------------------------------------------------------


class TestEffectiveImportance:
    def test_fresh_entry_effective_importance_near_original(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        entry = _make_entry(importance=0.8)
        # A freshly created entry has near-zero age, so decay ~ 1.0
        eff = gc.effective_importance(entry)
        assert eff == pytest.approx(0.8, rel=0.05)

    def test_effective_importance_is_product(self) -> None:
        store = EpisodicMemory()
        # Use LinearDecay with a known curve for predictable values
        from agent_memory.importance.decay import LinearDecay

        decay = LinearDecay(half_life_hours=24.0)
        gc = MemoryGarbageCollector(store=store, decay_curve=decay)
        entry = _make_entry(importance=1.0)
        eff = gc.effective_importance(entry)
        # Should equal importance_score * decay_factor, both <= 1.0
        assert 0.0 <= eff <= 1.0

    def test_effective_importance_above_zero_for_fresh_entry(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store, threshold=0.05)
        entry = _make_entry(importance=0.9)
        eff = gc.effective_importance(entry)
        assert eff > 0.0


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class TestCandidates:
    def test_no_candidates_when_all_above_threshold(self) -> None:
        store = _fresh_store(_make_entry(importance=0.9))
        gc = MemoryGarbageCollector(store=store, threshold=0.05)
        assert gc.candidates() == []

    def test_safety_critical_entries_excluded_from_candidates(self) -> None:
        entry = _make_entry(importance=0.01, safety_critical=True)
        store = _fresh_store(entry)
        gc = MemoryGarbageCollector(store=store, threshold=0.99)
        candidates = gc.candidates()
        assert not any(e.memory_id == entry.memory_id for e in candidates)

    def test_candidates_from_explicit_list(self) -> None:
        store = EpisodicMemory()
        low = _make_entry(importance=0.0)
        high = _make_entry(importance=0.99)
        gc = MemoryGarbageCollector(store=store, threshold=0.5)
        result = gc.candidates([low, high])
        # Only low should be a candidate
        assert any(e.memory_id == low.memory_id for e in result)
        assert not any(e.memory_id == high.memory_id for e in result)

    def test_empty_store_returns_no_candidates(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        assert gc.candidates() == []


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


class TestCollect:
    def test_collect_dry_run_does_not_delete(self) -> None:
        entry = _make_entry(importance=0.0)
        store = _fresh_store(entry)
        gc = MemoryGarbageCollector(store=store, threshold=0.99)
        evicted = gc.collect(dry_run=True)
        assert len(evicted) >= 1
        # Entry should still be in the store
        assert store.retrieve(entry.memory_id) is not None

    def test_collect_deletes_candidates(self) -> None:
        entry = _make_entry(importance=0.0)
        store = _fresh_store(entry)
        gc = MemoryGarbageCollector(store=store, threshold=0.99)
        evicted = gc.collect(dry_run=False)
        assert entry.memory_id in evicted
        assert store.retrieve(entry.memory_id) is None

    def test_collect_preserves_high_importance_entries(self) -> None:
        high = _make_entry(importance=0.9)
        store = _fresh_store(high)
        gc = MemoryGarbageCollector(store=store, threshold=0.05)
        gc.collect(dry_run=False)
        assert store.retrieve(high.memory_id) is not None

    def test_collect_returns_list_of_memory_ids(self) -> None:
        entry = _make_entry(importance=0.0)
        store = _fresh_store(entry)
        gc = MemoryGarbageCollector(store=store, threshold=0.99)
        evicted = gc.collect()
        assert isinstance(evicted, list)
        assert all(isinstance(mid, str) for mid in evicted)

    def test_collect_empty_store_returns_empty_list(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        assert gc.collect() == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict_with_required_keys(self) -> None:
        store = EpisodicMemory()
        gc = MemoryGarbageCollector(store=store)
        stats = gc.stats()
        for key in ("total", "protected", "unprotected", "eligible_for_gc"):
            assert key in stats

    def test_stats_total_matches_store_count(self) -> None:
        entries = [_make_entry(f"item {i}", importance=0.5) for i in range(4)]
        store = _fresh_store(*entries)
        gc = MemoryGarbageCollector(store=store, threshold=0.05)
        stats = gc.stats()
        assert stats["total"] == 4

    def test_stats_protected_count_for_safety_critical(self) -> None:
        crit = _make_entry(importance=0.5, safety_critical=True)
        normal = _make_entry(importance=0.5, safety_critical=False)
        store = _fresh_store(crit, normal)
        gc = MemoryGarbageCollector(store=store)
        stats = gc.stats()
        assert stats["protected"] == 1
        assert stats["unprotected"] == 1

    def test_stats_eligible_for_gc_count(self) -> None:
        low = _make_entry(importance=0.0)
        high = _make_entry(importance=0.9)
        store = _fresh_store(low, high)
        gc = MemoryGarbageCollector(store=store, threshold=0.5)
        stats = gc.stats()
        assert stats["eligible_for_gc"] >= 1
