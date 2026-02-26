"""Unit tests for WorkingMemory — FIFO queue with configurable capacity."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.memory.working import WorkingMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(content: str, layer: MemoryLayer = MemoryLayer.WORKING) -> MemoryEntry:
    return MemoryEntry(content=content, layer=layer)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestWorkingMemoryInit:
    def test_default_capacity_is_64(self) -> None:
        mem = WorkingMemory()
        assert mem.capacity == 64

    def test_custom_capacity(self) -> None:
        mem = WorkingMemory(capacity=10)
        assert mem.capacity == 10

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(capacity=0)

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(capacity=-1)

    def test_starts_empty(self) -> None:
        mem = WorkingMemory(capacity=8)
        assert mem.count() == 0

    def test_is_not_full_when_empty(self) -> None:
        mem = WorkingMemory(capacity=8)
        assert mem.is_full is False


# ---------------------------------------------------------------------------
# store / retrieve
# ---------------------------------------------------------------------------


class TestWorkingMemoryStoreRetrieve:
    def test_store_and_retrieve_by_id(self) -> None:
        mem = WorkingMemory(capacity=8)
        entry = _make_entry("hello")
        mem.store(entry)
        result = mem.retrieve(entry.memory_id)
        assert result is not None
        assert result.content == "hello"

    def test_retrieve_unknown_id_returns_none(self) -> None:
        mem = WorkingMemory(capacity=8)
        assert mem.retrieve("nonexistent-id") is None

    def test_store_updates_existing_entry(self) -> None:
        mem = WorkingMemory(capacity=8)
        entry = _make_entry("original")
        mem.store(entry)
        updated = entry.model_copy(update={"importance_score": 0.9})
        mem.store(updated)
        result = mem.retrieve(entry.memory_id)
        assert result is not None
        assert result.importance_score == 0.9

    def test_update_does_not_change_count(self) -> None:
        mem = WorkingMemory(capacity=8)
        entry = _make_entry("original")
        mem.store(entry)
        updated = entry.model_copy(update={"importance_score": 0.9})
        mem.store(updated)
        assert mem.count() == 1


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------


class TestWorkingMemoryFIFO:
    def test_count_increases_until_capacity(self) -> None:
        mem = WorkingMemory(capacity=3)
        mem.store(_make_entry("a"))
        mem.store(_make_entry("b"))
        mem.store(_make_entry("c"))
        assert mem.count() == 3
        assert mem.is_full is True

    def test_overflow_evicts_oldest_entry(self) -> None:
        mem = WorkingMemory(capacity=2)
        oldest = _make_entry("oldest")
        mem.store(oldest)
        mem.store(_make_entry("middle"))
        # Adding a third evicts the oldest
        mem.store(_make_entry("newest"))
        assert mem.retrieve(oldest.memory_id) is None

    def test_overflow_keeps_count_at_capacity(self) -> None:
        mem = WorkingMemory(capacity=2)
        for i in range(5):
            mem.store(_make_entry(f"entry {i}"))
        assert mem.count() == 2

    def test_evicted_entry_removed_from_index(self) -> None:
        mem = WorkingMemory(capacity=1)
        first = _make_entry("first")
        second = _make_entry("second")
        mem.store(first)
        mem.store(second)
        assert mem.retrieve(first.memory_id) is None
        assert mem.retrieve(second.memory_id) is not None


# ---------------------------------------------------------------------------
# all() / count()
# ---------------------------------------------------------------------------


class TestWorkingMemoryAllCount:
    def test_all_returns_all_entries(self) -> None:
        mem = WorkingMemory(capacity=8)
        entries = [_make_entry(f"e{i}") for i in range(3)]
        for e in entries:
            mem.store(e)
        ids = {e.memory_id for e in mem.all()}
        expected = {e.memory_id for e in entries}
        assert ids == expected

    def test_all_filtered_by_layer(self) -> None:
        mem = WorkingMemory(capacity=16)
        working_entry = _make_entry("w", MemoryLayer.WORKING)
        episodic_entry = _make_entry("e", MemoryLayer.EPISODIC)
        mem.store(working_entry)
        mem.store(episodic_entry)
        working_results = mem.all(layer=MemoryLayer.WORKING)
        assert len(working_results) == 1
        assert working_results[0].memory_id == working_entry.memory_id

    def test_count_without_layer(self) -> None:
        mem = WorkingMemory(capacity=8)
        mem.store(_make_entry("a"))
        mem.store(_make_entry("b"))
        assert mem.count() == 2

    def test_count_with_layer_filter(self) -> None:
        mem = WorkingMemory(capacity=16)
        mem.store(_make_entry("a", MemoryLayer.WORKING))
        mem.store(_make_entry("b", MemoryLayer.EPISODIC))
        mem.store(_make_entry("c", MemoryLayer.WORKING))
        assert mem.count(layer=MemoryLayer.WORKING) == 2
        assert mem.count(layer=MemoryLayer.EPISODIC) == 1


# ---------------------------------------------------------------------------
# delete / clear
# ---------------------------------------------------------------------------


class TestWorkingMemoryDeleteClear:
    def test_delete_existing_entry_returns_true(self) -> None:
        mem = WorkingMemory(capacity=8)
        entry = _make_entry("to delete")
        mem.store(entry)
        assert mem.delete(entry.memory_id) is True

    def test_delete_removes_entry(self) -> None:
        mem = WorkingMemory(capacity=8)
        entry = _make_entry("to delete")
        mem.store(entry)
        mem.delete(entry.memory_id)
        assert mem.retrieve(entry.memory_id) is None

    def test_delete_nonexistent_returns_false(self) -> None:
        mem = WorkingMemory(capacity=8)
        assert mem.delete("ghost-id") is False

    def test_clear_all_returns_count(self) -> None:
        mem = WorkingMemory(capacity=8)
        mem.store(_make_entry("a"))
        mem.store(_make_entry("b"))
        count = mem.clear()
        assert count == 2

    def test_clear_all_empties_store(self) -> None:
        mem = WorkingMemory(capacity=8)
        mem.store(_make_entry("a"))
        mem.clear()
        assert mem.count() == 0

    def test_clear_by_layer_only_removes_matching(self) -> None:
        mem = WorkingMemory(capacity=16)
        mem.store(_make_entry("w", MemoryLayer.WORKING))
        ep = _make_entry("e", MemoryLayer.EPISODIC)
        mem.store(ep)
        mem.clear(layer=MemoryLayer.WORKING)
        assert mem.count() == 1
        assert mem.retrieve(ep.memory_id) is not None


# ---------------------------------------------------------------------------
# Peek helpers
# ---------------------------------------------------------------------------


class TestWorkingMemoryPeek:
    def test_peek_oldest_returns_first_added(self) -> None:
        mem = WorkingMemory(capacity=8)
        first = _make_entry("first")
        mem.store(first)
        mem.store(_make_entry("second"))
        assert mem.peek_oldest() is not None
        assert mem.peek_oldest().memory_id == first.memory_id  # type: ignore[union-attr]

    def test_peek_newest_returns_last_added(self) -> None:
        mem = WorkingMemory(capacity=8)
        mem.store(_make_entry("first"))
        last = _make_entry("last")
        mem.store(last)
        assert mem.peek_newest() is not None
        assert mem.peek_newest().memory_id == last.memory_id  # type: ignore[union-attr]

    def test_peek_empty_returns_none(self) -> None:
        mem = WorkingMemory(capacity=8)
        assert mem.peek_oldest() is None
        assert mem.peek_newest() is None


# ---------------------------------------------------------------------------
# search (inherited substring implementation)
# ---------------------------------------------------------------------------


class TestWorkingMemorySearch:
    def test_search_finds_matching_content(self) -> None:
        mem = WorkingMemory(capacity=16)
        mem.store(_make_entry("the quick brown fox"))
        mem.store(_make_entry("an unrelated entry"))
        results = mem.search("quick fox")
        assert len(results) == 1
        assert "quick" in results[0].content

    def test_search_returns_empty_when_no_match(self) -> None:
        mem = WorkingMemory(capacity=16)
        mem.store(_make_entry("hello world"))
        assert mem.search("python") == []

    def test_search_is_case_insensitive(self) -> None:
        mem = WorkingMemory(capacity=16)
        mem.store(_make_entry("Hello World"))
        assert len(mem.search("hello world")) == 1
