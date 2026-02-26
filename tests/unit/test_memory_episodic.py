"""Unit tests for EpisodicMemory — temporal events with range queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _entry_at(content: str, offset_hours: float = 0.0) -> MemoryEntry:
    created = datetime.now(timezone.utc) - timedelta(hours=offset_hours)
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.EPISODIC,
        created_at=created,
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestEpisodicMemoryCRUD:
    def test_store_and_retrieve(self) -> None:
        mem = EpisodicMemory()
        entry = _entry_at("user asked about pricing")
        mem.store(entry)
        result = mem.retrieve(entry.memory_id)
        assert result is not None
        assert result.content == "user asked about pricing"

    def test_retrieve_missing_id_returns_none(self) -> None:
        mem = EpisodicMemory()
        assert mem.retrieve("no-such-id") is None

    def test_store_replaces_existing_entry(self) -> None:
        mem = EpisodicMemory()
        entry = _entry_at("original")
        mem.store(entry)
        updated = entry.model_copy(update={"content": "updated"})
        mem.store(updated)
        result = mem.retrieve(entry.memory_id)
        assert result is not None
        assert result.content == "updated"

    def test_delete_existing_returns_true(self) -> None:
        mem = EpisodicMemory()
        entry = _entry_at("to delete")
        mem.store(entry)
        assert mem.delete(entry.memory_id) is True

    def test_delete_removes_entry(self) -> None:
        mem = EpisodicMemory()
        entry = _entry_at("to delete")
        mem.store(entry)
        mem.delete(entry.memory_id)
        assert mem.retrieve(entry.memory_id) is None

    def test_delete_missing_returns_false(self) -> None:
        mem = EpisodicMemory()
        assert mem.delete("ghost") is False

    def test_clear_all_returns_count(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("a"))
        mem.store(_entry_at("b"))
        count = mem.clear()
        assert count == 2

    def test_clear_all_empties_store(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("a"))
        mem.clear()
        assert mem.count() == 0

    def test_clear_by_layer_only_removes_matching(self) -> None:
        mem = EpisodicMemory()
        ep = _entry_at("episodic content")
        sem = MemoryEntry(content="semantic content", layer=MemoryLayer.SEMANTIC)
        mem.store(ep)
        mem.store(sem)
        mem.clear(layer=MemoryLayer.EPISODIC)
        assert mem.retrieve(ep.memory_id) is None
        assert mem.retrieve(sem.memory_id) is not None


# ---------------------------------------------------------------------------
# all() ordering and count()
# ---------------------------------------------------------------------------


class TestEpisodicMemoryOrdering:
    def test_all_sorted_by_created_at_ascending(self) -> None:
        mem = EpisodicMemory()
        oldest = _entry_at("old", offset_hours=3)
        middle = _entry_at("middle", offset_hours=2)
        newest = _entry_at("new", offset_hours=1)
        for e in (newest, oldest, middle):
            mem.store(e)
        entries = list(mem.all())
        assert entries[0].memory_id == oldest.memory_id
        assert entries[-1].memory_id == newest.memory_id

    def test_count_all(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("a"))
        mem.store(_entry_at("b"))
        assert mem.count() == 2

    def test_count_by_layer(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("ep"))
        mem.store(MemoryEntry(content="sem", layer=MemoryLayer.SEMANTIC))
        assert mem.count(layer=MemoryLayer.EPISODIC) == 1
        assert mem.count(layer=MemoryLayer.SEMANTIC) == 1


# ---------------------------------------------------------------------------
# query_range()
# ---------------------------------------------------------------------------


class TestEpisodicMemoryRangeQuery:
    def test_range_returns_entries_within_window(self) -> None:
        mem = EpisodicMemory()
        now = datetime.now(timezone.utc)
        recent = _entry_at("recent", offset_hours=1)
        old = _entry_at("old", offset_hours=48)
        mem.store(recent)
        mem.store(old)

        start = now - timedelta(hours=2)
        end = now
        results = mem.query_range(start=start, end=end)
        ids = {e.memory_id for e in results}
        assert recent.memory_id in ids
        assert old.memory_id not in ids

    def test_range_inclusive_boundaries(self) -> None:
        mem = EpisodicMemory()
        now = datetime.now(timezone.utc)
        exactly_at_start = MemoryEntry(
            content="boundary",
            layer=MemoryLayer.EPISODIC,
            created_at=now - timedelta(hours=1),
        )
        mem.store(exactly_at_start)
        results = mem.query_range(
            start=now - timedelta(hours=1),
            end=now,
        )
        assert any(e.memory_id == exactly_at_start.memory_id for e in results)

    def test_range_empty_when_no_entries_in_window(self) -> None:
        mem = EpisodicMemory()
        old = _entry_at("very old", offset_hours=100)
        mem.store(old)
        now = datetime.now(timezone.utc)
        results = mem.query_range(start=now - timedelta(hours=1), end=now)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# most_recent() and oldest()
# ---------------------------------------------------------------------------


class TestEpisodicMemoryHelpers:
    def test_most_recent_returns_n_newest_in_reverse_order(self) -> None:
        mem = EpisodicMemory()
        entries = [_entry_at(f"entry {i}", offset_hours=float(i)) for i in range(5)]
        for e in entries:
            mem.store(e)
        recent = mem.most_recent(n=2)
        assert len(recent) == 2
        # most_recent returns newest first
        assert recent[0].memory_id == entries[0].memory_id

    def test_oldest_returns_n_oldest(self) -> None:
        mem = EpisodicMemory()
        entries = [_entry_at(f"entry {i}", offset_hours=float(4 - i)) for i in range(5)]
        for e in entries:
            mem.store(e)
        oldest = mem.oldest(n=2)
        assert len(oldest) == 2

    def test_most_recent_with_fewer_entries_than_n(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("only one"))
        assert len(mem.most_recent(n=10)) == 1

    def test_oldest_with_fewer_entries_than_n(self) -> None:
        mem = EpisodicMemory()
        mem.store(_entry_at("only one"))
        assert len(mem.oldest(n=10)) == 1
