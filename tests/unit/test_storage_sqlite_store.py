"""Unit tests for agent_memory.storage.sqlite_store — SQLiteStorage."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.storage.sqlite_store import (
    SQLiteStorage,
    _entry_to_row,
    _escape_fts_query,
    _parse_dt,
    _row_to_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        importance_score=importance,
        metadata=metadata or {},
    )


@pytest.fixture()
def store() -> Generator[SQLiteStorage, None, None]:
    """Provide an in-memory SQLiteStorage for each test."""
    with SQLiteStorage(db_path=":memory:") as s:
        yield s


# ---------------------------------------------------------------------------
# _parse_dt helper
# ---------------------------------------------------------------------------


class TestParseDt:
    def test_parses_iso_with_timezone(self) -> None:
        dt = _parse_dt("2024-01-15T10:30:00+00:00")
        assert dt.tzinfo is not None

    def test_naive_datetime_gets_utc(self) -> None:
        dt = _parse_dt("2024-01-15T10:30:00")
        assert dt.tzinfo is not None

    def test_returns_correct_time(self) -> None:
        from datetime import timezone
        dt = _parse_dt("2024-06-01T12:00:00")
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.hour == 12


# ---------------------------------------------------------------------------
# _escape_fts_query helper
# ---------------------------------------------------------------------------


class TestEscapeFtsQuery:
    def test_wraps_in_quotes(self) -> None:
        result = _escape_fts_query("hello world")
        assert result == '"hello world"'

    def test_escapes_internal_quotes(self) -> None:
        result = _escape_fts_query('say "hello"')
        assert '""' in result

    def test_empty_string(self) -> None:
        result = _escape_fts_query("")
        assert result == '""'


# ---------------------------------------------------------------------------
# _entry_to_row and _row_to_entry helpers
# ---------------------------------------------------------------------------


class TestRowConversion:
    def test_entry_to_row_returns_tuple_of_eight(self) -> None:
        entry = _make_entry()
        row = _entry_to_row(entry)
        assert isinstance(row, tuple)
        assert len(row) == 8

    def test_entry_to_row_memory_id_first(self) -> None:
        entry = _make_entry()
        row = _entry_to_row(entry)
        assert row[0] == entry.memory_id

    def test_entry_to_row_content_field(self) -> None:
        entry = _make_entry("hello world content")
        row = _entry_to_row(entry)
        assert row[3] == "hello world content"

    def test_entry_to_row_layer_as_string(self) -> None:
        entry = _make_entry(layer=MemoryLayer.SEMANTIC)
        row = _entry_to_row(entry)
        assert row[2] == "semantic"

    def test_entry_to_row_importance_float(self) -> None:
        entry = _make_entry(importance=0.75)
        row = _entry_to_row(entry)
        assert row[7] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# SQLiteStorage construction
# ---------------------------------------------------------------------------


class TestSQLiteStorageInit:
    def test_creates_in_memory_database(self) -> None:
        with SQLiteStorage(db_path=":memory:") as s:
            assert s.count() == 0

    def test_context_manager_closes_connection(self) -> None:
        with SQLiteStorage(db_path=":memory:") as s:
            pass
        # After __exit__, using the connection should fail
        with pytest.raises(Exception):
            s._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, store: SQLiteStorage) -> None:
        entry = _make_entry("stored content")
        store.save(entry)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.memory_id == entry.memory_id
        assert loaded.content == "stored content"

    def test_load_missing_returns_none(self, store: SQLiteStorage) -> None:
        assert store.load("ghost-id") is None

    def test_save_overwrites_existing(self, store: SQLiteStorage) -> None:
        entry = _make_entry("original")
        store.save(entry)
        updated = entry.model_copy(update={"importance_score": 0.9})
        store.save(updated)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.importance_score == pytest.approx(0.9)

    def test_load_preserves_metadata(self, store: SQLiteStorage) -> None:
        entry = _make_entry(metadata={"key": "value"})
        store.save(entry)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.metadata.get("key") == "value"

    def test_load_preserves_layer(self, store: SQLiteStorage) -> None:
        entry = _make_entry(layer=MemoryLayer.SEMANTIC)
        store.save(entry)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.layer is MemoryLayer.SEMANTIC

    def test_save_multiple_entries(self, store: SQLiteStorage) -> None:
        for i in range(5):
            store.save(_make_entry(f"entry {i}"))
        assert store.count() == 5


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_returns_true(self, store: SQLiteStorage) -> None:
        entry = _make_entry()
        store.save(entry)
        assert store.delete(entry.memory_id) is True

    def test_delete_removes_entry(self, store: SQLiteStorage) -> None:
        entry = _make_entry()
        store.save(entry)
        store.delete(entry.memory_id)
        assert store.load(entry.memory_id) is None

    def test_delete_missing_returns_false(self, store: SQLiteStorage) -> None:
        assert store.delete("ghost-id") is False

    def test_delete_reduces_count(self, store: SQLiteStorage) -> None:
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        store.save(e1)
        store.save(e2)
        store.delete(e1.memory_id)
        assert store.count() == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_matching_entry(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("python programming tutorial"))
        results = list(store.search("python"))
        assert len(results) >= 1

    def test_search_returns_empty_when_no_match(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("java programming tutorial"))
        results = list(store.search("quantum computing"))
        assert results == []

    def test_search_filters_by_layer(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("python tutorial", layer=MemoryLayer.SEMANTIC))
        store.save(_make_entry("python tutorial", layer=MemoryLayer.EPISODIC))
        results = list(store.search("python", layer=MemoryLayer.SEMANTIC))
        assert all(r.layer is MemoryLayer.SEMANTIC for r in results)

    def test_search_respects_limit(self, store: SQLiteStorage) -> None:
        for i in range(10):
            store.save(_make_entry(f"common term entry {i}"))
        results = list(store.search("common", limit=3))
        assert len(results) <= 3

    def test_search_returns_memory_entry_objects(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("test entry"))
        results = list(store.search("test"))
        assert all(isinstance(r, MemoryEntry) for r in results)


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_list_keys_all(self, store: SQLiteStorage) -> None:
        entries = [_make_entry(f"item {i}") for i in range(3)]
        for e in entries:
            store.save(e)
        keys = store.list_keys()
        expected_ids = {e.memory_id for e in entries}
        assert set(keys) == expected_ids

    def test_list_keys_by_layer(self, store: SQLiteStorage) -> None:
        e_sem = _make_entry("s", layer=MemoryLayer.SEMANTIC)
        e_epi = _make_entry("e", layer=MemoryLayer.EPISODIC)
        store.save(e_sem)
        store.save(e_epi)
        keys = store.list_keys(layer=MemoryLayer.SEMANTIC)
        assert keys == [e_sem.memory_id]

    def test_list_keys_respects_limit(self, store: SQLiteStorage) -> None:
        for i in range(10):
            store.save(_make_entry(f"item {i}"))
        keys = store.list_keys(limit=4)
        assert len(keys) == 4


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_all(self, store: SQLiteStorage) -> None:
        for i in range(4):
            store.save(_make_entry(f"item {i}"))
        assert store.count() == 4

    def test_count_by_layer(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("a", layer=MemoryLayer.WORKING))
        store.save(_make_entry("b", layer=MemoryLayer.WORKING))
        store.save(_make_entry("c", layer=MemoryLayer.SEMANTIC))
        assert store.count(layer=MemoryLayer.WORKING) == 2
        assert store.count(layer=MemoryLayer.SEMANTIC) == 1

    def test_count_empty_layer_returns_zero(self, store: SQLiteStorage) -> None:
        assert store.count(layer=MemoryLayer.PROCEDURAL) == 0

    def test_count_empty_store_returns_zero(self, store: SQLiteStorage) -> None:
        assert store.count() == 0


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all(self, store: SQLiteStorage) -> None:
        for i in range(5):
            store.save(_make_entry(f"item {i}"))
        removed = store.clear()
        assert removed == 5
        assert store.count() == 0

    def test_clear_by_layer(self, store: SQLiteStorage) -> None:
        store.save(_make_entry("a", layer=MemoryLayer.WORKING))
        store.save(_make_entry("b", layer=MemoryLayer.EPISODIC))
        removed = store.clear(layer=MemoryLayer.WORKING)
        assert removed == 1
        assert store.count() == 1

    def test_clear_empty_returns_zero(self, store: SQLiteStorage) -> None:
        assert store.clear() == 0
