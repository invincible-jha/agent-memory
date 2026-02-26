"""Unit tests for agent_memory.storage.memory_store — InMemoryStorage."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.storage.memory_store import InMemoryStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    source: MemorySource = MemorySource.USER_INPUT,
    importance: float = 0.5,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        source=source,
        importance_score=importance,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInMemoryStorageInit:
    def test_starts_empty(self) -> None:
        store = InMemoryStorage()
        assert len(store) == 0

    def test_repr_shows_count(self) -> None:
        store = InMemoryStorage()
        assert "InMemoryStorage" in repr(store)
        assert "count=0" in repr(store)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self) -> None:
        store = InMemoryStorage()
        entry = _make_entry("hello world")
        store.save(entry)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.memory_id == entry.memory_id
        assert loaded.content == "hello world"

    def test_load_missing_returns_none(self) -> None:
        store = InMemoryStorage()
        assert store.load("does-not-exist") is None

    def test_save_overwrites_existing_entry(self) -> None:
        store = InMemoryStorage()
        entry = _make_entry("original")
        store.save(entry)
        updated = entry.model_copy(update={"content": "updated"})
        store.save(updated)
        loaded = store.load(entry.memory_id)
        assert loaded is not None
        assert loaded.content == "updated"

    def test_save_multiple_entries(self) -> None:
        store = InMemoryStorage()
        entries = [_make_entry(f"entry {i}") for i in range(5)]
        for e in entries:
            store.save(e)
        assert len(store) == 5


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_returns_true(self) -> None:
        store = InMemoryStorage()
        entry = _make_entry()
        store.save(entry)
        assert store.delete(entry.memory_id) is True

    def test_delete_removes_entry(self) -> None:
        store = InMemoryStorage()
        entry = _make_entry()
        store.save(entry)
        store.delete(entry.memory_id)
        assert store.load(entry.memory_id) is None

    def test_delete_missing_returns_false(self) -> None:
        store = InMemoryStorage()
        assert store.delete("ghost-id") is False

    def test_delete_reduces_count(self) -> None:
        store = InMemoryStorage()
        e1 = _make_entry("a")
        e2 = _make_entry("b")
        store.save(e1)
        store.save(e2)
        store.delete(e1.memory_id)
        assert len(store) == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_matching_entry(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("the quick brown fox"))
        results = list(store.search("quick"))
        assert len(results) == 1

    def test_search_case_insensitive(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("Python is great"))
        results = list(store.search("python"))
        assert len(results) == 1

    def test_search_multi_token_all_must_match(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("apple banana cherry"))
        store.save(_make_entry("apple only here"))
        results = list(store.search("apple banana"))
        assert len(results) == 1
        assert results[0].content == "apple banana cherry"

    def test_search_no_match_returns_empty(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("totally unrelated"))
        results = list(store.search("quantum computing"))
        assert results == []

    def test_search_filters_by_layer(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("shared content", layer=MemoryLayer.WORKING))
        store.save(_make_entry("shared content", layer=MemoryLayer.SEMANTIC))
        results = list(store.search("shared", layer=MemoryLayer.WORKING))
        assert len(results) == 1
        assert results[0].layer is MemoryLayer.WORKING

    def test_search_respects_limit(self) -> None:
        store = InMemoryStorage()
        for i in range(10):
            store.save(_make_entry(f"common term entry {i}"))
        results = list(store.search("common", limit=3))
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_list_keys_returns_all_ids(self) -> None:
        store = InMemoryStorage()
        entries = [_make_entry(f"item {i}") for i in range(3)]
        for e in entries:
            store.save(e)
        keys = store.list_keys()
        assert set(keys) == {e.memory_id for e in entries}

    def test_list_keys_filtered_by_layer(self) -> None:
        store = InMemoryStorage()
        e_working = _make_entry("w", layer=MemoryLayer.WORKING)
        e_semantic = _make_entry("s", layer=MemoryLayer.SEMANTIC)
        store.save(e_working)
        store.save(e_semantic)
        keys = store.list_keys(layer=MemoryLayer.WORKING)
        assert keys == [e_working.memory_id]

    def test_list_keys_respects_limit(self) -> None:
        store = InMemoryStorage()
        for i in range(20):
            store.save(_make_entry(f"item {i}"))
        keys = store.list_keys(limit=5)
        assert len(keys) == 5


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all_removes_everything(self) -> None:
        store = InMemoryStorage()
        for i in range(5):
            store.save(_make_entry(f"item {i}"))
        removed = store.clear()
        assert removed == 5
        assert len(store) == 0

    def test_clear_by_layer_only_removes_that_layer(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("w", layer=MemoryLayer.WORKING))
        store.save(_make_entry("e", layer=MemoryLayer.EPISODIC))
        removed = store.clear(layer=MemoryLayer.WORKING)
        assert removed == 1
        assert len(store) == 1

    def test_clear_empty_store_returns_zero(self) -> None:
        store = InMemoryStorage()
        assert store.clear() == 0


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_all(self) -> None:
        store = InMemoryStorage()
        for i in range(4):
            store.save(_make_entry(f"item {i}"))
        assert store.count() == 4

    def test_count_by_layer(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("a", layer=MemoryLayer.WORKING))
        store.save(_make_entry("b", layer=MemoryLayer.WORKING))
        store.save(_make_entry("c", layer=MemoryLayer.SEMANTIC))
        assert store.count(layer=MemoryLayer.WORKING) == 2
        assert store.count(layer=MemoryLayer.SEMANTIC) == 1

    def test_count_empty_layer_returns_zero(self) -> None:
        store = InMemoryStorage()
        assert store.count(layer=MemoryLayer.PROCEDURAL) == 0


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


class TestLoadAll:
    def test_load_all_returns_all_entries(self) -> None:
        store = InMemoryStorage()
        entries = [_make_entry(f"item {i}") for i in range(3)]
        for e in entries:
            store.save(e)
        loaded = store.load_all()
        assert len(loaded) == 3

    def test_load_all_filtered_by_layer(self) -> None:
        store = InMemoryStorage()
        store.save(_make_entry("a", layer=MemoryLayer.WORKING))
        store.save(_make_entry("b", layer=MemoryLayer.EPISODIC))
        loaded = store.load_all(layer=MemoryLayer.WORKING)
        assert len(loaded) == 1
        assert loaded[0].layer is MemoryLayer.WORKING

    def test_load_all_respects_limit(self) -> None:
        store = InMemoryStorage()
        for i in range(10):
            store.save(_make_entry(f"item {i}"))
        loaded = store.load_all(limit=4)
        assert len(loaded) == 4


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


class TestContains:
    def test_contains_true_for_stored_entry(self) -> None:
        store = InMemoryStorage()
        entry = _make_entry()
        store.save(entry)
        assert entry.memory_id in store

    def test_contains_false_for_missing_entry(self) -> None:
        store = InMemoryStorage()
        assert "ghost-id" not in store
