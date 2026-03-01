"""Tests for async storage backends in agent-memory.

Coverage:
- AsyncInMemoryStorage: save/load/delete/clear/count/search/list_keys (8 tests)
- AsyncSQLiteStorage: same operations (8 tests, skipped if aiosqlite absent)
- Concurrent operations via asyncio.gather: 3 concurrent saves, reads (4 tests)
- Backward compatibility: sync API still works unchanged (3 tests)
Total: 23+ tests
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Optional

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.storage.async_memory_store import AsyncInMemoryStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.WORKING,
    importance: float = 0.5,
    memory_id: Optional[str] = None,
) -> MemoryEntry:
    """Build a minimal MemoryEntry for testing."""
    kwargs: dict[str, object] = {
        "content": content,
        "layer": layer,
        "importance_score": importance,
    }
    if memory_id is not None:
        kwargs["memory_id"] = memory_id
    return MemoryEntry(**kwargs)  # type: ignore[arg-type]


_aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None


# ---------------------------------------------------------------------------
# AsyncInMemoryStorage — core CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_inmemory_save_and_load() -> None:
    """save stores an entry; load retrieves it by memory_id."""
    backend = AsyncInMemoryStorage()
    entry = _make_entry("hello world", memory_id="e1")
    await backend.save(entry)
    loaded = await backend.load("e1")
    assert loaded is not None
    assert loaded.content == "hello world"


@pytest.mark.asyncio
async def test_async_inmemory_load_missing_returns_none() -> None:
    """load returns None when the key is not present."""
    backend = AsyncInMemoryStorage()
    result = await backend.load("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_async_inmemory_delete_existing_returns_true() -> None:
    """delete returns True and removes the entry."""
    backend = AsyncInMemoryStorage()
    entry = _make_entry("to delete", memory_id="del1")
    await backend.save(entry)
    deleted = await backend.delete("del1")
    assert deleted is True
    assert await backend.load("del1") is None


@pytest.mark.asyncio
async def test_async_inmemory_delete_missing_returns_false() -> None:
    """delete returns False when the key does not exist."""
    backend = AsyncInMemoryStorage()
    result = await backend.delete("ghost")
    assert result is False


@pytest.mark.asyncio
async def test_async_inmemory_clear_all() -> None:
    """clear() with no layer removes all entries and returns count."""
    backend = AsyncInMemoryStorage()
    for i in range(5):
        await backend.save(_make_entry(f"entry {i}", memory_id=f"id{i}"))
    removed = await backend.clear()
    assert removed == 5
    assert await backend.count() == 0


@pytest.mark.asyncio
async def test_async_inmemory_clear_by_layer() -> None:
    """clear(layer=...) only removes entries in that layer."""
    backend = AsyncInMemoryStorage()
    await backend.save(_make_entry("working", MemoryLayer.WORKING, memory_id="w1"))
    await backend.save(_make_entry("episodic", MemoryLayer.EPISODIC, memory_id="ep1"))
    removed = await backend.clear(layer=MemoryLayer.WORKING)
    assert removed == 1
    assert await backend.count() == 1
    assert await backend.load("ep1") is not None


@pytest.mark.asyncio
async def test_async_inmemory_count() -> None:
    """count() returns the total number of stored entries."""
    backend = AsyncInMemoryStorage()
    assert await backend.count() == 0
    await backend.save(_make_entry("a", memory_id="a"))
    await backend.save(_make_entry("b", memory_id="b"))
    assert await backend.count() == 2


@pytest.mark.asyncio
async def test_async_inmemory_search() -> None:
    """search returns entries whose content matches all query tokens."""
    backend = AsyncInMemoryStorage()
    await backend.save(_make_entry("the quick brown fox", memory_id="fox"))
    await backend.save(_make_entry("a lazy dog", memory_id="dog"))
    results = await backend.search("quick fox")
    assert len(results) == 1
    assert results[0].memory_id == "fox"


@pytest.mark.asyncio
async def test_async_inmemory_search_with_layer_filter() -> None:
    """search with layer filter excludes entries from other layers."""
    backend = AsyncInMemoryStorage()
    await backend.save(
        _make_entry("alpha beta", MemoryLayer.WORKING, memory_id="w")
    )
    await backend.save(
        _make_entry("alpha beta", MemoryLayer.EPISODIC, memory_id="e")
    )
    results = await backend.search("alpha", layer=MemoryLayer.EPISODIC)
    assert all(r.layer == MemoryLayer.EPISODIC for r in results)


@pytest.mark.asyncio
async def test_async_inmemory_list_keys() -> None:
    """list_keys returns all stored memory_ids."""
    backend = AsyncInMemoryStorage()
    ids = {"x1", "x2", "x3"}
    for i_id in ids:
        await backend.save(_make_entry("data", memory_id=i_id))
    keys = await backend.list_keys()
    assert set(keys) == ids


@pytest.mark.asyncio
async def test_async_inmemory_list_keys_with_layer_filter() -> None:
    """list_keys(layer=...) only returns keys for that layer."""
    backend = AsyncInMemoryStorage()
    await backend.save(_make_entry("w", MemoryLayer.WORKING, memory_id="wk"))
    await backend.save(_make_entry("s", MemoryLayer.SEMANTIC, memory_id="sm"))
    keys = await backend.list_keys(layer=MemoryLayer.WORKING)
    assert keys == ["wk"]


@pytest.mark.asyncio
async def test_async_inmemory_load_all() -> None:
    """load_all returns all entries, optionally filtered by layer."""
    backend = AsyncInMemoryStorage()
    await backend.save(_make_entry("one", MemoryLayer.WORKING, memory_id="o1"))
    await backend.save(_make_entry("two", MemoryLayer.WORKING, memory_id="o2"))
    await backend.save(_make_entry("three", MemoryLayer.EPISODIC, memory_id="o3"))
    all_entries = await backend.load_all()
    assert len(all_entries) == 3
    working_entries = await backend.load_all(layer=MemoryLayer.WORKING)
    assert len(working_entries) == 2


# ---------------------------------------------------------------------------
# AsyncSQLiteStorage — same operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_save_and_load() -> None:
    """AsyncSQLiteStorage: save stores and load retrieves by memory_id."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    entry = _make_entry("sqlite test", memory_id="s1")
    await backend.save(entry)
    loaded = await backend.load("s1")
    assert loaded is not None
    assert loaded.content == "sqlite test"


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_load_missing_returns_none() -> None:
    """AsyncSQLiteStorage: load returns None for a missing key."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    result = await backend.load("absent")
    assert result is None


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_delete() -> None:
    """AsyncSQLiteStorage: delete returns True for existing, False for missing."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    entry = _make_entry("will be deleted", memory_id="del_s")
    await backend.save(entry)
    assert await backend.delete("del_s") is True
    assert await backend.delete("del_s") is False


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_clear() -> None:
    """AsyncSQLiteStorage: clear() removes all entries."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    for i in range(3):
        await backend.save(_make_entry(f"entry {i}", memory_id=f"si{i}"))
    removed = await backend.clear()
    assert removed == 3
    assert await backend.count() == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_count() -> None:
    """AsyncSQLiteStorage: count returns correct number."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    assert await backend.count() == 0
    await backend.save(_make_entry("alpha", memory_id="sq_a"))
    await backend.save(_make_entry("beta", memory_id="sq_b"))
    assert await backend.count() == 2


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_search() -> None:
    """AsyncSQLiteStorage: search returns matching entries."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    await backend.save(_make_entry("database indexing", memory_id="db"))
    await backend.save(_make_entry("network latency", memory_id="net"))
    results = await backend.search("database")
    assert len(results) >= 1
    assert any(r.memory_id == "db" for r in results)


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_list_keys() -> None:
    """AsyncSQLiteStorage: list_keys returns all stored ids."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    ids = {"sq1", "sq2", "sq3"}
    for i_id in ids:
        await backend.save(_make_entry("data", memory_id=i_id))
    keys = await backend.list_keys()
    assert set(keys) == ids


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_overwrite() -> None:
    """AsyncSQLiteStorage: saving with the same id overwrites the entry."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    await backend.save(_make_entry("original", memory_id="ow"))
    await backend.save(_make_entry("updated", memory_id="ow"))
    loaded = await backend.load("ow")
    assert loaded is not None
    assert loaded.content == "updated"


# ---------------------------------------------------------------------------
# Concurrent operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_saves_inmemory() -> None:
    """Three concurrent saves all land without data corruption."""
    backend = AsyncInMemoryStorage()
    entries = [_make_entry(f"concurrent {i}", memory_id=f"c{i}") for i in range(3)]
    await asyncio.gather(*[backend.save(e) for e in entries])
    assert await backend.count() == 3


@pytest.mark.asyncio
async def test_concurrent_reads_inmemory() -> None:
    """Three concurrent loads all return the correct entries."""
    backend = AsyncInMemoryStorage()
    for i in range(3):
        await backend.save(_make_entry(f"r{i}", memory_id=f"r{i}"))
    results = await asyncio.gather(
        backend.load("r0"), backend.load("r1"), backend.load("r2")
    )
    assert all(r is not None for r in results)
    assert {r.memory_id for r in results if r is not None} == {"r0", "r1", "r2"}


@pytest.mark.asyncio
async def test_concurrent_saves_and_reads_inmemory() -> None:
    """Mixed concurrent saves and reads do not raise exceptions."""
    backend = AsyncInMemoryStorage()
    await backend.save(_make_entry("pre-existing", memory_id="pre"))

    save_tasks = [
        backend.save(_make_entry(f"new {i}", memory_id=f"n{i}")) for i in range(5)
    ]
    read_tasks = [backend.load("pre") for _ in range(5)]

    await asyncio.gather(*save_tasks, *read_tasks)
    # 1 pre-existing + 5 new
    assert await backend.count() == 6


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_concurrent_saves_sqlite() -> None:
    """Three concurrent saves to AsyncSQLiteStorage all succeed."""
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    backend = AsyncSQLiteStorage(db_path=":memory:")
    entries = [_make_entry(f"sq concurrent {i}", memory_id=f"sq_c{i}") for i in range(3)]
    await asyncio.gather(*[backend.save(e) for e in entries])
    assert await backend.count() == 3


# ---------------------------------------------------------------------------
# Backward compatibility — sync API unchanged
# ---------------------------------------------------------------------------


def test_sync_inmemory_save_load_still_works() -> None:
    """The synchronous InMemoryStorage API is unaffected by async changes."""
    from agent_memory.storage.memory_store import InMemoryStorage

    backend = InMemoryStorage()
    entry = _make_entry("sync content", memory_id="sync1")
    backend.save(entry)
    loaded = backend.load("sync1")
    assert loaded is not None
    assert loaded.content == "sync content"


def test_sync_inmemory_delete_still_works() -> None:
    """InMemoryStorage.delete still returns correct bool."""
    from agent_memory.storage.memory_store import InMemoryStorage

    backend = InMemoryStorage()
    entry = _make_entry("to remove", memory_id="rm")
    backend.save(entry)
    assert backend.delete("rm") is True
    assert backend.delete("rm") is False


def test_sync_sqlite_still_works() -> None:
    """SQLiteStorage continues to function correctly alongside async variant."""
    from agent_memory.storage.sqlite_store import SQLiteStorage

    backend = SQLiteStorage(db_path=":memory:")
    entry = _make_entry("sync sqlite", memory_id="ss1")
    backend.save(entry)
    loaded = backend.load("ss1")
    assert loaded is not None
    assert loaded.content == "sync sqlite"
    assert backend.delete("ss1") is True
