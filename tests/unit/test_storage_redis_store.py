"""Unit tests for agent_memory.storage.redis_store — RedisStorage (mocked)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.storage.redis_store import (
    RedisStorage,
    _deserialise,
    _entry_key,
    _layer_set_key,
    _serialise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test content",
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    source: MemorySource = MemorySource.USER_INPUT,
    safety_critical: bool = False,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        source=source,
        safety_critical=safety_critical,
        metadata=metadata or {},
    )


def _make_mock_client() -> MagicMock:
    """Return a MagicMock that behaves like a redis.Redis client."""
    client = MagicMock()
    pipeline_mock = MagicMock()
    client.pipeline.return_value = pipeline_mock
    pipeline_mock.__enter__ = MagicMock(return_value=pipeline_mock)
    pipeline_mock.__exit__ = MagicMock(return_value=False)
    return client


def _build_storage(client: MagicMock, ttl: int = 0) -> RedisStorage:
    return RedisStorage(client=client, ttl_seconds=ttl)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_entry_key_format(self) -> None:
        key = _entry_key("abc-123")
        assert key == "agent_memory:entry:abc-123"

    def test_layer_set_key_working(self) -> None:
        key = _layer_set_key(MemoryLayer.WORKING)
        assert key == "agent_memory:layer:working"

    def test_layer_set_key_episodic(self) -> None:
        key = _layer_set_key(MemoryLayer.EPISODIC)
        assert key == "agent_memory:layer:episodic"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_serialise_returns_flat_string_dict(self) -> None:
        entry = _make_entry()
        data = _serialise(entry)
        assert isinstance(data, dict)
        assert all(isinstance(v, str) for v in data.values())

    def test_serialise_contains_required_fields(self) -> None:
        entry = _make_entry("hello")
        data = _serialise(entry)
        for field in ("memory_id", "content", "layer", "importance_score",
                      "freshness_score", "source", "created_at", "last_accessed",
                      "access_count", "safety_critical", "metadata"):
            assert field in data, f"Missing field: {field}"

    def test_serialise_safety_critical_true(self) -> None:
        entry = _make_entry(safety_critical=True)
        data = _serialise(entry)
        assert data["safety_critical"] == "1"

    def test_serialise_safety_critical_false(self) -> None:
        entry = _make_entry(safety_critical=False)
        data = _serialise(entry)
        assert data["safety_critical"] == "0"

    def test_deserialise_roundtrip(self) -> None:
        entry = _make_entry("roundtrip test", metadata={"key": "value"})
        data = _serialise(entry)
        restored = _deserialise(data)
        assert restored.memory_id == entry.memory_id
        assert restored.content == entry.content
        assert restored.layer == entry.layer
        assert restored.source == entry.source
        assert restored.safety_critical == entry.safety_critical
        assert restored.metadata == {"key": "value"}

    def test_deserialise_attaches_utc_to_naive_datetime(self) -> None:
        entry = _make_entry()
        data = _serialise(entry)
        # Strip tzinfo from the datetime string to simulate naive stored value
        data["created_at"] = "2024-01-01T00:00:00"
        restored = _deserialise(data)
        assert restored.created_at.tzinfo is not None

    def test_deserialise_handles_missing_optional_fields(self) -> None:
        entry = _make_entry()
        data = _serialise(entry)
        # Remove optional fields with defaults
        data.pop("importance_score", None)
        data.pop("freshness_score", None)
        restored = _deserialise(data)
        assert restored.importance_score == 0.5
        assert restored.freshness_score == 1.0


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_calls_pipeline_hset(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client)
        entry = _make_entry("some content")
        storage.save(entry)
        pipeline = client.pipeline.return_value
        pipeline.hset.assert_called_once()

    def test_save_adds_to_all_keys_set(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client)
        entry = _make_entry()
        storage.save(entry)
        pipeline = client.pipeline.return_value
        pipeline.sadd.assert_any_call("agent_memory:all_keys", entry.memory_id)

    def test_save_adds_to_layer_set(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client)
        entry = _make_entry(layer=MemoryLayer.SEMANTIC)
        storage.save(entry)
        pipeline = client.pipeline.return_value
        pipeline.sadd.assert_any_call("agent_memory:layer:semantic", entry.memory_id)

    def test_save_with_ttl_calls_expire(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client, ttl=3600)
        entry = _make_entry()
        storage.save(entry)
        pipeline = client.pipeline.return_value
        pipeline.expire.assert_called_once()

    def test_save_without_ttl_does_not_call_expire(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client, ttl=0)
        entry = _make_entry()
        storage.save(entry)
        pipeline = client.pipeline.return_value
        pipeline.expire.assert_not_called()

    def test_save_calls_pipeline_execute(self) -> None:
        client = _make_mock_client()
        storage = _build_storage(client)
        storage.save(_make_entry())
        pipeline = client.pipeline.return_value
        pipeline.execute.assert_called_once()


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_returns_none_when_not_found(self) -> None:
        client = _make_mock_client()
        client.hgetall.return_value = {}
        storage = _build_storage(client)
        result = storage.load("ghost-id")
        assert result is None

    def test_load_returns_entry_when_found(self) -> None:
        client = _make_mock_client()
        entry = _make_entry("loaded content")
        client.hgetall.return_value = _serialise(entry)
        storage = _build_storage(client)
        result = storage.load(entry.memory_id)
        assert result is not None
        assert result.content == "loaded content"

    def test_load_calls_hgetall_with_correct_key(self) -> None:
        client = _make_mock_client()
        client.hgetall.return_value = {}
        storage = _build_storage(client)
        storage.load("test-id")
        client.hgetall.assert_called_once_with("agent_memory:entry:test-id")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_returns_false_when_not_found(self) -> None:
        client = _make_mock_client()
        client.hgetall.return_value = {}
        storage = _build_storage(client)
        assert storage.delete("ghost-id") is False

    def test_delete_returns_true_when_found(self) -> None:
        client = _make_mock_client()
        entry = _make_entry()
        client.hgetall.return_value = _serialise(entry)
        storage = _build_storage(client)
        assert storage.delete(entry.memory_id) is True

    def test_delete_calls_pipeline_delete_and_srem(self) -> None:
        client = _make_mock_client()
        entry = _make_entry(layer=MemoryLayer.WORKING)
        client.hgetall.return_value = _serialise(entry)
        storage = _build_storage(client)
        storage.delete(entry.memory_id)
        pipeline = client.pipeline.return_value
        pipeline.delete.assert_called_once_with(_entry_key(entry.memory_id))
        pipeline.srem.assert_any_call("agent_memory:all_keys", entry.memory_id)

    def test_delete_handles_invalid_layer_value_gracefully(self) -> None:
        client = _make_mock_client()
        entry = _make_entry()
        bad_data = _serialise(entry)
        bad_data["layer"] = "totally_invalid_layer"
        client.hgetall.return_value = bad_data
        storage = _build_storage(client)
        # Should not raise; bad layer just means layer set srem is skipped
        result = storage.delete(entry.memory_id)
        assert result is True


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_storage_with_entries(
        self, entries: list[MemoryEntry]
    ) -> tuple[RedisStorage, MagicMock]:
        client = _make_mock_client()
        # smembers returns the set of memory ids
        client.smembers.return_value = {e.memory_id for e in entries}
        # hgetall returns serialised data per key
        client.hgetall.side_effect = lambda key: next(
            (_serialise(e) for e in entries if _entry_key(e.memory_id) == key),
            {},
        )
        storage = _build_storage(client)
        return storage, client

    def test_search_finds_matching_content(self) -> None:
        entries = [
            _make_entry("python programming guide"),
            _make_entry("unrelated topic here"),
        ]
        storage, _ = self._make_storage_with_entries(entries)
        results = list(storage.search("python"))
        assert len(results) == 1
        assert results[0].content == "python programming guide"

    def test_search_returns_empty_when_no_match(self) -> None:
        entries = [_make_entry("absolutely nothing relevant")]
        storage, _ = self._make_storage_with_entries(entries)
        results = list(storage.search("quantum"))
        assert results == []

    def test_search_filters_by_layer(self) -> None:
        entries = [
            _make_entry("common term", layer=MemoryLayer.WORKING),
            _make_entry("common term", layer=MemoryLayer.SEMANTIC),
        ]
        client = _make_mock_client()
        client.smembers.return_value = {entries[0].memory_id}
        client.hgetall.side_effect = lambda key: next(
            (_serialise(e) for e in entries if _entry_key(e.memory_id) == key), {}
        )
        storage = _build_storage(client)
        results = list(storage.search("common", layer=MemoryLayer.WORKING))
        assert len(results) == 1

    def test_search_respects_limit(self) -> None:
        entries = [_make_entry(f"common entry {i}") for i in range(10)]
        storage, _ = self._make_storage_with_entries(entries)
        results = list(storage.search("common", limit=3))
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_list_keys_all(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {"id-1", "id-2", "id-3"}
        storage = _build_storage(client)
        keys = storage.list_keys()
        assert set(keys) == {"id-1", "id-2", "id-3"}

    def test_list_keys_by_layer(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {"id-a"}
        storage = _build_storage(client)
        keys = storage.list_keys(layer=MemoryLayer.SEMANTIC)
        client.smembers.assert_called_once_with("agent_memory:layer:semantic")
        assert keys == ["id-a"]

    def test_list_keys_sorted_and_limited(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {f"id-{i}" for i in range(10)}
        storage = _build_storage(client)
        keys = storage.list_keys(limit=3)
        assert len(keys) == 3


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all_returns_count_deleted(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {"id-1", "id-2"}
        storage = _build_storage(client)
        count = storage.clear()
        assert count == 2

    def test_clear_empty_returns_zero(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = set()
        storage = _build_storage(client)
        assert storage.clear() == 0

    def test_clear_all_deletes_global_set_key(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {"id-1"}
        storage = _build_storage(client)
        storage.clear()
        client.delete.assert_any_call("agent_memory:all_keys")

    def test_clear_by_layer_returns_count(self) -> None:
        client = _make_mock_client()
        client.smembers.return_value = {"id-x"}
        storage = _build_storage(client)
        count = storage.clear(layer=MemoryLayer.WORKING)
        assert count == 1


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_ping_returns_true_when_server_reachable(self) -> None:
        client = _make_mock_client()
        client.ping.return_value = True
        storage = _build_storage(client)
        assert storage.ping() is True

    def test_ping_returns_false_on_connection_error(self) -> None:
        import redis as redis_lib

        client = _make_mock_client()
        client.ping.side_effect = redis_lib.ConnectionError("refused")
        storage = _build_storage(client)
        assert storage.ping() is False
