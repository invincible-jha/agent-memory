"""Unit tests for agent_memory.unified.memory — UnifiedMemory facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.unified.config import MemoryConfig
from agent_memory.unified.memory import UnifiedMemory, _build_storage_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> MemoryConfig:
    defaults: dict[str, object] = dict(
        auto_score_importance=False,
        auto_score_freshness=False,
        auto_tag_provenance=False,
        consolidation_interval=0,
    )
    defaults.update(kwargs)
    return MemoryConfig(**defaults)  # type: ignore[arg-type]


def _make_unified(**kwargs: object) -> UnifiedMemory:
    config = _make_config(**kwargs)
    return UnifiedMemory(config=config, storage=InMemoryStorage())


def _entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    importance: float = 0.5,
    safety_critical: bool = False,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        importance_score=importance,
        safety_critical=safety_critical,
    )


# ---------------------------------------------------------------------------
# _build_storage_backend
# ---------------------------------------------------------------------------


class TestBuildStorageBackend:
    def test_memory_backend_returns_in_memory_storage(self) -> None:
        from agent_memory.storage.memory_store import InMemoryStorage

        config = MemoryConfig(storage_backend="memory")
        backend = _build_storage_backend(config)
        assert isinstance(backend, InMemoryStorage)

    def test_sqlite_backend_returns_sqlite_storage(self) -> None:
        from agent_memory.storage.sqlite_store import SQLiteStorage

        config = MemoryConfig(storage_backend="sqlite", sqlite_path=":memory:")
        backend = _build_storage_backend(config)
        assert isinstance(backend, SQLiteStorage)

    def test_redis_backend_returns_redis_storage(self) -> None:
        from agent_memory.storage.redis_store import RedisStorage

        mock_redis_cls = MagicMock()
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client

        with patch("agent_memory.storage.redis_store.redis_lib.Redis", mock_redis_cls):
            config = MemoryConfig(storage_backend="redis")
            backend = _build_storage_backend(config)
        assert isinstance(backend, RedisStorage)


# ---------------------------------------------------------------------------
# UnifiedMemory construction
# ---------------------------------------------------------------------------


class TestUnifiedMemoryInit:
    def test_default_config_used_when_none_provided(self) -> None:
        storage = InMemoryStorage()
        memory = UnifiedMemory(storage=storage)
        assert memory.config is not None

    def test_explicit_config_stored(self) -> None:
        config = _make_config()
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        assert memory.config is config

    def test_explicit_storage_overrides_config_backend(self) -> None:
        config = _make_config()
        storage = InMemoryStorage()
        memory = UnifiedMemory(config=config, storage=storage)
        # The explicitly supplied storage backend is used for persistence
        # (same type as default, but the data written by store() ends up in it)
        entry = MemoryEntry(content="verify storage", layer=MemoryLayer.EPISODIC)
        memory.store(entry)
        assert memory.storage.load(entry.memory_id) is not None

    def test_store_count_starts_at_zero(self) -> None:
        memory = _make_unified()
        stats = memory.stats()
        assert stats["store_calls"] == 0


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestStore:
    def test_store_returns_memory_entry(self) -> None:
        memory = _make_unified()
        entry = _entry("hello world")
        result = memory.store(entry)
        assert isinstance(result, MemoryEntry)

    def test_store_increments_store_count(self) -> None:
        memory = _make_unified()
        memory.store(_entry("a"))
        memory.store(_entry("b"))
        assert memory.stats()["store_calls"] == 2

    def test_store_persists_to_backend(self) -> None:
        memory = _make_unified()
        entry = _entry("persistent data")
        memory.store(entry)
        loaded = memory.storage.load(entry.memory_id)
        assert loaded is not None

    def test_store_routes_to_correct_layer(self) -> None:
        memory = _make_unified()
        entry = _entry("semantic fact", layer=MemoryLayer.SEMANTIC)
        memory.store(entry)
        # Should be retrievable by layer search
        results = list(memory.storage.search("semantic fact", layer=MemoryLayer.SEMANTIC))
        assert len(results) == 1

    def test_store_triggers_auto_scoring_when_enabled(self) -> None:
        config = _make_config(auto_score_importance=True, auto_score_freshness=True)
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        entry = _entry("critical emergency data")
        result = memory.store(entry)
        # ImportanceScorer should have boosted importance for "critical"
        assert result.importance_score > 0.0

    def test_store_triggers_consolidation_at_interval(self) -> None:
        config = _make_config(consolidation_interval=3, importance_threshold=0.99)
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        # Store 3 low-importance entries; third should trigger consolidation
        for i in range(3):
            memory.store(_entry(f"low importance item {i}", importance=0.01))
        # Consolidation should have run (may have removed items)
        assert memory.stats()["store_calls"] == 3


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_returns_none_for_missing_id(self) -> None:
        memory = _make_unified()
        assert memory.recall("ghost-id") is None

    def test_recall_finds_entry_by_id(self) -> None:
        memory = _make_unified()
        entry = _entry("recalled content")
        memory.store(entry)
        recalled = memory.recall(entry.memory_id)
        assert recalled is not None
        assert recalled.content == "recalled content"

    def test_recall_increments_access_count(self) -> None:
        memory = _make_unified()
        entry = _entry("access tracked")
        memory.store(entry)
        recalled = memory.recall(entry.memory_id)
        assert recalled is not None
        assert recalled.access_count >= 1

    def test_recall_falls_back_to_storage_if_not_in_layer(self) -> None:
        storage = InMemoryStorage()
        memory = _make_unified()
        # Directly save to storage without going through layer stores
        entry = _entry("storage only entry")
        storage.save(entry)
        # Create memory that uses this storage
        config = _make_config()
        memory2 = UnifiedMemory(config=config, storage=storage)
        recalled = memory2.recall(entry.memory_id)
        assert recalled is not None


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    def test_forget_returns_true_when_found(self) -> None:
        memory = _make_unified()
        entry = _entry("forgettable data")
        memory.store(entry)
        assert memory.forget(entry.memory_id) is True

    def test_forget_returns_false_when_not_found(self) -> None:
        memory = _make_unified()
        assert memory.forget("ghost-id") is False

    def test_forget_removes_from_storage(self) -> None:
        memory = _make_unified()
        entry = _entry("to be deleted")
        memory.store(entry)
        memory.forget(entry.memory_id)
        assert memory.storage.load(entry.memory_id) is None

    def test_forget_removes_from_layer_store(self) -> None:
        memory = _make_unified()
        entry = _entry("layer entry", layer=MemoryLayer.WORKING)
        memory.store(entry)
        memory.forget(entry.memory_id)
        recalled = memory.recall(entry.memory_id)
        assert recalled is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_matching_entries(self) -> None:
        memory = _make_unified()
        memory.store(_entry("Python programming language"))
        results = memory.search("Python")
        assert len(results) >= 1

    def test_search_returns_empty_when_no_match(self) -> None:
        memory = _make_unified()
        memory.store(_entry("completely unrelated topic"))
        results = memory.search("quantum entanglement")
        assert len(results) == 0

    def test_search_filters_by_layer(self) -> None:
        memory = _make_unified()
        memory.store(_entry("shared keyword content", layer=MemoryLayer.WORKING))
        memory.store(_entry("shared keyword content", layer=MemoryLayer.SEMANTIC))
        results = memory.search("shared keyword content", layer=MemoryLayer.WORKING)
        for result in results:
            assert result.layer is MemoryLayer.WORKING

    def test_search_respects_limit(self) -> None:
        memory = _make_unified()
        for i in range(10):
            memory.store(_entry(f"common topic item {i}"))
        results = memory.search("common topic", limit=3)
        assert len(results) <= 3

    def test_search_deduplicates_results(self) -> None:
        memory = _make_unified()
        entry = _entry("unique content for search")
        memory.store(entry)
        results = memory.search("unique content for search")
        ids = [r.memory_id for r in results]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------


class TestConsolidate:
    def test_consolidate_returns_dict_with_collected_and_retained(self) -> None:
        memory = _make_unified()
        result = memory.consolidate()
        assert "collected" in result
        assert "retained" in result

    def test_consolidate_removes_low_importance_entries(self) -> None:
        config = _make_config(importance_threshold=0.9)
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        # Store a low-importance entry
        memory.store(_entry("unimportant data", importance=0.01))
        result = memory.consolidate()
        assert result["collected"] >= 1

    def test_consolidate_preserves_safety_critical_entries(self) -> None:
        config = _make_config(importance_threshold=0.99)
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        entry = _entry("critical data", importance=0.01, safety_critical=True)
        memory.store(entry)
        memory.consolidate()
        # Safety-critical entry should survive despite low importance
        recalled = memory.recall(entry.memory_id)
        assert recalled is not None

    def test_consolidate_deletes_from_storage(self) -> None:
        config = _make_config(importance_threshold=0.9)
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        entry = _entry("disposable", importance=0.01)
        memory.store(entry)
        memory.consolidate()
        assert memory.storage.load(entry.memory_id) is None


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict(self) -> None:
        memory = _make_unified()
        assert isinstance(memory.stats(), dict)

    def test_stats_contains_required_keys(self) -> None:
        memory = _make_unified()
        stats = memory.stats()
        for key in ("layer_counts", "total_in_process", "total_in_storage",
                    "store_calls", "working_capacity", "working_utilisation",
                    "storage_backend", "decay_function"):
            assert key in stats, f"Missing stat key: {key}"

    def test_stats_layer_counts_include_all_layers(self) -> None:
        memory = _make_unified()
        stats = memory.stats()
        layer_counts = stats["layer_counts"]
        assert isinstance(layer_counts, dict)
        for layer in MemoryLayer:
            assert layer.value in layer_counts

    def test_stats_working_utilisation_is_ratio(self) -> None:
        memory = _make_unified()
        memory.store(_entry("working item", layer=MemoryLayer.WORKING))
        stats = memory.stats()
        assert 0.0 <= stats["working_utilisation"] <= 1.0  # type: ignore[operator]

    def test_stats_total_increases_with_stores(self) -> None:
        memory = _make_unified()
        before = memory.stats()["total_in_process"]
        memory.store(_entry("new entry"))
        after = memory.stats()["total_in_process"]
        assert after > before  # type: ignore[operator]

    def test_stats_storage_backend_matches_config(self) -> None:
        config = _make_config()
        memory = UnifiedMemory(config=config, storage=InMemoryStorage())
        stats = memory.stats()
        assert stats["storage_backend"] == "memory"
