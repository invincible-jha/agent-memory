"""Unit tests for agent_memory.memory.types — MemoryEntry, enums, computed fields."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from agent_memory.memory.types import (
    ImportanceLevel,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
)


# ---------------------------------------------------------------------------
# MemoryLayer enum
# ---------------------------------------------------------------------------


class TestMemoryLayer:
    def test_all_four_layers_exist(self) -> None:
        assert MemoryLayer.WORKING.value == "working"
        assert MemoryLayer.EPISODIC.value == "episodic"
        assert MemoryLayer.SEMANTIC.value == "semantic"
        assert MemoryLayer.PROCEDURAL.value == "procedural"

    def test_layer_from_string(self) -> None:
        assert MemoryLayer("working") is MemoryLayer.WORKING
        assert MemoryLayer("procedural") is MemoryLayer.PROCEDURAL

    def test_invalid_layer_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryLayer("unknown")


# ---------------------------------------------------------------------------
# MemorySource enum
# ---------------------------------------------------------------------------


class TestMemorySource:
    def test_all_sources_exist(self) -> None:
        expected = {"user_input", "tool_output", "agent_inference", "document", "external_api"}
        actual = {s.value for s in MemorySource}
        assert actual == expected

    def test_source_from_string(self) -> None:
        assert MemorySource("user_input") is MemorySource.USER_INPUT
        assert MemorySource("tool_output") is MemorySource.TOOL_OUTPUT


# ---------------------------------------------------------------------------
# ImportanceLevel enum and from_score
# ---------------------------------------------------------------------------


class TestImportanceLevel:
    def test_critical_threshold(self) -> None:
        assert ImportanceLevel.from_score(0.9) is ImportanceLevel.CRITICAL
        assert ImportanceLevel.from_score(1.0) is ImportanceLevel.CRITICAL

    def test_high_threshold(self) -> None:
        assert ImportanceLevel.from_score(0.7) is ImportanceLevel.HIGH
        assert ImportanceLevel.from_score(0.89) is ImportanceLevel.HIGH

    def test_medium_threshold(self) -> None:
        assert ImportanceLevel.from_score(0.5) is ImportanceLevel.MEDIUM
        assert ImportanceLevel.from_score(0.69) is ImportanceLevel.MEDIUM

    def test_low_threshold(self) -> None:
        assert ImportanceLevel.from_score(0.3) is ImportanceLevel.LOW
        assert ImportanceLevel.from_score(0.49) is ImportanceLevel.LOW

    def test_trivial_threshold(self) -> None:
        assert ImportanceLevel.from_score(0.0) is ImportanceLevel.TRIVIAL
        assert ImportanceLevel.from_score(0.29) is ImportanceLevel.TRIVIAL

    def test_boundary_exactly_09(self) -> None:
        assert ImportanceLevel.from_score(0.9) is ImportanceLevel.CRITICAL

    def test_all_level_values_exist(self) -> None:
        levels = {level.value for level in ImportanceLevel}
        assert "critical" in levels
        assert "trivial" in levels


# ---------------------------------------------------------------------------
# MemoryEntry creation and defaults
# ---------------------------------------------------------------------------


class TestMemoryEntryCreation:
    def test_minimal_creation_with_required_fields(self) -> None:
        entry = MemoryEntry(content="hello world", layer=MemoryLayer.WORKING)
        assert entry.content == "hello world"
        assert entry.layer is MemoryLayer.WORKING

    def test_auto_generated_memory_id_is_uuid(self) -> None:
        entry = MemoryEntry(content="test", layer=MemoryLayer.EPISODIC)
        assert len(entry.memory_id) == 36  # UUID4 length
        assert entry.memory_id.count("-") == 4

    def test_unique_ids_per_entry(self) -> None:
        entry_a = MemoryEntry(content="a", layer=MemoryLayer.WORKING)
        entry_b = MemoryEntry(content="b", layer=MemoryLayer.WORKING)
        assert entry_a.memory_id != entry_b.memory_id

    def test_default_importance_score(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.SEMANTIC)
        assert entry.importance_score == 0.5

    def test_default_freshness_score(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.SEMANTIC)
        assert entry.freshness_score == 1.0

    def test_default_source(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        assert entry.source is MemorySource.AGENT_INFERENCE

    def test_default_access_count(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        assert entry.access_count == 0

    def test_default_safety_critical_false(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        assert entry.safety_critical is False

    def test_default_metadata_empty_dict(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        assert entry.metadata == {}

    def test_created_at_is_utc_aware(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        assert entry.created_at.tzinfo is not None

    def test_custom_metadata(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.SEMANTIC,
            metadata={"key": "value", "entity": "Python"},
        )
        assert entry.metadata["entity"] == "Python"

    def test_importance_score_clamped_to_zero_lower_bound(self) -> None:
        with pytest.raises(Exception):
            MemoryEntry(content="x", layer=MemoryLayer.WORKING, importance_score=-0.1)

    def test_importance_score_clamped_to_one_upper_bound(self) -> None:
        with pytest.raises(Exception):
            MemoryEntry(content="x", layer=MemoryLayer.WORKING, importance_score=1.1)

    def test_safety_critical_flag(self) -> None:
        entry = MemoryEntry(content="critical info", layer=MemoryLayer.WORKING, safety_critical=True)
        assert entry.safety_critical is True


# ---------------------------------------------------------------------------
# MemoryEntry computed fields
# ---------------------------------------------------------------------------


class TestMemoryEntryComputedFields:
    def test_composite_score_is_product_of_importance_and_freshness(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.WORKING,
            importance_score=0.8,
            freshness_score=0.5,
        )
        assert entry.composite_score == pytest.approx(0.4, abs=1e-5)

    def test_composite_score_rounded_to_six_places(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.WORKING,
            importance_score=1 / 3,
            freshness_score=1 / 3,
        )
        # Just check it is a reasonable rounded float
        assert isinstance(entry.composite_score, float)

    def test_importance_level_computed_from_score(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.WORKING,
            importance_score=0.95,
        )
        assert entry.importance_level is ImportanceLevel.CRITICAL

    def test_importance_level_trivial(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.WORKING,
            importance_score=0.1,
        )
        assert entry.importance_level is ImportanceLevel.TRIVIAL

    def test_composite_score_zero_when_either_score_zero(self) -> None:
        entry = MemoryEntry(
            content="x",
            layer=MemoryLayer.WORKING,
            importance_score=0.0,
            freshness_score=1.0,
        )
        assert entry.composite_score == 0.0


# ---------------------------------------------------------------------------
# MemoryEntry.touch()
# ---------------------------------------------------------------------------


class TestMemoryEntryTouch:
    def test_touch_increments_access_count(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        touched = entry.touch()
        assert touched.access_count == 1

    def test_touch_is_idempotent_on_original(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        entry.touch()
        assert entry.access_count == 0  # original unchanged

    def test_touch_returns_new_entry(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        touched = entry.touch()
        assert touched is not entry

    def test_touch_updates_last_accessed(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        before = datetime.now(timezone.utc)
        time.sleep(0.01)
        touched = entry.touch()
        assert touched.last_accessed >= before

    def test_touch_preserves_content(self) -> None:
        entry = MemoryEntry(content="important content", layer=MemoryLayer.EPISODIC)
        touched = entry.touch()
        assert touched.content == "important content"

    def test_multiple_touches_accumulate_count(self) -> None:
        entry = MemoryEntry(content="x", layer=MemoryLayer.WORKING)
        after_one = entry.touch()
        after_two = after_one.touch()
        assert after_two.access_count == 2
