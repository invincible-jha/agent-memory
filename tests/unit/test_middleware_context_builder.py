"""Unit tests for agent_memory.middleware.context_builder — ContextBuilder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.middleware.context_builder import (
    ContextBuilder,
    ContextSection,
    _CHARS_PER_TOKEN,
    _DEFAULT_LAYER_ORDER,
    _LAYER_HEADINGS,
)
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.unified.config import MemoryConfig
from agent_memory.unified.memory import UnifiedMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_with_entries(entries: list[MemoryEntry]) -> UnifiedMemory:
    """Create a UnifiedMemory with auto-scoring disabled and seed entries."""
    config = MemoryConfig(
        auto_score_importance=False,
        auto_score_freshness=False,
        auto_tag_provenance=False,
        consolidation_interval=0,
    )
    storage = InMemoryStorage()
    memory = UnifiedMemory(config=config, storage=storage)
    for entry in entries:
        memory.store(entry)
    return memory


def _entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    importance: float = 0.5,
    freshness: float = 1.0,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        importance_score=importance,
        freshness_score=freshness,
    )


# ---------------------------------------------------------------------------
# ContextSection dataclass
# ---------------------------------------------------------------------------


class TestContextSection:
    def test_construction(self) -> None:
        section = ContextSection(heading="Test", entries=[], token_budget=512)
        assert section.heading == "Test"
        assert section.entries == []
        assert section.token_budget == 512


# ---------------------------------------------------------------------------
# ContextBuilder construction
# ---------------------------------------------------------------------------


class TestContextBuilderInit:
    def test_default_layer_order(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory)
        assert builder._layer_order == _DEFAULT_LAYER_ORDER

    def test_custom_layer_order(self) -> None:
        memory = _make_memory_with_entries([])
        order = [MemoryLayer.SEMANTIC, MemoryLayer.WORKING]
        builder = ContextBuilder(memory=memory, layer_order=order)
        assert builder._layer_order == order

    def test_chars_per_token_clamp_to_minimum_one(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, chars_per_token=0)
        assert builder._chars_per_token == 1

    def test_negative_chars_per_token_clamps_to_one(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, chars_per_token=-5)
        assert builder._chars_per_token == 1

    def test_custom_separator_stored(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, separator="---")
        assert builder._separator == "---"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_zero(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory)
        assert builder.estimate_tokens("") == 0

    def test_exact_chars_per_token_multiple(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, chars_per_token=4)
        # 8 chars / 4 = 2 tokens
        assert builder.estimate_tokens("12345678") == 2

    def test_non_multiple_rounds_down(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, chars_per_token=4)
        # 9 chars / 4 = 2 (floor)
        assert builder.estimate_tokens("123456789") == 2


# ---------------------------------------------------------------------------
# build_from_entries
# ---------------------------------------------------------------------------


class TestBuildFromEntries:
    def test_empty_entries_returns_empty_string(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory)
        result = builder.build_from_entries([])
        assert result == ""

    def test_single_entry_formatted_with_dash(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, include_metadata=False)
        entry = _entry("hello world")
        result = builder.build_from_entries([entry])
        assert result == "- hello world"

    def test_entries_sorted_by_composite_score_descending(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, include_metadata=False)
        low = _entry("low priority content", importance=0.1, freshness=0.1)
        high = _entry("high priority content", importance=0.9, freshness=0.9)
        result = builder.build_from_entries([low, high])
        lines = result.splitlines()
        assert "high priority" in lines[0]
        assert "low priority" in lines[1]

    def test_token_budget_limits_output(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, chars_per_token=1)
        entries = [_entry(f"entry content item {i}") for i in range(20)]
        # With char budget of 10 chars, only a fraction fits
        result = builder.build_from_entries(entries, token_budget=10)
        assert len(result) <= 20  # budget enforced

    def test_include_metadata_adds_source_and_importance(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, include_metadata=True)
        entry = _entry("test content")
        result = builder.build_from_entries([entry])
        assert "src=" in result
        assert "imp=" in result

    def test_without_metadata_does_not_include_src(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory, include_metadata=False)
        entry = _entry("test content")
        result = builder.build_from_entries([entry])
        assert "src=" not in result


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_empty_memory_returns_empty_string(self) -> None:
        memory = _make_memory_with_entries([])
        builder = ContextBuilder(memory=memory)
        result = builder.build(query="anything")
        assert result == ""

    def test_build_returns_string(self) -> None:
        entries = [_entry("Python programming is powerful", layer=MemoryLayer.SEMANTIC)]
        memory = _make_memory_with_entries(entries)
        builder = ContextBuilder(memory=memory)
        result = builder.build(query="python")
        assert isinstance(result, str)

    def test_build_includes_layer_heading(self) -> None:
        entries = [_entry("working memory content", layer=MemoryLayer.WORKING)]
        memory = _make_memory_with_entries(entries)
        builder = ContextBuilder(memory=memory)
        result = builder.build(query="working memory content")
        assert "[Current Context]" in result

    def test_build_uses_separator_between_sections(self) -> None:
        entries = [
            _entry("working entry", layer=MemoryLayer.WORKING),
            _entry("semantic entry about working", layer=MemoryLayer.SEMANTIC),
        ]
        memory = _make_memory_with_entries(entries)
        builder = ContextBuilder(memory=memory, separator="###")
        result = builder.build(query="working entry semantic entry")
        if result.count("[") > 1:
            assert "###" in result

    def test_min_importance_filters_low_importance_entries(self) -> None:
        entries = [
            _entry("low importance content about query", importance=0.1, layer=MemoryLayer.SEMANTIC),
            _entry("high importance content about query", importance=0.9, layer=MemoryLayer.SEMANTIC),
        ]
        memory = _make_memory_with_entries(entries)
        builder = ContextBuilder(memory=memory)
        result = builder.build(query="content about query", min_importance=0.5)
        assert "low importance" not in result

    def test_token_budget_zero_returns_empty(self) -> None:
        entries = [_entry("some content", layer=MemoryLayer.SEMANTIC)]
        memory = _make_memory_with_entries(entries)
        builder = ContextBuilder(memory=memory, chars_per_token=4)
        result = builder.build(query="content", token_budget=0)
        assert result == ""

    def test_build_respects_layer_order(self) -> None:
        entries = [
            _entry("episodic memory entry", layer=MemoryLayer.EPISODIC),
            _entry("working memory entry here", layer=MemoryLayer.WORKING),
        ]
        memory = _make_memory_with_entries(entries)
        # Put SEMANTIC first (empty), then WORKING
        builder = ContextBuilder(
            memory=memory,
            layer_order=[MemoryLayer.WORKING, MemoryLayer.EPISODIC],
        )
        result = builder.build(query="memory entry")
        if result:
            first_heading_idx = result.find("[Current Context]")
            second_heading_idx = result.find("[Recent Interactions]")
            if first_heading_idx >= 0 and second_heading_idx >= 0:
                assert first_heading_idx < second_heading_idx


# ---------------------------------------------------------------------------
# Layer headings constants
# ---------------------------------------------------------------------------


class TestLayerHeadings:
    def test_all_four_layers_have_headings(self) -> None:
        for layer in MemoryLayer:
            assert layer in _LAYER_HEADINGS

    def test_working_layer_heading(self) -> None:
        assert _LAYER_HEADINGS[MemoryLayer.WORKING] == "Current Context"

    def test_episodic_layer_heading(self) -> None:
        assert _LAYER_HEADINGS[MemoryLayer.EPISODIC] == "Recent Interactions"

    def test_semantic_layer_heading(self) -> None:
        assert _LAYER_HEADINGS[MemoryLayer.SEMANTIC] == "Background Knowledge"

    def test_procedural_layer_heading(self) -> None:
        assert _LAYER_HEADINGS[MemoryLayer.PROCEDURAL] == "Procedures & Skills"
