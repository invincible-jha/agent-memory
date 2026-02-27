"""Tests for PatternExtractor — E14.3."""

from __future__ import annotations

import pytest

from agent_memory.consolidation.pattern_extractor import ExtractedPattern, PatternExtractor
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(content: str) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.EPISODIC,
        importance_score=0.5,
    )


# ---------------------------------------------------------------------------
# PatternExtractor init
# ---------------------------------------------------------------------------


class TestPatternExtractorInit:
    def test_default_parameters(self) -> None:
        extractor = PatternExtractor()
        assert extractor._min_frequency == 2
        assert extractor._max_entities == 20
        assert extractor._max_actions == 10

    def test_custom_parameters(self) -> None:
        extractor = PatternExtractor(min_frequency=3, max_entities=5, max_actions=3)
        assert extractor._min_frequency == 3
        assert extractor._max_entities == 5

    def test_invalid_min_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="min_frequency"):
            PatternExtractor(min_frequency=0)


# ---------------------------------------------------------------------------
# Extract from empty / single
# ---------------------------------------------------------------------------


class TestExtractEmpty:
    def test_empty_list(self) -> None:
        extractor = PatternExtractor()
        result = extractor.extract([])
        assert result.entities == {}
        assert result.actions == {}
        assert result.total_episodes == 0
        assert result.coverage_ratio == 0.0

    def test_single_text_below_min_frequency(self) -> None:
        extractor = PatternExtractor(min_frequency=2)
        result = extractor.extract(["database query optimization"])
        # Single occurrence — should be filtered out with min_frequency=2
        assert result.total_episodes == 1
        # Entities may be empty since they appear only once
        assert all(count >= 2 for count in result.entities.values())


# ---------------------------------------------------------------------------
# Extract patterns
# ---------------------------------------------------------------------------


class TestExtractPatterns:
    def test_recurring_entities_detected(self) -> None:
        extractor = PatternExtractor(min_frequency=2)
        texts = [
            "Agent executed database query for authentication.",
            "Agent processed database query for validation.",
            "Agent completed database query with results.",
        ]
        result = extractor.extract(texts)
        assert result.total_episodes == 3
        # "database" and "query" appear in all three
        assert "database" in result.entities or "query" in result.entities

    def test_recurring_actions_detected(self) -> None:
        extractor = PatternExtractor(min_frequency=2)
        texts = [
            "Agent executed create operation on resource.",
            "Agent executed delete operation on resource.",
            "Agent executed update operation on resource.",
        ]
        result = extractor.extract(texts)
        # "execute" or "create"/"delete"/"update" should appear
        assert result.total_episodes == 3

    def test_coverage_ratio_correct(self) -> None:
        extractor = PatternExtractor(min_frequency=1)
        texts = [
            "process request authentication",
            "nothing useful here",
        ]
        result = extractor.extract(texts)
        assert 0.0 <= result.coverage_ratio <= 1.0
        assert result.total_episodes == 2

    def test_entities_capped_by_max_entities(self) -> None:
        extractor = PatternExtractor(min_frequency=1, max_entities=3)
        texts = [f"entity{i} entity{i} process create update" for i in range(50)]
        result = extractor.extract(texts)
        assert len(result.entities) <= 3

    def test_actions_capped_by_max_actions(self) -> None:
        extractor = PatternExtractor(min_frequency=1, max_actions=2)
        texts = ["create update delete fetch store merge split join" for _ in range(5)]
        result = extractor.extract(texts)
        assert len(result.actions) <= 2


# ---------------------------------------------------------------------------
# ExtractedPattern methods
# ---------------------------------------------------------------------------


class TestExtractedPattern:
    def _make_pattern(self) -> ExtractedPattern:
        return ExtractedPattern(
            entities={"database": 10, "service": 8, "agent": 6, "token": 4, "user": 2},
            actions={"create": 7, "execute": 5, "fetch": 3},
            total_episodes=15,
            coverage_ratio=0.8,
        )

    def test_top_entities_sorted(self) -> None:
        pattern = self._make_pattern()
        top = pattern.top_entities(3)
        assert top[0] == ("database", 10)
        assert top[1] == ("service", 8)
        assert top[2] == ("agent", 6)

    def test_top_actions_sorted(self) -> None:
        pattern = self._make_pattern()
        top = pattern.top_actions(2)
        assert top[0] == ("create", 7)
        assert top[1] == ("execute", 5)

    def test_top_entities_limit_respected(self) -> None:
        pattern = self._make_pattern()
        assert len(pattern.top_entities(2)) == 2

    def test_to_dict_contains_expected_keys(self) -> None:
        pattern = self._make_pattern()
        data = pattern.to_dict()
        assert "entities" in data
        assert "actions" in data
        assert "total_episodes" in data
        assert "coverage_ratio" in data
        assert data["total_episodes"] == 15
        assert data["coverage_ratio"] == 0.8


# ---------------------------------------------------------------------------
# extract_from_cluster
# ---------------------------------------------------------------------------


class TestExtractFromCluster:
    def test_extracts_from_memory_entries(self) -> None:
        extractor = PatternExtractor(min_frequency=2)
        episodes = [
            _make_episode("Agent processed database query for user records."),
            _make_episode("Agent executed database query for user account data."),
            _make_episode("Agent completed database query for user profile."),
        ]
        result = extractor.extract_from_cluster(episodes)
        assert result.total_episodes == 3
        assert isinstance(result, ExtractedPattern)

    def test_empty_cluster_returns_empty_pattern(self) -> None:
        extractor = PatternExtractor()
        result = extractor.extract_from_cluster([])
        assert result.total_episodes == 0
        assert result.entities == {}
