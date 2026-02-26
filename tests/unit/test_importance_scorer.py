"""Unit tests for agent_memory.importance.scorer — ImportanceScorer."""

from __future__ import annotations

import pytest

from agent_memory.importance.scorer import ImportanceScorer
from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.EPISODIC,
    source: MemorySource = MemorySource.AGENT_INFERENCE,
    safety_critical: bool = False,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=layer,
        source=source,
        safety_critical=safety_critical,
    )


# ---------------------------------------------------------------------------
# ImportanceScorer.score
# ---------------------------------------------------------------------------


class TestImportanceScorerScore:
    def test_safety_critical_entry_always_scores_one(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("some content", safety_critical=True)
        assert scorer.score(entry) == 1.0

    def test_score_returns_float_in_range(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("a normal memory entry")
        score = scorer.score(entry)
        assert 0.0 <= score <= 1.0

    def test_critical_keyword_raises_score(self) -> None:
        scorer = ImportanceScorer()
        # "critical" is in _CRITICAL_KEYWORDS
        entry_high = _make_entry("critical system failure detected")
        entry_low = _make_entry("the weather is nice today")
        assert scorer.score(entry_high) > scorer.score(entry_low)

    def test_high_keyword_raises_score_moderately(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("important deadline approaching")
        score = scorer.score(entry)
        assert score > 0.5

    def test_medium_keyword_gives_medium_score(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("should prefer the default approach here")
        score = scorer.score(entry)
        # Medium keyword + neutral source + episodic layer base
        assert 0.3 <= score <= 0.9

    def test_no_keyword_gives_neutral_score(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("the cat sat on the mat")
        score = scorer.score(entry)
        # Neutral content, no special keywords
        assert 0.0 < score < 1.0

    def test_tool_output_source_increases_score(self) -> None:
        scorer = ImportanceScorer()
        e_tool = _make_entry("some output", source=MemorySource.TOOL_OUTPUT)
        e_inference = _make_entry("some output", source=MemorySource.AGENT_INFERENCE)
        assert scorer.score(e_tool) > scorer.score(e_inference)

    def test_procedural_layer_has_highest_base(self) -> None:
        scorer = ImportanceScorer()
        e_proc = _make_entry("neutral content", layer=MemoryLayer.PROCEDURAL)
        e_epis = _make_entry("neutral content", layer=MemoryLayer.EPISODIC)
        assert scorer.score(e_proc) > scorer.score(e_epis)

    def test_score_clamped_to_one(self) -> None:
        scorer = ImportanceScorer()
        # Combine max keyword + best source + best layer
        entry = MemoryEntry(
            content="critical emergency stop",
            layer=MemoryLayer.PROCEDURAL,
            source=MemorySource.TOOL_OUTPUT,
        )
        score = scorer.score(entry)
        assert score <= 1.0

    def test_score_clamped_to_zero_at_minimum(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("x")
        score = scorer.score(entry)
        assert score >= 0.0

    def test_forbidden_keyword_triggers_critical_keyword_score(self) -> None:
        # "forbidden" is in _CRITICAL_KEYWORDS → keyword_score = 1.0
        # Final blended score = 1.0*0.5 + source*0.3 + layer*0.2
        # With AGENT_INFERENCE (0.5) and EPISODIC (0.5): 0.5+0.15+0.10 = 0.75
        scorer = ImportanceScorer()
        entry = _make_entry("forbidden operation detected")
        score = scorer.score(entry)
        assert score >= 0.7  # blended score with default source/layer

    def test_emergency_keyword_triggers_high_score(self) -> None:
        # "emergency" is in _CRITICAL_KEYWORDS → keyword_score = 1.0
        scorer = ImportanceScorer()
        entry = _make_entry("emergency evacuation required")
        score = scorer.score(entry)
        assert score >= 0.7


# ---------------------------------------------------------------------------
# ImportanceScorer.score_and_update
# ---------------------------------------------------------------------------


class TestScoreAndUpdate:
    def test_returns_new_entry_not_same_object(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("test content")
        updated = scorer.score_and_update(entry)
        assert updated is not entry

    def test_updated_entry_has_computed_importance_score(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("critical error occurred")
        updated = scorer.score_and_update(entry)
        expected = scorer.score(entry)
        assert updated.importance_score == pytest.approx(expected)

    def test_original_entry_unchanged(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("test content", source=MemorySource.AGENT_INFERENCE)
        original_score = entry.importance_score
        scorer.score_and_update(entry)
        assert entry.importance_score == original_score

    def test_safety_critical_updated_entry_scores_one(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("x", safety_critical=True)
        updated = scorer.score_and_update(entry)
        assert updated.importance_score == 1.0

    def test_content_preserved_after_update(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry("my important content")
        updated = scorer.score_and_update(entry)
        assert updated.content == "my important content"
