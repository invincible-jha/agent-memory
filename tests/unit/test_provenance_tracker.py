"""Unit tests for agent_memory.provenance.tracker — ProvenanceTracker."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.provenance.tracker import ProvenanceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "test",
    source: MemorySource = MemorySource.AGENT_INFERENCE,
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.SEMANTIC,
        source=source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# ProvenanceTracker.tag
# ---------------------------------------------------------------------------


class TestTag:
    def test_tag_returns_new_entry(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert tagged is not entry

    def test_tag_adds_reliability_key(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert "provenance_reliability" in tagged.metadata

    def test_tag_adds_tier_key(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert "provenance_tier" in tagged.metadata

    def test_tag_adds_tagged_at_key(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert "provenance_tagged_at" in tagged.metadata

    def test_tag_does_not_add_note_when_empty(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert "provenance_note" not in tagged.metadata

    def test_tag_adds_note_when_provided(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry, note="external verification")
        assert tagged.metadata.get("provenance_note") == "external verification"

    def test_tag_preserves_existing_metadata(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(metadata={"custom": "data"})
        tagged = tracker.tag(entry)
        assert tagged.metadata.get("custom") == "data"

    def test_tag_original_entry_unchanged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tracker.tag(entry)
        assert "provenance_reliability" not in entry.metadata

    def test_tool_output_gets_high_reliability(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.TOOL_OUTPUT)
        tagged = tracker.tag(entry)
        rel = float(tagged.metadata["provenance_reliability"])
        assert rel >= 0.9

    def test_agent_inference_gets_lower_reliability(self) -> None:
        tracker = ProvenanceTracker()
        e_tool = _make_entry(source=MemorySource.TOOL_OUTPUT)
        e_agent = _make_entry(source=MemorySource.AGENT_INFERENCE)
        tagged_tool = tracker.tag(e_tool)
        tagged_agent = tracker.tag(e_agent)
        rel_tool = float(tagged_tool.metadata["provenance_reliability"])
        rel_agent = float(tagged_agent.metadata["provenance_reliability"])
        assert rel_tool > rel_agent

    def test_reliability_value_parseable_as_float(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        # Should not raise
        float(tagged.metadata["provenance_reliability"])


# ---------------------------------------------------------------------------
# ProvenanceTracker.get_reliability
# ---------------------------------------------------------------------------


class TestGetReliability:
    def test_returns_stored_reliability_when_tagged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        rel = tracker.get_reliability(tagged)
        assert 0.0 <= rel <= 1.0

    def test_computes_from_source_when_not_tagged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.TOOL_OUTPUT)
        rel = tracker.get_reliability(entry)
        assert rel >= 0.9

    def test_handles_invalid_stored_reliability_gracefully(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(metadata={"provenance_reliability": "not-a-float"})
        # Should fall back to source-based computation
        rel = tracker.get_reliability(entry)
        assert 0.0 <= rel <= 1.0


# ---------------------------------------------------------------------------
# ProvenanceTracker.get_tier
# ---------------------------------------------------------------------------


class TestGetTier:
    def test_returns_stored_tier_when_tagged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        tier = tracker.get_tier(tagged)
        assert isinstance(tier, str)
        assert len(tier) > 0

    def test_derives_tier_from_source_when_not_tagged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.TOOL_OUTPUT)
        tier = tracker.get_tier(entry)
        assert tier == "VERIFIED_TOOL"

    def test_tool_output_tier_is_verified_tool(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.TOOL_OUTPUT)
        tagged = tracker.tag(entry)
        assert tracker.get_tier(tagged) == "VERIFIED_TOOL"


# ---------------------------------------------------------------------------
# ProvenanceTracker.is_tagged
# ---------------------------------------------------------------------------


class TestIsTagged:
    def test_untagged_entry_returns_false(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        assert tracker.is_tagged(entry) is False

    def test_tagged_entry_returns_true(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry)
        assert tracker.is_tagged(tagged) is True

    def test_entry_with_manual_reliability_key_returns_true(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(metadata={"provenance_reliability": "0.5"})
        assert tracker.is_tagged(entry) is True


# ---------------------------------------------------------------------------
# ProvenanceTracker.override_reliability
# ---------------------------------------------------------------------------


class TestOverrideReliability:
    def test_returns_new_entry_with_different_source(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.AGENT_INFERENCE)
        overridden = tracker.override_reliability(entry, MemorySource.TOOL_OUTPUT)
        assert overridden.source is MemorySource.TOOL_OUTPUT

    def test_overridden_entry_is_tagged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.USER_INPUT)
        overridden = tracker.override_reliability(entry, MemorySource.DOCUMENT)
        assert tracker.is_tagged(overridden)

    def test_overridden_reliability_matches_new_source(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.AGENT_INFERENCE)
        overridden = tracker.override_reliability(entry, MemorySource.TOOL_OUTPUT)
        rel = tracker.get_reliability(overridden)
        assert rel >= 0.9  # TOOL_OUTPUT reliability

    def test_original_entry_unchanged(self) -> None:
        tracker = ProvenanceTracker()
        entry = _make_entry(source=MemorySource.AGENT_INFERENCE)
        tracker.override_reliability(entry, MemorySource.TOOL_OUTPUT)
        assert entry.source is MemorySource.AGENT_INFERENCE
