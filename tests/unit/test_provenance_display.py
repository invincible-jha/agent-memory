"""Unit tests for agent_memory.provenance.display — ProvenanceDisplay."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.provenance.display import ProvenanceDisplay
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


def _tagged_entry(source: MemorySource = MemorySource.AGENT_INFERENCE) -> MemoryEntry:
    """Return an entry that has been tagged with provenance metadata."""
    tracker = ProvenanceTracker()
    entry = _make_entry(source=source)
    return tracker.tag(entry)


# ---------------------------------------------------------------------------
# ProvenanceDisplay construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_tracker_created(self) -> None:
        display = ProvenanceDisplay()
        assert display._tracker is not None

    def test_custom_tracker_used(self) -> None:
        tracker = ProvenanceTracker()
        display = ProvenanceDisplay(tracker=tracker)
        assert display._tracker is tracker


# ---------------------------------------------------------------------------
# ProvenanceDisplay.summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_string(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.summary(entry)
        assert isinstance(result, str)

    def test_summary_contains_source(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry(source=MemorySource.TOOL_OUTPUT)
        result = display.summary(entry)
        assert "tool_output" in result

    def test_summary_contains_tier(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry(source=MemorySource.TOOL_OUTPUT)
        result = display.summary(entry)
        assert "VERIFIED_TOOL" in result

    def test_summary_contains_reliability(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.summary(entry)
        assert "reliability=" in result

    def test_summary_contains_tagged_at(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.summary(entry)
        assert "tagged_at=" in result

    def test_summary_includes_note_when_present(self) -> None:
        display = ProvenanceDisplay()
        tracker = ProvenanceTracker()
        entry = _make_entry()
        tagged = tracker.tag(entry, note="verified by admin")
        result = display.summary(tagged)
        assert "verified by admin" in result

    def test_summary_excludes_note_when_absent(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.summary(entry)
        assert "note=" not in result

    def test_summary_uses_pipe_separator(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.summary(entry)
        assert " | " in result

    def test_summary_untagged_entry_uses_source_fallback(self) -> None:
        display = ProvenanceDisplay()
        entry = _make_entry(source=MemorySource.USER_INPUT)
        # Should not raise; falls back to source-based computation
        result = display.summary(entry)
        assert "user_input" in result


# ---------------------------------------------------------------------------
# ProvenanceDisplay.as_dict
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_returns_dict(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert isinstance(result, dict)

    def test_contains_memory_id(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert result["memory_id"] == entry.memory_id

    def test_contains_source(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry(source=MemorySource.DOCUMENT)
        result = display.as_dict(entry)
        assert result["source"] == "document"

    def test_contains_tier(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert "tier" in result
        assert isinstance(result["tier"], str)

    def test_contains_reliability(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert "reliability" in result
        assert isinstance(result["reliability"], float)

    def test_contains_tagged_at(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert "tagged_at" in result

    def test_tagged_at_is_string(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert isinstance(result["tagged_at"], str)

    def test_note_empty_when_not_set(self) -> None:
        display = ProvenanceDisplay()
        entry = _tagged_entry()
        result = display.as_dict(entry)
        assert result["note"] == ""

    def test_note_populated_when_set(self) -> None:
        display = ProvenanceDisplay()
        tracker = ProvenanceTracker()
        entry = tracker.tag(_make_entry(), note="my note")
        result = display.as_dict(entry)
        assert result["note"] == "my note"


# ---------------------------------------------------------------------------
# ProvenanceDisplay.format_batch
# ---------------------------------------------------------------------------


class TestFormatBatch:
    def test_empty_list_returns_empty(self) -> None:
        display = ProvenanceDisplay()
        assert display.format_batch([]) == []

    def test_returns_list_of_dicts(self) -> None:
        display = ProvenanceDisplay()
        entries = [_tagged_entry() for _ in range(3)]
        results = display.format_batch(entries)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_each_dict_has_memory_id(self) -> None:
        display = ProvenanceDisplay()
        entries = [_tagged_entry(), _tagged_entry()]
        results = display.format_batch(entries)
        for r, e in zip(results, entries):
            assert r["memory_id"] == e.memory_id

    def test_batch_preserves_order(self) -> None:
        display = ProvenanceDisplay()
        sources = [MemorySource.TOOL_OUTPUT, MemorySource.USER_INPUT, MemorySource.DOCUMENT]
        entries = [_tagged_entry(source=s) for s in sources]
        results = display.format_batch(entries)
        for result, source in zip(results, sources):
            assert result["source"] == source.value
