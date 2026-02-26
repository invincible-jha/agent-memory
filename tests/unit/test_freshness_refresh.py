"""Unit tests for agent_memory.freshness.refresh — RefreshTrigger."""

from __future__ import annotations

import pytest

from agent_memory.freshness.refresh import RefreshTrigger
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(content: str = "test content") -> MemoryEntry:
    return MemoryEntry(content=content, layer=MemoryLayer.SEMANTIC)


# ---------------------------------------------------------------------------
# mark_for_refresh
# ---------------------------------------------------------------------------


class TestMarkForRefresh:
    def test_returns_new_entry(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        marked = trigger.mark_for_refresh(entry)
        assert marked is not entry

    def test_sets_needs_refresh_flag(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        marked = trigger.mark_for_refresh(entry)
        assert marked.metadata.get("needs_refresh") == "true"

    def test_sets_refresh_requested_at_timestamp(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        marked = trigger.mark_for_refresh(entry)
        assert "refresh_requested_at" in marked.metadata

    def test_original_entry_not_modified(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        trigger.mark_for_refresh(entry)
        assert "needs_refresh" not in entry.metadata

    def test_preserves_existing_metadata(self) -> None:
        trigger = RefreshTrigger()
        entry = MemoryEntry(
            content="test",
            layer=MemoryLayer.SEMANTIC,
            metadata={"key": "value"},
        )
        marked = trigger.mark_for_refresh(entry)
        assert marked.metadata.get("key") == "value"

    def test_content_preserved(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry("important content here")
        marked = trigger.mark_for_refresh(entry)
        assert marked.content == "important content here"


# ---------------------------------------------------------------------------
# clear_refresh_flag
# ---------------------------------------------------------------------------


class TestClearRefreshFlag:
    def test_returns_new_entry(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        cleared = trigger.clear_refresh_flag(entry)
        assert cleared is not entry

    def test_removes_needs_refresh_flag(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        cleared = trigger.clear_refresh_flag(entry)
        assert "needs_refresh" not in cleared.metadata

    def test_removes_refresh_requested_at(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        cleared = trigger.clear_refresh_flag(entry)
        assert "refresh_requested_at" not in cleared.metadata

    def test_sets_last_verified_to_now(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        cleared = trigger.clear_refresh_flag(entry)
        assert "last_verified" in cleared.metadata

    def test_resets_freshness_score_to_one(self) -> None:
        trigger = RefreshTrigger()
        entry = MemoryEntry(content="stale", layer=MemoryLayer.EPISODIC, freshness_score=0.1)
        cleared = trigger.clear_refresh_flag(entry)
        assert cleared.freshness_score == 1.0

    def test_clear_on_unmarked_entry_does_not_raise(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        # Should not raise even if not previously marked
        cleared = trigger.clear_refresh_flag(entry)
        assert "needs_refresh" not in cleared.metadata


# ---------------------------------------------------------------------------
# needs_refresh
# ---------------------------------------------------------------------------


class TestNeedsRefresh:
    def test_unmarked_entry_returns_false(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        assert trigger.needs_refresh(entry) is False

    def test_marked_entry_returns_true(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        assert trigger.needs_refresh(entry) is True

    def test_cleared_entry_returns_false(self) -> None:
        trigger = RefreshTrigger()
        entry = trigger.mark_for_refresh(_make_entry())
        cleared = trigger.clear_refresh_flag(entry)
        assert trigger.needs_refresh(cleared) is False

    def test_case_insensitive_flag_check(self) -> None:
        trigger = RefreshTrigger()
        entry = MemoryEntry(
            content="test",
            layer=MemoryLayer.SEMANTIC,
            metadata={"needs_refresh": "TRUE"},
        )
        assert trigger.needs_refresh(entry) is True

    def test_empty_metadata_returns_false(self) -> None:
        trigger = RefreshTrigger()
        entry = _make_entry()
        assert trigger.needs_refresh(entry) is False


# ---------------------------------------------------------------------------
# filter_pending
# ---------------------------------------------------------------------------


class TestFilterPending:
    def test_empty_list_returns_empty(self) -> None:
        trigger = RefreshTrigger()
        assert trigger.filter_pending([]) == []

    def test_returns_only_marked_entries(self) -> None:
        trigger = RefreshTrigger()
        e1 = trigger.mark_for_refresh(_make_entry("a"))
        e2 = _make_entry("b")
        e3 = trigger.mark_for_refresh(_make_entry("c"))
        pending = trigger.filter_pending([e1, e2, e3])
        assert len(pending) == 2
        contents = {e.content for e in pending}
        assert "a" in contents
        assert "c" in contents

    def test_none_marked_returns_empty(self) -> None:
        trigger = RefreshTrigger()
        entries = [_make_entry(f"entry {i}") for i in range(5)]
        assert trigger.filter_pending(entries) == []

    def test_all_marked_returns_all(self) -> None:
        trigger = RefreshTrigger()
        entries = [trigger.mark_for_refresh(_make_entry(f"e{i}")) for i in range(3)]
        pending = trigger.filter_pending(entries)
        assert len(pending) == 3
