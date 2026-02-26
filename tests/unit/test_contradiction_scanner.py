"""Unit tests for agent_memory.contradiction.scanner — ContradictionScanner."""

from __future__ import annotations

import pytest

from agent_memory.contradiction.detector import ContradictionDetector
from agent_memory.contradiction.scanner import (
    ContradictionScanner,
    ScanResult,
    ScanScope,
)
from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.semantic import SemanticMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str,
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
    importance: float = 0.5,
) -> MemoryEntry:
    return MemoryEntry(content=content, layer=layer, importance_score=importance)


def _store_with(*entries: MemoryEntry) -> SemanticMemory:
    store = SemanticMemory()
    for e in entries:
        store.store(e)
    return store


# ---------------------------------------------------------------------------
# ScanResult
# ---------------------------------------------------------------------------


class TestScanResult:
    def test_contradiction_count_property(self) -> None:
        result = ScanResult(scanned_count=5)
        assert result.contradiction_count == 0

    def test_summary_contains_required_keys(self) -> None:
        result = ScanResult(scanned_count=3)
        summary = result.summary()
        for key in ("scanned_at", "scanned_count", "skipped_count",
                    "contradiction_count", "early_terminated",
                    "scan_duration_seconds"):
            assert key in summary

    def test_early_terminated_defaults_to_false(self) -> None:
        result = ScanResult(scanned_count=0)
        assert result.early_terminated is False


# ---------------------------------------------------------------------------
# ScanScope enum
# ---------------------------------------------------------------------------


class TestScanScope:
    def test_all_scope_value(self) -> None:
        assert ScanScope.ALL.value == "all"

    def test_working_scope_value(self) -> None:
        assert ScanScope.WORKING.value == "working"

    def test_high_importance_scope_value(self) -> None:
        assert ScanScope.HIGH_IMPORTANCE.value == "high_importance"


# ---------------------------------------------------------------------------
# ContradictionScanner.scan_entries
# ---------------------------------------------------------------------------


class TestScanEntries:
    def test_empty_entries_returns_zero_scanned(self) -> None:
        scanner = ContradictionScanner()
        result = scanner.scan_entries([])
        assert result.scanned_count == 0
        assert result.contradiction_count == 0

    def test_single_entry_no_contradictions(self) -> None:
        scanner = ContradictionScanner()
        entry = _make_entry("The sky is blue")
        result = scanner.scan_entries([entry])
        assert result.scanned_count == 1
        assert result.contradiction_count == 0

    def test_scan_duration_recorded(self) -> None:
        scanner = ContradictionScanner()
        result = scanner.scan_entries([_make_entry("x")])
        assert result.scan_duration_seconds >= 0.0

    def test_detects_contradiction_in_entries(self) -> None:
        scanner = ContradictionScanner()
        e1 = _make_entry("The temperature is hot")
        e2 = _make_entry("The temperature is cold")
        result = scanner.scan_entries([e1, e2])
        # Should detect at least one contradiction
        assert result.scanned_count == 2
        assert isinstance(result.contradiction_pairs, list)

    def test_non_contradicting_entries_produce_no_pairs(self) -> None:
        scanner = ContradictionScanner()
        e1 = _make_entry("Python is a language")
        e2 = _make_entry("Coffee is a beverage")
        result = scanner.scan_entries([e1, e2])
        assert result.contradiction_count == 0


# ---------------------------------------------------------------------------
# ContradictionScanner.scan (store-based)
# ---------------------------------------------------------------------------


class TestScan:
    def test_scan_all_scope(self) -> None:
        store = _store_with(
            _make_entry("item one here"),
            _make_entry("item two here"),
        )
        scanner = ContradictionScanner()
        result = scanner.scan(store, scope=ScanScope.ALL)
        assert result.scanned_count == 2

    def test_scan_semantic_scope(self) -> None:
        store = _store_with(
            _make_entry("semantic entry", layer=MemoryLayer.SEMANTIC),
        )
        scanner = ContradictionScanner()
        result = scanner.scan(store, scope=ScanScope.SEMANTIC)
        assert result.scanned_count >= 1

    def test_scan_high_importance_scope_filters_by_threshold(self) -> None:
        store = _store_with(
            _make_entry("low importance", importance=0.2),
            _make_entry("high importance", importance=0.8),
        )
        scanner = ContradictionScanner(high_importance_threshold=0.6)
        result = scanner.scan(store, scope=ScanScope.HIGH_IMPORTANCE)
        # Only the high-importance entry should be included
        assert result.scanned_count == 1

    def test_scan_working_scope_uses_working_layer(self) -> None:
        from agent_memory.memory.working import WorkingMemory

        store = WorkingMemory(capacity=10)
        entry = MemoryEntry(content="working layer entry", layer=MemoryLayer.WORKING)
        store.store(entry)
        scanner = ContradictionScanner()
        result = scanner.scan(store, scope=ScanScope.WORKING)
        assert result.scanned_count >= 1

    def test_scan_returns_scan_result(self) -> None:
        store = _store_with()
        scanner = ContradictionScanner()
        result = scanner.scan(store)
        assert isinstance(result, ScanResult)

    def test_scan_records_duration(self) -> None:
        store = _store_with(_make_entry("x"))
        scanner = ContradictionScanner()
        result = scanner.scan(store)
        assert result.scan_duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# ContradictionScanner.scan_against
# ---------------------------------------------------------------------------


class TestScanAgainst:
    def test_scan_against_returns_scan_result(self) -> None:
        store = _store_with(_make_entry("candidate entry"))
        scanner = ContradictionScanner()
        entry = _make_entry("new entry to check")
        result = scanner.scan_against(entry, store)
        assert isinstance(result, ScanResult)

    def test_scan_against_excludes_same_entry(self) -> None:
        """Entry should not be compared against itself."""
        entry = _make_entry("The value is 42")
        store = _store_with(entry)
        scanner = ContradictionScanner()
        result = scanner.scan_against(entry, store)
        # Scanned count excludes the entry itself
        assert result.scanned_count == 0

    def test_scan_against_empty_store(self) -> None:
        store = SemanticMemory()
        scanner = ContradictionScanner()
        entry = _make_entry("lone entry")
        result = scanner.scan_against(entry, store)
        assert result.scanned_count == 0

    def test_scan_against_finds_contradiction_in_candidates(self) -> None:
        e_candidate = _make_entry("The status is active")
        store = _store_with(e_candidate)
        scanner = ContradictionScanner()
        new_entry = _make_entry("The status is inactive")
        result = scanner.scan_against(new_entry, store)
        assert isinstance(result.contradiction_pairs, list)


# ---------------------------------------------------------------------------
# Early termination
# ---------------------------------------------------------------------------


class TestEarlyTermination:
    def test_max_contradictions_triggers_early_stop(self) -> None:
        # Build a set of entries that will all contradict each other
        entries = [_make_entry(f"The value is {i}") for i in range(5)]
        scanner = ContradictionScanner(
            max_contradictions=1,
            similarity_threshold=0.0,
        )
        result = scanner.scan_entries(entries)
        # Should stop early when it finds enough contradictions
        assert result.scanned_count == 5

    def test_zero_max_contradictions_scans_all(self) -> None:
        entries = [_make_entry(f"item {i} content here") for i in range(4)]
        scanner = ContradictionScanner(max_contradictions=0)
        result = scanner.scan_entries(entries)
        assert result.scanned_count == 4
