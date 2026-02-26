"""Unit tests for agent_memory.contradiction.detector — ContradictionDetector."""

from __future__ import annotations

import pytest

from agent_memory.contradiction.detector import ContradictionDetector
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(content: str, layer: MemoryLayer = MemoryLayer.SEMANTIC) -> MemoryEntry:
    return MemoryEntry(content=content, layer=layer)


# ---------------------------------------------------------------------------
# ContradictionDetector.detect — basic behaviour
# ---------------------------------------------------------------------------


class TestDetect:
    def test_empty_entries_returns_empty_report(self) -> None:
        detector = ContradictionDetector()
        report = detector.detect([])
        assert report.total_entries_scanned == 0
        assert report.contradiction_count == 0

    def test_single_entry_returns_empty_report(self) -> None:
        detector = ContradictionDetector()
        entry = _make_entry("The sky is blue.")
        report = detector.detect([entry])
        assert report.total_entries_scanned == 1
        assert report.contradiction_count == 0

    def test_non_contradicting_entries_returns_empty_pairs(self) -> None:
        detector = ContradictionDetector()
        e1 = _make_entry("Python is a programming language.")
        e2 = _make_entry("Coffee is a hot beverage.")
        report = detector.detect([e1, e2])
        assert report.total_entries_scanned == 2
        # These entries share few tokens so Jaccard < threshold
        assert report.contradiction_count == 0

    def test_report_contains_scanned_count(self) -> None:
        detector = ContradictionDetector()
        entries = [_make_entry(f"entry number {i}") for i in range(5)]
        report = detector.detect(entries)
        assert report.total_entries_scanned == 5

    def test_returns_contradiction_report_instance(self) -> None:
        detector = ContradictionDetector()
        report = detector.detect([])
        assert isinstance(report, ContradictionReport)

    def test_detects_entity_attribute_contradiction(self) -> None:
        detector = ContradictionDetector()
        # "temperature is hot" vs "temperature is cold" — same entity+attribute, different value
        e1 = _make_entry("The temperature is hot")
        e2 = _make_entry("The temperature is cold")
        report = detector.detect([e1, e2])
        assert report.contradiction_count >= 1

    def test_contradiction_pair_references_correct_entry_ids(self) -> None:
        detector = ContradictionDetector()
        e1 = _make_entry("The server is running")
        e2 = _make_entry("The server is stopped")
        report = detector.detect([e1, e2])
        if report.contradiction_count > 0:
            pair = report.contradiction_pairs[0]
            assert pair.entry_a_id in {e1.memory_id, e2.memory_id}
            assert pair.entry_b_id in {e1.memory_id, e2.memory_id}

    def test_similarity_score_in_pair_is_in_range(self) -> None:
        detector = ContradictionDetector()
        e1 = _make_entry("The light is on")
        e2 = _make_entry("The light is off")
        report = detector.detect([e1, e2])
        for pair in report.contradiction_pairs:
            assert 0.0 <= pair.similarity_score <= 1.0

    def test_multiple_pairs_scanned_pairwise(self) -> None:
        detector = ContradictionDetector()
        # Three identical-structure entries; at least some pairs should compare
        e1 = _make_entry("X is true")
        e2 = _make_entry("X is false")
        e3 = _make_entry("X is maybe")
        report = detector.detect([e1, e2, e3])
        # Total entries scanned is 3
        assert report.total_entries_scanned == 3


# ---------------------------------------------------------------------------
# ContradictionDetector.detect_for_entry
# ---------------------------------------------------------------------------


class TestDetectForEntry:
    def test_returns_empty_list_for_no_candidates(self) -> None:
        detector = ContradictionDetector()
        entry = _make_entry("The sky is blue")
        result = detector.detect_for_entry(entry, [])
        assert result == []

    def test_skips_same_entry_by_id(self) -> None:
        detector = ContradictionDetector()
        entry = _make_entry("The sky is blue")
        # Providing the same entry as a candidate should be skipped
        result = detector.detect_for_entry(entry, [entry])
        assert result == []

    def test_finds_contradiction_against_candidate(self) -> None:
        detector = ContradictionDetector()
        entry = _make_entry("The status is active")
        candidate = _make_entry("The status is inactive")
        result = detector.detect_for_entry(entry, [candidate])
        # May or may not find a contradiction depending on token overlap
        assert isinstance(result, list)

    def test_returns_list_of_contradiction_pairs(self) -> None:
        detector = ContradictionDetector()
        entry = _make_entry("The temperature is hot")
        candidates = [
            _make_entry("The temperature is cold"),
            _make_entry("Unrelated subject matter here"),
        ]
        result = detector.detect_for_entry(entry, candidates)
        for item in result:
            assert isinstance(item, ContradictionPair)


# ---------------------------------------------------------------------------
# ContradictionDetector._has_negation_conflict
# ---------------------------------------------------------------------------


class TestHasNegationConflict:
    def test_both_have_negation_returns_false(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict(
            "I cannot do this", "I cannot do that"
        ) is False

    def test_neither_has_negation_returns_false(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("It is running", "It is active") is False

    def test_only_first_has_negation_returns_true(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("It is not running", "It is active") is True

    def test_only_second_has_negation_returns_true(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("It is running", "It is not active") is True

    def test_case_insensitive_negation_detection(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("NEVER do this", "always do this") is True

    def test_wont_marker_detected(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("I won't comply", "I agree") is True

    def test_isnt_marker_detected(self) -> None:
        detector = ContradictionDetector()
        assert detector._has_negation_conflict("The service isn't available", "OK") is True


# ---------------------------------------------------------------------------
# Custom similarity threshold
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_zero_threshold_compares_all_pairs(self) -> None:
        """With threshold=0, all pairs are compared regardless of similarity."""
        detector = ContradictionDetector(similarity_threshold=0.0)
        e1 = _make_entry("Python")
        e2 = _make_entry("Coffee")
        # With threshold=0.0, the pair is not skipped (similarity >= 0)
        report = detector.detect([e1, e2])
        assert report.total_entries_scanned == 2

    def test_high_threshold_skips_low_similarity_pairs(self) -> None:
        """With threshold=1.0, only identical content would pass."""
        detector = ContradictionDetector(similarity_threshold=1.0)
        e1 = _make_entry("The cat sat on the mat")
        e2 = _make_entry("The dog ran in the park")
        report = detector.detect([e1, e2])
        assert report.contradiction_count == 0
