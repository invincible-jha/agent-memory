"""Contradiction detector — find conflicting memories via entity-attribute matching."""

from __future__ import annotations

from typing import Sequence

from agent_memory.contradiction.matcher import EntityAttributeMatcher
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.memory.types import MemoryEntry

# Similarity threshold above which two entries are considered to address the
# same topic and are therefore worth comparing for contradictions.
_TOPIC_SIMILARITY_THRESHOLD = 0.25


class ContradictionDetector:
    """Detect contradictions between memory entries.

    Detection strategy:
    1. For each pair of entries, check if their content is topically similar
       (Jaccard >= threshold) — if not, skip the pair.
    2. Extract entity-attribute-value triples from both.
    3. Flag pairs that share entity+attribute but differ in value.
    4. Also flag pairs with high textual similarity but opposite polarity
       (negation detection).
    """

    def __init__(
        self,
        similarity_threshold: float = _TOPIC_SIMILARITY_THRESHOLD,
    ) -> None:
        self._threshold = similarity_threshold
        self._matcher = EntityAttributeMatcher()

    def detect(self, entries: Sequence[MemoryEntry]) -> ContradictionReport:
        """Scan all entries and return a ContradictionReport."""
        entry_list = list(entries)
        report = ContradictionReport(total_entries_scanned=len(entry_list))

        for i in range(len(entry_list)):
            for j in range(i + 1, len(entry_list)):
                pair = self._compare(entry_list[i], entry_list[j])
                if pair is not None:
                    report.contradiction_pairs.append(pair)

        return report

    def detect_for_entry(
        self,
        entry: MemoryEntry,
        candidates: Sequence[MemoryEntry],
    ) -> list[ContradictionPair]:
        """Find contradictions between one entry and a set of candidates."""
        result: list[ContradictionPair] = []
        for candidate in candidates:
            if candidate.memory_id == entry.memory_id:
                continue
            pair = self._compare(entry, candidate)
            if pair is not None:
                result.append(pair)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare(
        self, entry_a: MemoryEntry, entry_b: MemoryEntry
    ) -> ContradictionPair | None:
        similarity = self._matcher.similarity(entry_a.content, entry_b.content)
        if similarity < self._threshold:
            return None

        triples_a = self._matcher.extract_triples(entry_a.content)
        triples_b = self._matcher.extract_triples(entry_b.content)
        conflicts = self._matcher.conflicts(triples_a, triples_b)

        if conflicts:
            conflict_descs = [
                f"entity={c[0].entity!r} attr={c[0].attribute!r}: "
                f"{c[0].value!r} vs {c[1].value!r}"
                for c in conflicts
            ]
            return ContradictionPair(
                entry_a_id=entry_a.memory_id,
                entry_b_id=entry_b.memory_id,
                entry_a_content=entry_a.content,
                entry_b_content=entry_b.content,
                conflict_description="; ".join(conflict_descs),
                similarity_score=round(similarity, 4),
            )

        # Negation check: high similarity but one sentence negates the other
        if similarity >= 0.5 and self._has_negation_conflict(
            entry_a.content, entry_b.content
        ):
            return ContradictionPair(
                entry_a_id=entry_a.memory_id,
                entry_b_id=entry_b.memory_id,
                entry_a_content=entry_a.content,
                entry_b_content=entry_b.content,
                conflict_description="Negation conflict detected",
                similarity_score=round(similarity, 4),
            )

        return None

    def _has_negation_conflict(self, text_a: str, text_b: str) -> bool:
        """Return True if exactly one text contains a negation marker."""
        import re

        negation_re = re.compile(r"\b(not|never|no|cannot|can't|won't|isn't|aren't)\b", re.IGNORECASE)
        a_has = bool(negation_re.search(text_a))
        b_has = bool(negation_re.search(text_b))
        return a_has != b_has


__all__ = ["ContradictionDetector"]
