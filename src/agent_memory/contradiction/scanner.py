"""Contradiction scanner — scans a memory store for contradictions at scale."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence

from agent_memory.contradiction.detector import ContradictionDetector
from agent_memory.contradiction.report import ContradictionPair
from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer


class ScanScope(str, Enum):
    """Which entries to include in a contradiction scan."""

    ALL = "all"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    HIGH_IMPORTANCE = "high_importance"


@dataclass
class ScanResult:
    """Aggregated result of a contradiction scan over a memory store."""

    scanned_count: int
    contradiction_pairs: list[ContradictionPair] = field(default_factory=list)
    skipped_count: int = 0
    early_terminated: bool = False
    scan_duration_seconds: float = 0.0
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def contradiction_count(self) -> int:
        return len(self.contradiction_pairs)

    def summary(self) -> dict[str, object]:
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "scanned_count": self.scanned_count,
            "skipped_count": self.skipped_count,
            "contradiction_count": self.contradiction_count,
            "early_terminated": self.early_terminated,
            "scan_duration_seconds": round(self.scan_duration_seconds, 3),
        }


# Default importance threshold used when scope is HIGH_IMPORTANCE
_HIGH_IMPORTANCE_THRESHOLD = 0.6


class ContradictionScanner:
    """Scan a memory store for contradictions using pairwise comparison.

    Pairwise comparison is O(n^2) in the number of entries. The scanner
    supports early termination (``max_contradictions``) and scoped scanning
    to limit the search space.

    Parameters
    ----------
    detector:
        The contradiction detector to use for pairwise comparisons.
        A default ``ContradictionDetector`` is created if not provided.
    max_contradictions:
        Stop scanning after this many contradictions are found. 0 means
        no limit (scan all pairs).
    similarity_threshold:
        Topic similarity threshold forwarded to the detector. Entries
        whose Jaccard similarity is below this value are skipped quickly.
    high_importance_threshold:
        Minimum importance_score for entries included in the
        ``HIGH_IMPORTANCE`` scope. Defaults to 0.6.
    """

    def __init__(
        self,
        detector: Optional[ContradictionDetector] = None,
        max_contradictions: int = 0,
        similarity_threshold: float = 0.25,
        high_importance_threshold: float = _HIGH_IMPORTANCE_THRESHOLD,
    ) -> None:
        self._detector = detector or ContradictionDetector(
            similarity_threshold=similarity_threshold
        )
        self._max_contradictions = max_contradictions
        self._high_importance_threshold = high_importance_threshold

    def scan(
        self,
        store: MemoryStore,
        scope: ScanScope = ScanScope.ALL,
    ) -> ScanResult:
        """Scan the store for contradictions within the given scope.

        Parameters
        ----------
        store:
            The memory store to scan.
        scope:
            Which entries to include. Defaults to ``ALL``.

        Returns
        -------
        ScanResult
            Contains all detected contradiction pairs and scan metadata.
        """
        import time

        start_time = time.monotonic()
        candidates = self._collect_candidates(store, scope)
        result = self._pairwise_scan(candidates)
        result.scan_duration_seconds = time.monotonic() - start_time
        return result

    def scan_entries(
        self,
        entries: Sequence[MemoryEntry],
    ) -> ScanResult:
        """Scan an explicit list of entries rather than a store.

        Parameters
        ----------
        entries:
            The entries to check for contradictions.

        Returns
        -------
        ScanResult
        """
        import time

        start_time = time.monotonic()
        result = self._pairwise_scan(list(entries))
        result.scan_duration_seconds = time.monotonic() - start_time
        return result

    def scan_against(
        self,
        entry: MemoryEntry,
        store: MemoryStore,
        scope: ScanScope = ScanScope.ALL,
    ) -> ScanResult:
        """Find contradictions between a single entry and all store entries.

        Useful for checking a newly added entry against the existing store
        without a full pairwise scan.

        Parameters
        ----------
        entry:
            The entry to check.
        store:
            The store to search within.
        scope:
            Restricts which stored entries are checked.

        Returns
        -------
        ScanResult
        """
        import time

        start_time = time.monotonic()
        candidates = self._collect_candidates(store, scope)
        # Exclude the entry itself if it is already in the store
        candidates = [c for c in candidates if c.memory_id != entry.memory_id]

        pairs = self._detector.detect_for_entry(entry, candidates)
        result = ScanResult(
            scanned_count=len(candidates),
            contradiction_pairs=pairs,
        )
        result.scan_duration_seconds = time.monotonic() - start_time
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_candidates(
        self,
        store: MemoryStore,
        scope: ScanScope,
    ) -> list[MemoryEntry]:
        """Return the filtered set of entries to scan based on scope."""
        layer_map: dict[ScanScope, MemoryLayer] = {
            ScanScope.WORKING: MemoryLayer.WORKING,
            ScanScope.EPISODIC: MemoryLayer.EPISODIC,
            ScanScope.SEMANTIC: MemoryLayer.SEMANTIC,
            ScanScope.PROCEDURAL: MemoryLayer.PROCEDURAL,
        }

        if scope in layer_map:
            return list(store.all(layer=layer_map[scope]))

        if scope == ScanScope.HIGH_IMPORTANCE:
            return [
                e for e in store.all()
                if e.importance_score >= self._high_importance_threshold
            ]

        # ScanScope.ALL
        return list(store.all())

    def _pairwise_scan(self, entries: list[MemoryEntry]) -> ScanResult:
        """Run pairwise contradiction detection with optional early termination."""
        found: list[ContradictionPair] = []
        skipped = 0
        early_terminated = False

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if self._max_contradictions > 0 and len(found) >= self._max_contradictions:
                    # Count remaining unchecked pairs as skipped
                    remaining = (len(entries) - i - 1) * len(entries) // 2
                    skipped += max(0, remaining)
                    early_terminated = True
                    break

                pair = self._detector._compare(entries[i], entries[j])
                if pair is not None:
                    found.append(pair)

            if early_terminated:
                break

        return ScanResult(
            scanned_count=len(entries),
            contradiction_pairs=found,
            skipped_count=skipped,
            early_terminated=early_terminated,
        )


__all__ = ["ContradictionScanner", "ScanResult", "ScanScope"]
