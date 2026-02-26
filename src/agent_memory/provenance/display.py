"""Provenance display — format provenance information for retrieval results."""

from __future__ import annotations

from agent_memory.memory.types import MemoryEntry
from agent_memory.provenance.tracker import ProvenanceTracker


class ProvenanceDisplay:
    """Format provenance information for human-readable and structured output."""

    def __init__(self, tracker: ProvenanceTracker | None = None) -> None:
        self._tracker = tracker or ProvenanceTracker()

    def summary(self, entry: MemoryEntry) -> str:
        """Return a single-line provenance summary string."""
        tier = self._tracker.get_tier(entry)
        rel = self._tracker.get_reliability(entry)
        tagged_at = entry.metadata.get("provenance_tagged_at", "unknown")
        note = entry.metadata.get("provenance_note", "")
        parts = [
            f"source={entry.source.value}",
            f"tier={tier}",
            f"reliability={rel:.2f}",
            f"tagged_at={tagged_at}",
        ]
        if note:
            parts.append(f"note={note!r}")
        return " | ".join(parts)

    def as_dict(self, entry: MemoryEntry) -> dict[str, str | float]:
        """Return provenance data as a structured dictionary."""
        return {
            "memory_id": entry.memory_id,
            "source": entry.source.value,
            "tier": self._tracker.get_tier(entry),
            "reliability": self._tracker.get_reliability(entry),
            "tagged_at": entry.metadata.get("provenance_tagged_at", ""),
            "note": entry.metadata.get("provenance_note", ""),
        }

    def format_batch(self, entries: list[MemoryEntry]) -> list[dict[str, str | float]]:
        """Format provenance for a list of entries."""
        return [self.as_dict(e) for e in entries]


__all__ = ["ProvenanceDisplay"]
