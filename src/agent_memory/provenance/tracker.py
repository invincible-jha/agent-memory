"""Provenance tracker — tag each memory with source and reliability metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_memory.memory.types import MemoryEntry, MemorySource
from agent_memory.provenance.reliability import SourceReliability


class ProvenanceTracker:
    """Attach provenance metadata to memory entries.

    Provenance information is stored in ``entry.metadata`` using well-known
    keys so it survives serialisation through any backend.
    """

    _RELIABILITY_KEY = "provenance_reliability"
    _TIER_KEY = "provenance_tier"
    _TAGGED_AT_KEY = "provenance_tagged_at"
    _SOURCE_NOTE_KEY = "provenance_note"

    def __init__(self) -> None:
        self._reliability = SourceReliability()

    def tag(
        self,
        entry: MemoryEntry,
        note: str = "",
    ) -> MemoryEntry:
        """Return a new entry with provenance metadata applied from its source."""
        rel_score = self._reliability.score(entry.source)
        tier = self._reliability.tier_label(entry.source)
        now_iso = datetime.now(timezone.utc).isoformat()
        updated_meta = dict(entry.metadata)
        updated_meta[self._RELIABILITY_KEY] = str(round(rel_score, 4))
        updated_meta[self._TIER_KEY] = tier
        updated_meta[self._TAGGED_AT_KEY] = now_iso
        if note:
            updated_meta[self._SOURCE_NOTE_KEY] = note
        return entry.model_copy(update={"metadata": updated_meta})

    def get_reliability(self, entry: MemoryEntry) -> float:
        """Return the stored reliability score, or re-compute from source."""
        raw = entry.metadata.get(self._RELIABILITY_KEY)
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
        return self._reliability.score(entry.source)

    def get_tier(self, entry: MemoryEntry) -> str:
        """Return the stored tier label, or derive from source."""
        return entry.metadata.get(self._TIER_KEY) or self._reliability.tier_label(entry.source)

    def is_tagged(self, entry: MemoryEntry) -> bool:
        """Return True if provenance metadata has been attached."""
        return self._RELIABILITY_KEY in entry.metadata

    def override_reliability(
        self,
        entry: MemoryEntry,
        new_source: MemorySource,
    ) -> MemoryEntry:
        """Re-tag an entry with a different source, updating all provenance fields."""
        updated = entry.model_copy(update={"source": new_source})
        return self.tag(updated)


__all__ = ["ProvenanceTracker"]
