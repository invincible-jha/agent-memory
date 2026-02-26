"""Refresh trigger — mark memories for re-verification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from agent_memory.memory.types import MemoryEntry


_NEEDS_REFRESH_KEY = "needs_refresh"
_REFRESH_REQUESTED_AT_KEY = "refresh_requested_at"


class RefreshTrigger:
    """Mark memory entries as requiring re-verification.

    This class does not perform actual refresh (that would require an external
    data source).  It stamps entries with metadata flags so downstream systems
    can act on them.
    """

    def mark_for_refresh(self, entry: MemoryEntry) -> MemoryEntry:
        """Return a new entry with the ``needs_refresh`` metadata flag set."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updated_meta = dict(entry.metadata)
        updated_meta[_NEEDS_REFRESH_KEY] = "true"
        updated_meta[_REFRESH_REQUESTED_AT_KEY] = now_iso
        return entry.model_copy(update={"metadata": updated_meta})

    def clear_refresh_flag(self, entry: MemoryEntry) -> MemoryEntry:
        """Return a new entry with the ``needs_refresh`` flag cleared and
        ``last_verified`` set to now."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updated_meta = dict(entry.metadata)
        updated_meta.pop(_NEEDS_REFRESH_KEY, None)
        updated_meta.pop(_REFRESH_REQUESTED_AT_KEY, None)
        updated_meta["last_verified"] = now_iso
        return entry.model_copy(
            update={
                "metadata": updated_meta,
                "freshness_score": 1.0,
            }
        )

    def needs_refresh(self, entry: MemoryEntry) -> bool:
        """Return True if the entry has been flagged for re-verification."""
        return entry.metadata.get(_NEEDS_REFRESH_KEY, "false").lower() == "true"

    def filter_pending(self, entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
        """Return all entries that are pending refresh."""
        return [e for e in entries if self.needs_refresh(e)]


__all__ = ["RefreshTrigger"]
