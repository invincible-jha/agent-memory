"""Contradiction report — structured representation of detected contradictions."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent_memory.memory.types import MemoryEntry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContradictionPair(BaseModel):
    """Two memory entries that contradict each other."""

    entry_a_id: str
    entry_b_id: str
    entry_a_content: str
    entry_b_content: str
    conflict_description: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    detected_at: datetime = Field(default_factory=_utcnow)
    resolved: bool = False
    resolution_strategy: str = ""
    winner_id: str = ""

    model_config = {"frozen": False}


class ContradictionReport(BaseModel):
    """Aggregate report of all contradictions found in a memory scan."""

    generated_at: datetime = Field(default_factory=_utcnow)
    total_entries_scanned: int = 0
    contradiction_pairs: list[ContradictionPair] = Field(default_factory=list)

    @property
    def contradiction_count(self) -> int:
        return len(self.contradiction_pairs)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for p in self.contradiction_pairs if not p.resolved)

    def mark_resolved(
        self,
        entry_a_id: str,
        entry_b_id: str,
        strategy: str,
        winner_id: str,
    ) -> None:
        """Mark a contradiction pair as resolved."""
        for pair in self.contradiction_pairs:
            if (pair.entry_a_id == entry_a_id and pair.entry_b_id == entry_b_id) or (
                pair.entry_a_id == entry_b_id and pair.entry_b_id == entry_a_id
            ):
                pair.resolved = True
                pair.resolution_strategy = strategy
                pair.winner_id = winner_id
                return

    def summary(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_scanned": self.total_entries_scanned,
            "total_contradictions": self.contradiction_count,
            "unresolved": self.unresolved_count,
            "resolved": self.contradiction_count - self.unresolved_count,
        }

    model_config = {"frozen": False}


__all__ = ["ContradictionPair", "ContradictionReport"]
