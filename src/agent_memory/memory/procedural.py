"""Procedural memory — named procedures with steps, preconditions, and keyword retrieval."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer


class ProcedureStep(BaseModel):
    """A single step within a named procedure."""

    step_number: int
    description: str
    expected_output: Optional[str] = None

    model_config = {"frozen": True}


class Procedure(BaseModel):
    """Structured procedure with ordered steps and optional preconditions."""

    name: str
    description: str
    steps: list[ProcedureStep] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    memory_id: str

    model_config = {"frozen": True}


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ProceduralMemory(MemoryStore):
    """Stores named procedures; retrieval by exact name or keyword matching.

    Procedures are stored as MemoryEntry objects with the structured
    procedure data embedded in the ``metadata`` dict via JSON, plus a
    ``Procedure`` object kept in a parallel registry for typed access.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._procedures: dict[str, Procedure] = {}
        # name -> memory_id for exact-name lookup
        self._name_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        self._entries[entry.memory_id] = entry

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(memory_id)

    def all(self, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        entries = list(self._entries.values())
        if layer is not None:
            entries = [e for e in entries if e.layer == layer]
        return entries

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            return len(self._entries)
        return sum(1 for e in self._entries.values() if e.layer == layer)

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._entries:
            return False
        entry = self._entries.pop(memory_id)
        proc_name = entry.metadata.get("procedure_name", "")
        if proc_name:
            self._name_index.pop(proc_name, None)
            self._procedures.pop(proc_name, None)
        return True

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            count = len(self._entries)
            self._entries.clear()
            self._procedures.clear()
            self._name_index.clear()
            return count
        to_delete = [mid for mid, e in self._entries.items() if e.layer == layer]
        for mid in to_delete:
            self.delete(mid)
        return len(to_delete)

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Keyword-based search across procedure names and descriptions."""
        query_words = _keywords(query)
        results: list[tuple[int, MemoryEntry]] = []
        for entry in self.all(layer=layer):
            entry_words = _keywords(entry.content)
            overlap = len(query_words & entry_words)
            if overlap > 0:
                results.append((overlap, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:limit]]

    # ------------------------------------------------------------------
    # Procedural-specific helpers
    # ------------------------------------------------------------------

    def store_procedure(
        self,
        procedure: Procedure,
        entry: MemoryEntry,
    ) -> None:
        """Register a typed Procedure alongside its MemoryEntry."""
        self.store(entry)
        self._procedures[procedure.name] = procedure
        self._name_index[procedure.name] = entry.memory_id

    def retrieve_procedure(self, name: str) -> Optional[Procedure]:
        """Return a typed Procedure by exact name."""
        return self._procedures.get(name)

    def retrieve_entry_by_name(self, name: str) -> Optional[MemoryEntry]:
        """Return the MemoryEntry for the named procedure."""
        mid = self._name_index.get(name)
        if mid is None:
            return None
        return self._entries.get(mid)

    def search_procedures(self, query: str) -> list[Procedure]:
        """Return procedures whose name or description matches query keywords."""
        query_words = _keywords(query)
        matches: list[tuple[int, Procedure]] = []
        for proc in self._procedures.values():
            proc_words = _keywords(proc.name) | _keywords(proc.description)
            overlap = len(query_words & proc_words)
            if overlap > 0:
                matches.append((overlap, proc))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in matches]

    def list_procedure_names(self) -> list[str]:
        return sorted(self._procedures.keys())


__all__ = ["ProceduralMemory", "Procedure", "ProcedureStep"]
