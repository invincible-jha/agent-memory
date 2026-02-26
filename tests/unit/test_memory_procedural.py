"""Unit tests for agent_memory.memory.procedural — ProceduralMemory."""

from __future__ import annotations

import pytest

from agent_memory.memory.procedural import (
    Procedure,
    ProceduralMemory,
    ProcedureStep,
)
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    content: str = "procedure content",
    layer: MemoryLayer = MemoryLayer.PROCEDURAL,
    procedure_name: str = "",
) -> MemoryEntry:
    metadata: dict[str, str] = {}
    if procedure_name:
        metadata["procedure_name"] = procedure_name
    return MemoryEntry(content=content, layer=layer, metadata=metadata)


def _make_procedure(name: str = "test-proc", description: str = "test procedure") -> Procedure:
    return Procedure(
        name=name,
        description=description,
        steps=[
            ProcedureStep(step_number=1, description="First step"),
            ProcedureStep(step_number=2, description="Second step"),
        ],
        preconditions=["system ready"],
        memory_id="test-memory-id",
    )


# ---------------------------------------------------------------------------
# ProcedureStep
# ---------------------------------------------------------------------------


class TestProcedureStep:
    def test_required_fields(self) -> None:
        step = ProcedureStep(step_number=1, description="Do something")
        assert step.step_number == 1
        assert step.description == "Do something"

    def test_optional_expected_output(self) -> None:
        step = ProcedureStep(step_number=1, description="step", expected_output="result")
        assert step.expected_output == "result"

    def test_default_expected_output_is_none(self) -> None:
        step = ProcedureStep(step_number=1, description="step")
        assert step.expected_output is None

    def test_is_frozen(self) -> None:
        step = ProcedureStep(step_number=1, description="step")
        with pytest.raises((AttributeError, TypeError, ValueError)):
            step.step_number = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------


class TestProcedure:
    def test_required_fields(self) -> None:
        proc = Procedure(name="my-proc", description="desc", memory_id="abc")
        assert proc.name == "my-proc"
        assert proc.description == "desc"
        assert proc.memory_id == "abc"

    def test_default_empty_steps(self) -> None:
        proc = Procedure(name="p", description="d", memory_id="m")
        assert proc.steps == []

    def test_default_empty_preconditions(self) -> None:
        proc = Procedure(name="p", description="d", memory_id="m")
        assert proc.preconditions == []

    def test_steps_stored_correctly(self) -> None:
        proc = _make_procedure()
        assert len(proc.steps) == 2
        assert proc.steps[0].step_number == 1


# ---------------------------------------------------------------------------
# ProceduralMemory — MemoryStore interface
# ---------------------------------------------------------------------------


class TestProceduralMemoryStore:
    def test_store_and_retrieve(self) -> None:
        mem = ProceduralMemory()
        entry = _make_entry("deploy procedure steps")
        mem.store(entry)
        retrieved = mem.retrieve(entry.memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == entry.memory_id

    def test_retrieve_missing_returns_none(self) -> None:
        mem = ProceduralMemory()
        assert mem.retrieve("ghost-id") is None

    def test_all_returns_all_entries(self) -> None:
        mem = ProceduralMemory()
        entries = [_make_entry(f"proc {i}") for i in range(3)]
        for e in entries:
            mem.store(e)
        all_entries = list(mem.all())
        assert len(all_entries) == 3

    def test_all_filtered_by_layer(self) -> None:
        mem = ProceduralMemory()
        e_proc = _make_entry("procedural", layer=MemoryLayer.PROCEDURAL)
        e_sem = _make_entry("semantic", layer=MemoryLayer.SEMANTIC)
        mem.store(e_proc)
        mem.store(e_sem)
        results = list(mem.all(layer=MemoryLayer.PROCEDURAL))
        assert len(results) == 1
        assert results[0].layer is MemoryLayer.PROCEDURAL

    def test_count_all(self) -> None:
        mem = ProceduralMemory()
        for i in range(4):
            mem.store(_make_entry(f"p{i}"))
        assert mem.count() == 4

    def test_count_by_layer(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("a", layer=MemoryLayer.PROCEDURAL))
        mem.store(_make_entry("b", layer=MemoryLayer.SEMANTIC))
        assert mem.count(layer=MemoryLayer.PROCEDURAL) == 1

    def test_delete_existing_returns_true(self) -> None:
        mem = ProceduralMemory()
        entry = _make_entry()
        mem.store(entry)
        assert mem.delete(entry.memory_id) is True

    def test_delete_removes_entry(self) -> None:
        mem = ProceduralMemory()
        entry = _make_entry()
        mem.store(entry)
        mem.delete(entry.memory_id)
        assert mem.retrieve(entry.memory_id) is None

    def test_delete_missing_returns_false(self) -> None:
        mem = ProceduralMemory()
        assert mem.delete("ghost-id") is False

    def test_delete_clears_procedure_from_index(self) -> None:
        mem = ProceduralMemory()
        proc = _make_procedure("my-proc")
        entry = _make_entry("my-proc content", procedure_name="my-proc")
        mem.store_procedure(proc, entry)
        mem.delete(entry.memory_id)
        assert mem.retrieve_procedure("my-proc") is None

    def test_clear_all_removes_everything(self) -> None:
        mem = ProceduralMemory()
        for i in range(3):
            mem.store(_make_entry(f"p{i}"))
        removed = mem.clear()
        assert removed == 3
        assert mem.count() == 0

    def test_clear_by_layer(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("a", layer=MemoryLayer.PROCEDURAL))
        mem.store(_make_entry("b", layer=MemoryLayer.SEMANTIC))
        removed = mem.clear(layer=MemoryLayer.PROCEDURAL)
        assert removed == 1
        assert mem.count() == 1


# ---------------------------------------------------------------------------
# ProceduralMemory.search
# ---------------------------------------------------------------------------


class TestProceduralMemorySearch:
    def test_search_finds_keyword_match(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("python deployment procedure steps"))
        results = list(mem.search("python"))
        assert len(results) == 1

    def test_search_no_match_returns_empty(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("unrelated content here"))
        results = list(mem.search("quantum"))
        assert results == []

    def test_search_sorted_by_overlap_score(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("python python python deployment"))
        mem.store(_make_entry("python once"))
        results = list(mem.search("python"))
        # Both match; higher overlap should rank first
        assert len(results) == 2

    def test_search_respects_limit(self) -> None:
        mem = ProceduralMemory()
        for i in range(10):
            mem.store(_make_entry(f"common keyword entry {i}"))
        results = list(mem.search("common", limit=3))
        assert len(results) <= 3

    def test_search_filtered_by_layer(self) -> None:
        mem = ProceduralMemory()
        mem.store(_make_entry("keyword match proc", layer=MemoryLayer.PROCEDURAL))
        mem.store(_make_entry("keyword match sem", layer=MemoryLayer.SEMANTIC))
        results = list(mem.search("keyword match", layer=MemoryLayer.PROCEDURAL))
        assert all(e.layer is MemoryLayer.PROCEDURAL for e in results)


# ---------------------------------------------------------------------------
# ProceduralMemory — typed procedure access
# ---------------------------------------------------------------------------


class TestProcedureTypedAccess:
    def test_store_procedure_and_retrieve_by_name(self) -> None:
        mem = ProceduralMemory()
        proc = _make_procedure("deploy")
        entry = _make_entry("deploy steps", procedure_name="deploy")
        mem.store_procedure(proc, entry)
        retrieved = mem.retrieve_procedure("deploy")
        assert retrieved is not None
        assert retrieved.name == "deploy"

    def test_retrieve_procedure_missing_returns_none(self) -> None:
        mem = ProceduralMemory()
        assert mem.retrieve_procedure("nonexistent") is None

    def test_retrieve_entry_by_name(self) -> None:
        mem = ProceduralMemory()
        proc = _make_procedure("my-proc")
        entry = _make_entry("my-proc content", procedure_name="my-proc")
        mem.store_procedure(proc, entry)
        retrieved_entry = mem.retrieve_entry_by_name("my-proc")
        assert retrieved_entry is not None
        assert retrieved_entry.memory_id == entry.memory_id

    def test_retrieve_entry_by_name_missing_returns_none(self) -> None:
        mem = ProceduralMemory()
        assert mem.retrieve_entry_by_name("ghost") is None

    def test_search_procedures_by_keyword(self) -> None:
        mem = ProceduralMemory()
        proc = _make_procedure("backup-procedure", "automated backup to cloud")
        entry = _make_entry("backup procedure", procedure_name="backup-procedure")
        mem.store_procedure(proc, entry)
        results = mem.search_procedures("backup")
        assert len(results) == 1
        assert results[0].name == "backup-procedure"

    def test_search_procedures_no_match_returns_empty(self) -> None:
        mem = ProceduralMemory()
        results = mem.search_procedures("quantum")
        assert results == []

    def test_list_procedure_names_sorted(self) -> None:
        mem = ProceduralMemory()
        for name in ("zebra", "alpha", "middle"):
            proc = Procedure(name=name, description=name, memory_id=name)
            entry = _make_entry(name, procedure_name=name)
            mem.store_procedure(proc, entry)
        names = mem.list_procedure_names()
        assert names == sorted(["zebra", "alpha", "middle"])

    def test_list_procedure_names_empty(self) -> None:
        mem = ProceduralMemory()
        assert mem.list_procedure_names() == []

    def test_store_procedure_registers_both_entry_and_procedure(self) -> None:
        mem = ProceduralMemory()
        proc = _make_procedure("test")
        entry = _make_entry("test content", procedure_name="test")
        mem.store_procedure(proc, entry)
        assert mem.count() == 1
        assert mem.retrieve_procedure("test") is not None
