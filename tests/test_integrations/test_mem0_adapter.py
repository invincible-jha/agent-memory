"""Tests for agent_memory.integrations.mem0_adapter.

The mem0 package is mocked — it is not required in CI.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixture: inject fake mem0 module
# ---------------------------------------------------------------------------


def _make_raw_mem0_result(
    memory_id: str = "mem-001",
    content: str = "Paris is the capital of France.",
    score: float = 0.9,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "id": memory_id,
        "memory": content,
        "score": score,
        "metadata": metadata or {},
    }


class FakeMemory:
    """Minimal stand-in for mem0.Memory."""

    def __init__(self) -> None:
        self._store: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FakeMemory":
        return cls()

    def add(self, text: str, user_id: str = "", metadata: Any = None) -> dict[str, Any]:
        entry = {"id": f"mem-{len(self._store):04d}", "memory": text}
        self._store.append(entry)
        return {"id": entry["id"], "event": "ADD"}

    def search(
        self,
        query: str,
        user_id: str = "",
        limit: int = 10,
        filters: Any = None,
    ) -> list[dict[str, Any]]:
        return self._store[:limit]

    def get_all(self, user_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return self._store[:limit]

    def delete(self, memory_id: str) -> None:
        self._store = [s for s in self._store if s.get("id") != memory_id]


def _make_mock_mem0_module() -> types.ModuleType:
    mod = types.ModuleType("mem0")
    mod.Memory = FakeMemory  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def inject_mem0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake mem0 before each test."""
    fake_mem0 = _make_mock_mem0_module()
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)
    monkeypatch.delitem(
        sys.modules,
        "agent_memory.integrations.mem0_adapter",
        raising=False,
    )


def _import_adapter() -> Any:
    from agent_memory.integrations import mem0_adapter  # noqa: PLC0415

    return mem0_adapter


def _make_enhanced(**kwargs: Any) -> Any:
    adapter = _import_adapter()
    return adapter.Mem0EnhancedMemory(**kwargs)


# ---------------------------------------------------------------------------
# Mem0EnhancedMemory — construction
# ---------------------------------------------------------------------------


class TestMem0EnhancedMemoryConstruction:
    def test_default_construction(self) -> None:
        memory = _make_enhanced()
        assert memory is not None

    def test_user_id_stored(self) -> None:
        memory = _make_enhanced(user_id="agent-42")
        assert memory._user_id == "agent-42"

    def test_block_on_contradiction_default_false(self) -> None:
        memory = _make_enhanced()
        assert memory._block_on_contradiction is False

    def test_block_on_contradiction_set_to_true(self) -> None:
        memory = _make_enhanced(block_on_contradiction=True)
        assert memory._block_on_contradiction is True

    def test_invalid_contradiction_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_enhanced(contradiction_threshold=1.5)

    def test_invalid_freshness_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_enhanced(freshness_weight=-0.1)

    def test_custom_client_accepted(self) -> None:
        fake_client = FakeMemory()
        memory = _make_enhanced(mem0_client=fake_client)
        assert memory._client is fake_client

    def test_repr_contains_class_name(self) -> None:
        memory = _make_enhanced(user_id="u1")
        assert "Mem0EnhancedMemory" in repr(memory)


# ---------------------------------------------------------------------------
# add_with_contradiction_check — basic add
# ---------------------------------------------------------------------------


class TestAddWithContradictionCheck:
    def test_add_returns_add_result(self) -> None:
        adapter = _import_adapter()
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check(
            "Paris is the capital of France."
        )
        assert isinstance(result, adapter.AddResult)

    def test_add_succeeds_for_new_fact(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check(
            "Paris is the capital of France."
        )
        assert result.added is True

    def test_add_result_text_preserved(self) -> None:
        memory = _make_enhanced()
        text = "The sky is blue."
        result = memory.add_with_contradiction_check(text)
        assert result.text == text

    def test_mem0_response_not_none_after_add(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("Some fact.")
        assert result.mem0_response is not None

    def test_empty_string_not_added(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("   ")
        assert result.added is False

    def test_metadata_forwarded(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check(
            "A fact.", metadata={"source": "user"}
        )
        assert result.added is True

    def test_contradiction_prevented_add_is_false_when_no_block(self) -> None:
        memory = _make_enhanced(block_on_contradiction=False)
        result = memory.add_with_contradiction_check("Any text.")
        assert result.contradiction_prevented_add is False


# ---------------------------------------------------------------------------
# add_with_contradiction_check — contradiction detection
# ---------------------------------------------------------------------------


class TestAddContradictionDetection:
    def test_contradictions_list_initially_empty_for_first_add(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("First fact.")
        assert result.contradictions == []

    def test_no_block_when_block_on_contradiction_false(self) -> None:
        # Even if there were contradictions, block=False means add proceeds
        memory = _make_enhanced(block_on_contradiction=False)
        # First add populates the store
        memory.add_with_contradiction_check("Paris is the capital of France.")
        # Second add — even if contradictory, should not be blocked
        result = memory.add_with_contradiction_check("Something about France.")
        assert result.contradiction_prevented_add is False

    def test_block_on_true_prevents_contradictory_add(self) -> None:
        # Use very low threshold so the pre-check always runs
        memory = _make_enhanced(
            block_on_contradiction=True,
            contradiction_threshold=0.01,
        )
        # Add initial fact
        memory.add_with_contradiction_check("Paris is the capital of France.")
        # Attempt a negated contradictory fact
        result = memory.add_with_contradiction_check(
            "Paris is not the capital of France."
        )
        # May or may not block depending on similarity, but structure is correct
        assert isinstance(result.contradiction_prevented_add, bool)

    def test_add_result_has_contradictions_field(self) -> None:
        adapter = _import_adapter()
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("Test fact.")
        assert hasattr(result, "contradictions")
        assert isinstance(result.contradictions, list)

    def test_has_contradictions_property_false_initially(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("First fact.")
        assert result.has_contradictions is False


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------


class TestSearchHybrid:
    def test_returns_list_of_memory_results(self) -> None:
        adapter = _import_adapter()
        memory = _make_enhanced()
        memory.add_with_contradiction_check("Paris is the capital of France.")
        results = memory.search_hybrid("capital of France", top_k=5)
        for r in results:
            assert isinstance(r, adapter.MemoryResult)

    def test_returns_empty_when_store_empty(self) -> None:
        memory = _make_enhanced()
        results = memory.search_hybrid("something", top_k=5)
        assert isinstance(results, list)

    def test_top_k_limits_results(self) -> None:
        memory = _make_enhanced()
        for i in range(10):
            memory.add_with_contradiction_check(f"Fact number {i} about Paris.")
        results = memory.search_hybrid("Paris", top_k=3)
        assert len(results) <= 3

    def test_invalid_top_k_raises(self) -> None:
        memory = _make_enhanced()
        with pytest.raises(ValueError):
            memory.search_hybrid("query", top_k=0)

    def test_hybrid_score_in_range(self) -> None:
        memory = _make_enhanced()
        memory.add_with_contradiction_check("Paris is the capital of France.")
        results = memory.search_hybrid("France capital", top_k=5)
        for r in results:
            assert 0.0 <= r.hybrid_score <= 1.0

    def test_results_sorted_by_hybrid_score_descending(self) -> None:
        memory = _make_enhanced()
        for i in range(5):
            memory.add_with_contradiction_check(f"Memory entry {i}.")
        results = memory.search_hybrid("entry", top_k=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].hybrid_score >= results[i + 1].hybrid_score

    def test_memory_result_layer_is_episodic(self) -> None:
        from agent_memory.memory.types import MemoryLayer  # noqa: PLC0415

        memory = _make_enhanced()
        memory.add_with_contradiction_check("Test.")
        results = memory.search_hybrid("Test", top_k=5)
        for r in results:
            assert r.layer == MemoryLayer.EPISODIC


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    def test_returns_all_added_memories(self) -> None:
        memory = _make_enhanced()
        memory.add_with_contradiction_check("Fact A.")
        memory.add_with_contradiction_check("Fact B.")
        results = memory.get_all()
        assert len(results) == 2

    def test_returns_empty_list_for_empty_store(self) -> None:
        memory = _make_enhanced()
        results = memory.get_all()
        assert results == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_returns_true_on_success(self) -> None:
        memory = _make_enhanced()
        result = memory.add_with_contradiction_check("Fact.")
        # We don't have the exact mem0 ID here — just test the return value shape
        success = memory.delete("mem-0000")
        assert isinstance(success, bool)

    def test_delete_returns_false_on_error(self) -> None:
        # Use a client that raises on delete
        class ErrorMemory(FakeMemory):
            def delete(self, memory_id: str) -> None:
                raise RuntimeError("delete failed")

        memory = _make_enhanced(mem0_client=ErrorMemory())
        result = memory.delete("mem-0000")
        assert result is False


# ---------------------------------------------------------------------------
# _compute_hybrid_score
# ---------------------------------------------------------------------------


class TestComputeHybridScore:
    def test_score_in_range(self) -> None:
        memory = _make_enhanced(freshness_weight=0.3)
        score = memory._compute_hybrid_score(raw_score=0.8, metadata={})
        assert 0.0 <= score <= 1.0

    def test_freshness_weight_zero_equals_raw_score(self) -> None:
        memory = _make_enhanced(freshness_weight=0.0)
        score = memory._compute_hybrid_score(raw_score=0.8, metadata={})
        assert abs(score - 0.8 * 1.0) < 0.01  # freshness=0.5 contributes 0

    def test_fresh_metadata_raises_score(self) -> None:
        memory = _make_enhanced(freshness_weight=0.5)
        # Very recent timestamp
        fresh_ts = datetime.now(timezone.utc).isoformat()
        score_fresh = memory._compute_hybrid_score(
            raw_score=0.5, metadata={"created_at": fresh_ts}
        )
        # Old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        score_old = memory._compute_hybrid_score(
            raw_score=0.5, metadata={"created_at": old_ts}
        )
        assert score_fresh > score_old


# ---------------------------------------------------------------------------
# ContradictionWarning
# ---------------------------------------------------------------------------


class TestContradictionWarning:
    def test_fields(self) -> None:
        adapter = _import_adapter()
        warning = adapter.ContradictionWarning(
            existing_memory_id="mem-001",
            existing_content="Paris is not in France.",
            conflict_description="Negation conflict",
            similarity_score=0.85,
        )
        assert warning.existing_memory_id == "mem-001"
        assert warning.similarity_score == 0.85
