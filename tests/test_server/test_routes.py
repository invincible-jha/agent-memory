"""Tests for agent_memory.server.routes."""
from __future__ import annotations

import pytest

from agent_memory.server import routes


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestHandleCreateMemory:
    def test_creates_valid_memory(self) -> None:
        body = {
            "content": "The sky is blue.",
            "layer": "semantic",
            "importance_score": 0.8,
            "source": "user_input",
        }
        status, data = routes.handle_create_memory(body)

        assert status == 201
        assert "memory_id" in data
        assert data["content"] == "The sky is blue."
        assert data["layer"] == "semantic"

    def test_creates_memory_with_defaults(self) -> None:
        body = {"content": "Agent task completed."}
        status, data = routes.handle_create_memory(body)

        assert status == 201
        assert data["layer"] == "episodic"
        assert data["importance_score"] == 0.5

    def test_rejects_invalid_layer(self) -> None:
        body = {"content": "Hello", "layer": "nonexistent-layer"}
        status, data = routes.handle_create_memory(body)

        assert status == 422
        assert "error" in data

    def test_rejects_missing_content(self) -> None:
        body = {"layer": "semantic"}
        status, data = routes.handle_create_memory(body)

        assert status == 422
        assert "error" in data

    def test_preserves_metadata(self) -> None:
        body = {
            "content": "Tool output recorded.",
            "metadata": {"source_tool": "calculator", "run_id": "r1"},
        }
        status, data = routes.handle_create_memory(body)

        assert status == 201
        assert data["metadata"]["source_tool"] == "calculator"

    def test_safety_critical_flag(self) -> None:
        body = {"content": "Critical security rule.", "safety_critical": True}
        status, data = routes.handle_create_memory(body)

        assert status == 201
        assert data["safety_critical"] is True

    def test_importance_score_clamped_high(self) -> None:
        body = {"content": "Test", "importance_score": 1.0}
        status, data = routes.handle_create_memory(body)

        assert status == 201
        assert data["importance_score"] == 1.0

    def test_importance_score_rejected_negative(self) -> None:
        body = {"content": "Test", "importance_score": -0.5}
        status, data = routes.handle_create_memory(body)

        assert status == 422


class TestHandleSearchMemories:
    def test_returns_empty_on_no_match(self) -> None:
        routes.handle_create_memory({"content": "The sky is blue."})

        status, data = routes.handle_search_memories("ocean")

        assert status == 200
        assert data["results"] == []
        assert data["total"] == 0

    def test_returns_matching_memories(self) -> None:
        routes.handle_create_memory({"content": "Python is a programming language."})
        routes.handle_create_memory({"content": "Java is also a language."})

        status, data = routes.handle_search_memories("programming")

        assert status == 200
        assert data["total"] == 1
        assert "Python" in data["results"][0]["content"]

    def test_case_insensitive_search(self) -> None:
        routes.handle_create_memory({"content": "Machine Learning is powerful."})

        status, data = routes.handle_search_memories("machine learning")

        assert status == 200
        assert data["total"] == 1

    def test_empty_query_returns_all(self) -> None:
        routes.handle_create_memory({"content": "First memory."})
        routes.handle_create_memory({"content": "Second memory."})

        status, data = routes.handle_search_memories("")

        assert status == 200
        assert data["total"] == 2

    def test_filters_by_layer(self) -> None:
        routes.handle_create_memory({"content": "Working data", "layer": "working"})
        routes.handle_create_memory({"content": "Semantic fact", "layer": "semantic"})

        status, data = routes.handle_search_memories("", layer="working")

        assert status == 200
        assert data["total"] == 1
        assert data["results"][0]["layer"] == "working"

    def test_respects_limit(self) -> None:
        for i in range(5):
            routes.handle_create_memory({"content": f"Memory entry {i}"})

        status, data = routes.handle_search_memories("Memory", limit=3)

        assert status == 200
        assert len(data["results"]) <= 3

    def test_results_sorted_by_composite_score(self) -> None:
        routes.handle_create_memory(
            {"content": "Low importance", "importance_score": 0.1}
        )
        routes.handle_create_memory(
            {"content": "High importance", "importance_score": 0.9}
        )

        status, data = routes.handle_search_memories("")

        assert status == 200
        assert data["results"][0]["importance_score"] >= data["results"][1]["importance_score"]


class TestHandleDeleteMemory:
    def test_deletes_existing_memory(self) -> None:
        _, created = routes.handle_create_memory({"content": "To be deleted."})
        memory_id = created["memory_id"]

        status, data = routes.handle_delete_memory(memory_id)

        assert status == 200
        assert data["deleted"] is True
        assert data["memory_id"] == memory_id

    def test_returns_404_for_missing_memory(self) -> None:
        status, data = routes.handle_delete_memory("nonexistent-id")

        assert status == 404
        assert "error" in data

    def test_memory_gone_after_delete(self) -> None:
        _, created = routes.handle_create_memory({"content": "Temporary."})
        memory_id = created["memory_id"]

        routes.handle_delete_memory(memory_id)

        status, data = routes.handle_get_memory(memory_id)
        assert status == 404


class TestHandleGetMemory:
    def test_returns_memory_entry(self) -> None:
        _, created = routes.handle_create_memory({"content": "Retrievable memory."})
        memory_id = created["memory_id"]

        status, data = routes.handle_get_memory(memory_id)

        assert status == 200
        assert data["memory_id"] == memory_id
        assert data["content"] == "Retrievable memory."

    def test_increments_access_count(self) -> None:
        _, created = routes.handle_create_memory({"content": "Accessed memory."})
        memory_id = created["memory_id"]

        routes.handle_get_memory(memory_id)
        _, data2 = routes.handle_get_memory(memory_id)

        assert data2["access_count"] >= 1

    def test_returns_404_for_missing(self) -> None:
        status, data = routes.handle_get_memory("no-such-id")

        assert status == 404


class TestHandleHealth:
    def test_returns_ok_status(self) -> None:
        status, data = routes.handle_health()

        assert status == 200
        assert data["status"] == "ok"
        assert data["service"] == "agent-memory"

    def test_reports_memory_count(self) -> None:
        routes.handle_create_memory({"content": "First."})
        routes.handle_create_memory({"content": "Second."})

        status, data = routes.handle_health()

        assert status == 200
        assert data["memory_count"] == 2
