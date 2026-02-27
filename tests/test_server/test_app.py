"""Tests for agent_memory.server.app — HTTP handler integration."""
from __future__ import annotations

from http.server import HTTPServer

import pytest

from agent_memory.server import routes
from agent_memory.server.app import AgentMemoryHandler, create_server


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestAgentMemoryHandlerHealth:
    def test_health_endpoint_returns_200(self) -> None:
        status, data = routes.handle_health()
        assert status == 200
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        status, data = routes.handle_health()
        assert data["service"] == "agent-memory"

    def test_health_reports_zero_count_initially(self) -> None:
        status, data = routes.handle_health()
        assert data["memory_count"] == 0


class TestAgentMemoryHandlerCreateMemory:
    def test_create_memory_returns_201(self) -> None:
        body = {"content": "Agent started task execution."}
        status, data = routes.handle_create_memory(body)
        assert status == 201
        assert "memory_id" in data

    def test_create_memory_default_layer(self) -> None:
        body = {"content": "Some content"}
        status, data = routes.handle_create_memory(body)
        assert status == 201
        assert data["layer"] == "episodic"

    def test_create_memory_all_layers(self) -> None:
        for layer in ["working", "episodic", "semantic", "procedural"]:
            body = {"content": f"Content for {layer}", "layer": layer}
            status, data = routes.handle_create_memory(body)
            assert status == 201
            assert data["layer"] == layer


class TestAgentMemoryHandlerSearch:
    def test_search_empty_results(self) -> None:
        status, data = routes.handle_search_memories("nothing")
        assert status == 200
        assert data["total"] == 0

    def test_search_finds_content(self) -> None:
        routes.handle_create_memory({"content": "Neural networks are powerful."})
        status, data = routes.handle_search_memories("neural")
        assert status == 200
        assert data["total"] == 1

    def test_search_multiple_results(self) -> None:
        routes.handle_create_memory({"content": "First memory about AI"})
        routes.handle_create_memory({"content": "Second memory about AI"})
        status, data = routes.handle_search_memories("AI")
        assert status == 200
        assert data["total"] == 2

    def test_search_query_preserved(self) -> None:
        status, data = routes.handle_search_memories("test-query")
        assert data["query"] == "test-query"


class TestAgentMemoryHandlerDelete:
    def test_delete_existing_memory(self) -> None:
        _, created = routes.handle_create_memory({"content": "Delete me."})
        memory_id = created["memory_id"]

        status, data = routes.handle_delete_memory(memory_id)
        assert status == 200
        assert data["deleted"] is True

    def test_delete_nonexistent_returns_404(self) -> None:
        status, data = routes.handle_delete_memory("fake-id-xyz")
        assert status == 404

    def test_deleted_memory_not_retrievable(self) -> None:
        _, created = routes.handle_create_memory({"content": "Gone soon."})
        memory_id = created["memory_id"]

        routes.handle_delete_memory(memory_id)
        status2, _ = routes.handle_get_memory(memory_id)
        assert status2 == 404


class TestAgentMemoryHandlerGetMemory:
    def test_get_existing_memory(self) -> None:
        _, created = routes.handle_create_memory({"content": "Retrieve this."})
        memory_id = created["memory_id"]

        status, data = routes.handle_get_memory(memory_id)
        assert status == 200
        assert data["content"] == "Retrieve this."

    def test_get_nonexistent_memory(self) -> None:
        status, data = routes.handle_get_memory("not-a-real-id")
        assert status == 404


class TestCreateServer:
    def test_create_server_returns_http_server(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert isinstance(server, HTTPServer)
        finally:
            server.server_close()

    def test_create_server_uses_correct_handler(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert server.RequestHandlerClass is AgentMemoryHandler
        finally:
            server.server_close()


class TestServerModels:
    def test_create_memory_request_required_fields(self) -> None:
        from agent_memory.server.models import CreateMemoryRequest

        req = CreateMemoryRequest(content="test content")
        assert req.layer == "episodic"
        assert req.importance_score == 0.5

    def test_memory_response_fields(self) -> None:
        from agent_memory.server.models import MemoryResponse

        resp = MemoryResponse(
            memory_id="m1",
            content="test",
            layer="episodic",
            importance_score=0.5,
            freshness_score=1.0,
            source="agent_inference",
            safety_critical=False,
            composite_score=0.5,
            importance_level="medium",
            created_at="2024-01-01T00:00:00",
            last_accessed="2024-01-01T00:00:00",
            access_count=0,
        )
        assert resp.memory_id == "m1"
        assert resp.content == "test"

    def test_search_response_fields(self) -> None:
        from agent_memory.server.models import SearchResponse

        resp = SearchResponse(query="test", results=[], total=0)
        assert resp.query == "test"
        assert resp.total == 0

    def test_health_response_defaults(self) -> None:
        from agent_memory.server.models import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.service == "agent-memory"
