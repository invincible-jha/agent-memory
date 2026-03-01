"""Tests for agent_memory.dashboard.server."""
from __future__ import annotations

import io
import json
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from agent_memory.dashboard.server import (
    DashboardDataSource,
    DashboardServer,
    _build_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> DashboardDataSource:
    source = DashboardDataSource()
    source.add_entry({"content": "The sky is blue", "layer": "semantic", "importance": 0.8})
    source.add_entry({"content": "User said hello at 9am", "layer": "episodic", "importance": 0.5})
    source.add_entry({"content": "Use retry on failure", "layer": "procedural", "importance": 0.9})
    source.add_entry({"content": "Current task: analyze report", "layer": "working", "importance": 0.6})
    source.add_graph_node({"id": "sky", "label": "Sky"})
    source.add_graph_node({"id": "blue", "label": "Blue"})
    source.add_graph_edge({"source": "sky", "target": "blue", "relation": "has_color"})
    return source


def _call_get(path: str, source: DashboardDataSource | None = None) -> bytes:
    if source is None:
        source = _make_source()
    handler_cls = _build_handler(source)
    output = io.BytesIO()
    request = MagicMock()
    srv = MagicMock()
    srv.server_address = ("127.0.0.1", 8082)
    handler = handler_cls.__new__(handler_cls)
    handler.request = request
    handler.client_address = ("127.0.0.1", 9999)
    handler.server = srv
    handler.rfile = io.BytesIO(b"")
    handler.wfile = output
    handler.path = path
    # Required by BaseHTTPRequestHandler.send_response / send_header
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.close_connection = True
    handler.do_GET()
    return output.getvalue()


def _parse_json(path: str, source: DashboardDataSource | None = None) -> dict[str, object]:
    raw = _call_get(path, source)
    body_start = raw.find(b"\r\n\r\n")
    body = raw[body_start + 4:] if body_start != -1 else raw[raw.find(b"\n\n") + 2:]
    return json.loads(body)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DashboardDataSource unit tests
# ---------------------------------------------------------------------------


class TestDashboardDataSource:
    def test_add_entry_assigns_id(self) -> None:
        source = DashboardDataSource()
        entry_id = source.add_entry({"content": "test", "layer": "semantic"})
        assert entry_id
        assert source.entry_count == 1

    def test_add_entry_default_layer(self) -> None:
        source = DashboardDataSource()
        source.add_entry({"content": "no layer"})
        memories = source.get_memories()
        assert memories[0]["layer"] == "working"

    def test_get_memories_all_layers(self) -> None:
        source = _make_source()
        memories = source.get_memories()
        assert len(memories) == 4

    def test_get_memories_filter_by_layer(self) -> None:
        source = _make_source()
        semantic = source.get_memories(layer="semantic")
        assert len(semantic) == 1
        assert all(m["layer"] == "semantic" for m in semantic)

    def test_get_memories_invalid_layer_returns_all(self) -> None:
        source = _make_source()
        # Invalid layer name is not in _VALID_LAYERS so filter is skipped
        memories = source.get_memories(layer="invalid-layer")
        assert len(memories) == 4  # all entries returned (filter ignored)

    def test_search_memories_finds_match(self) -> None:
        source = _make_source()
        results = source.search_memories("sky")
        assert len(results) == 1
        assert "sky" in str(results[0]["content"]).lower()

    def test_search_memories_case_insensitive(self) -> None:
        source = _make_source()
        results = source.search_memories("BLUE")
        assert len(results) >= 1

    def test_search_memories_no_match(self) -> None:
        source = _make_source()
        results = source.search_memories("zzz-no-match-xyz")
        assert results == []

    def test_get_stats_structure(self) -> None:
        source = _make_source()
        stats = source.get_stats()
        assert "total" in stats
        assert "by_layer" in stats
        assert stats["total"] == 4
        assert stats["by_layer"]["semantic"] == 1
        assert stats["graph_nodes"] == 2
        assert stats["graph_edges"] == 1

    def test_get_graph_structure(self) -> None:
        source = _make_source()
        graph = source.get_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1

    def test_max_entries_eviction(self) -> None:
        source = DashboardDataSource(max_entries=3)
        for i in range(5):
            source.add_entry({"content": f"entry {i}"})
        assert source.entry_count == 3


# ---------------------------------------------------------------------------
# HTTP handler — static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_root_returns_html(self) -> None:
        raw = _call_get("/")
        assert b"html" in raw.lower()

    def test_styles_css_served(self) -> None:
        raw = _call_get("/styles.css")
        assert b"text/css" in raw or b"--bg" in raw

    def test_app_js_served(self) -> None:
        raw = _call_get("/app.js")
        assert b"javascript" in raw or b"function" in raw


# ---------------------------------------------------------------------------
# HTTP handler — health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self) -> None:
        data = _parse_json("/health")
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        data = _parse_json("/health")
        assert data["service"] == "agent-memory-dashboard"

    def test_health_includes_count(self) -> None:
        data = _parse_json("/health")
        assert "entries" in data
        assert data["entries"] == 4


# ---------------------------------------------------------------------------
# HTTP handler — /api/memories
# ---------------------------------------------------------------------------


class TestMemoriesEndpoint:
    def test_memories_returns_list(self) -> None:
        data = _parse_json("/api/memories")
        assert "memories" in data
        assert data["count"] == 4

    def test_memories_filter_by_layer(self) -> None:
        data = _parse_json("/api/memories?layer=semantic")
        assert data["count"] == 1
        assert all(m["layer"] == "semantic" for m in data["memories"])

    def test_memories_total_field(self) -> None:
        data = _parse_json("/api/memories?layer=episodic")
        assert "total" in data
        assert data["total"] == 4  # total unfiltered


# ---------------------------------------------------------------------------
# HTTP handler — /api/stats
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_total(self) -> None:
        data = _parse_json("/api/stats")
        assert data["total"] == 4

    def test_stats_by_layer(self) -> None:
        data = _parse_json("/api/stats")
        assert "by_layer" in data
        by_layer = data["by_layer"]
        assert by_layer["semantic"] == 1
        assert by_layer["episodic"] == 1
        assert by_layer["working"] == 1
        assert by_layer["procedural"] == 1

    def test_stats_graph_counts(self) -> None:
        data = _parse_json("/api/stats")
        assert data["graph_nodes"] == 2
        assert data["graph_edges"] == 1


# ---------------------------------------------------------------------------
# HTTP handler — /api/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_search_returns_results(self) -> None:
        data = _parse_json("/api/search?q=sky")
        assert "results" in data
        assert data["count"] == 1

    def test_search_empty_query_returns_400(self) -> None:
        raw = _call_get("/api/search?q=")
        assert b"400" in raw

    def test_search_missing_query_returns_400(self) -> None:
        raw = _call_get("/api/search")
        assert b"400" in raw

    def test_search_no_match_returns_empty(self) -> None:
        data = _parse_json("/api/search?q=zzz-no-match")
        assert data["count"] == 0
        assert data["results"] == []


# ---------------------------------------------------------------------------
# HTTP handler — /api/graph
# ---------------------------------------------------------------------------


class TestGraphEndpoint:
    def test_graph_returns_nodes_and_edges(self) -> None:
        data = _parse_json("/api/graph")
        assert "nodes" in data
        assert "edges" in data

    def test_graph_node_count(self) -> None:
        data = _parse_json("/api/graph")
        assert len(data["nodes"]) == 2


# ---------------------------------------------------------------------------
# 404
# ---------------------------------------------------------------------------


class TestNotFound:
    def test_unknown_path_404(self) -> None:
        raw = _call_get("/api/unknown")
        assert b"404" in raw


# ---------------------------------------------------------------------------
# DashboardServer
# ---------------------------------------------------------------------------


class TestDashboardServer:
    def test_instantiation(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        assert server.address == "127.0.0.1:8082"

    def test_build_server_returns_http_server(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source, port=0)
        http_server = server.build_server()
        try:
            assert isinstance(http_server, HTTPServer)
        finally:
            http_server.server_close()

    def test_shutdown_noop_when_not_started(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        server.shutdown()  # must not raise
