"""HTTP dashboard server for agent-memory.

Serves a single-page web dashboard (memory browser, knowledge graph,
stats, search) using only the Python standard library.  No external
web frameworks are required.

Usage
-----
::

    from agent_memory.dashboard.server import DashboardServer, DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8082)
    server.start()  # blocks; Ctrl-C to stop
"""
from __future__ import annotations

import json
import mimetypes
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_STATIC_DIR = Path(__file__).parent / "static"

_VALID_LAYERS = {"working", "episodic", "semantic", "procedural"}

# ---------------------------------------------------------------------------
# In-memory data store
# ---------------------------------------------------------------------------


class DashboardDataSource:
    """Thread-safe in-memory store for memory entries and graph edges.

    Parameters
    ----------
    max_entries:
        Maximum number of memory records to retain.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self._max_entries = max_entries
        self._entries: list[dict[str, object]] = []
        self._graph_nodes: dict[str, dict[str, object]] = {}
        self._graph_edges: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_entry(self, entry: dict[str, object]) -> str:
        """Store a memory entry, returning its assigned ID."""
        entry_id = str(entry.get("id") or uuid.uuid4())
        record = dict(entry)
        record["id"] = entry_id
        record.setdefault("layer", "working")
        record.setdefault("timestamp", time.time())
        record.setdefault("importance", 0.5)
        self._entries.append(record)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        return entry_id

    def add_graph_node(self, node: dict[str, object]) -> None:
        """Add or update a knowledge graph node."""
        node_id = str(node.get("id") or uuid.uuid4())
        self._graph_nodes[node_id] = {"id": node_id, **node}

    def add_graph_edge(self, edge: dict[str, object]) -> None:
        """Add a directed edge between two graph nodes."""
        self._graph_edges.append(dict(edge))

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_memories(
        self,
        layer: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        """Return memory entries, optionally filtered by layer."""
        entries = self._entries
        if layer and layer in _VALID_LAYERS:
            entries = [e for e in entries if e.get("layer") == layer]
        sliced = entries[offset: offset + limit]
        return sliced

    def search_memories(self, query: str, limit: int = 50) -> list[dict[str, object]]:
        """Return entries whose content contains *query* (case-insensitive)."""
        query_lower = query.lower()
        results: list[dict[str, object]] = []
        for entry in reversed(self._entries):
            content = str(entry.get("content") or "").lower()
            if query_lower in content:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def get_stats(self) -> dict[str, object]:
        """Return memory statistics including per-layer counts."""
        layer_counts: dict[str, int] = {layer: 0 for layer in _VALID_LAYERS}
        for entry in self._entries:
            layer = str(entry.get("layer") or "working")
            if layer in layer_counts:
                layer_counts[layer] += 1

        importances = [float(e.get("importance") or 0.5) for e in self._entries]
        avg_importance = (
            round(sum(importances) / len(importances), 3) if importances else 0.0
        )

        return {
            "total": len(self._entries),
            "by_layer": layer_counts,
            "graph_nodes": len(self._graph_nodes),
            "graph_edges": len(self._graph_edges),
            "avg_importance": avg_importance,
        }

    def get_graph(self) -> dict[str, object]:
        """Return the knowledge graph as nodes and edges."""
        return {
            "nodes": list(self._graph_nodes.values()),
            "edges": self._graph_edges,
        }

    @property
    def entry_count(self) -> int:
        """Total number of stored memory entries."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _build_handler(data_source: DashboardDataSource) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP request handler bound to *data_source*."""

    class _Handler(BaseHTTPRequestHandler):
        _source = data_source

        def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
            pass

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            params = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/app.js":
                self._serve_static("app.js")
            elif path == "/styles.css":
                self._serve_static("styles.css")
            elif path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "service": "agent-memory-dashboard",
                    "entries": self._source.entry_count,
                })
            elif path == "/api/memories":
                layer = (params.get("layer") or [None])[0]
                limit = int((params.get("limit") or ["200"])[0])
                offset = int((params.get("offset") or ["0"])[0])
                memories = self._source.get_memories(layer=layer, limit=limit, offset=offset)
                self._send_json(200, {
                    "memories": memories,
                    "count": len(memories),
                    "total": self._source.entry_count,
                })
            elif path == "/api/stats":
                self._send_json(200, self._source.get_stats())
            elif path == "/api/search":
                query = (params.get("q") or [""])[0]
                limit = int((params.get("limit") or ["50"])[0])
                if not query:
                    self._send_json(400, {"error": "q parameter is required"})
                    return
                results = self._source.search_memories(query=query, limit=limit)
                self._send_json(200, {"results": results, "count": len(results), "query": query})
            elif path == "/api/graph":
                self._send_json(200, self._source.get_graph())
            else:
                self._send_json(404, {"error": "Not found", "path": path})

        def _serve_static(self, filename: str) -> None:
            file_path = _STATIC_DIR / filename
            if not file_path.exists():
                self._send_json(404, {"error": f"Static file not found: {filename}"})
                return
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, data: dict[str, object]) -> None:
            body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


# ---------------------------------------------------------------------------
# Server wrapper
# ---------------------------------------------------------------------------


class DashboardServer:
    """Agent-memory web dashboard server.

    Parameters
    ----------
    data_source:
        The data source to serve dashboard data from.
    host:
        Bind host (default ``"127.0.0.1"``).
    port:
        Bind port (default ``8082``).
    """

    def __init__(
        self,
        data_source: DashboardDataSource,
        host: str = "127.0.0.1",
        port: int = 8082,
    ) -> None:
        self._data_source = data_source
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None

    def build_server(self) -> HTTPServer:
        """Build and return the underlying ``HTTPServer`` without starting it."""
        handler_cls = _build_handler(self._data_source)
        server = HTTPServer((self._host, self._port), handler_cls)
        self._server = server
        return server

    def start(self) -> None:
        """Start the HTTP server and block until interrupted."""
        server = self.build_server()
        try:
            server.serve_forever()
        finally:
            server.server_close()

    def shutdown(self) -> None:
        """Stop the server if it is running."""
        if self._server is not None:
            self._server.shutdown()

    @property
    def address(self) -> str:
        """Return the server's bind address as ``host:port``."""
        return f"{self._host}:{self._port}"


__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
