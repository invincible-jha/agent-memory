"""HTTP server for agent-memory using stdlib http.server.

Routes:
    POST   /memories           — store a new memory entry
    GET    /search?q=...       — search memories by keyword
    GET    /memories/{id}      — retrieve a specific memory
    DELETE /memories/{id}      — delete a specific memory
    GET    /health             — health check

Usage:
    python -m agent_memory.server.app --port 8080
    python -m agent_memory.server.app --host 127.0.0.1 --port 9000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent_memory.server import routes

logger = logging.getLogger(__name__)

# URL pattern for /memories/{id}
_MEMORY_ID_PATTERN = re.compile(r"^/memories/([^/]+)$")


class AgentMemoryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the agent-memory server.

    Implements routing for GET, POST, and DELETE methods across all
    supported endpoints. All request bodies and responses use JSON.
    """

    def log_message(self, format: str, *args: object) -> None:
        """Override to route access logs through the Python logging system."""
        logger.debug(format, *args)

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        """Handle all GET requests by routing on the URL path."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        query_params = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            status, data = routes.handle_health()
            self._send_json(status, data)

        elif path == "/search":
            query = self._first_param(query_params, "q") or ""
            layer = self._first_param(query_params, "layer")
            limit_str = self._first_param(query_params, "limit")
            limit = int(limit_str) if limit_str and limit_str.isdigit() else 20
            status, data = routes.handle_search_memories(
                query=query, layer=layer, limit=limit
            )
            self._send_json(status, data)

        else:
            match = _MEMORY_ID_PATTERN.match(path)
            if match:
                memory_id = match.group(1)
                status, data = routes.handle_get_memory(memory_id)
                self._send_json(status, data)
            else:
                self._send_json(
                    404,
                    {"error": "Not found", "detail": f"No route for GET {path}"},
                )

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        """Handle all POST requests by routing on the URL path."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        body = self._read_json_body()
        if body is None:
            return

        if path == "/memories":
            status, data = routes.handle_create_memory(body)
            self._send_json(status, data)
        else:
            self._send_json(
                404, {"error": "Not found", "detail": f"No route for POST {path}"}
            )

    # ── DELETE ────────────────────────────────────────────────────────────────

    def do_DELETE(self) -> None:
        """Handle all DELETE requests by routing on the URL path."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        match = _MEMORY_ID_PATTERN.match(path)
        if match:
            memory_id = match.group(1)
            status, data = routes.handle_delete_memory(memory_id)
            self._send_json(status, data)
        else:
            self._send_json(
                404,
                {"error": "Not found", "detail": f"No route for DELETE {path}"},
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_json(self, status: int, data: dict[str, object]) -> None:
        """Serialize *data* to JSON and send an HTTP response with *status*."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, object] | None:
        """Read and parse the JSON request body.

        Returns None (and sends a 400 error response) if parsing fails.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}

        raw = self.rfile.read(content_length)
        try:
            parsed: dict[str, object] = json.loads(raw.decode("utf-8"))
            return parsed
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": "Invalid JSON", "detail": str(exc)})
            return None

    @staticmethod
    def _first_param(
        params: dict[str, list[str]], key: str
    ) -> str | None:
        """Return the first value for *key* from query parameters, or None."""
        values = params.get(key)
        return values[0] if values else None


def create_server(host: str = "0.0.0.0", port: int = 8080) -> HTTPServer:
    """Create (but do not start) the agent-memory HTTP server.

    Parameters
    ----------
    host:
        Bind address (default ``"0.0.0.0"`` — all interfaces).
    port:
        TCP port to listen on (default 8080).

    Returns
    -------
    HTTPServer
        A configured server instance ready to call ``serve_forever()`` on.
    """
    server = HTTPServer((host, port), AgentMemoryHandler)
    logger.info("agent-memory server created at http://%s:%d", host, port)
    return server


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Create and run the agent-memory HTTP server (blocking).

    Parameters
    ----------
    host:
        Bind address.
    port:
        TCP port.
    """
    server = create_server(host=host, port=port)
    logger.info("Serving agent-memory on http://%s:%d — press Ctrl-C to stop", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down agent-memory server.")
    finally:
        server.server_close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="agent-memory HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="TCP port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_server(host=args.host, port=args.port)
