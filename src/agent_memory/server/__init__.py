"""HTTP server mode for agent-memory.

Provides a lightweight stdlib-based HTTP API for memory storage and
retrieval without requiring any additional web framework dependencies.
"""
from __future__ import annotations

from agent_memory.server.app import AgentMemoryHandler, create_server, run_server

__all__ = ["AgentMemoryHandler", "create_server", "run_server"]
