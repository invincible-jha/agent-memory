"""agent-memory web dashboard subpackage.

Provides a self-contained HTTP dashboard for browsing memory entries,
visualizing knowledge graph relationships, and searching across all
memory layers.  Requires no external dependencies beyond the Python
standard library.

Usage
-----
::

    from agent_memory.dashboard import DashboardServer
    from agent_memory.dashboard.server import DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8082)
    server.start()
"""
from __future__ import annotations

from agent_memory.dashboard.server import DashboardServer, DashboardDataSource

__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
