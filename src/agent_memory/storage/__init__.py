"""Storage backends for agent-memory.

Available backends:

- ``InMemoryStorage`` — dict-based, no persistence, suitable for testing.
- ``SQLiteStorage``  — file-backed SQLite with FTS5 full-text search.
- ``RedisStorage``   — Redis hash storage with optional TTL (requires redis-py).
"""

from __future__ import annotations

from agent_memory.storage.base import StorageBackend
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.storage.sqlite_store import SQLiteStorage

__all__ = [
    "StorageBackend",
    "InMemoryStorage",
    "SQLiteStorage",
]

# RedisStorage is available but its import is guarded by the redis dependency
try:
    from agent_memory.storage.redis_store import RedisStorage

    __all__ = [*__all__, "RedisStorage"]
except ImportError:
    pass
