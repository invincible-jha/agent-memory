"""Storage backends for agent-memory.

Available backends (sync):

- ``InMemoryStorage`` — dict-based, no persistence, suitable for testing.
- ``SQLiteStorage``  — file-backed SQLite with FTS5 full-text search.
- ``RedisStorage``   — Redis hash storage with optional TTL (requires redis-py).

Available backends (async):

- ``AsyncStorageBackend``   — abstract base for all async backends.
- ``AsyncInMemoryStorage``  — dict-based async backend with asyncio.Lock.
- ``AsyncSQLiteStorage``    — aiosqlite-backed async backend (requires aiosqlite).
- ``AsyncRedisStorage``     — redis.asyncio-backed async backend (requires redis>=5).
"""

from __future__ import annotations

from agent_memory.storage.async_base import AsyncStorageBackend
from agent_memory.storage.async_memory_store import AsyncInMemoryStorage
from agent_memory.storage.base import StorageBackend
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.storage.sqlite_store import SQLiteStorage

__all__ = [
    "StorageBackend",
    "InMemoryStorage",
    "SQLiteStorage",
    "AsyncStorageBackend",
    "AsyncInMemoryStorage",
]

# RedisStorage — guarded by the redis dependency
try:
    from agent_memory.storage.redis_store import RedisStorage

    __all__ = [*__all__, "RedisStorage"]
except ImportError:
    pass

# AsyncSQLiteStorage — guarded by the aiosqlite dependency
try:
    from agent_memory.storage.async_sqlite_store import AsyncSQLiteStorage

    __all__ = [*__all__, "AsyncSQLiteStorage"]
except ImportError:
    pass

# AsyncRedisStorage — guarded by the redis[asyncio] dependency
try:
    from agent_memory.storage.async_redis_store import AsyncRedisStorage

    __all__ = [*__all__, "AsyncRedisStorage"]
except ImportError:
    pass
