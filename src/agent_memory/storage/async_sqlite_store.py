"""Async SQLite storage backend — requires aiosqlite (guarded import).

Uses the same schema and SQL as the synchronous ``SQLiteStorage`` so both
backends can operate against the same database file.

Classes
-------
- AsyncSQLiteStorage  — aiosqlite-backed async storage
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.storage.async_base import AsyncStorageBackend

_AIOSQLITE_IMPORT_ERROR = (
    "AsyncSQLiteStorage requires the 'aiosqlite' package. "
    "Install it with: pip install aiosqlite  or  pip install 'aumos-agent-memory[async]'"
)

# Re-use the same DDL and SQL constants from the sync store to keep
# schema in sync without duplicating them.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    key         TEXT NOT NULL,
    type        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.5
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories (key);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts (rowid, id, content)
    VALUES (new.rowid, new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, id, content)
    VALUES ('delete', old.rowid, old.id, old.content);
    INSERT INTO memories_fts (rowid, id, content)
    VALUES (new.rowid, new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, id, content)
    VALUES ('delete', old.rowid, old.id, old.content);
END;
"""

_INSERT_OR_REPLACE = """
INSERT OR REPLACE INTO memories
    (id, key, type, content, metadata, created_at, updated_at, importance)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_BY_ID = """
SELECT id, key, type, content, metadata, created_at, updated_at, importance
FROM memories
WHERE id = ?
"""

_DELETE_BY_ID = "DELETE FROM memories WHERE id = ?"

_LIST_KEYS_ALL = "SELECT id FROM memories ORDER BY updated_at DESC LIMIT ?"
_LIST_KEYS_LAYER = "SELECT id FROM memories WHERE type = ? ORDER BY updated_at DESC LIMIT ?"

_COUNT_ALL = "SELECT COUNT(*) FROM memories"
_COUNT_LAYER = "SELECT COUNT(*) FROM memories WHERE type = ?"

_CLEAR_ALL = "DELETE FROM memories"
_CLEAR_LAYER = "DELETE FROM memories WHERE type = ?"

_FTS_SEARCH_ALL = """
SELECT m.id, m.key, m.type, m.content, m.metadata,
       m.created_at, m.updated_at, m.importance
FROM memories_fts
JOIN memories m ON memories_fts.id = m.id
WHERE memories_fts MATCH ?
ORDER BY rank
LIMIT ?
"""

_FTS_SEARCH_LAYER = """
SELECT m.id, m.key, m.type, m.content, m.metadata,
       m.created_at, m.updated_at, m.importance
FROM memories_fts
JOIN memories m ON memories_fts.id = m.id
WHERE memories_fts MATCH ?
  AND m.type = ?
ORDER BY rank
LIMIT ?
"""

_FALLBACK_SEARCH_ALL = """
SELECT id, key, type, content, metadata, created_at, updated_at, importance
FROM memories
WHERE LOWER(content) LIKE ?
LIMIT ?
"""

_FALLBACK_SEARCH_LAYER = """
SELECT id, key, type, content, metadata, created_at, updated_at, importance
FROM memories
WHERE LOWER(content) LIKE ?
  AND type = ?
LIMIT ?
"""


class AsyncSQLiteStorage(AsyncStorageBackend):
    """Async SQLite storage backend using aiosqlite.

    Each call opens a short-lived connection via ``async with
    aiosqlite.connect()``, which is safe for concurrent callers as SQLite
    uses file-level locking in WAL mode.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to ``:memory:``.
    timeout:
        SQLite busy timeout in seconds.  Defaults to 5.0.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        timeout: float = 5.0,
    ) -> None:
        try:
            import aiosqlite as _aiosqlite  # noqa: F401
        except ImportError as exc:
            raise ImportError(_AIOSQLITE_IMPORT_ERROR) from exc

        self._db_path = str(db_path)
        self._timeout = timeout
        self._schema_initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        """Create tables once on first use."""
        if self._schema_initialised:
            return
        import aiosqlite

        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
        self._schema_initialised = True

    # ------------------------------------------------------------------
    # AsyncStorageBackend interface
    # ------------------------------------------------------------------

    async def save(self, entry: MemoryEntry) -> None:
        """Persist a memory entry, replacing any existing entry with the same ID."""
        import aiosqlite

        await self._ensure_schema()
        row = _entry_to_row(entry)
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(_INSERT_OR_REPLACE, row)
            await conn.commit()

    async def load(self, key: str) -> Optional[MemoryEntry]:
        """Load an entry by memory_id, returning None if not found."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(_SELECT_BY_ID, (key,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    async def delete(self, key: str) -> bool:
        """Delete an entry by memory_id. Returns True if it existed."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            cursor = await conn.execute(_DELETE_BY_ID, (key,))
            await conn.commit()
        return cursor.rowcount > 0

    async def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Search entries via FTS5, falling back to LIKE if FTS is unavailable."""
        import aiosqlite
        import sqlite3

        await self._ensure_schema()
        safe_query = _escape_fts_query(query)
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            conn.row_factory = aiosqlite.Row
            try:
                if layer is None:
                    async with conn.execute(_FTS_SEARCH_ALL, (safe_query, limit)) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with conn.execute(
                        _FTS_SEARCH_LAYER, (safe_query, layer.value, limit)
                    ) as cursor:
                        rows = await cursor.fetchall()
            except (aiosqlite.OperationalError, sqlite3.OperationalError):
                # FTS5 unavailable — fall back to LIKE
                like_pattern = f"%{query.lower()}%"
                if layer is None:
                    async with conn.execute(
                        _FALLBACK_SEARCH_ALL, (like_pattern, limit)
                    ) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with conn.execute(
                        _FALLBACK_SEARCH_LAYER, (like_pattern, layer.value, limit)
                    ) as cursor:
                        rows = await cursor.fetchall()
        return [_row_to_entry(row) for row in rows]

    async def list_keys(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Return up to ``limit`` stored memory_id values, newest first."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            if layer is None:
                async with conn.execute(_LIST_KEYS_ALL, (limit,)) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with conn.execute(_LIST_KEYS_LAYER, (layer.value, limit)) as cursor:
                    rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove entries. Returns number deleted."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            if layer is None:
                cursor = await conn.execute(_CLEAR_ALL)
            else:
                cursor = await conn.execute(_CLEAR_LAYER, (layer.value,))
            await conn.commit()
        return cursor.rowcount

    async def count(self, layer: Optional[MemoryLayer] = None) -> int:
        """Return the number of stored entries."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path, timeout=self._timeout) as conn:
            if layer is None:
                async with conn.execute(_COUNT_ALL) as cursor:
                    result = await cursor.fetchone()
            else:
                async with conn.execute(_COUNT_LAYER, (layer.value,)) as cursor:
                    result = await cursor.fetchone()
        return result[0] if result else 0

    def __repr__(self) -> str:
        return f"AsyncSQLiteStorage(db_path={self._db_path!r})"


# ------------------------------------------------------------------
# Row conversion helpers (identical logic to sync store)
# ------------------------------------------------------------------


def _entry_to_row(entry: MemoryEntry) -> tuple[str, str, str, str, str, str, str, float]:
    """Convert a MemoryEntry to a database row tuple."""
    return (
        entry.memory_id,
        entry.memory_id,
        entry.layer.value,
        entry.content,
        json.dumps(entry.metadata),
        entry.created_at.isoformat(),
        entry.last_accessed.isoformat(),
        entry.importance_score,
    )


def _row_to_entry(row: object) -> MemoryEntry:
    """Convert a database row to a MemoryEntry.

    Accepts both ``aiosqlite.Row`` (subscript access) and plain tuples.
    """
    # aiosqlite.Row supports both index and key access
    metadata: dict[str, str] = json.loads(row[4] or "{}")  # type: ignore[index]
    return MemoryEntry(
        memory_id=row[0],  # type: ignore[index]
        content=row[3],  # type: ignore[index]
        layer=MemoryLayer(row[2]),  # type: ignore[index]
        importance_score=float(row[7]),  # type: ignore[index]
        metadata=metadata,
        created_at=_parse_dt(row[5]),  # type: ignore[index]
        last_accessed=_parse_dt(row[6]),  # type: ignore[index]
    )


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _escape_fts_query(query: str) -> str:
    """Wrap the query in double quotes for FTS5 phrase matching."""
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


__all__ = ["AsyncSQLiteStorage"]
