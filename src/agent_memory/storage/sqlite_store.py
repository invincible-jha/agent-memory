"""SQLite storage backend with FTS5 full-text search."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.storage.base import StorageBackend


# DDL executed once on first connection
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


class SQLiteStorage(StorageBackend):
    """Persistent SQLite storage backend with FTS5 full-text search.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to ``:memory:`` for an
        in-memory database (useful for testing; data is lost when the object
        is garbage-collected).
    timeout:
        SQLite busy timeout in seconds. Defaults to 5.0.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        timeout: float = 5.0,
    ) -> None:
        self._db_path = str(db_path)
        self._timeout = timeout
        self._conn = self._connect()
        self._initialise_schema()

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> None:
        """Persist a memory entry, replacing any existing entry with the same ID."""
        row = _entry_to_row(entry)
        with self._conn:
            self._conn.execute(_INSERT_OR_REPLACE, row)

    def load(self, key: str) -> Optional[MemoryEntry]:
        """Load an entry by memory_id, returning None if not found."""
        cursor = self._conn.execute(_SELECT_BY_ID, (key,))
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def delete(self, key: str) -> bool:
        """Delete an entry by memory_id. Returns True if it existed."""
        with self._conn:
            cursor = self._conn.execute(_DELETE_BY_ID, (key,))
        return cursor.rowcount > 0

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Search entries via FTS5, falling back to LIKE if FTS is unavailable."""
        try:
            return self._fts_search(query, layer=layer, limit=limit)
        except sqlite3.OperationalError:
            return self._fallback_search(query, layer=layer, limit=limit)

    def list_keys(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Return up to ``limit`` stored memory_id values, newest first."""
        if layer is None:
            cursor = self._conn.execute(_LIST_KEYS_ALL, (limit,))
        else:
            cursor = self._conn.execute(_LIST_KEYS_LAYER, (layer.value, limit))
        return [row[0] for row in cursor.fetchall()]

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove entries. Returns number deleted."""
        with self._conn:
            if layer is None:
                cursor = self._conn.execute(_CLEAR_ALL)
            else:
                cursor = self._conn.execute(_CLEAR_LAYER, (layer.value,))
        return cursor.rowcount

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        """Return the number of stored entries."""
        if layer is None:
            cursor = self._conn.execute(_COUNT_ALL)
        else:
            cursor = self._conn.execute(_COUNT_LAYER, (layer.value,))
        result = cursor.fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> SQLiteStorage:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=self._timeout)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent reads (no-op for :memory:)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialise_schema(self) -> None:
        """Create tables and triggers on first use."""
        with self._conn:
            self._conn.executescript(_SCHEMA_SQL)

    def _fts_search(
        self,
        query: str,
        layer: Optional[MemoryLayer],
        limit: int,
    ) -> list[MemoryEntry]:
        # Escape special FTS5 characters to avoid syntax errors
        safe_query = _escape_fts_query(query)
        if layer is None:
            cursor = self._conn.execute(_FTS_SEARCH_ALL, (safe_query, limit))
        else:
            cursor = self._conn.execute(_FTS_SEARCH_LAYER, (safe_query, layer.value, limit))
        return [_row_to_entry(row) for row in cursor.fetchall()]

    def _fallback_search(
        self,
        query: str,
        layer: Optional[MemoryLayer],
        limit: int,
    ) -> list[MemoryEntry]:
        like_pattern = f"%{query.lower()}%"
        if layer is None:
            cursor = self._conn.execute(_FALLBACK_SEARCH_ALL, (like_pattern, limit))
        else:
            cursor = self._conn.execute(
                _FALLBACK_SEARCH_LAYER, (like_pattern, layer.value, limit)
            )
        return [_row_to_entry(row) for row in cursor.fetchall()]


# ------------------------------------------------------------------
# Row conversion helpers
# ------------------------------------------------------------------


def _entry_to_row(entry: MemoryEntry) -> tuple[str, str, str, str, str, str, str, float]:
    """Convert a MemoryEntry to a database row tuple."""
    return (
        entry.memory_id,                         # id
        entry.memory_id,                         # key (same as id for now)
        entry.layer.value,                       # type
        entry.content,                           # content
        json.dumps(entry.metadata),              # metadata
        entry.created_at.isoformat(),            # created_at
        entry.last_accessed.isoformat(),         # updated_at
        entry.importance_score,                  # importance
    )


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a database row to a MemoryEntry."""
    metadata: dict[str, str] = json.loads(row["metadata"] or "{}")
    # Rebuild a MemoryEntry from stored fields; non-persisted fields use defaults
    return MemoryEntry(
        memory_id=row["id"],
        content=row["content"],
        layer=MemoryLayer(row["type"]),
        importance_score=row["importance"],
        metadata=metadata,
        created_at=_parse_dt(row["created_at"]),
        last_accessed=_parse_dt(row["updated_at"]),
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


__all__ = ["SQLiteStorage"]
