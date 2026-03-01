"""SQLite-backed persistent vector store.

Uses only the Python standard library (``sqlite3``, ``json``, ``math``).
Vectors are JSON-encoded and stored as TEXT.  Similarity search loads all
vectors into memory and computes cosine similarity application-side.

This is suitable for small-to-medium workloads (up to ~100 K entries for
search, unlimited for storage).
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from agent_memory.vector.protocol import VectorStoreProtocol
from agent_memory.vector.types import VectorSearchResult

_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS vectors (
    key      TEXT PRIMARY KEY,
    vector   TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
)
"""

_COSINE_EPSILON: float = 1e-10


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns a value in ``[-1.0, 1.0]``.  Returns 0.0 if either vector is
    the zero vector.

    Parameters
    ----------
    a, b:
        Input float vectors of the same length.
    """
    dot: float = sum(x * y for x, y in zip(a, b))
    norm_a: float = math.sqrt(sum(x * x for x in a))
    norm_b: float = math.sqrt(sum(x * x for x in b))
    denom: float = norm_a * norm_b
    if denom < _COSINE_EPSILON:
        return 0.0
    return dot / denom


class SQLiteVectorStore(VectorStoreProtocol):
    """Persistent vector store backed by SQLite.

    All operations use parameterised SQL queries to prevent injection.
    Vectors and metadata are JSON-encoded for portability.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path: str = str(db_path)
        self._connection: sqlite3.Connection = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(_CREATE_TABLE_SQL)
        self._connection.commit()

    def upsert(
        self,
        key: str,
        vector: list[float],
        metadata: dict[str, object],
    ) -> None:
        """Insert or replace a vector entry.

        Parameters
        ----------
        key:
            Unique identifier for this entry.
        vector:
            Embedding as a list of floats.
        metadata:
            Arbitrary key-value metadata.  Must be JSON-serialisable.
        """
        vector_json: str = json.dumps(vector)
        metadata_json: str = json.dumps(metadata)
        self._connection.execute(
            "INSERT OR REPLACE INTO vectors (key, vector, metadata) VALUES (?, ?, ?)",
            (key, vector_json, metadata_json),
        )
        self._connection.commit()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Return the top-k most similar entries by cosine similarity.

        Loads all vectors from SQLite and computes similarity application-side.

        Parameters
        ----------
        query_vector:
            Query embedding as a list of floats.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum cosine similarity threshold (inclusive).

        Returns
        -------
        list[VectorSearchResult]
            Results sorted by cosine similarity descending.
        """
        if top_k <= 0:
            return []

        cursor = self._connection.execute(
            "SELECT key, vector, metadata FROM vectors"
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        scored: list[tuple[float, str, dict[str, object]]] = []
        for stored_key, vector_json, metadata_json in rows:
            stored_vector: list[float] = json.loads(vector_json)
            stored_metadata: dict[str, object] = json.loads(metadata_json)
            similarity: float = _cosine_similarity(query_vector, stored_vector)
            # Clamp to [0, 1] — cosine similarity can be negative for opposite vectors
            similarity = max(0.0, min(1.0, similarity))
            if similarity >= min_score:
                scored.append((similarity, stored_key, stored_metadata))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            VectorSearchResult(key=stored_key, score=score, metadata=meta)
            for score, stored_key, meta in scored[:top_k]
        ]

    def delete(self, key: str) -> None:
        """Remove a vector entry by key.

        Silently does nothing if the key does not exist.

        Parameters
        ----------
        key:
            The identifier of the entry to remove.
        """
        self._connection.execute(
            "DELETE FROM vectors WHERE key = ?", (key,)
        )
        self._connection.commit()

    def count(self) -> int:
        """Return the number of stored vector entries."""
        cursor = self._connection.execute("SELECT COUNT(*) FROM vectors")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Remove all stored vector entries."""
        self._connection.execute("DELETE FROM vectors")
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


__all__ = ["SQLiteVectorStore"]
