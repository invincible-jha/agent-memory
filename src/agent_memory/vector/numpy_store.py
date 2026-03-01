"""In-memory vector store backed by NumPy arrays.

Requires numpy::

    pip install aumos-agent-memory[vector]

Thread-safe: all mutating operations acquire a threading.Lock.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from agent_memory.vector.protocol import VectorStoreProtocol
from agent_memory.vector.types import VectorSearchResult

if TYPE_CHECKING:
    import numpy as np

_COSINE_EPSILON: float = 1e-10


def _require_numpy() -> object:
    try:
        import numpy  # type: ignore[import-untyped]

        return numpy
    except ImportError as exc:
        raise ImportError(
            "The 'numpy' package is required for NumpyVectorStore.  Install it with:\n\n"
            "    pip install aumos-agent-memory[vector]\n"
            "  or\n"
            "    pip install numpy"
        ) from exc


class NumpyVectorStore(VectorStoreProtocol):
    """Thread-safe in-memory vector store using NumPy for cosine similarity.

    Vectors are stored as NumPy ``ndarray`` objects keyed by a string
    identifier.  Similarity search performs a brute-force cosine similarity
    scan over all stored vectors.  This is suitable for up to tens of
    thousands of entries; use a dedicated ANN library for larger workloads.

    Parameters
    ----------
    dimension:
        Expected vector dimension.  If provided, mismatched vectors raise
        ``ValueError`` on :meth:`upsert`.  Pass ``None`` to infer from the
        first inserted vector.

    Raises
    ------
    ImportError
        If ``numpy`` is not installed.
    """

    def __init__(self, dimension: int | None = None) -> None:
        _require_numpy()
        self._dimension: int | None = dimension
        # key -> (vector_ndarray, metadata)
        self._store: dict[str, tuple[object, dict[str, object]]] = {}
        self._lock: threading.Lock = threading.Lock()

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
            Arbitrary key-value metadata.

        Raises
        ------
        ValueError
            If the vector length does not match the configured dimension.
        """
        import numpy as np  # type: ignore[import-untyped]

        array: np.ndarray = np.array(vector, dtype=np.float64)

        with self._lock:
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dimension}, "
                    f"got {len(vector)}"
                )
            self._store[key] = (array, dict(metadata))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Return the top-k most similar entries by cosine similarity.

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
        import numpy as np  # type: ignore[import-untyped]

        if top_k <= 0:
            return []

        query_array: np.ndarray = np.array(query_vector, dtype=np.float64)
        query_norm: float = float(np.linalg.norm(query_array))

        with self._lock:
            items = list(self._store.items())

        if not items:
            return []

        scored: list[tuple[float, str, dict[str, object]]] = []
        for stored_key, (stored_vector, stored_metadata) in items:
            vec: np.ndarray = stored_vector  # type: ignore[assignment]
            vec_norm: float = float(np.linalg.norm(vec))
            denom: float = query_norm * vec_norm
            if denom < _COSINE_EPSILON:
                similarity: float = 0.0
            else:
                similarity = float(np.dot(query_array, vec) / denom)
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
        with self._lock:
            self._store.pop(key, None)

    def count(self) -> int:
        """Return the number of stored vector entries."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all stored vector entries."""
        with self._lock:
            self._store.clear()


__all__ = ["NumpyVectorStore"]
