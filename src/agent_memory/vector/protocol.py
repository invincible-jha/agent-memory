"""Abstract base classes (protocols) for embedders and vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_memory.vector.types import VectorSearchResult


class EmbedderProtocol(ABC):
    """Contract for text-to-vector embedding implementations.

    All methods must be implemented by concrete embedder classes.
    Implementations should be stateless with respect to the input text;
    the same text must always produce the same vector.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text into a float vector.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            A fixed-length float vector of length :py:attr:`dimension`.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single call.

        Parameters
        ----------
        texts:
            A list of input strings.

        Returns
        -------
        list[list[float]]
            One vector per input text, in the same order, each of length
            :py:attr:`dimension`.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The fixed length of every vector produced by this embedder."""
        ...


class VectorStoreProtocol(ABC):
    """Contract for vector storage and similarity search implementations.

    Implementations must be safe for single-threaded use.  Thread-safety
    is optional and must be documented explicitly.
    """

    @abstractmethod
    def upsert(
        self,
        key: str,
        vector: list[float],
        metadata: dict[str, object],
    ) -> None:
        """Insert or update a vector entry.

        Parameters
        ----------
        key:
            Unique identifier (``memory_id``).  An existing entry with this
            key is silently replaced.
        vector:
            The embedding to store.
        metadata:
            Arbitrary metadata to associate with this entry.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for the most similar vectors.

        Parameters
        ----------
        query_vector:
            The query embedding.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum cosine similarity threshold; results below this score
            are excluded.

        Returns
        -------
        list[VectorSearchResult]
            Results sorted by score descending, up to *top_k* items.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a vector entry by key.

        Silently does nothing if the key does not exist.

        Parameters
        ----------
        key:
            The identifier of the entry to remove.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vector entries.

        Returns
        -------
        int
            Number of stored entries.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored vector entries."""
        ...


__all__ = ["EmbedderProtocol", "VectorStoreProtocol"]
