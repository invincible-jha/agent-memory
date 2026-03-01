"""SentenceTransformer-based embedder implementation.

Requires the ``sentence-transformers`` package::

    pip install aumos-agent-memory[vector]
"""

from __future__ import annotations

from agent_memory.vector.protocol import EmbedderProtocol

_DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
_DEFAULT_DIMENSION: int = 384


class SentenceTransformerEmbedder(EmbedderProtocol):
    """Embedder backed by a HuggingFace Sentence Transformer model.

    The default model is ``all-MiniLM-L6-v2`` (Apache 2.0 licence, 384
    dimensions).  Any model compatible with the ``sentence-transformers``
    library may be used by passing ``model_name``.

    Parameters
    ----------
    model_name:
        The Sentence Transformers model identifier.  Defaults to
        ``"all-MiniLM-L6-v2"``.

    Raises
    ------
    ImportError
        If the ``sentence-transformers`` package is not installed.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required for "
                "SentenceTransformerEmbedder.  Install it with:\n\n"
                "    pip install aumos-agent-memory[vector]\n"
                "  or\n"
                "    pip install sentence-transformers"
            ) from exc

        self._model_name: str = model_name
        self._model: object = SentenceTransformer(model_name)
        # Derive dimension from a probe embedding
        probe: list[float] = self._encode_single("probe")
        self._dimension: int = len(probe)

    def embed(self, text: str) -> list[float]:
        """Embed a single string into a float vector.

        Parameters
        ----------
        text:
            Input string to embed.

        Returns
        -------
        list[float]
            Float vector of length :py:attr:`dimension`.
        """
        return self._encode_single(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings into float vectors.

        Parameters
        ----------
        texts:
            Input strings to embed.

        Returns
        -------
        list[list[float]]
            One float vector per input string, in order.
        """
        if not texts:
            return []
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        embeddings: object = model.encode(texts, convert_to_numpy=True)
        return [list(map(float, row)) for row in embeddings]  # type: ignore[arg-type]

    @property
    def dimension(self) -> int:
        """The fixed length of every vector produced by this embedder."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """The Sentence Transformers model identifier in use."""
        return self._model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_single(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        model: SentenceTransformer = self._model  # type: ignore[assignment]
        result: object = model.encode([text], convert_to_numpy=True)
        return list(map(float, result[0]))  # type: ignore[index]


__all__ = ["SentenceTransformerEmbedder"]
