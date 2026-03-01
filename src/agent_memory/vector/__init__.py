"""Vector search subpackage for agent-memory.

Provides protocol definitions, a pure-Python in-memory vector store backed
by NumPy, and a SQLite-based persistent vector store.  An optional
SentenceTransformer embedder is available when ``sentence-transformers`` is
installed.

Optional extras required
------------------------
* ``numpy`` — for :class:`NumpyVectorStore`
* ``sentence-transformers`` — for :class:`SentenceTransformerEmbedder`

Both are installed with::

    pip install aumos-agent-memory[vector]
"""

from __future__ import annotations

from agent_memory.vector.protocol import EmbedderProtocol, VectorStoreProtocol
from agent_memory.vector.sqlite_vector_store import SQLiteVectorStore
from agent_memory.vector.types import VectorEntry, VectorSearchResult

__all__ = [
    "EmbedderProtocol",
    "VectorStoreProtocol",
    "VectorEntry",
    "VectorSearchResult",
    "SQLiteVectorStore",
]

# Guarded import — requires numpy extra
try:
    from agent_memory.vector.numpy_store import NumpyVectorStore as NumpyVectorStore

    __all__ += ["NumpyVectorStore"]
except ImportError:
    pass

# Guarded import — requires sentence-transformers extra
try:
    from agent_memory.vector.embedder import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )

    __all__ += ["SentenceTransformerEmbedder"]
except ImportError:
    pass
