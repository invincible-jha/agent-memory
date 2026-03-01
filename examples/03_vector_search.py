#!/usr/bin/env python3
"""Example: Vector Search

Demonstrates similarity-based memory retrieval using the SQLiteVectorStore
and EmbedderProtocol with a simple keyword-hash embedder.

Usage:
    python examples/03_vector_search.py

Requirements:
    pip install agent-memory
"""
from __future__ import annotations

import hashlib
import math

import agent_memory
from agent_memory import (
    EmbedderProtocol,
    SQLiteVectorStore,
    VectorEntry,
    VectorSearchResult,
)
import tempfile
import os


class KeywordHashEmbedder:
    """Simple deterministic embedder for demo purposes — not for production use."""

    DIMENSION: int = 8

    def embed(self, text: str) -> list[float]:
        """Generate a pseudo-embedding from keyword frequency."""
        words = text.lower().split()
        vector = [0.0] * self.DIMENSION
        for word in words:
            digest = hashlib.md5(word.encode()).digest()
            for i in range(self.DIMENSION):
                vector[i] += (digest[i] - 128) / 128.0
        magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / magnitude for v in vector]


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    # Step 1: Create SQLite vector store
    temp_dir = tempfile.mkdtemp(prefix="agent_vector_")
    db_path = os.path.join(temp_dir, "vectors.db")
    store = SQLiteVectorStore(db_path=db_path, dimension=KeywordHashEmbedder.DIMENSION)
    embedder = KeywordHashEmbedder()
    print(f"Vector store: {db_path} (dim={KeywordHashEmbedder.DIMENSION})")

    # Step 2: Add documents to the vector store
    documents: list[tuple[str, str]] = [
        ("doc-1", "Machine learning models require training data."),
        ("doc-2", "Python is widely used for data science and AI."),
        ("doc-3", "Neural networks consist of interconnected layers."),
        ("doc-4", "The agent remembered the user's preferences."),
        ("doc-5", "Database queries should be parameterised to prevent SQL injection."),
    ]

    print(f"\nIndexing {len(documents)} documents:")
    for doc_id, text in documents:
        vector = embedder.embed(text)
        entry = VectorEntry(id=doc_id, vector=vector, metadata={"text": text})
        store.add(entry)
        print(f"  [{doc_id}] indexed: {text[:50]}")

    # Step 3: Perform similarity search
    queries = ["neural network training", "Python programming", "user preferences"]
    print("\nVector similarity search results:")
    for query in queries:
        query_vector = embedder.embed(query)
        results: list[VectorSearchResult] = store.search(query_vector, top_k=2)
        print(f"\n  Query: '{query}'")
        for result in results:
            text = result.entry.metadata.get("text", "")
            print(f"    score={result.score:.3f} | {text[:60]}")


if __name__ == "__main__":
    main()
