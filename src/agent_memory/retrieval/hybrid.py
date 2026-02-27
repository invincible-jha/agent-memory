"""Hybrid retrieval — multi-signal retrieval fusion orchestrator.

:class:`HybridRetriever` combines three complementary retrieval signals:

1. **Vector retrieval** — semantic similarity via an optional vector store
   (injected via the :class:`VectorStoreProtocol` structural protocol).
2. **Graph retrieval** — knowledge graph traversal via :class:`GraphRetriever`.
3. **Recency scoring** — time-weighted decay via :class:`RecencyScorer`.

Results from all active backends are fused using :class:`ScoreFusion` and
returned as a ranked list of :class:`RetrievalResult` objects.

Degenerate behaviour
--------------------
- When no vector store is provided, the vector signal is omitted and the
  fusion weights are re-normalised across the remaining signals.
- When the graph or recency backend yields no results, only the available
  signals contribute to the final ranking.

Usage
-----
>>> from agent_memory.retrieval.hybrid import HybridRetriever, GraphRetriever
>>> graph = GraphRetriever()
>>> graph.add_edge("Paris", "capital_of", "France", weight=1.0)
>>> retriever = HybridRetriever(graph=graph)
>>> results = retriever.retrieve("Paris France", top_k=5)
>>> isinstance(results, list)
True
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_memory.retrieval.fusion import FusionWeights, ScoreFusion
from agent_memory.retrieval.graph_retriever import GraphRetriever, RetrievalResult
from agent_memory.retrieval.recency_scorer import RecencyScorer


# ---------------------------------------------------------------------------
# Vector store protocol (structural typing — no import dependency)
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Minimal protocol required of a vector store used by HybridRetriever.

    Any object implementing ``query(text: str, top_k: int) -> list[RetrievalResult]``
    satisfies this protocol.
    """

    def query(self, text: str, top_k: int) -> list[RetrievalResult]:
        """Search the vector store and return top_k results."""
        ...


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Multi-signal retrieval combining vector, graph, and recency signals.

    Parameters
    ----------
    vector_store:
        Optional vector store implementing :class:`VectorStoreProtocol`.
        When ``None`` the vector signal is skipped and the remaining
        weights are re-normalised.
    graph:
        Optional :class:`~agent_memory.retrieval.graph_retriever.GraphRetriever`
        instance.  When ``None`` a new empty graph is created.
    recency:
        Optional :class:`~agent_memory.retrieval.recency_scorer.RecencyScorer`
        instance.  When ``None`` a new scorer with default parameters
        (168 h half-life) is created.
    fusion:
        Optional :class:`~agent_memory.retrieval.fusion.ScoreFusion` instance.
        When ``None`` a fusion with default weights is created.

    Example
    -------
    >>> retriever = HybridRetriever()
    >>> results = retriever.retrieve("machine learning", top_k=5)
    >>> isinstance(results, list)
    True
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol | None = None,
        graph: GraphRetriever | None = None,
        recency: RecencyScorer | None = None,
        fusion: ScoreFusion | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._graph = graph if graph is not None else GraphRetriever()
        self._recency = recency if recency is not None else RecencyScorer()
        self._fusion = fusion if fusion is not None else ScoreFusion()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve the most relevant results for *query* across all signals.

        Parameters
        ----------
        query:
            Free-text query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Results ranked by fused score, highest first.
        """
        vector_results: list[RetrievalResult] = []
        graph_results: list[RetrievalResult] = []
        recency_results: list[RetrievalResult] = []

        # --- Vector signal ---
        if self._vector_store is not None:
            try:
                vector_results = self._vector_store.query(query, top_k=top_k)
            except Exception:
                vector_results = []

        # --- Graph signal ---
        try:
            graph_results = self._graph.retrieve(query, top_k=top_k)
        except Exception:
            graph_results = []

        # --- Recency signal: rank any content already retrieved ---
        # Collect unique contents and their timestamps from graph/vector results
        recency_results = self._recency_from_existing(
            vector_results + graph_results, top_k=top_k
        )

        return self._fusion.fuse(
            vector_results=vector_results,
            graph_results=graph_results,
            recency_results=recency_results,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recency_from_existing(
        self,
        existing: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Produce recency-scored versions of already-retrieved results.

        Extracts ``"timestamp"`` from each result's metadata (if present)
        and scores it with the recency scorer.  Results without a timestamp
        receive a score of 0.0.

        Parameters
        ----------
        existing:
            Results from vector and graph retrievers.
        top_k:
            Maximum results to return.

        Returns
        -------
        list[RetrievalResult]
        """
        from datetime import datetime, timezone

        scored: list[RetrievalResult] = []
        for result in existing:
            ts_raw = result.metadata.get("timestamp")
            if isinstance(ts_raw, datetime):
                timestamp = ts_raw
            elif isinstance(ts_raw, str):
                try:
                    timestamp = datetime.fromisoformat(ts_raw)
                except ValueError:
                    timestamp = None
            else:
                timestamp = None

            if timestamp is not None:
                recency_score = self._recency.score(timestamp)
            else:
                recency_score = 0.0

            scored.append(
                RetrievalResult(
                    content=result.content,
                    source="recency",
                    score=recency_score,
                    metadata=result.metadata,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
