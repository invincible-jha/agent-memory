"""Score fusion for hybrid retrieval.

:class:`ScoreFusion` combines results from the three retrieval backends
(vector, graph, recency) using a configurable weighted sum:

    fused_score = α·vector_score + β·graph_score + γ·recency_score

where α, β, γ are the :class:`FusionWeights` coefficients.

The default weights are:
- α (vector)  = 0.5
- β (graph)   = 0.3
- γ (recency) = 0.2

The fusion algorithm:

1. Collect all unique content strings across all three result lists.
2. For each content, gather the score from each backend that returned it
   (using 0.0 for backends that did not return it).
3. Compute the weighted sum of the three scores.
4. Normalise results to ``[0.0, 1.0]`` by dividing by the theoretical
   maximum (sum of all weights).
5. Sort by fused score descending and return the top *top_k*.

Usage
-----
>>> from agent_memory.retrieval.fusion import ScoreFusion, FusionWeights
>>> fusion = ScoreFusion(weights=FusionWeights(vector=0.6, graph=0.2, recency=0.2))
>>> results = fusion.fuse(vector_results=[], graph_results=[], recency_results=[], top_k=5)
>>> results
[]
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_memory.retrieval.graph_retriever import RetrievalResult


# ---------------------------------------------------------------------------
# FusionWeights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionWeights:
    """Weights for the three retrieval signals used in score fusion.

    Attributes
    ----------
    vector:
        Weight for vector similarity scores.  Default ``0.5``.
    graph:
        Weight for knowledge-graph traversal scores.  Default ``0.3``.
    recency:
        Weight for time-decay recency scores.  Default ``0.2``.

    Raises
    ------
    ValueError
        If the weights do not sum to approximately 1.0 (within 0.01).

    Example
    -------
    >>> FusionWeights(vector=0.5, graph=0.3, recency=0.2)
    FusionWeights(vector=0.5, graph=0.3, recency=0.2)
    """

    vector: float = 0.5
    graph: float = 0.3
    recency: float = 0.2

    def __post_init__(self) -> None:
        total = self.vector + self.graph + self.recency
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"FusionWeights must sum to 1.0, got {total:.4f} "
                f"(vector={self.vector}, graph={self.graph}, recency={self.recency})"
            )
        for name, value in (("vector", self.vector), ("graph", self.graph), ("recency", self.recency)):
            if value < 0.0:
                raise ValueError(f"Weight '{name}' must be >= 0.0, got {value}")


# ---------------------------------------------------------------------------
# ScoreFusion
# ---------------------------------------------------------------------------


class ScoreFusion:
    """Combines multiple retrieval signals using a weighted sum.

    Parameters
    ----------
    weights:
        A :class:`FusionWeights` instance specifying the α, β, γ coefficients.
        Defaults to ``FusionWeights()`` (vector=0.5, graph=0.3, recency=0.2).

    Example
    -------
    >>> from agent_memory.retrieval.graph_retriever import RetrievalResult
    >>> fusion = ScoreFusion()
    >>> v = [RetrievalResult("doc1", "vector", 0.9, {})]
    >>> g = [RetrievalResult("doc1", "graph", 0.7, {})]
    >>> r = [RetrievalResult("doc1", "recency", 0.5, {})]
    >>> results = fusion.fuse(v, g, r, top_k=5)
    >>> len(results) == 1
    True
    >>> results[0].content == "doc1"
    True
    """

    def __init__(self, weights: FusionWeights = FusionWeights()) -> None:
        self._weights = weights

    @property
    def weights(self) -> FusionWeights:
        """The configured fusion weights."""
        return self._weights

    def fuse(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        recency_results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Fuse results from all three backends into a single ranked list.

        Parameters
        ----------
        vector_results:
            Results from the vector store (source tag ``"vector"``).
        graph_results:
            Results from the graph retriever (source tag ``"graph"``).
        recency_results:
            Results from the recency scorer (source tag ``"recency"``).
        top_k:
            Maximum number of fused results to return.

        Returns
        -------
        list[RetrievalResult]
            Merged results with ``source="hybrid"`` sorted by fused score
            descending, up to *top_k* items.
        """
        if top_k <= 0:
            return []

        # Index each result list by content string
        vector_scores: dict[str, float] = {r.content: r.score for r in vector_results}
        graph_scores: dict[str, float] = {r.content: r.score for r in graph_results}
        recency_scores: dict[str, float] = {r.content: r.score for r in recency_results}
        metadata_map: dict[str, dict] = {}

        # Collect metadata: prefer vector > graph > recency ordering
        for r in recency_results:
            metadata_map.setdefault(r.content, r.metadata)
        for r in graph_results:
            metadata_map.setdefault(r.content, r.metadata)
        for r in vector_results:
            metadata_map.setdefault(r.content, r.metadata)

        # Union of all unique content keys
        all_contents = (
            set(vector_scores.keys())
            | set(graph_scores.keys())
            | set(recency_scores.keys())
        )

        if not all_contents:
            return []

        w = self._weights
        max_possible = w.vector + w.graph + w.recency  # = 1.0 by invariant

        fused: list[tuple[float, str]] = []
        for content in all_contents:
            v_score = vector_scores.get(content, 0.0)
            g_score = graph_scores.get(content, 0.0)
            r_score = recency_scores.get(content, 0.0)
            raw = w.vector * v_score + w.graph * g_score + w.recency * r_score
            normalised = round(min(1.0, max(0.0, raw / max_possible)), 6)
            fused.append((normalised, content))

        fused.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievalResult(
                content=content,
                source="hybrid",
                score=score,
                metadata=metadata_map.get(content, {}),
            )
            for score, content in fused[:top_k]
        ]
