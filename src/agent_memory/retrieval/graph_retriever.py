"""Knowledge graph traversal retriever.

:class:`GraphRetriever` maintains an in-memory knowledge graph as an
adjacency list.  Edges are labelled triples of the form
``(source_node, relation, target_node, weight)``.

Retrieval uses breadth-first search (BFS) from nodes whose labels
contain any token from the query string, traversing up to *max_hops*
edges and collecting reached nodes as candidate results.

Each result is assigned a path-based score:

    score = node_weight × hop_decay^hop_distance

where ``hop_decay`` defaults to 0.8 (80% score per additional hop).  This
gives direct neighbours a higher score than distant nodes.

Usage
-----
>>> graph = GraphRetriever()
>>> graph.add_edge("Paris", "capital_of", "France", weight=1.0)
>>> graph.add_edge("France", "located_in", "Europe", weight=0.8)
>>> results = graph.retrieve("Paris", max_hops=2, top_k=5)
>>> len(results) >= 1
True
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# RetrievalResult — shared result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result from any backend.

    Attributes
    ----------
    content:
        The retrieved text content or node label.
    source:
        Which backend produced this result: ``"vector"``, ``"graph"``,
        or ``"recency"``.
    score:
        Relevance score in ``[0.0, 1.0]``.
    metadata:
        Arbitrary key-value pairs attached to the result.
    """

    content: str
    source: str  # "vector", "graph", "recency"
    score: float
    metadata: dict[str, object]


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """Knowledge graph traversal retriever using adjacency lists (BFS).

    The graph stores directed, weighted edges.  Multi-edges between the
    same pair of nodes with different relations are all retained.

    Parameters
    ----------
    hop_decay:
        Score multiplier applied per additional hop from the seed nodes.
        Must be in ``(0, 1]``.  Defaults to ``0.8``.

    Example
    -------
    >>> g = GraphRetriever()
    >>> g.add_edge("Alice", "knows", "Bob", weight=1.0)
    >>> g.add_edge("Bob", "works_at", "Acme", weight=0.9)
    >>> results = g.retrieve("Alice", max_hops=2, top_k=5)
    >>> any(r.content == "Acme" for r in results)
    True
    """

    def __init__(self, hop_decay: float = 0.8) -> None:
        if not (0.0 < hop_decay <= 1.0):
            raise ValueError(f"hop_decay must be in (0, 1], got {hop_decay}")
        self._hop_decay = hop_decay
        # Adjacency list: node_label -> list of (relation, target_label, weight)
        self._graph: dict[str, list[tuple[str, str, float]]] = {}
        # Reverse adjacency for undirected search (also traverse incoming edges)
        self._reverse_graph: dict[str, list[tuple[str, str, float]]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """Add a directed, weighted edge to the knowledge graph.

        Parameters
        ----------
        source:
            Label of the source node.
        relation:
            Relation type string (e.g. ``"capital_of"``, ``"knows"``).
        target:
            Label of the target node.
        weight:
            Edge weight in ``(0.0, 1.0]``.  Higher weight means greater
            relevance of this edge.

        Raises
        ------
        ValueError
            If *weight* is not in ``(0.0, 1.0]``.
        """
        if not (0.0 < weight <= 1.0):
            raise ValueError(f"weight must be in (0, 1], got {weight}")

        if source not in self._graph:
            self._graph[source] = []
        self._graph[source].append((relation, target, weight))

        # Ensure target node exists in the graph even if it has no outgoing edges
        if target not in self._graph:
            self._graph[target] = []

        # Maintain reverse graph for bidirectional traversal
        if target not in self._reverse_graph:
            self._reverse_graph[target] = []
        self._reverse_graph[target].append((relation, source, weight))

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._graph)

    def edge_count(self) -> int:
        """Return the total number of directed edges in the graph."""
        return sum(len(edges) for edges in self._graph.values())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 5,
        bidirectional: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve nodes reachable from seed nodes matching *query*.

        Seed selection: any node whose label shares at least one token
        with the query (case-insensitive).

        BFS traversal up to *max_hops* hops collects candidates.  Each
        candidate is scored as ``seed_weight × hop_decay^hop_distance``
        where ``seed_weight`` is the maximum edge weight on any path from
        the seed to the candidate.

        Parameters
        ----------
        query:
            Free-text query used to identify seed nodes.
        max_hops:
            Maximum number of hops from a seed node.
        top_k:
            Maximum number of results to return.
        bidirectional:
            When ``True`` also traverse reverse edges so that nodes
            mentioning any query token as a *target* can seed results.

        Returns
        -------
        list[RetrievalResult]
            Candidates sorted by score descending, up to *top_k* entries.
            Each entry has ``source="graph"``.
        """
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        query_tokens = _tokenise(query)
        if not query_tokens or not self._graph:
            return []

        # Find seed nodes
        seeds = [
            node
            for node in self._graph
            if any(tok in node.lower() for tok in query_tokens)
        ]
        if not seeds:
            return []

        # BFS — track best score per node
        best_scores: dict[str, float] = {}

        for seed in seeds:
            # Initial score for seed itself
            seed_score = 1.0
            if seed not in best_scores or best_scores[seed] < seed_score:
                best_scores[seed] = seed_score

            # BFS queue: (current_node, hops_remaining, accumulated_score)
            queue: deque[tuple[str, int, float]] = deque()
            queue.append((seed, max_hops, seed_score))
            visited_from_seed: set[str] = {seed}

            while queue:
                current, hops_left, current_score = queue.popleft()

                if hops_left == 0:
                    continue

                # Forward edges
                for relation, neighbour, weight in self._graph.get(current, []):
                    hop_score = current_score * weight * self._hop_decay
                    if neighbour not in best_scores or best_scores[neighbour] < hop_score:
                        best_scores[neighbour] = hop_score
                    if neighbour not in visited_from_seed:
                        visited_from_seed.add(neighbour)
                        queue.append((neighbour, hops_left - 1, hop_score))

                # Reverse edges (bidirectional)
                if bidirectional:
                    for relation, source_node, weight in self._reverse_graph.get(current, []):
                        hop_score = current_score * weight * self._hop_decay
                        if source_node not in best_scores or best_scores[source_node] < hop_score:
                            best_scores[source_node] = hop_score
                        if source_node not in visited_from_seed:
                            visited_from_seed.add(source_node)
                            queue.append((source_node, hops_left - 1, hop_score))

        # Exclude seeds themselves from results (they are query terms, not answers)
        # But include them if max_hops == 0 (trivial retrieval)
        results = [
            RetrievalResult(
                content=node,
                source="graph",
                score=round(min(1.0, max(0.0, score)), 6),
                metadata={"hops": 0, "graph_node": node},
            )
            for node, score in best_scores.items()
        ]

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase tokeniser — extracts alphanumeric tokens."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())
