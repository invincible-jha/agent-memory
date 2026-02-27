"""EpisodicConsolidator — cluster similar episodes and extract semantic facts.

Uses commodity bag-of-words cosine similarity to group related episodic
memories, then promotes stable patterns as semantic memory entries.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Stop-words for tokenisation
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being have has had do does did
    will would could should may might shall must can need dare ought
    used am at by for in of on or so and but nor yet with as if
    it its this that these those i me my we our you your he him
    his she her they them their what which who whom when where why how
    all both each few more most other some such no only same than then
    too very just""".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words, return tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _build_term_vector(tokens: list[str]) -> dict[str, float]:
    """Build a normalised term-frequency vector from token list."""
    if not tokens:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    length = len(tokens)
    return {term: count / length for term, count in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two term-frequency vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(vec_a.get(term, 0.0) * vec_b.get(term, 0.0) for term in vec_a)
    magnitude_a = math.sqrt(sum(v * v for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeCluster:
    """A cluster of similar episodic memory entries.

    Parameters
    ----------
    cluster_id:
        Sequential integer identifier for this cluster.
    episode_ids:
        Memory IDs of the episodes in this cluster.
    centroid_terms:
        Most frequent terms across the cluster (top 10).
    average_importance:
        Mean importance score of all episodes in the cluster.
    """

    cluster_id: int
    episode_ids: list[str]
    centroid_terms: list[str]
    average_importance: float


@dataclass
class ConsolidationResult:
    """Result of a consolidation run.

    Parameters
    ----------
    clusters:
        Episode clusters found during consolidation.
    semantic_entries:
        New semantic MemoryEntry objects extracted from stable clusters.
    episodes_processed:
        Total number of episodic entries examined.
    clusters_formed:
        Number of clusters that exceeded the minimum cluster size.
    """

    clusters: list[EpisodeCluster] = field(default_factory=list)
    semantic_entries: list[MemoryEntry] = field(default_factory=list)
    episodes_processed: int = 0
    clusters_formed: int = 0


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EpisodicConsolidator:
    """Cluster similar episodes and extract stable patterns as semantic facts.

    Algorithm (commodity, no proprietary logic):
    1. Tokenise each episode's content into a bag-of-words TF vector.
    2. Greedily cluster episodes whose cosine similarity >= ``similarity_threshold``.
    3. For clusters with >= ``min_cluster_size`` members, emit a new semantic
       MemoryEntry summarising the cluster's centroid terms.

    Parameters
    ----------
    similarity_threshold:
        Minimum cosine similarity [0, 1] for two episodes to be grouped.
        Defaults to 0.35.
    min_cluster_size:
        Minimum number of episodes in a cluster before a semantic fact is
        extracted. Defaults to 2.
    max_centroid_terms:
        Number of top terms to include in the centroid summary. Defaults to 10.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        min_cluster_size: int = 2,
        max_centroid_terms: int = 10,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        if min_cluster_size < 1:
            raise ValueError(
                f"min_cluster_size must be >= 1, got {min_cluster_size}"
            )
        self._similarity_threshold = similarity_threshold
        self._min_cluster_size = min_cluster_size
        self._max_centroid_terms = max_centroid_terms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, episodes: Sequence[MemoryEntry]) -> ConsolidationResult:
        """Run consolidation over a collection of episodic entries.

        Parameters
        ----------
        episodes:
            Episodic MemoryEntry objects to cluster. Non-episodic entries are
            silently skipped.

        Returns
        -------
        ConsolidationResult
            Clusters found and new semantic entries derived from them.
        """
        episodic_only = [e for e in episodes if e.layer == MemoryLayer.EPISODIC]
        result = ConsolidationResult(episodes_processed=len(episodic_only))

        if not episodic_only:
            return result

        # Build TF vectors for each episode
        vectors: dict[str, dict[str, float]] = {
            entry.memory_id: _build_term_vector(_tokenise(entry.content))
            for entry in episodic_only
        }

        # Greedy single-pass clustering
        raw_clusters = self._greedy_cluster(episodic_only, vectors)

        # Build result objects
        for cluster_index, member_ids in enumerate(raw_clusters):
            if not member_ids:
                continue

            members = [e for e in episodic_only if e.memory_id in member_ids]
            centroid_terms = self._compute_centroid_terms(members)
            avg_importance = (
                sum(m.importance_score for m in members) / len(members)
            )

            cluster = EpisodeCluster(
                cluster_id=cluster_index,
                episode_ids=list(member_ids),
                centroid_terms=centroid_terms,
                average_importance=round(avg_importance, 4),
            )
            result.clusters.append(cluster)

            if len(member_ids) >= self._min_cluster_size:
                result.clusters_formed += 1
                semantic_entry = self._make_semantic_entry(cluster)
                result.semantic_entries.append(semantic_entry)

        return result

    def similarity(self, entry_a: MemoryEntry, entry_b: MemoryEntry) -> float:
        """Return the cosine similarity between two memory entries.

        Parameters
        ----------
        entry_a:
            First memory entry.
        entry_b:
            Second memory entry.

        Returns
        -------
        float
            Cosine similarity in [0, 1].
        """
        vec_a = _build_term_vector(_tokenise(entry_a.content))
        vec_b = _build_term_vector(_tokenise(entry_b.content))
        return _cosine_similarity(vec_a, vec_b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _greedy_cluster(
        self,
        entries: list[MemoryEntry],
        vectors: dict[str, dict[str, float]],
    ) -> list[set[str]]:
        """Greedy single-pass clustering algorithm.

        Each entry is assigned to the first cluster whose centroid has
        cosine similarity >= threshold, or starts a new cluster otherwise.
        Cluster centroids are recomputed as members join.
        """
        clusters: list[set[str]] = []
        cluster_centroids: list[dict[str, float]] = []

        for entry in entries:
            entry_vec = vectors[entry.memory_id]
            assigned = False

            for cluster_index, centroid in enumerate(cluster_centroids):
                similarity = _cosine_similarity(entry_vec, centroid)
                if similarity >= self._similarity_threshold:
                    clusters[cluster_index].add(entry.memory_id)
                    # Update centroid: average with incoming vector
                    cluster_centroids[cluster_index] = self._average_vectors(
                        centroid, entry_vec, len(clusters[cluster_index])
                    )
                    assigned = True
                    break

            if not assigned:
                clusters.append({entry.memory_id})
                cluster_centroids.append(dict(entry_vec))

        return clusters

    def _average_vectors(
        self,
        current_centroid: dict[str, float],
        new_vector: dict[str, float],
        current_size: int,
    ) -> dict[str, float]:
        """Return an updated centroid by incorporating a new member vector."""
        all_terms = set(current_centroid) | set(new_vector)
        weight_old = (current_size - 1) / current_size
        weight_new = 1.0 / current_size
        return {
            term: (
                current_centroid.get(term, 0.0) * weight_old
                + new_vector.get(term, 0.0) * weight_new
            )
            for term in all_terms
        }

    def _compute_centroid_terms(self, members: list[MemoryEntry]) -> list[str]:
        """Return the top N terms by aggregate TF across all cluster members."""
        term_totals: dict[str, float] = defaultdict(float)
        for member in members:
            for term, freq in _build_term_vector(_tokenise(member.content)).items():
                term_totals[term] += freq
        ranked = sorted(term_totals.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in ranked[: self._max_centroid_terms]]

    def _make_semantic_entry(self, cluster: EpisodeCluster) -> MemoryEntry:
        """Create a semantic MemoryEntry from a cluster's centroid terms."""
        terms_text = ", ".join(cluster.centroid_terms)
        content = (
            f"Consolidated pattern from {len(cluster.episode_ids)} episodes: {terms_text}."
        )
        return MemoryEntry(
            content=content,
            layer=MemoryLayer.SEMANTIC,
            importance_score=min(1.0, cluster.average_importance * 1.1),
            source=MemorySource.AGENT_INFERENCE,
            metadata={
                "cluster_id": str(cluster.cluster_id),
                "episode_count": str(len(cluster.episode_ids)),
                "source_episodes": ",".join(cluster.episode_ids[:5]),
            },
        )


__all__ = [
    "EpisodicConsolidator",
    "EpisodeCluster",
    "ConsolidationResult",
]
