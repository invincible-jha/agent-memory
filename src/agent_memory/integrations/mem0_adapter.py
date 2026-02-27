"""Mem0 adapter — wraps Mem0 with AumOS hybrid retrieval and contradiction detection.

Enriches the Mem0 ``Memory`` client with:
- Contradiction checking before every ``add()`` call
- Hybrid retrieval that merges Mem0 results with AumOS scoring
- Structured ``MemoryResult`` objects with provenance metadata

Install the extra to use this module::

    pip install aumos-agent-memory[mem0]

Usage
-----
::

    from agent_memory.integrations.mem0_adapter import Mem0EnhancedMemory

    memory = Mem0EnhancedMemory(user_id="agent-42")
    result = memory.add_with_contradiction_check(
        text="The capital of France is Paris.",
        metadata={"source": "user_input"},
    )
    hits = memory.search_hybrid("What is the capital of France?", top_k=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory.contradiction.detector import _has_negation, _pairwise_cosine
from agent_memory.contradiction.models import MemoryEntry as LightweightEntry
from agent_memory.memory.types import MemoryLayer, MemorySource

try:
    from mem0 import Memory  # type: ignore[import-untyped]
except ImportError as _import_error:
    raise ImportError(
        "Mem0 is required for this adapter. "
        "Install it with: pip install aumos-agent-memory[mem0]"
    ) from _import_error

logger = logging.getLogger(__name__)


@dataclass
class ContradictionWarning:
    """Describes a contradiction found before a memory add operation.

    Parameters
    ----------
    existing_memory_id:
        Identifier of the conflicting existing memory.
    existing_content:
        Text content of the conflicting existing memory.
    conflict_description:
        Human-readable description of how the entries conflict.
    similarity_score:
        Topic similarity score between the two entries (higher = more related).
    """

    existing_memory_id: str
    existing_content: str
    conflict_description: str
    similarity_score: float


@dataclass
class AddResult:
    """Result of ``add_with_contradiction_check()``.

    Parameters
    ----------
    added:
        True when the memory was successfully persisted.
    mem0_response:
        Raw response dict from the Mem0 ``add()`` call.
    contradictions:
        Any contradictions detected against existing memories.
    contradiction_prevented_add:
        True if a high-confidence contradiction blocked the add.
    text:
        The text that was (or was attempted to be) added.
    """

    added: bool
    mem0_response: Any
    contradictions: list[ContradictionWarning]
    contradiction_prevented_add: bool
    text: str

    @property
    def has_contradictions(self) -> bool:
        """True when at least one contradiction was detected."""
        return len(self.contradictions) > 0


@dataclass
class MemoryResult:
    """A single memory hit from hybrid retrieval.

    Parameters
    ----------
    memory_id:
        Identifier in the Mem0 store.
    content:
        Text content of the memory.
    raw_score:
        Similarity score returned by Mem0's search.
    hybrid_score:
        Composite score combining Mem0 similarity with AumOS freshness
        and importance heuristics.
    metadata:
        Key-value metadata attached to this memory.
    layer:
        AumOS memory layer classification (always EPISODIC for Mem0 entries).
    """

    memory_id: str
    content: str
    raw_score: float
    hybrid_score: float
    metadata: dict[str, Any]
    layer: MemoryLayer = MemoryLayer.EPISODIC


class Mem0EnhancedMemory:
    """Wraps Mem0 Memory with AumOS hybrid retrieval and contradiction detection.

    All memory add operations are checked against existing memories before
    being committed. Retrieval results are re-ranked using a hybrid score
    that combines Mem0's raw similarity with AumOS heuristics.

    Parameters
    ----------
    user_id:
        The Mem0 user namespace for memory operations.
    mem0_client:
        Pre-configured Mem0 ``Memory`` instance. When None, a new client
        is created with default configuration.
    mem0_config:
        Optional configuration dict passed to ``Memory()`` when
        ``mem0_client`` is None.
    contradiction_threshold:
        Minimum similarity score above which two memories are considered
        potentially contradictory. Default: 0.25.
    block_on_contradiction:
        When True, detected contradictions prevent the add from executing.
        When False (default), contradictions are warned but add proceeds.
    freshness_weight:
        Weight [0, 1] for freshness in the hybrid score calculation.
        Default: 0.3.

    Examples
    --------
    ::

        memory = Mem0EnhancedMemory(user_id="agent-42")
        result = memory.add_with_contradiction_check(
            text="Paris is the capital of France.",
            metadata={"source": "verified"},
        )
        hits = memory.search_hybrid("capital of France", top_k=3)
        for hit in hits:
            print(f"{hit.hybrid_score:.3f} | {hit.content}")
    """

    def __init__(
        self,
        user_id: str = "default",
        mem0_client: Optional[Any] = None,
        mem0_config: Optional[dict[str, Any]] = None,
        contradiction_threshold: float = 0.25,
        block_on_contradiction: bool = False,
        freshness_weight: float = 0.3,
    ) -> None:
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError(
                f"contradiction_threshold must be in [0.0, 1.0], got {contradiction_threshold}"
            )
        if not 0.0 <= freshness_weight <= 1.0:
            raise ValueError(
                f"freshness_weight must be in [0.0, 1.0], got {freshness_weight}"
            )

        self._user_id = user_id
        self._contradiction_threshold = contradiction_threshold
        self._block_on_contradiction = block_on_contradiction
        self._freshness_weight = freshness_weight
        if mem0_client is not None:
            self._client = mem0_client
        elif mem0_config is not None:
            self._client = Memory.from_config(mem0_config)
        else:
            self._client = Memory()

    # ------------------------------------------------------------------
    # Add with contradiction check
    # ------------------------------------------------------------------

    def add_with_contradiction_check(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        source: MemorySource = MemorySource.AGENT_INFERENCE,
    ) -> AddResult:
        """Add a memory after checking for contradictions with existing entries.

        Fetches semantically similar existing memories from Mem0 and checks
        the candidate text against them using AumOS ContradictionDetector.
        If contradictions are found and ``block_on_contradiction`` is True,
        the add is aborted.

        Parameters
        ----------
        text:
            The memory text to store.
        metadata:
            Optional key-value metadata to attach to the memory.
        source:
            AumOS memory source classification (informational only).

        Returns
        -------
        AddResult
            Contains the Mem0 response, detected contradictions, and whether
            the add was blocked.
        """
        if not text.strip():
            return AddResult(
                added=False,
                mem0_response=None,
                contradictions=[],
                contradiction_prevented_add=False,
                text=text,
            )

        contradictions = self._check_contradictions(text)

        if contradictions and self._block_on_contradiction:
            logger.warning(
                "Add blocked: %d contradiction(s) detected for text: %.80s",
                len(contradictions),
                text,
            )
            return AddResult(
                added=False,
                mem0_response=None,
                contradictions=contradictions,
                contradiction_prevented_add=True,
                text=text,
            )

        if contradictions:
            logger.info(
                "%d contradiction(s) detected but add proceeds (block_on_contradiction=False)",
                len(contradictions),
            )

        add_metadata: dict[str, Any] = {"aumos_source": source.value}
        if metadata:
            add_metadata.update(metadata)

        mem0_response = self._client.add(
            text,
            user_id=self._user_id,
            metadata=add_metadata,
        )

        logger.debug("Memory added for user %r: %.80s", self._user_id, text)

        return AddResult(
            added=True,
            mem0_response=mem0_response,
            contradictions=contradictions,
            contradiction_prevented_add=False,
            text=text,
        )

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryResult]:
        """Search Mem0 and re-rank results using a hybrid AumOS score.

        Retrieves up to ``top_k * 2`` raw results from Mem0, then computes
        a hybrid score for each result by combining Mem0's raw similarity
        score with a freshness heuristic derived from the result's metadata.
        Returns the top ``top_k`` results sorted by hybrid score (descending).

        Parameters
        ----------
        query:
            The semantic query string.
        top_k:
            Maximum number of results to return after re-ranking.
        filters:
            Optional Mem0 filter dict passed to the search call.

        Returns
        -------
        list[MemoryResult]
            Re-ranked memory results, highest hybrid score first.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        search_kwargs: dict[str, Any] = {
            "query": query,
            "user_id": self._user_id,
            "limit": top_k * 2,
        }
        if filters:
            search_kwargs["filters"] = filters

        raw_results: list[Any] = self._client.search(**search_kwargs)

        if not raw_results:
            return []

        scored_results = [self._to_memory_result(raw) for raw in raw_results]
        scored_results.sort(key=lambda r: r.hybrid_score, reverse=True)

        return scored_results[:top_k]

    def get_all(self, limit: int = 100) -> list[MemoryResult]:
        """Retrieve all memories for the current user.

        Parameters
        ----------
        limit:
            Maximum number of memories to fetch. Default: 100.

        Returns
        -------
        list[MemoryResult]
            All stored memories sorted by hybrid score descending.
        """
        raw_results: list[Any] = self._client.get_all(
            user_id=self._user_id,
            limit=limit,
        )
        scored = [self._to_memory_result(raw) for raw in raw_results]
        scored.sort(key=lambda r: r.hybrid_score, reverse=True)
        return scored

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by its Mem0 identifier.

        Parameters
        ----------
        memory_id:
            The Mem0 memory identifier.

        Returns
        -------
        bool
            True when deletion succeeded; False on error.
        """
        try:
            self._client.delete(memory_id=memory_id)
            logger.debug("Deleted memory %r", memory_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete memory %r: %s", memory_id, exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_contradictions(self, candidate_text: str) -> list[ContradictionWarning]:
        """Search for existing memories and check for contradictions."""
        try:
            similar_raw = self._client.search(
                query=candidate_text,
                user_id=self._user_id,
                limit=20,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Contradiction pre-check search failed: %s", exc)
            return []

        if not similar_raw:
            return []

        existing_entries = [self._raw_to_lightweight(raw) for raw in similar_raw]

        warnings: list[ContradictionWarning] = []
        for existing in existing_entries:
            similarity = _pairwise_cosine(candidate_text, existing.content)
            if similarity < self._contradiction_threshold:
                continue

            # Check for negation conflict (one sentence negates the other)
            candidate_has_neg = _has_negation(candidate_text)
            existing_has_neg = _has_negation(existing.content)
            negation_conflict = candidate_has_neg != existing_has_neg

            if negation_conflict or similarity >= 0.5:
                if negation_conflict:
                    description = (
                        f"Negation conflict detected (similarity={similarity:.4f}): "
                        f"one entry contains explicit negation while the other does not."
                    )
                else:
                    description = (
                        f"High topical overlap (similarity={similarity:.4f}) suggests "
                        f"incompatible factual claims about the same subject."
                    )
                warnings.append(
                    ContradictionWarning(
                        existing_memory_id=existing.id,
                        existing_content=existing.content,
                        conflict_description=description,
                        similarity_score=round(similarity, 4),
                    )
                )
        return warnings

    def _raw_to_lightweight(self, raw: Any) -> LightweightEntry:
        """Convert a Mem0 result dict to a LightweightEntry for contradiction checking."""
        if isinstance(raw, dict):
            memory_id = str(raw.get("id", "") or "")
            content = str(raw.get("memory", "") or raw.get("text", "") or "")
        else:
            memory_id = str(getattr(raw, "id", "") or "")
            content = str(getattr(raw, "memory", "") or getattr(raw, "text", "") or "")

        return LightweightEntry(
            id=memory_id or "unknown",
            content=content,
            timestamp=datetime.now(timezone.utc),
        )

    def _to_memory_result(self, raw: Any) -> MemoryResult:
        """Convert a Mem0 search result to a MemoryResult with hybrid scoring."""
        if isinstance(raw, dict):
            memory_id = str(raw.get("id", "") or "")
            content = str(raw.get("memory", "") or raw.get("text", "") or "")
            raw_score = float(raw.get("score", 0.5) or 0.5)
            metadata: dict[str, Any] = dict(raw.get("metadata", {}) or {})
        else:
            memory_id = str(getattr(raw, "id", "") or "")
            content = str(getattr(raw, "memory", "") or getattr(raw, "text", "") or "")
            raw_score = float(getattr(raw, "score", 0.5) or 0.5)
            metadata = dict(getattr(raw, "metadata", {}) or {})

        hybrid_score = self._compute_hybrid_score(raw_score=raw_score, metadata=metadata)

        return MemoryResult(
            memory_id=memory_id,
            content=content,
            raw_score=raw_score,
            hybrid_score=hybrid_score,
            metadata=metadata,
            layer=MemoryLayer.EPISODIC,
        )

    def _compute_hybrid_score(
        self,
        raw_score: float,
        metadata: dict[str, Any],
    ) -> float:
        """Compute a hybrid relevance score combining Mem0 similarity with freshness.

        The hybrid score blends the Mem0 raw similarity score with a freshness
        signal. Freshness is estimated from the ``created_at`` metadata field
        when present; otherwise a neutral score of 0.5 is used.

        Score formula::

            hybrid = (1 - freshness_weight) * raw_score
                   + freshness_weight * freshness_estimate

        Parameters
        ----------
        raw_score:
            Mem0 cosine similarity score for this result.
        metadata:
            Metadata dict from the Mem0 result.

        Returns
        -------
        float
            Hybrid score in [0.0, 1.0].
        """
        freshness = self._estimate_freshness(metadata)
        similarity_weight = 1.0 - self._freshness_weight
        hybrid = (similarity_weight * raw_score) + (self._freshness_weight * freshness)
        return round(min(1.0, max(0.0, hybrid)), 6)

    def _estimate_freshness(self, metadata: dict[str, Any]) -> float:
        """Estimate a freshness score from result metadata.

        Returns a value in [0.0, 1.0] where 1.0 = created in the last hour
        and 0.0 = older than 30 days.
        """
        created_at_raw = metadata.get("created_at") or metadata.get("timestamp")
        if created_at_raw is None:
            return 0.5  # unknown freshness — neutral

        try:
            if isinstance(created_at_raw, datetime):
                created_at = created_at_raw
            else:
                created_at = datetime.fromisoformat(str(created_at_raw))

            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_seconds = max(0.0, (now - created_at).total_seconds())
            thirty_days_seconds = 30 * 24 * 3600

            # Linear decay from 1.0 (age=0) to 0.0 (age >= 30 days)
            return max(0.0, 1.0 - age_seconds / thirty_days_seconds)
        except (ValueError, TypeError, OverflowError):
            return 0.5

    def __repr__(self) -> str:
        return (
            f"Mem0EnhancedMemory("
            f"user_id={self._user_id!r}, "
            f"block_on_contradiction={self._block_on_contradiction!r})"
        )
