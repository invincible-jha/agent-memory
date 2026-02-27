"""Contradiction detector — find conflicting memories via entity-attribute matching
and TF-IDF cosine similarity.

Two detectors are exported:

ContradictionDetector (original, backward-compatible)
    Entity-attribute-value triple extraction with Jaccard topic similarity.
    Used by ContradictionScanner and the rest of the existing codebase.

TFIDFContradictionDetector
    Fully self-contained detector that operates on the lightweight
    ``contradiction.models.MemoryEntry`` frozen dataclass.  Requires no
    external embedding model — all similarity is computed via a built-in
    TF-IDF implementation backed by cosine distance.  Considers temporal
    distance, topic overlap (TF-IDF cosine), and semantic negation to
    classify contradictions into ``ContradictionType`` categories.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from agent_memory.contradiction.matcher import EntityAttributeMatcher
from agent_memory.contradiction.models import (
    Contradiction,
    ContradictionType,
)
from agent_memory.contradiction.models import MemoryEntry as LightweightMemoryEntry
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.memory.types import MemoryEntry

# Similarity threshold above which two entries are considered to address the
# same topic and are therefore worth comparing for contradictions.
_TOPIC_SIMILARITY_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Original ContradictionDetector (entity-attribute strategy, backward-compatible)
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """Detect contradictions between memory entries.

    Detection strategy:
    1. For each pair of entries, check if their content is topically similar
       (Jaccard >= threshold) — if not, skip the pair.
    2. Extract entity-attribute-value triples from both.
    3. Flag pairs that share entity+attribute but differ in value.
    4. Also flag pairs with high textual similarity but opposite polarity
       (negation detection).
    """

    def __init__(
        self,
        similarity_threshold: float = _TOPIC_SIMILARITY_THRESHOLD,
    ) -> None:
        self._threshold = similarity_threshold
        self._matcher = EntityAttributeMatcher()

    def detect(self, entries: Sequence[MemoryEntry]) -> ContradictionReport:
        """Scan all entries and return a ContradictionReport."""
        entry_list = list(entries)
        report = ContradictionReport(total_entries_scanned=len(entry_list))

        for i in range(len(entry_list)):
            for j in range(i + 1, len(entry_list)):
                pair = self._compare(entry_list[i], entry_list[j])
                if pair is not None:
                    report.contradiction_pairs.append(pair)

        return report

    def detect_for_entry(
        self,
        entry: MemoryEntry,
        candidates: Sequence[MemoryEntry],
    ) -> list[ContradictionPair]:
        """Find contradictions between one entry and a set of candidates."""
        result: list[ContradictionPair] = []
        for candidate in candidates:
            if candidate.memory_id == entry.memory_id:
                continue
            pair = self._compare(entry, candidate)
            if pair is not None:
                result.append(pair)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare(
        self, entry_a: MemoryEntry, entry_b: MemoryEntry
    ) -> ContradictionPair | None:
        similarity = self._matcher.similarity(entry_a.content, entry_b.content)
        if similarity < self._threshold:
            return None

        triples_a = self._matcher.extract_triples(entry_a.content)
        triples_b = self._matcher.extract_triples(entry_b.content)
        conflicts = self._matcher.conflicts(triples_a, triples_b)

        if conflicts:
            conflict_descs = [
                f"entity={c[0].entity!r} attr={c[0].attribute!r}: "
                f"{c[0].value!r} vs {c[1].value!r}"
                for c in conflicts
            ]
            return ContradictionPair(
                entry_a_id=entry_a.memory_id,
                entry_b_id=entry_b.memory_id,
                entry_a_content=entry_a.content,
                entry_b_content=entry_b.content,
                conflict_description="; ".join(conflict_descs),
                similarity_score=round(similarity, 4),
            )

        # Negation check: high similarity but one sentence negates the other
        if similarity >= 0.5 and self._has_negation_conflict(
            entry_a.content, entry_b.content
        ):
            return ContradictionPair(
                entry_a_id=entry_a.memory_id,
                entry_b_id=entry_b.memory_id,
                entry_a_content=entry_a.content,
                entry_b_content=entry_b.content,
                conflict_description="Negation conflict detected",
                similarity_score=round(similarity, 4),
            )

        return None

    def _has_negation_conflict(self, text_a: str, text_b: str) -> bool:
        """Return True if exactly one text contains a negation marker."""
        negation_re = re.compile(
            r"\b(not|never|no|cannot|can't|won't|isn't|aren't)\b", re.IGNORECASE
        )
        a_has = bool(negation_re.search(text_a))
        b_has = bool(negation_re.search(text_b))
        return a_has != b_has


# ---------------------------------------------------------------------------
# TF-IDF helpers (no external dependency)
# ---------------------------------------------------------------------------

# Extended stop-word set used for TF-IDF tokenisation
_STOP_WORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being have has had do does did
    will would could should may might shall must can need dare ought
    used to am at by for in of on or so and but nor yet with as if
    it its this that these those i me my we our you your he him
    his she her they them their what which who whom when where why how
    all both each few more most other some such no only same than then
    too very just also about into over after""".split()
)

_NEGATION_WORDS: frozenset[str] = frozenset(
    "not never no cannot cant wont isnt arent dont doesnt didnt "
    "wasnt werent hasnt havent hadnt shouldnt wouldnt couldnt "
    "neednt darednt oughtnt usednt".split()
)

# Temporal inconsistency threshold: entries more than this many seconds apart
# are candidates for TEMPORAL_INCONSISTENCY when they discuss the same topic.
_TEMPORAL_INCONSISTENCY_SECONDS: float = 30 * 24 * 3600  # 30 days


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words, return content tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute raw term frequency (count / document length)."""
    if not tokens:
        return {}
    counter = Counter(tokens)
    length = len(tokens)
    return {term: count / length for term, count in counter.items()}


def _build_tfidf_vector(
    tokens: list[str],
    df_map: dict[str, int],
    num_docs: int,
) -> dict[str, float]:
    """Compute TF-IDF weighted vector for a tokenised document.

    Uses the standard smoothed IDF: log((N + 1) / (df + 1)) + 1
    where N is the total number of documents in the corpus.
    """
    tf_map = _tf(tokens)
    vector: dict[str, float] = {}
    for term, tf_val in tf_map.items():
        df = df_map.get(term, 0)
        idf = math.log((num_docs + 1) / (df + 1)) + 1.0
        vector[term] = tf_val * idf
    return vector


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    shared_terms = vec_a.keys() & vec_b.keys()
    dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _pairwise_cosine(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two text strings.

    The mini two-document corpus is built on the fly so IDF is document-relative.
    For a larger corpus callers should use ``TFIDFContradictionDetector`` directly
    which batches the IDF computation.
    """
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    # Build document-frequency map over the two-doc corpus
    df_map: dict[str, int] = {}
    for token in set(tokens_a):
        df_map[token] = df_map.get(token, 0) + 1
    for token in set(tokens_b):
        df_map[token] = df_map.get(token, 0) + 1

    vec_a = _build_tfidf_vector(tokens_a, df_map, num_docs=2)
    vec_b = _build_tfidf_vector(tokens_b, df_map, num_docs=2)
    return _cosine_similarity(vec_a, vec_b)


def _has_negation(text: str) -> bool:
    """Return True if the text contains any negation marker."""
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    return bool(tokens & _NEGATION_WORDS)


def _temporal_distance_seconds(
    entry_a: LightweightMemoryEntry,
    entry_b: LightweightMemoryEntry,
) -> float:
    """Return absolute temporal distance in seconds between two entries."""
    from datetime import timezone

    def to_utc(dt_val: object) -> object:
        from datetime import datetime

        if not isinstance(dt_val, datetime):
            return dt_val
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)

    ts_a = to_utc(entry_a.timestamp)
    ts_b = to_utc(entry_b.timestamp)

    from datetime import datetime

    if isinstance(ts_a, datetime) and isinstance(ts_b, datetime):
        return abs((ts_a - ts_b).total_seconds())
    return 0.0


def _classify_contradiction_type(
    entry_a: LightweightMemoryEntry,
    entry_b: LightweightMemoryEntry,
    similarity: float,
    temporal_seconds: float,
    negation_conflict: bool,
) -> tuple[ContradictionType, float, str]:
    """Classify the contradiction type and return (type, confidence, explanation).

    Classification priority:
    1. DIRECT_NEGATION   — exactly one entry contains negation words and
                           similarity is high enough to indicate they discuss
                           the same claim.
    2. TEMPORAL_INCONSISTENCY — entries are very similar but far apart in time.
    3. FACTUAL_CONFLICT  — high similarity with no negation; treated as a
                           factual claim conflict.
    4. PREFERENCE_CHANGE — moderate similarity with temporal gap — preference
                           or stance has shifted.
    """
    if negation_conflict and similarity >= 0.4:
        confidence = min(1.0, 0.5 + similarity * 0.5)
        explanation = (
            f"One entry contains explicit negation while the other does not, "
            f"with TF-IDF cosine similarity {similarity:.4f}.  Content A: "
            f"'{entry_a.content[:80]}' | Content B: '{entry_b.content[:80]}'"
        )
        return ContradictionType.DIRECT_NEGATION, confidence, explanation

    if temporal_seconds >= _TEMPORAL_INCONSISTENCY_SECONDS and similarity >= 0.35:
        days = temporal_seconds / 86400
        confidence = min(1.0, 0.4 + similarity * 0.4)
        explanation = (
            f"Entries are {days:.1f} days apart and discuss the same topic "
            f"(similarity {similarity:.4f}).  The older entry may be stale."
        )
        return ContradictionType.TEMPORAL_INCONSISTENCY, confidence, explanation

    if similarity >= 0.5:
        confidence = min(1.0, 0.3 + similarity * 0.7)
        explanation = (
            f"High topical overlap (similarity {similarity:.4f}) suggests "
            f"the entries make incompatible factual claims about the same subject."
        )
        return ContradictionType.FACTUAL_CONFLICT, confidence, explanation

    # Moderate similarity + temporal gap → preference change
    confidence = min(1.0, 0.2 + similarity * 0.6)
    days = temporal_seconds / 86400
    explanation = (
        f"Moderate topical overlap (similarity {similarity:.4f}) with a "
        f"{days:.1f}-day gap suggests a preference or stance change."
    )
    return ContradictionType.PREFERENCE_CHANGE, confidence, explanation


# ---------------------------------------------------------------------------
# TFIDFContradictionDetector
# ---------------------------------------------------------------------------


class TFIDFContradictionDetector:
    """Contradiction detector using built-in TF-IDF cosine similarity.

    No external embedding model or library is required.  All similarity
    computation is performed via a self-contained TF-IDF implementation.

    The detector operates on the lightweight ``contradiction.models.MemoryEntry``
    frozen dataclass, which is separate from the Pydantic-based model used by
    the rest of the codebase.

    Detection logic
    ---------------
    For each pair of entries the detector computes:

    1. **TF-IDF cosine similarity** — entries below ``similarity_threshold``
       are skipped (they discuss unrelated topics).
    2. **Negation analysis** — checks whether exactly one entry contains
       negation markers.
    3. **Temporal distance** — weighted into the final contradiction score via
       ``temporal_weight``.
    4. **Type classification** — assigns a ``ContradictionType`` based on the
       above signals.

    Parameters
    ----------
    similarity_threshold:
        Minimum TF-IDF cosine similarity for two entries to be considered
        topically related.  Pairs below this threshold are never flagged as
        contradictions.  Default: 0.85.
    temporal_weight:
        How heavily to weight temporal distance when computing the combined
        contradiction score.  Range [0, 1].  Higher values make the detector
        more sensitive to time-separated entries.  Default: 0.3.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        temporal_weight: float = 0.3,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold!r}"
            )
        if not 0.0 <= temporal_weight <= 1.0:
            raise ValueError(
                f"temporal_weight must be in [0, 1], got {temporal_weight!r}"
            )
        self._similarity_threshold = similarity_threshold
        self._temporal_weight = temporal_weight

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @property
    def temporal_weight(self) -> float:
        return self._temporal_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        memories: list[LightweightMemoryEntry],
    ) -> list[Contradiction]:
        """Scan all entries pairwise and return detected contradictions.

        Parameters
        ----------
        memories:
            The list of memory entries to scan.  Pairwise O(n^2) comparison.

        Returns
        -------
        list[Contradiction]
            All detected contradictions.  Empty list if none found.
        """
        result: list[Contradiction] = []
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                contradiction = self.check_pair(memories[i], memories[j])
                if contradiction is not None:
                    result.append(contradiction)
        return result

    def check_pair(
        self,
        a: LightweightMemoryEntry,
        b: LightweightMemoryEntry,
    ) -> Contradiction | None:
        """Check a single pair of entries for contradictions.

        Returns ``None`` if no contradiction is detected; otherwise returns a
        ``Contradiction`` describing the conflict.

        Parameters
        ----------
        a:
            The first memory entry.
        b:
            The second memory entry.

        Returns
        -------
        Contradiction | None
        """
        # Fast path: use pre-computed embeddings when available for both entries
        if a.embedding is not None and b.embedding is not None:
            similarity = _cosine_similarity_dense(a.embedding, b.embedding)
        else:
            similarity = _pairwise_cosine(a.content, b.content)

        if similarity < self._similarity_threshold:
            return None

        negation_conflict = _has_negation(a.content) != _has_negation(b.content)
        temporal_seconds = _temporal_distance_seconds(a, b)

        # Apply temporal weight as a penalty on the raw similarity score
        # so that temporally distant entries need higher topic similarity
        # to be flagged as contradictions.
        adjusted_similarity = similarity * (1.0 - self._temporal_weight) + similarity * (
            1.0 - min(1.0, temporal_seconds / _TEMPORAL_INCONSISTENCY_SECONDS)
        ) * self._temporal_weight

        # Re-check threshold with adjusted similarity
        if adjusted_similarity < self._similarity_threshold and not negation_conflict:
            return None

        contradiction_type, confidence, explanation = _classify_contradiction_type(
            a, b, similarity, temporal_seconds, negation_conflict
        )

        return Contradiction(
            entry_a=a,
            entry_b=b,
            similarity_score=round(similarity, 6),
            contradiction_type=contradiction_type,
            confidence=round(confidence, 6),
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Dense vector helpers (for pre-computed embeddings)
# ---------------------------------------------------------------------------


def _cosine_similarity_dense(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two dense float vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    if not vec_a:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


__all__ = [
    "ContradictionDetector",
    "TFIDFContradictionDetector",
    # TF-IDF helpers exposed for testing and reuse
    "_tokenise",
    "_tf",
    "_build_tfidf_vector",
    "_cosine_similarity",
    "_cosine_similarity_dense",
    "_pairwise_cosine",
    "_has_negation",
    "_temporal_distance_seconds",
    "_classify_contradiction_type",
    "_TEMPORAL_INCONSISTENCY_SECONDS",
    "_STOP_WORDS",
    "_NEGATION_WORDS",
]
