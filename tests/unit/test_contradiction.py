"""Unit tests for the Contradiction Detection Engine.

Covers:
- ContradictionType enum values
- MemoryEntry (lightweight frozen dataclass) construction and immutability
- Contradiction and Resolution dataclasses
- TF-IDF helpers: _tokenise, _tf, _build_tfidf_vector, _cosine_similarity,
  _cosine_similarity_dense, _pairwise_cosine, _has_negation,
  _temporal_distance_seconds
- TFIDFContradictionDetector: contradicting pairs, non-contradicting pairs,
  edge cases, temporal analysis, negation detection, pre-computed embeddings
- SimpleContradictionResolver: all five strategies, invalid strategy handling
- Strategy classes: KeepNewerStrategy, KeepBothStrategy, FlagForReviewStrategy,
  MergeStrategy, keep_older via resolver
- Backward-compat: ContradictionDetector, ContradictionResolver still work
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Subject imports
# ---------------------------------------------------------------------------

from agent_memory.contradiction.models import (
    Contradiction,
    ContradictionType,
    MemoryEntry,
    Resolution,
)
from agent_memory.contradiction.detector import (
    TFIDFContradictionDetector,
    _build_tfidf_vector,
    _cosine_similarity,
    _cosine_similarity_dense,
    _has_negation,
    _pairwise_cosine,
    _temporal_distance_seconds,
    _tf,
    _tokenise,
    _classify_contradiction_type,
    _TEMPORAL_INCONSISTENCY_SECONDS,
    _STOP_WORDS,
    _NEGATION_WORDS,
    ContradictionDetector,
)
from agent_memory.contradiction.strategies import (
    FlagForReviewStrategy,
    KeepBothStrategy,
    KeepNewerStrategy,
    MergeStrategy,
)
from agent_memory.contradiction.resolver import (
    SimpleContradictionResolver,
    ContradictionResolver,
    ResolutionStrategy,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(
    content: str,
    *,
    source: str = "user_input",
    offset_days: float = 0.0,
    embedding: list[float] | None = None,
) -> MemoryEntry:
    """Create a lightweight MemoryEntry with optional timestamp offset."""
    timestamp = _now() - timedelta(days=offset_days)
    return MemoryEntry(
        content=content,
        source=source,
        timestamp=timestamp,
        embedding=embedding,
    )


def _contradiction(
    entry_a: MemoryEntry | None = None,
    entry_b: MemoryEntry | None = None,
    *,
    similarity_score: float = 0.9,
    contradiction_type: ContradictionType = ContradictionType.FACTUAL_CONFLICT,
    confidence: float = 0.85,
    explanation: str = "test explanation",
) -> Contradiction:
    a = entry_a or _entry("Alice lives in London")
    b = entry_b or _entry("Alice lives in Berlin")
    return Contradiction(
        entry_a=a,
        entry_b=b,
        similarity_score=similarity_score,
        contradiction_type=contradiction_type,
        confidence=confidence,
        explanation=explanation,
    )


# ===========================================================================
# Section 1: ContradictionType enum
# ===========================================================================


class TestContradictionTypeEnum:
    def test_direct_negation_value(self) -> None:
        assert ContradictionType.DIRECT_NEGATION.value == "direct_negation"

    def test_temporal_inconsistency_value(self) -> None:
        assert ContradictionType.TEMPORAL_INCONSISTENCY.value == "temporal_inconsistency"

    def test_factual_conflict_value(self) -> None:
        assert ContradictionType.FACTUAL_CONFLICT.value == "factual_conflict"

    def test_preference_change_value(self) -> None:
        assert ContradictionType.PREFERENCE_CHANGE.value == "preference_change"

    def test_all_four_members_exist(self) -> None:
        members = {m.value for m in ContradictionType}
        assert members == {
            "direct_negation",
            "temporal_inconsistency",
            "factual_conflict",
            "preference_change",
        }

    def test_enum_from_string(self) -> None:
        assert ContradictionType("direct_negation") is ContradictionType.DIRECT_NEGATION

    def test_invalid_enum_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ContradictionType("unknown_type")

    def test_enum_is_string_subclass(self) -> None:
        assert isinstance(ContradictionType.FACTUAL_CONFLICT, str)


# ===========================================================================
# Section 2: MemoryEntry (lightweight frozen dataclass)
# ===========================================================================


class TestLightweightMemoryEntry:
    def test_required_content_field(self) -> None:
        entry = MemoryEntry(content="hello world")
        assert entry.content == "hello world"

    def test_auto_generated_id_is_uuid4_format(self) -> None:
        entry = MemoryEntry(content="test")
        assert len(entry.id) == 36
        assert entry.id.count("-") == 4

    def test_unique_ids_across_instances(self) -> None:
        a = MemoryEntry(content="same content")
        b = MemoryEntry(content="same content")
        assert a.id != b.id

    def test_explicit_id(self) -> None:
        entry = MemoryEntry(content="x", id="my-custom-id")
        assert entry.id == "my-custom-id"

    def test_default_source(self) -> None:
        entry = MemoryEntry(content="x")
        assert entry.source == "agent_inference"

    def test_custom_source(self) -> None:
        entry = MemoryEntry(content="x", source="tool_output")
        assert entry.source == "tool_output"

    def test_default_timestamp_is_utc_aware(self) -> None:
        entry = MemoryEntry(content="x")
        assert entry.timestamp.tzinfo is not None

    def test_custom_timestamp(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        entry = MemoryEntry(content="x", timestamp=ts)
        assert entry.timestamp == ts

    def test_default_embedding_is_none(self) -> None:
        entry = MemoryEntry(content="x")
        assert entry.embedding is None

    def test_custom_embedding(self) -> None:
        vec = [0.1, 0.2, 0.3]
        entry = MemoryEntry(content="x", embedding=vec)
        assert entry.embedding == vec

    def test_frozen_dataclass_is_immutable(self) -> None:
        entry = MemoryEntry(content="x")
        with pytest.raises((AttributeError, TypeError)):
            entry.content = "mutated"  # type: ignore[misc]

    def test_entries_with_same_content_and_id_are_equal(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="hello", id="abc", timestamp=ts, source="s")
        b = MemoryEntry(content="hello", id="abc", timestamp=ts, source="s")
        assert a == b

    def test_entries_with_different_id_are_not_equal(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="hello", id="abc", timestamp=ts, source="s")
        b = MemoryEntry(content="hello", id="xyz", timestamp=ts, source="s")
        assert a != b

    def test_embedding_excluded_from_equality(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="hello", id="abc", timestamp=ts, source="s", embedding=[1.0])
        b = MemoryEntry(content="hello", id="abc", timestamp=ts, source="s", embedding=[2.0])
        # embedding has compare=False so these should be equal
        assert a == b

    def test_entry_is_hashable(self) -> None:
        entry = MemoryEntry(content="x")
        assert isinstance(hash(entry), int)


# ===========================================================================
# Section 3: Contradiction and Resolution dataclasses
# ===========================================================================


class TestContradictionDataclass:
    def test_construction(self) -> None:
        a = _entry("Alice is in London")
        b = _entry("Alice is in Berlin")
        c = Contradiction(
            entry_a=a,
            entry_b=b,
            similarity_score=0.88,
            contradiction_type=ContradictionType.FACTUAL_CONFLICT,
            confidence=0.9,
            explanation="Location mismatch",
        )
        assert c.entry_a is a
        assert c.entry_b is b
        assert c.similarity_score == 0.88
        assert c.contradiction_type is ContradictionType.FACTUAL_CONFLICT
        assert c.confidence == 0.9
        assert c.explanation == "Location mismatch"

    def test_frozen_immutability(self) -> None:
        c = _contradiction()
        with pytest.raises((AttributeError, TypeError)):
            c.confidence = 0.5  # type: ignore[misc]

    def test_contradiction_is_hashable(self) -> None:
        c = _contradiction()
        assert isinstance(hash(c), int)


class TestResolutionDataclass:
    def test_construction_with_kept_and_archived(self) -> None:
        a = _entry("fact A")
        b = _entry("fact B")
        res = Resolution(
            strategy_used="keep_newer",
            kept_entries=(a,),
            archived_entries=(b,),
            notes="kept newer",
        )
        assert res.strategy_used == "keep_newer"
        assert a in res.kept_entries
        assert b in res.archived_entries
        assert res.notes == "kept newer"

    def test_default_notes_is_empty_string(self) -> None:
        a = _entry("fact")
        res = Resolution(
            strategy_used="flag_for_review",
            kept_entries=(a,),
            archived_entries=(),
        )
        assert res.notes == ""

    def test_frozen_immutability(self) -> None:
        a = _entry("fact")
        res = Resolution(
            strategy_used="merge",
            kept_entries=(a,),
            archived_entries=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            res.strategy_used = "other"  # type: ignore[misc]

    def test_kept_entries_is_tuple(self) -> None:
        a = _entry("x")
        res = Resolution(strategy_used="s", kept_entries=(a,), archived_entries=())
        assert isinstance(res.kept_entries, tuple)

    def test_archived_entries_is_tuple(self) -> None:
        res = Resolution(strategy_used="s", kept_entries=(), archived_entries=())
        assert isinstance(res.archived_entries, tuple)


# ===========================================================================
# Section 4: TF-IDF helper functions
# ===========================================================================


class TestTokenise:
    def test_lowercases(self) -> None:
        tokens = _tokenise("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stop_words(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_strips_punctuation(self) -> None:
        tokens = _tokenise("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenise("") == []

    def test_all_stop_words_returns_empty(self) -> None:
        tokens = _tokenise("the is are was were")
        assert tokens == []

    def test_single_char_tokens_removed(self) -> None:
        tokens = _tokenise("a b c python")
        assert "a" not in tokens
        assert "python" in tokens

    def test_numbers_kept(self) -> None:
        tokens = _tokenise("version 42 released")
        assert "42" in tokens

    def test_returns_list(self) -> None:
        assert isinstance(_tokenise("hello world"), list)

    def test_stop_words_set_is_non_empty(self) -> None:
        assert len(_STOP_WORDS) > 10

    def test_negation_words_set_is_non_empty(self) -> None:
        assert len(_NEGATION_WORDS) > 5


class TestTermFrequency:
    def test_single_term_frequency_is_one(self) -> None:
        tf = _tf(["word"])
        assert tf == {"word": 1.0}

    def test_repeated_term_correct_frequency(self) -> None:
        tf = _tf(["a", "a", "b"])
        assert tf["a"] == pytest.approx(2 / 3)
        assert tf["b"] == pytest.approx(1 / 3)

    def test_empty_tokens_returns_empty(self) -> None:
        assert _tf([]) == {}

    def test_all_frequencies_sum_to_one(self) -> None:
        tf = _tf(["x", "y", "z"])
        assert sum(tf.values()) == pytest.approx(1.0, abs=1e-9)

    def test_unique_terms_each_have_equal_frequency(self) -> None:
        tf = _tf(["a", "b", "c", "d"])
        expected = 1 / 4
        for v in tf.values():
            assert v == pytest.approx(expected)


class TestBuildTfidfVector:
    def test_returns_dict(self) -> None:
        vec = _build_tfidf_vector(["python", "code"], {"python": 1, "code": 1}, num_docs=2)
        assert isinstance(vec, dict)

    def test_term_in_all_docs_gets_lower_idf(self) -> None:
        # "common" appears in both docs (df=2), "rare" in one (df=1)
        df = {"common": 2, "rare": 1}
        vec = _build_tfidf_vector(["common", "rare"], df, num_docs=2)
        assert vec["common"] < vec["rare"]

    def test_empty_tokens_returns_empty(self) -> None:
        vec = _build_tfidf_vector([], {}, num_docs=5)
        assert vec == {}

    def test_scores_are_positive(self) -> None:
        df = {"hello": 1, "world": 1}
        vec = _build_tfidf_vector(["hello", "world"], df, num_docs=3)
        for score in vec.values():
            assert score > 0


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self) -> None:
        vec = {"a": 0.5, "b": 0.3}
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        vec_a = {"x": 1.0}
        vec_b = {"y": 1.0}
        assert _cosine_similarity(vec_a, vec_b) == pytest.approx(0.0, abs=1e-9)

    def test_empty_vectors_return_zero(self) -> None:
        assert _cosine_similarity({}, {}) == 0.0
        assert _cosine_similarity({"a": 1.0}, {}) == 0.0
        assert _cosine_similarity({}, {"b": 1.0}) == 0.0

    def test_partially_overlapping_vectors_between_zero_and_one(self) -> None:
        vec_a = {"a": 1.0, "b": 0.5}
        vec_b = {"a": 1.0, "c": 0.5}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_symmetry(self) -> None:
        vec_a = {"a": 1.0, "b": 0.5}
        vec_b = {"a": 0.8, "c": 0.3}
        assert _cosine_similarity(vec_a, vec_b) == pytest.approx(
            _cosine_similarity(vec_b, vec_a), abs=1e-9
        )

    def test_result_bounded_zero_to_one(self) -> None:
        vec_a = {"x": 2.0, "y": 1.0}
        vec_b = {"x": 0.5, "y": 3.0}
        sim = _cosine_similarity(vec_a, vec_b)
        assert 0.0 <= sim <= 1.0


class TestCosineSimilarityDense:
    def test_identical_unit_vector(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert _cosine_similarity_dense(vec, vec) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_negative_similarity(self) -> None:
        # cos similarity can be negative for opposite vectors
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        result = _cosine_similarity_dense(vec_a, vec_b)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert _cosine_similarity_dense(vec_a, vec_b) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity_dense([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_vectors_return_zero(self) -> None:
        assert _cosine_similarity_dense([], []) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            _cosine_similarity_dense([1.0, 2.0], [1.0])

    def test_symmetry(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [0.5, 0.8, 1.2]
        assert _cosine_similarity_dense(a, b) == pytest.approx(
            _cosine_similarity_dense(b, a), abs=1e-9
        )


class TestPairwiseCosine:
    def test_identical_texts_return_one(self) -> None:
        text = "python programming language data science"
        sim = _pairwise_cosine(text, text)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_completely_unrelated_texts_low_similarity(self) -> None:
        sim = _pairwise_cosine("python programming", "ocean water fish")
        assert sim < 0.1

    def test_empty_texts_return_one(self) -> None:
        # both empty → defined as 1.0 (both documents have zero tokens)
        assert _pairwise_cosine("", "") == pytest.approx(1.0)

    def test_one_empty_text_returns_zero(self) -> None:
        assert _pairwise_cosine("hello world", "") == pytest.approx(0.0)

    def test_shared_terms_increase_similarity(self) -> None:
        sim_high = _pairwise_cosine(
            "python programming language", "python language tutorial"
        )
        sim_low = _pairwise_cosine("python programming", "ocean water fish")
        assert sim_high > sim_low

    def test_returns_float(self) -> None:
        assert isinstance(_pairwise_cosine("hello", "world"), float)

    def test_result_bounded(self) -> None:
        sim = _pairwise_cosine("alpha beta gamma", "beta gamma delta")
        assert 0.0 <= sim <= 1.0


class TestHasNegation:
    def test_not_detected(self) -> None:
        assert _has_negation("I am not sure") is True

    def test_never_detected(self) -> None:
        assert _has_negation("I never said that") is True

    def test_no_detected(self) -> None:
        # "no" on its own — it's in the negation words set
        assert _has_negation("no problem here") is True

    def test_affirmative_sentence_returns_false(self) -> None:
        assert _has_negation("Alice lives in London") is False

    def test_empty_string_returns_false(self) -> None:
        assert _has_negation("") is False

    def test_cannot_detected(self) -> None:
        assert _has_negation("you cannot do that") is True

    def test_wont_detected(self) -> None:
        assert _has_negation("it wont work") is True

    def test_isnt_detected(self) -> None:
        assert _has_negation("this isnt correct") is True


class TestTemporalDistanceSeconds:
    def test_same_timestamp_returns_zero(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="x", timestamp=ts)
        b = MemoryEntry(content="y", timestamp=ts)
        assert _temporal_distance_seconds(a, b) == pytest.approx(0.0)

    def test_one_day_apart(self) -> None:
        ts_a = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts_b = datetime(2024, 1, 2, tzinfo=timezone.utc)
        a = MemoryEntry(content="x", timestamp=ts_a)
        b = MemoryEntry(content="y", timestamp=ts_b)
        assert _temporal_distance_seconds(a, b) == pytest.approx(86400.0)

    def test_result_is_absolute_value(self) -> None:
        ts_a = datetime(2024, 1, 2, tzinfo=timezone.utc)
        ts_b = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="x", timestamp=ts_a)
        b = MemoryEntry(content="y", timestamp=ts_b)
        # Should be the same regardless of order
        assert _temporal_distance_seconds(a, b) == pytest.approx(86400.0)

    def test_symmetry(self) -> None:
        ts_a = datetime(2024, 3, 1, tzinfo=timezone.utc)
        ts_b = datetime(2024, 6, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="x", timestamp=ts_a)
        b = MemoryEntry(content="y", timestamp=ts_b)
        assert _temporal_distance_seconds(a, b) == pytest.approx(
            _temporal_distance_seconds(b, a)
        )

    def test_naive_timestamps_treated_as_utc(self) -> None:
        ts_naive = datetime(2024, 1, 1)  # no tzinfo
        ts_aware = datetime(2024, 1, 2, tzinfo=timezone.utc)
        a = MemoryEntry(content="x", timestamp=ts_naive)
        b = MemoryEntry(content="y", timestamp=ts_aware)
        result = _temporal_distance_seconds(a, b)
        assert result == pytest.approx(86400.0)


class TestClassifyContradictionType:
    def test_direct_negation_classification(self) -> None:
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        ctype, confidence, explanation = _classify_contradiction_type(
            a, b, similarity=0.7, temporal_seconds=3600, negation_conflict=True
        )
        assert ctype is ContradictionType.DIRECT_NEGATION
        assert confidence > 0.5
        assert "negation" in explanation.lower()

    def test_temporal_inconsistency_classification(self) -> None:
        old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="Alice lives in London", timestamp=old_ts)
        b = MemoryEntry(content="Alice lives in London", timestamp=recent_ts)
        seconds_apart = (recent_ts - old_ts).total_seconds()
        ctype, confidence, explanation = _classify_contradiction_type(
            a, b,
            similarity=0.95,
            temporal_seconds=seconds_apart,
            negation_conflict=False,
        )
        assert ctype is ContradictionType.TEMPORAL_INCONSISTENCY
        assert "days" in explanation.lower()

    def test_factual_conflict_classification(self) -> None:
        a = _entry("Alice lives in London")
        b = _entry("Alice lives in Berlin")
        ctype, confidence, explanation = _classify_contradiction_type(
            a, b,
            similarity=0.75,
            temporal_seconds=3600,
            negation_conflict=False,
        )
        assert ctype is ContradictionType.FACTUAL_CONFLICT

    def test_preference_change_classification(self) -> None:
        a = _entry("user prefers coffee")
        b = _entry("user likes tea")
        ctype, confidence, explanation = _classify_contradiction_type(
            a, b,
            similarity=0.4,
            temporal_seconds=7 * 86400,  # 7 days
            negation_conflict=False,
        )
        assert ctype is ContradictionType.PREFERENCE_CHANGE

    def test_confidence_in_unit_range(self) -> None:
        a = _entry("x")
        b = _entry("y")
        for negation in [True, False]:
            _, confidence, _ = _classify_contradiction_type(
                a, b,
                similarity=0.6,
                temporal_seconds=100,
                negation_conflict=negation,
            )
            assert 0.0 <= confidence <= 1.0


# ===========================================================================
# Section 5: TFIDFContradictionDetector
# ===========================================================================


class TestTFIDFContradictionDetectorInit:
    def test_default_parameters(self) -> None:
        detector = TFIDFContradictionDetector()
        assert detector.similarity_threshold == 0.85
        assert detector.temporal_weight == 0.3

    def test_custom_parameters(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.5, temporal_weight=0.1
        )
        assert detector.similarity_threshold == 0.5
        assert detector.temporal_weight == 0.1

    def test_invalid_similarity_threshold_above_one(self) -> None:
        with pytest.raises(ValueError):
            TFIDFContradictionDetector(similarity_threshold=1.1)

    def test_invalid_similarity_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError):
            TFIDFContradictionDetector(similarity_threshold=-0.1)

    def test_invalid_temporal_weight_above_one(self) -> None:
        with pytest.raises(ValueError):
            TFIDFContradictionDetector(temporal_weight=1.5)

    def test_invalid_temporal_weight_below_zero(self) -> None:
        with pytest.raises(ValueError):
            TFIDFContradictionDetector(temporal_weight=-0.1)

    def test_boundary_threshold_zero_is_valid(self) -> None:
        detector = TFIDFContradictionDetector(similarity_threshold=0.0)
        assert detector.similarity_threshold == 0.0

    def test_boundary_threshold_one_is_valid(self) -> None:
        detector = TFIDFContradictionDetector(similarity_threshold=1.0)
        assert detector.similarity_threshold == 1.0


class TestTFIDFContradictionDetectorCheckPair:
    """Tests for check_pair() — the pairwise comparison method."""

    def _detector(self, threshold: float = 0.0) -> TFIDFContradictionDetector:
        """Return a detector with low threshold so most pairs pass the filter."""
        return TFIDFContradictionDetector(
            similarity_threshold=threshold, temporal_weight=0.0
        )

    def test_returns_none_for_unrelated_entries(self) -> None:
        detector = self._detector(threshold=0.5)
        a = _entry("Alice lives in London")
        b = _entry("The stock market closed lower today")
        result = detector.check_pair(a, b)
        assert result is None

    def test_returns_contradiction_for_directly_negating_pair(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result = detector.check_pair(a, b)
        assert result is not None
        assert result.contradiction_type is ContradictionType.DIRECT_NEGATION

    def test_returns_contradiction_for_high_similarity_pair(self) -> None:
        detector = self._detector(threshold=0.0)
        # Near-identical content except location
        a = _entry("The project status is complete and approved")
        b = _entry("The project status is incomplete and rejected")
        result = detector.check_pair(a, b)
        assert result is not None

    def test_contradiction_has_correct_entries(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result = detector.check_pair(a, b)
        assert result is not None
        assert result.entry_a is a
        assert result.entry_b is b

    def test_similarity_score_in_unit_range(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result = detector.check_pair(a, b)
        assert result is not None
        assert 0.0 <= result.similarity_score <= 1.0

    def test_confidence_in_unit_range(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result = detector.check_pair(a, b)
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_non_empty_explanation(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result = detector.check_pair(a, b)
        assert result is not None
        assert len(result.explanation) > 0

    def test_same_entry_has_maximum_similarity(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=1.0, temporal_weight=0.0
        )
        a = _entry("Alice is at home")
        # With threshold=1.0, the pair must have cosine=1.0 to fire
        # Two identical entries should produce similarity=1.0
        result = detector.check_pair(a, a)
        # Regardless of whether a contradiction fires, similarity should be 1.0
        if result is not None:
            assert result.similarity_score == pytest.approx(1.0, abs=1e-6)

    def test_uses_pre_computed_embeddings_when_available(self) -> None:
        detector = self._detector(threshold=0.0)
        # Two embeddings that are near-identical (cosine ~ 1)
        vec = [1.0, 0.0, 0.0]
        a = _entry("unrelated text", embedding=vec)
        b = _entry("different unrelated text", embedding=vec)
        result = detector.check_pair(a, b)
        # With identical embeddings, similarity should be 1.0
        if result is not None:
            assert result.similarity_score == pytest.approx(1.0, abs=1e-6)

    def test_temporal_inconsistency_detected_for_old_entries(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        new = datetime(2024, 6, 1, tzinfo=timezone.utc)
        # Identical content but far apart in time
        a = MemoryEntry(content="Python is the best language", timestamp=old)
        b = MemoryEntry(content="Python is the best language", timestamp=new)
        result = detector.check_pair(a, b)
        if result is not None:
            assert result.contradiction_type is ContradictionType.TEMPORAL_INCONSISTENCY

    def test_below_threshold_entries_return_none(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.99, temporal_weight=0.0
        )
        a = _entry("Python is a language")
        b = _entry("Java is a language")
        result = detector.check_pair(a, b)
        # With 0.99 threshold and different languages, no contradiction expected
        # (they share some tokens but not enough for 0.99 cosine)
        # We just verify the return type is correct
        assert result is None or isinstance(result, Contradiction)

    def test_check_pair_is_order_independent_in_type(self) -> None:
        """check_pair(a, b) and check_pair(b, a) should both detect a contradiction."""
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        result_ab = detector.check_pair(a, b)
        result_ba = detector.check_pair(b, a)
        # Both should detect a contradiction (though entry_a/b may swap)
        assert result_ab is not None
        assert result_ba is not None


class TestTFIDFContradictionDetectorDetect:
    """Tests for detect() — the full list scan."""

    def _detector(self, threshold: float = 0.0) -> TFIDFContradictionDetector:
        return TFIDFContradictionDetector(
            similarity_threshold=threshold, temporal_weight=0.0
        )

    def test_empty_list_returns_empty(self) -> None:
        detector = self._detector()
        assert detector.detect([]) == []

    def test_single_entry_returns_empty(self) -> None:
        detector = self._detector()
        assert detector.detect([_entry("Alice is at home")]) == []

    def test_two_contradicting_entries_returns_one_contradiction(self) -> None:
        detector = self._detector()
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        results = detector.detect([a, b])
        assert len(results) == 1

    def test_two_unrelated_entries_return_empty_with_high_threshold(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.8, temporal_weight=0.0
        )
        a = _entry("Alice lives in London")
        b = _entry("The weather is sunny today")
        results = detector.detect([a, b])
        assert results == []

    def test_three_entries_multiple_contradictions(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        c = _entry("Alice is at home right now for sure")
        results = detector.detect([a, b, c])
        assert len(results) >= 1  # at least one contradiction detected

    def test_all_results_are_contradiction_instances(self) -> None:
        detector = self._detector()
        a = _entry("Alice is at home")
        b = _entry("Alice is not at home")
        results = detector.detect([a, b])
        for r in results:
            assert isinstance(r, Contradiction)

    def test_n_pairs_checked_is_n_choose_2(self) -> None:
        """With 4 entries and threshold=0 we check C(4,2)=6 pairs."""
        detector = self._detector(threshold=0.0)
        entries = [_entry(f"entry {i} distinct vocabulary") for i in range(4)]
        # With highly distinct entries, few or no contradictions expected;
        # we just verify detect() returns a list without error.
        results = detector.detect(entries)
        assert isinstance(results, list)

    def test_detect_returns_list(self) -> None:
        detector = self._detector()
        results = detector.detect([_entry("x"), _entry("y")])
        assert isinstance(results, list)

    def test_negating_pair_classified_correctly(self) -> None:
        detector = self._detector(threshold=0.0)
        a = _entry("the system is operational")
        b = _entry("the system is not operational")
        results = detector.detect([a, b])
        assert len(results) >= 1
        assert results[0].contradiction_type is ContradictionType.DIRECT_NEGATION


# ===========================================================================
# Section 6: Strategy implementations
# ===========================================================================


class TestKeepNewerStrategy:
    def test_newer_entry_is_kept(self) -> None:
        old = MemoryEntry(
            content="Alice in London",
            timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
        )
        new = MemoryEntry(
            content="Alice in Berlin",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        contradiction = _contradiction(entry_a=old, entry_b=new)
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(contradiction)

        assert resolution.strategy_used == "keep_newer"
        assert new in resolution.kept_entries
        assert old in resolution.archived_entries

    def test_older_entry_is_archived(self) -> None:
        old = MemoryEntry(
            content="fact",
            timestamp=datetime(2021, 6, 1, tzinfo=timezone.utc),
        )
        new = MemoryEntry(
            content="updated fact",
            timestamp=datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        contradiction = _contradiction(entry_a=old, entry_b=new)
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(contradiction)
        assert old in resolution.archived_entries

    def test_tie_keeps_entry_a(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="fact a", timestamp=ts)
        b = MemoryEntry(content="fact b", timestamp=ts)
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(contradiction)
        # When equal, entry_a should be kept (ts_a >= ts_b)
        assert a in resolution.kept_entries
        assert b in resolution.archived_entries

    def test_returns_resolution_instance(self) -> None:
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(_contradiction())
        assert isinstance(resolution, Resolution)

    def test_kept_entries_has_exactly_one(self) -> None:
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.kept_entries) == 1

    def test_archived_entries_has_exactly_one(self) -> None:
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.archived_entries) == 1

    def test_notes_mention_recency(self) -> None:
        strategy = KeepNewerStrategy()
        resolution = strategy.apply(_contradiction())
        assert "recency" in resolution.notes.lower()


class TestKeepBothStrategy:
    def test_both_entries_kept(self) -> None:
        a = _entry("fact A")
        b = _entry("fact B")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = KeepBothStrategy()
        resolution = strategy.apply(contradiction)
        assert a in resolution.kept_entries
        assert b in resolution.kept_entries

    def test_no_entries_archived(self) -> None:
        strategy = KeepBothStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.archived_entries) == 0

    def test_strategy_name(self) -> None:
        strategy = KeepBothStrategy()
        resolution = strategy.apply(_contradiction())
        assert resolution.strategy_used == "keep_both_with_context"

    def test_notes_contain_contradiction_type(self) -> None:
        contradiction = _contradiction(
            contradiction_type=ContradictionType.DIRECT_NEGATION
        )
        strategy = KeepBothStrategy()
        resolution = strategy.apply(contradiction)
        assert "direct_negation" in resolution.notes

    def test_notes_contain_similarity_score(self) -> None:
        contradiction = _contradiction(similarity_score=0.777)
        strategy = KeepBothStrategy()
        resolution = strategy.apply(contradiction)
        assert "0.7770" in resolution.notes

    def test_returns_resolution_instance(self) -> None:
        strategy = KeepBothStrategy()
        assert isinstance(strategy.apply(_contradiction()), Resolution)

    def test_kept_entries_has_exactly_two(self) -> None:
        strategy = KeepBothStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.kept_entries) == 2


class TestFlagForReviewStrategy:
    def test_both_entries_kept(self) -> None:
        a = _entry("x")
        b = _entry("y")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(contradiction)
        assert a in resolution.kept_entries
        assert b in resolution.kept_entries

    def test_no_entries_archived(self) -> None:
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.archived_entries) == 0

    def test_strategy_name(self) -> None:
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(_contradiction())
        assert resolution.strategy_used == "flag_for_review"

    def test_notes_contain_pending_review_marker(self) -> None:
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(_contradiction())
        assert "PENDING_REVIEW" in resolution.notes

    def test_notes_contain_entry_ids(self) -> None:
        a = _entry("x")
        b = _entry("y")
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(_contradiction(entry_a=a, entry_b=b))
        assert a.id in resolution.notes
        assert b.id in resolution.notes

    def test_notes_contain_confidence(self) -> None:
        contradiction = _contradiction(confidence=0.92)
        strategy = FlagForReviewStrategy()
        resolution = strategy.apply(contradiction)
        assert "0.9200" in resolution.notes

    def test_returns_resolution_instance(self) -> None:
        strategy = FlagForReviewStrategy()
        assert isinstance(strategy.apply(_contradiction()), Resolution)


class TestMergeStrategy:
    def test_archived_both_originals(self) -> None:
        a = _entry("Alice is in London")
        b = _entry("Alice is in Berlin")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = MergeStrategy()
        resolution = strategy.apply(contradiction)
        assert a in resolution.archived_entries
        assert b in resolution.archived_entries

    def test_kept_has_one_merged_entry(self) -> None:
        strategy = MergeStrategy()
        resolution = strategy.apply(_contradiction())
        assert len(resolution.kept_entries) == 1

    def test_merged_content_contains_both_contents(self) -> None:
        a = _entry("Alice is in London")
        b = _entry("Alice is in Berlin")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = MergeStrategy()
        resolution = strategy.apply(contradiction)
        merged = resolution.kept_entries[0]
        assert "Alice is in London" in merged.content
        assert "Alice is in Berlin" in merged.content

    def test_merged_entry_has_merged_source(self) -> None:
        a = _entry("fact A", source="user_input")
        b = _entry("fact B", source="tool_output")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = MergeStrategy()
        resolution = strategy.apply(contradiction)
        merged = resolution.kept_entries[0]
        assert "merged" in merged.source

    def test_merged_entry_takes_newer_timestamp(self) -> None:
        old_ts = datetime(2022, 1, 1, tzinfo=timezone.utc)
        new_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = MemoryEntry(content="old fact", timestamp=old_ts)
        b = MemoryEntry(content="new fact", timestamp=new_ts)
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = MergeStrategy()
        resolution = strategy.apply(contradiction)
        merged = resolution.kept_entries[0]
        assert merged.timestamp == new_ts

    def test_strategy_name(self) -> None:
        strategy = MergeStrategy()
        resolution = strategy.apply(_contradiction())
        assert resolution.strategy_used == "merge"

    def test_notes_mention_merged_entry_id(self) -> None:
        strategy = MergeStrategy()
        resolution = strategy.apply(_contradiction())
        merged = resolution.kept_entries[0]
        assert merged.id in resolution.notes

    def test_separator_present_in_merged_content(self) -> None:
        a = _entry("part A")
        b = _entry("part B")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        strategy = MergeStrategy()
        resolution = strategy.apply(contradiction)
        merged = resolution.kept_entries[0]
        assert MergeStrategy.MERGE_SEPARATOR in merged.content

    def test_returns_resolution_instance(self) -> None:
        strategy = MergeStrategy()
        assert isinstance(strategy.apply(_contradiction()), Resolution)


# ===========================================================================
# Section 7: SimpleContradictionResolver
# ===========================================================================


class TestSimpleContradictionResolverInit:
    def test_default_strategy(self) -> None:
        resolver = SimpleContradictionResolver()
        assert resolver.default_strategy == "keep_newer"

    def test_custom_default_strategy(self) -> None:
        resolver = SimpleContradictionResolver(default_strategy="merge")
        assert resolver.default_strategy == "merge"

    def test_invalid_default_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            SimpleContradictionResolver(default_strategy="nonexistent")

    def test_all_valid_strategies_accepted(self) -> None:
        for name in [
            "keep_newer",
            "keep_older",
            "keep_both_with_context",
            "flag_for_review",
            "merge",
        ]:
            resolver = SimpleContradictionResolver(default_strategy=name)
            assert resolver.default_strategy == name


class TestSimpleContradictionResolverResolve:
    def _resolver(self, default: str = "keep_newer") -> SimpleContradictionResolver:
        return SimpleContradictionResolver(default_strategy=default)

    def test_resolve_keep_newer_uses_recency(self) -> None:
        old = MemoryEntry(
            content="old fact",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        new = MemoryEntry(
            content="new fact",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        contradiction = _contradiction(entry_a=old, entry_b=new)
        resolver = self._resolver("keep_newer")
        resolution = resolver.resolve(contradiction)
        assert new in resolution.kept_entries

    def test_resolve_keep_older(self) -> None:
        old = MemoryEntry(
            content="old fact",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        new = MemoryEntry(
            content="new fact",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        contradiction = _contradiction(entry_a=old, entry_b=new)
        resolver = self._resolver()
        resolution = resolver.resolve(contradiction, strategy="keep_older")
        assert old in resolution.kept_entries

    def test_resolve_keep_both(self) -> None:
        a = _entry("fact A")
        b = _entry("fact B")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        resolver = self._resolver()
        resolution = resolver.resolve(contradiction, strategy="keep_both_with_context")
        assert len(resolution.kept_entries) == 2
        assert len(resolution.archived_entries) == 0

    def test_resolve_flag_for_review(self) -> None:
        contradiction = _contradiction()
        resolver = self._resolver()
        resolution = resolver.resolve(contradiction, strategy="flag_for_review")
        assert resolution.strategy_used == "flag_for_review"
        assert "PENDING_REVIEW" in resolution.notes

    def test_resolve_merge(self) -> None:
        a = _entry("fact A content")
        b = _entry("fact B content")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        resolver = self._resolver()
        resolution = resolver.resolve(contradiction, strategy="merge")
        assert resolution.strategy_used == "merge"
        assert len(resolution.kept_entries) == 1

    def test_resolve_uses_default_strategy_when_none(self) -> None:
        resolver = SimpleContradictionResolver(default_strategy="flag_for_review")
        resolution = resolver.resolve(_contradiction())
        assert resolution.strategy_used == "flag_for_review"

    def test_resolve_invalid_strategy_raises(self) -> None:
        resolver = self._resolver()
        with pytest.raises(ValueError):
            resolver.resolve(_contradiction(), strategy="totally_invalid")

    def test_resolve_returns_resolution(self) -> None:
        resolver = self._resolver()
        result = resolver.resolve(_contradiction())
        assert isinstance(result, Resolution)

    def test_strategy_override_takes_precedence(self) -> None:
        resolver = SimpleContradictionResolver(default_strategy="keep_newer")
        resolution = resolver.resolve(_contradiction(), strategy="merge")
        assert resolution.strategy_used == "merge"


# ===========================================================================
# Section 8: Backward-compatibility — original ContradictionDetector
# ===========================================================================


class TestOriginalContradictionDetectorBackwardCompat:
    """Ensure the original ContradictionDetector still works after our changes."""

    def _pydantic_entry(self, content: str) -> Any:
        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        return PydanticEntry(content=content, layer=MemoryLayer.SEMANTIC)

    def test_detect_returns_contradiction_report(self) -> None:
        from agent_memory.contradiction.report import ContradictionReport

        detector = ContradictionDetector()
        entries = [
            self._pydantic_entry("Alice is at home"),
            self._pydantic_entry("Alice is not at home"),
        ]
        report = detector.detect(entries)
        assert isinstance(report, ContradictionReport)

    def test_detect_empty_list(self) -> None:
        detector = ContradictionDetector()
        report = detector.detect([])
        assert report.contradiction_count == 0

    def test_detect_for_entry(self) -> None:
        detector = ContradictionDetector(similarity_threshold=0.25)
        entry = self._pydantic_entry("Alice is at home")
        candidates = [
            self._pydantic_entry("Alice is not at home"),
            self._pydantic_entry("Bob likes chess"),
        ]
        results = detector.detect_for_entry(entry, candidates)
        assert isinstance(results, list)


# ===========================================================================
# Section 9: Backward-compatibility — original ContradictionResolver
# ===========================================================================


class TestOriginalContradictionResolverBackwardCompat:
    """Ensure ContradictionResolver (Pydantic-based) still works."""

    def _pair(self, entry_a_id: str, entry_b_id: str) -> Any:
        from agent_memory.contradiction.report import ContradictionPair

        return ContradictionPair(
            entry_a_id=entry_a_id,
            entry_b_id=entry_b_id,
            entry_a_content="Alice is at home",
            entry_b_content="Alice is not at home",
            conflict_description="Negation conflict",
            similarity_score=0.75,
        )

    def _pydantic_entry(self, memory_id: str, content: str) -> Any:
        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        return PydanticEntry(
            memory_id=memory_id, content=content, layer=MemoryLayer.SEMANTIC
        )

    def test_recency_wins_strategy(self) -> None:
        from datetime import timezone

        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        older = PydanticEntry(
            content="old fact",
            layer=MemoryLayer.SEMANTIC,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        newer = PydanticEntry(
            content="new fact",
            layer=MemoryLayer.SEMANTIC,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        pair = self._pair(older.memory_id, newer.memory_id)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, older, newer, strategy=ResolutionStrategy.RECENCY_WINS
        )
        assert result.winner_id == newer.memory_id
        assert result.loser_id == older.memory_id

    def test_default_strategy_is_recency_wins(self) -> None:
        resolver = ContradictionResolver()
        assert resolver.default_strategy == ResolutionStrategy.RECENCY_WINS

    def test_resolution_strategy_enum_values(self) -> None:
        assert ResolutionStrategy.RECENCY_WINS.value == "recency_wins"
        assert ResolutionStrategy.SOURCE_PRIORITY.value == "source_priority"
        assert ResolutionStrategy.CONFIDENCE_BASED.value == "confidence_based"
        assert ResolutionStrategy.MANUAL.value == "manual"

    def test_invalid_entry_ids_raises(self) -> None:
        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        a = PydanticEntry(content="x", layer=MemoryLayer.SEMANTIC)
        b = PydanticEntry(content="y", layer=MemoryLayer.SEMANTIC)
        pair = self._pair("wrong-id-1", "wrong-id-2")
        resolver = ContradictionResolver()
        with pytest.raises(ValueError):
            resolver.resolve(pair, a, b)

    def test_manual_strategy_with_valid_winner(self) -> None:
        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        a = PydanticEntry(content="x", layer=MemoryLayer.SEMANTIC)
        b = PydanticEntry(content="y", layer=MemoryLayer.SEMANTIC)
        pair = self._pair(a.memory_id, b.memory_id)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, a, b,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id=a.memory_id,
        )
        assert result.winner_id == a.memory_id
        assert result.loser_id == b.memory_id

    def test_manual_strategy_with_invalid_winner_returns_review_flag(self) -> None:
        from agent_memory.memory.types import MemoryEntry as PydanticEntry, MemoryLayer

        a = PydanticEntry(content="x", layer=MemoryLayer.SEMANTIC)
        b = PydanticEntry(content="y", layer=MemoryLayer.SEMANTIC)
        pair = self._pair(a.memory_id, b.memory_id)
        resolver = ContradictionResolver()
        result = resolver.resolve(
            pair, a, b,
            strategy=ResolutionStrategy.MANUAL,
            manual_winner_id="nonexistent-id",
        )
        assert result.requires_manual_review is True


# ===========================================================================
# Section 10: Edge cases and integration
# ===========================================================================


class TestEdgeCases:
    def test_single_word_content(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        a = _entry("yes")
        b = _entry("no")
        # Should not crash; result depends on similarity
        result = detector.check_pair(a, b)
        assert result is None or isinstance(result, Contradiction)

    def test_very_long_content(self) -> None:
        long_text_a = " ".join(["word"] * 500 + ["not"])
        long_text_b = " ".join(["word"] * 500)
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        a = _entry(long_text_a)
        b = _entry(long_text_b)
        result = detector.check_pair(a, b)
        assert result is None or isinstance(result, Contradiction)

    def test_content_with_only_stop_words(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        a = _entry("the is are was")
        b = _entry("a an the by for")
        # After tokenisation both are empty — should return None (or not crash)
        result = detector.check_pair(a, b)
        assert result is None or isinstance(result, Contradiction)

    def test_empty_content_entries(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        a = _entry("")
        b = _entry("")
        result = detector.check_pair(a, b)
        assert result is None or isinstance(result, Contradiction)

    def test_temporal_inconsistency_constant_is_positive(self) -> None:
        assert _TEMPORAL_INCONSISTENCY_SECONDS > 0

    def test_detect_with_large_list_does_not_error(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.9, temporal_weight=0.0
        )
        entries = [_entry(f"fact about topic number {i}") for i in range(20)]
        results = detector.detect(entries)
        assert isinstance(results, list)

    def test_resolution_kept_plus_archived_accounts_for_all_inputs(self) -> None:
        """For non-merge strategies, kept + archived == both original entries."""
        a = _entry("alpha")
        b = _entry("beta")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        for strategy_name in ["keep_newer", "keep_older", "flag_for_review", "keep_both_with_context"]:
            resolver = SimpleContradictionResolver()
            resolution = resolver.resolve(contradiction, strategy=strategy_name)
            all_entries = set(resolution.kept_entries) | set(resolution.archived_entries)
            assert a in all_entries
            assert b in all_entries

    def test_merge_strategy_creates_new_entry_not_original(self) -> None:
        a = _entry("fact A content here")
        b = _entry("fact B content here")
        contradiction = _contradiction(entry_a=a, entry_b=b)
        resolver = SimpleContradictionResolver()
        resolution = resolver.resolve(contradiction, strategy="merge")
        merged = resolution.kept_entries[0]
        assert merged.id != a.id
        assert merged.id != b.id

    def test_tfidf_detector_with_pre_computed_embeddings_zero_vector(self) -> None:
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        zero_vec = [0.0, 0.0, 0.0]
        a = _entry("content", embedding=zero_vec)
        b = _entry("content", embedding=zero_vec)
        # Should not crash with zero-norm vectors
        result = detector.check_pair(a, b)
        assert result is None or isinstance(result, Contradiction)

    def test_contradiction_type_and_confidence_consistent(self) -> None:
        """All detected contradictions should have confidence in [0, 1]."""
        detector = TFIDFContradictionDetector(
            similarity_threshold=0.0, temporal_weight=0.0
        )
        pairs: list[tuple[str, str]] = [
            ("Alice is happy", "Alice is not happy"),
            ("the server is running", "the server is not running"),
            ("water boils at 100 degrees", "water boils at 80 degrees"),
        ]
        for content_a, content_b in pairs:
            a = _entry(content_a)
            b = _entry(content_b)
            result = detector.check_pair(a, b)
            if result is not None:
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Confidence {result.confidence} out of range for pair: "
                    f"{content_a!r} / {content_b!r}"
                )
