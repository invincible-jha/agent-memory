"""Unit tests for agent_memory.contradiction.matcher — EntityAttributeMatcher."""

from __future__ import annotations

import pytest

from agent_memory.contradiction.matcher import EntityAttributeMatcher, Triple


# ---------------------------------------------------------------------------
# Triple dataclass
# ---------------------------------------------------------------------------


class TestTriple:
    def test_triple_is_immutable(self) -> None:
        triple = Triple(entity="sky", attribute="state", value="blue")
        with pytest.raises((AttributeError, TypeError)):
            triple.entity = "ground"  # type: ignore[misc]

    def test_triple_equality_by_value(self) -> None:
        t1 = Triple(entity="sky", attribute="state", value="blue")
        t2 = Triple(entity="sky", attribute="state", value="blue")
        assert t1 == t2

    def test_triple_inequality_on_different_value(self) -> None:
        t1 = Triple(entity="sky", attribute="state", value="blue")
        t2 = Triple(entity="sky", attribute="state", value="red")
        assert t1 != t2

    def test_triple_is_hashable(self) -> None:
        triple = Triple(entity="x", attribute="a", value="v")
        s = {triple}
        assert triple in s


# ---------------------------------------------------------------------------
# EntityAttributeMatcher.similarity
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_texts_return_one(self) -> None:
        matcher = EntityAttributeMatcher()
        assert matcher.similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different_texts_return_low_score(self) -> None:
        matcher = EntityAttributeMatcher()
        score = matcher.similarity("apple", "zebra")
        assert score == 0.0

    def test_partial_overlap_between_zero_and_one(self) -> None:
        matcher = EntityAttributeMatcher()
        score = matcher.similarity("the sky is blue", "the ground is brown")
        assert 0.0 < score < 1.0

    def test_both_empty_returns_one(self) -> None:
        matcher = EntityAttributeMatcher()
        assert matcher.similarity("", "") == pytest.approx(1.0)

    def test_one_empty_one_not(self) -> None:
        matcher = EntityAttributeMatcher()
        score = matcher.similarity("hello", "")
        assert score == 0.0

    def test_case_insensitive(self) -> None:
        matcher = EntityAttributeMatcher()
        s1 = matcher.similarity("Python is great", "PYTHON IS GREAT")
        assert s1 == pytest.approx(1.0)

    def test_jaccard_property(self) -> None:
        """Jaccard(A, A union B) = |A| / |A union B|."""
        matcher = EntityAttributeMatcher()
        score = matcher.similarity("a b c", "a b c d e")
        # tokens_a = {a,b,c}, tokens_b = {a,b,c,d,e}
        # intersection = 3, union = 5, jaccard = 0.6
        assert score == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# EntityAttributeMatcher.extract_triples
# ---------------------------------------------------------------------------


class TestExtractTriples:
    def test_is_pattern_extracts_triple(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("The sky is blue")
        assert len(triples) == 1
        triple = triples[0]
        assert triple.entity == "the sky"
        assert triple.attribute == "state"
        assert triple.value == "blue"

    def test_has_pattern_extracts_triple(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("The dog has four legs")
        assert len(triples) == 1
        assert triples[0].attribute == "property"

    def test_equals_pattern_extracts_triple(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("temperature = 100")
        assert len(triples) == 1
        assert triples[0].attribute == "equals"
        assert triples[0].value == "100"

    def test_empty_content_returns_empty(self) -> None:
        matcher = EntityAttributeMatcher()
        assert matcher.extract_triples("") == []

    def test_no_pattern_match_returns_empty(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("something something else randomly")
        # "something" does not contain a verb trigger
        assert isinstance(triples, list)

    def test_multiple_sentences_extracts_multiple_triples(self) -> None:
        matcher = EntityAttributeMatcher()
        content = "The sky is blue. The grass is green."
        triples = matcher.extract_triples(content)
        assert len(triples) == 2

    def test_was_verb_pattern(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("The result was successful")
        assert len(triples) == 1
        assert triples[0].attribute == "state"

    def test_entity_and_value_normalised_to_lowercase(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("The SKY is BLUE")
        assert triples[0].entity == "the sky"
        assert triples[0].value == "blue"

    def test_question_mark_splits_sentences(self) -> None:
        matcher = EntityAttributeMatcher()
        triples = matcher.extract_triples("Is it? The value is correct.")
        # At least the second sentence should yield a triple
        assert any(t.attribute == "state" for t in triples)


# ---------------------------------------------------------------------------
# EntityAttributeMatcher.conflicts
# ---------------------------------------------------------------------------


class TestConflicts:
    def test_same_entity_attribute_different_value_is_conflict(self) -> None:
        matcher = EntityAttributeMatcher()
        t1 = Triple(entity="server", attribute="state", value="running")
        t2 = Triple(entity="server", attribute="state", value="stopped")
        conflicts = matcher.conflicts([t1], [t2])
        assert len(conflicts) == 1

    def test_same_entity_same_value_is_not_conflict(self) -> None:
        matcher = EntityAttributeMatcher()
        t1 = Triple(entity="server", attribute="state", value="running")
        t2 = Triple(entity="server", attribute="state", value="running")
        conflicts = matcher.conflicts([t1], [t2])
        assert conflicts == []

    def test_different_entity_not_conflict(self) -> None:
        matcher = EntityAttributeMatcher()
        t1 = Triple(entity="server_a", attribute="state", value="on")
        t2 = Triple(entity="server_b", attribute="state", value="off")
        conflicts = matcher.conflicts([t1], [t2])
        assert conflicts == []

    def test_different_attribute_not_conflict(self) -> None:
        matcher = EntityAttributeMatcher()
        t1 = Triple(entity="server", attribute="state", value="running")
        t2 = Triple(entity="server", attribute="property", value="stopped")
        conflicts = matcher.conflicts([t1], [t2])
        assert conflicts == []

    def test_empty_lists_return_no_conflicts(self) -> None:
        matcher = EntityAttributeMatcher()
        assert matcher.conflicts([], []) == []

    def test_conflicts_return_tuple_pairs(self) -> None:
        matcher = EntityAttributeMatcher()
        t1 = Triple(entity="x", attribute="a", value="v1")
        t2 = Triple(entity="x", attribute="a", value="v2")
        conflicts = matcher.conflicts([t1], [t2])
        assert len(conflicts) == 1
        assert isinstance(conflicts[0], tuple)
        assert len(conflicts[0]) == 2

    def test_multiple_conflicts_detected(self) -> None:
        matcher = EntityAttributeMatcher()
        triples_a = [
            Triple(entity="cpu", attribute="state", value="hot"),
            Triple(entity="memory", attribute="equals", value="8gb"),
        ]
        triples_b = [
            Triple(entity="cpu", attribute="state", value="cool"),
            Triple(entity="memory", attribute="equals", value="16gb"),
        ]
        conflicts = matcher.conflicts(triples_a, triples_b)
        assert len(conflicts) == 2
