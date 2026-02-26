"""Unit tests for agent_memory.provenance.reliability — SourceReliability."""

from __future__ import annotations

import pytest

from agent_memory.memory.types import MemorySource
from agent_memory.provenance.reliability import SourceReliability


# ---------------------------------------------------------------------------
# SourceReliability.score
# ---------------------------------------------------------------------------


class TestScore:
    def test_tool_output_highest_score(self) -> None:
        rel = SourceReliability()
        assert rel.score(MemorySource.TOOL_OUTPUT) == pytest.approx(0.95)

    def test_document_score(self) -> None:
        rel = SourceReliability()
        assert rel.score(MemorySource.DOCUMENT) == pytest.approx(0.85)

    def test_external_api_score(self) -> None:
        rel = SourceReliability()
        assert rel.score(MemorySource.EXTERNAL_API) == pytest.approx(0.80)

    def test_user_input_score(self) -> None:
        rel = SourceReliability()
        assert rel.score(MemorySource.USER_INPUT) == pytest.approx(0.65)

    def test_agent_inference_lowest_score(self) -> None:
        rel = SourceReliability()
        assert rel.score(MemorySource.AGENT_INFERENCE) == pytest.approx(0.45)

    def test_all_scores_in_range(self) -> None:
        rel = SourceReliability()
        for source in MemorySource:
            score = rel.score(source)
            assert 0.0 <= score <= 1.0, f"Score out of range for {source}: {score}"

    def test_ordering_tool_gt_document_gt_api_gt_user_gt_agent(self) -> None:
        rel = SourceReliability()
        scores = [
            rel.score(MemorySource.TOOL_OUTPUT),
            rel.score(MemorySource.DOCUMENT),
            rel.score(MemorySource.EXTERNAL_API),
            rel.score(MemorySource.USER_INPUT),
            rel.score(MemorySource.AGENT_INFERENCE),
        ]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], f"Score ordering broken at index {i}"


# ---------------------------------------------------------------------------
# SourceReliability.rank
# ---------------------------------------------------------------------------


class TestRank:
    def test_rank_sorts_descending_by_reliability(self) -> None:
        rel = SourceReliability()
        sources = [
            MemorySource.AGENT_INFERENCE,
            MemorySource.TOOL_OUTPUT,
            MemorySource.USER_INPUT,
        ]
        ranked = rel.rank(sources)
        scores = [rel.score(s) for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_empty_list_returns_empty(self) -> None:
        rel = SourceReliability()
        assert rel.rank([]) == []

    def test_rank_single_item(self) -> None:
        rel = SourceReliability()
        result = rel.rank([MemorySource.DOCUMENT])
        assert result == [MemorySource.DOCUMENT]

    def test_rank_preserves_all_sources(self) -> None:
        rel = SourceReliability()
        sources = list(MemorySource)
        ranked = rel.rank(sources)
        assert set(ranked) == set(sources)

    def test_rank_all_sources_ordered_correctly(self) -> None:
        rel = SourceReliability()
        all_sources = list(MemorySource)
        ranked = rel.rank(all_sources)
        assert ranked[0] is MemorySource.TOOL_OUTPUT
        assert ranked[-1] is MemorySource.AGENT_INFERENCE


# ---------------------------------------------------------------------------
# SourceReliability.is_verified
# ---------------------------------------------------------------------------


class TestIsVerified:
    def test_tool_output_is_verified(self) -> None:
        rel = SourceReliability()
        assert rel.is_verified(MemorySource.TOOL_OUTPUT) is True

    def test_document_is_verified(self) -> None:
        rel = SourceReliability()
        assert rel.is_verified(MemorySource.DOCUMENT) is True

    def test_external_api_is_verified(self) -> None:
        rel = SourceReliability()
        # 0.80 >= 0.80 → verified
        assert rel.is_verified(MemorySource.EXTERNAL_API) is True

    def test_user_input_is_not_verified(self) -> None:
        rel = SourceReliability()
        # 0.65 < 0.80 → not verified
        assert rel.is_verified(MemorySource.USER_INPUT) is False

    def test_agent_inference_is_not_verified(self) -> None:
        rel = SourceReliability()
        assert rel.is_verified(MemorySource.AGENT_INFERENCE) is False


# ---------------------------------------------------------------------------
# SourceReliability.tier_label
# ---------------------------------------------------------------------------


class TestTierLabel:
    def test_tool_output_returns_verified_tool(self) -> None:
        rel = SourceReliability()
        assert rel.tier_label(MemorySource.TOOL_OUTPUT) == "VERIFIED_TOOL"

    def test_document_returns_verified_document(self) -> None:
        rel = SourceReliability()
        assert rel.tier_label(MemorySource.DOCUMENT) == "VERIFIED_DOCUMENT"

    def test_external_api_returns_verified_document(self) -> None:
        rel = SourceReliability()
        # 0.80 >= 0.80 → "VERIFIED_DOCUMENT" tier
        assert rel.tier_label(MemorySource.EXTERNAL_API) == "VERIFIED_DOCUMENT"

    def test_user_input_returns_user_input(self) -> None:
        rel = SourceReliability()
        assert rel.tier_label(MemorySource.USER_INPUT) == "USER_INPUT"

    def test_agent_inference_returns_agent_inference(self) -> None:
        rel = SourceReliability()
        assert rel.tier_label(MemorySource.AGENT_INFERENCE) == "AGENT_INFERENCE"

    def test_all_sources_return_string_labels(self) -> None:
        rel = SourceReliability()
        for source in MemorySource:
            label = rel.tier_label(source)
            assert isinstance(label, str)
            assert len(label) > 0
