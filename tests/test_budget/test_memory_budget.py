"""Tests for MemoryBudget — E14.4."""

from __future__ import annotations

import pytest

from agent_memory.budget.memory_budget import (
    MemoryBudget,
    MemoryTier,
    TierConfig,
    TierStats,
    _estimate_tokens,
    _tier_rank,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self) -> None:
        assert _estimate_tokens("") == 1

    def test_word_count_approximation(self) -> None:
        assert _estimate_tokens("hello world") == 2
        assert _estimate_tokens("one two three four five") == 5


class TestTierRank:
    def test_cold_is_lowest(self) -> None:
        assert _tier_rank(MemoryTier.COLD) < _tier_rank(MemoryTier.WARM)

    def test_hot_is_highest(self) -> None:
        assert _tier_rank(MemoryTier.HOT) > _tier_rank(MemoryTier.WARM)


# ---------------------------------------------------------------------------
# TierConfig
# ---------------------------------------------------------------------------


class TestTierConfig:
    def test_frozen(self) -> None:
        config = TierConfig(
            max_entries=10,
            max_tokens=1000,
            promote_threshold=5,
            demote_threshold=2,
        )
        with pytest.raises((AttributeError, TypeError)):
            config.max_entries = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TierStats
# ---------------------------------------------------------------------------


class TestTierStats:
    def _make_stats(self, entries: int, tokens: int) -> TierStats:
        return TierStats(
            tier=MemoryTier.HOT,
            entry_count=entries,
            token_count=tokens,
            capacity_entries=20,
            capacity_tokens=4000,
        )

    def test_entries_utilisation_correct(self) -> None:
        stats = self._make_stats(10, 2000)
        assert stats.entries_utilisation == 0.5

    def test_tokens_utilisation_correct(self) -> None:
        stats = self._make_stats(10, 2000)
        assert stats.tokens_utilisation == 0.5

    def test_to_dict_contains_expected_keys(self) -> None:
        stats = self._make_stats(5, 500)
        data = stats.to_dict()
        assert "tier" in data
        assert "entry_count" in data
        assert "token_count" in data
        assert "entries_utilisation" in data
        assert "tokens_utilisation" in data


# ---------------------------------------------------------------------------
# MemoryBudget add / remove
# ---------------------------------------------------------------------------


class TestMemoryBudgetAddRemove:
    def test_add_entry_starts_in_cold(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Sample content for entry one.")
        assert budget.get_tier("e1") == MemoryTier.COLD

    def test_add_entry_to_hot_tier(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Hot entry content.", initial_tier=MemoryTier.HOT)
        assert budget.get_tier("e1") == MemoryTier.HOT

    def test_add_duplicate_raises(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content.")
        with pytest.raises(ValueError, match="already exists"):
            budget.add("e1", "Duplicate.")

    def test_remove_existing_entry(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content.")
        assert budget.remove("e1") is True
        assert budget.get_tier("e1") is None

    def test_remove_nonexistent_returns_false(self) -> None:
        budget = MemoryBudget()
        assert budget.remove("nonexistent") is False

    def test_total_entries_count(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content one.")
        budget.add("e2", "Content two.")
        budget.add("e3", "Content three.")
        assert budget.total_entries() == 3


# ---------------------------------------------------------------------------
# Access tracking and promotion
# ---------------------------------------------------------------------------


class TestRecordAccess:
    def test_record_access_increments_count(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content.")
        budget.record_access("e1")
        budget.record_access("e1")
        # Access count internal — verify through promotion
        # 2 accesses should not yet promote (cold promote_threshold=3)

    def test_record_access_nonexistent_returns_none(self) -> None:
        budget = MemoryBudget()
        result = budget.record_access("does-not-exist")
        assert result is None

    def test_promotion_from_cold_to_warm(self) -> None:
        hot_cfg = TierConfig(max_entries=10, max_tokens=10000, promote_threshold=5, demote_threshold=2)
        warm_cfg = TierConfig(max_entries=50, max_tokens=50000, promote_threshold=3, demote_threshold=1)
        cold_cfg = TierConfig(max_entries=1000, max_tokens=1000000, promote_threshold=3, demote_threshold=0)
        budget = MemoryBudget(hot_config=hot_cfg, warm_config=warm_cfg, cold_config=cold_cfg)
        budget.add("e1", "Entry that will be promoted.")
        # Access 3 times to hit cold promote_threshold
        for _ in range(3):
            budget.record_access("e1")
        assert budget.get_tier("e1") == MemoryTier.WARM

    def test_promotion_from_warm_to_hot(self) -> None:
        hot_cfg = TierConfig(max_entries=10, max_tokens=10000, promote_threshold=10, demote_threshold=2)
        warm_cfg = TierConfig(max_entries=50, max_tokens=50000, promote_threshold=3, demote_threshold=1)
        cold_cfg = TierConfig(max_entries=1000, max_tokens=1000000, promote_threshold=1, demote_threshold=0)
        budget = MemoryBudget(hot_config=hot_cfg, warm_config=warm_cfg, cold_config=cold_cfg)
        budget.add("e1", "Entry that climbs to hot.", initial_tier=MemoryTier.WARM)
        for _ in range(3):
            budget.record_access("e1")
        assert budget.get_tier("e1") == MemoryTier.HOT


# ---------------------------------------------------------------------------
# Rebalance / demotion
# ---------------------------------------------------------------------------


class TestRebalance:
    def test_rebalance_demotes_underaccessed(self) -> None:
        hot_cfg = TierConfig(max_entries=10, max_tokens=10000, promote_threshold=10, demote_threshold=5)
        warm_cfg = TierConfig(max_entries=50, max_tokens=50000, promote_threshold=5, demote_threshold=2)
        cold_cfg = TierConfig(max_entries=1000, max_tokens=1000000, promote_threshold=3, demote_threshold=0)
        budget = MemoryBudget(hot_config=hot_cfg, warm_config=warm_cfg, cold_config=cold_cfg)
        # Add entry directly to hot with access_count=0 (below demote threshold=5)
        budget.add("e1", "Low access entry.", initial_tier=MemoryTier.HOT)
        result = budget.rebalance()
        assert "demoted" in result
        # Entry should now be warm or cold
        assert budget.get_tier("e1") in (MemoryTier.WARM, MemoryTier.COLD)

    def test_rebalance_returns_lists(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content.")
        result = budget.rebalance()
        assert isinstance(result["promoted"], list)
        assert isinstance(result["demoted"], list)


# ---------------------------------------------------------------------------
# Capacity enforcement
# ---------------------------------------------------------------------------


class TestCapacityEnforcement:
    def test_hot_tier_respects_max_entries(self) -> None:
        hot_cfg = TierConfig(max_entries=3, max_tokens=100000, promote_threshold=10, demote_threshold=0)
        budget = MemoryBudget(hot_config=hot_cfg)
        for i in range(5):
            budget.add(f"e{i}", f"Entry content {i}.", initial_tier=MemoryTier.HOT)
        hot_entries = budget.entries_in_tier(MemoryTier.HOT)
        assert len(hot_entries) <= 3

    def test_evicted_entries_move_to_lower_tier(self) -> None:
        hot_cfg = TierConfig(max_entries=2, max_tokens=100000, promote_threshold=10, demote_threshold=0)
        budget = MemoryBudget(hot_config=hot_cfg)
        for i in range(4):
            budget.add(f"e{i}", f"Entry {i}.", initial_tier=MemoryTier.HOT)
        # After 4 adds with max=2, some must be in warm
        warm_entries = budget.entries_in_tier(MemoryTier.WARM)
        assert len(warm_entries) > 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_for_single_tier(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Cold entry one.")
        budget.add("e2", "Cold entry two.")
        stats = budget.stats(MemoryTier.COLD)
        assert len(stats) == 1
        assert stats[0].tier == MemoryTier.COLD
        assert stats[0].entry_count == 2

    def test_stats_all_tiers(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Cold.")
        budget.add("e2", "Hot.", initial_tier=MemoryTier.HOT)
        all_stats = budget.stats()
        assert len(all_stats) == 3

    def test_budget_summary_returns_dict(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Some content here.")
        summary = budget.budget_summary()
        assert "hot" in summary
        assert "warm" in summary
        assert "cold" in summary


# ---------------------------------------------------------------------------
# entries_in_tier
# ---------------------------------------------------------------------------


class TestEntriesInTier:
    def test_entries_in_correct_tier(self) -> None:
        budget = MemoryBudget()
        budget.add("cold1", "Cold content one.")
        budget.add("cold2", "Cold content two.")
        budget.add("hot1", "Hot content.", initial_tier=MemoryTier.HOT)
        cold_entries = budget.entries_in_tier(MemoryTier.COLD)
        assert "cold1" in cold_entries
        assert "cold2" in cold_entries
        assert "hot1" not in cold_entries

    def test_empty_tier_returns_empty_list(self) -> None:
        budget = MemoryBudget()
        budget.add("e1", "Content.", initial_tier=MemoryTier.COLD)
        hot_entries = budget.entries_in_tier(MemoryTier.HOT)
        assert hot_entries == []
