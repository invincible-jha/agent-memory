"""MemoryBudget — tiered memory management with hot/warm/cold tiers.

Hot tier:  In-context window entries — highest access frequency, token-limited.
Warm tier: Cached entries — recent but not currently in context, access-aware.
Cold tier: Stored entries — long-term storage, lowest access frequency.

Entries are automatically promoted and demoted based on access frequency
and configurable token budgets for each tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MemoryTier(str, Enum):
    """Storage tier for a memory entry."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single memory tier.

    Parameters
    ----------
    max_entries:
        Maximum number of entries allowed in this tier.
    max_tokens:
        Maximum total token budget for entries in this tier.
        Each entry's token cost is estimated as ``len(content.split())``.
    promote_threshold:
        Access count at or above which an entry is promoted to a higher tier.
    demote_threshold:
        Access count below which an entry is demoted to a lower tier.
    """

    max_entries: int
    max_tokens: int
    promote_threshold: int
    demote_threshold: int


_DEFAULT_HOT_CONFIG = TierConfig(
    max_entries=20,
    max_tokens=4000,
    promote_threshold=10,  # Already hot — no higher tier
    demote_threshold=3,
)

_DEFAULT_WARM_CONFIG = TierConfig(
    max_entries=100,
    max_tokens=20000,
    promote_threshold=5,
    demote_threshold=1,
)

_DEFAULT_COLD_CONFIG = TierConfig(
    max_entries=10000,
    max_tokens=1_000_000,
    promote_threshold=3,
    demote_threshold=0,  # Nothing below cold
)


@dataclass
class TierStats:
    """Statistics for a single tier.

    Parameters
    ----------
    tier:
        The tier identifier.
    entry_count:
        Number of entries currently in this tier.
    token_count:
        Total estimated token count for all entries.
    capacity_entries:
        Maximum entries allowed.
    capacity_tokens:
        Maximum tokens allowed.
    """

    tier: MemoryTier
    entry_count: int
    token_count: int
    capacity_entries: int
    capacity_tokens: int

    @property
    def entries_utilisation(self) -> float:
        """Fraction of entry capacity used (0 – 1)."""
        return self.entry_count / self.capacity_entries if self.capacity_entries else 0.0

    @property
    def tokens_utilisation(self) -> float:
        """Fraction of token capacity used (0 – 1)."""
        return self.token_count / self.capacity_tokens if self.capacity_tokens else 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "tier": self.tier.value,
            "entry_count": self.entry_count,
            "token_count": self.token_count,
            "capacity_entries": self.capacity_entries,
            "capacity_tokens": self.capacity_tokens,
            "entries_utilisation": round(self.entries_utilisation, 4),
            "tokens_utilisation": round(self.tokens_utilisation, 4),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _estimate_tokens(content: str) -> int:
    """Estimate token count using whitespace word count (commodity approximation)."""
    return max(1, len(content.split()))


@dataclass
class _BudgetEntry:
    """Internal record tracking an entry's placement and usage."""

    entry_id: str
    content: str
    tier: MemoryTier
    access_count: int = 0
    last_accessed: datetime = field(default_factory=_utcnow)
    added_at: datetime = field(default_factory=_utcnow)
    importance_score: float = 0.5

    @property
    def token_cost(self) -> int:
        return _estimate_tokens(self.content)


class MemoryBudget:
    """Tiered memory budget controller.

    Manages which memory entries occupy the hot, warm, and cold tiers based
    on access frequency and configurable token/entry caps. Promotion and
    demotion happen automatically when ``record_access`` or ``rebalance`` is called.

    Parameters
    ----------
    hot_config:
        Configuration for the hot tier. Defaults to 20 entries / 4k tokens.
    warm_config:
        Configuration for the warm tier. Defaults to 100 entries / 20k tokens.
    cold_config:
        Configuration for the cold tier. Defaults to 10000 entries / 1M tokens.
    """

    def __init__(
        self,
        hot_config: Optional[TierConfig] = None,
        warm_config: Optional[TierConfig] = None,
        cold_config: Optional[TierConfig] = None,
    ) -> None:
        self._hot_config = hot_config or _DEFAULT_HOT_CONFIG
        self._warm_config = warm_config or _DEFAULT_WARM_CONFIG
        self._cold_config = cold_config or _DEFAULT_COLD_CONFIG
        self._entries: dict[str, _BudgetEntry] = {}

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add(
        self,
        entry_id: str,
        content: str,
        initial_tier: MemoryTier = MemoryTier.COLD,
        importance_score: float = 0.5,
    ) -> None:
        """Add an entry to the budget system.

        Parameters
        ----------
        entry_id:
            Unique identifier for this entry.
        content:
            The text content (used for token estimation).
        initial_tier:
            Starting tier for the entry. Defaults to COLD.
        importance_score:
            Importance score in [0, 1] — used when evicting from over-capacity tiers.

        Raises
        ------
        ValueError
            If an entry with the same ``entry_id`` already exists.
        """
        if entry_id in self._entries:
            raise ValueError(f"Entry {entry_id!r} already exists in budget.")
        self._entries[entry_id] = _BudgetEntry(
            entry_id=entry_id,
            content=content,
            tier=initial_tier,
            importance_score=importance_score,
        )
        self._enforce_capacity(initial_tier)

    def remove(self, entry_id: str) -> bool:
        """Remove an entry from the budget.

        Parameters
        ----------
        entry_id:
            The entry to remove.

        Returns
        -------
        bool
            True if removed, False if not found.
        """
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        return True

    def record_access(self, entry_id: str) -> Optional[MemoryTier]:
        """Record an access event for an entry, potentially promoting it.

        Parameters
        ----------
        entry_id:
            The entry that was accessed.

        Returns
        -------
        MemoryTier | None
            The new tier if the entry was promoted, otherwise None.
            Returns None if the entry is not found.
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        entry.access_count += 1
        entry.last_accessed = _utcnow()
        return self._maybe_promote(entry)

    def get_tier(self, entry_id: str) -> Optional[MemoryTier]:
        """Return the current tier for an entry, or None if not found."""
        entry = self._entries.get(entry_id)
        return entry.tier if entry else None

    def entries_in_tier(self, tier: MemoryTier) -> list[str]:
        """Return all entry IDs currently in the given tier, sorted by access count descending."""
        tier_entries = [e for e in self._entries.values() if e.tier == tier]
        tier_entries.sort(key=lambda e: e.access_count, reverse=True)
        return [e.entry_id for e in tier_entries]

    # ------------------------------------------------------------------
    # Promotion / demotion
    # ------------------------------------------------------------------

    def rebalance(self) -> dict[str, list[str]]:
        """Run a full rebalance pass — demote under-accessed entries.

        Returns
        -------
        dict[str, list[str]]
            Maps ``"promoted"`` and ``"demoted"`` to lists of affected entry IDs.
        """
        promoted: list[str] = []
        demoted: list[str] = []

        for entry in list(self._entries.values()):
            new_tier = self._evaluate_tier(entry)
            if new_tier != entry.tier:
                if _tier_rank(new_tier) > _tier_rank(entry.tier):
                    promoted.append(entry.entry_id)
                else:
                    demoted.append(entry.entry_id)
                entry.tier = new_tier

        # Enforce capacity after reassignment
        for tier in MemoryTier:
            self._enforce_capacity(tier)

        return {"promoted": promoted, "demoted": demoted}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self, tier: Optional[MemoryTier] = None) -> list[TierStats]:
        """Return tier statistics.

        Parameters
        ----------
        tier:
            If provided, return stats for that tier only. Otherwise returns
            stats for all three tiers.

        Returns
        -------
        list[TierStats]
            One TierStats per requested tier.
        """
        tiers = [tier] if tier else list(MemoryTier)
        result: list[TierStats] = []
        for t in tiers:
            tier_entries = [e for e in self._entries.values() if e.tier == t]
            token_count = sum(e.token_cost for e in tier_entries)
            config = self._config_for(t)
            result.append(
                TierStats(
                    tier=t,
                    entry_count=len(tier_entries),
                    token_count=token_count,
                    capacity_entries=config.max_entries,
                    capacity_tokens=config.max_tokens,
                )
            )
        return result

    def total_entries(self) -> int:
        """Return total number of entries across all tiers."""
        return len(self._entries)

    def budget_summary(self) -> dict[str, object]:
        """Return a human-readable summary of budget utilisation."""
        all_stats = self.stats()
        return {
            s.tier.value: {
                "entries": s.entry_count,
                "tokens": s.token_count,
                "entries_pct": round(s.entries_utilisation * 100, 1),
                "tokens_pct": round(s.tokens_utilisation * 100, 1),
            }
            for s in all_stats
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _config_for(self, tier: MemoryTier) -> TierConfig:
        if tier == MemoryTier.HOT:
            return self._hot_config
        if tier == MemoryTier.WARM:
            return self._warm_config
        return self._cold_config

    def _evaluate_tier(self, entry: _BudgetEntry) -> MemoryTier:
        """Determine the correct tier for an entry based on access count."""
        access = entry.access_count
        current = entry.tier

        if current == MemoryTier.COLD and access >= self._cold_config.promote_threshold:
            return MemoryTier.WARM
        if current == MemoryTier.WARM and access >= self._warm_config.promote_threshold:
            return MemoryTier.HOT
        if current == MemoryTier.WARM and access < self._warm_config.demote_threshold:
            return MemoryTier.COLD
        if current == MemoryTier.HOT and access < self._hot_config.demote_threshold:
            return MemoryTier.WARM
        return current

    def _maybe_promote(self, entry: _BudgetEntry) -> Optional[MemoryTier]:
        """Check if an entry should be promoted after an access event."""
        new_tier = self._evaluate_tier(entry)
        if new_tier != entry.tier and _tier_rank(new_tier) > _tier_rank(entry.tier):
            entry.tier = new_tier
            self._enforce_capacity(new_tier)
            return new_tier
        return None

    def _enforce_capacity(self, tier: MemoryTier) -> None:
        """Evict lowest-importance entries from a tier if over capacity."""
        config = self._config_for(tier)
        tier_entries = [e for e in self._entries.values() if e.tier == tier]

        # Sort by importance ascending — evict lowest importance first
        tier_entries.sort(key=lambda e: (e.importance_score, e.access_count))

        # Enforce entry count limit
        while len(tier_entries) > config.max_entries:
            victim = tier_entries.pop(0)
            self._demote_one(victim)

        # Enforce token limit
        token_total = sum(e.token_cost for e in tier_entries)
        while token_total > config.max_tokens and tier_entries:
            victim = tier_entries.pop(0)
            token_total -= victim.token_cost
            self._demote_one(victim)

    def _demote_one(self, entry: _BudgetEntry) -> None:
        """Move an entry to the next lower tier."""
        if entry.tier == MemoryTier.HOT:
            entry.tier = MemoryTier.WARM
        elif entry.tier == MemoryTier.WARM:
            entry.tier = MemoryTier.COLD
        # Already cold — cannot demote further


def _tier_rank(tier: MemoryTier) -> int:
    """Return a numeric rank for tier comparison (higher = hotter)."""
    return {MemoryTier.COLD: 0, MemoryTier.WARM: 1, MemoryTier.HOT: 2}[tier]


__all__ = ["MemoryBudget", "MemoryTier", "TierConfig", "TierStats"]
