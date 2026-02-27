"""Tests for agent_memory.contradiction.resolution and strategy subpackage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.contradiction.models import MemoryEntry
from agent_memory.contradiction.resolution import (
    ContradictionResolver,
    ResolutionResult,
    ResolutionStrategyBase,
)
from agent_memory.contradiction.resolution_strategies import (
    LatestWinsStrategy,
    MergeResolutionStrategy,
    UserConfirmationStrategy,
)
from agent_memory.contradiction.resolution_strategies.merge import _merge_contents, _split_sentences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _entry(content: str, ts: datetime | None = None, source: str = "user_input") -> MemoryEntry:
    timestamp = ts or datetime.now(timezone.utc)
    return MemoryEntry(content=content, timestamp=timestamp, source=source)


# ---------------------------------------------------------------------------
# Tests: ResolutionResult dataclass
# ---------------------------------------------------------------------------


class TestResolutionResult:
    def test_is_frozen(self) -> None:
        result = ResolutionResult(
            strategy_used="latest_wins",
            action="replace",
            resolved_content="content",
            requires_user_input=False,
            explanation="test",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.action = "merge"  # type: ignore[misc]

    def test_resolved_content_can_be_none(self) -> None:
        result = ResolutionResult(
            strategy_used="user_confirmation",
            action="defer",
            resolved_content=None,
            requires_user_input=True,
            explanation="Please confirm.",
        )
        assert result.resolved_content is None

    def test_valid_actions(self) -> None:
        valid = {"replace", "merge", "defer", "keep_both"}
        for action in valid:
            result = ResolutionResult(
                strategy_used="x", action=action,
                resolved_content=None, requires_user_input=False, explanation=""
            )
            assert result.action == action


# ---------------------------------------------------------------------------
# Tests: LatestWinsStrategy
# ---------------------------------------------------------------------------


class TestLatestWinsStrategy:
    def test_name(self) -> None:
        s = LatestWinsStrategy()
        assert s.name == "latest_wins"

    def test_newer_entry_wins(self) -> None:
        old = _entry("Old content", ts=_utc(2026, 1, 1))
        new = _entry("New content", ts=_utc(2026, 1, 10))
        result = LatestWinsStrategy().resolve(old, new)
        assert result.resolved_content == "New content"

    def test_older_entry_wins_when_existing_is_newer(self) -> None:
        newer_existing = _entry("Existing newer", ts=_utc(2026, 1, 10))
        older_new = _entry("Older incoming", ts=_utc(2026, 1, 1))
        result = LatestWinsStrategy().resolve(newer_existing, older_new)
        assert result.resolved_content == "Existing newer"

    def test_tie_keeps_existing(self) -> None:
        ts = _utc(2026, 1, 5)
        existing = _entry("Existing", ts=ts)
        new = _entry("New", ts=ts)
        result = LatestWinsStrategy().resolve(existing, new)
        assert result.resolved_content == "Existing"

    def test_action_is_replace(self) -> None:
        old = _entry("old", ts=_utc(2026, 1, 1))
        new = _entry("new", ts=_utc(2026, 1, 10))
        result = LatestWinsStrategy().resolve(old, new)
        assert result.action == "replace"

    def test_requires_user_input_false(self) -> None:
        old = _entry("old", ts=_utc(2026, 1, 1))
        new = _entry("new", ts=_utc(2026, 1, 10))
        result = LatestWinsStrategy().resolve(old, new)
        assert result.requires_user_input is False

    def test_strategy_used_is_latest_wins(self) -> None:
        old = _entry("old", ts=_utc(2026, 1, 1))
        new = _entry("new", ts=_utc(2026, 1, 10))
        result = LatestWinsStrategy().resolve(old, new)
        assert result.strategy_used == "latest_wins"

    def test_explanation_contains_timestamps(self) -> None:
        old = _entry("old", ts=_utc(2026, 1, 1))
        new = _entry("new", ts=_utc(2026, 1, 10))
        result = LatestWinsStrategy().resolve(old, new)
        assert "2026" in result.explanation

    def test_naive_timestamps_handled(self) -> None:
        naive_old = datetime(2026, 1, 1)
        naive_new = datetime(2026, 1, 10)
        old = MemoryEntry(content="old", timestamp=naive_old)
        new = MemoryEntry(content="new", timestamp=naive_new)
        result = LatestWinsStrategy().resolve(old, new)
        assert result.resolved_content == "new"


# ---------------------------------------------------------------------------
# Tests: UserConfirmationStrategy
# ---------------------------------------------------------------------------


class TestUserConfirmationStrategy:
    def test_name(self) -> None:
        s = UserConfirmationStrategy()
        assert s.name == "user_confirmation"

    def test_action_is_defer(self) -> None:
        existing = _entry("A is B")
        new = _entry("A is C")
        result = UserConfirmationStrategy().resolve(existing, new)
        assert result.action == "defer"

    def test_resolved_content_is_none(self) -> None:
        existing = _entry("A is B")
        new = _entry("A is C")
        result = UserConfirmationStrategy().resolve(existing, new)
        assert result.resolved_content is None

    def test_requires_user_input_true(self) -> None:
        existing = _entry("A is B")
        new = _entry("A is C")
        result = UserConfirmationStrategy().resolve(existing, new)
        assert result.requires_user_input is True

    def test_explanation_contains_both_contents(self) -> None:
        existing = _entry("Alice lives in London.")
        new = _entry("Alice lives in Berlin.")
        result = UserConfirmationStrategy().resolve(existing, new)
        assert "London" in result.explanation
        assert "Berlin" in result.explanation

    def test_strategy_used_is_user_confirmation(self) -> None:
        existing = _entry("x")
        new = _entry("y")
        result = UserConfirmationStrategy().resolve(existing, new)
        assert result.strategy_used == "user_confirmation"

    def test_custom_prompt_template(self) -> None:
        template = "Choose: [{existing}] or [{new}]?"
        s = UserConfirmationStrategy(prompt_template=template)
        existing = _entry("option A")
        new = _entry("option B")
        result = s.resolve(existing, new)
        assert "option A" in result.explanation
        assert "option B" in result.explanation

    def test_long_content_truncated_in_explanation(self) -> None:
        long_content = "x" * 300
        existing = _entry(long_content)
        new = _entry("short content")
        result = UserConfirmationStrategy().resolve(existing, new)
        # The explanation should not contain the full 300-char string verbatim
        assert len(result.explanation) < 1000  # sanity check


# ---------------------------------------------------------------------------
# Tests: MergeResolutionStrategy
# ---------------------------------------------------------------------------


class TestMergeResolutionStrategy:
    def test_name(self) -> None:
        s = MergeResolutionStrategy()
        assert s.name == "merge"

    def test_action_is_merge(self) -> None:
        existing = _entry("Alice works at Acme.")
        new = _entry("Alice lives in London.")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.action == "merge"

    def test_resolved_content_is_not_none(self) -> None:
        existing = _entry("Alice works at Acme.")
        new = _entry("Alice lives in London.")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.resolved_content is not None

    def test_merged_content_contains_both_sentences(self) -> None:
        existing = _entry("Alice works at Acme.")
        new = _entry("Alice lives in London.")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.resolved_content is not None
        assert "Acme" in result.resolved_content
        assert "London" in result.resolved_content

    def test_duplicate_sentences_not_repeated(self) -> None:
        shared = "Alice works at Acme."
        existing = _entry(f"{shared} She likes coffee.")
        new = _entry(f"{shared} She lives in London.")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.resolved_content is not None
        # "Alice works at Acme." should appear only once
        assert result.resolved_content.count("Acme") == 1

    def test_requires_user_input_false(self) -> None:
        existing = _entry("x")
        new = _entry("y")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.requires_user_input is False

    def test_strategy_used_is_merge(self) -> None:
        existing = _entry("x")
        new = _entry("y")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert result.strategy_used == "merge"

    def test_explanation_describes_merge(self) -> None:
        existing = _entry("Fact A.")
        new = _entry("Fact B.")
        result = MergeResolutionStrategy().resolve(existing, new)
        assert "sentence" in result.explanation.lower() or "merged" in result.explanation.lower()


class TestMergeHelpers:
    def test_split_sentences_basic(self) -> None:
        text = "Hello world. This is a test. Done."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_split_sentences_single(self) -> None:
        text = "Only one sentence."
        sentences = _split_sentences(text)
        assert sentences == ["Only one sentence."]

    def test_split_sentences_exclamation(self) -> None:
        text = "Yes! No."
        sentences = _split_sentences(text)
        assert len(sentences) == 2

    def test_split_sentences_empty_string(self) -> None:
        assert _split_sentences("") == []

    def test_merge_contents_no_overlap(self) -> None:
        merged = _merge_contents("Fact A.", "Fact B.")
        assert "Fact A" in merged
        assert "Fact B" in merged

    def test_merge_contents_with_overlap(self) -> None:
        merged = _merge_contents("Shared. Unique A.", "Shared. Unique B.")
        assert "Shared" in merged
        assert "Unique A" in merged
        assert "Unique B" in merged
        # "Shared" should appear only once
        assert merged.count("Shared") == 1


# ---------------------------------------------------------------------------
# Tests: ContradictionResolver
# ---------------------------------------------------------------------------


class TestContradictionResolver:
    def test_default_strategy_is_latest_wins(self) -> None:
        resolver = ContradictionResolver()
        assert resolver.default_strategy == "latest_wins"

    def test_custom_default_strategy(self) -> None:
        resolver = ContradictionResolver(default_strategy="merge")
        assert resolver.default_strategy == "merge"

    def test_invalid_default_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown default strategy"):
            ContradictionResolver(default_strategy="nonexistent")

    def test_registered_strategies_includes_builtins(self) -> None:
        resolver = ContradictionResolver()
        strategies = resolver.registered_strategies
        assert "latest_wins" in strategies
        assert "user_confirmation" in strategies
        assert "merge" in strategies

    def test_resolve_uses_default_strategy(self) -> None:
        resolver = ContradictionResolver(default_strategy="merge")
        existing = _entry("Alice works at Acme.")
        new = _entry("Alice lives in London.")
        result = resolver.resolve(existing, new)
        assert result.strategy_used == "merge"

    def test_resolve_with_explicit_strategy(self) -> None:
        resolver = ContradictionResolver()
        existing = _entry("old", ts=_utc(2026, 1, 1))
        new = _entry("new", ts=_utc(2026, 1, 10))
        result = resolver.resolve(existing, new, strategy="latest_wins")
        assert result.strategy_used == "latest_wins"

    def test_resolve_user_confirmation(self) -> None:
        resolver = ContradictionResolver()
        existing = _entry("A is B.")
        new = _entry("A is C.")
        result = resolver.resolve(existing, new, strategy="user_confirmation")
        assert result.action == "defer"
        assert result.requires_user_input is True

    def test_resolve_unknown_strategy_raises(self) -> None:
        resolver = ContradictionResolver()
        existing = _entry("x")
        new = _entry("y")
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolver.resolve(existing, new, strategy="does_not_exist")

    def test_register_strategy_custom(self) -> None:
        class KeepBothCustom(ResolutionStrategyBase):
            @property
            def name(self) -> str:
                return "keep_both_custom"

            def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
                return ResolutionResult(
                    strategy_used=self.name,
                    action="keep_both",
                    resolved_content=f"{existing.content} | {new.content}",
                    requires_user_input=False,
                    explanation="Kept both entries.",
                )

        resolver = ContradictionResolver()
        resolver.register_strategy("keep_both_custom", KeepBothCustom())
        assert "keep_both_custom" in resolver.registered_strategies

    def test_register_strategy_custom_resolves(self) -> None:
        class KeepBothCustom(ResolutionStrategyBase):
            @property
            def name(self) -> str:
                return "keep_both_custom"

            def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
                return ResolutionResult(
                    strategy_used=self.name,
                    action="keep_both",
                    resolved_content=f"{existing.content} | {new.content}",
                    requires_user_input=False,
                    explanation="Kept both entries.",
                )

        resolver = ContradictionResolver()
        resolver.register_strategy("keep_both_custom", KeepBothCustom())
        result = resolver.resolve(
            _entry("A"), _entry("B"), strategy="keep_both_custom"
        )
        assert result.action == "keep_both"
        assert result.strategy_used == "keep_both_custom"

    def test_register_invalid_strategy_type_raises(self) -> None:
        resolver = ContradictionResolver()
        with pytest.raises(TypeError, match="ResolutionStrategyBase"):
            resolver.register_strategy("bad", object())  # type: ignore[arg-type]

    def test_register_strategy_overrides_existing(self) -> None:
        class AlwaysMerge(ResolutionStrategyBase):
            @property
            def name(self) -> str:
                return "latest_wins"  # Override built-in

            def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
                return ResolutionResult(
                    strategy_used="latest_wins",
                    action="merge",
                    resolved_content="overridden",
                    requires_user_input=False,
                    explanation="Overridden.",
                )

        resolver = ContradictionResolver()
        resolver.register_strategy("latest_wins", AlwaysMerge())
        result = resolver.resolve(_entry("x"), _entry("y"))
        assert result.action == "merge"

    def test_resolve_batch_returns_correct_count(self) -> None:
        resolver = ContradictionResolver()
        pairs = [
            (_entry("A"), _entry("B")),
            (_entry("C"), _entry("D")),
            (_entry("E"), _entry("F")),
        ]
        results = resolver.resolve_batch(pairs)
        assert len(results) == 3

    def test_resolve_batch_all_use_same_strategy(self) -> None:
        resolver = ContradictionResolver()
        pairs = [
            (_entry("A"), _entry("B")),
            (_entry("C"), _entry("D")),
        ]
        results = resolver.resolve_batch(pairs, strategy="user_confirmation")
        assert all(r.strategy_used == "user_confirmation" for r in results)

    def test_resolve_batch_empty_input(self) -> None:
        resolver = ContradictionResolver()
        results = resolver.resolve_batch([])
        assert results == []

    def test_resolve_returns_resolution_result(self) -> None:
        resolver = ContradictionResolver()
        result = resolver.resolve(_entry("x"), _entry("y"))
        assert isinstance(result, ResolutionResult)
