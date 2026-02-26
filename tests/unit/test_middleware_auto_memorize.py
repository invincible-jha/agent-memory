"""Unit tests for agent_memory.middleware.auto_memorize — AutoMemorizeMiddleware."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.middleware.auto_memorize import AutoMemorizeMiddleware, Interaction
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.unified.config import MemoryConfig
from agent_memory.unified.memory import UnifiedMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory() -> UnifiedMemory:
    config = MemoryConfig(
        auto_score_importance=False,
        auto_score_freshness=False,
        auto_tag_provenance=False,
        consolidation_interval=0,
    )
    return UnifiedMemory(config=config, storage=InMemoryStorage())


def _make_interaction(
    user_input: str = "Hello agent",
    agent_response: str = "Hello user, how can I help?",
    session_id: str = "",
    turn_id: str = "",
    extra_metadata: dict[str, str] | None = None,
) -> Interaction:
    return Interaction(
        user_input=user_input,
        agent_response=agent_response,
        session_id=session_id,
        turn_id=turn_id,
        extra_metadata=extra_metadata or {},
    )


# ---------------------------------------------------------------------------
# Interaction dataclass
# ---------------------------------------------------------------------------


class TestInteraction:
    def test_required_fields(self) -> None:
        interaction = Interaction(user_input="hi", agent_response="hello")
        assert interaction.user_input == "hi"
        assert interaction.agent_response == "hello"

    def test_default_session_id_empty(self) -> None:
        interaction = Interaction(user_input="x", agent_response="y")
        assert interaction.session_id == ""

    def test_default_turn_id_empty(self) -> None:
        interaction = Interaction(user_input="x", agent_response="y")
        assert interaction.turn_id == ""

    def test_default_extra_metadata_empty(self) -> None:
        interaction = Interaction(user_input="x", agent_response="y")
        assert interaction.extra_metadata == {}

    def test_occurred_at_is_utc_aware(self) -> None:
        interaction = Interaction(user_input="x", agent_response="y")
        assert interaction.occurred_at.tzinfo is not None


# ---------------------------------------------------------------------------
# AutoMemorizeMiddleware construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_stored_count_starts_at_zero(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        assert middleware.stored_count == 0

    def test_importance_boost_clamped_to_zero_minimum(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, importance_boost=-1.0)
        assert middleware._importance_boost == 0.0

    def test_importance_boost_clamped_to_one_maximum(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, importance_boost=5.0)
        assert middleware._importance_boost == 1.0

    def test_valid_importance_boost(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, importance_boost=0.1)
        assert middleware._importance_boost == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# AutoMemorizeMiddleware.record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_returns_list(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        result = middleware.record(_make_interaction())
        assert isinstance(result, list)

    def test_record_both_turns_stores_two_entries(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_user_turns=True, store_agent_turns=True
        )
        result = middleware.record(_make_interaction())
        assert len(result) == 2

    def test_record_only_user_turn_stores_one_entry(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_user_turns=True, store_agent_turns=False
        )
        result = middleware.record(_make_interaction())
        assert len(result) == 1
        assert result[0].source is MemorySource.USER_INPUT

    def test_record_only_agent_turn_stores_one_entry(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_user_turns=False, store_agent_turns=True
        )
        result = middleware.record(_make_interaction())
        assert len(result) == 1
        assert result[0].source is MemorySource.AGENT_INFERENCE

    def test_record_neither_turn_stores_nothing(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_user_turns=False, store_agent_turns=False
        )
        result = middleware.record(_make_interaction())
        assert len(result) == 0

    def test_record_skips_short_user_input(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, min_content_length=10)
        result = middleware.record(_make_interaction(user_input="hi", agent_response="Hello there user"))
        # User input is too short; only agent entry stored
        assert len(result) == 1
        assert result[0].source is MemorySource.AGENT_INFERENCE

    def test_record_skips_short_agent_response(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, min_content_length=10)
        result = middleware.record(_make_interaction(user_input="Hello agent", agent_response="ok"))
        # Agent response is too short; only user entry stored
        assert len(result) == 1
        assert result[0].source is MemorySource.USER_INPUT

    def test_user_entry_layer_is_episodic(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        result = middleware.record(_make_interaction())
        assert result[0].layer is MemoryLayer.EPISODIC

    def test_agent_entry_layer_is_episodic(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_user_turns=False)
        result = middleware.record(_make_interaction())
        assert result[0].layer is MemoryLayer.EPISODIC

    def test_user_entry_metadata_role_is_user(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        result = middleware.record(_make_interaction())
        assert result[0].metadata.get("role") == "user"

    def test_agent_entry_metadata_role_is_agent(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_user_turns=False)
        result = middleware.record(_make_interaction())
        assert result[0].metadata.get("role") == "agent"

    def test_session_id_in_metadata_when_provided(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        interaction = _make_interaction(session_id="sess-123")
        result = middleware.record(interaction)
        assert result[0].metadata.get("session_id") == "sess-123"

    def test_session_id_absent_when_empty(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        interaction = _make_interaction(session_id="")
        result = middleware.record(interaction)
        assert "session_id" not in result[0].metadata

    def test_turn_id_in_metadata_when_provided(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        interaction = _make_interaction(turn_id="turn-42")
        result = middleware.record(interaction)
        assert result[0].metadata.get("turn_id") == "turn-42"

    def test_extra_metadata_merged_into_entry(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        interaction = _make_interaction(extra_metadata={"topic": "finance"})
        result = middleware.record(interaction)
        assert result[0].metadata.get("topic") == "finance"

    def test_occurred_at_used_as_created_at(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        interaction = _make_interaction()
        result = middleware.record(interaction)
        # created_at should match the interaction's occurred_at
        assert result[0].created_at == interaction.occurred_at

    def test_importance_boost_applied(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_agent_turns=False, importance_boost=0.3
        )
        interaction = _make_interaction()
        result = middleware.record(interaction)
        # The boosted score should exceed the default 0.5
        assert result[0].importance_score > 0.5

    def test_importance_boost_capped_at_one(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(
            memory=memory, store_agent_turns=False, importance_boost=1.0
        )
        interaction = _make_interaction()
        result = middleware.record(interaction)
        assert result[0].importance_score <= 1.0

    def test_stored_count_increments_per_entry(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        middleware.record(_make_interaction())  # 2 entries
        middleware.record(_make_interaction())  # 2 more
        assert middleware.stored_count == 4

    def test_stored_entries_are_memory_entry_instances(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        result = middleware.record(_make_interaction())
        assert all(isinstance(e, MemoryEntry) for e in result)


# ---------------------------------------------------------------------------
# AutoMemorizeMiddleware.record_raw
# ---------------------------------------------------------------------------


class TestRecordRaw:
    def test_record_raw_returns_list(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        result = middleware.record_raw(
            user_input="What is the weather?",
            agent_response="The weather is sunny today.",
        )
        assert isinstance(result, list)

    def test_record_raw_stores_entries(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        result = middleware.record_raw("Hello", "Hi there")
        assert len(result) == 2

    def test_record_raw_with_session_and_turn_ids(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        result = middleware.record_raw(
            user_input="test question here",
            agent_response="answer",
            session_id="s1",
            turn_id="t1",
        )
        assert result[0].metadata.get("session_id") == "s1"
        assert result[0].metadata.get("turn_id") == "t1"

    def test_record_raw_extra_metadata(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory, store_agent_turns=False)
        result = middleware.record_raw(
            user_input="test question here",
            agent_response="answer",
            extra_metadata={"custom": "value"},
        )
        assert result[0].metadata.get("custom") == "value"

    def test_record_raw_increments_stored_count(self) -> None:
        memory = _make_memory()
        middleware = AutoMemorizeMiddleware(memory=memory)
        middleware.record_raw("user input here", "agent response here")
        assert middleware.stored_count == 2
