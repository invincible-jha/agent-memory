"""Unit tests for agent_memory framework adapters."""
from __future__ import annotations

import pytest

from agent_memory.adapters import (
    AnthropicMemoryBridge,
    CrewAIMemoryBridge,
    LangChainMemoryBridge,
    MicrosoftMemoryBridge,
    OpenAIMemoryBridge,
)


# ---------------------------------------------------------------------------
# LangChainMemoryBridge
# ---------------------------------------------------------------------------


class TestLangChainMemoryBridge:
    def test_construction_no_args(self) -> None:
        bridge = LangChainMemoryBridge()
        assert bridge.memory_store is None
        assert bridge._items == []

    def test_construction_with_store(self) -> None:
        sentinel = object()
        bridge = LangChainMemoryBridge(memory_store=sentinel)
        assert bridge.memory_store is sentinel

    def test_store_conversation_returns_str(self) -> None:
        bridge = LangChainMemoryBridge()
        entry_id = bridge.store_conversation([{"role": "user", "content": "hi"}])
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_store_conversation_appends_to_items(self) -> None:
        bridge = LangChainMemoryBridge()
        bridge.store_conversation(["msg1", "msg2"])
        assert len(bridge._items) == 1
        assert bridge._items[0]["count"] == 2

    def test_retrieve_context_returns_list(self) -> None:
        bridge = LangChainMemoryBridge()
        bridge.store_conversation(["hello", "world"])
        result = bridge.retrieve_context("hello")
        assert isinstance(result, list)

    def test_retrieve_context_respects_k(self) -> None:
        bridge = LangChainMemoryBridge()
        for i in range(10):
            bridge.store_conversation([f"msg{i}"])
        result = bridge.retrieve_context("msg", k=3)
        assert len(result) <= 3

    def test_clear_memory_empties_items(self) -> None:
        bridge = LangChainMemoryBridge()
        bridge.store_conversation(["a", "b"])
        bridge.clear_memory()
        assert bridge._items == []

    def test_store_returns_unique_ids(self) -> None:
        bridge = LangChainMemoryBridge()
        id1 = bridge.store_conversation(["a"])
        id2 = bridge.store_conversation(["b"])
        assert id1 != id2


# ---------------------------------------------------------------------------
# CrewAIMemoryBridge
# ---------------------------------------------------------------------------


class TestCrewAIMemoryBridge:
    def test_construction_no_args(self) -> None:
        bridge = CrewAIMemoryBridge()
        assert bridge.memory_store is None
        assert bridge._items == []

    def test_construction_with_store(self) -> None:
        sentinel = object()
        bridge = CrewAIMemoryBridge(memory_store=sentinel)
        assert bridge.memory_store is sentinel

    def test_store_task_result_returns_str(self) -> None:
        bridge = CrewAIMemoryBridge()
        entry_id = bridge.store_task_result("analyse_data", "result text")
        assert isinstance(entry_id, str)

    def test_store_task_result_appends_item(self) -> None:
        bridge = CrewAIMemoryBridge()
        bridge.store_task_result("task_a", "output")
        assert len(bridge._items) == 1
        assert bridge._items[0]["task_name"] == "task_a"

    def test_retrieve_knowledge_returns_list(self) -> None:
        bridge = CrewAIMemoryBridge()
        bridge.store_task_result("summarise", "important data summary")
        results = bridge.retrieve_knowledge("summarise")
        assert isinstance(results, list)
        assert len(results) == 1

    def test_retrieve_knowledge_no_match_returns_empty(self) -> None:
        bridge = CrewAIMemoryBridge()
        bridge.store_task_result("task_x", "output y")
        results = bridge.retrieve_knowledge("zzz_no_match")
        assert results == []

    def test_get_crew_memory_returns_list(self) -> None:
        bridge = CrewAIMemoryBridge()
        bridge.store_task_result("t1", "r1")
        bridge.store_task_result("t2", "r2")
        memory = bridge.get_crew_memory()
        assert isinstance(memory, list)
        assert len(memory) == 2

    def test_store_returns_unique_ids(self) -> None:
        bridge = CrewAIMemoryBridge()
        id1 = bridge.store_task_result("t1", "r1")
        id2 = bridge.store_task_result("t2", "r2")
        assert id1 != id2


# ---------------------------------------------------------------------------
# OpenAIMemoryBridge
# ---------------------------------------------------------------------------


class TestOpenAIMemoryBridge:
    def test_construction_no_args(self) -> None:
        bridge = OpenAIMemoryBridge()
        assert bridge.memory_store is None
        assert bridge._items == []

    def test_construction_with_store(self) -> None:
        sentinel = object()
        bridge = OpenAIMemoryBridge(memory_store=sentinel)
        assert bridge.memory_store is sentinel

    def test_store_messages_returns_str(self) -> None:
        bridge = OpenAIMemoryBridge()
        entry_id = bridge.store_messages("thread-1", ["hello", "world"])
        assert isinstance(entry_id, str)

    def test_store_messages_keyed_by_thread(self) -> None:
        bridge = OpenAIMemoryBridge()
        bridge.store_messages("thread-1", ["a"])
        bridge.store_messages("thread-2", ["b"])
        assert len(bridge.get_thread_memory("thread-1")) == 1
        assert len(bridge.get_thread_memory("thread-2")) == 1

    def test_retrieve_context_returns_list(self) -> None:
        bridge = OpenAIMemoryBridge()
        bridge.store_messages("thread-1", ["find me"])
        result = bridge.retrieve_context("thread-1", "find")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_retrieve_context_empty_when_no_match(self) -> None:
        bridge = OpenAIMemoryBridge()
        bridge.store_messages("thread-1", ["hello"])
        result = bridge.retrieve_context("thread-1", "zzz_no_match")
        assert result == []

    def test_get_thread_memory_returns_list(self) -> None:
        bridge = OpenAIMemoryBridge()
        bridge.store_messages("t1", ["msg1"])
        bridge.store_messages("t1", ["msg2"])
        memory = bridge.get_thread_memory("t1")
        assert isinstance(memory, list)
        assert len(memory) == 2

    def test_get_thread_memory_empty_for_unknown_thread(self) -> None:
        bridge = OpenAIMemoryBridge()
        assert bridge.get_thread_memory("nonexistent") == []


# ---------------------------------------------------------------------------
# AnthropicMemoryBridge
# ---------------------------------------------------------------------------


class TestAnthropicMemoryBridge:
    def test_construction_no_args(self) -> None:
        bridge = AnthropicMemoryBridge()
        assert bridge.memory_store is None
        assert bridge._items == []

    def test_construction_with_store(self) -> None:
        sentinel = object()
        bridge = AnthropicMemoryBridge(memory_store=sentinel)
        assert bridge.memory_store is sentinel

    def test_store_conversation_returns_str(self) -> None:
        bridge = AnthropicMemoryBridge()
        entry_id = bridge.store_conversation("conv-1", [{"role": "user", "content": "hi"}])
        assert isinstance(entry_id, str)

    def test_store_conversation_appends_item(self) -> None:
        bridge = AnthropicMemoryBridge()
        bridge.store_conversation("conv-1", ["a", "b"])
        assert len(bridge._items) == 1
        assert bridge._items[0]["conversation_id"] == "conv-1"

    def test_retrieve_context_returns_list(self) -> None:
        bridge = AnthropicMemoryBridge()
        bridge.store_conversation("conv-1", ["searchable content"])
        result = bridge.retrieve_context("searchable")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_retrieve_context_no_match(self) -> None:
        bridge = AnthropicMemoryBridge()
        bridge.store_conversation("conv-1", ["hello"])
        result = bridge.retrieve_context("zzz_no_match")
        assert result == []

    def test_get_session_memory_returns_list(self) -> None:
        bridge = AnthropicMemoryBridge()
        bridge.store_conversation("sess-1", ["a"])
        bridge.store_conversation("sess-2", ["b"])
        memory = bridge.get_session_memory("sess-1")
        assert isinstance(memory, list)
        assert len(memory) == 1

    def test_get_session_memory_empty_for_unknown(self) -> None:
        bridge = AnthropicMemoryBridge()
        assert bridge.get_session_memory("unknown") == []


# ---------------------------------------------------------------------------
# MicrosoftMemoryBridge
# ---------------------------------------------------------------------------


class TestMicrosoftMemoryBridge:
    def test_construction_no_args(self) -> None:
        bridge = MicrosoftMemoryBridge()
        assert bridge.memory_store is None
        assert bridge._items == []

    def test_construction_with_store(self) -> None:
        sentinel = object()
        bridge = MicrosoftMemoryBridge(memory_store=sentinel)
        assert bridge.memory_store is sentinel

    def test_store_turn_returns_str(self) -> None:
        bridge = MicrosoftMemoryBridge()
        entry_id = bridge.store_turn("conv-1", {"text": "Hello"})
        assert isinstance(entry_id, str)

    def test_store_turn_appends_item(self) -> None:
        bridge = MicrosoftMemoryBridge()
        bridge.store_turn("conv-1", "turn content")
        assert len(bridge._items) == 1
        assert bridge._items[0]["conversation_id"] == "conv-1"

    def test_retrieve_state_returns_dict(self) -> None:
        bridge = MicrosoftMemoryBridge()
        bridge.store_turn("conv-1", "first turn")
        bridge.store_turn("conv-1", "second turn")
        state = bridge.retrieve_state("conv-1")
        assert isinstance(state, dict)
        assert state["turn_count"] == 2
        assert state["conversation_id"] == "conv-1"

    def test_retrieve_state_empty_conversation(self) -> None:
        bridge = MicrosoftMemoryBridge()
        state = bridge.retrieve_state("unknown")
        assert state["turn_count"] == 0

    def test_get_conversation_memory_returns_list(self) -> None:
        bridge = MicrosoftMemoryBridge()
        bridge.store_turn("conv-1", "t1")
        bridge.store_turn("conv-1", "t2")
        bridge.store_turn("conv-2", "t3")
        memory = bridge.get_conversation_memory("conv-1")
        assert isinstance(memory, list)
        assert len(memory) == 2

    def test_get_conversation_memory_empty_for_unknown(self) -> None:
        bridge = MicrosoftMemoryBridge()
        assert bridge.get_conversation_memory("unknown") == []

    def test_all_classes_importable_from_init(self) -> None:
        from agent_memory.adapters import (
            AnthropicMemoryBridge,
            CrewAIMemoryBridge,
            LangChainMemoryBridge,
            MicrosoftMemoryBridge,
            OpenAIMemoryBridge,
        )
        assert LangChainMemoryBridge is not None
        assert CrewAIMemoryBridge is not None
        assert OpenAIMemoryBridge is not None
        assert AnthropicMemoryBridge is not None
        assert MicrosoftMemoryBridge is not None
