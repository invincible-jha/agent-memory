"""OpenAI Agents SDK adapter for agent_memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIMemoryBridge:
    """Memory bridge adapter for the OpenAI Agents SDK.

    Stores and retrieves message history keyed by thread ID, mirroring the
    thread-based memory model of the OpenAI Assistants API.

    Usage::

        from agent_memory.adapters.openai_agents import OpenAIMemoryBridge
        bridge = OpenAIMemoryBridge()
    """

    def __init__(self, memory_store: Any = None) -> None:
        self.memory_store = memory_store
        self._items: list[dict[str, Any]] = []
        logger.info("OpenAIMemoryBridge initialized.")

    def store_messages(self, thread_id: str, messages: list[Any]) -> str:
        """Persist a list of messages for a given thread.

        Returns a unique memory entry ID.
        """
        entry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": entry_id,
            "type": "thread_messages",
            "thread_id": thread_id,
            "messages": [str(m) for m in messages],
            "count": len(messages),
        }
        self._items.append(entry)
        logger.debug("Stored %d messages for thread=%s id=%s", len(messages), thread_id, entry_id)
        return entry_id

    def retrieve_context(self, thread_id: str, query: str) -> list[dict[str, Any]]:
        """Retrieve message entries for a thread that contain the query string.

        Returns matching entries in insertion order.
        """
        results: list[dict[str, Any]] = []
        for item in self._items:
            if item.get("thread_id") != thread_id:
                continue
            if any(query.lower() in msg.lower() for msg in item.get("messages", [])):
                results.append(item)
        return results

    def get_thread_memory(self, thread_id: str) -> list[dict[str, Any]]:
        """Return all stored memory entries for the given thread."""
        return [item for item in self._items if item.get("thread_id") == thread_id]
