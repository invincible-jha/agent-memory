"""Anthropic SDK adapter for agent_memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicMemoryBridge:
    """Memory bridge adapter for the Anthropic SDK.

    Stores conversation turns keyed by conversation ID and provides context
    retrieval across Anthropic messages API sessions.

    Usage::

        from agent_memory.adapters.anthropic_sdk import AnthropicMemoryBridge
        bridge = AnthropicMemoryBridge()
    """

    def __init__(self, memory_store: Any = None) -> None:
        self.memory_store = memory_store
        self._items: list[dict[str, Any]] = []
        logger.info("AnthropicMemoryBridge initialized.")

    def store_conversation(self, conversation_id: str, messages: list[Any]) -> str:
        """Persist a conversation's messages under a conversation ID.

        Returns a unique memory entry ID.
        """
        entry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": entry_id,
            "type": "conversation",
            "conversation_id": conversation_id,
            "messages": [str(m) for m in messages],
            "count": len(messages),
        }
        self._items.append(entry)
        logger.debug(
            "Stored conversation conversation_id=%s entry_id=%s messages=%d",
            conversation_id,
            entry_id,
            len(messages),
        )
        return entry_id

    def retrieve_context(self, query: str) -> list[dict[str, Any]]:
        """Retrieve memory entries whose messages contain the query string.

        Returns matching entries across all conversations.
        """
        results: list[dict[str, Any]] = []
        for item in self._items:
            if any(query.lower() in msg.lower() for msg in item.get("messages", [])):
                results.append(item)
        return results

    def get_session_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Return all stored entries for the given session / conversation ID."""
        return [item for item in self._items if item.get("conversation_id") == session_id]
