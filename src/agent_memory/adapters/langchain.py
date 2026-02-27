"""LangChain adapter for agent_memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class LangChainMemoryBridge:
    """Memory bridge adapter for LangChain.

    Bridges LangChain conversation messages to the agent_memory storage
    layer, enabling retrieval of past context without depending on the
    LangChain package.

    Usage::

        from agent_memory.adapters.langchain import LangChainMemoryBridge
        bridge = LangChainMemoryBridge()
    """

    def __init__(self, memory_store: Any = None) -> None:
        self.memory_store = memory_store
        self._items: list[dict[str, Any]] = []
        logger.info("LangChainMemoryBridge initialized.")

    def store_conversation(self, messages: list[Any]) -> str:
        """Persist a list of LangChain messages to the memory store.

        Returns a unique memory entry ID for later retrieval.
        """
        entry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": entry_id,
            "type": "conversation",
            "messages": [str(m) for m in messages],
            "count": len(messages),
        }
        self._items.append(entry)
        logger.debug("Stored conversation with id=%s, messages=%d", entry_id, len(messages))
        return entry_id

    def retrieve_context(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most recent k memory entries relevant to the query.

        Returns a list of memory entry dictionaries (naive recency ranking).
        """
        conversation_items = [item for item in self._items if item.get("type") == "conversation"]
        return conversation_items[-k:]

    def clear_memory(self) -> None:
        """Remove all stored memory entries."""
        self._items.clear()
        logger.info("LangChainMemoryBridge memory cleared.")
