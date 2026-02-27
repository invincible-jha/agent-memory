"""Microsoft Agents adapter for agent_memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MicrosoftMemoryBridge:
    """Memory bridge adapter for Microsoft Agents.

    Stores per-turn conversation state keyed by conversation ID, aligning
    with the Bot Framework conversation state model.

    Usage::

        from agent_memory.adapters.microsoft_agents import MicrosoftMemoryBridge
        bridge = MicrosoftMemoryBridge()
    """

    def __init__(self, memory_store: Any = None) -> None:
        self.memory_store = memory_store
        self._items: list[dict[str, Any]] = []
        logger.info("MicrosoftMemoryBridge initialized.")

    def store_turn(self, conversation_id: str, turn: Any) -> str:
        """Persist a single conversation turn for a given conversation.

        Returns a unique memory entry ID.
        """
        entry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": entry_id,
            "type": "turn",
            "conversation_id": conversation_id,
            "turn": str(turn) if turn is not None else "",
        }
        self._items.append(entry)
        logger.debug("Stored turn for conversation_id=%s entry_id=%s", conversation_id, entry_id)
        return entry_id

    def retrieve_state(self, conversation_id: str) -> dict[str, Any]:
        """Retrieve the aggregated state for a conversation.

        Returns a dict with turn count and the most recent turn text.
        """
        turns = [item for item in self._items if item.get("conversation_id") == conversation_id]
        latest_turn = turns[-1].get("turn", "") if turns else ""
        return {
            "conversation_id": conversation_id,
            "turn_count": len(turns),
            "latest_turn": latest_turn,
        }

    def get_conversation_memory(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return all stored turn entries for the given conversation."""
        return [item for item in self._items if item.get("conversation_id") == conversation_id]
