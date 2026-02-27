"""CrewAI adapter for agent_memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class CrewAIMemoryBridge:
    """Memory bridge adapter for CrewAI.

    Stores task results and agent knowledge produced during a CrewAI crew
    execution, making them available for cross-task retrieval.

    Usage::

        from agent_memory.adapters.crewai import CrewAIMemoryBridge
        bridge = CrewAIMemoryBridge()
    """

    def __init__(self, memory_store: Any = None) -> None:
        self.memory_store = memory_store
        self._items: list[dict[str, Any]] = []
        logger.info("CrewAIMemoryBridge initialized.")

    def store_task_result(self, task_name: str, result: Any) -> str:
        """Persist a task result to the memory store.

        Returns a unique memory entry ID.
        """
        entry_id = str(uuid.uuid4())
        entry: dict[str, Any] = {
            "id": entry_id,
            "type": "task_result",
            "task_name": task_name,
            "result": str(result) if result is not None else "",
        }
        self._items.append(entry)
        logger.debug("Stored task result for task=%s id=%s", task_name, entry_id)
        return entry_id

    def retrieve_knowledge(self, query: str) -> list[dict[str, Any]]:
        """Retrieve stored knowledge entries that match the query.

        Returns task result entries (naive full-scan; replace with vector
        search in production).
        """
        results: list[dict[str, Any]] = []
        for item in self._items:
            if query.lower() in item.get("task_name", "").lower() or query.lower() in item.get("result", "").lower():
                results.append(item)
        return results

    def get_crew_memory(self) -> list[dict[str, Any]]:
        """Return all stored memory entries for the current crew run."""
        return list(self._items)
