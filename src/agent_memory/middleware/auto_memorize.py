"""AutoMemorizeMiddleware — automatically stores agent interactions as episodic memories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.unified.memory import UnifiedMemory


@dataclass
class Interaction:
    """A single agent interaction turn (user input + agent response)."""

    user_input: str
    agent_response: str
    session_id: str = ""
    turn_id: str = ""
    extra_metadata: dict[str, str] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AutoMemorizeMiddleware:
    """Middleware that automatically stores agent interactions as episodic memories.

    Wrap your agent's response loop with this middleware to build up a rich
    episodic memory history without any manual ``store()`` calls.

    Parameters
    ----------
    memory:
        The ``UnifiedMemory`` instance to write interactions into.
    source:
        The ``MemorySource`` to assign to auto-created entries.
        Defaults to ``USER_INPUT`` for user turns and ``AGENT_INFERENCE``
        for agent turns.
    store_user_turns:
        When True (default), user inputs are stored as episodic memories.
    store_agent_turns:
        When True (default), agent responses are stored as episodic memories.
    min_content_length:
        Interactions shorter than this character count are skipped.
        Defaults to 5.
    importance_boost:
        Fixed addition to the base importance score for auto-stored entries.
        Use a small positive value (e.g., 0.05) to slightly elevate
        interaction memories above heuristic baseline.
    """

    def __init__(
        self,
        memory: UnifiedMemory,
        store_user_turns: bool = True,
        store_agent_turns: bool = True,
        min_content_length: int = 5,
        importance_boost: float = 0.0,
    ) -> None:
        self._memory = memory
        self._store_user = store_user_turns
        self._store_agent = store_agent_turns
        self._min_length = min_content_length
        self._importance_boost = max(0.0, min(1.0, importance_boost))
        self._stored_count = 0

    @property
    def stored_count(self) -> int:
        """Total number of entries auto-stored by this middleware instance."""
        return self._stored_count

    def record(self, interaction: Interaction) -> list[MemoryEntry]:
        """Process one interaction turn and store relevant entries.

        Parameters
        ----------
        interaction:
            The interaction turn to record.

        Returns
        -------
        list[MemoryEntry]
            The stored entries (0–2 entries depending on config).
        """
        stored: list[MemoryEntry] = []

        if self._store_user and len(interaction.user_input) >= self._min_length:
            user_entry = self._make_entry(
                content=interaction.user_input,
                source=MemorySource.USER_INPUT,
                interaction=interaction,
                role="user",
            )
            stored.append(self._memory.store(user_entry))
            self._stored_count += 1

        if self._store_agent and len(interaction.agent_response) >= self._min_length:
            agent_entry = self._make_entry(
                content=interaction.agent_response,
                source=MemorySource.AGENT_INFERENCE,
                interaction=interaction,
                role="agent",
            )
            stored.append(self._memory.store(agent_entry))
            self._stored_count += 1

        return stored

    def record_raw(
        self,
        user_input: str,
        agent_response: str,
        session_id: str = "",
        turn_id: str = "",
        extra_metadata: Optional[dict[str, str]] = None,
    ) -> list[MemoryEntry]:
        """Convenience wrapper: build an Interaction and call ``record()``.

        Parameters
        ----------
        user_input:
            The user's message.
        agent_response:
            The agent's reply.
        session_id:
            Optional session identifier.
        turn_id:
            Optional turn counter or UUID.
        extra_metadata:
            Additional metadata to attach to both stored entries.

        Returns
        -------
        list[MemoryEntry]
            Stored entries.
        """
        interaction = Interaction(
            user_input=user_input,
            agent_response=agent_response,
            session_id=session_id,
            turn_id=turn_id,
            extra_metadata=extra_metadata or {},
        )
        return self.record(interaction)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        content: str,
        source: MemorySource,
        interaction: Interaction,
        role: str,
    ) -> MemoryEntry:
        """Build a MemoryEntry for an interaction turn."""
        metadata: dict[str, str] = {
            "role": role,
            "occurred_at": interaction.occurred_at.isoformat(),
        }
        if interaction.session_id:
            metadata["session_id"] = interaction.session_id
        if interaction.turn_id:
            metadata["turn_id"] = interaction.turn_id
        metadata.update(interaction.extra_metadata)

        entry = MemoryEntry(
            content=content,
            layer=MemoryLayer.EPISODIC,
            source=source,
            created_at=interaction.occurred_at,
            metadata=metadata,
        )

        if self._importance_boost > 0.0:
            boosted = min(1.0, entry.importance_score + self._importance_boost)
            entry = entry.model_copy(update={"importance_score": boosted})

        return entry


__all__ = ["AutoMemorizeMiddleware", "Interaction"]
