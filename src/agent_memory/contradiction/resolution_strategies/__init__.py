"""Contradiction resolution strategy implementations (subpackage).

This subpackage exposes resolution strategies for use with
:class:`~agent_memory.contradiction.resolution.ContradictionResolver`.

Available strategies
--------------------
- :class:`LatestWinsStrategy`       — use the most recent memory entry
- :class:`UserConfirmationStrategy` — defer the decision to the user
- :class:`MergeResolutionStrategy`  — merge non-conflicting content parts
"""
from __future__ import annotations

from agent_memory.contradiction.resolution_strategies.latest_wins import LatestWinsStrategy
from agent_memory.contradiction.resolution_strategies.merge import MergeResolutionStrategy
from agent_memory.contradiction.resolution_strategies.user_confirmation import UserConfirmationStrategy

__all__ = [
    "LatestWinsStrategy",
    "MergeResolutionStrategy",
    "UserConfirmationStrategy",
]
