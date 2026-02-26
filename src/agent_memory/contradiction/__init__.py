"""Contradiction detection, reporting, resolution, and scanning."""

from __future__ import annotations

from agent_memory.contradiction.detector import ContradictionDetector
from agent_memory.contradiction.matcher import EntityAttributeMatcher, Triple
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.contradiction.resolver import (
    ContradictionResolver,
    ResolutionResult,
    ResolutionStrategy,
)
from agent_memory.contradiction.scanner import ContradictionScanner, ScanResult, ScanScope

__all__ = [
    "ContradictionDetector",
    "EntityAttributeMatcher",
    "Triple",
    "ContradictionPair",
    "ContradictionReport",
    "ContradictionResolver",
    "ResolutionResult",
    "ResolutionStrategy",
    "ContradictionScanner",
    "ScanResult",
    "ScanScope",
]
