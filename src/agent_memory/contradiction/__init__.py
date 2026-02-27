"""Contradiction detection, reporting, resolution, and scanning.

Original symbols (backward-compatible):
    ContradictionDetector, EntityAttributeMatcher, Triple,
    ContradictionPair, ContradictionReport,
    ContradictionResolver, ResolutionResult, ResolutionStrategy,
    ContradictionScanner, ScanResult, ScanScope

New TF-IDF engine symbols:
    TFIDFContradictionDetector,
    SimpleContradictionResolver,
    MemoryEntry (lightweight frozen dataclass, from contradiction.models),
    Contradiction, Resolution, ContradictionType,
    KeepNewerStrategy, KeepBothStrategy, FlagForReviewStrategy, MergeStrategy
"""

from __future__ import annotations

from agent_memory.contradiction.detector import ContradictionDetector, TFIDFContradictionDetector
from agent_memory.contradiction.matcher import EntityAttributeMatcher, Triple
from agent_memory.contradiction.models import (
    Contradiction,
    ContradictionType,
    Resolution,
)
from agent_memory.contradiction.models import MemoryEntry as LightweightMemoryEntry
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.contradiction.resolver import (
    ContradictionResolver,
    ResolutionResult,
    ResolutionStrategy,
    SimpleContradictionResolver,
)
from agent_memory.contradiction.scanner import ContradictionScanner, ScanResult, ScanScope
from agent_memory.contradiction.strategies import (
    FlagForReviewStrategy,
    KeepBothStrategy,
    KeepNewerStrategy,
    MergeStrategy,
    ResolutionStrategyProtocol,
)

__all__ = [
    # Original symbols
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
    # New TF-IDF engine symbols
    "TFIDFContradictionDetector",
    "SimpleContradictionResolver",
    "LightweightMemoryEntry",
    "Contradiction",
    "Resolution",
    "ContradictionType",
    "KeepNewerStrategy",
    "KeepBothStrategy",
    "FlagForReviewStrategy",
    "MergeStrategy",
    "ResolutionStrategyProtocol",
]
