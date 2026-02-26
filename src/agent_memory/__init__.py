"""agent-memory — Agent memory and knowledge management with 4-layer cognitive architecture.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> from agent_memory import UnifiedMemory, MemoryEntry, MemoryLayer
>>> mem = UnifiedMemory()
>>> entry = MemoryEntry(content="The sky is blue", layer=MemoryLayer.SEMANTIC)
>>> mem.store(entry)
>>> results = mem.search("sky")
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__: str = "0.1.0"

# ---------------------------------------------------------------------------
# Core memory types
# ---------------------------------------------------------------------------

from agent_memory.memory.types import (
    ImportanceLevel,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
)

# ---------------------------------------------------------------------------
# Cognitive layer stores
# ---------------------------------------------------------------------------

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.procedural import Procedure, ProceduralMemory, ProcedureStep
from agent_memory.memory.semantic import SemanticMemory
from agent_memory.memory.working import WorkingMemory

# ---------------------------------------------------------------------------
# Scoring subsystems
# ---------------------------------------------------------------------------

from agent_memory.freshness.decay import FreshnessDecay
from agent_memory.freshness.scorer import FreshnessScorer
from agent_memory.freshness.validator import StaleMemoryDetector
from agent_memory.importance.decay import DecayCurve, ExponentialDecay, LinearDecay, StepDecay
from agent_memory.importance.gc import MemoryGarbageCollector
from agent_memory.importance.safety import SafetyMemoryProtector
from agent_memory.importance.scorer import ImportanceScorer

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

from agent_memory.provenance.reliability import SourceReliability
from agent_memory.provenance.tracker import ProvenanceTracker

# ---------------------------------------------------------------------------
# Contradiction detection and resolution
# ---------------------------------------------------------------------------

from agent_memory.contradiction.detector import ContradictionDetector
from agent_memory.contradiction.matcher import EntityAttributeMatcher, Triple
from agent_memory.contradiction.report import ContradictionPair, ContradictionReport
from agent_memory.contradiction.resolver import (
    ContradictionResolver,
    ResolutionResult,
    ResolutionStrategy,
)
from agent_memory.contradiction.scanner import ContradictionScanner, ScanResult, ScanScope

# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

from agent_memory.storage.base import StorageBackend
from agent_memory.storage.memory_store import InMemoryStorage
from agent_memory.storage.sqlite_store import SQLiteStorage

# ---------------------------------------------------------------------------
# Unified orchestrator
# ---------------------------------------------------------------------------

from agent_memory.unified.config import MemoryConfig
from agent_memory.unified.memory import UnifiedMemory

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

from agent_memory.retrieval.ranking import RankingWeights, ResultRanker
from agent_memory.retrieval.search import MemorySearchEngine, SearchResult

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

from agent_memory.middleware.auto_memorize import AutoMemorizeMiddleware, Interaction
from agent_memory.middleware.context_builder import ContextBuilder, ContextSection

# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------

from agent_memory.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Guarded optional exports
# ---------------------------------------------------------------------------

try:
    from agent_memory.storage.redis_store import RedisStorage as RedisStorage
except ImportError:
    pass  # redis not installed

__all__ = [
    # Version
    "__version__",
    # Core types
    "MemoryEntry",
    "MemoryLayer",
    "MemorySource",
    "ImportanceLevel",
    # Cognitive layer stores
    "MemoryStore",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "Procedure",
    "ProcedureStep",
    # Importance subsystem
    "ImportanceScorer",
    "DecayCurve",
    "LinearDecay",
    "ExponentialDecay",
    "StepDecay",
    "MemoryGarbageCollector",
    "SafetyMemoryProtector",
    # Freshness subsystem
    "FreshnessScorer",
    "FreshnessDecay",
    "StaleMemoryDetector",
    # Provenance
    "ProvenanceTracker",
    "SourceReliability",
    # Contradiction
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
    # Storage backends
    "StorageBackend",
    "InMemoryStorage",
    "SQLiteStorage",
    # Unified
    "UnifiedMemory",
    "MemoryConfig",
    # Retrieval
    "MemorySearchEngine",
    "SearchResult",
    "ResultRanker",
    "RankingWeights",
    # Middleware
    "AutoMemorizeMiddleware",
    "Interaction",
    "ContextBuilder",
    "ContextSection",
    # Plugins
    "PluginRegistry",
]
