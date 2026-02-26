"""Configuration dataclass for UnifiedMemory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class MemoryConfig:
    """Configuration for a UnifiedMemory instance.

    Parameters
    ----------
    working_capacity:
        Maximum number of entries in the working memory buffer.
        Older entries are evicted when the buffer is full.
    episodic_max_items:
        Soft limit on the number of episodic memory entries.  When
        exceeded during consolidation, the oldest low-importance entries
        are promoted or discarded.  0 means no limit.
    decay_function:
        Freshness decay mode: ``"linear"``, ``"exponential"``, or
        ``"step"``.
    storage_backend:
        Which persistent storage backend to use: ``"sqlite"``,
        ``"redis"``, or ``"memory"`` (in-process, no persistence).
    sqlite_path:
        Path for the SQLite database file.  Ignored unless
        ``storage_backend == "sqlite"``.  Defaults to
        ``"agent_memory.db"`` in the current working directory.
    redis_host:
        Redis server hostname.  Ignored unless
        ``storage_backend == "redis"``.
    redis_port:
        Redis server port.  Defaults to 6379.
    redis_db:
        Redis database index.  Defaults to 0.
    redis_ttl_seconds:
        Per-entry TTL in Redis.  0 means no expiry.
    consolidation_interval:
        How many ``store()`` calls between automatic consolidation passes.
        0 disables automatic consolidation.
    importance_threshold:
        Entries with composite_score below this are candidates for
        garbage collection during consolidation.
    auto_score_importance:
        When True, the ImportanceScorer re-scores every entry on store().
    auto_score_freshness:
        When True, the FreshnessScorer re-scores every entry on store().
    auto_tag_provenance:
        When True, the ProvenanceTracker tags every entry on store().
    """

    working_capacity: int = 64
    episodic_max_items: int = 0
    decay_function: Literal["linear", "exponential", "step"] = "exponential"
    storage_backend: Literal["sqlite", "redis", "memory"] = "memory"
    sqlite_path: str = "agent_memory.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 0
    consolidation_interval: int = 100
    importance_threshold: float = 0.05
    auto_score_importance: bool = True
    auto_score_freshness: bool = True
    auto_tag_provenance: bool = True

    def __post_init__(self) -> None:
        if self.working_capacity < 1:
            raise ValueError(f"working_capacity must be >= 1, got {self.working_capacity}")
        if self.episodic_max_items < 0:
            raise ValueError(
                f"episodic_max_items must be >= 0, got {self.episodic_max_items}"
            )
        if self.consolidation_interval < 0:
            raise ValueError(
                f"consolidation_interval must be >= 0, got {self.consolidation_interval}"
            )
        if not 0.0 <= self.importance_threshold <= 1.0:
            raise ValueError(
                f"importance_threshold must be in [0, 1], got {self.importance_threshold}"
            )
        valid_decay = {"linear", "exponential", "step"}
        if self.decay_function not in valid_decay:
            raise ValueError(
                f"decay_function must be one of {sorted(valid_decay)!r}, "
                f"got {self.decay_function!r}"
            )
        valid_backends = {"sqlite", "redis", "memory"}
        if self.storage_backend not in valid_backends:
            raise ValueError(
                f"storage_backend must be one of {sorted(valid_backends)!r}, "
                f"got {self.storage_backend!r}"
            )


__all__ = ["MemoryConfig"]
