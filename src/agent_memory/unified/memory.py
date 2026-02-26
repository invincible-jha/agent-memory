"""UnifiedMemory — orchestrates all 4 cognitive layers with storage and scoring."""

from __future__ import annotations

from typing import Optional, Sequence

from agent_memory.freshness.scorer import FreshnessScorer
from agent_memory.importance.gc import MemoryGarbageCollector
from agent_memory.importance.scorer import ImportanceScorer
from agent_memory.memory.base import MemoryStore
from agent_memory.memory.episodic import EpisodicMemory
from agent_memory.memory.procedural import ProceduralMemory
from agent_memory.memory.semantic import SemanticMemory
from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.memory.working import WorkingMemory
from agent_memory.provenance.tracker import ProvenanceTracker
from agent_memory.storage.base import StorageBackend
from agent_memory.unified.config import MemoryConfig


def _build_storage_backend(config: MemoryConfig) -> StorageBackend:
    """Construct the appropriate StorageBackend from the config."""
    if config.storage_backend == "sqlite":
        from agent_memory.storage.sqlite_store import SQLiteStorage

        return SQLiteStorage(db_path=config.sqlite_path)

    if config.storage_backend == "redis":
        from agent_memory.storage.redis_store import RedisStorage

        return RedisStorage(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            ttl_seconds=config.redis_ttl_seconds,
        )

    # Default: in-memory
    from agent_memory.storage.memory_store import InMemoryStorage

    return InMemoryStorage()


class UnifiedMemory:
    """Orchestrator that ties together the 4 cognitive memory layers.

    Provides a single entry-point for storing, recalling, and managing
    memories across working, episodic, semantic, and procedural layers.
    Scoring (importance, freshness) and provenance tagging are applied
    automatically on every ``store()`` call based on configuration.

    Parameters
    ----------
    config:
        Configuration dataclass; a default ``MemoryConfig`` is used if
        not provided.
    storage:
        Explicit storage backend; overrides ``config.storage_backend``
        when provided.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        self._config = config or MemoryConfig()
        self._storage = storage or _build_storage_backend(self._config)

        # Cognitive layers
        self._working = WorkingMemory(capacity=self._config.working_capacity)
        self._episodic = EpisodicMemory()
        self._semantic = SemanticMemory()
        self._procedural = ProceduralMemory()

        # Subsystems
        self._importance_scorer = ImportanceScorer()
        self._freshness_scorer = FreshnessScorer()
        self._provenance_tracker = ProvenanceTracker()

        self._store_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> MemoryConfig:
        return self._config

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Store a memory entry across the appropriate cognitive layer and backend.

        Scoring and provenance tagging are applied according to config before
        the entry is written. Automatic consolidation is triggered when the
        ``consolidation_interval`` threshold is met.

        Parameters
        ----------
        entry:
            The memory entry to store.

        Returns
        -------
        MemoryEntry
            The final (possibly score-updated) entry that was persisted.
        """
        processed = self._apply_processing(entry)

        # Route to the correct cognitive layer
        layer_store = self._layer_store(processed.layer)
        layer_store.store(processed)

        # Persist to durable backend
        self._storage.save(processed)

        self._store_count += 1
        if (
            self._config.consolidation_interval > 0
            and self._store_count % self._config.consolidation_interval == 0
        ):
            self.consolidate()

        return processed

    def recall(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single entry by memory_id.

        Checks in-process layers first (fast path), then falls back to the
        durable storage backend.  The entry's access metadata is updated.

        Parameters
        ----------
        memory_id:
            The ID of the entry to retrieve.

        Returns
        -------
        MemoryEntry | None
            The entry, or None if not found anywhere.
        """
        # Fast path: in-process layers
        for layer_store in self._all_layer_stores():
            entry = layer_store.retrieve(memory_id)
            if entry is not None:
                touched = entry.touch()
                layer_store.store(touched)
                self._storage.save(touched)
                return touched

        # Slow path: durable storage
        entry = self._storage.load(memory_id)
        if entry is not None:
            touched = entry.touch()
            self._storage.save(touched)
            return touched

        return None

    def forget(self, memory_id: str) -> bool:
        """Remove an entry from all layers and durable storage.

        Parameters
        ----------
        memory_id:
            The ID of the entry to remove.

        Returns
        -------
        bool
            True if the entry was found and removed from at least one location.
        """
        removed = False
        for layer_store in self._all_layer_stores():
            if layer_store.delete(memory_id):
                removed = True
        if self._storage.delete(memory_id):
            removed = True
        return removed

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Search for entries matching a query string.

        Searches the durable storage backend (which may support FTS), then
        de-duplicates against entries returned from in-process layers for
        completeness.

        Parameters
        ----------
        query:
            Free-text query string.
        layer:
            Optional layer filter.
        limit:
            Maximum number of results.

        Returns
        -------
        Sequence[MemoryEntry]
            Matching entries, ordered by backend relevance.
        """
        backend_results = list(self._storage.search(query, layer=layer, limit=limit))
        seen_ids = {e.memory_id for e in backend_results}

        # Include any in-process matches not yet in storage
        tokens = query.lower().split()
        for layer_store in self._all_layer_stores():
            for entry in layer_store.all(layer=layer):
                if entry.memory_id in seen_ids:
                    continue
                if all(token in entry.content.lower() for token in tokens):
                    backend_results.append(entry)
                    seen_ids.add(entry.memory_id)
                if len(backend_results) >= limit:
                    break

        return backend_results[:limit]

    def consolidate(self) -> dict[str, int]:
        """Run a garbage-collection pass across all in-process layers.

        Removes entries whose effective importance (importance * decay) falls
        below the configured threshold.  Safety-critical entries are never
        removed.

        Returns
        -------
        dict[str, int]
            Summary with keys ``"collected"`` and ``"retained"``.
        """
        collected = 0
        for layer_store in self._all_layer_stores():
            gc = MemoryGarbageCollector(
                store=layer_store,
                threshold=self._config.importance_threshold,
            )
            evicted_ids = gc.collect(dry_run=False)
            collected += len(evicted_ids)
            # Sync deletions to durable storage
            for memory_id in evicted_ids:
                self._storage.delete(memory_id)

        total = sum(s.count() for s in self._all_layer_stores())
        return {"collected": collected, "retained": total}

    def stats(self) -> dict[str, object]:
        """Return aggregate statistics about the current memory state.

        Returns
        -------
        dict[str, object]
            Contains total counts per layer, overall totals, and backend type.
        """
        layer_counts: dict[str, int] = {}
        total_in_process = 0
        for layer in MemoryLayer:
            layer_store = self._layer_store(layer)
            layer_counts[layer.value] = layer_store.count()
            total_in_process += layer_counts[layer.value]

        storage_total = self._storage.count()

        return {
            "layer_counts": layer_counts,
            "total_in_process": total_in_process,
            "total_in_storage": storage_total,
            "store_calls": self._store_count,
            "working_capacity": self._config.working_capacity,
            "working_utilisation": round(
                layer_counts.get("working", 0) / max(1, self._config.working_capacity), 3
            ),
            "storage_backend": self._config.storage_backend,
            "decay_function": self._config.decay_function,
        }

    # ------------------------------------------------------------------
    # Layer routing helpers
    # ------------------------------------------------------------------

    def _layer_store(self, layer: MemoryLayer) -> MemoryStore:
        """Return the in-process MemoryStore for the given layer."""
        mapping: dict[MemoryLayer, MemoryStore] = {
            MemoryLayer.WORKING: self._working,
            MemoryLayer.EPISODIC: self._episodic,
            MemoryLayer.SEMANTIC: self._semantic,
            MemoryLayer.PROCEDURAL: self._procedural,
        }
        return mapping[layer]

    def _all_layer_stores(self) -> list[MemoryStore]:
        return [self._working, self._episodic, self._semantic, self._procedural]

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _apply_processing(self, entry: MemoryEntry) -> MemoryEntry:
        """Apply scoring and provenance tagging in the configured order."""
        result = entry
        if self._config.auto_score_importance:
            result = self._importance_scorer.score_and_update(result)
        if self._config.auto_score_freshness:
            result = self._freshness_scorer.score_and_update(result)
        if self._config.auto_tag_provenance:
            result = self._provenance_tracker.tag(result)
        return result


__all__ = ["UnifiedMemory"]
