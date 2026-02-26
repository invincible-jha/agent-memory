"""Redis storage backend — guarded import, requires redis-py."""

from __future__ import annotations

try:
    import redis as redis_lib
except ImportError as _redis_import_error:  # pragma: no cover
    raise ImportError(
        "RedisStorage requires the 'redis' package. "
        "Install it with: pip install redis"
    ) from _redis_import_error

import json
from datetime import datetime, timezone
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.storage.base import StorageBackend

# Key prefixes used in Redis
_PREFIX = "agent_memory"
_ENTRY_PREFIX = f"{_PREFIX}:entry"
_LAYER_SET_PREFIX = f"{_PREFIX}:layer"
_ALL_KEYS_SET = f"{_PREFIX}:all_keys"


def _entry_key(memory_id: str) -> str:
    return f"{_ENTRY_PREFIX}:{memory_id}"


def _layer_set_key(layer: MemoryLayer) -> str:
    return f"{_LAYER_SET_PREFIX}:{layer.value}"


class RedisStorage(StorageBackend):
    """Redis-backed storage using hashes for entries and sets for indexing.

    Each memory entry is stored as a Redis hash at key
    ``agent_memory:entry:<memory_id>``. Two index structures are maintained:
    - A global set ``agent_memory:all_keys`` containing all memory_ids.
    - Per-layer sets ``agent_memory:layer:<layer>`` for layer-scoped queries.

    Parameters
    ----------
    host:
        Redis server hostname. Defaults to ``"localhost"``.
    port:
        Redis server port. Defaults to ``6379``.
    db:
        Redis database index. Defaults to ``0``.
    password:
        Optional Redis password.
    ttl_seconds:
        Optional TTL for each entry in seconds. 0 means no expiry.
    client:
        Pre-constructed redis.Redis client (overrides host/port/db/password).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 0,
        client: Optional[redis_lib.Redis] = None,  # type: ignore[type-arg]
    ) -> None:
        if client is not None:
            self._client: redis_lib.Redis = client  # type: ignore[type-arg]
        else:
            self._client = redis_lib.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> None:
        """Persist an entry as a Redis hash with optional TTL."""
        entry_key = _entry_key(entry.memory_id)
        serialised = _serialise(entry)

        pipeline = self._client.pipeline()
        pipeline.hset(entry_key, mapping=serialised)  # type: ignore[arg-type]
        if self._ttl > 0:
            pipeline.expire(entry_key, self._ttl)
        pipeline.sadd(_ALL_KEYS_SET, entry.memory_id)
        pipeline.sadd(_layer_set_key(entry.layer), entry.memory_id)
        pipeline.execute()

    def load(self, key: str) -> Optional[MemoryEntry]:
        """Load an entry by memory_id, returning None if not found."""
        data = self._client.hgetall(_entry_key(key))
        if not data:
            return None
        return _deserialise(data)  # type: ignore[arg-type]

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        entry_key = _entry_key(key)
        # Load to find the layer before deleting
        data = self._client.hgetall(entry_key)
        if not data:
            return False
        layer_value = data.get("layer", "")

        pipeline = self._client.pipeline()
        pipeline.delete(entry_key)
        pipeline.srem(_ALL_KEYS_SET, key)
        if layer_value:
            try:
                layer = MemoryLayer(layer_value)
                pipeline.srem(_layer_set_key(layer), key)
            except ValueError:
                pass
        pipeline.execute()
        return True

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """Substring search across all entries (scans content field)."""
        entries = self.load_all(layer=layer, limit=limit * 10)
        query_lower = query.lower()
        tokens = query_lower.split()
        results: list[MemoryEntry] = []
        for entry in entries:
            content_lower = entry.content.lower()
            if all(token in content_lower for token in tokens):
                results.append(entry)
            if len(results) >= limit:
                break
        return results

    def list_keys(
        self,
        layer: Optional[MemoryLayer] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Return stored memory_id values, up to ``limit``."""
        if layer is None:
            all_keys = self._client.smembers(_ALL_KEYS_SET)
        else:
            all_keys = self._client.smembers(_layer_set_key(layer))
        # smembers returns a set — sort for deterministic ordering
        return sorted(all_keys)[:limit]  # type: ignore[return-value]

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        """Remove entries. Returns count deleted."""
        if layer is None:
            all_ids: list[str] = list(self._client.smembers(_ALL_KEYS_SET))  # type: ignore[arg-type]
        else:
            all_ids = list(self._client.smembers(_layer_set_key(layer)))  # type: ignore[arg-type]

        if not all_ids:
            return 0

        pipeline = self._client.pipeline()
        for memory_id in all_ids:
            pipeline.delete(_entry_key(memory_id))
        pipeline.execute()

        # Rebuild index sets
        if layer is None:
            self._client.delete(_ALL_KEYS_SET)
            # Also clear all layer sets
            for lyr in MemoryLayer:
                self._client.delete(_layer_set_key(lyr))
        else:
            # Remove cleared IDs from global and layer sets
            pipeline = self._client.pipeline()
            for memory_id in all_ids:
                pipeline.srem(_ALL_KEYS_SET, memory_id)
            pipeline.delete(_layer_set_key(layer))
            pipeline.execute()

        return len(all_ids)

    def ping(self) -> bool:
        """Return True if the Redis server is reachable."""
        try:
            return bool(self._client.ping())
        except redis_lib.ConnectionError:
            return False


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------


def _serialise(entry: MemoryEntry) -> dict[str, str]:
    """Convert a MemoryEntry to a flat dict of strings for Redis hset."""
    return {
        "memory_id": entry.memory_id,
        "content": entry.content,
        "layer": entry.layer.value,
        "importance_score": str(entry.importance_score),
        "freshness_score": str(entry.freshness_score),
        "source": entry.source.value,
        "created_at": entry.created_at.isoformat(),
        "last_accessed": entry.last_accessed.isoformat(),
        "access_count": str(entry.access_count),
        "safety_critical": "1" if entry.safety_critical else "0",
        "metadata": json.dumps(entry.metadata),
    }


def _parse_dt(value: str) -> datetime:
    """Parse an ISO 8601 datetime string, attaching UTC if naive."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _deserialise(data: dict[str, str]) -> MemoryEntry:
    """Reconstruct a MemoryEntry from a Redis hash dict."""
    metadata: dict[str, str] = json.loads(data.get("metadata", "{}"))
    return MemoryEntry(
        memory_id=data["memory_id"],
        content=data["content"],
        layer=MemoryLayer(data["layer"]),
        importance_score=float(data.get("importance_score", "0.5")),
        freshness_score=float(data.get("freshness_score", "1.0")),
        source=MemorySource(data.get("source", "agent_inference")),
        created_at=_parse_dt(data["created_at"]),
        last_accessed=_parse_dt(data["last_accessed"]),
        access_count=int(data.get("access_count", "0")),
        safety_critical=data.get("safety_critical", "0") == "1",
        metadata=metadata,
    )


__all__ = ["RedisStorage"]
