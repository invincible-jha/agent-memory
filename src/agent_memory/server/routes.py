"""Route handler functions for the agent-memory HTTP server.

Each function accepts parsed request data and returns a tuple of
(status_code, response_dict). The HTTP handler in app.py calls these
functions and serializes the results to JSON.
"""
from __future__ import annotations


from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource
from agent_memory.server.models import (
    CreateMemoryRequest,
    DeleteMemoryResponse,
    ErrorResponse,
    HealthResponse,
    MemoryResponse,
    SearchResponse,
)


# Module-level in-memory store (reset between tests)
_memory_store: dict[str, MemoryEntry] = {}


def reset_state() -> None:
    """Reset all shared state — used in tests and for clean restarts."""
    global _memory_store
    _memory_store = {}


def _entry_to_response(entry: MemoryEntry) -> MemoryResponse:
    """Convert a MemoryEntry to a MemoryResponse."""
    return MemoryResponse(
        memory_id=entry.memory_id,
        content=entry.content,
        layer=entry.layer.value,
        importance_score=entry.importance_score,
        freshness_score=entry.freshness_score,
        source=entry.source.value,
        safety_critical=entry.safety_critical,
        composite_score=entry.composite_score,
        importance_level=entry.importance_level.value,
        created_at=entry.created_at.isoformat(),
        last_accessed=entry.last_accessed.isoformat(),
        access_count=entry.access_count,
        metadata=entry.metadata,
    )


def handle_create_memory(body: dict[str, object]) -> tuple[int, dict[str, object]]:
    """Handle POST /memories.

    Parameters
    ----------
    body:
        Parsed JSON request body.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    try:
        request = CreateMemoryRequest.model_validate(body)
    except Exception as exc:
        return 422, ErrorResponse(error="Validation error", detail=str(exc)).model_dump()

    try:
        layer = MemoryLayer(request.layer)
    except ValueError:
        return 422, ErrorResponse(
            error="Invalid layer",
            detail=f"Layer {request.layer!r} is not valid. "
            f"Choose from: {[l.value for l in MemoryLayer]}",
        ).model_dump()

    try:
        source = MemorySource(request.source)
    except ValueError:
        source = MemorySource.AGENT_INFERENCE

    entry = MemoryEntry(
        content=request.content,
        layer=layer,
        importance_score=request.importance_score,
        source=source,
        safety_critical=request.safety_critical,
        metadata=request.metadata,
    )
    _memory_store[entry.memory_id] = entry

    return 201, _entry_to_response(entry).model_dump()


def handle_search_memories(
    query: str,
    layer: str | None = None,
    limit: int = 20,
) -> tuple[int, dict[str, object]]:
    """Handle GET /search?q=...

    Performs case-insensitive substring matching against memory content.

    Parameters
    ----------
    query:
        Search term.
    layer:
        Optional layer filter.
    limit:
        Maximum results to return.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    query_lower = query.lower()
    results: list[MemoryEntry] = []

    for entry in _memory_store.values():
        if query_lower and query_lower not in entry.content.lower():
            continue
        if layer and entry.layer.value != layer:
            continue
        results.append(entry)

    # Sort by composite_score descending
    results.sort(key=lambda e: e.composite_score, reverse=True)
    results = results[:limit]

    response = SearchResponse(
        query=query,
        results=[_entry_to_response(e) for e in results],
        total=len(results),
    )
    return 200, response.model_dump()


def handle_delete_memory(memory_id: str) -> tuple[int, dict[str, object]]:
    """Handle DELETE /memories/{id}.

    Parameters
    ----------
    memory_id:
        The memory identifier from the URL path.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    if memory_id not in _memory_store:
        return 404, ErrorResponse(
            error="Not found",
            detail=f"Memory {memory_id!r} not found.",
        ).model_dump()

    del _memory_store[memory_id]
    response = DeleteMemoryResponse(memory_id=memory_id, deleted=True)
    return 200, response.model_dump()


def handle_get_memory(memory_id: str) -> tuple[int, dict[str, object]]:
    """Handle GET /memories/{id}.

    Parameters
    ----------
    memory_id:
        The memory identifier.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    entry = _memory_store.get(memory_id)
    if entry is None:
        return 404, ErrorResponse(
            error="Not found",
            detail=f"Memory {memory_id!r} not found.",
        ).model_dump()

    # Touch access count
    entry = entry.touch()
    _memory_store[memory_id] = entry

    return 200, _entry_to_response(entry).model_dump()


def handle_health() -> tuple[int, dict[str, object]]:
    """Handle GET /health.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    response = HealthResponse(memory_count=len(_memory_store))
    return 200, response.model_dump()


__all__ = [
    "reset_state",
    "handle_create_memory",
    "handle_search_memories",
    "handle_delete_memory",
    "handle_get_memory",
    "handle_health",
]
