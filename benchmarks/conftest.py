"""Shared fixtures for agent-memory benchmark scripts.

These helpers are imported directly by benchmark scripts (not via pytest).
They set up pre-populated InMemoryStorage instances at different sizes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package source and benchmarks datasets directory are on sys.path
_REPO_ROOT = Path(__file__).parent.parent
_SRC = _REPO_ROOT / "src"
_BENCHMARKS = _REPO_ROOT / "benchmarks"
for _path in [str(_SRC), str(_BENCHMARKS)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent_memory.memory.types import MemoryLayer
from agent_memory.storage.memory_store import InMemoryStorage
from datasets.synthetic_memories import generate_memory_dataset


def build_store(n_memories: int, seed: int = 42) -> InMemoryStorage:
    """Return an InMemoryStorage pre-populated with n_memories entries.

    Parameters
    ----------
    n_memories:
        Number of MemoryEntry objects to insert.
    seed:
        Fixed seed for reproducibility.

    Returns
    -------
    InMemoryStorage
        Populated store ready for benchmarking.
    """
    entries, _ = generate_memory_dataset(
        n_memories=n_memories,
        seed=seed,
        layer=MemoryLayer.SEMANTIC,
    )
    store = InMemoryStorage()
    for entry in entries:
        store.save(entry)
    return store


__all__ = ["build_store"]
