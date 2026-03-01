#!/usr/bin/env python3
"""Example: Advanced Configuration

Demonstrates configuring UnifiedMemory with custom storage backends,
importance scoring, and freshness decay settings.

Usage:
    python examples/02_configuration.py

Requirements:
    pip install agent-memory
"""
from __future__ import annotations

import agent_memory
from agent_memory import (
    UnifiedMemory,
    MemoryConfig,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    ImportanceLevel,
    InMemoryStorage,
    SQLiteStorage,
    ImportanceScorer,
    FreshnessScorer,
)
import tempfile
import os


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    # Step 1: Configure memory with custom settings
    config = MemoryConfig(
        max_working_memory_entries=50,
        max_episodic_entries=500,
        freshness_decay_hours=24.0,
        importance_threshold=0.3,
    )
    print(f"Config: max_working={config.max_working_memory_entries}, "
          f"decay_hours={config.freshness_decay_hours}")

    # Step 2: Use SQLite storage for persistence
    temp_dir = tempfile.mkdtemp(prefix="agent_memory_")
    db_path = os.path.join(temp_dir, "memory.db")
    storage = SQLiteStorage(db_path=db_path)
    print(f"\nSQLite storage: {db_path}")

    # Step 3: Create UnifiedMemory with custom config + storage
    unified = UnifiedMemory(config=config, storage=storage)

    # Step 4: Store entries with explicit importance
    entries: list[MemoryEntry] = [
        MemoryEntry(
            content="Critical: system is in maintenance mode",
            layer=MemoryLayer.WORKING,
            source=MemorySource.SYSTEM,
            importance=ImportanceLevel.CRITICAL,
        ),
        MemoryEntry(
            content="User completed onboarding on Tuesday",
            layer=MemoryLayer.EPISODIC,
            source=MemorySource.USER,
            importance=ImportanceLevel.MEDIUM,
        ),
        MemoryEntry(
            content="The speed of light is approximately 299,792,458 m/s",
            layer=MemoryLayer.SEMANTIC,
            source=MemorySource.SYSTEM,
            importance=ImportanceLevel.LOW,
        ),
    ]

    for entry in entries:
        unified.store(entry)

    print(f"\nStored {len(entries)} entries.")

    # Step 5: Search with scoring
    scorer = ImportanceScorer()
    results = unified.search("maintenance")
    print(f"\nSearch 'maintenance' -> {len(results)} result(s):")
    for result in results:
        score = scorer.score(result)
        print(f"  [{result.layer.value}] importance={result.importance.value} "
              f"| score={score:.2f} | {result.content[:60]}")

    # Step 6: Check freshness
    freshness_scorer = FreshnessScorer()
    all_entries = unified.search("light")
    for entry in all_entries:
        freshness = freshness_scorer.score(entry)
        print(f"\nFreshness score for '{entry.content[:40]}...': {freshness:.2f}")


if __name__ == "__main__":
    main()
