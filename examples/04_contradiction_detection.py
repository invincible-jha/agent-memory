#!/usr/bin/env python3
"""Example: Contradiction Detection

Demonstrates how to detect and resolve contradictory memories using
the ContradictionScanner and ContradictionResolver.

Usage:
    python examples/04_contradiction_detection.py

Requirements:
    pip install agent-memory
"""
from __future__ import annotations

import agent_memory
from agent_memory import (
    Memory,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    ImportanceLevel,
    ContradictionScanner,
    ContradictionResolver,
    ResolutionStrategy,
    ScanScope,
    UnifiedMemory,
    MemoryConfig,
    InMemoryStorage,
)


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    # Step 1: Create a unified memory with in-memory storage
    config = MemoryConfig()
    storage = InMemoryStorage()
    unified = UnifiedMemory(config=config, storage=storage)

    # Step 2: Store conflicting facts about the same entity
    conflicting_entries: list[MemoryEntry] = [
        MemoryEntry(
            content="The meeting is scheduled for 2pm on Friday.",
            layer=MemoryLayer.WORKING,
            source=MemorySource.USER,
            importance=ImportanceLevel.HIGH,
            metadata={"entity": "meeting", "attribute": "time"},
        ),
        MemoryEntry(
            content="The meeting has been moved to 3pm on Friday.",
            layer=MemoryLayer.WORKING,
            source=MemorySource.SYSTEM,
            importance=ImportanceLevel.HIGH,
            metadata={"entity": "meeting", "attribute": "time"},
        ),
        MemoryEntry(
            content="Alice is the project lead.",
            layer=MemoryLayer.SEMANTIC,
            source=MemorySource.USER,
            importance=ImportanceLevel.MEDIUM,
            metadata={"entity": "Alice", "attribute": "role"},
        ),
        MemoryEntry(
            content="Bob is the project lead.",
            layer=MemoryLayer.SEMANTIC,
            source=MemorySource.SYSTEM,
            importance=ImportanceLevel.MEDIUM,
            metadata={"entity": "project_lead", "attribute": "person"},
        ),
    ]

    for entry in conflicting_entries:
        unified.store(entry)

    print(f"Stored {len(conflicting_entries)} entries (with intentional contradictions).")

    # Step 3: Scan for contradictions
    scanner = ContradictionScanner(memory=unified)
    scan_result = scanner.scan(scope=ScanScope.ALL)
    print(f"\nContradiction scan complete:")
    print(f"  Contradictions found: {scan_result.contradiction_count}")
    print(f"  Scanned entries: {scan_result.total_scanned}")

    if scan_result.report:
        for pair in scan_result.report.pairs[:3]:
            print(f"\n  Contradiction:")
            print(f"    A: {pair.entry_a.content[:60]}")
            print(f"    B: {pair.entry_b.content[:60]}")
            print(f"    Confidence: {pair.confidence:.2f}")

    # Step 4: Resolve contradictions using most-recent strategy
    if scan_result.report and scan_result.report.pairs:
        resolver = ContradictionResolver(strategy=ResolutionStrategy.MOST_RECENT)
        for pair in scan_result.report.pairs[:1]:
            resolution = resolver.resolve(pair)
            print(f"\nResolution (most-recent strategy):")
            print(f"  Kept: {resolution.kept_entry.content[:60]}")
            print(f"  Discarded: {resolution.discarded_entry.content[:60] if resolution.discarded_entry else 'N/A'}")
            print(f"  Rationale: {resolution.rationale}")


if __name__ == "__main__":
    main()
