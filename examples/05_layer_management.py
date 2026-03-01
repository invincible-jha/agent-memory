#!/usr/bin/env python3
"""Example: Layer Management

Demonstrates the four cognitive memory layers — working, episodic,
semantic, and procedural — with garbage collection and provenance.

Usage:
    python examples/05_layer_management.py

Requirements:
    pip install agent-memory
"""
from __future__ import annotations

import agent_memory
from agent_memory import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    Procedure,
    ProcedureStep,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    ImportanceLevel,
    MemoryGarbageCollector,
    ProvenanceTracker,
    SourceReliability,
)


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    # Step 1: Working memory — short-term task context
    working = WorkingMemory(capacity=10)
    working.add(MemoryEntry(
        content="Current task: summarise Q4 report",
        layer=MemoryLayer.WORKING,
        source=MemorySource.SYSTEM,
        importance=ImportanceLevel.HIGH,
    ))
    working.add(MemoryEntry(
        content="User is in EST timezone",
        layer=MemoryLayer.WORKING,
        source=MemorySource.USER,
        importance=ImportanceLevel.LOW,
    ))
    print(f"Working memory entries: {working.count()}")

    # Step 2: Episodic memory — event history
    episodic = EpisodicMemory()
    episodic.add(MemoryEntry(
        content="User asked for help with data analysis on Monday",
        layer=MemoryLayer.EPISODIC,
        source=MemorySource.USER,
        importance=ImportanceLevel.MEDIUM,
    ))
    episodic.add(MemoryEntry(
        content="Agent completed report generation successfully",
        layer=MemoryLayer.EPISODIC,
        source=MemorySource.SYSTEM,
        importance=ImportanceLevel.HIGH,
    ))
    print(f"Episodic memory entries: {episodic.count()}")

    # Step 3: Semantic memory — factual knowledge
    semantic = SemanticMemory()
    semantic.add(MemoryEntry(
        content="Company fiscal year runs January to December",
        layer=MemoryLayer.SEMANTIC,
        source=MemorySource.SYSTEM,
        importance=ImportanceLevel.MEDIUM,
    ))
    print(f"Semantic memory entries: {semantic.count()}")

    # Step 4: Procedural memory — learned procedures
    procedural = ProceduralMemory()
    report_procedure = Procedure(
        name="generate_report",
        steps=[
            ProcedureStep(order=1, action="fetch_data", description="Retrieve data from source"),
            ProcedureStep(order=2, action="analyse_data", description="Run statistical analysis"),
            ProcedureStep(order=3, action="format_output", description="Format results as markdown"),
        ],
    )
    procedural.add_procedure(report_procedure)
    retrieved = procedural.get_procedure("generate_report")
    print(f"Procedural memory: '{retrieved.name}' with {len(retrieved.steps)} steps")

    # Step 5: Provenance tracking
    tracker = ProvenanceTracker()
    entry = MemoryEntry(
        content="Revenue grew 12% YoY",
        layer=MemoryLayer.SEMANTIC,
        source=MemorySource.SYSTEM,
        importance=ImportanceLevel.HIGH,
    )
    tracker.record(entry=entry, source_reliability=SourceReliability.HIGH)
    reliability = tracker.get_reliability(entry.id)
    print(f"\nProvenance for '{entry.content[:30]}...': reliability={reliability.name}")

    # Step 6: Garbage collection
    all_entries = working.all() + episodic.all() + semantic.all()
    gc = MemoryGarbageCollector(importance_threshold=ImportanceLevel.MEDIUM)
    collected = gc.collect(all_entries)
    print(f"\nGarbage collection: {len(all_entries)} entries -> {len(collected)} retained")


if __name__ == "__main__":
    main()
