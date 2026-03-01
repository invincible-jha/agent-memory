#!/usr/bin/env python3
"""Example: Quickstart

Demonstrates the minimal setup for agent-memory using the Memory
convenience class to store and retrieve memories.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-memory
"""
from __future__ import annotations

import agent_memory
from agent_memory import Memory, MemoryLayer


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    # Step 1: Create a zero-config memory instance
    mem = Memory()
    print(f"Memory instance created: {mem}")

    # Step 2: Store some memories
    mem.remember("The project deadline is March 15th.", layer=MemoryLayer.WORKING)
    mem.remember("Python was created by Guido van Rossum.", layer=MemoryLayer.SEMANTIC)
    mem.remember("User prefers concise responses.", layer=MemoryLayer.SEMANTIC)
    print("\nStored 3 memories across layers.")

    # Step 3: Retrieve relevant memories
    results = mem.recall("deadline")
    print(f"\nRecall 'deadline' -> {len(results)} result(s):")
    for result in results:
        print(f"  [{result.layer.value}] {result.content[:80]}")

    # Step 4: Retrieve from semantic layer
    semantic_results = mem.recall("Python programming language")
    print(f"\nRecall 'Python' -> {len(semantic_results)} result(s):")
    for result in semantic_results:
        print(f"  [{result.layer.value}] {result.content[:80]}")

    print("\nQuickstart complete.")


if __name__ == "__main__":
    main()
