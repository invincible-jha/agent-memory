#!/usr/bin/env python3
"""Example: CrewAI Memory Integration

Demonstrates sharing agent-memory between multiple CrewAI agents so
they can access a common knowledge base.

Usage:
    python examples/07_crewai_memory.py

Requirements:
    pip install agent-memory crewai
"""
from __future__ import annotations

try:
    from crewai import Agent, Task, Crew, Process
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

import agent_memory
from agent_memory import (
    Memory,
    MemoryLayer,
    MemoryEntry,
    MemorySource,
    ImportanceLevel,
    ContextBuilder,
    ContextSection,
)


def populate_shared_memory(mem: Memory) -> None:
    """Populate a shared memory store with project context."""
    facts: list[tuple[str, MemoryLayer]] = [
        ("Project Phoenix targets enterprise healthcare clients", MemoryLayer.SEMANTIC),
        ("Budget approved: $2.4M for FY2026", MemoryLayer.SEMANTIC),
        ("Tech stack: Next.js, Python, PostgreSQL", MemoryLayer.SEMANTIC),
        ("Launch date: Q3 2026", MemoryLayer.WORKING),
        ("Primary contact: Sarah Chen, VP Engineering", MemoryLayer.SEMANTIC),
    ]
    for content, layer in facts:
        mem.remember(content, layer=layer)


def agent_with_memory(role: str, mem: Memory, query: str) -> str:
    """Simulate a CrewAI agent that uses shared memory."""
    context_builder = ContextBuilder(memory=mem)
    context = context_builder.build(
        query=query,
        sections=[ContextSection.RELEVANT_FACTS],
        max_tokens=200,
    )
    context_text = " ".join(s.content for s in context.sections)
    return f"[{role}] Based on context: {context_text[:120]}... -> Task complete."


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    if not _CREWAI_AVAILABLE:
        print("crewai not installed — demonstrating shared memory pattern only.")
        print("Install with: pip install crewai")

    # Step 1: Create and populate shared memory store
    shared_mem = Memory()
    populate_shared_memory(shared_mem)
    print(f"Shared memory populated.")

    # Step 2: Simulate multiple agents drawing from shared memory
    agents: list[tuple[str, str]] = [
        ("Product Manager", "What is the project timeline and budget?"),
        ("Tech Lead", "What is the technology stack for this project?"),
        ("Account Manager", "Who is the primary contact for this project?"),
    ]

    print("\nAgent outputs using shared memory:")
    for role, query in agents:
        result = agent_with_memory(role=role, mem=shared_mem, query=query)
        print(f"  {result}")

    # Step 3: If CrewAI available, demonstrate real crew setup
    if _CREWAI_AVAILABLE:
        print("\nBuilding CrewAI crew with memory-aware tasks:")

        def make_memory_task(agent_role: str, query: str) -> str:
            return agent_with_memory(agent_role, shared_mem, query)

        research_agent = Agent(
            role="Research Analyst",
            goal="Answer questions using available context",
            backstory="Expert at synthesising project information.",
            verbose=False,
        )
        task = Task(
            description="Summarise the Project Phoenix key facts.",
            agent=research_agent,
            expected_output="A concise project summary.",
        )
        crew = Crew(agents=[research_agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        print(f"  CrewAI output: {str(result)[:100]}")

    # Step 4: Show memory recall across all agents' stored info
    print(f"\nShared memory recall 'budget':")
    for entry in shared_mem.recall("budget"):
        print(f"  {entry.content}")


if __name__ == "__main__":
    main()
