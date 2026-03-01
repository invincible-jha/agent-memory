#!/usr/bin/env python3
"""Example: LangChain Memory Integration

Demonstrates using agent-memory as the backing store for LangChain
conversation history via the AutoMemorizeMiddleware.

Usage:
    python examples/06_langchain_memory.py

Requirements:
    pip install agent-memory langchain
"""
from __future__ import annotations

try:
    from langchain.memory import ConversationBufferMemory
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

import agent_memory
from agent_memory import (
    Memory,
    MemoryLayer,
    AutoMemorizeMiddleware,
    Interaction,
    ContextBuilder,
    ContextSection,
)


def simulate_conversation(mem: Memory, turns: list[tuple[str, str]]) -> None:
    """Store conversation turns and build context."""
    middleware = AutoMemorizeMiddleware(memory=mem)
    for user_msg, agent_reply in turns:
        interaction = Interaction(user_input=user_msg, agent_response=agent_reply)
        middleware.process(interaction)


def main() -> None:
    print(f"agent-memory version: {agent_memory.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("langchain not installed — demonstrating agent-memory middleware only.")
        print("Install with: pip install langchain")

    # Step 1: Create memory store
    mem = Memory()

    # Step 2: Simulate a multi-turn conversation
    conversation: list[tuple[str, str]] = [
        ("My name is Alice and I work in finance.", "Nice to meet you, Alice!"),
        ("I need help preparing the Q3 earnings report.", "Of course! Let me help you with the Q3 report."),
        ("The deadline is next Friday.", "Noted — the Q3 report deadline is next Friday."),
        ("My preferred chart style is bar charts.", "I'll use bar charts for all visualisations."),
    ]

    simulate_conversation(mem, conversation)
    print(f"Stored {len(conversation)} conversation turns in memory.")

    # Step 3: Build context for the next agent call
    context_builder = ContextBuilder(memory=mem)
    context = context_builder.build(
        query="What do we know about Alice?",
        sections=[ContextSection.RECENT_INTERACTIONS, ContextSection.RELEVANT_FACTS],
        max_tokens=500,
    )

    print(f"\nContext built for query 'What do we know about Alice?':")
    print(f"  Sections included: {len(context.sections)}")
    for section in context.sections:
        print(f"  [{section.label}] {section.content[:100]}{'...' if len(section.content) > 100 else ''}")

    # Step 4: Recall specific facts
    preferences = mem.recall("chart style preferences")
    print(f"\nRecall 'chart style preferences' -> {len(preferences)} result(s):")
    for result in preferences:
        print(f"  {result.content[:80]}")

    deadline_memories = mem.recall("deadline")
    print(f"\nRecall 'deadline' -> {len(deadline_memories)} result(s):")
    for result in deadline_memories:
        print(f"  {result.content[:80]}")


if __name__ == "__main__":
    main()
