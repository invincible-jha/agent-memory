"""Framework adapters for agent_memory.

Each adapter bridges a specific agent framework's memory/message model to the
agent_memory storage layer without requiring the framework to be installed.
"""
from __future__ import annotations

from agent_memory.adapters.anthropic_sdk import AnthropicMemoryBridge
from agent_memory.adapters.crewai import CrewAIMemoryBridge
from agent_memory.adapters.langchain import LangChainMemoryBridge
from agent_memory.adapters.microsoft_agents import MicrosoftMemoryBridge
from agent_memory.adapters.openai_agents import OpenAIMemoryBridge

__all__ = [
    "AnthropicMemoryBridge",
    "CrewAIMemoryBridge",
    "LangChainMemoryBridge",
    "MicrosoftMemoryBridge",
    "OpenAIMemoryBridge",
]
