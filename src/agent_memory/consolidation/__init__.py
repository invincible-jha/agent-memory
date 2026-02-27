"""Episodic-to-semantic memory consolidation."""

from __future__ import annotations

from agent_memory.consolidation.consolidator import ConsolidationResult, EpisodicConsolidator
from agent_memory.consolidation.pattern_extractor import ExtractedPattern, PatternExtractor

__all__ = [
    "EpisodicConsolidator",
    "ConsolidationResult",
    "PatternExtractor",
    "ExtractedPattern",
]
