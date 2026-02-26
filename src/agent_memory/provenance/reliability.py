"""Source reliability rankings for memory provenance."""

from __future__ import annotations

from agent_memory.memory.types import MemorySource


# Reliability scores in [0, 1] — higher is more trustworthy
_RELIABILITY_MAP: dict[MemorySource, float] = {
    MemorySource.TOOL_OUTPUT: 0.95,      # VERIFIED_TOOL
    MemorySource.DOCUMENT: 0.85,          # VERIFIED_DOCUMENT
    MemorySource.EXTERNAL_API: 0.80,      # trusted external feed
    MemorySource.USER_INPUT: 0.65,        # USER_INPUT
    MemorySource.AGENT_INFERENCE: 0.45,   # AGENT_INFERENCE / UNKNOWN
}


class SourceReliability:
    """Map memory sources to reliability scores.

    Tier ordering (descending):
    VERIFIED_TOOL (TOOL_OUTPUT) > VERIFIED_DOCUMENT (DOCUMENT) >
    EXTERNAL_API > USER_INPUT > AGENT_INFERENCE > UNKNOWN
    """

    def score(self, source: MemorySource) -> float:
        """Return the reliability score for a given source."""
        return _RELIABILITY_MAP.get(source, 0.3)

    def rank(self, sources: list[MemorySource]) -> list[MemorySource]:
        """Return sources sorted from most to least reliable."""
        return sorted(sources, key=self.score, reverse=True)

    def is_verified(self, source: MemorySource) -> bool:
        """Return True for sources considered 'verified' (score >= 0.80)."""
        return self.score(source) >= 0.80

    def tier_label(self, source: MemorySource) -> str:
        """Return a human-readable reliability tier label."""
        score = self.score(source)
        if score >= 0.90:
            return "VERIFIED_TOOL"
        if score >= 0.80:
            return "VERIFIED_DOCUMENT"
        if score >= 0.65:
            return "USER_INPUT"
        if score >= 0.45:
            return "AGENT_INFERENCE"
        return "UNKNOWN"


__all__ = ["SourceReliability"]
