"""ContextBuilder — assembles a memory-enriched context string for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from agent_memory.memory.types import MemoryEntry, MemoryLayer
from agent_memory.unified.memory import UnifiedMemory


@dataclass
class ContextSection:
    """A named section within the assembled context."""

    heading: str
    entries: list[MemoryEntry]
    token_budget: int


# Approximate characters per token (conservative estimate)
_CHARS_PER_TOKEN = 4

# Section headings for each memory layer
_LAYER_HEADINGS: dict[MemoryLayer, str] = {
    MemoryLayer.WORKING: "Current Context",
    MemoryLayer.EPISODIC: "Recent Interactions",
    MemoryLayer.SEMANTIC: "Background Knowledge",
    MemoryLayer.PROCEDURAL: "Procedures & Skills",
}

# Default order in which layers appear in the context block
_DEFAULT_LAYER_ORDER: list[MemoryLayer] = [
    MemoryLayer.WORKING,
    MemoryLayer.EPISODIC,
    MemoryLayer.SEMANTIC,
    MemoryLayer.PROCEDURAL,
]


class ContextBuilder:
    """Build a memory-enriched context string from a UnifiedMemory instance.

    The builder queries each memory layer, ranks entries by composite score
    (importance * freshness), and formats them into a structured text block
    that fits within a token budget.

    Parameters
    ----------
    memory:
        The ``UnifiedMemory`` to query.
    layer_order:
        Order in which layers are included in the context.  Working memory
        always comes first if not explicitly ordered otherwise.
    chars_per_token:
        Conversion factor for estimating token count from character count.
        Defaults to 4, which is a conservative approximation.
    include_metadata:
        When True, minimal metadata (source, importance) is appended to
        each entry line.
    separator:
        String placed between context sections.
    """

    def __init__(
        self,
        memory: UnifiedMemory,
        layer_order: Optional[list[MemoryLayer]] = None,
        chars_per_token: int = _CHARS_PER_TOKEN,
        include_metadata: bool = False,
        separator: str = "\n\n",
    ) -> None:
        self._memory = memory
        self._layer_order = layer_order or _DEFAULT_LAYER_ORDER
        self._chars_per_token = max(1, chars_per_token)
        self._include_metadata = include_metadata
        self._separator = separator

    def build(
        self,
        query: str,
        token_budget: int = 2048,
        min_importance: float = 0.0,
        top_per_layer: int = 10,
    ) -> str:
        """Build a context string relevant to the given query.

        Parameters
        ----------
        query:
            The current user query or topic; used to rank retrieved entries.
        token_budget:
            Maximum number of tokens for the entire context block.
        min_importance:
            Exclude entries with importance_score below this threshold.
        top_per_layer:
            Maximum entries retrieved per layer before budget trimming.

        Returns
        -------
        str
            Formatted context string ready for inclusion in a prompt.
        """
        char_budget = token_budget * self._chars_per_token
        sections: list[str] = []
        used_chars = 0

        for layer in self._layer_order:
            if used_chars >= char_budget:
                break

            remaining_budget = char_budget - used_chars
            section_text = self._build_section(
                layer=layer,
                query=query,
                char_budget=remaining_budget,
                min_importance=min_importance,
                top_per_layer=top_per_layer,
            )
            if section_text:
                sections.append(section_text)
                used_chars += len(section_text)

        return self._separator.join(sections)

    def build_from_entries(
        self,
        entries: Sequence[MemoryEntry],
        token_budget: int = 2048,
    ) -> str:
        """Build a context string from an explicit list of entries.

        Parameters
        ----------
        entries:
            Pre-selected entries to format.
        token_budget:
            Character budget (token_budget * chars_per_token).

        Returns
        -------
        str
            Formatted context string.
        """
        char_budget = token_budget * self._chars_per_token
        sorted_entries = sorted(entries, key=lambda e: e.composite_score, reverse=True)
        lines: list[str] = []
        used = 0
        for entry in sorted_entries:
            line = self._format_entry(entry)
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count for a text string."""
        return len(text) // self._chars_per_token

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_section(
        self,
        layer: MemoryLayer,
        query: str,
        char_budget: int,
        min_importance: float,
        top_per_layer: int,
    ) -> str:
        """Build a single section for one memory layer."""
        search_results = self._memory.search(query, layer=layer, limit=top_per_layer)
        entries = [
            e for e in search_results
            if e.importance_score >= min_importance
        ]

        if not entries:
            return ""

        # Sort by composite score descending
        entries_sorted = sorted(entries, key=lambda e: e.composite_score, reverse=True)

        heading = _LAYER_HEADINGS.get(layer, layer.value.replace("_", " ").title())
        header_line = f"[{heading}]"
        lines: list[str] = [header_line]
        used = len(header_line)

        for entry in entries_sorted:
            line = self._format_entry(entry)
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1  # +1 for the newline

        if len(lines) <= 1:
            # Only the header — no entries fit
            return ""

        return "\n".join(lines)

    def _format_entry(self, entry: MemoryEntry) -> str:
        """Format a single MemoryEntry as a context line."""
        content = entry.content.strip()
        if not self._include_metadata:
            return f"- {content}"

        importance = f"{entry.importance_score:.2f}"
        source = entry.source.value
        return f"- {content} [src={source}, imp={importance}]"


__all__ = ["ContextBuilder", "ContextSection"]
