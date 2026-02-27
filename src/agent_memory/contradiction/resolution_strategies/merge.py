"""Merge resolution strategy.

:class:`MergeResolutionStrategy` attempts to combine the content of two
contradicting memory entries into a single merged entry.

The merge algorithm is intentionally simple (no LLM dependency):

1. Split each entry's content into sentences by splitting on ``. `` and
   ``! `` and ``? `` sentence boundaries.
2. Collect the union of sentences from both entries, deduplicating
   near-identical sentences (case-insensitive, strip whitespace).
3. Join the merged sentences back into a single string.

This is a commodity, deterministic merge suitable for cases where the
two entries are believed to contain complementary (not truly conflicting)
information — for example, two partial descriptions of the same entity.

When the entries appear truly contradictory (same sentence in both but
with different content), both versions are included with a separator.

Usage
-----
>>> from agent_memory.contradiction.models import MemoryEntry
>>> from agent_memory.contradiction.strategies.merge import MergeResolutionStrategy
>>> existing = MemoryEntry(content="Alice works at Acme. She likes coffee.")
>>> new = MemoryEntry(content="Alice works at Acme. She lives in London.")
>>> strategy = MergeResolutionStrategy()
>>> result = strategy.resolve(existing, new)
>>> result.action
'merge'
>>> result.resolved_content is not None
True
>>> "coffee" in result.resolved_content and "London" in result.resolved_content
True
"""
from __future__ import annotations

import re

from agent_memory.contradiction.models import MemoryEntry
from agent_memory.contradiction.resolution import ResolutionResult, ResolutionStrategyBase


class MergeResolutionStrategy(ResolutionStrategyBase):
    """Merge non-conflicting parts of two contradicting memory entries.

    The strategy produces a single merged content string that is the
    union of unique sentences from both entries.  The action is always
    ``"merge"``.

    Parameters
    ----------
    separator:
        String used to join merged sentences.  Defaults to ``" "``
        (single space after each sentence's own punctuation).

    Example
    -------
    >>> strategy = MergeResolutionStrategy()
    >>> strategy.name
    'merge'
    """

    _MERGE_MARKER: str = " [ALSO: "
    _MERGE_MARKER_CLOSE: str = "]"

    def __init__(self, separator: str = " ") -> None:
        self._separator = separator

    @property
    def name(self) -> str:
        return "merge"

    def resolve(self, existing: MemoryEntry, new: MemoryEntry) -> ResolutionResult:
        """Merge the content of *existing* and *new* into a single entry.

        Parameters
        ----------
        existing:
            The entry already in memory.
        new:
            The incoming entry that contradicts *existing*.

        Returns
        -------
        ResolutionResult
            Result with ``action="merge"`` and the merged content in
            ``resolved_content``.  ``requires_user_input=False``.
        """
        merged_content = _merge_contents(
            existing.content,
            new.content,
            separator=self._separator,
        )

        unique_sentences_count = len(_split_sentences(merged_content))
        explanation = (
            f"Merged {len(_split_sentences(existing.content))} sentence(s) from existing "
            f"and {len(_split_sentences(new.content))} sentence(s) from new entry into "
            f"{unique_sentences_count} unique sentence(s)."
        )

        return ResolutionResult(
            strategy_used=self.name,
            action="merge",
            resolved_content=merged_content,
            requires_user_input=False,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences by common punctuation boundaries.

    Parameters
    ----------
    text:
        Input text to split.

    Returns
    -------
    list[str]
        List of non-empty, stripped sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalise_sentence(sentence: str) -> str:
    """Return a normalised key for deduplication (lowercase, stripped)."""
    return sentence.lower().strip().rstrip(".!?")


def _merge_contents(existing: str, new: str, separator: str = " ") -> str:
    """Merge two content strings by combining unique sentences.

    Parameters
    ----------
    existing:
        Content of the existing memory entry.
    new:
        Content of the new memory entry.
    separator:
        String used between sentences in the output.

    Returns
    -------
    str
        Merged content string.
    """
    existing_sentences = _split_sentences(existing)
    new_sentences = _split_sentences(new)

    # Build ordered union, deduplicating by normalised form
    seen_normalised: set[str] = set()
    merged: list[str] = []

    for sentence in existing_sentences:
        key = _normalise_sentence(sentence)
        if key not in seen_normalised:
            seen_normalised.add(key)
            merged.append(sentence)

    for sentence in new_sentences:
        key = _normalise_sentence(sentence)
        if key not in seen_normalised:
            seen_normalised.add(key)
            merged.append(sentence)

    if not merged:
        # Fallback: concatenate with separator
        return f"{existing}{separator}{new}".strip()

    return separator.join(merged)
