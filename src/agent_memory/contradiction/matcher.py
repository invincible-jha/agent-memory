"""Entity-attribute matcher — extract entity-attribute-value triples and compare."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Triple:
    """An entity-attribute-value triple extracted from memory content."""

    entity: str
    attribute: str
    value: str


# Simple pattern: "<entity> <is/are/was/has/have> <value>"
# More patterns can be added without changing the public interface.
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # "X is Y" / "X are Y" / "X was Y"
    (re.compile(r"^(.+?)\s+(?:is|are|was|were)\s+(.+)$", re.IGNORECASE), "state", "value"),
    # "X has Y" / "X have Y"
    (re.compile(r"^(.+?)\s+(?:has|have|had)\s+(.+)$", re.IGNORECASE), "property", "value"),
    # "X = Y" (key=value style)
    (re.compile(r"^(.+?)\s*=\s*(.+)$"), "equals", "value"),
]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class EntityAttributeMatcher:
    """Extract entity-attribute-value triples and detect conflicts."""

    def extract_triples(self, content: str) -> list[Triple]:
        """Extract structured triples from free-form content."""
        results: list[Triple] = []
        sentences = re.split(r"[.!?;]", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern, attribute, _ in _PATTERNS:
                match = pattern.match(sentence)
                if match:
                    entity = _normalise(match.group(1))
                    value = _normalise(match.group(2))
                    if entity and value:
                        results.append(Triple(entity=entity, attribute=attribute, value=value))
                        break
        return results

    def conflicts(
        self,
        triples_a: Sequence[Triple],
        triples_b: Sequence[Triple],
    ) -> list[tuple[Triple, Triple]]:
        """Return pairs of triples that conflict (same entity+attribute, different value)."""
        pairs: list[tuple[Triple, Triple]] = []
        for ta in triples_a:
            for tb in triples_b:
                if (
                    ta.entity == tb.entity
                    and ta.attribute == tb.attribute
                    and ta.value != tb.value
                ):
                    pairs.append((ta, tb))
        return pairs

    def similarity(self, text_a: str, text_b: str) -> float:
        """Simple token-level Jaccard similarity between two texts."""
        tokens_a = set(re.findall(r"[a-z0-9]+", text_a.lower()))
        tokens_b = set(re.findall(r"[a-z0-9]+", text_b.lower()))
        if not tokens_a and not tokens_b:
            return 1.0
        union = tokens_a | tokens_b
        intersection = tokens_a & tokens_b
        return len(intersection) / len(union)


__all__ = ["EntityAttributeMatcher", "Triple"]
