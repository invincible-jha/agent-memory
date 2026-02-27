"""PatternExtractor — extract common entities and actions from episode clusters.

Provides commodity NLP-lite extraction of repeated nouns/verbs across a
group of episodic memory entries using simple frequency analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Simple token classification heuristics
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being have has had do does did
    will would could should may might shall must can need dare ought used
    am at by for in of on or so and but nor yet with as if it its this
    that these those i me my we our you your he him his she her they
    them their what which who whom when where why how all both each few
    more most other some such no only same than then too very just""".split()
)

# Simple action verbs — commodity list of common English verbs
_COMMON_VERBS: frozenset[str] = frozenset(
    """run ran create created update updated delete deleted read write get set
    fetch send receive process handle validate check call invoke execute deploy
    start stop restart fail succeed complete finish log report monitor store
    retrieve merge split join connect disconnect open close load save export
    import configure register unregister scan detect alert notify respond""".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words."""
    words = re.findall(r"[a-z][a-z0-9_-]*", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _is_likely_entity(token: str) -> bool:
    """Heuristic: longer tokens that are not common verbs tend to be entities."""
    return len(token) >= 4 and token not in _COMMON_VERBS


def _is_likely_action(token: str) -> bool:
    """Heuristic: token appears in the common verb list."""
    # Also check verb stems by checking if the base form is in the set
    base_forms = {token, token.rstrip("ed"), token.rstrip("ing"), token.rstrip("s")}
    return bool(base_forms & _COMMON_VERBS)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedPattern:
    """Common entities and actions found across a cluster of episodes.

    Parameters
    ----------
    entities:
        Recurring noun-like tokens with their frequency counts.
    actions:
        Recurring verb-like tokens with their frequency counts.
    total_episodes:
        Number of episodes used as input.
    coverage_ratio:
        Fraction of episodes where at least one top entity/action appeared.
    """

    entities: dict[str, int]
    actions: dict[str, int]
    total_episodes: int
    coverage_ratio: float

    def top_entities(self, limit: int = 5) -> list[tuple[str, int]]:
        """Return top entities by frequency.

        Parameters
        ----------
        limit:
            Maximum number of results.

        Returns
        -------
        list[tuple[str, int]]
            Sorted (entity, count) pairs, highest first.
        """
        return sorted(self.entities.items(), key=lambda x: x[1], reverse=True)[:limit]

    def top_actions(self, limit: int = 5) -> list[tuple[str, int]]:
        """Return top actions by frequency.

        Parameters
        ----------
        limit:
            Maximum number of results.

        Returns
        -------
        list[tuple[str, int]]
            Sorted (action, count) pairs, highest first.
        """
        return sorted(self.actions.items(), key=lambda x: x[1], reverse=True)[:limit]

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "entities": dict(self.top_entities(10)),
            "actions": dict(self.top_actions(10)),
            "total_episodes": self.total_episodes,
            "coverage_ratio": self.coverage_ratio,
        }


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class PatternExtractor:
    """Extract common entities and actions from a collection of text snippets.

    Uses simple token frequency analysis — no ML dependencies required.

    Parameters
    ----------
    min_frequency:
        Minimum number of occurrences for a token to be considered a pattern.
        Defaults to 2.
    max_entities:
        Maximum number of entity patterns to return. Defaults to 20.
    max_actions:
        Maximum number of action patterns to return. Defaults to 10.
    """

    def __init__(
        self,
        min_frequency: int = 2,
        max_entities: int = 20,
        max_actions: int = 10,
    ) -> None:
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")
        self._min_frequency = min_frequency
        self._max_entities = max_entities
        self._max_actions = max_actions

    def extract(self, texts: list[str]) -> ExtractedPattern:
        """Extract recurring entity and action patterns from text snippets.

        Parameters
        ----------
        texts:
            List of text strings to analyse (e.g. episode content fields).

        Returns
        -------
        ExtractedPattern
            Entities and actions found above the minimum frequency threshold.
        """
        if not texts:
            return ExtractedPattern(
                entities={},
                actions={},
                total_episodes=0,
                coverage_ratio=0.0,
            )

        entity_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        covered_count = 0

        for text in texts:
            tokens = _tokenise(text)
            token_set = set(tokens)

            episode_entities = {t for t in token_set if _is_likely_entity(t)}
            episode_actions = {t for t in token_set if _is_likely_action(t)}

            entity_counter.update(episode_entities)
            action_counter.update(episode_actions)

            if episode_entities or episode_actions:
                covered_count += 1

        # Filter by minimum frequency
        filtered_entities = {
            token: count
            for token, count in entity_counter.items()
            if count >= self._min_frequency
        }
        filtered_actions = {
            token: count
            for token, count in action_counter.items()
            if count >= self._min_frequency
        }

        # Trim to max limits
        top_entities = dict(
            Counter(filtered_entities).most_common(self._max_entities)
        )
        top_actions = dict(
            Counter(filtered_actions).most_common(self._max_actions)
        )

        coverage_ratio = covered_count / len(texts) if texts else 0.0

        return ExtractedPattern(
            entities=top_entities,
            actions=top_actions,
            total_episodes=len(texts),
            coverage_ratio=round(coverage_ratio, 4),
        )

    def extract_from_cluster(
        self,
        episodes: list[object],
        content_attr: str = "content",
    ) -> ExtractedPattern:
        """Extract patterns from objects with a content field.

        Parameters
        ----------
        episodes:
            Objects that have a ``content`` string attribute (e.g. MemoryEntry).
        content_attr:
            Name of the attribute to read text from. Defaults to ``"content"``.

        Returns
        -------
        ExtractedPattern
            Entities and actions extracted from the episode contents.
        """
        texts = [getattr(ep, content_attr, "") for ep in episodes]
        return self.extract([str(t) for t in texts if t])


__all__ = ["PatternExtractor", "ExtractedPattern"]
