"""Semantic memory — facts with TF-IDF keyword retrieval and entity linking."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Optional, Sequence

from agent_memory.memory.base import MemoryStore
from agent_memory.memory.types import MemoryEntry, MemoryLayer

# ---------------------------------------------------------------------------
# Stop-words used during tokenisation
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being have has had do does did
    will would could should may might shall must can need dare ought
    used to am at by for in of on or so and but nor yet with as if
    it its it's this that these those i me my we our you your he him
    his she her they them their what which who whom when where why how
    all both each few more most other some such no only same than then
    too very just""".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words, return tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


class SemanticMemory(MemoryStore):
    """Long-term factual knowledge with keyword-based TF-IDF retrieval.

    Each entry can carry an optional ``entity`` key in its metadata dict to
    enable entity-scoped lookups.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        # Inverted index: token -> set of memory_ids
        self._inverted: dict[str, set[str]] = defaultdict(set)
        # Entity index: entity_name -> set of memory_ids
        self._entity_index: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        # Remove old index entries if updating
        if entry.memory_id in self._entries:
            self._remove_from_index(entry.memory_id)
        self._entries[entry.memory_id] = entry
        self._add_to_index(entry)

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(memory_id)

    def all(self, layer: Optional[MemoryLayer] = None) -> Sequence[MemoryEntry]:
        entries = list(self._entries.values())
        if layer is not None:
            entries = [e for e in entries if e.layer == layer]
        return entries

    def count(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            return len(self._entries)
        return sum(1 for e in self._entries.values() if e.layer == layer)

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._entries:
            return False
        self._remove_from_index(memory_id)
        del self._entries[memory_id]
        return True

    def clear(self, layer: Optional[MemoryLayer] = None) -> int:
        if layer is None:
            count = len(self._entries)
            self._entries.clear()
            self._inverted.clear()
            self._entity_index.clear()
            return count
        to_delete = [mid for mid, e in self._entries.items() if e.layer == layer]
        for mid in to_delete:
            self.delete(mid)
        return len(to_delete)

    def search(
        self,
        query: str,
        layer: Optional[MemoryLayer] = None,
        limit: int = 20,
    ) -> Sequence[MemoryEntry]:
        """TF-IDF ranked keyword search over stored entries."""
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = self._tfidf_scores(tokens, layer=layer)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self._entries[mid] for mid, _ in ranked[:limit] if mid in self._entries]

    # ------------------------------------------------------------------
    # Entity-linked retrieval
    # ------------------------------------------------------------------

    def retrieve_by_entity(self, entity: str) -> Sequence[MemoryEntry]:
        """Return all memories associated with a named entity."""
        mids = self._entity_index.get(entity.lower(), set())
        return [self._entries[mid] for mid in mids if mid in self._entries]

    def linked_entities(self) -> list[str]:
        """Return all known entity names."""
        return sorted(self._entity_index.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_to_index(self, entry: MemoryEntry) -> None:
        tokens = _tokenise(entry.content)
        for token in set(tokens):
            self._inverted[token].add(entry.memory_id)
        entity = entry.metadata.get("entity", "").lower()
        if entity:
            self._entity_index[entity].add(entry.memory_id)

    def _remove_from_index(self, memory_id: str) -> None:
        entry = self._entries.get(memory_id)
        if entry is None:
            return
        tokens = _tokenise(entry.content)
        for token in set(tokens):
            self._inverted[token].discard(memory_id)
            if not self._inverted[token]:
                del self._inverted[token]
        entity = entry.metadata.get("entity", "").lower()
        if entity:
            self._entity_index[entity].discard(memory_id)
            if not self._entity_index[entity]:
                del self._entity_index[entity]

    def _tfidf_scores(
        self,
        tokens: list[str],
        layer: Optional[MemoryLayer] = None,
    ) -> dict[str, float]:
        """Compute a simple TF-IDF score per candidate document."""
        num_docs = len(self._entries) or 1
        candidate_ids: set[str] = set()
        for token in tokens:
            candidate_ids.update(self._inverted.get(token, set()))

        if layer is not None:
            candidate_ids = {
                mid for mid in candidate_ids
                if mid in self._entries and self._entries[mid].layer == layer
            }

        scores: dict[str, float] = {}
        for mid in candidate_ids:
            entry = self._entries[mid]
            doc_tokens = _tokenise(entry.content)
            tf_sum = 0.0
            for token in tokens:
                tf = doc_tokens.count(token) / (len(doc_tokens) or 1)
                df = len(self._inverted.get(token, set()))
                idf = math.log((num_docs + 1) / (df + 1)) + 1.0
                tf_sum += tf * idf
            scores[mid] = tf_sum
        return scores


__all__ = ["SemanticMemory"]
