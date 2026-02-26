"""Unit tests for SemanticMemory — TF-IDF keyword retrieval and entity linking."""

from __future__ import annotations

import pytest

from agent_memory.memory.semantic import SemanticMemory, _tokenise
from agent_memory.memory.types import MemoryEntry, MemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sem(content: str, entity: str = "", **kwargs: object) -> MemoryEntry:
    metadata: dict[str, str] = {}
    if entity:
        metadata["entity"] = entity
    return MemoryEntry(
        content=content,
        layer=MemoryLayer.SEMANTIC,
        metadata=metadata,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_lowercases_input(self) -> None:
        tokens = _tokenise("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stop_words(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_strips_punctuation(self) -> None:
        tokens = _tokenise("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_returns_empty_for_all_stop_words(self) -> None:
        tokens = _tokenise("the is are was")
        assert tokens == []

    def test_filters_single_char_tokens(self) -> None:
        tokens = _tokenise("a b c python")
        # single letters removed, 'python' kept
        assert "python" in tokens
        assert "a" not in tokens


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestSemanticMemoryCRUD:
    def test_store_and_retrieve(self) -> None:
        mem = SemanticMemory()
        entry = _sem("Python is a programming language")
        mem.store(entry)
        result = mem.retrieve(entry.memory_id)
        assert result is not None
        assert result.content == "Python is a programming language"

    def test_retrieve_missing_id_returns_none(self) -> None:
        mem = SemanticMemory()
        assert mem.retrieve("no-such-id") is None

    def test_store_replaces_and_updates_index(self) -> None:
        mem = SemanticMemory()
        entry = _sem("cats are mammals")
        mem.store(entry)
        updated = entry.model_copy(update={"content": "dogs are mammals"})
        mem.store(updated)
        # Old content should not be findable
        old_results = mem.search("cats")
        assert not any(r.memory_id == entry.memory_id for r in old_results)
        # New content should be findable
        new_results = mem.search("dogs")
        assert any(r.memory_id == updated.memory_id for r in new_results)

    def test_delete_existing(self) -> None:
        mem = SemanticMemory()
        entry = _sem("to delete")
        mem.store(entry)
        assert mem.delete(entry.memory_id) is True
        assert mem.retrieve(entry.memory_id) is None

    def test_delete_cleans_inverted_index(self) -> None:
        mem = SemanticMemory()
        entry = _sem("unique keyword only here")
        mem.store(entry)
        mem.delete(entry.memory_id)
        results = mem.search("unique keyword")
        assert len(results) == 0

    def test_delete_missing_returns_false(self) -> None:
        mem = SemanticMemory()
        assert mem.delete("ghost") is False

    def test_clear_all(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("a"))
        mem.store(_sem("b"))
        count = mem.clear()
        assert count == 2
        assert mem.count() == 0

    def test_clear_by_layer(self) -> None:
        mem = SemanticMemory()
        sem = _sem("semantic")
        ep = MemoryEntry(content="episodic", layer=MemoryLayer.EPISODIC)
        mem.store(sem)
        mem.store(ep)
        mem.clear(layer=MemoryLayer.SEMANTIC)
        assert mem.retrieve(sem.memory_id) is None
        assert mem.retrieve(ep.memory_id) is not None


# ---------------------------------------------------------------------------
# TF-IDF search
# ---------------------------------------------------------------------------


class TestSemanticMemorySearch:
    def test_search_finds_relevant_entries(self) -> None:
        mem = SemanticMemory()
        relevant = _sem("neural networks deep learning machine learning")
        irrelevant = _sem("classical music orchestra concert")
        mem.store(relevant)
        mem.store(irrelevant)
        results = mem.search("deep learning")
        ids = [r.memory_id for r in results]
        assert relevant.memory_id in ids
        assert irrelevant.memory_id not in ids

    def test_search_empty_query_returns_empty(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("some content"))
        results = mem.search("")
        assert results == []

    def test_search_stop_words_only_returns_empty(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("python programming language"))
        results = mem.search("the is are")
        assert len(results) == 0

    def test_search_respects_limit(self) -> None:
        mem = SemanticMemory()
        for i in range(10):
            mem.store(_sem(f"python programming language version {i}"))
        results = mem.search("python", limit=3)
        assert len(results) <= 3

    def test_search_with_layer_filter(self) -> None:
        mem = SemanticMemory()
        sem = _sem("python programming")
        ep = MemoryEntry(content="python programming", layer=MemoryLayer.EPISODIC)
        mem.store(sem)
        mem.store(ep)
        results = mem.search("python", layer=MemoryLayer.SEMANTIC)
        assert all(r.layer is MemoryLayer.SEMANTIC for r in results)

    def test_higher_tf_idf_entry_ranked_first(self) -> None:
        mem = SemanticMemory()
        # Entry with more query-term density
        dense = _sem("python python python programming")
        sparse = _sem("python is used in data science")
        mem.store(dense)
        mem.store(sparse)
        results = mem.search("python")
        assert len(results) >= 1
        # Dense entry should rank first
        assert results[0].memory_id == dense.memory_id


# ---------------------------------------------------------------------------
# Entity-linked retrieval
# ---------------------------------------------------------------------------


class TestSemanticMemoryEntityIndex:
    def test_retrieve_by_entity(self) -> None:
        mem = SemanticMemory()
        entry = _sem("Python is a high-level language", entity="Python")
        mem.store(entry)
        results = mem.retrieve_by_entity("Python")
        assert any(r.memory_id == entry.memory_id for r in results)

    def test_retrieve_by_entity_case_insensitive(self) -> None:
        mem = SemanticMemory()
        entry = _sem("facts about python", entity="Python")
        mem.store(entry)
        results = mem.retrieve_by_entity("python")
        assert any(r.memory_id == entry.memory_id for r in results)

    def test_retrieve_by_entity_unknown_entity_returns_empty(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("some fact", entity="Python"))
        assert mem.retrieve_by_entity("UnknownEntity") == []

    def test_linked_entities_lists_known_entities(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("fact about python", entity="python"))
        mem.store(_sem("fact about rust", entity="rust"))
        entities = mem.linked_entities()
        assert "python" in entities
        assert "rust" in entities

    def test_linked_entities_sorted(self) -> None:
        mem = SemanticMemory()
        mem.store(_sem("fact", entity="zebra"))
        mem.store(_sem("fact", entity="alpha"))
        entities = mem.linked_entities()
        assert entities == sorted(entities)

    def test_delete_removes_entity_from_index(self) -> None:
        mem = SemanticMemory()
        entry = _sem("about python", entity="python")
        mem.store(entry)
        mem.delete(entry.memory_id)
        assert "python" not in mem.linked_entities()
