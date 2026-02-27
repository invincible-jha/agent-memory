"""Test that the 3-line quickstart API works for agent-memory."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_memory import Memory

    m = Memory()
    assert m is not None


def test_quickstart_add() -> None:
    from agent_memory import Memory

    m = Memory()
    entry = m.add("The sky is blue", user_id="user-1")
    assert entry is not None


def test_quickstart_search() -> None:
    from agent_memory import Memory

    m = Memory()
    m.add("Python is a programming language", user_id="alice")
    results = m.search("programming language")
    assert isinstance(results, list)


def test_quickstart_search_finds_stored() -> None:
    from agent_memory import Memory

    m = Memory()
    m.add("Elephants are the largest land animals", user_id="test-user")
    results = m.search("elephants")
    assert len(results) >= 1


def test_quickstart_clear() -> None:
    from agent_memory import Memory

    m = Memory()
    m.add("Some text to clear", user_id="user-x")
    m.clear()
    results = m.search("text")
    assert len(results) == 0


def test_quickstart_repr() -> None:
    from agent_memory import Memory

    m = Memory()
    assert "Memory" in repr(m)
