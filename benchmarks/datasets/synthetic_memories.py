"""Synthetic memory dataset generator with known ground truth.

Generates reproducible sets of MemoryEntry objects with deterministic
content, importance scores, and freshness scores. The ground truth mapping
is included so benchmark scripts can compute precision/recall exactly.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Optional

from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource


# ---------------------------------------------------------------------------
# Topic vocabulary — 20 distinct topics with 5 keywords each.
# Fixed vocabulary ensures deterministic retrieval ground truth.
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "python_async": ["async", "await", "coroutine", "event loop", "asyncio"],
    "database_indexing": ["index", "btree", "query plan", "scan", "postgres"],
    "machine_learning": ["gradient", "backpropagation", "epoch", "loss function", "tensor"],
    "kubernetes": ["pod", "namespace", "deployment", "ingress", "kubelet"],
    "oauth2": ["access token", "refresh token", "authorization code", "scope", "bearer"],
    "rest_api": ["endpoint", "http method", "status code", "json body", "rate limit"],
    "vector_search": ["embedding", "cosine similarity", "nearest neighbor", "faiss", "annoy"],
    "redis_cache": ["redis", "ttl", "eviction", "lua script", "pub sub"],
    "typescript": ["interface", "generic type", "union type", "type guard", "narrowing"],
    "docker": ["dockerfile", "layer", "container", "build context", "registry"],
    "react_hooks": ["useState", "useEffect", "custom hook", "dependency array", "memoization"],
    "sql_joins": ["inner join", "left join", "full outer join", "on clause", "cross join"],
    "encryption": ["aes256", "rsa keypair", "tls handshake", "cipher suite", "entropy"],
    "git_workflow": ["rebase", "merge commit", "cherry pick", "stash", "bisect"],
    "grpc": ["protobuf", "streaming rpc", "service definition", "channel", "interceptor"],
    "kafka": ["partition", "consumer group", "offset", "topic", "broker"],
    "monitoring": ["prometheus", "alert rule", "scrape interval", "cardinality", "metric"],
    "terraform": ["provider", "resource block", "state file", "plan output", "workspace"],
    "agent_planning": ["goal decomposition", "task queue", "dependency graph", "retry policy", "subtask"],
    "llm_prompting": ["system prompt", "few shot", "chain of thought", "temperature", "top_p"],
}

TOPIC_NAMES: list[str] = list(TOPIC_KEYWORDS.keys())


@dataclass
class GroundTruthEntry:
    """Associates a MemoryEntry with its canonical topic for evaluation."""

    entry: MemoryEntry
    topic: str
    query_keywords: list[str]


def _generate_content(topic: str, keywords: list[str], rng: random.Random) -> str:
    """Generate a deterministic sentence for a topic using its keywords."""
    keyword = rng.choice(keywords)
    templates = [
        f"The {keyword} is essential for understanding {topic.replace('_', ' ')} concepts.",
        f"When working with {topic.replace('_', ' ')}, remember that {keyword} matters most.",
        f"A common pattern in {topic.replace('_', ' ')} involves configuring the {keyword} correctly.",
        f"The {keyword} component of {topic.replace('_', ' ')} requires careful attention.",
        f"Experienced engineers use {keyword} as a core primitive in {topic.replace('_', ' ')}.",
    ]
    return rng.choice(templates)


def generate_memory_dataset(
    n_memories: int = 1000,
    seed: int = 42,
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
) -> tuple[list[MemoryEntry], dict[str, list[str]]]:
    """Generate a reproducible synthetic memory dataset.

    Parameters
    ----------
    n_memories:
        Total number of MemoryEntry objects to generate.
    seed:
        Random seed for reproducibility.
    layer:
        The MemoryLayer to assign to all generated entries.

    Returns
    -------
    tuple of:
        - list of MemoryEntry objects
        - dict mapping topic name to list of memory_ids for that topic
          (the ground truth for retrieval evaluation)
    """
    rng = random.Random(seed)
    entries: list[MemoryEntry] = []
    ground_truth: dict[str, list[str]] = {topic: [] for topic in TOPIC_NAMES}

    for index in range(n_memories):
        topic = TOPIC_NAMES[index % len(TOPIC_NAMES)]
        keywords = TOPIC_KEYWORDS[topic]
        content = _generate_content(topic, keywords, rng)
        memory_id = f"mem-{uuid.UUID(int=index + 1)}"

        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            layer=layer,
            importance_score=round(rng.uniform(0.3, 0.95), 3),
            freshness_score=round(rng.uniform(0.4, 1.0), 3),
            source=MemorySource.DOCUMENT,
        )
        entries.append(entry)
        ground_truth[topic].append(memory_id)

    return entries, ground_truth


def generate_query_for_topic(topic: str) -> str:
    """Return the primary search keyword for a topic.

    The first keyword in the topic's vocabulary is used as the canonical
    query. This ensures the query matches only entries from that topic.
    """
    return TOPIC_KEYWORDS[topic][0]


def generate_ground_truth_entries(
    n_memories: int = 100,
    seed: int = 42,
    layer: MemoryLayer = MemoryLayer.SEMANTIC,
) -> list[GroundTruthEntry]:
    """Generate GroundTruthEntry objects for precision/recall evaluation.

    Parameters
    ----------
    n_memories:
        Total number of entries to generate.
    seed:
        Random seed for reproducibility.
    layer:
        Memory layer for all entries.

    Returns
    -------
    list of GroundTruthEntry
        Each entry has its memory and the canonical topic+keywords.
    """
    entries, ground_truth = generate_memory_dataset(n_memories, seed, layer)
    result: list[GroundTruthEntry] = []
    memory_to_topic: dict[str, str] = {}
    for topic, memory_ids in ground_truth.items():
        for memory_id in memory_ids:
            memory_to_topic[memory_id] = topic

    for entry in entries:
        topic = memory_to_topic[entry.memory_id]
        result.append(
            GroundTruthEntry(
                entry=entry,
                topic=topic,
                query_keywords=TOPIC_KEYWORDS[topic],
            )
        )
    return result


__all__ = [
    "TOPIC_KEYWORDS",
    "TOPIC_NAMES",
    "GroundTruthEntry",
    "generate_memory_dataset",
    "generate_query_for_topic",
    "generate_ground_truth_entries",
]
