"""Benchmark: Retrieval precision@k for agent-memory.

Inserts N memories with known ground-truth topic labels, then queries
with the canonical keyword for each topic and measures precision@1/5/10.

Competitor context
------------------
- Mem0 claims 26% accuracy improvement over naive retrieval in their
  published benchmarks (https://github.com/mem0ai/mem0, README, 2024-11).
- Letta (formerly MemGPT) uses tiered storage but publishes no precision@k.

This benchmark establishes a baseline for agent-memory's substring-based
retrieval so future embedding-based improvements can be measured against it.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Bootstrap path before any local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_memory.memory.types import MemoryLayer
from agent_memory.storage.memory_store import InMemoryStorage
from conftest import build_store
from datasets.synthetic_memories import (
    TOPIC_NAMES,
    generate_memory_dataset,
    generate_query_for_topic,
)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute precision@k: fraction of top-k results that are relevant.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of memory_ids returned by the search.
    relevant_ids:
        Set of memory_ids that are relevant (same topic).
    k:
        Cut-off rank.

    Returns
    -------
    float
        Precision in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute recall@k: fraction of relevant items found in top-k.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of memory_ids returned by the search.
    relevant_ids:
        Complete set of relevant memory_ids.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Recall in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def run_benchmark(
    n_memories: int = 1000,
    k_values: list[int] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Measure retrieval precision@k and recall@k.

    Parameters
    ----------
    n_memories:
        Total memories inserted into the store.
    k_values:
        List of k cut-offs to evaluate. Defaults to [1, 5, 10].
    seed:
        Fixed seed for reproducibility.

    Returns
    -------
    dict
        Benchmark results with precision/recall per k value.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    entries, ground_truth = generate_memory_dataset(
        n_memories=n_memories, seed=seed, layer=MemoryLayer.SEMANTIC
    )
    store = InMemoryStorage()
    for entry in entries:
        store.save(entry)

    per_topic_results: list[dict[str, float]] = []
    latencies_ms: list[float] = []

    for topic in TOPIC_NAMES:
        query = generate_query_for_topic(topic)
        relevant_ids: set[str] = set(ground_truth.get(topic, []))
        if not relevant_ids:
            continue

        start = time.perf_counter()
        results = store.search(query, layer=MemoryLayer.SEMANTIC, limit=max(k_values))
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        retrieved_ids = [entry.memory_id for entry in results]
        topic_scores: dict[str, float] = {}
        for k in k_values:
            topic_scores[f"precision_at_{k}"] = precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            topic_scores[f"recall_at_{k}"] = recall_at_k(
                retrieved_ids, relevant_ids, k
            )
        per_topic_results.append(topic_scores)

    # Aggregate across all topics
    aggregate: dict[str, float] = {}
    for k in k_values:
        precisions = [r[f"precision_at_{k}"] for r in per_topic_results]
        recalls = [r[f"recall_at_{k}"] for r in per_topic_results]
        f1_scores: list[float] = []
        for p, r in zip(precisions, recalls):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0.0)
        aggregate[f"mean_precision_at_{k}"] = round(statistics.mean(precisions), 4)
        aggregate[f"mean_recall_at_{k}"] = round(statistics.mean(recalls), 4)
        aggregate[f"mean_f1_at_{k}"] = round(statistics.mean(f1_scores), 4)

    n_latencies = len(latencies_ms)
    sorted_latencies = sorted(latencies_ms)

    return {
        "benchmark": "retrieval_precision",
        "n_memories": n_memories,
        "n_topics": len(TOPIC_NAMES),
        "k_values": k_values,
        "seed": seed,
        "metrics": aggregate,
        "query_latency_ms": {
            "p50": round(sorted_latencies[int(n_latencies * 0.50)], 3),
            "p95": round(sorted_latencies[int(n_latencies * 0.95)], 3),
            "p99": round(
                sorted_latencies[min(int(n_latencies * 0.99), n_latencies - 1)], 3
            ),
            "mean": round(statistics.mean(latencies_ms), 3),
        },
    }


if __name__ == "__main__":
    print("Running retrieval precision benchmark (n=1000 memories)...")
    result = run_benchmark(n_memories=1000)
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
