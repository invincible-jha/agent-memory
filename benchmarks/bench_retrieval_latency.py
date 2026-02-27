"""Benchmark: Retrieval latency at different store sizes for agent-memory.

Measures query latency (p50/p95/p99) with store sizes of 100, 1K, and 10K.
This characterises how the substring search scales with data volume.

Competitor context
------------------
Letta's published architecture notes O(n) scan costs for non-indexed stores.
Mem0 uses vector embeddings (requires external model) — not comparable to
pure in-process substring matching. This baseline is for the built-in store.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_memory.memory.types import MemoryLayer
from conftest import build_store
from datasets.synthetic_memories import TOPIC_NAMES, generate_query_for_topic


def _measure_query_latencies(
    store_size: int,
    n_queries: int,
    seed: int,
) -> dict[str, float]:
    """Populate a store of store_size entries and run n_queries, return stats.

    Parameters
    ----------
    store_size:
        Number of memories in the store.
    n_queries:
        Number of search operations to measure.
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with p50/p95/p99/mean latencies in milliseconds.
    """
    store = build_store(n_memories=store_size, seed=seed)
    queries = [generate_query_for_topic(t) for t in TOPIC_NAMES] * (
        n_queries // len(TOPIC_NAMES) + 1
    )
    queries = queries[:n_queries]

    latencies_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        store.search(query, layer=MemoryLayer.SEMANTIC, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    return {
        "p50_ms": round(sorted_lats[int(n * 0.50)], 4),
        "p95_ms": round(sorted_lats[int(n * 0.95)], 4),
        "p99_ms": round(sorted_lats[min(int(n * 0.99), n - 1)], 4),
        "mean_ms": round(statistics.mean(latencies_ms), 4),
        "min_ms": round(min(latencies_ms), 4),
        "max_ms": round(max(latencies_ms), 4),
    }


def run_benchmark(
    store_sizes: list[int] | None = None,
    n_queries: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Measure retrieval latency across multiple store sizes.

    Parameters
    ----------
    store_sizes:
        List of store sizes to evaluate. Defaults to [100, 1000, 10000].
    n_queries:
        Queries to run per store size.
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with latency stats per store size.
    """
    if store_sizes is None:
        store_sizes = [100, 1_000, 10_000]

    size_results: dict[str, dict[str, float]] = {}
    for size in store_sizes:
        print(f"  Measuring store_size={size:,} with {n_queries} queries...")
        size_results[str(size)] = _measure_query_latencies(
            store_size=size,
            n_queries=n_queries,
            seed=seed,
        )

    return {
        "benchmark": "retrieval_latency",
        "store_sizes": store_sizes,
        "n_queries_per_size": n_queries,
        "seed": seed,
        "results_by_store_size": size_results,
    }


if __name__ == "__main__":
    print("Running retrieval latency benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "latency_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
