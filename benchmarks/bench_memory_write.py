"""Benchmark: Write throughput (memories/second) for agent-memory.

Measures how quickly the InMemoryStorage backend can accept new entries.
Reports memories/second and write latency p50/p95/p99.

Competitor context
------------------
Mem0's managed cloud backend introduces network round-trips not present in
in-process storage. This benchmark measures the local storage layer only,
establishing the upper bound before network latency is added.
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
from agent_memory.storage.memory_store import InMemoryStorage
from datasets.synthetic_memories import generate_memory_dataset


def run_benchmark(
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Measure write throughput and per-write latency.

    Parameters
    ----------
    n_iterations:
        Number of individual write operations to measure.
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with throughput (memories/sec) and per-write latency stats.
    """
    entries, _ = generate_memory_dataset(
        n_memories=n_iterations, seed=seed, layer=MemoryLayer.SEMANTIC
    )
    store = InMemoryStorage()

    individual_latencies_ms: list[float] = []

    # Warmup — write 10 entries to prime any interpreter JIT paths
    warmup_count = min(10, len(entries))
    for entry in entries[:warmup_count]:
        store.save(entry)
    store.clear()

    # Timed run
    overall_start = time.perf_counter()
    for entry in entries:
        start = time.perf_counter()
        store.save(entry)
        elapsed_ms = (time.perf_counter() - start) * 1000
        individual_latencies_ms.append(elapsed_ms)
    overall_elapsed = time.perf_counter() - overall_start

    throughput = n_iterations / overall_elapsed
    sorted_lats = sorted(individual_latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "memory_write_throughput",
        "n_iterations": n_iterations,
        "seed": seed,
        "total_elapsed_seconds": round(overall_elapsed, 4),
        "throughput_memories_per_second": round(throughput, 1),
        "per_write_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4),
            "p95": round(sorted_lats[int(n * 0.95)], 4),
            "p99": round(sorted_lats[min(int(n * 0.99), n - 1)], 4),
            "mean": round(statistics.mean(individual_latencies_ms), 4),
            "min": round(min(individual_latencies_ms), 4),
            "max": round(max(individual_latencies_ms), 4),
        },
    }


if __name__ == "__main__":
    print("Running memory write throughput benchmark (n=1000 writes)...")
    result = run_benchmark(n_iterations=1000)
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "write_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
