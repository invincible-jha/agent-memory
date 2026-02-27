"""Comparison visualiser for agent-memory benchmark results.

Loads JSON result files from the results/ directory and prints a formatted
comparison table. Adds known competitor reference numbers where available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Competitor reference data (from public sources only)
# ---------------------------------------------------------------------------

COMPETITOR_NOTES = {
    "retrieval_precision": (
        "Mem0 reports 26% accuracy improvement (precision-based) over naive "
        "retrieval in their 2024-11 README benchmark using embedding-based search. "
        "Source: https://github.com/mem0ai/mem0 README, Nov 2024."
    ),
    "retrieval_latency": (
        "Letta (MemGPT) uses tiered paged storage with O(n) scan costs for "
        "cold memory tiers. No published latency numbers for direct comparison."
    ),
    "memory_write_throughput": (
        "Mem0 cloud SDK incurs network round-trips (~50-200ms per write). "
        "In-process stores are not directly compared in their published benchmarks."
    ),
}


def _fmt_table(rows: list[tuple[str, str]], title: str) -> None:
    """Print a simple two-column table."""
    col1_width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, value in rows:
        print(f"  {key:<{col1_width}} {value}")


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


def display_precision_results(data: dict[str, object]) -> None:
    """Display precision/recall benchmark results."""
    metrics = data.get("metrics", {})
    rows: list[tuple[str, str]] = []
    for k in [1, 5, 10]:
        rows.append((f"Precision@{k}", f"{metrics.get(f'mean_precision_at_{k}', 'N/A'):.4f}"))
        rows.append((f"Recall@{k}", f"{metrics.get(f'mean_recall_at_{k}', 'N/A'):.4f}"))
        rows.append((f"F1@{k}", f"{metrics.get(f'mean_f1_at_{k}', 'N/A'):.4f}"))
    lat = data.get("query_latency_ms", {})
    rows.append(("Query latency p50 (ms)", str(lat.get("p50", "N/A"))))
    rows.append(("Query latency p95 (ms)", str(lat.get("p95", "N/A"))))
    rows.append(("Query latency p99 (ms)", str(lat.get("p99", "N/A"))))
    _fmt_table(rows, "Retrieval Precision Results")
    print(f"\n  Note: {COMPETITOR_NOTES['retrieval_precision']}")


def display_latency_results(data: dict[str, object]) -> None:
    """Display latency benchmark results across store sizes."""
    size_results = data.get("results_by_store_size", {})
    rows: list[tuple[str, str]] = []
    for size, stats in size_results.items():
        rows.append((f"Store size {size} — p50 (ms)", str(stats.get("p50_ms"))))
        rows.append((f"Store size {size} — p95 (ms)", str(stats.get("p95_ms"))))
        rows.append((f"Store size {size} — p99 (ms)", str(stats.get("p99_ms"))))
    _fmt_table(rows, "Retrieval Latency by Store Size")
    print(f"\n  Note: {COMPETITOR_NOTES['retrieval_latency']}")


def display_write_results(data: dict[str, object]) -> None:
    """Display write throughput benchmark results."""
    lat = data.get("per_write_latency_ms", {})
    rows: list[tuple[str, str]] = [
        ("Throughput (memories/sec)", str(data.get("throughput_memories_per_second"))),
        ("Per-write p50 (ms)", str(lat.get("p50"))),
        ("Per-write p95 (ms)", str(lat.get("p95"))),
        ("Per-write p99 (ms)", str(lat.get("p99"))),
        ("Total elapsed (sec)", str(data.get("total_elapsed_seconds"))),
    ]
    _fmt_table(rows, "Write Throughput Results")
    print(f"\n  Note: {COMPETITOR_NOTES['memory_write_throughput']}")


def main() -> None:
    """Load all result files and display comparison tables."""
    results_dir = Path(__file__).parent / "results"

    baseline = _load_json(results_dir / "baseline.json")
    latency = _load_json(results_dir / "latency_baseline.json")
    write = _load_json(results_dir / "write_baseline.json")

    if baseline:
        display_precision_results(baseline)
    else:
        print("No baseline.json found. Run bench_retrieval_precision.py first.")

    if latency:
        display_latency_results(latency)
    else:
        print("No latency_baseline.json found. Run bench_retrieval_latency.py first.")

    if write:
        display_write_results(write)
    else:
        print("No write_baseline.json found. Run bench_memory_write.py first.")

    print("\n" + "=" * 60)
    print("  To regenerate all results:")
    print("    python benchmarks/bench_retrieval_precision.py")
    print("    python benchmarks/bench_retrieval_latency.py")
    print("    python benchmarks/bench_memory_write.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
