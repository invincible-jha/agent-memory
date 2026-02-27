# agent-memory Benchmarks

Reproducible benchmark suite for agent-memory retrieval precision, latency, and write throughput.

## Quick Start

Run all benchmarks from the repo root:

```bash
cd repos/agent-memory
python benchmarks/bench_retrieval_precision.py
python benchmarks/bench_retrieval_latency.py
python benchmarks/bench_memory_write.py
python benchmarks/compare.py
```

## Benchmarks

### bench_retrieval_precision.py

**What it measures:** Precision@k, Recall@k, and F1@k (k = 1, 5, 10) for the InMemoryStorage substring search.

**Method:**
1. Generates 1,000 synthetic MemoryEntry objects across 20 topics (50 per topic).
2. Each topic has a canonical query keyword embedded in all its entries.
3. Runs the query for each topic and measures how many top-k results belong to the correct topic.

**Key metrics:**
- `mean_precision_at_1` — fraction of searches where the #1 result is correct
- `mean_precision_at_5` — fraction of the top-5 results that are correct
- `mean_f1_at_10` — harmonic mean of precision and recall at top-10

**Competitor reference:**
Mem0 reports 26% accuracy improvement over naive retrieval (embedding-based, requires external model).
Source: https://github.com/mem0ai/mem0 README, November 2024.

---

### bench_retrieval_latency.py

**What it measures:** Query latency (p50/p95/p99/mean) at store sizes of 100, 1K, and 10K entries.

**Method:**
200 search queries are issued per store size. Each query is timed individually.

**Key metrics:**
- `p50_ms`, `p95_ms`, `p99_ms` — latency percentiles in milliseconds
- Scaling behaviour from 100 to 10K entries (linear for substring scan)

---

### bench_memory_write.py

**What it measures:** Write throughput (memories/second) and per-write latency.

**Method:**
1,000 MemoryEntry objects are written one-at-a-time. A 10-entry warmup is discarded.

**Key metrics:**
- `throughput_memories_per_second` — total write rate
- `per_write_latency_ms` p50/p95/p99

---

## Interpreting Results

- Results are saved to `results/` as JSON files.
- Use `compare.py` to display all results in a formatted table with competitor notes.
- All datasets are synthetic (no downloads required) with a fixed `seed=42`.
- Results should be stable across runs on the same hardware within ~5%.

## Dataset Design

`datasets/synthetic_memories.py` generates entries with exactly one topic keyword per entry.
This makes ground truth trivial to compute: a search for topic keyword T should return
all entries from topic T and nothing else. This gives a clear precision/recall ceiling.

## Competitor Numbers (public only)

| Competitor | Metric | Source |
|------------|--------|--------|
| Mem0 | 26% accuracy improvement (embedding-based) | GitHub README, Nov 2024 |
| Letta | O(n) scan for cold storage (no published ms numbers) | Architecture docs |

Note: Mem0's accuracy metric uses embedding similarity which requires an external model.
The baseline here uses substring matching — a different (simpler) retrieval strategy.
