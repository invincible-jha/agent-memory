# agent-memory

Agent Memory & Knowledge Management — 4-layer cognitive memory, contradiction detection, forgetting policies.

[![CI](https://github.com/invincible-jha/agent-memory/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-memory/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/aumos-agent-memory.svg)](https://pypi.org/project/aumos-agent-memory/)
[![Python versions](https://img.shields.io/pypi/pyversions/aumos-agent-memory.svg)](https://pypi.org/project/aumos-agent-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/invincible-jha/agent-memory/blob/main/LICENSE)

---

## Installation

```bash
pip install aumos-agent-memory
```

Verify the installation:

```bash
agent-memory version
```

---

## Quick Start

```python
import agent_memory

# See examples/01_quickstart.py for a complete working example
```

---

## Key Features

- **Four-layer cognitive memory model** — `WORKING` (active context), `EPISODIC` (events), `SEMANTIC` (facts and concepts), and `PROCEDURAL` (how-to knowledge) — each with its own storage, retrieval, and eviction semantics
- **Importance scoring** with five qualitative tiers (`CRITICAL` through `TRIVIAL`) and a numeric `[0, 1]` score, plus time-based decay that reduces importance automatically as memories age
- **Contradiction detection and resolution** — `ContradictionDetector` identifies conflicting `MemoryEntry` pairs and `ContradictionResolver` applies configurable resolution strategies
- **Provenance tracking** records the source and trust level of every memory item, with `reliability` scores that factor into retrieval ranking
- **Three storage backends** — in-memory, SQLite, and Redis — all behind the `MemoryStore` ABC; `UnifiedMemory` provides a single facade across all four layers
- **`AutoMemorizeMiddleware` and `ContextBuilder`** work together to automatically memorize tool outputs and construct retrieval-augmented context windows for LLM calls
- **Freshness validation** and forced refresh policies prevent stale semantic memories from being surfaced in time-sensitive retrieval queries

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-memory)
- [PyPI Package](https://pypi.org/project/aumos-agent-memory/)
- [Architecture](architecture.md)
- [Contributing](https://github.com/invincible-jha/agent-memory/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/invincible-jha/agent-memory/blob/main/CHANGELOG.md)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/invincible-jha/agent-memory/blob/main/LICENSE) for full terms.

---

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure.
