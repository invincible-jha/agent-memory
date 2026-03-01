# agent-memory

Agent memory and knowledge management with 4-layer cognitive architecture

[![CI](https://github.com/aumos-ai/agent-memory/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-memory/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-memory.svg)](https://pypi.org/project/agent-memory/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-memory.svg)](https://pypi.org/project/agent-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- Four-layer cognitive memory model — `WORKING` (active context), `EPISODIC` (events), `SEMANTIC` (facts and concepts), and `PROCEDURAL` (how-to knowledge) — each with its own storage, retrieval, and eviction semantics
- Importance scoring with five qualitative tiers (`CRITICAL` through `TRIVIAL`) and a numeric `[0, 1]` score, plus time-based decay that reduces importance automatically as memories age
- Contradiction detection and resolution: `ContradictionDetector` identifies conflicting `MemoryEntry` pairs and `ContradictionResolver` applies configurable resolution strategies
- Provenance tracking records the source and trust level of every memory item, with `reliability` scores that factor into retrieval ranking
- Three storage backends — in-memory, SQLite, and Redis — all behind the `MemoryStore` ABC; `UnifiedMemory` provides a single facade across all four layers
- `AutoMemorizeMiddleware` and `ContextBuilder` work together to automatically memorize tool outputs and construct retrieval-augmented context windows for LLM calls
- Freshness validation and forced refresh policies prevent stale semantic memories from being surfaced in time-sensitive retrieval queries

## Current Limitations

> **Transparency note**: We list known limitations to help you evaluate fit.

- **Vector Search**: TF-IDF only — no neural/embedding-based search yet. No pgvector or Pinecone integration.
- **Async**: Synchronous API only. Async storage backends not yet available.
- **Scale**: In-memory and SQLite backends; not designed for >100K entries without external storage.

## Quick Start

Install from PyPI:

```bash
pip install agent-memory
```

Verify the installation:

```bash
agent-memory version
```

Basic usage:

```python
import agent_memory

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

For production deployments requiring SLA-backed support and advanced
integrations, contact the maintainers or see the commercial extensions documentation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
