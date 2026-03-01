# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Minimal working example with the Memory convenience class |
| 02 | [Configuration](02_configuration.py) | Custom storage, importance scoring, and freshness decay |
| 03 | [Vector Search](03_vector_search.py) | Similarity-based retrieval with SQLiteVectorStore |
| 04 | [Contradiction Detection](04_contradiction_detection.py) | Detect and resolve conflicting memories |
| 05 | [Layer Management](05_layer_management.py) | Working, episodic, semantic, and procedural layers |
| 06 | [LangChain Memory](06_langchain_memory.py) | agent-memory as LangChain conversation backing store |
| 07 | [CrewAI Memory](07_crewai_memory.py) | Shared memory across multiple CrewAI agents |

## Running the examples

```bash
pip install agent-memory
python examples/01_quickstart.py
```

For framework integrations:

```bash
pip install langchain    # for example 06
pip install crewai       # for example 07
```
