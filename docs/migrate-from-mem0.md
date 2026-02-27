# Migrating from Mem0 to agent-memory

agent-memory is a local-first, cognitive-layer memory library for AI agents. It models memory the
way cognitive science describes it — with four distinct layers (working, episodic, semantic,
procedural), automatic importance scoring, forgetting policies, and built-in contradiction
detection. If you are using Mem0 today, this guide shows how to migrate incrementally or run both
in coexistence.

---

## Feature Comparison

| Capability | Mem0 | agent-memory |
|---|---|---|
| Storage architecture | Single vector store | 4-layer cognitive model: working / episodic / semantic / procedural |
| Vector search | Yes (cloud API) | Yes — via `MemorySearchEngine` (pluggable backend) |
| Auto-categorization | Yes — cloud AI | Configurable `ImportanceScorer` and `MemoryLayer` assignment |
| Contradiction detection | No | Yes — `ContradictionDetector` with entity-attribute triple matching and TF-IDF cosine similarity |
| Forgetting / decay | No | Yes — `ExponentialDecay`, `LinearDecay`, `StepDecay`, configurable per layer |
| Knowledge graph traversal | No | Planned via provenance graph |
| Provenance tracking | No | Yes — `ProvenanceTracker` with `SourceReliability` scoring |
| Freshness scoring | No | Yes — `FreshnessScorer` and `StaleMemoryDetector` |
| Self-hosted / local | No (cloud) | Yes — SQLite or in-memory out of the box; Redis optional |
| Python API | Yes | Yes |
| LangChain integration | Yes | Yes — via `AutoMemorizeMiddleware` |
| Multi-agent isolation | Workspace-based (cloud) | Per-agent `UnifiedMemory` instances with isolated storage |
| Structured knowledge | No | Yes — `SemanticMemory` with graph-like entity linking |
| Procedural memory | No | Yes — `ProceduralMemory` with `Procedure` and `ProcedureStep` |

---

## Installation

```bash
# Core install — SQLite storage, no external services required
pip install agent-memory

# With Redis storage backend
pip install "agent-memory[redis]"
```

---

## Step 1 — Replace Mem0 Client Initialization

**Before (Mem0):**

```python
from mem0 import Memory

memory = Memory()
# or with configuration
memory = Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    }
})
```

**After (agent-memory — in-memory quickstart):**

```python
from agent_memory import Memory

# Zero-config in-process memory — no external service required
memory = Memory()
```

**After (agent-memory — SQLite persistence):**

```python
from agent_memory import UnifiedMemory, MemoryConfig
from agent_memory.storage.sqlite_store import SQLiteStorage

config = MemoryConfig(agent_id="agent-001", max_working_memory=100)
storage = SQLiteStorage(db_path="agent_memory.db")

memory = UnifiedMemory(config=config, storage=storage)
```

---

## Step 2 — Replace add() / search() / get_all()

Mem0 uses a flat `add` / `search` / `get_all` interface. agent-memory maps those to `store` and
`search` on `UnifiedMemory`, but adds layer-aware routing.

**Before (Mem0):**

```python
# Add a memory
memory.add("The project deadline is March 15th", user_id="alice")

# Search memories
results = memory.search("project deadline", user_id="alice")
for result in results:
    print(result["memory"])

# Get all memories
all_memories = memory.get_all(user_id="alice")
```

**After (agent-memory):**

```python
from agent_memory import UnifiedMemory, MemoryEntry, MemoryLayer, MemorySource

memory = UnifiedMemory()

# Store — choose the appropriate cognitive layer
entry = MemoryEntry(
    content="The project deadline is March 15th",
    layer=MemoryLayer.EPISODIC,       # or SEMANTIC, WORKING, PROCEDURAL
    source=MemorySource.CONVERSATION,
    agent_id="agent-001",
    tags=["project", "deadline"],
)
memory.store(entry)

# Search — hybrid retrieval across all layers
results = memory.search("project deadline", top_k=5)
for result in results:
    print(result.entry.content, result.score)
```

---

## Step 3 — Use the 4-Layer Cognitive Architecture

The four layers are not just organizational — they have different decay rates, retrieval weights,
and lifecycle behaviors.

```python
from agent_memory import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    Procedure,
    ProcedureStep,
)

# Working memory — active context, short-lived, cleared between sessions
working = WorkingMemory(capacity=10)
working.store(MemoryEntry(
    content="Current task: summarize meeting notes",
    layer=MemoryLayer.WORKING,
    source=MemorySource.SYSTEM,
))

# Episodic memory — what happened, when, in what order
episodic = EpisodicMemory()
episodic.store(MemoryEntry(
    content="User asked about Q3 revenue on 2026-01-15",
    layer=MemoryLayer.EPISODIC,
    source=MemorySource.CONVERSATION,
))

# Semantic memory — facts and relationships
semantic = SemanticMemory()
semantic.store(MemoryEntry(
    content="Paris is the capital of France",
    layer=MemoryLayer.SEMANTIC,
    source=MemorySource.DOCUMENT,
    tags=["geography", "europe"],
))

# Procedural memory — how to perform a multi-step task
procedural = ProceduralMemory()
procedure = Procedure(
    name="file_expense_report",
    steps=[
        ProcedureStep(order=1, action="Collect all receipts"),
        ProcedureStep(order=2, action="Enter amounts in expense form"),
        ProcedureStep(order=3, action="Submit for manager approval"),
    ],
)
procedural.store_procedure(procedure)
```

---

## Step 4 — Enable Contradiction Detection

This has no equivalent in Mem0. When agents accumulate memories from multiple sources, facts can
contradict each other. agent-memory detects this automatically.

```python
from agent_memory import (
    UnifiedMemory,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    ContradictionScanner,
    ScanScope,
)

memory = UnifiedMemory()

# Store two contradictory facts
memory.store(MemoryEntry(
    content="The meeting is on Monday at 2pm",
    layer=MemoryLayer.EPISODIC,
    source=MemorySource.CONVERSATION,
))
memory.store(MemoryEntry(
    content="The meeting is on Tuesday at 3pm",
    layer=MemoryLayer.EPISODIC,
    source=MemorySource.EMAIL,
))

# Scan for contradictions
scanner = ContradictionScanner()
scan_result = scanner.scan(
    entries=memory.get_all(),
    scope=ScanScope.FULL,
)

for pair in scan_result.contradiction_pairs:
    print(f"Contradiction: {pair.entry_a.content!r} vs {pair.entry_b.content!r}")
    print(f"  Confidence: {pair.confidence:.2f}")
```

You can also resolve contradictions by source reliability:

```python
from agent_memory import ContradictionResolver, ResolutionStrategy

resolver = ContradictionResolver(strategy=ResolutionStrategy.MOST_RECENT)
resolution = resolver.resolve(pair)
print(f"Winner: {resolution.winner.content}")
print(f"Rationale: {resolution.rationale}")
```

---

## Step 5 — Configure Forgetting Policies

Mem0 retains everything indefinitely. agent-memory supports configurable decay so working memory
does not accumulate stale context.

```python
from agent_memory import (
    ImportanceScorer,
    ExponentialDecay,
    MemoryGarbageCollector,
    MemoryEntry,
    MemoryLayer,
    MemorySource,
)

# Exponential decay — memories lose importance over time
decay = ExponentialDecay(half_life_days=7.0)
scorer = ImportanceScorer(decay_curve=decay)

entry = MemoryEntry(
    content="Remind me to call Sarah",
    layer=MemoryLayer.WORKING,
    source=MemorySource.USER,
)
current_importance = scorer.score(entry)

# Garbage collector removes entries below an importance threshold
collector = MemoryGarbageCollector(scorer=scorer, min_importance=0.1)
entries = memory.get_all()
retained, removed = collector.collect(entries)
print(f"Retained {len(retained)} memories, removed {len(removed)}")
```

---

## Step 6 — Auto-Memorize Middleware (LangChain / any agent loop)

```python
from agent_memory import AutoMemorizeMiddleware, Interaction, UnifiedMemory

memory = UnifiedMemory()
middleware = AutoMemorizeMiddleware(memory=memory, agent_id="agent-001")

# Wrap your agent loop
async def agent_loop(user_input: str) -> str:
    interaction = Interaction(user_message=user_input, agent_id="agent-001")
    response = await call_llm(user_input)
    interaction.agent_response = response

    # Automatically extracts and stores salient facts from the exchange
    await middleware.process(interaction)
    return response
```

---

## Coexistence: Adding Contradiction Detection on Top of Mem0

If you want to keep Mem0 as your primary vector store but gain contradiction detection and
forgetting policies, pull memories from Mem0 into agent-memory entries and run the scanner
as a post-retrieval step.

```bash
pip install agent-memory mem0ai
```

```python
from mem0 import Memory as Mem0Memory
from agent_memory import (
    MemoryEntry,
    MemoryLayer,
    MemorySource,
    ContradictionScanner,
    ScanScope,
)

mem0 = Mem0Memory()
scanner = ContradictionScanner()

def search_with_contradiction_check(query: str, user_id: str) -> list[MemoryEntry]:
    # Retrieve from Mem0 as normal
    raw_results = mem0.search(query, user_id=user_id)

    # Wrap raw dicts as lightweight MemoryEntry objects
    entries = [
        MemoryEntry(
            content=r["memory"],
            layer=MemoryLayer.SEMANTIC,
            source=MemorySource.USER,
        )
        for r in raw_results
    ]

    # Check for contradictions before returning to the agent
    scan = scanner.scan(entries, scope=ScanScope.FULL)
    if scan.contradiction_pairs:
        print(f"Warning: {len(scan.contradiction_pairs)} contradictions in retrieved memories")

    return entries
```

This pattern requires no changes to your Mem0 setup and adds contradiction awareness to any
existing agent.

---

## What You Gain by Switching

1. **4-layer cognitive architecture** — working, episodic, semantic, and procedural layers model
   how agents actually use memory, with separate decay rates and retrieval weights per layer.
2. **Contradiction detection** — entity-attribute triple matching and TF-IDF cosine similarity
   automatically flag conflicting facts, which is critical when memories come from multiple sources.
3. **Forgetting policies** — configurable decay curves (exponential, linear, step) prevent working
   memory from becoming a growing list of stale context.
4. **Provenance tracking** — every memory entry records its source and reliability score, so the
   agent can prefer authoritative sources when conflicts arise.
5. **Freshness scoring** — `StaleMemoryDetector` flags entries that have not been reinforced within
   a configurable window.
6. **No cloud dependency** — all storage is local (SQLite, in-memory, or Redis). Your agent's
   memory never leaves your infrastructure.

## What You Keep

- Mem0's vector search results are fully compatible as input to `MemoryEntry` objects, so you can
  migrate incrementally without rewriting retrieval logic.
- If you use LangChain, the `AutoMemorizeMiddleware` wraps your existing agent loop and does not
  require changes to chain structure.
- SQLite storage is schema-versioned, so you can export existing Mem0 memories to a SQLite file
  and pick up where you left off.
