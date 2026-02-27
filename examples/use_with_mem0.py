"""Example: Using agent-memory with Mem0.

Install with:
    pip install "aumos-agent-memory[mem0]"

This example demonstrates wrapping Mem0's memory API with
agent-memory's contradiction detection and hybrid re-ranking.
New memories are checked for contradictions with existing ones,
and retrieval results are re-ranked using recency and relevance.
"""
from __future__ import annotations

# from agent_memory.integrations.mem0_adapter import (
#     Mem0EnhancedMemory,
#     ContradictionWarning,
# )

# --- Initialize enhanced memory ---
# memory = Mem0EnhancedMemory(
#     mem0_api_key="m0-...",  # Your Mem0 API key
#     freshness_weight=0.3,  # 30% recency boost in re-ranking
# )

# --- Store a memory (with contradiction check) ---
# result = memory.add(
#     content="User prefers dark mode.",
#     user_id="u-12345",
# )
# if result.contradictions:
#     for warning in result.contradictions:
#         print(f"Contradiction: '{warning.new_text}' vs '{warning.existing_text}'")
#         print(f"  Similarity: {warning.similarity:.2f}")

# --- Search with hybrid re-ranking ---
# results = memory.search(
#     query="What are the user's UI preferences?",
#     user_id="u-12345",
#     top_k=5,
# )
# for item in results:
#     print(f"  [{item.score:.3f}] {item.content}")

# --- Store a contradicting memory ---
# result2 = memory.add(
#     content="User prefers light mode.",
#     user_id="u-12345",
# )
# # ContradictionWarning will flag "dark mode" vs "light mode"

print("Example: use_with_mem0.py")
print("Uncomment the code above and install mem0ai to run.")
