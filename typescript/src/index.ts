/**
 * @aumos/agent-memory
 *
 * TypeScript client for the AumOS agent-memory framework.
 * Provides multi-layer memory management, hybrid retrieval,
 * knowledge graph access, and memory compaction.
 */

// Client and configuration
export type { AgentMemoryClient, AgentMemoryClientConfig } from "./client.js";
export { createAgentMemoryClient } from "./client.js";

// Core memory types
export type {
  MemoryLayer,
  MemorySource,
  ImportanceLevel,
  MemoryEntry,
  KnowledgeNode,
  KnowledgeEdge,
  KnowledgeGraph,
  EpisodeRecord,
  RetrievalQuery,
  RetrievalResult,
  CompactResult,
  StoreRequest,
  SearchQuery,
  ForgetQuery,
  ForgetResult,
  ApiError,
  ApiResult,
} from "./types.js";
