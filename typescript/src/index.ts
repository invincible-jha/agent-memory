/**
 * @aumos/agent-memory
 *
 * TypeScript client for the AumOS agent-memory framework.
 * Provides multi-layer memory management, hybrid retrieval,
 * knowledge graph access, and memory compaction.
 *
 * The client is now backed by @aumos/sdk-core for automatic retry,
 * typed error hierarchy, and request lifecycle events.
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

// Re-export sdk-core error hierarchy for callers that want to instanceof-check
export {
  AumosError,
  NetworkError,
  TimeoutError,
  HttpError,
  RateLimitError,
  ValidationError,
  ServerError,
  AbortError,
} from "@aumos/sdk-core";

// Re-export event emitter type for listeners attached via client.events
export type { SdkEventEmitter, SdkEventMap } from "@aumos/sdk-core";
