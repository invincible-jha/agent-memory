/**
 * TypeScript interfaces for the agent-memory framework.
 *
 * Mirrors the Pydantic/dataclass models defined in:
 *   agent_memory.memory.types          — MemoryLayer, MemoryEntry, MemorySource
 *   agent_memory.retrieval.graph_retriever — RetrievalResult
 *   agent_memory.memory.episodic        — EpisodicMemory
 *
 * All interfaces use readonly fields to match Python frozen models.
 */

// ---------------------------------------------------------------------------
// Memory layer enumerations
// ---------------------------------------------------------------------------

/**
 * The four cognitive memory layers supported by the framework.
 * Maps to the MemoryLayer enum in agent_memory.memory.types.
 *
 *   - working    — short-lived, session-scoped scratchpad memory
 *   - episodic   — time-ordered event log (what happened, when)
 *   - semantic   — factual knowledge and learned concepts
 *   - procedural — learned skills, workflows, and action templates
 */
export type MemoryLayer = "working" | "episodic" | "semantic" | "procedural";

/**
 * Origin of a memory entry.
 * Maps to the MemorySource enum in agent_memory.memory.types.
 */
export type MemorySource =
  | "user_input"
  | "tool_output"
  | "agent_inference"
  | "document"
  | "external_api";

/**
 * Qualitative importance tier for a memory entry.
 * Maps to the ImportanceLevel enum in agent_memory.memory.types.
 */
export type ImportanceLevel = "critical" | "high" | "medium" | "low" | "trivial";

// ---------------------------------------------------------------------------
// Core memory entry
// ---------------------------------------------------------------------------

/**
 * A single memory record stored across any layer.
 * Maps to MemoryEntry in agent_memory.memory.types.
 */
export interface MemoryEntry {
  /** Unique identifier for this memory record (UUID v4). */
  readonly memory_id: string;
  /** The textual content of this memory. */
  readonly content: string;
  /** Which cognitive memory layer this entry belongs to. */
  readonly layer: MemoryLayer;
  /** Numeric importance score in [0, 1]. Higher is more important. */
  readonly importance_score: number;
  /** Numeric freshness score in [0, 1]. Decays over time. */
  readonly freshness_score: number;
  /** Origin of this memory entry. */
  readonly source: MemorySource;
  /** ISO-8601 UTC timestamp when this entry was first created. */
  readonly created_at: string;
  /** ISO-8601 UTC timestamp of the most recent access. */
  readonly last_accessed: string;
  /** Number of times this entry has been retrieved. */
  readonly access_count: number;
  /** When true this entry is protected from automated forgetting. */
  readonly safety_critical: boolean;
  /** Derived composite relevance score: importance_score * freshness_score. */
  readonly composite_score: number;
  /** Qualitative importance tier derived from importance_score. */
  readonly importance_level: ImportanceLevel;
  /** Arbitrary string key/value metadata. */
  readonly metadata: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Knowledge graph types
// ---------------------------------------------------------------------------

/**
 * A single node in the knowledge graph.
 */
export interface KnowledgeNode {
  /** Node label — also serves as the unique identifier within the graph. */
  readonly label: string;
  /** Outgoing edges from this node. */
  readonly edges: readonly KnowledgeEdge[];
}

/**
 * A directed, weighted edge in the knowledge graph.
 */
export interface KnowledgeEdge {
  /** Relation type string (e.g. "capital_of", "knows", "related_to"). */
  readonly relation: string;
  /** Label of the target node. */
  readonly target: string;
  /** Edge weight in (0.0, 1.0]. Higher weight means greater relevance. */
  readonly weight: number;
}

/**
 * Snapshot of the knowledge graph structure.
 */
export interface KnowledgeGraph {
  /** Total number of nodes in the graph. */
  readonly node_count: number;
  /** Total number of directed edges in the graph. */
  readonly edge_count: number;
  /** All nodes with their outgoing edges. */
  readonly nodes: readonly KnowledgeNode[];
}

// ---------------------------------------------------------------------------
// Episode record
// ---------------------------------------------------------------------------

/**
 * A time-bounded episode record returned from episodic memory queries.
 * Groups temporally adjacent events into a named episode.
 */
export interface EpisodeRecord {
  /** Unique identifier for this episode. */
  readonly episode_id: string;
  /** Human-readable title or label for this episode. */
  readonly title: string;
  /** ISO-8601 UTC timestamp when the episode started. */
  readonly started_at: string;
  /** ISO-8601 UTC timestamp when the episode ended (null if ongoing). */
  readonly ended_at: string | null;
  /** Memory entries that make up this episode. */
  readonly entries: readonly MemoryEntry[];
  /** Number of entries in this episode. */
  readonly entry_count: number;
  /** Arbitrary metadata attached to this episode. */
  readonly metadata: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Retrieval types
// ---------------------------------------------------------------------------

/**
 * Query parameters for memory retrieval operations.
 */
export interface RetrievalQuery {
  /** Free-text query string used to find relevant memories. */
  readonly query: string;
  /** Restrict retrieval to a specific memory layer (null = all layers). */
  readonly layer?: MemoryLayer;
  /** Maximum number of results to return (default 10). */
  readonly top_k?: number;
  /** Minimum composite score threshold for inclusion (0.0–1.0). */
  readonly min_score?: number;
}

/**
 * A single result from a retrieval or search operation.
 * Maps to RetrievalResult in agent_memory.retrieval.graph_retriever.
 */
export interface RetrievalResult {
  /** The retrieved text content or node label. */
  readonly content: string;
  /** Which backend produced this result: "vector", "graph", or "recency". */
  readonly source: "vector" | "graph" | "recency";
  /** Relevance score in [0.0, 1.0]. */
  readonly score: number;
  /** Arbitrary metadata from the retrieval backend. */
  readonly metadata: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Compact operation result
// ---------------------------------------------------------------------------

/**
 * Result returned after a memory compaction operation.
 */
export interface CompactResult {
  /** Total number of entries before compaction. */
  readonly entries_before: number;
  /** Total number of entries after compaction. */
  readonly entries_after: number;
  /** Number of entries that were removed. */
  readonly entries_removed: number;
  /** Number of entries that were merged or consolidated. */
  readonly entries_merged: number;
  /** Wall-clock milliseconds the compaction took. */
  readonly duration_ms: number;
  /** ISO-8601 UTC timestamp when compaction completed. */
  readonly completed_at: string;
}

// ---------------------------------------------------------------------------
// Request/response payload types
// ---------------------------------------------------------------------------

/**
 * Request body for the store operation.
 */
export interface StoreRequest {
  /** The textual content to store. */
  readonly content: string;
  /** Memory layer to store this entry in. */
  readonly layer: MemoryLayer;
  /** Importance score [0, 1] (default 0.5). */
  readonly importance_score?: number;
  /** Origin of this memory (default "agent_inference"). */
  readonly source?: MemorySource;
  /** Whether this entry is safety-critical (default false). */
  readonly safety_critical?: boolean;
  /** Arbitrary key/value metadata. */
  readonly metadata?: Readonly<Record<string, string>>;
}

/**
 * Query parameters for the search endpoint.
 */
export interface SearchQuery {
  /** Free-text search string. */
  readonly query: string;
  /** Restrict to a specific layer (omit for all layers). */
  readonly layer?: MemoryLayer;
  /** Maximum number of results (default 10). */
  readonly limit?: number;
}

/**
 * Query parameters for the forget endpoint.
 */
export interface ForgetQuery {
  /** Specific memory ID to remove (takes precedence over other filters). */
  readonly memoryId?: string;
  /** Remove all entries in this layer. */
  readonly layer?: MemoryLayer;
  /** Remove entries below this importance score threshold. */
  readonly belowImportance?: number;
}

/**
 * Response from the forget operation.
 */
export interface ForgetResult {
  /** Number of entries removed. */
  readonly removed_count: number;
  /** IDs of the removed entries. */
  readonly removed_ids: readonly string[];
}

// ---------------------------------------------------------------------------
// API result wrapper (shared pattern)
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-memory API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
