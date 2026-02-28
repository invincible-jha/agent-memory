/**
 * HTTP client for the agent-memory API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentMemoryClient } from "@aumos/agent-memory";
 *
 * const client = createAgentMemoryClient({ baseUrl: "http://localhost:8060" });
 *
 * // Store a new memory in the semantic layer
 * const stored = await client.store({
 *   content: "The Eiffel Tower is located in Paris, France.",
 *   layer: "semantic",
 *   importance_score: 0.8,
 *   source: "document",
 * });
 *
 * if (stored.ok) {
 *   console.log("Memory stored:", stored.data.memory_id);
 * }
 *
 * // Hybrid retrieval across all layers
 * const results = await client.search({ query: "Eiffel Tower", limit: 5 });
 * if (results.ok) {
 *   for (const result of results.data) {
 *     console.log(`[${result.source}] score=${result.score} — ${result.content}`);
 *   }
 * }
 * ```
 */

import type {
  ApiError,
  ApiResult,
  CompactResult,
  ForgetQuery,
  ForgetResult,
  KnowledgeGraph,
  MemoryEntry,
  RetrievalQuery,
  RetrievalResult,
  SearchQuery,
  StoreRequest,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentMemoryClient. */
export interface AgentMemoryClientConfig {
  /** Base URL of the agent-memory server (e.g. "http://localhost:8060"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-memory server. */
export interface AgentMemoryClient {
  /**
   * Persist a new memory entry in the specified layer.
   *
   * The server assigns a unique memory_id and computes the initial
   * freshness score. Returns the full MemoryEntry as stored.
   *
   * @param request - Content, layer, importance, source, and metadata.
   * @returns The persisted MemoryEntry with server-assigned fields.
   */
  store(request: StoreRequest): Promise<ApiResult<MemoryEntry>>;

  /**
   * Retrieve a specific memory entry by its ID.
   *
   * Accessing a memory updates its last_accessed timestamp and
   * increments access_count on the server.
   *
   * @param memoryId - The UUID of the memory entry to retrieve.
   * @returns The MemoryEntry if found, or a 404 error result.
   */
  retrieve(memoryId: string): Promise<ApiResult<MemoryEntry>>;

  /**
   * Perform a hybrid (vector + graph + recency) search across memory layers.
   *
   * Combines semantic similarity, knowledge graph traversal, and recency
   * signals, fusing results into a single ranked list.
   *
   * @param query - Query string, optional layer filter, and result limit.
   * @returns Ranked RetrievalResult array from all active retrieval backends.
   */
  search(query: SearchQuery): Promise<ApiResult<readonly RetrievalResult[]>>;

  /**
   * Remove one or more memory entries matching the given criteria.
   *
   * When memoryId is provided, only that specific entry is removed.
   * When layer is provided without memoryId, all entries in that layer
   * are removed. When belowImportance is provided, entries with an
   * importance_score below the threshold are candidates for removal.
   *
   * Safety-critical entries are never removed by automated forget calls.
   *
   * @param query - Removal criteria: by ID, layer, or importance threshold.
   * @returns ForgetResult with removed count and IDs.
   */
  forget(query: ForgetQuery): Promise<ApiResult<ForgetResult>>;

  /**
   * Retrieve the current knowledge graph structure.
   *
   * Returns all nodes and directed edges from the semantic knowledge graph,
   * including node labels, relation types, and edge weights.
   *
   * @returns KnowledgeGraph snapshot with nodes, edges, and counts.
   */
  getKnowledgeGraph(): Promise<ApiResult<KnowledgeGraph>>;

  /**
   * Trigger a memory compaction pass.
   *
   * Compaction removes stale low-importance entries, merges near-duplicate
   * content, and refreshes freshness scores based on access patterns.
   * The operation runs server-side and returns a summary of changes made.
   *
   * @param options - Optional compaction parameters.
   * @returns CompactResult with before/after counts and duration.
   */
  compact(options?: {
    layer?: import("./types.js").MemoryLayer;
    importanceThreshold?: number;
  }): Promise<ApiResult<CompactResult>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-memory server.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentMemoryClient instance.
 */
export function createAgentMemoryClient(
  config: AgentMemoryClientConfig,
): AgentMemoryClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async store(request: StoreRequest): Promise<ApiResult<MemoryEntry>> {
      return fetchJson<MemoryEntry>(
        `${baseUrl}/memory`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async retrieve(memoryId: string): Promise<ApiResult<MemoryEntry>> {
      return fetchJson<MemoryEntry>(
        `${baseUrl}/memory/${encodeURIComponent(memoryId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async search(
      query: SearchQuery,
    ): Promise<ApiResult<readonly RetrievalResult[]>> {
      const params = new URLSearchParams({ query: query.query });
      if (query.layer !== undefined) {
        params.set("layer", query.layer);
      }
      if (query.limit !== undefined) {
        params.set("limit", String(query.limit));
      }

      return fetchJson<readonly RetrievalResult[]>(
        `${baseUrl}/memory/search?${params.toString()}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async forget(query: ForgetQuery): Promise<ApiResult<ForgetResult>> {
      return fetchJson<ForgetResult>(
        `${baseUrl}/memory/forget`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify({
            memory_id: query.memoryId,
            layer: query.layer,
            below_importance: query.belowImportance,
          }),
        },
        timeoutMs,
      );
    },

    async getKnowledgeGraph(): Promise<ApiResult<KnowledgeGraph>> {
      return fetchJson<KnowledgeGraph>(
        `${baseUrl}/memory/graph`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async compact(
      options: {
        layer?: import("./types.js").MemoryLayer;
        importanceThreshold?: number;
      } = {},
    ): Promise<ApiResult<CompactResult>> {
      return fetchJson<CompactResult>(
        `${baseUrl}/memory/compact`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify({
            layer: options.layer,
            importance_threshold: options.importanceThreshold,
          }),
        },
        timeoutMs,
      );
    },
  };
}

// ---------------------------------------------------------------------------
// Re-export advanced retrieval query type
// ---------------------------------------------------------------------------

export type {
  StoreRequest,
  MemoryEntry,
  SearchQuery,
  RetrievalQuery,
  RetrievalResult,
  ForgetQuery,
  ForgetResult,
  KnowledgeGraph,
  CompactResult,
};
