/**
 * HTTP client for the agent-memory API.
 *
 * Backed by @aumos/sdk-core's createHttpClient which provides automatic retry
 * with exponential backoff, typed error hierarchy, request lifecycle events,
 * and abort signal support.
 *
 * The public API surface is unchanged — all methods still return ApiResult<T>
 * so existing callers require no migration work.
 *
 * @example
 * ```ts
 * import { createAgentMemoryClient } from "@aumos/agent-memory";
 *
 * const client = createAgentMemoryClient({ baseUrl: "http://localhost:8060" });
 *
 * // Attach a retry observer via sdk-core events
 * client.events.on("request:retry", ({ payload }) => {
 *   console.warn(`Memory API retry attempt ${payload.attempt}`);
 * });
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
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  ServerError,
  ValidationError,
  AumosError,
} from "@aumos/sdk-core";

import type { HttpClient, SdkEventEmitter } from "@aumos/sdk-core";

import type {
  ApiError,
  ApiResult,
  CompactResult,
  ForgetQuery,
  ForgetResult,
  KnowledgeGraph,
  MemoryEntry,
  MemoryLayer,
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
  /** Optional maximum retry count. Defaults to 3. */
  readonly maxRetries?: number;
}

// ---------------------------------------------------------------------------
// Internal adapter — bridges HttpClient throws into ApiResult<T>
// ---------------------------------------------------------------------------

function extractApiError(body: unknown, fallbackMessage: string): ApiError {
  if (
    body !== null &&
    typeof body === "object" &&
    "error" in body &&
    typeof (body as Record<string, unknown>)["error"] === "string"
  ) {
    const candidate = body as Partial<{ error: string; detail: string }>;
    return {
      error: candidate.error ?? fallbackMessage,
      detail: candidate.detail ?? "",
    };
  }
  return { error: fallbackMessage, detail: "" };
}

async function executeApiCall<T>(
  call: () => Promise<T>,
): Promise<ApiResult<T>> {
  try {
    const data = await call();
    return { ok: true, data };
  } catch (error: unknown) {
    if (error instanceof RateLimitError) {
      return {
        ok: false,
        error: extractApiError(error.body, "Rate limit exceeded"),
        status: 429,
      };
    }
    if (error instanceof ValidationError) {
      return {
        ok: false,
        error: {
          error: "Validation failed",
          detail: Object.entries(error.fields)
            .map(([field, messages]) => `${field}: ${messages.join(", ")}`)
            .join("; "),
        },
        status: 422,
      };
    }
    if (error instanceof ServerError) {
      return {
        ok: false,
        error: extractApiError(error.body, `Server error: HTTP ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: extractApiError(error.body, `HTTP error: ${error.statusCode}`),
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: {
          error: "Network error",
          detail: error instanceof Error ? error.message : String(error),
        },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unknown error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-memory server. */
export interface AgentMemoryClient {
  /**
   * Typed event emitter exposed from the underlying sdk-core HttpClient.
   * Attach listeners here to observe request lifecycle, retries, and errors.
   *
   * @example
   * ```ts
   * client.events.on("request:retry", ({ payload }) => {
   *   console.warn(`Retry attempt ${payload.attempt}, delay ${payload.delayMs}ms`);
   * });
   * ```
   */
  readonly events: SdkEventEmitter;

  /**
   * Persist a new memory entry in the specified layer.
   *
   * @param request - Content, layer, importance, source, and metadata.
   * @returns The persisted MemoryEntry with server-assigned fields.
   */
  store(request: StoreRequest): Promise<ApiResult<MemoryEntry>>;

  /**
   * Retrieve a specific memory entry by its ID.
   *
   * @param memoryId - The UUID of the memory entry to retrieve.
   * @returns The MemoryEntry if found, or a 404 error result.
   */
  retrieve(memoryId: string): Promise<ApiResult<MemoryEntry>>;

  /**
   * Perform a hybrid (vector + graph + recency) search across memory layers.
   *
   * @param query - Query string, optional layer filter, and result limit.
   * @returns Ranked RetrievalResult array from all active retrieval backends.
   */
  search(query: SearchQuery): Promise<ApiResult<readonly RetrievalResult[]>>;

  /**
   * Remove one or more memory entries matching the given criteria.
   *
   * @param query - Removal criteria: by ID, layer, or importance threshold.
   * @returns ForgetResult with removed count and IDs.
   */
  forget(query: ForgetQuery): Promise<ApiResult<ForgetResult>>;

  /**
   * Retrieve the current knowledge graph structure.
   *
   * @returns KnowledgeGraph snapshot with nodes, edges, and counts.
   */
  getKnowledgeGraph(): Promise<ApiResult<KnowledgeGraph>>;

  /**
   * Trigger a memory compaction pass.
   *
   * @param options - Optional compaction parameters.
   * @returns CompactResult with before/after counts and duration.
   */
  compact(options?: {
    layer?: MemoryLayer;
    importanceThreshold?: number;
  }): Promise<ApiResult<CompactResult>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-memory server.
 *
 * Internally uses @aumos/sdk-core's createHttpClient for automatic retry,
 * typed errors, and request lifecycle events. The public API remains identical
 * to the previous version — all methods return ApiResult<T>.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentMemoryClient instance.
 */
export function createAgentMemoryClient(
  config: AgentMemoryClientConfig,
): AgentMemoryClient {
  const httpClient: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    maxRetries: config.maxRetries ?? 3,
    defaultHeaders: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(config.headers as Record<string, string> | undefined),
    },
  });

  return {
    events: httpClient.events,

    store(request: StoreRequest): Promise<ApiResult<MemoryEntry>> {
      return executeApiCall(() =>
        httpClient.post<MemoryEntry>("/memory", request).then((r) => r.data),
      );
    },

    retrieve(memoryId: string): Promise<ApiResult<MemoryEntry>> {
      return executeApiCall(() =>
        httpClient
          .get<MemoryEntry>(`/memory/${encodeURIComponent(memoryId)}`)
          .then((r) => r.data),
      );
    },

    search(query: SearchQuery): Promise<ApiResult<readonly RetrievalResult[]>> {
      const queryParams: Record<string, string> = { query: query.query };
      if (query.layer !== undefined) {
        queryParams["layer"] = query.layer;
      }
      if (query.limit !== undefined) {
        queryParams["limit"] = String(query.limit);
      }

      return executeApiCall(() =>
        httpClient
          .get<readonly RetrievalResult[]>("/memory/search", { queryParams })
          .then((r) => r.data),
      );
    },

    forget(query: ForgetQuery): Promise<ApiResult<ForgetResult>> {
      return executeApiCall(() =>
        httpClient
          .post<ForgetResult>("/memory/forget", {
            memory_id: query.memoryId,
            layer: query.layer,
            below_importance: query.belowImportance,
          })
          .then((r) => r.data),
      );
    },

    getKnowledgeGraph(): Promise<ApiResult<KnowledgeGraph>> {
      return executeApiCall(() =>
        httpClient.get<KnowledgeGraph>("/memory/graph").then((r) => r.data),
      );
    },

    compact(
      options: { layer?: MemoryLayer; importanceThreshold?: number } = {},
    ): Promise<ApiResult<CompactResult>> {
      return executeApiCall(() =>
        httpClient
          .post<CompactResult>("/memory/compact", {
            layer: options.layer,
            importance_threshold: options.importanceThreshold,
          })
          .then((r) => r.data),
      );
    },
  };
}
