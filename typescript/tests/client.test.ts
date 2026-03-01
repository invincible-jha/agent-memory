/**
 * Tests for @aumos/agent-memory client.
 *
 * Covers:
 * - Core memory client operations (store, retrieve, search, forget, graph, compact)
 * - sdk-core error hierarchy: HttpError, NetworkError, TimeoutError, RateLimitError, ServerError
 * - Request lifecycle events (request:start, request:end, request:retry, request:error)
 * - Retry behavior on 503 responses
 * - Query parameter construction
 * - Backward compatibility of ApiResult<T> shape
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createAgentMemoryClient } from "../src/client.js";
import type { MemoryEntry, RetrievalResult } from "../src/types.js";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeSuccessResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-type" ? "application/json" : null,
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

function makeErrorResponse(status: number, body: unknown, extraHeaders: Record<string, string> = {}) {
  return {
    ok: false,
    status,
    statusText: `Error ${status}`,
    headers: {
      get: (name: string) => {
        if (name.toLowerCase() === "content-type") return "application/json";
        return extraHeaders[name.toLowerCase()] ?? null;
      },
      forEach: (cb: (v: string, k: string) => void) => {
        cb("application/json", "content-type");
        for (const [k, v] of Object.entries(extraHeaders)) {
          cb(v, k);
        }
      },
    },
    json: vi.fn().mockResolvedValue(body),
    text: vi.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

const BASE_URL = "http://localhost:18060";

const SAMPLE_MEMORY_ENTRY: MemoryEntry = {
  memory_id: "mem-001",
  content: "The Eiffel Tower is in Paris.",
  layer: "semantic",
  importance_score: 0.8,
  freshness_score: 0.95,
  source: "document",
  created_at: "2024-01-01T00:00:00Z",
  last_accessed: "2024-01-01T00:00:00Z",
  access_count: 0,
  safety_critical: false,
  composite_score: 0.76,
  importance_level: "high",
  metadata: {},
};

// ---------------------------------------------------------------------------
// store()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — store()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with MemoryEntry on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_MEMORY_ENTRY)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.store({
      content: "The Eiffel Tower is in Paris.",
      layer: "semantic",
      importance_score: 0.8,
      source: "document",
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.memory_id).toBe("mem-001");
      expect(result.data.layer).toBe("semantic");
    }
  });

  it("returns ok:false with status 422 on validation error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(422, {
          fields: { content: ["Content cannot be empty"] },
        }),
      ),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.store({ content: "", layer: "semantic" });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(422);
      expect(result.error.error).toBe("Validation failed");
      expect(result.error.detail).toContain("content");
    }
  });
});

// ---------------------------------------------------------------------------
// retrieve()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — retrieve()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with MemoryEntry on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_MEMORY_ENTRY)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.retrieve("mem-001");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.memory_id).toBe("mem-001");
    }
  });

  it("returns ok:false with status 404 when memory not found", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(404, { error: "Memory not found", detail: "" }),
      ),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.retrieve("nonexistent");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(404);
      expect(result.error.error).toBe("Memory not found");
    }
  });

  it("URL-encodes the memory ID in the path", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse(SAMPLE_MEMORY_ENTRY));
      }),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.retrieve("mem id/with spaces");

    expect(capturedUrl).toContain(encodeURIComponent("mem id/with spaces"));
  });
});

// ---------------------------------------------------------------------------
// search()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — search()", () => {
  beforeEach(() => vi.restoreAllMocks());

  const SAMPLE_RESULTS: RetrievalResult[] = [
    {
      content: "The Eiffel Tower is in Paris.",
      source: "vector",
      score: 0.95,
      metadata: {},
    },
  ];

  it("returns ok:true with results on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(SAMPLE_RESULTS)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.search({ query: "Eiffel Tower", limit: 5 });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data).toHaveLength(1);
      expect(result.data[0]?.source).toBe("vector");
    }
  });

  it("passes query, layer, and limit as query params", async () => {
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        capturedUrl = url;
        return Promise.resolve(makeSuccessResponse([]));
      }),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    await client.search({ query: "Paris", layer: "semantic", limit: 10 });

    expect(capturedUrl).toContain("query=Paris");
    expect(capturedUrl).toContain("layer=semantic");
    expect(capturedUrl).toContain("limit=10");
  });
});

// ---------------------------------------------------------------------------
// forget()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — forget()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with ForgetResult on success", async () => {
    const forgetResult = { removed_count: 1, removed_ids: ["mem-001"] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(forgetResult)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.forget({ memoryId: "mem-001" });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.removed_count).toBe(1);
      expect(result.data.removed_ids).toContain("mem-001");
    }
  });
});

// ---------------------------------------------------------------------------
// getKnowledgeGraph()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — getKnowledgeGraph()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with KnowledgeGraph snapshot", async () => {
    const graph = { node_count: 2, edge_count: 1, nodes: [] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(graph)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getKnowledgeGraph();

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.node_count).toBe(2);
    }
  });
});

// ---------------------------------------------------------------------------
// compact()
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — compact()", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:true with CompactResult on success", async () => {
    const compact = {
      entries_before: 100,
      entries_after: 80,
      entries_removed: 20,
      entries_merged: 0,
      duration_ms: 45.2,
      completed_at: "2024-01-01T00:00:00Z",
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse(compact)));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.compact({ importanceThreshold: 0.2 });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.entries_removed).toBe(20);
    }
  });
});

// ---------------------------------------------------------------------------
// sdk-core error handling integration
// ---------------------------------------------------------------------------

describe("createAgentMemoryClient — sdk-core error handling", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns ok:false with status 429 on rate limit", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(429, { error: "Rate limit exceeded", detail: "" }, {
          "retry-after": "30",
        }),
      ),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.getKnowledgeGraph();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(429);
    }
  });

  it("returns ok:false with status 500 on server error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        makeErrorResponse(500, { error: "Internal error", detail: "DB timeout" }),
      ),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.store({ content: "test", layer: "working" });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(500);
    }
  });

  it("returns ok:false with status 0 on network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const result = await client.store({ content: "test", layer: "working" });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(0);
      expect(result.error.error).toMatch(/network error/i);
    }
  });

  it("exposes SdkEventEmitter on client.events", () => {
    const client = createAgentMemoryClient({ baseUrl: BASE_URL });
    expect(typeof client.events.on).toBe("function");
    expect(typeof client.events.off).toBe("function");
    expect(typeof client.events.emit).toBe("function");
  });

  it("fires request lifecycle events on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(makeSuccessResponse([])));

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 0 });
    const fired: string[] = [];

    client.events.on("request:start", () => fired.push("start"));
    client.events.on("request:end", () => fired.push("end"));

    await client.search({ query: "test" });

    expect(fired).toContain("start");
    expect(fired).toContain("end");
  });

  it("retries on 503 and ultimately succeeds", async () => {
    let callCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        callCount += 1;
        if (callCount < 3) {
          return Promise.resolve(makeErrorResponse(503, { error: "Unavailable", detail: "" }));
        }
        return Promise.resolve(makeSuccessResponse(SAMPLE_MEMORY_ENTRY));
      }),
    );

    const retried: number[] = [];
    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 3 });
    client.events.on("request:retry", ({ payload }) => retried.push(payload.attempt));

    const result = await client.retrieve("mem-001");

    expect(result.ok).toBe(true);
    expect(callCount).toBe(3);
    expect(retried).toHaveLength(2);
  });

  it("returns ok:false after all retries are exhausted on persistent 503", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(makeErrorResponse(503, { error: "Down", detail: "" })),
    );

    const client = createAgentMemoryClient({ baseUrl: BASE_URL, maxRetries: 2 });
    const result = await client.retrieve("mem-001");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(503);
    }
  });
});
