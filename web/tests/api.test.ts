import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, analyze, ask, health, search } from "@/lib/api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("api client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function mockFetch() {
    return globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
  }

  it("posts the analyse payload as JSON", async () => {
    mockFetch().mockResolvedValue(jsonResponse({ document_id: "doc-1", entities: [] }));

    await analyze({ text: "hello", analyses: ["sentiment"] });

    const [url, init] = mockFetch().mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/analyze$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ text: "hello", analyses: ["sentiment"] });
    expect((init.headers as Record<string, string>)["content-type"]).toBe("application/json");
  });

  it("returns the parsed body", async () => {
    mockFetch().mockResolvedValue(jsonResponse({ status: "ok", documents: 3 }));
    await expect(health()).resolves.toEqual({ status: "ok", documents: 3 });
  });

  it("sends the search query and top_k", async () => {
    mockFetch().mockResolvedValue(jsonResponse([]));
    await search("refunds", 3);

    const [, init] = mockFetch().mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({ query: "refunds", top_k: 3 });
  });

  it("puts a namespaced document id in the qa path", async () => {
    mockFetch().mockResolvedValue(jsonResponse({ answer: "yes", supported: true }));
    await ask("kb/refunds", "how long?");

    const [url] = mockFetch().mock.calls[0] as [string];
    expect(url).toMatch(/\/documents\/kb\/refunds\/qa$/);
  });

  it("surfaces the server's own error message", async () => {
    mockFetch().mockResolvedValue(
      jsonResponse({ detail: "the 'qa' analysis needs a question" }, 422),
    );

    await expect(analyze({ text: "x", analyses: ["qa"] })).rejects.toThrow(
      "the 'qa' analysis needs a question",
    );
  });

  it("flattens a FastAPI validation error into a readable message", async () => {
    mockFetch().mockResolvedValue(
      jsonResponse(
        { detail: [{ msg: "String should have at least 1 character", loc: ["body", "text"] }] },
        422,
      ),
    );

    await expect(analyze({ text: "" })).rejects.toThrow(
      "String should have at least 1 character at body.text",
    );
  });

  it("carries the status code on the error", async () => {
    mockFetch().mockResolvedValue(jsonResponse({ detail: "no document 'kb/nope'" }, 404));

    const error = await analyze({ text: "x" }).catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
  });

  it("falls back to the status text when the body is not JSON", async () => {
    mockFetch().mockResolvedValue(
      new Response("<html>oops</html>", { status: 500, statusText: "Internal Server Error" }),
    );

    await expect(health()).rejects.toThrow("Internal Server Error");
  });

  it("reports a network failure as status 0 rather than an HTTP error", async () => {
    // "Never reached the server" is a different problem from "the server said no", and the
    // message needs to say which.
    mockFetch().mockRejectedValue(new TypeError("Failed to fetch"));

    const error = await health().catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(0);
    expect((error as ApiError).message).toContain("cannot reach the API");
  });
});
