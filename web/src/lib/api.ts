/**
 * Typed client for the analyzer API.
 *
 * Errors carry the server's own message where there is one. A generic "request failed"
 * turns a 422 that explains exactly what was wrong with the request into a shrug, and the
 * backend already returns a useful `detail` for every error it raises.
 */

import type {
  AnalyzeRequest,
  AnalyzeResponse,
  HealthResponse,
  QAAnswer,
  SearchMatch,
} from "@/lib/types";

export const DEFAULT_API_URL = "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function baseUrl(): string {
  return (process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL).replace(/\/+$/, "");
}

/** Pull the most useful message out of an error response body. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") {
        return detail;
      }
      // FastAPI validation errors arrive as a list of objects.
      if (Array.isArray(detail)) {
        const first = detail[0] as { msg?: unknown; loc?: unknown } | undefined;
        if (first && typeof first.msg === "string") {
          const where = Array.isArray(first.loc) ? ` at ${first.loc.join(".")}` : "";
          return `${first.msg}${where}`;
        }
      }
    }
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return response.statusText || `request failed with status ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl()}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // A network failure is not an HTTP status; 0 marks "never reached the server".
    throw new ApiError(
      `cannot reach the API at ${baseUrl()}: ${cause instanceof Error ? cause.message : String(cause)}`,
      0,
    );
  }

  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

export function analyze(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function search(query: string, topK = 5): Promise<SearchMatch[]> {
  return request<SearchMatch[]>("/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}

export function ask(documentId: string, question: string): Promise<QAAnswer> {
  return request<QAAnswer>(`/documents/${documentId}/qa`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/healthz");
}
