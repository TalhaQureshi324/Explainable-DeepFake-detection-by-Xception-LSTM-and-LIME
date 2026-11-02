/**
 * API access layer — the ONLY module that talks HTTP to the inference backend.
 * Presentation components never call fetch directly.
 *
 * All requests go through the same-origin proxy configured in next.config.mjs
 * (`/api/inference/:path*` -> INFERENCE_API_URL). This avoids CORS entirely and
 * keeps the deployed backend URL a server-side secret.
 */

import type {
  AnalysisCreateAccepted,
  AnalysisResponse,
  HealthResponse,
  ModelsResponse,
  ReportData,
  ReadyResponse,
} from "./types";

const API_BASE = "/api/inference";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail || `API error ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "The inference backend could not be reached.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function getReady(): Promise<ReadyResponse> {
  return requestJson<ReadyResponse>("/ready");
}

export function getModels(): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>("/v1/models");
}

export function createAnalysis(
  file: File,
  opts: { modelVersion?: string; explain?: boolean } = {},
): Promise<AnalysisCreateAccepted> {
  const form = new FormData();
  form.append("file", file);
  if (opts.modelVersion) form.append("model_version", opts.modelVersion);
  if (opts.explain !== undefined) form.append("explain", String(opts.explain));
  return requestJson<AnalysisCreateAccepted>("/v1/analyses", {
    method: "POST",
    body: form,
  });
}

export function getAnalysis(analysisId: string): Promise<AnalysisResponse> {
  return requestJson<AnalysisResponse>(`/v1/analyses/${encodeURIComponent(analysisId)}`);
}

export function getReport(analysisId: string): Promise<ReportData> {
  return requestJson<ReportData>(`/v1/analyses/${encodeURIComponent(analysisId)}/report`);
}

/** Artifacts (face preview / LIME PNGs) also flow through the proxy. */
export function artifactUrl(relativePath: string | null | undefined): string | null {
  if (!relativePath) return null;
  return `${API_BASE}${relativePath}`;
}

/**
 * Poll an analysis until it reaches a terminal state.
 * - onState fires for every observed state change (drives the progress UI).
 * - Aborts cleanly via AbortSignal (React StrictMode double-mount, navigation).
 * - Guards against a backend restart losing the in-memory job (404 -> ApiError).
 */
export async function pollAnalysis(
  analysisId: string,
  opts: {
    onState?: (state: AnalysisResponse) => void;
    intervalMs?: number;
    timeoutMs?: number;
    signal?: AbortSignal;
  } = {},
): Promise<AnalysisResponse> {
  const { onState, intervalMs = 1500, timeoutMs = 15 * 60 * 1000, signal } = opts;
  const started = Date.now();
  let lastStatus: string | null = null;

  while (Date.now() - started < timeoutMs) {
    if (signal?.aborted) throw new DOMException("Polling aborted", "AbortError");
    const analysis = await getAnalysis(analysisId);
    if (analysis.status !== lastStatus) {
      lastStatus = analysis.status;
      onState?.(analysis);
    }
    if (analysis.status === "COMPLETED" || analysis.status === "FAILED") return analysis;
    await new Promise<void>((resolve, reject) => {
      const t = setTimeout(resolve, intervalMs);
      signal?.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new DOMException("Polling aborted", "AbortError"));
      }, { once: true });
    });
  }
  throw new ApiError(504, "Analysis timed out while polling the backend.");
}
