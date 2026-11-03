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
  AuthResponse,
  HealthResponse,
  HistoryFilters,
  HistoryResponse,
  HistoryStats,
  MeResponse,
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

/**
 * Download the server-rendered PDF for a completed analysis.
 * The backend renders from the SAME structured report document as the JSON
 * endpoint, so numbers can never diverge between web and PDF.
 */
export async function downloadReportPdf(analysisId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/v1/analyses/${encodeURIComponent(analysisId)}/report.pdf`,
  );
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
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `deepfake_report_${analysisId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
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

// ---- Phase 4: auth + history (cookie session via the same-origin proxy) ----

async function requestJsonAuth<T>(path: string, init?: RequestInit): Promise<T> {
  return requestJson<T>(path, { credentials: "include", ...init });
}

export function register(email: string, password: string): Promise<AuthResponse> {
  return requestJsonAuth<AuthResponse>("/v1/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function login(email: string, password: string): Promise<AuthResponse> {
  return requestJsonAuth<AuthResponse>("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function logout(): Promise<{ ok: boolean }> {
  return requestJsonAuth<{ ok: boolean }>("/v1/auth/logout", { method: "POST" });
}

export function me(): Promise<MeResponse> {
  return requestJsonAuth<MeResponse>("/v1/auth/me");
}

export function getHistory(filters: HistoryFilters = {}): Promise<HistoryResponse> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v) q.set(k, String(v));
  }
  const qs = q.toString();
  return requestJsonAuth<HistoryResponse>(`/v1/history${qs ? `?${qs}` : ""}`);
}

export function getHistoryStats(): Promise<HistoryStats> {
  return requestJsonAuth<HistoryStats>("/v1/history/stats");
}

export function deleteAnalysis(analysisId: string): Promise<{ deleted: boolean }> {
  return requestJsonAuth<{ deleted: boolean }>(
    `/v1/analyses/${encodeURIComponent(analysisId)}`,
    { method: "DELETE" },
  );
}
