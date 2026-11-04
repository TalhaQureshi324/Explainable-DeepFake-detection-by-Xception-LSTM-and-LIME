/**
 * Inference provider types (Phase 5-HF).
 *
 * Two interchangeable inference providers behind one interface:
 *  - "huggingface" (default): browser talks DIRECTLY to our public Hugging
 *    Face Space via @gradio/client — no media crosses Vercel.
 *  - "fastapi" (optional/future): the modular FastAPI backend from Phases 1-4
 *    (kept in the repository, selectable via NEXT_PUBLIC_INFERENCE_MODE).
 *
 * Both providers produce a client-side `StoredAnalysis` record shaped after
 * the Phase 1 API contract so the results, report and PDF surfaces stay
 * provider-agnostic. Fields the Space does not provide are `null` and are
 * displayed as such — never invented.
 */

import type { AnalysisResponse, WindowScore } from "@/lib/api/types";

export type InferenceProviderId = "huggingface" | "fastapi";

/** Subset of the API contract the HF basic endpoint can populate. */
export type StoredAnalysis = AnalysisResponse & {
  /** Markdown block returned verbatim by the Space (display fallback). */
  raw_markdown?: string | null;
  /** True when the record came from the rich structured Space endpoint. */
  structured?: boolean;
};

export interface PredictOptions {
  enableXai: boolean;
  filename: string;
  mimeType?: string;
  /** Optional progress callback (provider-dependent phases). */
  onStatus?: (message: string) => void;
}

export interface InferenceProvider {
  readonly id: InferenceProviderId;
  /** Human-readable status line while the job runs (queue, cold start…). */
  statusLabel?: (phase: "connecting" | "running") => string;
  predict(file: File, opts: PredictOptions): Promise<StoredAnalysis>;
}

export function newAnalysisId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** Session-scoped registry: analyses live for the tab lifetime only. */
const memory = new Map<string, StoredAnalysis>();

export function saveAnalysis(record: StoredAnalysis): void {
  memory.set(record.analysis_id, record);
}

export function getStoredAnalysis(id: string): StoredAnalysis | null {
  return memory.get(id) ?? null;
}

export function windowScoresToWindows(windows: Array<{ label: string; score: number }>): WindowScore[] {
  return windows.map((w) => ({
    label: w.label,
    score: w.score,
    frame_start: null,
    frame_end: null,
    start_seconds: null,
    end_seconds: null,
  }));
}
