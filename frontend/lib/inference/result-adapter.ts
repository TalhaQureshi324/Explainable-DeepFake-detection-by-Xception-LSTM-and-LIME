/**
 * Result adapter (Phase 5-HF): converts what the Space actually returns into
 * the StoredAnalysis contract. Values the Space does not provide stay null;
 * the honest threshold (0.75, the Space's own constant) is used — never the
 * FastAPI development threshold.
 */

import type { StoredAnalysis } from "./types";
import { newAnalysisId } from "./types";

export const SPACE_THRESHOLD = 0.75;
export const SPACE_THRESHOLD_VERSION = "hf-space-original";

/**
 * The Space result Markdown embeds a small table:
 *   | Confidence | 62.34% |
 *   | Average Score | 0.6234 (1 = Fake, 0 = Real) |
 *   | Std Deviation | 0.1234 |
 *   | Ensemble Windows | 7 |
 *   | Input Type | Video (32 face frames extracted) |
 * plus a verdict header line "🟥 FAKE" / "🟩 REAL".
 */
export function parseResultMarkdown(md: string): {
  label: "REAL" | "FAKE" | null;
  confidence: number | null;
  fakeScore: number | null;
  stdDev: number | null;
  windows: number | null;
  inputType: string | null;
} {
  const plain = md ?? "";
  // The Space result markdown starts with a verdict emoji:
  //   "## \u{1F7E5} FAKE" / "## \u{1F7E9} REAL"
  // (FAKE/Real words alone are ambiguous - the score legend mentions both)
  const label = /\u{1F7E5}/u.test(plain)
    ? "FAKE"
    : /\u{1F7E9}/u.test(plain)
      ? "REAL"
      : null;

  const grab = (re: RegExp): number | null => {
    const m = plain.match(re);
    if (!m) return null;
    const v = parseFloat(m[1].replace("%", ""));
    return Number.isFinite(v) ? v : null;
  };
  const confidence = grab(/\|\s*\|?\s*Confidence\s*\|\s*([\d.]+)\s*%/i) ?? grab(/Confidence\D{0,6}([\d.]+)\s*%/i);
  const fakeScore = grab(/Average Score\D{0,6}([\d.]+)/i);
  const stdDev = grab(/Std Deviation\D{0,6}([\d.]+)/i);
  const windows = grab(/Ensemble Windows\D{0,6}(\d+)/i);
  const inputTypeMatch = plain.match(/\|\s*\|?\s*Input Type\s*\|\s*([^|\n]+)/i);

  return {
    label,
    confidence,
    fakeScore: fakeScore !== null && fakeScore > 1 ? fakeScore / 100 : fakeScore,
    stdDev,
    windows: windows !== null ? parseInt(String(windows), 10) : null,
    inputType: inputTypeMatch ? inputTypeMatch[1].trim() : null,
  };
}

export interface SpaceRawResult {
  markdown: string;
  facePath: string | null;
  limePath: string | null;
  faceUrl: string | null;
  limeUrl: string | null;
  seconds: number | null;
}

/**
 * Build a StoredAnalysis from a basic `/predict_deepfake` result.
 * `resolveFileUrl` turns the client-specific file object into a public URL.
 */
export function basicResultToAnalysis(
  raw: SpaceRawResult,
  opts: { filename: string; mediaType: "video" | "image"; spaceId: string },
): StoredAnalysis {
  const parsed = parseResultMarkdown(raw.markdown);
  const id = newAnalysisId();
  const warnings: string[] = [
    "This analysis ran on the public Hugging Face Space; results are not saved and are lost when you leave this page.",
  ];
  if (!raw.limeUrl && opts.mediaType) {
    // LIME simply wasn't requested — not an error.
  }
  return {
    analysis_id: id,
    status: raw.markdown ? "COMPLETED" : "FAILED",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    media: {
      type: opts.mediaType,
      filename: opts.filename,
      size_bytes: null,
      sha256: null,
      width: null,
      height: null,
      fps: null,
      duration_seconds: null,
      frames_extracted:
        parsed.inputType && /(\d+)\s+face frames/.test(parsed.inputType)
          ? parseInt(parsed.inputType.match(/(\d+)\s+face frames/)![1], 10)
          : opts.mediaType === "image"
            ? 1
            : null,
      frames_analyzed: null,
      valid_face_detections: null,
    },
    prediction: {
      label: parsed.label,
      fake_score: parsed.fakeScore,
      threshold: SPACE_THRESHOLD,
      threshold_version: SPACE_THRESHOLD_VERSION,
      method: "hf_space_sliding_window_ensemble",
      score_is_calibrated: false,
    },
    model: {
      name: "Xception-LSTM",
      version: "space-preferred checkpoint",
      description: "Loaded by the Hugging Face Space (v2 preferred, v1 fallback)",
      checkpoint_sha256: null,
      input_frames: null,
      input_size: 299,
    },
    processing: {
      duration_seconds: raw.seconds,
      warnings,
      error: null,
      error_code: null,
    },
    explainability: {
      available: Boolean(raw.limeUrl),
      method: raw.limeUrl ? "LIME" : null,
      artifact_id: null,
      limitation:
        "LIME highlights regions of a single replicated frame that influenced the model. It does not prove those pixels were manipulated and does not explain the temporal pathway.",
    },
    artifacts: {
      processed_face: raw.faceUrl,
      lime: raw.limeUrl,
    },
    windows: [],
    report_available: false, // upgraded when the structured endpoint is available
    raw_markdown: raw.markdown,
    structured: false,
  };
}

/**
 * Structured `/analyze_structured` result (available once the extended Space
 * app is deployed) maps almost 1:1 onto the Phase 1 contract.
 */
export function structuredResultToAnalysis(payload: Record<string, unknown>): StoredAnalysis {
  const record = payload as unknown as StoredAnalysis;
  return {
    ...record,
    structured: true,
    report_available: record.status === "COMPLETED",
  };
}
