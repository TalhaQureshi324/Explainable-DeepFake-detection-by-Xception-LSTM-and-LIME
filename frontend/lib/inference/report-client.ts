/**
 * Client-side report builder for provider results (Phase 5-HF).
 *
 * Mirrors backend `services/report_data.build_report` so the interactive
 * report and (Space-generated) PDF stay consistent with what the API contract
 * produces in FastAPI mode. Same rule: numbers come from the stored analysis
 * record only — nothing invented.
 */

import type { MediaInfo, ModelInfo, Prediction, ProcessingInfo } from "@/lib/api/types";
import type { StoredAnalysis } from "./types";

export const REPORT_LIMITATIONS = [
  "Scores are sigmoid outputs from an uncalibrated model; they are not proven probabilities.",
  "Models were trained only on the DeepFake Detection (DFD) subset of FaceForensics++; performance on other manipulation methods is unknown.",
  "Only the most confident face per frame is analyzed; multi-face videos are not fully covered.",
  "LIME explains a replicated static frame; it does not attribute the temporal pathway.",
  "Guest analyses are session-scoped and are not permanently stored.",
];

export function buildClientReport(analysis: StoredAnalysis): Record<string, unknown> {
  const media = (analysis.media ?? {}) as Partial<MediaInfo>;
  const prediction = (analysis.prediction ?? {}) as Partial<Prediction>;
  const model = (analysis.model ?? {}) as Partial<ModelInfo>;
  const processing = (analysis.processing ?? { warnings: [] }) as Partial<ProcessingInfo>;
  const windows = analysis.windows ?? [];

  const scores = windows.map((w) => w.score).filter((s) => typeof s === "number");
  const variation =
    scores.length > 0
      ? {
          min: Math.min(...scores),
          max: Math.max(...scores),
          stdev:
            scores.length > 1
              ? Math.sqrt(
                  scores.reduce((acc: number, s: number) => acc + (s - scores.reduce((a, b) => a + b, 0) / scores.length) ** 2, 0) /
                    scores.length,
                )
              : 0,
        }
      : null;

  const label = prediction.label ?? "—";
  const threshold = prediction.threshold ?? null;
  const stance =
    label === "FAKE"
      ? `The model scored this media above the configured decision threshold, which maps to its FAKE class. The score exceeds the threshold by ${
          threshold != null && prediction.fake_score != null ? (prediction.fake_score - threshold).toFixed(3) : "?"
        }.`
      : `The model scored this media below the configured decision threshold, which maps to its REAL class.${
          threshold != null && prediction.fake_score != null
            ? ` The score falls short of the threshold by ${Math.abs(threshold - prediction.fake_score).toFixed(3)}.`
            : ""
        }`;

  return {
    analysis_id: analysis.analysis_id,
    generated_from_status: analysis.status,
    overview: {
      created_at: analysis.created_at,
      media_type: media.type ?? null,
      model_version: model.version ?? null,
      assessment: prediction.label ?? null,
      assessment_basis: `fake_score ${prediction.fake_score ?? "?"} vs decision threshold ${
        prediction.threshold ?? "?"
      } (${prediction.threshold_version ?? "?"})`,
    },
    media: Object.fromEntries(
      (Object.entries(media) as Array<[string, unknown]>).filter(
        ([, v]) => v !== null && v !== undefined,
      ),
    ),
    prediction: {
      fake_score: prediction.fake_score,
      threshold: prediction.threshold,
      threshold_version: prediction.threshold_version,
      label: prediction.label,
      method: prediction.method,
      score_is_calibrated: prediction.score_is_calibrated ?? false,
      window_variation: variation,
    },
    timeline: {
      note: "Window scores are placed on the source timeline when fps and source frame indices are available from the inference backend.",
      windows,
      threshold: prediction.threshold ?? null,
    },
    explainability: analysis.explainability ?? { available: false },
    artifacts: analysis.artifacts ?? {},
    technical: {
      model,
      frames_analyzed: media.frames_analyzed ?? null,
      prediction_method: prediction.method ?? null,
      processing_duration_seconds: processing.duration_seconds ?? null,
      warnings: processing.warnings ?? [],
    },
    interpretation: `${stance} Scoring method: ${prediction.method ?? "unknown"}. Because the model is uncalibrated and was evaluated as weak on held-out data (see evaluation/EVALUATION.md), this assessment must be treated as a research-grade indication and reviewed by a human before any consequential use.${
      (processing.warnings ?? []).length ? " Quality warnings were raised during processing." : ""
    }`,
    limitations: REPORT_LIMITATIONS,
  };
}
