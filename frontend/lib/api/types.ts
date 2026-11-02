/**
 * Typed interfaces mirroring backend/app/schemas/analysis.py (API contract v1).
 * The backend guarantees additive compatibility: new optional fields may appear;
 * existing fields keep their types. Keep these types in sync — they are the
 * single frontend source of truth for inference responses.
 */

export type JobState =
  | "QUEUED"
  | "PREPROCESSING"
  | "INFERENCING"
  | "EXPLAINING"
  | "GENERATING_REPORT"
  | "COMPLETED"
  | "FAILED";

export type PredictionMethod =
  | "full_sequence"
  | "sliding_window_ensemble"
  | "replicated_frame_sequence";

export interface WindowScore {
  label: string;
  score: number;
  frame_start: number | null;
  frame_end: number | null;
  start_seconds: number | null;
  end_seconds: number | null;
}

export interface MediaInfo {
  type: "video" | "image";
  filename: string;
  size_bytes: number | null;
  sha256: string | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  duration_seconds: number | null;
  frames_extracted: number | null;
  frames_analyzed: number | null;
  valid_face_detections: number | null;
}

export interface Prediction {
  label: "REAL" | "FAKE";
  fake_score: number;
  threshold: number;
  threshold_version: string;
  method: PredictionMethod;
  score_is_calibrated: boolean;
}

export interface ModelInfo {
  name: string;
  version: string;
  description: string | null;
  checkpoint_sha256: string | null;
  input_frames: number | null;
  input_size: number | null;
}

export interface ProcessingInfo {
  duration_seconds: number | null;
  warnings: string[];
  error: string | null;
  error_code: string | null;
}

export interface ExplainabilityInfo {
  available: boolean;
  method: string | null;
  artifact_id: string | null;
  limitation: string | null;
}

export interface ArtifactsInfo {
  processed_face: string | null;
  lime: string | null;
}

export interface AnalysisResponse {
  analysis_id: string;
  status: JobState;
  created_at: string;
  updated_at: string;
  media: MediaInfo | null;
  prediction: Prediction | null;
  model: ModelInfo | null;
  processing: ProcessingInfo;
  explainability: ExplainabilityInfo;
  artifacts: ArtifactsInfo;
  windows: WindowScore[];
  report_available: boolean;
}

export interface AnalysisCreateAccepted {
  analysis_id: string;
  status: string;
  poll_url: string;
}

export interface ModelSummary {
  version: string;
  description: string | null;
  available: boolean;
  file: string | null;
  checkpoint_sha256: string | null;
  input_frames: number | null;
  input_size: number | null;
  decision_threshold: number | null;
  loaded: boolean;
}

export interface ModelsResponse {
  preferred_version: string | null;
  models: ModelSummary[];
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ReadyResponse {
  ready: boolean;
  model_version: string | null;
  detail: string | null;
}

export interface ReportTimelineWindow {
  label: string;
  score: number;
  frame_start: number | null;
  frame_end: number | null;
  start_seconds: number | null;
  end_seconds: number | null;
}

export interface ReportData {
  analysis_id: string;
  generated_from_status: string;
  overview: {
    created_at: string;
    media_type: string | null;
    model_version: string | null;
    assessment: string | null;
    assessment_basis: string;
  };
  media: Record<string, string | number>;
  prediction: {
    fake_score: number | null;
    threshold: number | null;
    threshold_version: string | null;
    label: string | null;
    method: string | null;
    score_is_calibrated: boolean;
    window_variation: { min: number; max: number; stdev: number } | null;
  };
  timeline: {
    note: string;
    windows: ReportTimelineWindow[];
    threshold: number | null;
  };
  explainability: ExplainabilityInfo | Record<string, unknown>;
  technical: {
    model: ModelInfo | Record<string, unknown>;
    frames_analyzed: number | null;
    prediction_method: string | null;
    processing_duration_seconds: number | null;
    warnings: string[];
  };
  interpretation: string;
  limitations: string[];
}
