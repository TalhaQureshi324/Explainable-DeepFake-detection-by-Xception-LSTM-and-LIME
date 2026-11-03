"""Pydantic schemas for analyses and predictions (API contract v1)."""

from typing import List, Optional
from pydantic import BaseModel, Field

# This schema is versioned: additive, optional fields only. The Next.js
# frontend and future Supabase persistence depend on it staying compatible.


class WindowScoreOut(BaseModel):
    label: str
    score: float
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None


class MediaInfo(BaseModel):
    type: str                                   # "video" | "image"
    filename: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    frames_extracted: Optional[int] = None
    frames_analyzed: Optional[int] = None
    valid_face_detections: Optional[int] = None


class PredictionOut(BaseModel):
    label: str | None                           # "REAL" | "FAKE" (null while processing)
    fake_score: float | None
    threshold: float | None
    threshold_version: str | None
    method: str | None                          # full_sequence | sliding_window_ensemble | replicated_frame_sequence
    score_is_calibrated: bool = Field(
        default=False,
        description="Sigmoid outputs are NOT calibrated probabilities; "
        "no calibration evidence exists for these checkpoints.",
    )


class ModelInfo(BaseModel):
    name: str | None = "Xception-LSTM"
    version: str | None
    description: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    input_frames: Optional[int] = None
    input_size: Optional[int] = None


class ProcessingInfo(BaseModel):
    duration_seconds: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    error_code: Optional[str] = None


class ExplainabilityInfo(BaseModel):
    available: bool = False
    method: Optional[str] = None
    artifact_id: Optional[str] = None
    limitation: Optional[str] = None


class ArtifactsInfo(BaseModel):
    processed_face: Optional[str] = None   # relative URL to the processed-face PNG
    lime: Optional[str] = None             # relative URL to the LIME PNG (when requested)


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str                                 # QUEUED..COMPLETED/FAILED
    created_at: str
    updated_at: str
    media: Optional[MediaInfo] = None
    prediction: Optional[PredictionOut] = None
    model: Optional[ModelInfo] = None
    processing: ProcessingInfo = ProcessingInfo()
    explainability: ExplainabilityInfo = ExplainabilityInfo()
    artifacts: ArtifactsInfo = ArtifactsInfo()
    windows: List[WindowScoreOut] = Field(default_factory=list)
    report_available: bool = False

    class Config:
        protected_namespaces = ()  # allow fields starting with model_* if added later


class AnalysisCreateAccepted(BaseModel):
    analysis_id: str
    status: str = "QUEUED"
    poll_url: str


class ModelSummary(BaseModel):
    version: str
    description: Optional[str] = None
    available: bool = True
    file: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    input_frames: Optional[int] = None
    input_size: Optional[int] = None
    decision_threshold: Optional[float] = None
    loaded: bool = False


class ModelsResponse(BaseModel):
    preferred_version: Optional[str] = None
    models: List[ModelSummary]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v1"


class ReadyResponse(BaseModel):
    ready: bool
    model_version: Optional[str] = None
    detail: Optional[str] = None
