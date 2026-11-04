/**
 * Provider factory (Phase 5).
 *
 * NEXT_PUBLIC_INFERENCE_MODE selects the inference backend:
 *   "huggingface" (default) — public HF Space, browser-direct, guest mode
 *   "fastapi"               — the modular Phase 1-4 backend (auth + history)
 *
 * Both return the same StoredAnalysis contract; results/report/PDF surfaces
 * are provider-agnostic.
 */

import { createHuggingFaceProvider } from "./huggingface-provider";
import { createFastApiProvider } from "./fastapi-provider";
import type { InferenceProvider } from "./types";

export type InferenceMode = "huggingface" | "fastapi";

export function inferenceMode(): InferenceMode {
  return process.env.NEXT_PUBLIC_INFERENCE_MODE === "fastapi" ? "fastapi" : "huggingface";
}

export function getInferenceProvider(): InferenceProvider {
  if (inferenceMode() === "fastapi") {
    // The FastAPI provider wraps lib/api/client.ts (POST /v1/analyses + polling).
    return createFastApiProvider();
  }
  return createHuggingFaceProvider();
}
