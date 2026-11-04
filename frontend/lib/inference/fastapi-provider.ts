/**
 * FastAPI provider (optional mode): wraps the Phase 1-4 backend so the same
 * StoredAnalysis contract works when INFERENCE_API_URL is hosted again
 * (selected via NEXT_PUBLIC_INFERENCE_MODE=fastapi).
 */

import { createAnalysis, getAnalysis } from "@/lib/api/client";
import type { InferenceProvider, PredictOptions, StoredAnalysis } from "./types";
import { saveAnalysis } from "./types";

export function createFastApiProvider(): InferenceProvider {
  return {
    id: "fastapi",
    async predict(file: File, opts: PredictOptions) {
      const accepted = await createAnalysis(file, {
        modelVersion: undefined,
        explain: opts.enableXai,
      });
      // poll to completion
      for (let i = 0; i < 200; i++) {
        const analysis = await getAnalysis(accepted.analysis_id);
        if (analysis.status === "COMPLETED" || analysis.status === "FAILED") {
          const record = analysis as unknown as StoredAnalysis;
          saveAnalysis(record);
          if (record.status === "FAILED") {
            throw new Error(record.processing?.error ?? "Analysis failed.");
          }
          return record;
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      throw new Error("Timed out waiting for the analysis.");
    },
  };
}

