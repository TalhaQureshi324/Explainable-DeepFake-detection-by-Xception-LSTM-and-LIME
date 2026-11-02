"use client";

/**
 * Analysis polling hook: owns the AbortController lifecycle so polling is
 * cleaned up on unmount/navigation and React StrictMode double-mounts.
 */

import { useEffect, useState } from "react";
import { ApiError, pollAnalysis } from "./client";
import type { AnalysisResponse } from "./types";

export type PollPhase =
  | { kind: "loading" }
  | { kind: "polling"; analysis: AnalysisResponse }
  | { kind: "done"; analysis: AnalysisResponse }
  | { kind: "error"; error: ApiError };

export function useAnalysisPolling(analysisId: string) {
  const [phase, setPhase] = useState<PollPhase>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setPhase({ kind: "loading" });

    pollAnalysis(analysisId, {
      signal: controller.signal,
      onState: (a) => setPhase({ kind: "polling", analysis: a }),
    })
      .then((a) => setPhase({ kind: "done", analysis: a }))
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setPhase({
          kind: "error",
          error:
            e instanceof ApiError
              ? e
              : new ApiError(0, e instanceof Error ? e.message : "unknown error"),
        });
      });

    return () => controller.abort();
  }, [analysisId]);

  return phase;
}

/** Maps backend error codes to localized, user-safe messages. */
export function errorCodeToMessageKey(code: string | null | undefined): string | null {
  switch (code) {
    case "NO_FACE_DETECTED":
      return "errNoFace";
    case "INSUFFICIENT_FRAMES":
      return "errInsufficientFrames";
    case "MEDIA_UNREADABLE":
      return "errMediaUnreadable";
    case "MODEL_UNAVAILABLE":
      return "errModelUnavailable";
    case "SERVER_BUSY":
      return "errServerBusy";
    case "INTERNAL_ERROR":
      return "errInternal";
    default:
      return null;
  }
}
