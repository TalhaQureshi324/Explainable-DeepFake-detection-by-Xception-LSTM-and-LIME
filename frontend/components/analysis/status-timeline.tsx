"use client";

/**
 * Live processing-state display. Renders ONLY the states the backend actually
 * reports (docs/API.md) — no invented percentages.
 */

import { CheckCircle2, CircleDashed, Loader2, XCircle } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import type { JobState } from "@/lib/api/types";

const ORDER: JobState[] = [
  "QUEUED",
  "PREPROCESSING",
  "INFERENCING",
  "EXPLAINING",
  "GENERATING_REPORT",
];

const LABEL_KEYS: Record<JobState, keyof ReturnType<typeof useI18n>["t"]> = {
  QUEUED: "statusQueued",
  PREPROCESSING: "statusPreprocessing",
  INFERENCING: "statusInferencing",
  EXPLAINING: "statusExplaining",
  GENERATING_REPORT: "statusReport",
  COMPLETED: "statusCompleted",
  FAILED: "statusFailed",
};

export function StatusTimeline({
  current,
  explainRequested,
  failedDetail,
}: {
  current: JobState;
  /** EXPLAINING only ever happens when the user asked for LIME. */
  explainRequested?: boolean;
  failedDetail?: string | null;
}) {
  const { t } = useI18n();
  const failed = current === "FAILED";
  const completed = current === "COMPLETED";
  const steps: JobState[] = explainRequested
    ? ORDER
    : ORDER.filter((s) => s !== "EXPLAINING");

  const currentIndex = steps.indexOf(current as JobState);

  return (
    <ol className="space-y-1" aria-live="polite">
      {steps.map((state, i) => {
        const done = completed || (currentIndex >= 0 && i < currentIndex);
        const active = !completed && !failed && current === state;
        return (
          <li
            key={state}
            className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm ${
              active ? "bg-accent/10 text-ink-1" : done ? "text-ink-2" : "text-ink-3"
            }`}
          >
            {done ? (
              <CheckCircle2 className="h-4 w-4 shrink-0 text-verdict-real" aria-hidden />
            ) : active ? (
              <Loader2 className="h-4 w-4 shrink-0 animate-spin text-accent" aria-hidden />
            ) : (
              <CircleDashed className="h-4 w-4 shrink-0" aria-hidden />
            )}
            <span className={active ? "font-semibold" : undefined}>
              {t[LABEL_KEYS[state]] as string}
            </span>
          </li>
        );
      })}
      {failed ? (
        <li className="flex items-center gap-3 rounded-lg bg-verdict-fake/10 px-3 py-2.5 text-sm font-semibold text-verdict-fake">
          <XCircle className="h-4 w-4 shrink-0" aria-hidden />
          {t.statusFailed}
        </li>
      ) : null}
      {failed && failedDetail ? (
        <li className="px-3 pb-1 text-xs text-ink-2">{failedDetail}</li>
      ) : null}
    </ol>
  );
}
