"use client";

/**
 * Analysis page (processing states): polls GET /v1/analyses/{id} and shows the
 * live backend state machine plus recoverable failure views. 404s (backend
 * restart loses in-memory jobs) are handled explicitly.
 */

import { use } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { errorCodeToMessageKey, useAnalysisPolling } from "@/lib/api/use-analysis";
import { StatusTimeline } from "@/components/analysis/status-timeline";
import { Alert, Badge, ButtonLink, Skeleton } from "@/components/ui/primitives";

export default function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { t } = useI18n();
  const phase = useAnalysisPolling(id);

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <div className="mb-5 flex items-center justify-between">
        <Link
          href="/analyze"
          className="inline-flex items-center gap-1.5 text-sm text-ink-2 transition-colors hover:text-ink-1"
        >
          <ArrowLeft className="h-4 w-4 rtl:rotate-180" aria-hidden />
          {t.navAnalyze}
        </Link>
        <span className="font-mono text-xs text-ink-3" dir="ltr">
          {t.analysisId}: {id.slice(0, 8)}…
        </span>
      </div>

      <h1 className="mb-6 text-2xl font-bold tracking-tight">{t.statusTitle}</h1>

      {phase.kind === "loading" && (
        <div className="mx-auto max-w-xl">
          <Skeleton className="h-64" />
        </div>
      )}

      {phase.kind === "error" && (
        <Alert tone="danger">
          {phase.error.status === 404
            ? t.notFound
            : phase.error.status === 504
              ? t.timeout
              : phase.error.status === 0
                ? t.backendDown
                : phase.error.detail}
        </Alert>
      )}

      {phase.kind === "polling" && (
        <div className="mx-auto max-w-xl">
          <StatusTimeline current={phase.analysis.status} />
          <p className="mt-3 text-center text-xs text-ink-3">{t.statusNote}</p>
        </div>
      )}

      {phase.kind === "done" && phase.analysis.status === "COMPLETED" && (
        <div className="mx-auto max-w-xl">
          <p className="text-sm text-ink-2">
            {t.resultsTitle}:{" "}
            <Badge tone={phase.analysis.prediction?.label === "FAKE" ? "fake" : "real"}>
              {phase.analysis.prediction?.label}
            </Badge>{" "}
            <span className="font-mono">{phase.analysis.prediction?.fake_score?.toFixed(4)}</span>
          </p>
        </div>
      )}

      {phase.kind === "done" && phase.analysis.status === "FAILED" && (
        <div className="mx-auto max-w-xl">
          <StatusTimeline
            current="FAILED"
            failedDetail={
              errorCodeToMessageKey(phase.analysis.processing?.error_code)
                ? (t[
                    errorCodeToMessageKey(phase.analysis.processing?.error_code) as keyof typeof t
                  ] as string)
                : phase.analysis.processing?.error ?? null
            }
          />
          <div className="mt-6 flex justify-center gap-3">
            <ButtonLink href="/analyze" variant="secondary">
              {t.retry}
            </ButtonLink>
            <ButtonLink href="/" variant="ghost">
              {t.backHome}
            </ButtonLink>
          </div>
        </div>
      )}
    </div>
  );
}
