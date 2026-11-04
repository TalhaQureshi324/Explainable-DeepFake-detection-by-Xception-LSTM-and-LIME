"use client";

/**
 * Full results view for a COMPLETED analysis. Every value comes from the API
 * response; scores are explicitly labelled uncalibrated, the active threshold
 * and its version are shown, and image analyses surface the replicated-frame
 * compatibility warning.
 */

import Image from "next/image";
import { AlertTriangle, CircleCheck, CircleX, FileText } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { artifactUrl } from "@/lib/api/client";
import type { AnalysisResponse } from "@/lib/api/types";
import {
  Alert, ButtonLink, Card, CardContent, CardHeader, CardTitle,
} from "@/components/ui/primitives";
import { WindowsChart } from "./windows-chart";

const METHOD_KEY: Record<string, string> = {
  full_sequence: "methodFull",
  sliding_window_ensemble: "methodEnsemble",
  replicated_frame_sequence: "methodReplicated",
};

function fmtSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  return s >= 1 ? `${s.toFixed(1)}s` : `${Math.round(s * 1000)}ms`;
}

function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2 last:border-b-0">
      <dt className="shrink-0 text-xs text-ink-3">{label}</dt>
      <dd className="break-all text-right font-mono text-xs text-ink-1">{children}</dd>
    </div>
  );
}

export function ResultView({ analysis }: { analysis: AnalysisResponse }) {
  const { t } = useI18n();
  const p = analysis.prediction;
  const m = analysis.media;
  if (!p) return null;

  const fake = p.label === "FAKE";
  const faceUrl = artifactUrl(analysis.artifacts?.processed_face);
  const limeUrl = artifactUrl(analysis.artifacts?.lime);
  const isImage = m?.type === "image";
  const score = p.fake_score ?? null;
  const thr = p.threshold ?? null;

  return (
    <div className="grid gap-5 lg:grid-cols-5">
      {/* Verdict card */}
      <Card className={`animate-fade-up border-2 ${fake ? "border-verdict-fake/50" : "border-verdict-real/50"} lg:col-span-2`}>
        <CardContent className="p-6 text-center">
          {fake ? (
            <CircleX className="mx-auto h-12 w-12 text-verdict-fake" aria-hidden />
          ) : (
            <CircleCheck className="mx-auto h-12 w-12 text-verdict-real" aria-hidden />
          )}
          <p className="mt-3 text-3xl font-bold tracking-tight" style={{ color: fake ? "rgb(var(--verdict-fake))" : "rgb(var(--verdict-real))" }}>
            {fake ? t.verdictFake : t.verdictReal}
          </p>
          <p className="mt-1 text-xs text-ink-3">
            {fake ? t.verdictFake : t.verdictReal} · {t.resultsTitle}
          </p>

          <div className="mt-6 rounded-lg bg-surface-2 p-4" dir="ltr">
            <p className="text-xs text-ink-3">{t.fakeScore}</p>
            <p className="mt-1 font-mono text-2xl font-bold">
              {score !== null ? score.toFixed(4) : "—"}
            </p>
            {/* score vs threshold bar */}
            <div className="relative mt-3 h-2 rounded-full bg-surface-3">
              <div
                className="absolute inset-y-0 rounded-full"
                style={{
                  width: `${Math.min(100, (score ?? 0) * 100)}%`,
                  background: fake ? "rgb(var(--verdict-fake))" : "rgb(var(--verdict-real))",
                }}
              />
              <div
                className="absolute -top-1 h-4 w-0.5 bg-ink-1"
                style={{ left: `${Math.min(100, (thr ?? 0) * 100)}%` }}
                title={`${t.threshold}: ${thr ?? "—"}`}
              />
            </div>
            <div className="mt-1 flex justify-between text-[11px] text-ink-3">
              <span>0</span>
              <span>
                {t.threshold}: {thr ?? "—"} ({p.threshold_version})
              </span>
              <span>1</span>
            </div>
          </div>

          <Alert tone="info" className="mt-4">{t.uncalibrated}</Alert>

          <ButtonLink
            href={`/analysis/${analysis.analysis_id}/report`}
            className="mt-4 w-full"
            variant="secondary"
          >
            <FileText className="h-4 w-4 rtl:rotate-180" aria-hidden />
            {t.openReport}
          </ButtonLink>
        </CardContent>
      </Card>

      {/* Details */}
      <div className="grid gap-5 lg:col-span-3">
        {/* Method + model */}
        <Card>
          <CardHeader>
            <CardTitle>{t.reportSectionTechnical}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-x-8 sm:grid-cols-2">
            <dl>
              <MetaRow label={t.modelVersion}>{analysis.model?.version ?? "—"}</MetaRow>
              <MetaRow label={t.method}>{t[METHOD_KEY[p.method] as keyof typeof t] as string ?? p.method}</MetaRow>
              <MetaRow label={t.thresholdVersion}>{p.threshold_version}</MetaRow>
            </dl>
            <dl>
              <MetaRow label="checkpoint">{(analysis.model?.checkpoint_sha256 ?? "—").slice(0, 16)}…</MetaRow>
              <MetaRow label="input">{analysis.model?.input_frames ?? "—"} × {analysis.model?.input_size ?? "—"}</MetaRow>
              <MetaRow label={t.processingTime}>{fmtSeconds(analysis.processing?.duration_seconds)}</MetaRow>
            </dl>
          </CardContent>
        </Card>

        {/* Warnings */}
        {isImage && (
          <Alert tone="warning">
            <span className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              {t.imageCompatWarning}
            </span>
          </Alert>
        )}
        {(analysis.processing?.warnings?.length ?? 0) > 0 && (
          <Alert tone="warning" title={t.warnings}>
            <ul className="list-inside list-disc space-y-1">
              {analysis.processing.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </Alert>
        )}

        {/* Window chart (videos) */}
        {m?.type === "video" && analysis.windows?.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>{t.timelineTitle}</CardTitle>
            </CardHeader>
            <WindowsChart windows={analysis.windows} threshold={p.threshold} />
          </Card>
        )}

        {/* Artifacts: face + LIME */}
        <div className="grid gap-5 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t.facePreview}</CardTitle>
            </CardHeader>
            <CardContent>
              {faceUrl ? (
                <div className="overflow-hidden rounded-lg border border-line bg-black">
                  <Image
                    src={faceUrl}
                    alt={t.facePreview}
                    width={299}
                    height={299}
                    unoptimized
                    className="mx-auto h-auto w-full max-w-[299px] object-contain"
                  />
                </div>
              ) : (
                <p className="text-sm text-ink-3">{t.noData}</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>{t.limeTitle}</CardTitle>
            </CardHeader>
            <CardContent>
              {limeUrl ? (
                <>
                  <div className="overflow-hidden rounded-lg border border-line bg-black">
                    <Image
                      src={limeUrl}
                      alt={t.limeTitle}
                      width={299}
                      height={299}
                      unoptimized
                      className="mx-auto h-auto w-full max-w-[299px] object-contain"
                    />
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-ink-3">{t.limeLimitation}</p>
                </>
              ) : (
                <p className="text-sm text-ink-3">{t.noData}</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Media metadata */}
        <Card>
          <CardHeader>
            <CardTitle>{t.reportSectionMedia}</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid gap-x-8 sm:grid-cols-2">
              <div>
                <MetaRow label="file">{m?.filename ?? "—"}</MetaRow>
                <MetaRow label="sha256">{(m?.sha256 ?? "—").slice(0, 24)}…</MetaRow>
                <MetaRow label="resolution">{m?.width && m?.height ? `${m.width}×${m.height}` : "—"}</MetaRow>
              </div>
              <div>
                <MetaRow label="duration">
                  {m?.duration_seconds != null ? `${Number(m.duration_seconds).toFixed(2)}s` : "—"}
                </MetaRow>
                <MetaRow label="fps">{m?.fps ?? "—"}</MetaRow>
                <MetaRow label="frames">
                  {m?.frames_analyzed ?? "—"} / {m?.frames_extracted ?? "—"}
                </MetaRow>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
