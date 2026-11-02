"use client";

/**
 * Report preview (Phase 3 groundwork): renders the structured report document
 * from GET /v1/analyses/{id}/report. One source of truth — the future PDF must
 * show exactly these numbers. The PDF button is deliberately non-functional.
 */

import { useEffect, useState } from "react";
import { ArrowLeft, Download, FileText } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { ApiError, artifactUrl, getReport } from "@/lib/api/client";
import type { ReportData } from "@/lib/api/types";
import { WindowsChart } from "@/components/analysis/windows-chart";
import {
  Alert, Badge, Button, ButtonLink, Card, CardContent, CardHeader, CardTitle, Skeleton,
} from "@/components/ui/primitives";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Kv({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-1.5 last:border-b-0">
      <span className="shrink-0 text-xs text-ink-3">{k}</span>
      <span className="break-all text-right font-mono text-xs">{v}</span>
    </div>
  );
}

export default function ReportPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const { t } = useI18n();
  const [report, setReport] = useState<ReportData | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    let alive = true;
    getReport(id)
      .then((r) => alive && setReport(r))
      .catch((e: ApiError) => alive && setError(e));
    return () => {
      alive = false;
    };
  }, [id]);

  // Artifact ids are deterministic (face_<id[:20]>) and the LIME id comes from
  // the analysis record via the report's explainability block.
  const faceUrl = artifactUrl(
    `/v1/artifacts/face_${(report?.analysis_id ?? id).replace(/-/g, "").slice(0, 20)}`,
  );
  const limeId = (report?.explainability as { artifact_id?: string } | undefined)?.artifact_id;
  const limeUrl = artifactUrl(limeId ? `/v1/artifacts/${limeId}` : null);

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <div className="mb-5 flex items-center justify-between">
        <ButtonLink href={`/analysis/${id}`} variant="ghost" size="sm">
          <ArrowLeft className="h-4 w-4 rtl:rotate-180" aria-hidden />
          {t.resultsTitle}
        </ButtonLink>
        <span className="font-mono text-xs text-ink-3" dir="ltr">
          {t.analysisId}: {id.slice(0, 8)}…
        </span>
      </div>

      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">{t.reportTitle}</h1>
        <div className="relative group">
          <Button variant="secondary" size="sm" disabled aria-disabled="true">
            <Download className="h-4 w-4" aria-hidden />
            {t.pdfButton}
          </Button>
          <span className="pointer-events-none absolute -top-9 start-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-ink-1 px-2 py-1 text-[11px] text-surface-1 opacity-0 transition-opacity group-hover:opacity-100 rtl:translate-x-1/2">
            {t.pdfDisabledNote}
          </span>
        </div>
      </div>

      {error && (
        <Alert tone="danger" title={error.status === 409 ? t.statusTitle : t.uploadFailed}>
          {error.status === 409
            ? t.noData
            : error.status === 404
              ? t.notFound
              : error.detail}
        </Alert>
      )}

      {!report && !error && (
        <div className="grid gap-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
        </div>
      )}

      {report && (
        <div className="grid gap-5">
          {/* 1. Overview */}
          <Section title={t.reportSectionOverview}>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={report.overview.assessment === "FAKE" ? "fake" : "real"}>
                <FileText className="h-3 w-3" aria-hidden />
                {report.overview.assessment}
              </Badge>
              <Badge tone="neutral">{report.overview.media_type}</Badge>
              <Badge tone="accent">{report.overview.model_version}</Badge>
            </div>
            <p className="mt-3 font-mono text-xs text-ink-2" dir="ltr">
              {report.overview.assessment_basis}
            </p>
            <p className="mt-1 text-xs text-ink-3" dir="ltr">
              {t.analysisId}: {report.analysis_id} · {t.created}: {report.overview.created_at}
            </p>
          </Section>

          {/* 3. Prediction summary */}
          <Section title={t.reportSectionPrediction}>
            <div className="grid gap-x-8 sm:grid-cols-2">
              <div>
                <Kv k={t.fakeScore} v={report.prediction.fake_score?.toFixed(6) ?? "—"} />
                <Kv k={t.threshold} v={report.prediction.threshold ?? "—"} />
                <Kv k={t.thresholdVersion} v={report.prediction.threshold_version ?? "—"} />
              </div>
              <div>
                <Kv k={t.method} v={report.prediction.method ?? "—"} />
                <Kv k="score_is_calibrated" v={String(report.prediction.score_is_calibrated)} />
                <Kv
                  k="window stdev"
                  v={report.prediction.window_variation?.stdev?.toFixed(4) ?? "—"}
                />
              </div>
            </div>
            <Alert tone="info" className="mt-3">{t.uncalibrated}</Alert>
          </Section>

          {/* 4. Timeline */}
          {report.timeline.windows?.length > 0 ? (
            <Section title={t.reportSectionTimeline}>
              <WindowsChart
                windows={report.timeline.windows}
                threshold={report.timeline.threshold}
              />
            </Section>
          ) : null}

          {/* 5+6. Face + LIME */}
          <div className="grid gap-5 sm:grid-cols-2">
            <Section title={t.reportSectionFace}>
              {faceUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={faceUrl}
                  alt={t.reportSectionFace}
                  className="mx-auto max-h-64 rounded-lg border border-line object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.visibility = "hidden";
                  }}
                />
              ) : (
                <p className="text-sm text-ink-3">{t.noData}</p>
              )}
            </Section>
            <Section title={t.reportSectionLime}>
              {limeUrl ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={limeUrl}
                    alt={t.limeTitle}
                    className="mx-auto max-h-64 rounded-lg border border-line object-contain"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.visibility = "hidden";
                    }}
                  />
                  <p className="mt-3 text-xs leading-relaxed text-ink-3">{t.limeLimitation}</p>
                </>
              ) : (
                <p className="text-sm text-ink-3">{t.noData}</p>
              )}
            </Section>
          </div>

          {/* 7. Technical */}
          <Section title={t.reportSectionTechnical}>
            <div className="grid gap-x-8 sm:grid-cols-2">
              <div>
                <Kv k="model" v={report.technical.model ? String((report.technical.model as { version?: string }).version ?? "—") : "—"} />
                <Kv k="checkpoint" v={String((report.technical.model as { checkpoint_sha256?: string })?.checkpoint_sha256 ?? "—").slice(0, 24) + "…"} />
                <Kv k="frames" v={report.technical.frames_analyzed ?? "—"} />
              </div>
              <div>
                <Kv k="method" v={report.technical.prediction_method ?? "—"} />
                <Kv k="duration" v={report.technical.processing_duration_seconds != null ? `${report.technical.processing_duration_seconds}s` : "—"} />
                <Kv k="status" v={report.generated_from_status} />
              </div>
            </div>
          </Section>

          {/* Interpretation */}
          <Section title={t.reportInterpretation}>
            <p className="text-sm leading-relaxed text-ink-2">{report.interpretation}</p>
          </Section>

          {/* 8. Warnings + limitations */}
          <Section title={t.reportSectionLimits}>
            {(report.technical.warnings?.length ?? 0) > 0 && (
              <ul className="mb-3 list-inside list-disc text-sm text-ink-2">
                {report.technical.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
            <ul className="list-inside list-disc space-y-1.5 text-xs leading-relaxed text-ink-3">
              {report.limitations?.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </Section>
        </div>
      )}
    </div>
  );
}
