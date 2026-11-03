"use client";

/**
 * Dashboard (Phase 4E): genuinely calculated summary metrics over the signed-in
 * user's persisted analyses, plus recent activity. These numbers describe
 * stored model outputs — not real-world prevalence and not model accuracy.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { useI18n } from "@/lib/i18n/context";
import { useAuth } from "@/lib/auth/context";
import { getHistoryStats } from "@/lib/api/client";
import type { HistoryStats } from "@/lib/api/types";
import {
  Alert, Badge, ButtonLink, Card, CardContent, Skeleton,
} from "@/components/ui/primitives";

export default function DashboardPage() {
  const { t } = useI18n();
  const { user, loading: authLoading } = useAuth();
  const [stats, setStats] = useState<HistoryStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;
    getHistoryStats()
      .then(setStats)
      .catch((e) => setError(e?.detail ?? "Failed to load stats."));
  }, [user]);

  if (authLoading) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-10">
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="mx-auto max-w-md px-4 py-16">
        <Alert tone="info" title={t.authRequired}>
          <div className="mt-3 flex gap-2">
            <ButtonLink href="/login" size="sm">{t.signIn}</ButtonLink>
            <ButtonLink href="/register" size="sm" variant="secondary">{t.signUp}</ButtonLink>
          </div>
        </Alert>
      </div>
    );
  }

  const cards: Array<{ key: keyof HistoryStats; label: string; tone?: "accent" | "real" | "fake" | "warning" }> = [
    { key: "total", label: t.statTotal },
    { key: "completed", label: t.statCompleted, tone: "accent" },
    { key: "images", label: t.statImages },
    { key: "videos", label: t.statVideos },
    { key: "fakes", label: t.statFakes, tone: "fake" },
    { key: "reals", label: t.statReals, tone: "real" },
    { key: "failed", label: t.statFailed, tone: "warning" },
  ];

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">{t.dashboardTitle}</h1>
        <ButtonLink href="/history" variant="ghost" size="sm">{t.navHistory}</ButtonLink>
      </div>

      {error && <Alert tone="danger" className="mt-4">{error}</Alert>}

      {!stats && !error ? (
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Array.from({ length: 7 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : stats ? (
        <>
          <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {cards.map((c) => (
              <Card key={c.key}>
                <CardContent className="p-4">
                  <p className="text-xs text-ink-3">{c.label}</p>
                  <p className="mt-1 text-2xl font-bold" style={
                    c.tone === "fake" ? { color: "rgb(var(--verdict-fake))" }
                      : c.tone === "real" ? { color: "rgb(var(--verdict-real))" }
                        : undefined
                  }>
                    {Number(stats[c.key] ?? 0)}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          <h2 className="mt-8 text-lg font-semibold">{t.recentAnalyses}</h2>
          <div className="mt-3 grid gap-3">
            {stats.recent.length === 0 ? (
              <Card>
                <CardContent className="p-8 text-center text-sm text-ink-2">
                  {t.emptyHistory}
                  <div className="mt-4">
                    <ButtonLink href="/analyze" size="sm">{t.navAnalyze}</ButtonLink>
                  </div>
                </CardContent>
              </Card>
            ) : (
              stats.recent.map((r) => (
                <Card key={r.id}>
                  <CardContent className="flex flex-wrap items-center gap-3 p-4">
                    <Badge tone={r.label === "FAKE" ? "fake" : r.label === "REAL" ? "real" : "neutral"}>
                      {r.label ?? r.status}
                    </Badge>
                    <Badge tone="neutral">{r.media_type}</Badge>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{r.filename}</p>
                      <p className="text-xs text-ink-3" dir="ltr">
                        {new Date(r.created_at).toLocaleString()}
                      </p>
                    </div>
                    <Link
                      href={`/analysis/${r.id}`}
                      className="rounded-lg border border-line px-3 py-1.5 text-xs transition-colors hover:bg-surface-3"
                    >
                      {r.status === "COMPLETED" ? t.openReport : t.statusTitle}
                    </Link>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </>
      ) : null}
    </div>
  );
}
