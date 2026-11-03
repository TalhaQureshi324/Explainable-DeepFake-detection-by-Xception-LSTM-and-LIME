"use client";

/**
 * Analysis history (Phase 4E): cursor-paginated, filterable list of the
 * signed-in user's persisted analyses. Server-side pagination; deletion
 * cascades to stored artifacts via the owner-verified API.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { FileText, Loader2, Search, Trash2 } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { useAuth } from "@/lib/auth/context";
import { ApiError, deleteAnalysis, getHistory } from "@/lib/api/client";
import type { HistoryItem } from "@/lib/api/types";
import {
  Alert, Badge, Button, ButtonLink, Card, CardContent, Select, Skeleton,
} from "@/components/ui/primitives";

export default function HistoryPage() {
  const { t } = useI18n();
  const { user, loading: authLoading } = useAuth();

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [label, setLabel] = useState("");
  const [mediaType, setMediaType] = useState("");
  const [status, setStatus] = useState("");
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const firstLoad = useRef(true);

  const load = useCallback(async (reset: boolean, currentCursor: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory({
        label: label || undefined,
        media_type: mediaType || undefined,
        status: status || undefined,
        search: search || undefined,
        cursor: reset ? undefined : currentCursor ?? undefined,
        limit: 12,
      });
      setItems((old) => (reset ? res.items : [...old, ...res.items]));
      setCursor(res.next_cursor);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setLoading(false);
      firstLoad.current = false;
    }
  }, [label, mediaType, status, search]);

  useEffect(() => {
    if (user) load(true, null);
  }, [user, load]);

  useEffect(() => {
    // refetch when filters change (debounced for the search box)
    if (!user) return;
    const h = setTimeout(() => load(true, null), search ? 350 : 0);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, mediaType, status, search, user]);

  const onDelete = async (id: string) => {
    if (!window.confirm(t.deleteConfirm)) return;
    setDeletingId(id);
    try {
      await deleteAnalysis(id);
      setItems((old) => old.filter((i) => i.id !== id));
      setNotice(t.deleted);
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setDeletingId(null);
    }
  };

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

  return (
    <div className="mx-auto max-w-5xl px-4 py-10">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">{t.historyTitle}</h1>
        <ButtonLink href="/dashboard" variant="ghost" size="sm">{t.navDashboard}</ButtonLink>
      </div>
      <p className="mt-1 text-xs text-ink-3">{t.historyNote}</p>

      {/* Filters */}
      <div className="mt-5 flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-52">
          <Search className="absolute start-2.5 top-2.5 h-4 w-4 text-ink-3" aria-hidden />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t.searchPlaceholder}
            aria-label={t.searchPlaceholder}
            className="h-10 w-full rounded-lg border border-line bg-surface-1 ps-9 pe-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          />
        </div>
        <Select value={mediaType} onChange={(e) => setMediaType(e.target.value)} aria-label="media type">
          <option value="">{t.filterAll}</option>
          <option value="video">{t.filterVideo}</option>
          <option value="image">{t.filterImage}</option>
        </Select>
        <Select value={label} onChange={(e) => setLabel(e.target.value)} aria-label="verdict">
          <option value="">{t.filterAll}</option>
          <option value="REAL">{t.filterReal}</option>
          <option value="FAKE">{t.filterFake}</option>
        </Select>
        <Select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="status">
          <option value="">{t.filterAll}</option>
          <option value="COMPLETED">{t.filterCompleted}</option>
          <option value="FAILED">{t.filterFailed}</option>
        </Select>
      </div>

      {notice && <Alert tone="info" className="mt-4">{notice}</Alert>}
      {error && <Alert tone="danger" className="mt-4">{error.detail}</Alert>}

      {/* List */}
      <div className="mt-5 grid gap-3">
        {loading && items.length === 0 ? (
          <>
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </>
        ) : items.length === 0 && !loading ? (
          <Card>
            <CardContent className="p-8 text-center text-sm text-ink-2">
              {t.emptyHistory}
              <div className="mt-4">
                <ButtonLink href="/analyze" size="sm">{t.navAnalyze}</ButtonLink>
              </div>
            </CardContent>
          </Card>
        ) : (
          items.map((item) => (
            <Card key={item.id} className="transition-colors hover:bg-surface-2">
              <CardContent className="flex flex-wrap items-center gap-3 p-4">
                <Badge tone={item.label === "FAKE" ? "fake" : item.label === "REAL" ? "real" : "neutral"}>
                  {item.label ?? item.status}
                </Badge>
                <Badge tone="neutral">{item.media_type}</Badge>
                {item.status !== "COMPLETED" && (
                  <Badge tone={item.status === "FAILED" ? "warning" : "accent"}>{item.status}</Badge>
                )}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{item.filename}</p>
                  <p className="text-xs text-ink-3" dir="ltr">
                    {item.fake_score != null ? `score ${item.fake_score}` : ""} ·{" "}
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  <Link
                    href={`/analysis/${item.id}`}
                    className="rounded-lg border border-line px-3 py-1.5 text-xs transition-colors hover:bg-surface-3"
                  >
                    {item.status === "COMPLETED" ? t.openReport : t.statusTitle}
                  </Link>
                  {item.status === "COMPLETED" && (
                    <a
                      href={`/api/inference/v1/analyses/${item.id}/report.pdf`}
                      download
                      className="rounded-lg border border-line px-3 py-1.5 text-xs transition-colors hover:bg-surface-3"
                    >
                      <FileText className="h-3.5 w-3.5 inline rtl:hidden" aria-hidden />
                      PDF
                    </a>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDelete(item.id)}
                    disabled={deletingId === item.id}
                    aria-label={t.deleteAction}
                    className="text-verdict-fake"
                  >
                    {deletingId === item.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                    ) : (
                      <Trash2 className="h-4 w-4" aria-hidden />
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {cursor && !loading && (
        <div className="mt-5 text-center">
          <Button variant="secondary" onClick={() => load(false, cursor)}>{t.loadMore}</Button>
        </div>
      )}
    </div>
  );
}
