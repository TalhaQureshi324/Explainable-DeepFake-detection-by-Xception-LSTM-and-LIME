"use client";

/**
 * Media upload + analysis launcher (subphase 2B/2C).
 * Validation mirrors the backend's actual rules (docs/API.md): extension allow
 * list and DEEPFAKE_MAX_UPLOAD_MB (default 200 MB). Backend limits are
 * surfaced as a constant here; the backend remains the enforcement point.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, FileVideo, ImageIcon, Loader2, UploadCloud } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import {
  Alert, Badge, Button, Card, CardContent, Select, Switch,
} from "@/components/ui/primitives";
import { ApiError, createAnalysis, getModels } from "@/lib/api/client";
import type { ModelSummary } from "@/lib/api/types";

const VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const MAX_UPLOAD_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB || 200);

type Preview = { url: string; kind: "video" | "image"; name: string; sizeMb: number };

export default function AnalyzePage() {
  const { t } = useI18n();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [explain, setExplain] = useState(false);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelVersion, setModelVersion] = useState<string>("");
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let alive = true;
    getModels()
      .then((res) => {
        if (!alive) return;
        setModels(res.models.filter((m) => m.available));
        setModelVersion(res.preferred_version ?? res.models[0]?.version ?? "");
      })
      .catch((e: ApiError) => alive && setModelsError(e.detail));
    return () => {
      alive = false;
    };
  }, []);

  const validate = useCallback(
    (f: File): string | null => {
      const ext = `.${f.name.split(".").pop()?.toLowerCase() ?? ""}`;
      const isVideo = VIDEO_EXT.includes(ext);
      const isImage = IMAGE_EXT.includes(ext);
      if (!isVideo && !isImage) return t.badType;
      if (f.size === 0) return t.emptyFile;
      if (f.size > MAX_UPLOAD_MB * 1024 * 1024) return t.tooLarge(MAX_UPLOAD_MB);
      return null;
    },
    [t],
  );

  const acceptFile = useCallback(
    (f: File) => {
      const problem = validate(f);
      setError(problem);
      if (problem) {
        setFile(null);
        setPreview(null);
        return;
      }
      const ext = `.${f.name.split(".").pop()?.toLowerCase() ?? ""}`;
      const kind = VIDEO_EXT.includes(ext) ? "video" : "image";
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return { url: URL.createObjectURL(f), kind, name: f.name, sizeMb: f.size / (1024 * 1024) };
      });
      setFile(f);
    },
    [validate],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) acceptFile(f);
    },
    [acceptFile],
  );

  const submit = useCallback(async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await createAnalysis(file, {
        modelVersion: modelVersion || undefined,
        explain,
      });
      router.push(`/analysis/${res.analysis_id}`);
    } catch (e) {
      const apiErr = e as ApiError;
      setError(apiErr.status === 0 ? t.backendDown : `${t.uploadFailed}: ${apiErr.detail}`);
      setSubmitting(false);
    }
  }, [file, submitting, modelVersion, explain, router, t]);

  const exts = useMemo(() => [...VIDEO_EXT, ...IMAGE_EXT].join(","), []);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <h1 className="text-2xl font-bold tracking-tight">{t.uploadTitle}</h1>
      <p className="mt-2 text-sm text-ink-2">{t.uploadSubtitle}</p>

      {modelsError && (
        <Alert tone="danger" className="mt-4" title={t.backendDown}>
          {modelsError}
        </Alert>
      )}

      {/* Dropzone */}
      <div
        role="button"
        tabIndex={0}
        aria-label={t.dropHere}
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        className={`mt-6 cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
          dragging ? "border-accent bg-accent/5" : "border-line bg-surface-1 hover:bg-surface-2"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={exts}
          className="hidden"
          onChange={(e) => e.target.files?.[0] && acceptFile(e.target.files[0])}
        />
        <UploadCloud className="mx-auto h-10 w-10 text-accent" aria-hidden />
        <p className="mt-3 font-medium">{t.dropHere}</p>
        <p className="mt-1 text-xs text-ink-3">
          {t.dropOr} {t.browseFiles}
        </p>
        <p className="mt-3 text-xs text-ink-3">{t.dropFormats}</p>
        <p className="text-xs text-ink-3">{t.dropLimits(MAX_UPLOAD_MB)}</p>
      </div>

      {error && (
        <Alert tone="danger" className="mt-4">
          <span className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" aria-hidden /> {error}
          </span>
        </Alert>
      )}

      {/* Preview + options */}
      {preview && (
        <Card className="mt-6 animate-fade-up">
          <CardContent className="p-5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold">{t.selectedFile}</p>
              <Badge tone={preview.kind === "video" ? "accent" : "neutral"}>
                {preview.kind === "video" ? (
                  <FileVideo className="h-3.5 w-3.5" aria-hidden />
                ) : (
                  <ImageIcon className="h-3.5 w-3.5" aria-hidden />
                )}
                {preview.kind}
              </Badge>
            </div>
            <div className="mt-4 overflow-hidden rounded-lg border border-line bg-black">
              {preview.kind === "video" ? (
                <video src={preview.url} controls className="mx-auto max-h-72 w-full" />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={preview.url}
                  alt={preview.name}
                  className="mx-auto max-h-72 w-auto object-contain"
                />
              )}
            </div>
            <p className="mt-2 truncate font-mono text-xs text-ink-3">
              {preview.name} · {preview.sizeMb.toFixed(1)} MB
            </p>
          </CardContent>
        </Card>
      )}

      <Card className="mt-4">
        <CardContent className="grid gap-5 p-5 sm:grid-cols-2">
          <div>
            <label htmlFor="model" className="mb-1.5 block text-sm font-medium">
              {t.modelLabel}
            </label>
            <Select
              id="model"
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
              disabled={models.length === 0}
              className="w-full"
            >
              {models.map((m) => (
                <option key={m.version} value={m.version}>
                  {m.version} — {m.input_frames} frames · thr {m.decision_threshold}
                </option>
              ))}
            </Select>
          </div>
          <div className="flex items-end pb-1">
            <Switch
              id="explain"
              checked={explain}
              onChange={setExplain}
              label={t.explainToggle}
              hint={t.explainHint}
            />
          </div>
        </CardContent>
      </Card>

      <div className="mt-6 flex justify-end">
        <Button size="lg" onClick={submit} disabled={!file || submitting || !!error}>
          {submitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {t.analyzing}
            </>
          ) : (
            t.analyzeButton
          )}
        </Button>
      </div>
    </div>
  );
}
