/**
 * Hugging Face provider (Phase 5): talks DIRECTLY to our public Gradio Space
 * from the browser via the official @gradio/client. Media never crosses
 * Vercel. No token is used — the Space is public.
 *
 * Two endpoints:
 *  - "/predict_deepfake"  — the original, always available (Markdown + images)
 *  - "/analyze_structured" — the extended structured endpoint (deployed with
 *    the hf_space/ app in this repository); used automatically when present,
 *    with graceful fallback to the basic endpoint.
 */

import { Client, handle_file } from "@gradio/client";
import type { StoredAnalysis } from "./types";
import {
  SPACE_THRESHOLD,
  basicResultToAnalysis,
  structuredResultToAnalysis,
} from "./result-adapter";
import type { InferenceProvider, PredictOptions } from "./types";

const SPACE_ID = process.env.NEXT_PUBLIC_HF_SPACE_ID || "MuhammadTalha121/Deep-Fake-Detection";

type GradioFile = { url?: string; path?: string } | string | null | undefined;

function fileUrl(value: GradioFile): string | null {
  if (!value) return null;
  if (typeof value === "string") return value;
  return value.url ?? (value.path && value.path.startsWith("http") ? value.path : null);
}

let clientPromise: Promise<Client> | null = null;

function connect(): Promise<Client> {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_ID).catch((err: unknown) => {
      clientPromise = null; // allow retry (Space waking up, transient errors)
      throw err;
    });
  }
  return clientPromise;
}

interface PredictDeps {
  onStatus?: (message: string) => void;
}

async function runPredict(
  file: File,
  opts: PredictOptions,
  structured: boolean,
  deps?: PredictDeps,
): Promise<StoredAnalysis> {
  deps?.onStatus?.("connecting");
  const app = await connect();

  const payload: Record<string, unknown> = {
    input_file: handle_file(file),
    enable_xai: opts.enableXai,
  };
  if (!structured) {
    payload.lang_display = "English";
  }

  deps?.onStatus?.("running");
  const started = Date.now();
  let data: unknown[];
  try {
    const result = await app.predict(structured ? "/analyze_structured" : "/predict_deepfake", payload);
    data = result.data as unknown[];
  } catch (err: unknown) {
    if (structured) {
      // extended endpoint not deployed yet -> fall back to the basic endpoint
      return runPredict(file, opts, false, deps);
    }
    const message = err instanceof Error ? err.message : String(err);
    if (/sleep|asleep|paused/i.test(message)) {
      throw new Error(
        "The Hugging Face Space is asleep or paused. Open the Space once to wake it, then retry.",
      );
    }
    throw new Error(`Hugging Face Space error: ${message}`);
  }
  const seconds = (Date.now() - started) / 1000;

  const markdown = typeof data?.[0] === "string" ? (data[0] as string) : "";
  const faceUrl = fileUrl(data?.[1] as GradioFile);
  const limeUrl = fileUrl(data?.[2] as GradioFile);

  if (structured) {
    // [0] = JSON string, [1] face, [2] lime, [3] pdf (when generated)
    let payloadStructured: Record<string, unknown> | null = null;
    try {
      payloadStructured = JSON.parse(data?.[0] as string);
    } catch {
      payloadStructured = null;
    }
    if (!payloadStructured) {
      throw new Error("The structured endpoint returned an unreadable result.");
    }
    const record = structuredResultToAnalysis(payloadStructured);
    record.artifacts = {
      processed_face: faceUrl ?? record.artifacts?.processed_face ?? null,
      lime: limeUrl ?? record.artifacts?.lime ?? null,
    };
    (record.artifacts as unknown as Record<string, unknown>).pdf = fileUrl(data?.[3] as GradioFile);
    record.processing = record.processing ?? {};
    record.processing.duration_seconds = record.processing.duration_seconds ?? seconds;
    return record;
  }

  return basicResultToAnalysis(
    {
      markdown,
      facePath: null,
      limePath: null,
      faceUrl,
      limeUrl,
      seconds,
    },
    { filename: opts.filename, mediaType: file.type.startsWith("video") ? "video" : "image", spaceId: SPACE_ID },
  );
}

export function createHuggingFaceProvider(): InferenceProvider {
  return {
    id: "huggingface",
    statusLabel: (phase) =>
      phase === "connecting"
        ? "Connecting to the Hugging Face Space (may wake it from sleep)…"
        : "Running inference on the Hugging Face Space…",
    async predict(file: File, opts: PredictOptions & { onStatus?: (m: string) => void }) {
      const { onStatus, ...rest } = opts;
      const structured = process.env.NEXT_PUBLIC_HF_STRUCTURED !== "false";
      try {
        return await runPredict(file, rest, structured, { onStatus });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (/analyze_structured/i.test(message)) {
          // direct fallback when the error names the missing endpoint
          return runPredict(file, rest, false, { onStatus });
        }
        throw err;
      }
    },
  };
}

export { SPACE_ID, SPACE_THRESHOLD };
