/**
 * Runtime inference-API proxy (Phase 5).
 *
 * Replaces the build-time `next.config.mjs` rewrite: Vercel service bindings
 * and project environment variables are available to functions at RUNTIME
 * only, so the backend base URL is resolved here on every request instead of
 * being baked into the build manifest.
 *
 * The frontend calls relative `/api/inference/:path*` URLs; this handler
 * forwards them to the FastAPI backend at INFERENCE_API_URL (default
 * http://127.0.0.1:8000 for local development). Cookies, uploads, PDFs and
 * other binary responses pass through unchanged.
 *
 * NOTE on platform limits: serverless request bodies are limited (4.5 MB on
 * Vercel serverless functions). Large video uploads in a deployed
 * frontend-only setup should use a direct-to-backend or direct-to-Supabase
 * Storage upload path instead of this proxy (see docs/ARCHITECTURE.md).
 */

export const dynamic = "force-dynamic";

const DEFAULT_BACKEND = "http://127.0.0.1:8000";

function backendBase(): string {
  const url = process.env.INFERENCE_API_URL || DEFAULT_BACKEND;
  return url.replace(/\/+$/, "");
}

type Ctx = { params: Promise<{ path?: string[] }> };

async function handle(req: Request, ctx: Ctx): Promise<Response> {
  const { path } = await ctx.params;
  const target = `${backendBase()}/${(path ?? []).join("/")}${req.url.includes("?") ? `?${new URL(req.url).search}` : ""}`;

  const headers = new Headers();
  for (const name of ["content-type", "authorization", "accept", "content-disposition", "cookie"]) {
    const value = req.headers.get(name);
    if (value) headers.set(name, value);
  }

  let body: ArrayBuffer | undefined;
  if (!["GET", "HEAD"].includes(req.method)) {
    body = await req.arrayBuffer();
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body,
      redirect: "manual",
    });
  } catch {
    return Response.json(
      { detail: "The inference backend could not be reached." },
      { status: 502 },
    );
  }

  const responseHeaders = new Headers();
  for (const name of ["content-type", "content-length", "content-disposition"]) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  // session cookies must survive the proxy (httpOnly, SameSite=Lax from FastAPI)
  const setCookies = upstream.headers.getSetCookie?.() ?? [];
  for (const cookie of setCookies) {
    responseHeaders.append("set-cookie", cookie);
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
export const OPTIONS = handle;
