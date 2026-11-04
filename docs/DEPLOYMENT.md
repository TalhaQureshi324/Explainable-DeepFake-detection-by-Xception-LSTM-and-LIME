# Deployment — Vercel Services configuration

Status: configuration committed and CLI-validated; **deployment architecture
pending your confirmation** (see checklist at the end).

## What ships on Vercel

`vercel.json` (repository root) declares **one service**:

| Service | Root | Framework | Public route | Purpose |
|---------|------|-----------|--------------|---------|
| `frontend` | `frontend/` | `nextjs` | `/` (catch-all rewrite) | Next.js app: UI, analysis workflow, reports, runtime API proxy |

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "services": {
    "frontend": {
      "root": "frontend/",
      "framework": "nextjs",
      "buildCommand": "npm run build"
    }
  },
  "rewrites": [
    { "source": "/(.*)", "destination": { "service": "frontend" } }
  ]
}
```

There is deliberately **no `backend` service**: per the feasibility assessment
below, the TensorFlow inference workload does not run reliably on Vercel, and
adding a backend service that cannot perform inference would be exactly the
"nonfunctional service to satisfy the example configuration" the task rules
forbid. When a backend service is ever added, expose nothing publicly and
declare the binding on `frontend`:

```json
"frontend": {
  "bindings": [
    { "type": "service", "service": "backend", "format": "url", "env": "INFERENCE_API_URL" }
  ]
}
```

## Runtime environment (was build-time — changed in this phase)

`frontend/app/api/inference/[...path]/route.ts` is a **runtime proxy route
handler** replacing the old build-time rewrite. `INFERENCE_API_URL` is read on
every request, so it works with:

- local dev: `.env.local` or the default `http://127.0.0.1:8000`
- Vercel project environment variable (set to the externally hosted backend URL)
- a future Vercel service binding (runtime-injected)

This was verified live: the same build served requests to different backends
depending on the server-process environment.

Platform caveat carried over from the architecture plan: **serverless request
bodies are limited (~4.5 MB)**. Everything except large media uploads works
through the proxy (auth, history, reports, PDFs, small images). Large video
uploads from a deployed frontend should go direct to the backend origin (enable
`CORS_ALLOW_ORIGINS` on the backend) or — better — direct-to-Supabase Storage
with the signed-upload flow planned for Phase 4C completion.

## Why the TensorFlow backend is not deployed on Vercel

| Constraint | Our backend | Vercel reality |
|-----------|-------------|----------------|
| Deployment size | TF wheel + 2 × 166 MB checkpoints ≫ 250 MB function limit | Serverless functions cap at 250 MB uncompressed; containers allow more but… |
| Inference duration | 10–60 s per video CPU; LIME adds minutes | Fluid default 300 s / max 800 s — long jobs at risk |
| Upload size | Up to 200 MB videos | 4.5 MB request body limit on serverless |
| Background workers | Durable queue dispatcher + lease renewal + retention sweeper are long-lived threads | Functions are request-scoped; continuous workers need Daemons/containers + Supabase mode (SQLite is ephemeral on serverless) |
| Cold starts | 166 MB model load per cold instance | Multi-hundred-MB cold starts per request burst |

**Recommended production architecture** (unchanged from Phase 1 planning):

1. **Next.js frontend on Vercel** — this `vercel.json`.
2. **FastAPI inference backend on dedicated Python compute** — Hugging Face
   Space container, a small VM, or any container host running the same
   `uvicorn` app with the checkpoints on disk.
3. **Supabase** — persistence, auth, storage (Phase 4 schema/RLS ready).

## Validation performed

- `vercel.json` parses as valid JSON; service + catch-all rewrite structure
  match the documented Services schema.
- **Vercel CLI 65.11.0 detected the service**: `vercel dev -L` printed
  "Detected services: • frontend [Next.js]" (schema accepted by the real CLI).
  The full local dev run then failed inside the CLI's builder cache with
  `EPERM: symlink` — a known Windows permission limitation (enable Developer
  Mode or run on macOS/Linux); not a configuration error.
- `npm run build` green **without any build-time env** (runtime resolution).
- Live proxy verification: health, auth cookie flow (register/login through
  the route handler, protected history accessible), PDF binary passthrough —
  all through the new handler with `INFERENCE_API_URL` set on the server
  process only.
- `python -m pytest backend/tests` → 42 passed (incl. opt-in CORS change).

## Deployment steps (after confirmation)

1. Import the repository into Vercel (Services beta enabled).
2. Set project env: `INFERENCE_API_URL=https://<hosted-backend>` (runtime).
3. Deploy; the `frontend` service serves at the project domain.
4. Point the hosted backend's `CORS_ALLOW_ORIGINS` at the Vercel domain if
   direct uploads are enabled.

## Confirmation checklist (reply to approve/adjust)

- [ ] Service name `frontend` only (backend deliberately excluded — confirm or
      request a container-runtime backend service for later).
- [ ] Public route: `/` → `frontend` (catch-all; API proxy lives inside it).
- [ ] `INFERENCE_API_URL` as a **project environment variable** now, with the
      service-binding declaration reserved for when a `backend` service exists.
