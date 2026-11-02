# Frontend — DeepFake Forensics (Phase 2)

Next.js 14 (App Router) + TypeScript + Tailwind CSS + Recharts + Lucide.
Talks to the FastAPI inference backend (`backend/`) over a same-origin proxy.

## Quick start (local end-to-end)

```bash
# 1. Backend (repo root) — put checkpoints in models_store/ or set DEEPFAKE_MODELS_DIR
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 2. Frontend (frontend/) — the proxy target is baked at BUILD time
cd frontend
npm install
INFERENCE_API_URL=http://127.0.0.1:8000 npm run build   # proxy target for prod server
npm run dev                                             # http://localhost:3000
```

> **Important:** Next.js bakes `rewrites()` into the build manifest, so
> `INFERENCE_API_URL` must be set when BUILDING (`npm run build`), not just at
> runtime. On Vercel, set `INFERENCE_API_URL` as a project environment
> variable and redeploy — same mechanism, zero code changes.

All frontend HTTP calls go to relative `/api/inference/:path*` routes which the
config proxies to the backend, so there is no CORS and the deployed backend URL
never reaches the browser.

## Structure

```
frontend/
├── app/
│   ├── page.tsx                    # Homepage: hero, disclaimer, architecture
│   ├── analyze/page.tsx            # Upload: dropzone, validation, preview, model select
│   ├── analysis/[id]/page.tsx      # Polling + live states + full results
│   └── analysis/[id]/report/page.tsx  # Report preview (Phase 3 groundwork)
├── components/
│   ├── ui/primitives.tsx           # shadcn-style local primitives
│   ├── site/chrome.tsx             # header (language/theme switches) + footer
│   └── analysis/                   # status timeline, result view, windows chart
└── lib/
    ├── api/types.ts                # mirrors backend Pydantic schemas
    ├── api/client.ts               # the ONLY fetch layer + polling
    ├── api/use-analysis.ts         # polling hook + error-code mapping
    ├── i18n/                       # EN/UR dictionaries + RTL provider
    └── theme/provider.tsx          # light/dark provider (class strategy)
```

## Behavior guarantees

- Only real backend states are rendered — no simulated progress percentages.
- Scores are labelled **uncalibrated** everywhere they appear.
- Image analyses display the replicated-frame compatibility warning.
- `?lang=ur` / `?theme=dark` URL params override the persisted choice (handy
  for demos and screenshots); otherwise choices persist in localStorage.
- Backend restarts make Phase 1 in-memory jobs disappear; the analysis page
  surfaces the 404 as an explanatory state instead of an error blob.
- PDF button on the report page is intentionally disabled until Phase 3.

## Testing

```bash
npm run build     # type + lint + production build (must pass)
npm run lint
```

Backend contract tests: `python -m pytest backend/tests` (34 tests).
End-to-end evidence with real media: `docs/E2E_EVIDENCE_PHASE2.md`.
