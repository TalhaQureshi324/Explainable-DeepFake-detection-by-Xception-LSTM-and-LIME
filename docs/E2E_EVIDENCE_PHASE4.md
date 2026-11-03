# Phase 4 End-to-End Verification Evidence

Date: 2026-11-04 · Branch: `phase-4-supabase` · Persistence mode: **local SQLite**
(identical interface to the Supabase backend — see "Supabase activation" below).
All tests ran through the Next.js production proxy against the real FastAPI
backend and the real V2 checkpoint.

## Verification matrix (master-plan list → result)

| # | Requirement | Result |
|---|-------------|--------|
| 1 | User registration + authentication | ✅ register 201 (pbkdf2-hashed), duplicate 409, wrong password 401, login 200 + httpOnly cookie |
| 2 | Image analysis with persistence | ✅ (deletion test path) COMPLETED as owner, retrievable from repository |
| 3 | Video analysis with persistence | ✅ submitted as user A → COMPLETED → REAL 0.325892 stored |
| 4 | Report after FastAPI restart | ✅ backend process restarted (store wiped) — report served from SQLite |
| 5 | PDF after restart | ✅ `application/pdf`, 150 KB, values consistent |
| 6 | History after refresh | ✅ server-side fetch (no client state) — history/dashboard pages |
| 7 | Filtering + pagination | ✅ label/media_type/status/search filters; `limit=1` → cursor → page 2 |
| 8 | Deletion + artifacts | ✅ owner DELETE → 200; subsequent GET 404; local caches + uploaded media removed |
| 9 | Cross-user isolation | ✅ user B gets 404 on A's analysis/report/PDF; B's history empty |
| 10 | Storage link expiry/auth | ✅ local mode serves artifacts server-side by validated id; Supabase mode uses signed URLs (code + policy in 0003_storage.sql; live check once credentials exist) |
| 11 | Failed upload cleanup | ✅ sweeper purges upload dirs of non-active analyses after 24 h grace (`AnalysisWorker.sweep_now`) |
| 12 | Interrupted worker recovery | ✅ backend hard-killed mid-INFERENCING → lease expired → periodic reclaim re-queued → re-ran → COMPLETED, same score, media deleted after success |
| 13 | No duplicate records | ✅ enqueue uses `insert or ignore` / `resolution=ignore-duplicates`; unique(analysis_id) on jobs |
| 14 | Inference prediction regression | ✅ 0.325892 on the reference video — identical across Phases 1–4 |
| 15 | PDF value consistency | ✅ PDF renders from the persisted report document; values match API responses |
| 16 | Next.js production build | ✅ green (typed routes, ESLint) |
| 17 | EN/UR + dark/light + mobile workflows | ✅ unchanged and re-captured (`history_urdu_dark.png`, mobile nav row) |

## Defects found by verification and fixed

1. **Stalled-job reclaim only ran at startup** — a worker that died within its
   lease left the job `RUNNING` forever. Reclaim now also runs periodically in
   the dispatcher; plus exhausted jobs are dead-lettered (`DEAD`) at claim time.
2. **Rebuilt records were not re-registered in the hot store** after a restart —
   the re-run crashed reading warnings from a missing store entry. `_load_job`
   now re-registers, restoring both status polling and processing updates.
3. **SQLite self-deadlock** — `update_analysis` held the non-reentrant lock
   while nesting `_replace_windows`/`add_artifact` acquisitions. Fixed with an
   `RLock`; the interrupted job then recovered fully.
4. **Contract hardening** — `AnalysisResponse` now tolerates NULL
   model/prediction fields for analyses mid-rerun (additive-compatible).
5. Guest/persisted access split: unauthenticated access to saved analyses
   returns 401 (no existence disclosure) — tests updated accordingly.

## Screenshots (docs/screenshots/)

- `login_page.png`, `register_page_dark.png` — auth screens (light/dark)
- `dashboard_page.png` — signed-in dashboard with genuinely calculated stats
  (Total 2 · Completed 1 · Videos 2 · REAL 2) and recent analyses
- `history_page.png`, `history_urdu_dark.png` — history list + Urdu/RTL dark

## Supabase activation (when the project exists)

1. Run `supabase/migrations/0001_init.sql` → `0004_job_rpc.sql`,
   `supabase/seed_test.sql`, then `0003_storage.sql` (buckets) in the SQL editor.
2. Set `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (backend) and `SUPABASE_ANON_KEY`
   (frontend build env). The repository/storage/auth implementations switch
   automatically; no code changes. RLS + storage policies in 0002/0003 enforce
   the same ownership rules the local mode enforces in code.
3. Live Supabase-specific checks (signed-URL expiry, GoTrue email flows) are the
   remaining items to run once credentials exist — every other behavior above is
   mode-independent and already verified.

## Known limitations

- Local auth mode is for development/demo: pbkdf2-hashed users in SQLite, no
  email verification/magic links (Supabase mode provides those via GoTrue).
- Job retries are per-backend-process workers (bounded thread pool); horizontal
  worker scaling works in Supabase mode via the RPC claim but is untested here.
- PDF/artifact retention defaults: artifacts + PDFs kept until user deletion;
  `RETENTION_DAYS` (optional) prunes analyses older than the horizon.
