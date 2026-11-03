-- Phase 4B: Row Level Security — users can only touch their own rows.
-- The Python backend uses the service-role key server-side (bypasses RLS for
-- worker bookkeeping); every user-facing path verifies ownership explicitly.

alter table analyses enable row level security;
alter table analysis_windows enable row level security;
alter table analysis_artifacts enable row level security;
alter table analysis_jobs enable row level security;
alter table model_versions enable row level security;

drop policy if exists analyses_owner_all on analyses;
create policy analyses_owner_all on analyses
  for all using (user_id = auth.uid()) with check (user_id = auth.uid());

drop policy if exists analysis_windows_owner_all on analysis_windows;
create policy analysis_windows_owner_all on analysis_windows
  for all using (
    exists (select 1 from analyses a
            where a.id = analysis_windows.analysis_id and a.user_id = auth.uid())
  ) with check (
    exists (select 1 from analyses a
            where a.id = analysis_windows.analysis_id and a.user_id = auth.uid())
  );

drop policy if exists analysis_artifacts_owner_all on analysis_artifacts;
create policy analysis_artifacts_owner_all on analysis_artifacts
  for all using (
    exists (select 1 from analyses a
            where a.id = analysis_artifacts.analysis_id and a.user_id = auth.uid())
  ) with check (
    exists (select 1 from analyses a
            where a.id = analysis_artifacts.analysis_id and a.user_id = auth.uid())
  );

drop policy if exists analysis_jobs_owner_read on analysis_jobs;
create policy analysis_jobs_owner_read on analysis_jobs
  for select using (
    exists (select 1 from analyses a
            where a.id = analysis_jobs.analysis_id and a.user_id = auth.uid())
  );

drop policy if exists model_versions_read_all on model_versions;
create policy model_versions_read_all on model_versions
  for select using (true);
