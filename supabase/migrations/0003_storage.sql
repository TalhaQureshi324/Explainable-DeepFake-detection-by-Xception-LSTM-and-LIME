-- Phase 4C: private storage buckets + retention defaults.
-- Run once in the Supabase SQL editor (storage schema is managed by Supabase).

insert into storage.buckets (id, name, public, file_size_limit)
values
  ('analysis-uploads',   'analysis-uploads',   false, 2147483648)  -- 2 GB raw media
  on conflict (id) do nothing;
insert into storage.buckets (id, name, public, file_size_limit)
values
  ('analysis-artifacts', 'analysis-artifacts', false, 26214400)   -- 25 MB previews/heatmaps
  on conflict (id) do nothing;
insert into storage.buckets (id, name, public, file_size_limit)
values
  ('analysis-reports',   'analysis-reports',   false, 26214400)   -- 25 MB PDFs
  on conflict (id) do nothing;

-- Object-level RLS: owner-scoped paths ("<uid>/..."), service key bypasses.
drop policy if exists uploads_owner_rw on storage.objects;
create policy uploads_owner_rw on storage.objects
  for all to authenticated
  using (bucket_id = 'analysis-uploads' and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'analysis-uploads' and (storage.foldername(name))[1] = auth.uid()::text);

drop policy if exists artifacts_owner_read on storage.objects;
create policy artifacts_owner_read on storage.objects
  for select to authenticated
  using (bucket_id in ('analysis-artifacts', 'analysis-reports')
         and (storage.foldername(name))[1] = auth.uid()::text);
