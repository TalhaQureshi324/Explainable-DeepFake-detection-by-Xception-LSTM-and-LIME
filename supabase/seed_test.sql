-- Seed / test fixtures: local development and CI only.
-- Creates one demo model registry row. For local-mode auth testing, register
-- users through the app itself (backend /v1/auth/register) — Supabase-managed
-- auth users cannot be seeded from plain SQL without the auth schema's
-- crypt() helpers, and we deliberately do not touch auth.users here.

insert into model_versions (version, name, checkpoint_sha256, input_frames, input_size, preprocessing_version, notes)
values
  ('v2', 'Xception-LSTM', '9ee14530166b842ce0aa6ac74500ac138993f5fbe82a927df7ddee0ffc261228', 32, 299, 'dnn-ellipse-01',
   'Preferred 32-frame checkpoint; sliding-window ensemble at inference. Score = uncalibrated P(FAKE). Threshold dev default 0.60 (dev-v1).')
  on conflict (version) do nothing;

insert into model_versions (version, name, checkpoint_sha256, input_frames, input_size, preprocessing_version, notes)
values
  ('v1', 'Xception-LSTM', '3c101cbf3baeda47d223370f72421a157d775b0046388f2fdfa08f8642c1dcab', 10, 299, 'dnn-ellipse-01',
   'Fallback 10-frame checkpoint; full-sequence scoring.')
  on conflict (version) do nothing;
