-- Phase 4D: atomic job claiming for multiple/competing workers.
-- PostgREST alone cannot do compare-and-swap updates, so claiming runs here
-- with FOR UPDATE SKIP LOCKED (safe under concurrent workers).

create or replace function claim_job(p_worker_id text, p_lease_seconds int default 300)
returns table (job_id uuid, analysis_id uuid, attempts int, max_attempts int)
language plpgsql
as $$
declare
  v_job record;
begin
  update analysis_jobs
    set state = 'RUNNING',
        worker_id = p_worker_id,
        lease_until = now() + make_interval(secs => p_lease_seconds),
        attempts = attempts + 1,
        updated_at = now()
    where id = (
      select id from analysis_jobs
        where state = 'QUEUED' and run_after <= now() and attempts < max_attempts
        order by run_after
        for update skip locked
        limit 1
    )
    returning id, analysis_id, attempts, max_attempts into v_job;
  if v_job is null then
    -- dead-letter exhausted queued jobs alongside claiming
    update analysis_jobs set state = 'DEAD', updated_at = now()
      where state = 'QUEUED' and attempts >= max_attempts;
    return;
  end if;
  return query select v_job.id, v_job.analysis_id, v_job.attempts, v_job.max_attempts;
end;
$$;

create or replace function reclaim_stalled_jobs()
returns int
language plpgsql
as $$
declare
  v_count int;
begin
  update analysis_jobs
    set state = 'QUEUED', worker_id = null, lease_until = null, updated_at = now()
    where state = 'RUNNING'
      and (lease_until is null or lease_until < now() - interval '60 seconds');
  get diagnostics v_count = row_count;
  return v_count;
end;
$$;
