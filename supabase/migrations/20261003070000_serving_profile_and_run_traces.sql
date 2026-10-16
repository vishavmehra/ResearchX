-- Serving profile + provable run traces (2026-10-03). Additive only: no table, column or row is removed.
--
-- 1. Free-plan quota (500 MB): frozen records are stored losslessly compact (services/storage/records.py). Provenance
--    blocks are deduplicated into rx.provenance and file-constant keys are kept once in rx.load_checks.constants.
--    load_checks.status marks files kept only in the frozen release (ARCHIVE_ONLY; still sha256 + count checked).
-- 2. Provable execution: every research request gets research_run_id = 'rx_' + ULID and every pipeline stage writes a
--    run_events row with its REAL status. The UI renders traces only from these rows. A model-only answer is
--    mode = 'fallback' and must carry fallback_reason, so it can never look like a completed pipeline run.

create table rx.provenance (id text primary key, provenance jsonb not null);
alter table rx.provenance enable row level security;

alter table rx.load_checks
  add column status    text  not null default 'LOADED' check (status in ('LOADED','ARCHIVE_ONLY')),
  add column constants jsonb not null default '{}';

do $$
declare t text;
begin
  foreach t in array array['source_records','papers','studies','citations','experiments','evidence','results','associations',
                           'artifacts','artifact_relations','comparisons','study_links','contradictions','abstention_cases',
                           'research_tasks','hard_cases','canonical_entities','graph_nodes','graph_edges','other_records'] loop
    execute format('alter table rx.%I add column provenance_id text references rx.provenance(id)', t);
    execute format('comment on column rx.%I.record is %L', t,
                   'lossless compact record: services/storage/records.py expand() rebuilds the frozen record exactly');
  end loop;
end $$;

alter table rx.research_runs
  add column research_run_id  text not null unique check (research_run_id ~ '^rx_[0-9A-HJKMNP-TV-Z]{26}$'),
  add column normalized_query text,
  add column mode             text not null default 'pipeline' check (mode in ('pipeline','fallback')),
  add column fallback_reason  text,
  add constraint research_runs_fallback_needs_reason check (mode = 'pipeline' or fallback_reason is not null);

alter table rx.run_events
  add column status      text not null check (status in ('running','completed','skipped','not_required','failed')),
  add column error       text,
  add column finished_at timestamptz,
  add constraint run_events_step_known check (step in (
    'query_understanding','laya_decision','research_strategy','model_routing','cache_lookup','bm25','pgvector',
    'hybrid_fusion','reranking','evidence_assessment','evidence_gap','tavily','external_sources','source_validation',
    'evidence_extraction','experiment_extraction','contradiction_detection','evidence_selection','synthesis',
    'citation_validation','abstention','fallback','answer'));
create index on rx.run_events (step, status);
create index on rx.retrieval_traces (run_id);
alter table rx.messages add constraint messages_research_run_fk
  foreign key (research_run_id) references rx.research_runs(id) on delete set null;

create view rx.architecture_metrics with (security_invoker = true) as
select step as stage, status, count(*) as invocations, round(avg(latency_ms)) as avg_latency_ms,
       percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms
from rx.run_events group by step, status;

create view rx.run_summary with (security_invoker = true) as
select mode, status, count(*) as runs,
       round(avg(extract(epoch from (finished_at - started_at)) * 1000)) as avg_total_ms
from rx.research_runs group by mode, status;
