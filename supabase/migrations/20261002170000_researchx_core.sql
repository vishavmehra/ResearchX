-- ResearchX core schema (M2, 2026-10-02). Supabase PostgreSQL is the runtime system of record
-- (docs/ARCHITECTURE.md §50). Frozen dataset records are stored verbatim in `record jsonb` (never altered);
-- a few typed columns are extracted for joins and filters. Every row carries the dataset release it came from.
-- RLS is enabled on every table with no policies: only the backend (postgres / service role) reads or writes;
-- the browser never talks to these tables directly.

create extension if not exists vector with schema extensions;
create extension if not exists pg_trgm with schema extensions;

create schema if not exists rx;

-- ---------------------------------------------------------------- dataset releases + load audit
create table rx.dataset_releases (
  version            text primary key,
  manifest_sha256    text not null,
  processing_commit  text not null,
  schema_version     text not null,
  manifest           jsonb not null,
  loaded_at          timestamptz not null default now(),
  load_status        text not null default 'LOADING' check (load_status in ('LOADING','VERIFIED','FAILED'))
);

create table rx.load_checks (
  release      text not null references rx.dataset_releases(version),
  source_file  text not null,
  file_sha256  text not null,
  manifest_records integer,
  loaded_records   integer not null,
  ok           boolean not null,
  checked_at   timestamptz not null default now(),
  primary key (release, source_file)
);

-- ---------------------------------------------------------------- frozen records (verbatim jsonb)
create table rx.source_records (id text primary key, release text not null, source_name text, record jsonb not null);
create table rx.papers        (id text primary key, release text not null, study_id text, arxiv_id text, doi text, title text,
                               publication_year int, is_relevant boolean, split text, record jsonb not null);
create table rx.studies       (id text primary key, release text not null, record jsonb not null);
create table rx.citations     (citing_paper_id text not null, cited_paper_id text not null, release text not null, record jsonb not null,
                               primary key (citing_paper_id, cited_paper_id));
create table rx.experiments   (id text primary key, release text not null, paper_id text not null, study_id text, experiment_type text,
                               depth_tier text, split text, record jsonb not null);
create table rx.evidence      (id text primary key, release text not null, paper_id text, experiment_id text, field_name text not null,
                               status text, location_kind text, source_file text not null, record jsonb not null);
create table rx.results       (id text primary key, release text not null, paper_id text, record jsonb not null);
create table rx.associations  (id text primary key, release text not null, result_id text not null, experiment_id text, status text not null,
                               record jsonb not null);
create table rx.artifacts     (id text primary key, release text not null, artifact_type text, repository text, record jsonb not null);
create table rx.artifact_relations (id text primary key, release text not null, paper_id text, experiment_id text, field_name text,
                               relation text, record jsonb not null);
create table rx.comparisons   (id text primary key, release text not null, experiment_a text, experiment_b text, label text,
                               is_hard_negative boolean, source_file text not null, record jsonb not null);
create table rx.study_links   (id text primary key, release text not null, record_a text, record_b text, relationship text, record jsonb not null);
create table rx.contradictions (id text primary key, release text not null, subject_id text, field text, record jsonb not null);
create table rx.abstention_cases (id text primary key, release text not null, expected_behavior text, record jsonb not null);
create table rx.research_tasks (id text primary key, release text not null, task_type text, expected_behavior text, record jsonb not null);
create table rx.hard_cases    (id text primary key, release text not null, case_type text, record jsonb not null);
create table rx.canonical_entities (id text primary key, release text not null, entity_type text, normalized_name text, preferred_name text,
                               record jsonb not null);
create table rx.graph_nodes   (id text primary key, release text not null, node_type text, ref_id text, record jsonb not null);
create table rx.graph_edges   (id text primary key, release text not null, edge_type text, source text not null, target text not null,
                               record jsonb not null);
create table rx.other_records (id text not null, release text not null, record_type text not null, source_file text not null,
                               record jsonb not null, primary key (record_type, id));

create index on rx.experiments (paper_id);
create index on rx.evidence (experiment_id);
create index on rx.evidence (paper_id, field_name);
create index on rx.associations (experiment_id);
create index on rx.associations (result_id);
create index on rx.artifact_relations (paper_id);
create index on rx.comparisons (experiment_a);
create index on rx.comparisons (experiment_b);
create index on rx.graph_edges (source);
create index on rx.graph_edges (target);
create index on rx.citations (cited_paper_id);
create index on rx.canonical_entities (entity_type, normalized_name);

-- ---------------------------------------------------------------- retrieval units + embeddings
-- chunks are the stable retrieval units of parsed documents (paragraph / table / caption / footnote) plus
-- paper-level title+abstract units; unit_id equals the evidence location id inside the paper.
create table rx.chunks (
  id            text primary key,                 -- '<paper_id>#<unit_id>'
  release       text not null,
  paper_id      text not null,
  unit_type     text not null check (unit_type in ('paper','paragraph','table','caption','footnote')),
  unit_id       text not null,
  section_title text,
  text          text not null,
  token_estimate int not null,
  source_record_id text,
  source_sha256 text,
  content_license text,
  redistribution_allowed boolean,
  tsv           tsvector generated always as (to_tsvector('english', coalesce(section_title,'') || ' ' || text)) stored
);
create index on rx.chunks (paper_id);
create index chunks_tsv_idx on rx.chunks using gin (tsv);

-- embeddings live apart from chunks so the model / dimension can change without touching the corpus.
create table rx.chunk_embeddings (
  chunk_id     text not null references rx.chunks(id) on delete cascade,
  model        text not null,
  dims         int not null,
  embedding    extensions.halfvec not null,       -- dimension fixed per model; HNSW index created per model after benchmarking
  created_at   timestamptz not null default now(),
  primary key (chunk_id, model)
);

-- ---------------------------------------------------------------- runtime: conversations, runs, traces
create table rx.conversations (
  id          uuid primary key default gen_random_uuid(),
  user_id     text not null,
  title       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create table rx.messages (
  id              uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references rx.conversations(id) on delete cascade,
  role            text not null check (role in ('user','assistant','system','tool')),
  content         text not null,
  input_modality  text not null default 'text' check (input_modality in ('text','voice')),
  research_run_id uuid,
  created_at      timestamptz not null default now()
);
create index on rx.messages (conversation_id, created_at);

create table rx.research_runs (
  id               uuid primary key default gen_random_uuid(),
  conversation_id  uuid references rx.conversations(id) on delete set null,
  query            text not null,
  status           text not null default 'RUNNING' check (status in ('RUNNING','ANSWERED','PARTIAL','CONFLICTING','ABSTAINED','FAILED')),
  dataset_release  text,
  routing          jsonb not null default '{}',
  answer           jsonb,
  usage            jsonb not null default '{}',
  error            text,
  started_at       timestamptz not null default now(),
  finished_at      timestamptz
);
create table rx.run_events (
  id          bigserial primary key,
  run_id      uuid not null references rx.research_runs(id) on delete cascade,
  step        text not null,
  detail      jsonb not null,
  latency_ms  int,
  created_at  timestamptz not null default now()
);
create index on rx.run_events (run_id, id);

create table rx.retrieval_traces (
  id            bigserial primary key,
  run_id        uuid references rx.research_runs(id) on delete cascade,
  query         text not null,
  method        text not null check (method in ('bm25','vector','hybrid','hybrid_rerank','structured')),
  config        jsonb not null,
  candidates    jsonb not null,
  latency_ms    int,
  created_at    timestamptz not null default now()
);

-- runtime sources: RUNTIME (one answer) -> CANDIDATE (validated) -> PROMOTED (ingested in a release)
create table rx.runtime_sources (
  id                uuid primary key default gen_random_uuid(),
  url               text not null,
  provider          text not null,
  source_type       text not null default 'unknown',
  title             text,
  authors           jsonb,
  doi               text,
  arxiv_id          text,
  published_date    date,
  retrieved_at      timestamptz not null default now(),
  content_sha256    text,
  license           text,
  reliability       jsonb not null default '{}',
  state             text not null default 'RUNTIME' check (state in ('RUNTIME','CANDIDATE','PROMOTED','REJECTED')),
  validation        jsonb not null default '{}',
  originating_run_id uuid references rx.research_runs(id) on delete set null,
  promoted_release  text,
  unique (url, content_sha256)
);

create table rx.eval_runs (
  id            uuid primary key default gen_random_uuid(),
  suite         text not null,
  config        jsonb not null,
  dataset_release text not null,
  metrics       jsonb,
  created_at    timestamptz not null default now()
);
create table rx.eval_results (
  id         bigserial primary key,
  eval_run_id uuid not null references rx.eval_runs(id) on delete cascade,
  item_id    text not null,
  result     jsonb not null
);

create table rx.cache_entries (
  key           text primary key,
  kind          text not null check (kind in ('retrieval','api','semantic','model')),
  dataset_release text,
  model         text,
  value         jsonb not null,
  query_embedding extensions.halfvec,
  created_at    timestamptz not null default now(),
  expires_at    timestamptz
);

-- ---------------------------------------------------------------- lock down: backend-only access
do $$
declare t record;
begin
  for t in select tablename from pg_tables where schemaname = 'rx' loop
    execute format('alter table rx.%I enable row level security', t.tablename);
  end loop;
end $$;
revoke all on schema rx from anon, authenticated;
