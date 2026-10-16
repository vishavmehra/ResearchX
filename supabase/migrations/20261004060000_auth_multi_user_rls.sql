-- Multi-user isolation (2026-10-04; docs/SECURITY.md). Supabase Auth users own their conversations, messages, research
-- runs (with stage traces), answer-cache rows and private documents. The backend verifies every access token with
-- Supabase Auth and runs each user-owned query as role `authenticated` with the verified JWT claims (SET LOCAL ROLE +
-- request.jwt.claims), so these policies apply even though the backend connects as postgres (BYPASSRLS). Schema rx is
-- not exposed through the Data API; if it ever is, the same policies protect it. Additive only: nothing is dropped.
-- Shared corpus tables (papers, chunks, evidence, ...) stay backend-only and are never user-owned.

-- 1. profiles: one row per auth user; credentials stay in Supabase Auth
create table rx.profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  display_name text check (char_length(display_name) <= 80),
  avatar_url   text check (avatar_url is null or avatar_url ~ '^https://'),
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

-- 2. ownership: user-owned rows carry the auth user id; deleting the auth user cascades to all of them
alter table rx.conversations alter column user_id type uuid using user_id::uuid;
alter table rx.conversations alter column user_id set default auth.uid();
alter table rx.conversations add constraint conversations_user_fk foreign key (user_id) references auth.users(id) on delete cascade not valid;
alter table rx.messages add column user_id uuid default auth.uid() references auth.users(id) on delete cascade,
                        add column metadata jsonb not null default '{}';
alter table rx.research_runs add column user_id uuid default auth.uid() references auth.users(id) on delete cascade;
alter table rx.cache_entries add column user_id uuid references auth.users(id) on delete cascade;

-- 3. private documents: tenant-owned, never part of the shared corpus (visibility is private only; no promotion path)
create table rx.user_documents (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null default auth.uid() references auth.users(id) on delete cascade,
  title          text not null check (char_length(title) between 1 and 200),
  media_type     text not null,
  bytes          int  not null check (bytes > 0),
  content_sha256 text not null,
  storage_path   text,
  visibility     text not null default 'private' check (visibility = 'private'),
  status         text not null default 'PROCESSING' check (status in ('PROCESSING','READY','FAILED')),
  chunks         int  not null default 0,
  error          text,
  created_at     timestamptz not null default now(),
  unique (user_id, content_sha256)
);
create table rx.user_chunks (
  id          bigserial primary key,
  document_id uuid not null references rx.user_documents(id) on delete cascade,
  user_id     uuid not null default auth.uid() references auth.users(id) on delete cascade,
  ordinal     int  not null,
  text        text not null,
  embedding   extensions.halfvec(256) not null,
  unique (document_id, ordinal)
);
alter table rx.profiles enable row level security;
alter table rx.user_documents enable row level security;
alter table rx.user_chunks enable row level security;

-- 4. indexes for per-user listing and pagination
create index on rx.conversations (user_id, updated_at desc);
create index on rx.messages (user_id);
create index on rx.research_runs (user_id, started_at desc);
create index on rx.research_runs (conversation_id, started_at);
create index on rx.cache_entries (user_id);
create index on rx.user_documents (user_id, created_at desc);
create index on rx.user_chunks (user_id);

-- 5. privileges for the authenticated role (rows are then filtered by the policies below)
grant usage on schema rx to authenticated;
grant select, insert, update, delete on rx.profiles, rx.conversations, rx.messages, rx.research_runs, rx.run_events,
  rx.retrieval_traces, rx.cache_entries, rx.user_documents, rx.user_chunks to authenticated;
grant usage, select on sequence rx.run_events_id_seq, rx.retrieval_traces_id_seq, rx.user_chunks_id_seq to authenticated;
grant select on rx.architecture_metrics, rx.run_summary to authenticated;  -- security_invoker views: own rows only

-- 6. ownership policies: SELECT / INSERT / UPDATE / DELETE only on the caller's own rows (auth.uid())
create policy profiles_own on rx.profiles for all to authenticated
  using (id = (select auth.uid())) with check (id = (select auth.uid()));
create policy conversations_own on rx.conversations for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy messages_own on rx.messages for all to authenticated
  using (user_id = (select auth.uid()) and exists (select 1 from rx.conversations c where c.id = conversation_id and c.user_id = (select auth.uid())))
  with check (user_id = (select auth.uid()) and exists (select 1 from rx.conversations c where c.id = conversation_id and c.user_id = (select auth.uid())));
create policy research_runs_own on rx.research_runs for all to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()) and (conversation_id is null or
              exists (select 1 from rx.conversations c where c.id = conversation_id and c.user_id = (select auth.uid()))));
create policy run_events_own on rx.run_events for all to authenticated
  using (exists (select 1 from rx.research_runs r where r.id = run_id and r.user_id = (select auth.uid())))
  with check (exists (select 1 from rx.research_runs r where r.id = run_id and r.user_id = (select auth.uid())));
create policy retrieval_traces_own on rx.retrieval_traces for all to authenticated
  using (exists (select 1 from rx.research_runs r where r.id = run_id and r.user_id = (select auth.uid())))
  with check (exists (select 1 from rx.research_runs r where r.id = run_id and r.user_id = (select auth.uid())));
create policy cache_entries_own on rx.cache_entries for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy user_documents_own on rx.user_documents for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy user_chunks_own on rx.user_chunks for all to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()) and exists (select 1 from rx.user_documents d where d.id = document_id and d.user_id = (select auth.uid())));

-- 7. MFA: once a user has a verified factor, every user-owned row requires an aal2 session (restrictive = AND-ed)
create function rx.has_verified_mfa() returns boolean language sql stable security definer set search_path = '' as $$
  select exists (select 1 from auth.mfa_factors where user_id = (select auth.uid()) and status = 'verified')
$$;
revoke execute on function rx.has_verified_mfa() from public, anon;
grant execute on function rx.has_verified_mfa() to authenticated;
do $$
declare
  t text;
  cond text := '((select auth.jwt() ->> ''aal'') = ''aal2'' or not (select rx.has_verified_mfa()))';
begin
  foreach t in array array['profiles','conversations','messages','research_runs','run_events','retrieval_traces','cache_entries',
                           'user_documents','user_chunks'] loop
    execute format('create policy %I on rx.%I as restrictive for all to authenticated using %s with check %s', t || '_mfa_aal2', t, cond, cond);
  end loop;
end $$;

comment on table rx.profiles is 'User profile (no credentials). Owner: the auth user; removed with the account.';
comment on table rx.user_documents is 'Private user documents. Never indexed into the shared corpus; retrieval filters by owner.';
comment on column rx.cache_entries.user_id is 'Answer cache scope: rows are private to this user (no cross-user cache hits).';
