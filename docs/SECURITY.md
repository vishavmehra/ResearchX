# ResearchX — Security

Scope: the runtime product (Next.js app in `apps/web`, FastAPI service in `services/api`, Supabase Auth, PostgreSQL
with RLS, Supabase Storage). The dataset pipeline's secret rules are in `planning/DECISIONS.md` §34. Architecture
overview: `docs/ARCHITECTURE.md` §60 (prompt injection) and §62 (multi-user isolation).

## 1. Principles

1. Identity comes from Supabase Auth only. The backend derives the user from a verified access token; a `user_id`
   in a request body or query is never trusted (unknown fields are rejected).
2. Authorization is enforced in PostgreSQL with row-level security, not only in application code or the UI.
3. Secrets stay on the server. Only `NEXT_PUBLIC_*` values (project URL, publishable key) reach the browser.
4. Every research run is attributable: run id, owner, stages and sources are persisted, and only the owner can
   read them.

## 2. Authentication

| Flow | Implementation |
|---|---|
| Sign-up, e-mail verification | `supabase.auth.signUp` with a redirect to `/auth/callback`, which exchanges the code for a session |
| Sign-in, sign-out | password sign-in in the browser; sessions are `@supabase/ssr` cookies; sign-out ends the session |
| Password reset | `resetPasswordForEmail` link to `/auth/callback?next=/reset-password`, then `updateUser` |
| Google / GitHub sign-in (OAuth, 2026-10-08) | `signInWithOAuth` with PKCE; the provider returns to `/auth/callback?flow=oauth`, which exchanges the code for a session (MFA users then go to `/mfa`). Buttons appear only for providers switched on in Supabase Auth (read from the public `/auth/v1/settings`). Supabase links a provider identity to an existing account automatically only when both carry the same verified email |
| Connected accounts | Settings → Security lists every identity; `linkIdentity` adds a provider (`flow=link`), `unlinkIdentity` removes one (never the last). An account without a password confirms account deletion by signing in again with its provider (`flow=reauth`) |
| Callback errors | provider errors (`?error=&error_code=`) and failed code exchanges go back to the sign-in page, or to Settings for linking, with a fixed message; the destination is a same-site path only |
| Session check on pages | Next.js `proxy.ts` calls `getUser()` (validated by Supabase Auth, cookies refreshed) on every page |
| Session check on the API | `Authorization: Bearer <access token>`; the API calls Supabase `GET /auth/v1/user` (rejects expired and revoked sessions), caches the result at most 30 s or until expiry, and reads claims only after that check |
| MFA (TOTP) | enrol (QR code and manual key) and remove in Settings → Security; challenge at `/mfa`; a user with a verified factor must reach AAL2: the proxy sends AAL1 sessions to `/mfa`, the API answers `403 mfa_required`, and restrictive RLS policies deny AAL1 access to every owned table |
| Account deletion | requires a sign-in within the last 10 minutes (`amr` timestamp); deletes the user's storage objects, then the auth user, which cascades to every owned row; the shared corpus is untouched |

## 3. Authorization and isolation

- RLS is enabled on all nine user-owned tables: `profiles`, `conversations`, `messages`, `research_runs`,
  `run_events`, `retrieval_traces`, `cache_entries`, `user_documents`, `user_chunks`. Each has an owner policy and
  a restrictive policy that requires AAL2 when the user has a verified factor (`rx.has_verified_mfa()`).
- User statements run inside `user_tx`: `SET LOCAL ROLE authenticated` plus the verified claims as
  `request.jwt.claims`, so RLS applies even though the API holds a database connection.
- The privileged database connection (`DATABASE_URL`) is used for the shared public corpus, for trace writes of
  runs without a user scope, and for account deletion. The Supabase service-role key is not used by ResearchX.
- Object identifiers in URLs (conversations, runs, documents) are looked up as the user: another user's id returns
  404, never data (IDOR tests in §6).
- Retrieval requires a `RetrievalScope`; there is no default. A run searches the public corpus plus only the
  scope user's documents (RLS and an explicit owner filter). Private text is never cached for other users.
- The answer cache is per user (key includes the user id; RLS hides other users' rows). So is the external-provider
  result cache (`cache_entries` kind `api`, 2026-10-08).
- Knowledge Map (2026-10-08): the canonical graph reads only the shared corpus tables (a test asserts the graph modules
  touch no user table). A run's graph lives in `research_runs.answer` and is read as the user with an owner filter;
  another user's run id returns the same 404 as a missing one, for the graph, Ask, diff and promotion routes. Private
  document nodes exist only in their owner's run graphs, never in the corpus index, search or shared metadata, and
  cannot be proposed for promotion. A promotion proposal records no user id in the shared `runtime_sources` table.
- Storage: a private bucket; objects live under the user's id and three storage policies restrict them to the
  owner.

## 4. API hardening

- Strict request models (`extra=forbid`), bounded question length (2,000 characters) and uploads (2 MB per file,
  10 documents, 400 chunks).
- Per-user rate limit on research runs (`RESEARCHX_RATE_PER_MIN`, default 12).
- CORS allowlist (`RESEARCHX_CORS_ORIGINS`); the browser normally calls the API same-origin through the Next.js
  rewrite.
- Security headers on every page: `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy` (microphone for dictation only), and a
  `Content-Security-Policy` (`apps/web/next.config.ts`): everything from the page's own origin, `connect-src` adds only
  the Supabase project (https and wss); scripts and styles allow inline (Next.js bootstrap), `unsafe-eval` only in
  development.
- Error responses carry no stack traces or secrets; traces store no credentials.
- External content (papers, web pages, uploads) is data: it is delimited, never executed, and model calls that read
  it hold no tools or secrets (`docs/ARCHITECTURE.md` §60). Web results pass a deterministic source gate before use.
- Prompt injection inside sources (2026-10-14): prompts tell every model that sources are untrusted quoted data; a
  web result with instruction-like text is rejected by the source gate; a synthesis claim whose quote reads like an
  instruction is rejected (`instruction_like_quote`); a runtime fact extracted from text carrying instructions is
  rejected (`instruction_like_source`). Answers are rendered only from claims and facts that pass deterministic
  validation, so injected text cannot add an unsupported statement.

## 5. Secrets and configuration

| Variable | Where | Secret |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | browser | no (public by design; RLS protects data) |
| `RESEARCHX_API_ORIGIN` | Next.js server (rewrite target) | no |
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY` | API (token verification) | no |
| `DATABASE_URL` | API only | yes |
| `NEBIUS_API_KEY`, `TAVILY_API_KEY`, `S2_API_KEY` | API only | yes |
| `RESEARCHX_CORS_ORIGINS`, `RESEARCHX_RATE_PER_MIN` | API | no |

Local values live in git-ignored `.env` / `apps/web/.env.local`; production values in the deployment platform's
secret store. `.env.example` lists names only.

## 6. Verification

- `tests/test_auth_unit.py` (6, offline): missing and rejected tokens, claims must match the user Supabase returns,
  MFA state and assurance level, short verification cache, `user_tx` runs as `authenticated` with verified claims,
  retrieval scope has no default.
- `tests/test_security_integration.py` (10, live, `RX_INTEGRATION=1`): generic error on invalid credentials; API
  requires a valid session; refresh restores a session and global logout kills it; users see only their own
  conversations and cannot modify others'; forged `user_id` in bodies is rejected; RLS blocks cross-user access
  even with hand-written SQL; runs, traces and answer cache are private; private documents never cross tenants;
  TOTP enrolment, challenge, AAL2 enforcement and removal; account deletion removes every owned record.
- `tests/test_injection.py` (4, offline): instruction-like quotes and sources are rejected; injected text never
  becomes a claim. `tests/test_routing.py` (20, offline): no request field can name a model, a user-private context
  never reaches a provider not approved for private documents, the embedding route is never generative, unhealthy
  routes hand over to their fallback, in-process caches are keyed per user.

## 7. Manual configuration (cannot be automated from the repository)

1. Supabase Auth → URL configuration: set the production Site URL and add `<origin>/auth/callback` to the redirect
   allowlist for every deployed origin (e-mail confirmation and password-reset links use it).
2. Supabase Auth → SMTP: configure a custom SMTP provider before real users sign up; the built-in mailer allows
   only a few e-mails per hour.
3. Deployment platform: set the variables of §5 for the web app and the API; never expose `DATABASE_URL` or
   provider keys to the web project.
4. Google sign-in: Google Cloud Console → APIs & Services → Credentials → OAuth client ID (Web application), authorized
   redirect URI `https://<project-ref>.supabase.co/auth/v1/callback`; then Supabase Auth → Sign In / Providers →
   Google: enable, paste the client ID and secret. The OAuth consent screen needs the app name, support e-mail and
   the `email`, `profile` and `openid` scopes.
5. GitHub sign-in: GitHub → Settings → Developer settings → OAuth Apps → New OAuth App, callback URL
   `https://<project-ref>.supabase.co/auth/v1/callback`; then Supabase Auth → Sign In / Providers → GitHub: enable,
   paste the client ID and a client secret.
6. Connecting providers to an existing account: Supabase Auth → Sign In / Providers → "Allow manual linking" (off
   by default; until then Settings shows that connecting is not enabled).
7. Redirect allowlist (item 1) must include `<origin>/auth/callback**` for every origin, including
   `http://localhost:4730` for local testing, or the provider round trip falls back to the Site URL.

Status 2026-10-15: Google and GitHub are enabled in the project (`/auth/v1/settings` reports both); `/auth/v1/authorize`
redirects to accounts.google.com and github.com with a client id and the project's callback. A full sign-in needs a
person's own Google or GitHub account, so the round trip back to the app was not automated.

## 8. Residual risks

- The rate limiter is in memory, per API instance; with several instances the effective limit multiplies.
- Credentials were shared in chat during development, and one password is reused across two services (owner's
  decision to rotate; `planning/DECISIONS.md` §34).
- The database is at 455 MB of the 500 MB Free-plan quota (2026-10-15). Above 450 MB the API stops writing answer
  and provider-result cache rows (guard `cache.write_limit_mb`), which is the case now; traces still grow. When full,
  Supabase switches the database to read-only and sign-ups, conversations and traces fail until space is freed
  (pruning old evaluation traces or a plan upgrade: the owner's decision).
- Laya runs as an advisory sidecar in shadow mode: its injection answers are traced, never enforced. On a small
  hand-labelled set it found 12 of 14 injection attempts with no false alarm, against 2 of 14 for the deterministic
  pattern (`data/reports/laya_eval.md`); enforcement still relies on data delimiting, tool-free model calls, the source
  gate and deterministic validation.

## 9. Model routing and caches (2026-10-14)

- Routing never grants permissions: the router chooses among configured routes only; no request field names a model
  or a route; Laya (shadow mode) never reaches it, and its optional advise mode (off) could only raise the synthesis
  complexity hint (`services/model_gateway/router.py`, `docs/ARCHITECTURE.md` §67).
- A context holding the user's private document text goes only to a provider marked `private_data: true` in
  `config/models.yaml` (Nebius, ResearchX's own inference provider). Private documents are never sent to the search
  providers (S2, OpenAlex, arXiv, CORE, OpenReview, Tavily): those receive the question's keyword terms (scholarly
  APIs) or the question as asked (Tavily), never document text.
- Caches never cross users: the answer and provider-result caches are rows with the user's id under RLS; the
  in-process query-embedding and reranker caches key every entry by user id (and model, version, release, the full
  candidate texts). Failed or cut-off provider searches and abstained answers are never cached. Uploading or deleting a
  private document empties the user's answer cache; deleting a conversation removes its runs' cache rows.
- Semantic caching is deferred: similarity cannot establish that two research questions ask for the same fact.
- Dictation uses the browser's speech-recognition service (audio leaves the device to the browser vendor); it is
  opt-in per use and labelled in the UI.
