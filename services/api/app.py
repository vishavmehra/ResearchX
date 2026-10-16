"""ResearchX HTTP API (docs/API_SPEC.md §40-60, docs/SECURITY.md).

Every route except /api/health requires `Authorization: Bearer <Supabase access token>`. Supabase Auth verifies the
token (services/auth/supabase.py) and the user id comes only from that verified token, never from a request body (all
bodies forbid unknown fields, so a forged user_id is rejected). User-owned data is read and written inside user_tx():
role `authenticated` + the verified claims, so PostgreSQL RLS enforces ownership even if a route had a bug. Users with
a verified MFA factor must hold an aal2 session. The shared corpus is read through the backend service connection; no
service-role key exists in this process, and nothing secret is ever returned.

  GET    /api/health                                   public status (no user data)
  GET    /api/me | PATCH /api/me                       profile, email, assurance level, MFA status
  DELETE /api/account                                  delete the account and everything it owns (recent sign-in)
  POST   /api/research                                 run research: JSON, or SSE of the run's real stage events
  GET    /api/research/{research_run_id}/trace         own runs only
  GET    /api/conversations?q=&limit=&before=          own chats, most recent first
  POST   /api/conversations | PATCH | DELETE /api/conversations/{id}
  GET    /api/conversations/{id}/messages?limit=&before=
  GET    /api/documents | POST /api/documents | DELETE /api/documents/{id}   private documents
  GET    /api/metrics/architecture                     own stage metrics + global aggregates (no content)
  GET    /api/graph/stats|search|landscape|neighborhood|node|edge|diff|ask   Knowledge Map over the shared corpus
  GET    /api/graph/run/{research_run_id} | /api/graph/runs | /api/graph/discoveries   own runs' runtime graphs only
  POST   /api/graph/promotions                         propose an external source of an own run for review (never automatic)
  GET    /api/analysis/paper-diff|benchmark|recipe|discrepancy|transfer|plan   the eight analyses (docs/API_SPEC.md §53)
  POST   /api/analysis/papers/ask|claim|review          Paper-to-Paper Diff from chat text; Claim-to-Evidence Auditor; review audit
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from services.analysis import benchmark as bench
from services.analysis import claims, discrepancy, enrich, planner, recipe, review, transfer
from services.analysis import papers as paper_diff
from services.analysis import runtime_papers as rtp
from services.auth import supabase as auth
from services.auth.supabase import AuthError, AuthUser, user_tx
from services.env import load_env
from services.graph import knowledge_map as km
from services.graph import runtime as runtime_graph
from services.laya import decider as laya
from services.retrieval.scope import RetrievalScope
from services.runtime import documents, reader, tools
from services.runtime.orchestrator import Orchestrator
from services.runtime.trace import RunTracer
from services.storage.db import connect

load_env()
ROOT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"^rx_[0-9A-HJKMNP-TV-Z]{26}$")
RATE_PER_MIN = int(os.environ.get("RESEARCHX_RATE_PER_MIN", "12"))
QUOTA_MB = 500
REAUTH_WINDOW_S = 600

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """One-time costs are paid at startup in the background, not by the first user: the Knowledge Map index (~6.7 s),
    the BM25 index, the model catalog and the database connection. Requests are served meanwhile; one that needs a part
    still warming waits for that part only. RESEARCHX_WARMUP=0 turns it off (tests, one-off scripts)."""
    if os.environ.get("RESEARCHX_WARMUP", "1") != "0":  # two threads: the graph index and the orchestrator use their own connections
        threading.Thread(target=_warm, args=(lambda: graph(),), name="warmup-graph", daemon=True).start()
        threading.Thread(target=_warm, args=(_warm_orchestrator,), name="warmup-runtime", daemon=True).start()
    yield


def _warm_orchestrator() -> None:
    o = orch()
    o.bm25()
    o.q("select 1")
    o.gw.resolve_model("embedding")


def _warm(part) -> None:
    try:
        part()
    except Exception:  # an optimisation only: a failure shows up, and is retried, on the first request that needs the part
        pass


app = FastAPI(title="ResearchX API", version="0.2.0", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get("RESEARCHX_CORS_ORIGINS", "http://localhost:3000").split(",") if o],
                   allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type", "X-Filename"])

_orch: Orchestrator | None = None
_db = None
_orch_lock, _db_lock = threading.Lock(), threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


# ---------------------------------------------------------------------------------------------- infrastructure
def orch() -> Orchestrator:
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator()
        return _orch


def _conn():
    global _db
    if _db is None or _db.closed or _db.broken:
        _db = connect(autocommit=True)
    return _db


@contextmanager
def as_user(user: AuthUser):
    """A transaction on the shared connection that runs as the user: every statement inside is subject to RLS."""
    with _db_lock:
        with user_tx(_conn(), user) as c:
            yield c


def service(sql: str, params: tuple = ()) -> list[tuple]:
    """Shared-corpus reads, global aggregates and the single account-deletion statement. Never for user data reads."""
    with _db_lock:
        cur = _conn().execute(sql, params)
        return cur.fetchall() if cur.description else []


def rate_limit(key: str, per_min: int = RATE_PER_MIN) -> None:
    # ponytail: in-memory sliding window per serverless instance; move to a shared store if abuse becomes real
    now, hits = time.time(), _hits[key]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= per_min:
        raise HTTPException(429, "Rate limit reached: try again in a minute.")
    hits.append(now)


@app.exception_handler(AuthError)
def _auth_error(_: Request, e: AuthError):
    return JSONResponse({"detail": {"code": e.code, "message": e.message}}, status_code=e.status)


def current_user(authorization: str | None = Header(None)) -> AuthUser:
    token = authorization[7:].strip() if authorization and authorization[:7].lower() == "bearer " else None
    user = auth.verify(token)
    if user.needs_mfa:  # a verified factor exists but this session has not passed the MFA challenge
        raise AuthError(403, "mfa_required", "Complete two-factor authentication to continue.")
    return user


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # identity is never accepted from the body


class ResearchRequest(Strict):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    input_modality: Literal["text", "voice"] = "text"
    stream: bool = False


class NewConversation(Strict):
    title: str | None = Field(default=None, max_length=120)


class Rename(Strict):
    title: str = Field(min_length=1, max_length=120)


class ProfileUpdate(Strict):
    display_name: str | None = Field(default=None, max_length=80)


class DeleteAccount(Strict):
    confirm: Literal["DELETE MY ACCOUNT"]


def title_from(query: str) -> str:
    t = " ".join(query.split())
    return t if len(t) <= 60 else t[:60].rsplit(" ", 1)[0] + "…"


def like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def public(result: dict, conversation_id: str) -> dict:
    keys = ("research_run_id", "status", "mode", "fallback_reason", "intent", "answer", "claims", "sources", "cited_sources", "contradictions",
            "cached", "origin_research_run_id", "total_ms", "trace_persisted", "usage", "routing", "stages",
            "evidence_status", "fields", "request", "rejected_facts", "rejected_candidates",  # field-level grounding outcome
            "unused_sources",  # external results found but not used, with why
            "graph")  # the run's runtime research graph (services/graph/runtime.py)
    out = {k: result.get(k) for k in keys}
    out["removed"] = [{"reason": r.get("reason"), "text": r.get("text")} for r in result.get("removed") or []]
    out["conversation_id"] = conversation_id
    return out


# ---------------------------------------------------------------------------------------------- public
@app.get("/api/health")
def health():
    o = orch()
    out = {"api": "ok", "release": o.release, "pipeline_version": o.cfg["pipeline_version"],
           "bm25_artifact": (ROOT / "data" / "indexes" / "bm25" / f"v{o.release}.npz").exists(),
           "auth": "supabase" if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_PUBLISHABLE_KEY") else "unconfigured",
           "capabilities": {"nebius_models": bool(os.environ.get("NEBIUS_API_KEY")), "tavily": bool(os.environ.get("TAVILY_API_KEY")),
                            "semantic_scholar_key": bool(os.environ.get("S2_API_KEY")),
                            "laya": {k: v for k, v in laya.status(o.cfg).items() if k != "error"}}}
    try:
        status = service("select load_status from rx.dataset_releases where version=%s", (o.release,))
        size = float(service("select sum(pg_database_size(datname)) from pg_database")[0][0])
        out.update(database="ok", release_status=status[0][0] if status else "MISSING", database_mb=round(size / 1e6), quota_mb=QUOTA_MB,
                   quota_used=round(size / (QUOTA_MB * 1e6), 3))
    except Exception:
        out["database"] = "unreachable"
    return out


# ---------------------------------------------------------------------------------------------- account
@app.get("/api/me")
def me(user: AuthUser = Depends(current_user)):
    meta = user.claims.get("user_metadata") or {}
    seed = str(meta.get("full_name") or meta.get("name") or "").strip()[:80] or None  # OAuth providers supply a name
    with as_user(user) as c:
        c.execute("insert into rx.profiles (id, display_name) values (%s, %s) on conflict (id) do nothing", (user.id, seed))
        name, avatar, created = c.execute("select display_name, avatar_url, created_at from rx.profiles where id = %s", (user.id,)).fetchone()
    return {"id": user.id, "email": user.email, "aal": user.aal, "mfa_enabled": user.mfa_enabled,
            "profile": {"display_name": name, "avatar_url": avatar, "created_at": created}}


@app.patch("/api/me")
def update_me(body: ProfileUpdate, user: AuthUser = Depends(current_user)):
    with as_user(user) as c:
        c.execute("insert into rx.profiles (id, display_name) values (%s, %s) on conflict (id) do update set display_name = excluded.display_name, "
                  "updated_at = now()", (user.id, (body.display_name or "").strip() or None))
    return {"ok": True}


@app.delete("/api/account")
def delete_account(body: DeleteAccount, user: AuthUser = Depends(current_user)):
    if not any(m.get("timestamp", 0) >= time.time() - REAUTH_WINDOW_S for m in user.claims.get("amr") or []):
        raise AuthError(403, "reauthentication_required", "Sign in again to confirm it is you, then delete the account within 10 minutes.")
    try:
        removed = documents.delete_all_objects(user, documents.Storage(user))  # originals first: SQL cannot remove Storage files
    except documents.DocumentError as e:
        raise HTTPException(e.status, e.message) from None
    service("delete from auth.users where id = %s", (user.id,))  # cascades to profile, chats, messages, runs, traces, cache, documents
    auth.forget(user.token)
    return {"deleted": True, "storage_objects_removed": removed}


# ---------------------------------------------------------------------------------------------- research
@app.post("/api/research")
def research(req: ResearchRequest, user: AuthUser = Depends(current_user)):
    rate_limit(user.id)
    with as_user(user) as c:
        conv = str(req.conversation_id) if req.conversation_id else None
        if conv and not c.execute("select 1 from rx.conversations where id = %s and user_id = %s", (conv, user.id)).fetchone():
            raise HTTPException(404, "Conversation not found.")
        if not conv:
            conv = str(c.execute("insert into rx.conversations (user_id, title) values (%s, %s) returning id", (user.id, title_from(req.query))).fetchone()[0])
        c.execute("insert into rx.messages (conversation_id, user_id, role, content, input_modality) values (%s,%s,'user',%s,%s)",
                  (conv, user.id, req.query, req.input_modality))
    scope = RetrievalScope(user)

    def finish(result: dict) -> dict:
        with as_user(user) as c:
            run = c.execute("select id from rx.research_runs where research_run_id = %s", (result["research_run_id"],)).fetchone()
            c.execute("insert into rx.messages (conversation_id, user_id, role, content, research_run_id, metadata) values (%s,%s,'assistant',%s,%s,%s)",
                      (conv, user.id, result["answer"], run[0] if run else None, json.dumps({"status": result["status"], "mode": result["mode"]})))
            c.execute("update rx.conversations set updated_at = now() where id = %s", (conv,))
        return public(result, conv)

    if not req.stream:
        return finish(orch().run(req.query, scope, conv))

    events: queue.Queue = queue.Queue()

    def work():
        try:
            events.put(("result", finish(orch().run(req.query, scope, conv, on_event=lambda e: events.put((e["type"], e))))))
        except Exception:  # details stay server-side; the run's own trace records what failed
            events.put(("error", {"error": "The research run failed."}))

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            kind, payload = events.get()
            yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
            if kind in ("result", "error"):
                return

    # no-transform: proxies (the Next.js rewrite) must not compress, which would buffer the stage events until the run ends
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.get("/api/research/{research_run_id}/trace")
def trace(research_run_id: str, user: AuthUser = Depends(current_user)):
    if not RUN_ID.match(research_run_id):
        raise HTTPException(422, "Invalid research_run_id (expected rx_ + ULID).")
    with as_user(user) as c:
        run = c.execute("select id, research_run_id, conversation_id, query, normalized_query, status, mode, fallback_reason, dataset_release, "
                        "routing, usage, answer, error, started_at, finished_at from rx.research_runs where research_run_id = %s and user_id = %s",
                        (research_run_id, user.id)).fetchone()
        if not run:
            raise HTTPException(404, "Unknown research_run_id.")  # same answer for other users' runs: no existence oracle
        uid, rid, conv, query, nq, status, mode, fb, release, routing, usage, answer, err, started, finished = run
        stages = c.execute("select step, status, detail, error, latency_ms, created_at, finished_at from rx.run_events where run_id = %s order by id",
                           (uid,)).fetchall()
        retrieval = c.execute("select method, config, candidates, latency_ms from rx.retrieval_traces where run_id = %s order by id", (uid,)).fetchall()
    return {"research_run_id": rid, "conversation_id": str(conv) if conv else None, "query": query, "normalized_query": nq, "status": status,
            "mode": mode, "fallback_reason": fb, "dataset_release": release, "routing": routing, "usage": usage, "error": err,
            "started_at": started, "finished_at": finished,
            "stages": [{"stage": s, "status": st, "detail": d, "error": e, "latency_ms": ms, "started_at": a, "finished_at": b}
                       for s, st, d, e, ms, a, b in stages],
            "retrieval": [{"method": m, "config": cfg, "candidates": cand, "latency_ms": ms} for m, cfg, cand, ms in retrieval],
            "answer": answer}


# ---------------------------------------------------------------------------------------------- conversations
@app.get("/api/conversations")
def conversations(q: str | None = Query(None, max_length=200), limit: int = Query(30, ge=1, le=100), before: datetime | None = None,
                  user: AuthUser = Depends(current_user)):
    sql, params = "select id, title, created_at, updated_at from rx.conversations where user_id = %s", [user.id]
    if before:
        sql += " and updated_at < %s"
        params.append(before)
    if q and q.strip():
        sql += (" and (title ilike %s escape '\\' or exists (select 1 from rx.messages m where m.conversation_id = rx.conversations.id "
                "and m.content ilike %s escape '\\'))")
        params += [like(q.strip()), like(q.strip())]
    sql += " order by updated_at desc limit %s"
    params.append(limit)
    with as_user(user) as c:
        rows = c.execute(sql, tuple(params)).fetchall()
    return {"conversations": [{"id": str(i), "title": t, "created_at": ca, "updated_at": ua} for i, t, ca, ua in rows], "has_more": len(rows) == limit}


@app.post("/api/conversations")
def new_conversation(body: NewConversation, user: AuthUser = Depends(current_user)):
    with as_user(user) as c:
        i = c.execute("insert into rx.conversations (user_id, title) values (%s, %s) returning id", (user.id, (body.title or "New research").strip())).fetchone()[0]
    return {"id": str(i), "title": (body.title or "New research").strip()}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: uuid.UUID, body: Rename, user: AuthUser = Depends(current_user)):
    with as_user(user) as c:
        row = c.execute("update rx.conversations set title = %s, updated_at = now() where id = %s and user_id = %s returning id",
                        (body.title.strip(), str(conversation_id), user.id)).fetchone()
    if not row:
        raise HTTPException(404, "Conversation not found.")
    return {"id": str(conversation_id), "title": body.title.strip()}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: uuid.UUID, user: AuthUser = Depends(current_user)):
    cid = str(conversation_id)
    with as_user(user) as c:
        runs = [r[0] for r in c.execute("select research_run_id from rx.research_runs where conversation_id = %s and user_id = %s", (cid, user.id)).fetchall()]
        if runs:  # cached answers produced by this chat's runs go too
            c.execute("delete from rx.cache_entries where user_id = %s and value->>'research_run_id' = any(%s)", (user.id, runs))
        c.execute("delete from rx.research_runs where conversation_id = %s and user_id = %s", (cid, user.id))
        if not c.execute("delete from rx.conversations where id = %s and user_id = %s returning id", (cid, user.id)).fetchone():
            raise HTTPException(404, "Conversation not found.")  # rolls back: nothing of another user is touched
    return {"deleted": True, "research_runs_deleted": len(runs)}


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: uuid.UUID, limit: int = Query(50, ge=1, le=200), before: datetime | None = None,
             user: AuthUser = Depends(current_user)):
    cid = str(conversation_id)
    with as_user(user) as c:
        conv = c.execute("select title from rx.conversations where id = %s and user_id = %s", (cid, user.id)).fetchone()
        if not conv:
            raise HTTPException(404, "Conversation not found.")
        rows = c.execute("select m.id, m.role, m.content, m.input_modality, m.created_at, r.research_run_id, r.answer, "
                         "round(extract(epoch from (r.finished_at - r.started_at)) * 1000)::int from rx.messages m "
                         "left join rx.research_runs r on r.id = m.research_run_id where m.conversation_id = %s"
                         + (" and m.created_at < %s" if before else "") + " order by m.created_at desc, m.id desc limit %s",
                         (cid, before, limit) if before else (cid, limit)).fetchall()
    # a stored answer is written before the run's total time is known: the run row's own timestamps give it
    return {"title": conv[0], "has_more": len(rows) == limit,
            "messages": [{"id": str(i), "role": role, "content": content, "input_modality": mod, "created_at": at, "research_run_id": rid,
                          "result": {**ans, "total_ms": ans.get("total_ms") or ms} if isinstance(ans, dict) else ans}
                         for i, role, content, mod, at, rid, ans, ms in reversed(rows)]}


# ---------------------------------------------------------------------------------------------- private documents
def _uq(user: AuthUser):
    def run(sql: str, params: tuple = ()) -> list[tuple]:
        with as_user(user) as c:
            cur = c.execute(sql, params)
            return cur.fetchall() if cur.description else []
    return run


@app.get("/api/documents")
def list_documents(user: AuthUser = Depends(current_user)):
    rows = _uq(user)("select id, title, media_type, bytes, status, chunks, error, created_at from rx.user_documents where user_id = %s "
                     "order by created_at desc", (user.id,))
    return {"documents": [{"id": str(i), "title": t, "media_type": m, "bytes": b, "status": s, "chunks": n, "error": e, "created_at": at}
                          for i, t, m, b, s, n, e, at in rows],
            "limits": {"documents": documents.MAX_DOCS, "passages": documents.MAX_CHUNKS, "bytes": documents.MAX_BYTES}}


@app.post("/api/documents")
async def upload_document(request: Request, content_type: str = Header(...), x_filename: str | None = Header(None),
                          user: AuthUser = Depends(current_user)):
    if int(request.headers.get("content-length") or 0) > documents.MAX_BYTES:
        raise HTTPException(413, f"Files must be at most {documents.MAX_BYTES // 1_000_000} MB.")
    data = await request.body()
    media = content_type.split(";")[0].strip().lower()
    title = unquote(x_filename or "Untitled document")
    try:
        out = await run_in_threadpool(documents.ingest, _uq(user), user, orch().gw, documents.Storage(user), title, media, data)
    except documents.DocumentError as e:
        raise HTTPException(e.status, e.message) from None
    await run_in_threadpool(_forget_answers, user)
    return out


def _forget_answers(user: AuthUser) -> None:
    """A private document changed what this user's questions can be answered from: their cached answers are dropped (as
    the user, RLS; nobody else's are touched). Search-result caches stay: they hold public provider results only."""
    _uq(user)("delete from rx.cache_entries where user_id = %s and kind = 'model'", (user.id,))


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: uuid.UUID, user: AuthUser = Depends(current_user)):
    try:
        found = documents.delete_document(_uq(user), user, documents.Storage(user), str(document_id))
    except documents.DocumentError as e:
        raise HTTPException(e.status, e.message) from None
    if not found:
        raise HTTPException(404, "Document not found.")
    _forget_answers(user)  # cached answers may quote the deleted document
    return {"deleted": True}


# ---------------------------------------------------------------------------------------------- observability
@app.get("/api/metrics/architecture")
def architecture_metrics(user: AuthUser = Depends(current_user)):
    sql = "select stage, status, invocations, avg_latency_ms, p95_latency_ms from rx.architecture_metrics order by stage, status"
    with as_user(user) as c:
        mine = c.execute(sql).fetchall()
    every = service(sql)  # aggregates over all runs: counts and latencies only, no query text or user ids
    shape = lambda rows: [{"stage": s, "status": st, "invocations": n, "avg_latency_ms": a, "p95_latency_ms": p} for s, st, n, a, p in rows]  # noqa: E731
    return {"mine": shape(mine), "global": shape(every)}


# ---------------------------------------------------------------------------------------------- knowledge map
# Canonical layer: the shared public corpus (services/graph/knowledge_map.py), the same for every signed-in user. Runtime
# layer: each research run's graph, stored in the run row and read only as its owner (RLS + owner filter). Private
# documents exist only in their owner's run graphs; the canonical index never reads a user table.
NODE_ID = re.compile(r"^(paper|study|exp|ent|res|cfg|ev|run|doc|rt):\S{1,300}$")
_gdb, _gdb_lock = None, threading.Lock()


class PromotionRequest(Strict):
    research_run_id: str = Field(pattern=RUN_ID.pattern)
    node_id: str = Field(min_length=3, max_length=320)


def graph_q(sql: str, params: tuple = ()) -> list[tuple]:
    """Shared-corpus reads for the Knowledge Map on their own connection, so browsing never queues behind research runs."""
    global _gdb
    with _gdb_lock:
        if _gdb is None or _gdb.closed or _gdb.broken:
            _gdb = connect(autocommit=True)
        try:
            cur = _gdb.execute(sql, params)
            return cur.fetchall() if cur.description else []
        except Exception:
            _gdb = None
            raise


def graph() -> km.GraphIndex:
    return km.get_index(graph_q)


def node_id(value: str | None) -> str:
    if not NODE_ID.match(value or ""):
        raise HTTPException(422, "Invalid node id.")
    return value


def csv_of(value: str | None, allowed: tuple[str, ...], what: str) -> list[str]:
    items = [x for x in (value or "").split(",") if x]
    if bad := [x for x in items if x not in allowed]:
        raise HTTPException(422, f"Unknown {what}: {', '.join(bad[:5])}.")
    return items


def graph_call(fn, *args):
    try:
        return fn(*args)
    except KeyError:
        raise HTTPException(404, "Not found in the knowledge map.") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


def run_graph(user: AuthUser, research_run_id: str) -> dict:
    """A run's runtime graph, read as its owner. Another user's run gets the same 404 as a missing one (no existence oracle)."""
    if not RUN_ID.match(research_run_id or ""):
        raise HTTPException(422, "Invalid research_run_id (expected rx_ + ULID).")
    with as_user(user) as c:
        row = c.execute("select answer->'graph' from rx.research_runs where research_run_id = %s and user_id = %s", (research_run_id, user.id)).fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "No knowledge graph for this research run.")
    g = row[0]
    urls = sorted({n["provenance"]["url"] for n in g["nodes"] if n["origin"] == "runtime_external" and n["provenance"].get("url")})
    if urls:  # reviewed state of external sources: shared metadata without user linkage
        runtime_graph.apply_review_states(g, {(u, h): s for u, h, s in service(
            "select url, content_sha256, state from rx.runtime_sources where url = any(%s)", (urls,))})
    return g


def decorated(g: dict) -> dict:
    """Canonical nodes of a run graph carry the corpus index's current label and summary."""
    idx = graph()
    for n in g["nodes"]:
        if n["origin"] == "canonical" and (s := idx.node(n["id"])):
            n["label"], n["canonical_meta"], n["flags"] = s["label"], s["meta"], sorted(set(n["flags"]) | set(s["flags"]))
    return g


@app.get("/api/graph/stats")
def graph_stats(user: AuthUser = Depends(current_user)):
    return graph().stats()


@app.get("/api/graph/search")
def graph_search(q: str = Query(min_length=2, max_length=200), types: str | None = Query(None, max_length=300), limit: int = Query(20, ge=1, le=50),
                 user: AuthUser = Depends(current_user)):
    return {"results": graph().search(q, csv_of(types, km.NODE_TYPES, "node types"), limit)}


@app.get("/api/graph/landscape")
def graph_landscape(root: str | None = Query(None, max_length=320), user: AuthUser = Depends(current_user)):
    return graph_call(graph().landscape, node_id(root) if root else None)


@app.get("/api/graph/neighborhood")
def graph_neighborhood(node: str = Query(max_length=320), node_types: str | None = Query(None, max_length=300),
                       edge_types: str | None = Query(None, max_length=400), limit: int = Query(40, ge=1, le=100),
                       offset: int = Query(0, ge=0, le=100_000), user: AuthUser = Depends(current_user)):
    return graph_call(graph().neighborhood, node_id(node), csv_of(node_types, km.NODE_TYPES, "node types"),
                      csv_of(edge_types, km.EDGE_TYPES, "edge types"), limit, offset)


@app.get("/api/graph/node")
def graph_node(id: str = Query(max_length=320), user: AuthUser = Depends(current_user)):  # noqa: A002 (query parameter name)
    return graph_call(graph().node_detail, node_id(id))


@app.get("/api/graph/edge")
def graph_edge(type: str = Query(max_length=40), source: str = Query(max_length=320), target: str = Query(max_length=320),  # noqa: A002
               user: AuthUser = Depends(current_user)):
    return graph_call(graph().edge_detail, type, node_id(source), node_id(target))


@app.get("/api/graph/diff")
def graph_diff(a: str = Query(max_length=320), b: str = Query(max_length=320), run: str | None = Query(None, max_length=40),
               user: AuthUser = Depends(current_user)):
    a, b = node_id(a), node_id(b)
    if a == b or not all(x.startswith(("exp:", "rt:exp:")) for x in (a, b)):
        raise HTTPException(422, "Compare two different experiments.")
    extra = {}
    if runtime := [x for x in (a, b) if x.startswith("rt:exp:")]:
        if not run:
            raise HTTPException(422, "A runtime experiment belongs to a research run: pass run=<research_run_id>.")
        g = run_graph(user, run)
        nodes = {n["id"]: n for n in g["nodes"]}
        for x in runtime:
            if x not in nodes:
                raise HTTPException(404, "Experiment not in this research run's graph.")
            extra[x] = (runtime_graph.runtime_experiment(g, x), nodes[x])
    out = graph_call(graph().diff, a, b, extra)
    out.pop("_raw", None)
    return out


@app.get("/api/graph/ask")
def graph_ask(node: str = Query(max_length=320), q: str = Query(max_length=40), run: str | None = Query(None, max_length=40),
              user: AuthUser = Depends(current_user)):
    node = node_id(node)
    if node.startswith("run:"):
        return graph_call(runtime_graph.ask_run, run_graph(user, node.removeprefix("run:")), q)
    if run:  # a node as seen in one of the user's runs: runtime nodes need it, corpus nodes may come with it
        g = run_graph(user, run)
        if any(n["id"] == node and n["origin"] != "canonical" for n in g["nodes"]):
            return graph_call(runtime_graph.ask_node, g, node, q, graph())
    if node.startswith(("rt:", "doc:")):
        raise HTTPException(422, "A runtime node belongs to a research run: pass run=<research_run_id>.")
    return graph_call(graph().ask, node, q)


@app.get("/api/graph/run/{research_run_id}")
def graph_run(research_run_id: str, user: AuthUser = Depends(current_user)):
    return decorated(run_graph(user, research_run_id))


@app.get("/api/graph/runs")
def graph_runs(limit: int = Query(20, ge=1, le=50), user: AuthUser = Depends(current_user)):
    with as_user(user) as c:
        rows = c.execute("select research_run_id, query, status, started_at, answer->'graph'->'stats' from rx.research_runs "
                         "where user_id = %s and answer->'graph' is not null order by started_at desc limit %s", (user.id, limit)).fetchall()
    return {"runs": [{"research_run_id": r, "query": q, "status": s, "started_at": a, "stats": st} for r, q, s, a, st in rows]}


@app.get("/api/graph/discoveries")
def graph_discoveries(days: int = Query(1, ge=1, le=30), user: AuthUser = Depends(current_user)):
    """What the user's own runs found outside the corpus (and in their private documents) in the last `days` days."""
    with as_user(user) as c:
        rows = c.execute("select research_run_id, query, started_at, answer->'graph'->'nodes' from rx.research_runs where user_id = %s "
                         "and started_at > now() - make_interval(days => %s) and answer->'graph' is not null order by started_at desc limit 100",
                         (user.id, days)).fetchall()
    found: dict[str, dict] = {}
    for rid, query, at, nodes in rows:
        for n in nodes or []:
            if n["origin"] in ("runtime_external", "private_document"):
                item = found.setdefault(n["id"], {**{k: n[k] for k in ("id", "type", "label", "origin", "status")},
                                                  "provider": n["provenance"].get("provider"), "url": n["provenance"].get("url"), "runs": []})
                if runtime_graph.RANK[n["status"]] > runtime_graph.RANK[item["status"]]:
                    item["status"] = n["status"]
                item["runs"].append({"research_run_id": rid, "query": query, "started_at": at})
    items = sorted(found.values(), key=lambda i: (-runtime_graph.RANK[i["status"]], i["label"]))
    return {"days": days, "runs": len(rows), "counts": dict(Counter(i["status"] for i in items)), "items": items[:200]}


@app.post("/api/graph/promotions")
def graph_promote(body: PromotionRequest, user: AuthUser = Depends(current_user)):
    """Propose one external source of the caller's run for the next corpus release. It becomes a CANDIDATE in the shared
    runtime-source table (no user linkage); only the reviewed release pipeline can promote it."""
    rate_limit(f"promote:{user.id}")
    g = run_graph(user, body.research_run_id)
    try:
        verdict = runtime_graph.promotion_checks(g, body.node_id)
    except KeyError:
        raise HTTPException(404, "Node not in this research run's graph.") from None
    if not verdict["eligible"]:
        return {"state": "NOT_ELIGIBLE", **verdict}
    p = next(n for n in g["nodes"] if n["id"] == body.node_id)["provenance"]
    licence = None
    if p.get("doi"):
        try:
            licence = tools.unpaywall_lookup(p["doi"], timeout=4)
        except Exception as e:  # recorded; the release pipeline re-checks the licence anyway
            licence = {"error": type(e).__name__}
    note = json.dumps({"promotion": {"requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": verdict["checks"],
                                     "licence": licence, "graph_version": g.get("version")}})
    if service("update rx.runtime_sources set state = 'CANDIDATE', validation = validation || %s::jsonb "
               "where url = %s and content_sha256 = %s and state = 'RUNTIME' returning state", (note, p["url"], p["content_sha256"])):
        return {"state": "CANDIDATE", **verdict, "licence": licence}
    current = service("select state from rx.runtime_sources where url = %s and content_sha256 = %s", (p["url"], p["content_sha256"]))
    if not current:
        raise HTTPException(404, "The source record is no longer available for review.")
    return {"state": current[0][0], **verdict, "licence": licence, "note": "already proposed or reviewed"}


# ---------------------------------------------------------------------------------------------- analysis features
# The eight analyses (services/analysis, docs/ARCHITECTURE.md §66): Paper-to-Paper Diff, Result Discrepancy Investigator,
# Reproducibility Recipe Builder, Next Experiment Planner, Claim-to-Evidence Auditor, Generalization and Transferability
# Analyzer, Literature Review Completeness Auditor, Benchmark Reality Explorer. They read the same validated records, field
# facts, configuration identity, comparison engine and Research Trace; Paper-to-Paper Diff and Benchmark Reality Explorer
# also read runtime papers on request (services/analysis/runtime_papers.py, §69). Each call is a research run of the caller
# (owner-only trace, RLS); the run row keeps a compact summary and the inputs, not the evidence payload (DB quota).
ANALYSIS_KEEP = ("kind", "version", "summary", "verdict", "inputs", "missing", "conflicting")
ANALYSIS_PER_MIN = int(os.environ.get("RESEARCHX_ANALYSIS_PER_MIN", "60"))  # deterministic corpus reads; model/provider calls keep RATE_PER_MIN


def analysis(user: AuthUser, kind: str, title: str, work, per_min: int = ANALYSIS_PER_MIN) -> dict:
    """Run one analysis as a traced research run: `work(tr)` returns (run status, result)."""
    rate_limit(f"analysis:{user.id}", per_min)
    o = orch()
    tr = RunTracer(o.connect, title, " ".join(title.casefold().split()), o.release, user=user)
    try:
        status, result = work(tr)
    except KeyError:
        tr.finish("FAILED", {"kind": kind}, {}, {}, error="not found")
        raise HTTPException(404, "Not found in the corpus.") from None
    except ValueError as e:
        tr.finish("FAILED", {"kind": kind}, {}, {}, error=str(e)[:200])
        raise HTTPException(422, str(e)) from None
    except Exception as e:
        tr.finish("FAILED", {"kind": kind}, {}, {}, error=type(e).__name__)
        raise
    meta = tr.finish(status, {k: result[k] for k in ANALYSIS_KEEP if k in result}, dict(result.get("usage") or {}), dict(result.get("routing") or {}))
    return {**result, "status": status, "research_run_id": tr.research_run_id, "stages": tr.stages, "total_ms": meta["total_ms"],
            "trace_persisted": not meta["trace_errors"]}


def _checked_extract(o):
    def extract(req, fields, sources):  # the runtime rules of the chat path too: another model's row, size or column is not this value
        facts = o._extract(req, fields, sources, Counter(), "fulltext_extraction")[0]
        cands = {s["chunk_id"]: {**s, "unit_type": "external", "paper_id": None} for s in sources}
        for f in facts:
            o._check_fact(f, req, cands, {}, {})
        return [f for f in facts if f.valid]
    return extract


def full_text_fill(idx):
    """Request-time full-text reading for settings the records leave unknown (services/analysis/enrich.py), with the
    shared schema-bound extractor; None when full-text reading is switched off (config evidence.fulltext_reading)."""
    o = orch()
    if not o.cfg["evidence"].get("fulltext_reading", True):
        return None
    extract = _checked_extract(o)
    return lambda models, tr: enrich.fill(idx, models, tr, extract)


class RuntimeTools:
    """The provider and model calls behind runtime papers (services/analysis/runtime_papers.py) for one user: scholarly
    search through the shared provider phase (routing need `papers`, per-user cache, source gate), Semantic Scholar
    identifier lookups, arXiv full texts, and the schema-bound extractors with their deterministic checks. Built only when
    the caller opted in to papers outside the corpus: without it, references are resolved against the corpus alone."""

    def __init__(self, user: AuthUser):
        o, self.user = orch(), user
        self.search = lambda tr, q: o._external(tr, q, Counter(), RetrievalScope(user), [q], need="papers")[0]
        self.lookup = lambda ident: tools.s2_paper(ident, timeout=6)
        self.read = lambda arxiv: reader.read(arxiv, timeout=8)
        self.extract = _checked_extract(o)
        self.extract_results = lambda sources: o._extract_results(sources, Counter())

    def load(self, idx, sources: list[dict], tr) -> list[dict]:
        return rtp.load_many(idx, sources, self.user.id, self.read, self.extract, self.extract_results, tr) if sources else []


def _resolve_papers(idx, user: AuthUser, tr, refs: list[str], topic: str | None, external: bool, ids: list[str]) -> tuple:
    """Corpus ids + typed references + a topic -> (overlay index, paper ids to compare, runtime summary)."""
    rt = RuntimeTools(user) if external else None
    resolutions = rtp.resolve(idx, refs, tr, rt.search if rt else None, rt.lookup if rt else None) if refs else []
    discovery = rtp.discover(idx, topic, tr, rt.search if rt else None) if topic else None
    picked = (discovery or {}).get("picked", [])
    ids = list(ids) + [r["paper_id"] for r in resolutions if r["kind"] == "corpus"] + [p["paper_id"] for p in picked if p["kind"] == "corpus"]
    sources = [r["source"] for r in resolutions if r["kind"] == "runtime"] + [p["source"] for p in picked if p["kind"] == "runtime"]
    loaded = rt.load(idx, sources[:rtp.MAX_REFS], tr) if rt else []
    ok = [p for p in loaded if p.get("status") == "completed"]
    read = {p["key"] for p in ok}  # a corpus paper that could not be read now is still compared as the corpus has it
    ids += [r["paper_id"] for r in resolutions if r["kind"] == "runtime" and r.get("paper_id") and rtp.paper_key(r["source"]) not in read]
    return rtp.Overlay(idx, ok), list(dict.fromkeys(ids + [p["key"] for p in ok])), rtp.summary(loaded, resolutions, discovery)


@app.get("/api/analysis/recipe")
def analysis_recipe(experiment: str | None = Query(None, max_length=320), paper: str | None = Query(None, max_length=320),
                    user: AuthUser = Depends(current_user)):
    """Reproducibility Recipe Builder: the recipe of one experiment (a corpus id, or a runtime experiment you read); with paper= alone, the paper's experiments to choose from."""
    idx = graph()
    if experiment is None:
        pid = node_id(paper)
        if pid not in idx.paper:
            raise HTTPException(404, "Paper not in the corpus.")
        return {"kind": "recipe_options", "paper": idx.node(pid), "experiments": recipe.options(idx, pid)}
    xid = node_id(experiment)
    index = rtp.user_overlay(idx, user.id) if xid.startswith("rt:") else idx  # a runtime experiment this user has read

    def work(tr):
        r = recipe.build(index, xid, orch().release, tr, fill=full_text_fill(idx))
        return ("PARTIAL" if r["missing"] or r["conflicting"] else "ANSWERED"), r
    # may read a full text with the extraction model: the per-user rate of research runs applies
    return analysis(user, "recipe", f"Reproduction recipe for {xid}", work, per_min=RATE_PER_MIN)


def _papers_run(user: AuthUser, ids: list[str], refs: list[str], topic: str | None, external: bool, title: str) -> dict:
    """Paper-to-Paper Diff over corpus ids, typed references and a topic. Two to four papers: the diff; one: its
    configuration inventory; none: the resolutions and candidates, so the user can pick (ABSTAINED)."""
    idx = graph()

    def work(tr):
        ov, pids, rt = _resolve_papers(idx, user, tr, refs, topic, external, ids)
        if not pids:
            return "ABSTAINED", {"kind": "paper_resolution", "version": paper_diff.VERSION, "runtime": rt,
                                 "summary": {"papers": 0, "references": len(refs)}, "inputs": {"refs": refs, "topic": topic, "external": external},
                                 "notes": ["No paper could be resolved; pick one of the candidates, give a DOI or arXiv id, or turn on papers outside the corpus."]}
        fill = full_text_fill(idx)
        r = paper_diff.inventory_view(ov, pids[0], tr, fill) if len(pids) == 1 else paper_diff.diff(ov, pids[:paper_diff.MAX_PAPERS], tr, fill)
        r["runtime"] = rt
        r["inputs"] = {**r["inputs"], "refs": refs, "topic": topic, "external": external, "runtime_papers": [p["key"] for p in rt["papers"]],
                       "dropped": pids[paper_diff.MAX_PAPERS:]}
        unresolved = [x for x in rt["resolutions"] if x["kind"] not in ("corpus", "runtime")]
        return ("PARTIAL" if unresolved or any(p.get("status") != "completed" for p in rt["papers"]) else "ANSWERED"), r
    return analysis(user, "paper_diff", title, work, per_min=RATE_PER_MIN)


@app.get("/api/analysis/paper-diff")
def analysis_paper_diff(papers: str = Query("", max_length=1400), ref: list[str] = Query([], max_length=rtp.MAX_REFS),
                        topic: str | None = Query(None, max_length=rtp.TOPIC_CHARS), external: bool = False, user: AuthUser = Depends(current_user)):
    """Paper-to-Paper Diff: two to four papers side by side with a fairness verdict. Papers are corpus ids (`papers`),
    typed references (`ref`: a title, DOI, arXiv id or URL) or a `topic`; with `external=true` the ones outside the corpus
    are looked up and read at request time (only the typed text goes to the scholarly providers). One paper: its
    configurations (several pre-training / fine-tuning configurations, seeds, splits, protocols)."""
    ids = [node_id(p) for p in papers.split(",") if p]
    if not (ids or ref or topic):
        raise HTTPException(422, "Name two to four papers: titles, DOIs, arXiv ids or URLs, or a topic.")
    if any(len(r) > rtp.REF_CHARS for r in ref):
        raise HTTPException(422, f"A reference is a title, DOI, arXiv id or URL of at most {rtp.REF_CHARS} characters.")
    title = "Paper-to-Paper Diff: " + ", ".join(ids + [" ".join(r.split())[:60] for r in ref] + ([f"topic {topic}"] if topic else []))
    return _papers_run(user, ids, ref, topic, external, title[:300])


class PapersAsk(Strict):
    text: str = Field(min_length=3, max_length=1200)
    external: bool = False


@app.post("/api/analysis/papers/ask")
def analysis_papers_ask(body: PapersAsk, user: AuthUser = Depends(current_user)):
    """Paper-to-Paper Diff from what a researcher writes: "Compare <A> vs <B>, including papers not in the corpus",
    "Find recent papers on <method> and compare their experimental settings", or one paper ("Does <A> report several
    pre-training configurations?"). No ids needed."""
    parsed = rtp.papers_from_text(body.text)
    refs = [r for r in parsed.get("refs", []) if r][: rtp.MAX_REFS]
    if any(len(r) > rtp.REF_CHARS for r in refs):
        raise HTTPException(422, f"Name the papers by title, DOI, arXiv id or URL (at most {rtp.REF_CHARS} characters each).")
    external = body.external or bool(re.search(r"\b(?:not (?:already )?in|outside|beyond)\b[^.]*\bcorpus\b|\bincluding papers\b", body.text, re.I))
    return _papers_run(user, [], refs, parsed.get("topic"), external, ("Paper-to-Paper Diff: " + " ".join(body.text.split()))[:300])


@app.get("/api/analysis/discrepancy")
def analysis_discrepancy(a: str | None = Query(None, max_length=320), b: str | None = Query(None, max_length=320),
                         benchmark: str | None = Query(None, max_length=320), user: AuthUser = Depends(current_user)):
    """Result Discrepancy Investigator: why two experiments report different outcomes; without a and b, the pairs worth investigating."""
    idx = graph()
    if a is None or b is None:  # one side or none: the pairs worth investigating (with that experiment, when one is given)
        rate_limit(f"analysis:{user.id}", ANALYSIS_PER_MIN)
        one = node_id(a or b) if (a or b) else None
        return {"kind": "discrepancy_candidates", "experiment": one, **discrepancy.candidates(idx, node_id(benchmark) if benchmark else None, experiment=one)}
    a, b = node_id(a), node_id(b)
    index = rtp.user_overlay(idx, user.id) if a.startswith("rt:") or b.startswith("rt:") else idx  # runtime experiments this user has read

    def work(tr):
        r = discrepancy.investigate(index, a, b, tr, fill=full_text_fill(idx))
        status = {"CONFLICTING_RECORDS": "CONFLICTING", "UNDETERMINED": "PARTIAL", "UNEXPLAINED": "PARTIAL"}.get(r["verdict"], "ANSWERED")
        return status, r
    return analysis(user, "discrepancy", f"Why do {a} and {b} report different results?", work, per_min=RATE_PER_MIN)


@app.get("/api/analysis/benchmarks")
def analysis_benchmarks(user: AuthUser = Depends(current_user)):
    """Benchmarks with the most reported results and their metrics (selectors of the Benchmark Reality Explorer)."""
    return {"benchmarks": bench.catalog(graph())}


@app.get("/api/analysis/benchmark")
def analysis_benchmark(benchmark: str | None = Query(None, max_length=320), metric: str | None = Query(None, max_length=320),
                       reference: str | None = Query(None, max_length=320), family: str | None = Query(None, max_length=120),
                       year_from: int | None = Query(None, ge=1900, le=2100), year_to: int | None = Query(None, ge=1900, le=2100),
                       q: str | None = Query(None, max_length=400), external: bool = False, ref: list[str] = Query([], max_length=rtp.MAX_REFS),
                       user: AuthUser = Depends(current_user)):
    """Benchmark Reality Explorer: every reported result for one benchmark and metric with its comparability class. The
    benchmark and metric are ids, or named in `q` ("Which ImageNet top-1 results are comparable ...?"). With
    `external=true`, papers outside the corpus that report on the benchmark (found from its name, plus any `ref`) are read
    at request time and their validated results join the table, marked runtime; the same comparability rules apply."""
    idx = graph()
    if q and not (benchmark and metric):
        try:
            benchmark, metric = bench.resolve(idx, q)
        except KeyError:
            raise HTTPException(404, "The question names no benchmark of the corpus catalog; pick one from the list.") from None
    if not (benchmark and metric):
        raise HTTPException(422, "Give a benchmark and a metric, or a question that names them.")
    args = (node_id(benchmark), node_id(metric), node_id(reference) if reference else None, family, year_from, year_to)

    def work(tr):
        index, rt = idx, None
        if external or ref:
            if args[0] not in idx.ent or args[1] not in idx.ent:
                raise KeyError(args[0])
            names = (idx.ent[args[0]]["name"], idx.ent[args[1]]["name"])
            tools_ = RuntimeTools(user) if external else None
            resolutions = rtp.resolve(idx, list(ref), tr, tools_.search if tools_ else None, tools_.lookup if tools_ else None) if ref else []
            found = rtp.discover(idx, f"{names[0]} {names[1]}", tr, tools_.search, include_corpus=False) if tools_ else None
            sources = [r["source"] for r in resolutions if r["kind"] == "runtime"]
            sources += [p["source"] for p in (found or {}).get("picked", []) if p["kind"] == "runtime"]
            loaded = tools_.load(idx, sources[:rtp.MAX_DISCOVER + len(ref)], tr) if tools_ else []
            ok = [p for p in loaded if p.get("status") == "completed"]
            index = rtp.Overlay(idx, ok, align={"benchmark": (args[0], names[0]), "metric": (args[1], names[1])})
            rt = rtp.summary(loaded, resolutions, found)
        r = bench.explore(index, args[0], args[1], tr, *args[2:])
        if rt is not None:
            r["runtime"] = rt
            r["inputs"] = {**r["inputs"], "external": external, "refs": list(ref), "runtime_papers": [p["key"] for p in rt["papers"]]}
        if q:
            r["inputs"]["question"] = q
        return ("ANSWERED" if r["summary"]["ranked"] else "PARTIAL"), r
    # reading papers outside the corpus calls providers and the extraction model: the per-user rate of research runs applies
    return analysis(user, "benchmark", f"Benchmark Reality Explorer: {args[0]} / {args[1]}", work,
                    per_min=RATE_PER_MIN if (external or ref) else ANALYSIS_PER_MIN)


class ClaimRequest(Strict):
    claim: str = Field(min_length=8, max_length=600)


@app.post("/api/analysis/claim")
def analysis_claim(body: ClaimRequest, user: AuthUser = Depends(current_user)):
    """Claim-to-Evidence Auditor: what the corpus records say about a stated claim: supporting, contradicting and inconclusive evidence."""
    idx = graph()

    def work(tr):
        r = claims.audit(idx, body.claim, tr, fill=full_text_fill(idx))
        # research_runs.status allows ANSWERED / PARTIAL / CONFLICTING / ABSTAINED / FAILED; no assessable evidence abstains
        return {"SUPPORTED": "ANSWERED", "CONTRADICTED": "ANSWERED", "MIXED": "CONFLICTING"}.get(r["verdict"], "ABSTAINED"), r
    return analysis(user, "claim", "Claim audit: " + " ".join(body.claim.split())[:200], work, per_min=RATE_PER_MIN)


@app.get("/api/analysis/transfer")
def analysis_transfer(model: str = Query(min_length=2, max_length=200), source: str | None = Query(None, max_length=200),
                      target: str | None = Query(None, max_length=200), user: AuthUser = Depends(current_user)):
    """Generalization and Transferability Analyzer: where else a model has been evaluated and what changed: direct transfers, other papers, untested benchmarks."""
    idx = graph()

    def work(tr):
        r = transfer.analyze(idx, model, tr, source, target)
        return {"OBSERVED": "ANSWERED", "NO_DIRECT_TRANSFER": "PARTIAL"}.get(r["verdict"], "ABSTAINED"), r
    return analysis(user, "transfer", f"Transferability: {' '.join(model.split())}", work)


@app.get("/api/analysis/plan")
def analysis_plan(experiment: str | None = Query(None, max_length=320), query: str | None = Query(None, max_length=300),
                  goal: str | None = Query(None, max_length=600), user: AuthUser = Depends(current_user)):
    """Next Experiment Planner: next experiments for one corpus experiment: reproduce, ablate, missing baselines, robustness, reporting.
    `query` ("model on dataset", e.g. from the chat) picks the best-matching experiment when no id is given."""
    idx = graph()
    if not experiment and not query:
        raise HTTPException(422, "Give an experiment id, or a model and dataset (e.g. '<model> on <dataset>').")
    try:
        xid = node_id(experiment) if experiment else planner.resolve(idx, query)
    except KeyError:
        raise HTTPException(404, f"No corpus experiment matches '{query}'. Name the model as in the paper, and the dataset.") from None

    def work(tr):
        r = planner.plan(idx, xid, tr, goal)
        return ("ANSWERED" if r["recommendations"] else "PARTIAL"), r
    return analysis(user, "plan", f"Next experiments for {xid}", work)


class ReviewRequest(Strict):
    text: str = Field(min_length=20, max_length=20000)
    runtime_search: bool = False  # opt-in: only the recognised entity names are sent to the literature providers


@app.post("/api/analysis/review")
def analysis_review(body: ReviewRequest, user: AuthUser = Depends(current_user)):
    """Literature Review Completeness Auditor: coverage and evidence gaps of a research question or draft literature review."""
    idx, o = graph(), orch()
    search = ((lambda tr, q: o._external(tr, q, Counter(), RetrievalScope(user), [q], need="literature")[0])
              if body.runtime_search else None)

    def work(tr):
        r = review.audit(idx, body.text, tr, o.release, search)
        return ("ANSWERED" if r["verdict"] == "GAPS_FOUND" else "PARTIAL"), r
    # a runtime search calls external providers: the stricter per-user rate of research runs applies
    return analysis(user, "review", "Literature review audit", work, per_min=RATE_PER_MIN if search else ANALYSIS_PER_MIN)
