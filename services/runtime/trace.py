"""Research-run tracing: provable execution (docs/API_SPEC.md §40-48, planning/DECISIONS.md §35).

Every research request gets research_run_id = 'rx_' + ULID, and every pipeline stage that actually ran writes one
rx.run_events row with its REAL status (completed | skipped | not_required | failed). A stage that was never reached
has no row; the UI renders only these rows. Stage details carry counts, ids, model names and timings, never secrets
or prompt text.

The database is ~100 ms away, so rows are written by a background writer in order and finish() waits for the flush:
when a response is returned, its trace is complete. on_event receives the same events for live streaming (SSE).
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager

import psycopg

from services.auth.supabase import rls_claims

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32 (ULID alphabet)


def new_run_id() -> str:
    """'rx_' + ULID: 48-bit millisecond timestamp + 80 random bits, 26 Crockford base32 characters (sortable)."""
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    return "rx_" + "".join(_B32[(n >> s) & 31] for s in range(125, -1, -5))


class Stage:
    def __init__(self, name: str):
        self.name, self.status, self.detail, self.error = name, "completed", {}, None

    def skip(self, reason: str, status: str = "skipped") -> None:
        """Mark the stage as not executed (status skipped or not_required) and say why."""
        self.status, self.detail["reason"] = status, reason


class RunTracer:
    def __init__(self, connect: Callable, query: str, normalized_query: str, release: str, conversation_id: str | None = None,
                 on_event: Callable[[dict], None] | None = None, user=None, budgets: dict[str, int] | None = None):
        """user: the verified AuthUser. Its run rows are written as that user (role authenticated + claims), so RLS
        checks ownership of every insert; None = offline/system run (no owner, invisible to every user).
        budgets: stage name -> latency budget in ms; a stage that ran records its budget and whether it went over."""
        self.research_run_id, self.run_uuid = new_run_id(), str(uuid.uuid4())
        self.on_event, self.t0, self.stages, self.budgets = on_event, time.perf_counter(), [], budgets or {}
        self._q: queue.Queue = queue.Queue()
        self._errors: list[str] = []
        self._writer = threading.Thread(target=self._drain, args=(connect, user), daemon=True)
        self._writer.start()
        self._sql("insert into rx.research_runs (id, research_run_id, conversation_id, user_id, query, normalized_query, dataset_release) "
                  "values (%s,%s,%s,%s,%s,%s,%s)", (self.run_uuid, self.research_run_id, conversation_id, user.id if user else None,
                                                    query, normalized_query, release))
        self._emit({"type": "run", "research_run_id": self.research_run_id, "status": "RUNNING"})

    @contextmanager
    def stage(self, name: str, **detail):
        st, started = Stage(name), time.time()
        st.detail.update(detail)
        self._emit({"type": "stage", "stage": name, "status": "running"})
        t = time.perf_counter()
        try:
            yield st
        except Exception as e:  # recorded as failed, then re-raised for the orchestrator to decide
            st.status, st.error = "failed", f"{type(e).__name__}: {str(e)[:300]}"
            raise
        finally:
            self._stage_row(st, started, round((time.perf_counter() - t) * 1000))

    def record(self, name: str, status: str, latency_ms: int = 0, error: str | None = None, **detail) -> None:
        """One-shot row for a stage that did not run (skipped / not_required) or ran outside a `with` block (threads)."""
        st = Stage(name)
        st.status, st.detail, st.error = status, detail, error
        self._stage_row(st, time.time() - latency_ms / 1000, latency_ms)

    def persist(self, sql: str, params: tuple) -> None:
        """Queue a write that belongs to this run (runtime sources, cache rows); flushed before finish() returns."""
        self._sql(sql, params)

    def retrieval(self, method: str, query: str, config: dict, candidates: list, latency_ms: int) -> None:
        self._sql("insert into rx.retrieval_traces (run_id, query, method, config, candidates, latency_ms) values (%s,%s,%s,%s,%s,%s)",
                  (self.run_uuid, query, method, json.dumps(config), json.dumps(candidates), latency_ms))

    def finish(self, status: str, answer: dict | None, usage: dict, routing: dict, mode: str = "pipeline",
               fallback_reason: str | None = None, error: str | None = None) -> dict:
        self._sql("update rx.research_runs set status=%s, answer=%s, usage=%s, routing=%s, mode=%s, fallback_reason=%s, error=%s, "
                  "finished_at=now() where id=%s", (status, json.dumps(answer), json.dumps(usage), json.dumps(routing), mode,
                                                    fallback_reason, error, self.run_uuid))
        self._q.put(None)
        self._writer.join(timeout=60)
        total = round((time.perf_counter() - self.t0) * 1000)
        self._emit({"type": "done", "research_run_id": self.research_run_id, "status": status, "mode": mode, "total_ms": total,
                    "trace_persisted": not self._errors and not self._writer.is_alive()})
        return {"total_ms": total, "trace_errors": self._errors}

    def _stage_row(self, st: Stage, started: float, latency_ms: int) -> None:
        if st.name in self.budgets and st.status not in ("skipped", "not_required"):
            st.detail.update(budget_ms=self.budgets[st.name], over_budget=latency_ms > self.budgets[st.name])
        self.stages.append({"stage": st.name, "status": st.status, "latency_ms": latency_ms, "detail": st.detail, "error": st.error})
        self._sql("insert into rx.run_events (run_id, step, status, detail, error, latency_ms, created_at, finished_at) "
                  "values (%s,%s,%s,%s,%s,%s,to_timestamp(%s),now())",
                  (self.run_uuid, st.name, st.status, json.dumps(st.detail, default=str), st.error, latency_ms, started))
        self._emit({"type": "stage", "stage": st.name, "status": st.status, "latency_ms": latency_ms, "detail": st.detail, "error": st.error})

    def _emit(self, event: dict) -> None:
        if self.on_event:
            self.on_event(event)

    def _sql(self, sql: str, params: tuple) -> None:
        self._q.put((sql, params))

    def _drain(self, connect: Callable, user) -> None:
        """Writes queued rows in order. Rows queued together go as one simple-query message (one network round trip; the
        database is ~110 ms away): BEGIN, the run owner's role and claims (SET LOCAL, so RLS checks every insert), the
        rows, COMMIT. Connections are reused across runs; SET LOCAL leaves nothing behind for the next run."""
        try:
            conn = _take_connection(connect)
        except Exception as e:
            self._errors.append(f"trace connection failed: {type(e).__name__}")
            conn = None
        prefix, done = None, False
        while not done:
            batch = [self._q.get()]
            while True:  # everything already queued joins this round trip
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if batch[-1] is None:
                done, batch = True, batch[:-1]
            if conn is None or not batch:
                continue
            cur, stmts = psycopg.ClientCursor(conn), []
            for sql, params in batch:
                try:
                    stmts.append(cur.mogrify(sql, params))
                except Exception as e:  # a row that cannot be bound is reported, the others are written
                    self._errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            if not stmts:
                continue
            try:
                if prefix is None:
                    prefix = (f"set local role authenticated; select set_config('request.jwt.claims', {cur.mogrify('%s', (rls_claims(user),))}, true); "
                              if user is not None else "")
                try:
                    cur.execute(f"begin; {prefix}{'; '.join(stmts)}; commit")
                except psycopg.OperationalError:  # a pooled connection the server closed: one fresh connection, one retry
                    conn.close()
                    conn = connect(autocommit=True)
                    cur = psycopg.ClientCursor(conn)
                    cur.execute(f"begin; {prefix}{'; '.join(stmts)}; commit")
            except psycopg.OperationalError as e:  # the database is unreachable: keep the error, stop writing
                self._errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                conn = None
            except psycopg.Error:  # one bad row must not lose the rest: end the failed batch, then row by row
                conn.rollback()
                for s in stmts:
                    try:
                        cur.execute(f"begin; {prefix}{s}; commit")
                    except psycopg.Error as e:
                        conn.rollback()
                        self._errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            except Exception as e:
                self._errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                conn = None
        if conn is not None:
            _give_connection(conn)


# Idle trace-writer connections, reused across runs: opening one costs ~0.8 s (TLS + auth to the pooler), measured
# 2026-10-09. Each run takes one for its writer thread and gives it back when its trace is flushed.
_POOL: list = []
_POOL_LOCK = threading.Lock()
POOL_SIZE = 4


def _take_connection(connect: Callable):
    with _POOL_LOCK:
        while _POOL:
            conn = _POOL.pop()
            if not conn.closed:
                return conn
    return connect(autocommit=True)


def _give_connection(conn) -> None:
    if conn.closed or conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        conn.close()
        return
    with _POOL_LOCK:
        if len(_POOL) < POOL_SIZE:
            _POOL.append(conn)
            return
    conn.close()
