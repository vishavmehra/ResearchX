"""Knowledge Map integration tests against the real Supabase project, through the real API (in-process).

Opt-in (network + database + model calls):   RX_INTEGRATION=1 uv run pytest -q tests/test_knowledge_graph_integration.py
Two throwaway users are created (deleting them cascades to every row they own). One research run is stored for user A
with a runtime graph built by the real builder on a synthetic external source; one synthetic runtime-source row is
created for the promotion test and deleted at the end. The corpus itself is only read.
"""

import json
import os
import secrets
import uuid

import httpx
import pytest

from services.env import load_env

load_env()
pytestmark = pytest.mark.skipif(os.environ.get("RX_INTEGRATION") != "1", reason="set RX_INTEGRATION=1 to run against Supabase")

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
H = {"apikey": os.environ.get("SUPABASE_PUBLISHABLE_KEY", ""), "Content-Type": "application/json"}
SUFFICIENT = "What learning rate was used to train HRNetV2 + OCR on ADE20K?"  # answered from the corpus (grounding eval)
LABELS = {"ALIGNED_EXPERIMENT", "MATERIALLY_DIFFERENT", "RELATED_CONTEXT", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}


def sign_in(email: str, password: str) -> dict:
    r = httpx.post(f"{URL}/auth/v1/token?grant_type=password", headers=H, json={"email": email, "password": password}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()


def bearer(user: dict) -> dict:
    return {"Authorization": f"Bearer {user['token']}"}


@pytest.fixture(scope="module")
def admin():
    from services.storage.db import connect
    conn, created = connect(autocommit=True), []

    def make_user() -> dict:
        uid, email, pw = str(uuid.uuid4()), f"rx-test-{secrets.token_hex(5)}@mailinator.com", secrets.token_urlsafe(18) + "Aa1!"
        conn.execute("insert into auth.users (instance_id, id, aud, role, email, encrypted_password, email_confirmed_at, raw_app_meta_data, "
                     "raw_user_meta_data, created_at, updated_at, confirmation_token, recovery_token, email_change_token_new, email_change) "
                     "values ('00000000-0000-0000-0000-000000000000', %s, 'authenticated', 'authenticated', %s, "
                     "extensions.crypt(%s, extensions.gen_salt('bf')), now(), '{\"provider\":\"email\",\"providers\":[\"email\"]}', '{}', "
                     "now(), now(), '', '', '', '')", (uid, email, pw))
        conn.execute("insert into auth.identities (user_id, provider_id, identity_data, provider, last_sign_in_at, created_at, updated_at) "
                     "values (%s, %s, %s, 'email', now(), now(), now())", (uid, uid, json.dumps({"sub": uid, "email": email, "email_verified": True})))
        created.append(uid)
        return {"id": uid, "token": sign_in(email, pw)["access_token"]}

    yield conn, make_user
    for uid in created:
        conn.execute("delete from auth.users where id = %s", (uid,))
    conn.close()


@pytest.fixture(scope="module")
def api():
    from fastapi.testclient import TestClient

    from services.api.app import app
    return TestClient(app)


@pytest.fixture(scope="module")
def ab(admin):
    return admin[1](), admin[1]()


@pytest.fixture(scope="module")
def stored(admin, ab):
    """A research run of user A whose graph holds an external source with validated facts and a private document. The
    runtime experiment uses a real corpus benchmark, so identity resolution and comparisons hit real data."""
    from services.graph import runtime as rg
    from services.runtime.orchestrator import Orchestrator
    from services.runtime.trace import new_run_id

    conn = admin[0]
    xid, bench = conn.execute("select id, record->'evaluation'->'benchmark'->>'raw_value' from rx.experiments "
                              "where record->'evaluation'->'benchmark'->>'status' = 'KNOWN' order by id limit 1").fetchone()
    token = secrets.token_hex(4)
    url, sha = f"https://example.org/rx-test/{token}", secrets.token_hex(32)
    ext = {"sid": "S1", "kind": "external", "chunk_id": url, "provider": "openalex", "url": url, "title": f"Runtime test source {token}",
           "doi": f"10.5555/rx-test-{token}", "arxiv_id": None, "year": 2026, "retrieved_at": "2026-10-08T05:00:00+00:00", "content_sha256": sha,
           "reliability": "scholarly", "source_id": f"test:{token}", "also_found_by": [], "role": "SUPPORTS", "text": "a learning rate of 0.02"}
    doc = {"sid": "S2", "kind": "private", "chunk_id": "private:1", "document_id": str(uuid.uuid4()), "title": f"Private notes {token}",
           "text": "notes", "role": "RELATED"}
    fact = {"field": "training.learning_rate", "label": "learning rate", "value": "0.02", "normalized": 0.02, "span": "a learning rate of 0.02",
            "chunk_id": url, "sid": "S1", "evidence_id": None, "experiment_ids": [], "origin": "runtime extraction", "config": "EXACT",
            "config_detail": "", "dataset_ok": True, "start": None, "end": None, "locator": None, "metric": None}
    rid = new_run_id()
    query = f"What learning rate was used to train RX-Test-{token} on {bench}?"
    result = {"status": "ANSWERED", "sources": [ext, doc], "cited_sources": ["S1"], "claims": [], "contradictions": []}
    inputs = {"query": query, "intent": "protocol", "asked_unknown": True, "request": {"model": f"RX-Test-{token}", "dataset": bench, "metric": None},
              "external": [ext], "facts": [fact], "rejected": [], "conflicts": [], "result_ids": {}}
    graph = rg.build(rid, result, inputs, Orchestrator().q)
    conn.execute("insert into rx.research_runs (research_run_id, user_id, query, normalized_query, dataset_release, status, mode, answer) "
                 "values (%s, %s, %s, %s, '0.2.0', 'ANSWERED', 'pipeline', %s)", (rid, ab[0]["id"], query, query.lower(), json.dumps({"graph": graph})))
    conn.execute("insert into rx.runtime_sources (url, provider, source_type, title, doi, content_sha256, reliability, validation) "
                 "values (%s, 'openalex', 'paper', %s, %s, %s, '{\"tier\": \"scholarly\"}', '{}')", (url, ext["title"], ext["doi"], sha))
    rt = next(n["id"] for n in graph["nodes"] if n["id"].startswith("rt:exp:"))
    yield {"rid": rid, "graph": graph, "rt_exp": rt, "corpus_exp": xid, "ext": f"rt:paper:doi:10.5555/rx-test-{token}",
           "doc": f"doc:{doc['document_id']}", "token": token, "url": url}
    conn.execute("delete from rx.runtime_sources where url = %s", (url,))


# ------------------------------------------------------------------------------------------------ access
def test_every_graph_route_requires_a_session(api):
    for path in ("/api/graph/stats", "/api/graph/search?q=ab", "/api/graph/landscape", "/api/graph/neighborhood?node=paper:x", "/api/graph/node?id=paper:x",
                 "/api/graph/edge?type=CITES&source=paper:a&target=paper:b", "/api/graph/diff?a=exp:a&b=exp:b", "/api/graph/ask?node=paper:x&q=experiments",
                 "/api/graph/run/rx_01ABCDEFGHJKMNPQRSTVWXYZ01", "/api/graph/runs", "/api/graph/discoveries"):
        assert api.get(path).status_code == 401, path
    assert api.post("/api/graph/promotions", json={"research_run_id": "rx_01ABCDEFGHJKMNPQRSTVWXYZ01", "node_id": "rt:x"}).status_code == 401


# ------------------------------------------------------------------------------------------------ canonical graph on real data
def test_the_canonical_graph_is_the_loaded_corpus(api, ab, admin):
    from services.canonicalization.entities import entity_id
    from services.datasets.schemas import CanonicalEntityType as E

    conn, a = admin[0], ab[0]
    experiments, papers, contradictions = conn.execute(
        "select (select count(*) from rx.experiments), (select count(*) from rx.papers), (select count(*) from rx.contradictions)").fetchone()
    s = api.get("/api/graph/stats", headers=bearer(a)).json()
    assert (s["experiments"], s["papers"], s["contradictions"]) == (experiments, papers, contradictions)
    raw, = conn.execute("select record->'evaluation'->'benchmark'->>'raw_value' from rx.experiments "
                        "where record->'evaluation'->'benchmark'->>'status' = 'KNOWN' order by id limit 1").fetchone()
    eid = entity_id(E.BENCHMARK, raw)
    name, = conn.execute("select preferred_name from rx.canonical_entities where id = %s", (eid,)).fetchone()
    assert eid in {r["id"] for r in api.get("/api/graph/search", params={"q": name}, headers=bearer(a)).json()["results"]}
    nb = api.get("/api/graph/neighborhood", params={"node": eid, "limit": 10}, headers=bearer(a)).json()
    assert nb["nodes"][0]["id"] == eid and 1 <= len(nb["nodes"]) - 1 <= 10 and nb["total"] >= 1
    assert all(e["derivation"] and e["why"] for e in nb["edges"])
    assert api.get("/api/graph/neighborhood", params={"node": eid, "limit": 1000}, headers=bearer(a)).status_code == 422  # pages are bounded
    assert api.get("/api/graph/neighborhood", params={"node": "paper:does-not-exist"}, headers=bearer(a)).status_code == 404


# ------------------------------------------------------------------------------------------------ runtime graph: isolation
def test_run_graphs_and_private_nodes_are_visible_only_to_their_owner(api, ab, stored):
    a, b = ab
    mine = api.get(f"/api/graph/run/{stored['rid']}", headers=bearer(a))
    assert mine.status_code == 200
    origins = {n["id"]: n["origin"] for n in mine.json()["nodes"]}
    assert origins[stored["doc"]] == "private_document" and origins[stored["ext"]] == "runtime_external"
    assert api.get(f"/api/graph/run/{stored['rid']}", headers=bearer(b)).status_code == 404  # same answer as a missing run
    assert api.get("/api/graph/ask", params={"node": f"run:{stored['rid']}", "q": "discoveries"}, headers=bearer(b)).status_code == 404
    assert api.get("/api/graph/diff", params={"a": stored["rt_exp"], "b": stored["corpus_exp"], "run": stored["rid"]}, headers=bearer(b)).status_code == 404
    assert stored["rid"] not in {r["research_run_id"] for r in api.get("/api/graph/runs", headers=bearer(b)).json()["runs"]}
    theirs = {i["id"] for i in api.get("/api/graph/discoveries", headers=bearer(b)).json()["items"]}
    assert not theirs & {stored["ext"], stored["doc"]}
    found = {i["id"] for i in api.get("/api/graph/discoveries", headers=bearer(a)).json()["items"]}
    assert {stored["ext"], stored["doc"]} <= found
    assert not api.get("/api/graph/search", params={"q": f"Private notes {stored['token']}"}, headers=bearer(a)).json()["results"]  # never in the corpus index


# ------------------------------------------------------------------------------------------------ runtime graph: diff, ask, promotion
def test_a_runtime_experiment_connects_to_diff_and_ask_the_map(api, ab, stored):
    a = ab[0]
    diff = api.get("/api/graph/diff", params={"a": stored["rt_exp"], "b": stored["corpus_exp"], "run": stored["rid"]}, headers=bearer(a)).json()
    assert diff["label"] in LABELS and any(r["field"] == "training.learning_rate" and r["value_a"] == 0.02 for r in diff["fields"])
    closest = api.get("/api/graph/ask", params={"node": stored["rt_exp"], "q": "closest", "run": stored["rid"]}, headers=bearer(a)).json()
    assert closest["items"] and all(i["id"].startswith("exp:") and i["relation"] in LABELS for i in closest["items"])
    found = api.get("/api/graph/ask", params={"node": f"run:{stored['rid']}", "q": "discoveries"}, headers=bearer(a)).json()
    assert stored["ext"] in {i["id"] for i in found["items"]}


def test_promotion_is_explicit_reviewed_and_never_touches_the_corpus(api, ab, admin, stored):
    conn, (a, b) = admin[0], ab
    before = conn.execute("select (select count(*) from rx.papers), (select count(*) from rx.experiments)").fetchone()
    assert api.post("/api/graph/promotions", json={"research_run_id": stored["rid"], "node_id": stored["ext"]}, headers=bearer(b)).status_code == 404
    private = api.post("/api/graph/promotions", json={"research_run_id": stored["rid"], "node_id": stored["doc"]}, headers=bearer(a)).json()
    assert private["state"] == "NOT_ELIGIBLE"
    r = api.post("/api/graph/promotions", json={"research_run_id": stored["rid"], "node_id": stored["ext"]}, headers=bearer(a))
    assert r.status_code == 200 and r.json()["state"] == "CANDIDATE", r.text
    state, note = conn.execute("select state, validation->'promotion' from rx.runtime_sources where url = %s", (stored["url"],)).fetchone()
    assert state == "CANDIDATE" and note["checks"]
    assert conn.execute("select (select count(*) from rx.papers), (select count(*) from rx.experiments)").fetchone() == before
    again = api.get(f"/api/graph/run/{stored['rid']}", headers=bearer(a)).json()
    assert next(n for n in again["nodes"] if n["id"] == stored["ext"])["meta"]["review_state"] == "CANDIDATE"


# ------------------------------------------------------------------------------------------------ live runs: gating and grounding
def test_external_research_runs_only_when_internal_evidence_is_insufficient():
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator

    o = Orchestrator()
    ok = o.run(SUFFICIENT, RetrievalScope(None))
    stages = {s["stage"]: s for s in ok["stages"]}
    assert stages["external_sources"]["status"] == "not_required"
    assert stages["graph_enrichment"]["status"] == "completed" and ok["graph"]["stats"]["by_origin"].get("runtime_external", 0) == 0
    assert ok["claims"] and all(c["verified"] for c in ok["claims"]), "the answer still comes only from verified citations"
    assert ok["trace_persisted"], "every stage row, graph_enrichment included, is accepted by the trace store"

    gap = o.run(f"What learning rate was used to train QX-{secrets.token_hex(3)} on an unpublished benchmark?", RetrievalScope(None))
    stages = {s["stage"]: s for s in gap["stages"]}
    assert stages["external_sources"]["status"] in ("completed", "failed")  # attempted, whatever the providers did
    assert gap["status"] == "ABSTAINED" and not gap["claims"], "no value is reported without validated evidence"
    assert stages["graph_enrichment"]["status"] == "completed"
    assert all(n["status"] != "validated" for n in gap["graph"]["nodes"] if n["origin"] == "runtime_external")
