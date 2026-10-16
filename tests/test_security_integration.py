"""Security integration tests against the real Supabase project: authentication, RLS isolation, IDOR, MFA (TOTP),
answer-cache isolation, private-document isolation and account deletion, through the real API (in-process).

Opt-in (network + database + model calls):   RX_INTEGRATION=1 uv run pytest -q tests/test_security_integration.py
Test users are created as already-confirmed accounts with the backend's admin database credential (the built-in mailer
allows only a few e-mails per hour), then signed in through the real Supabase Auth API. Deleting the auth users at the
end cascades to every rx row they own.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import time
import uuid

import httpx
import pytest

from services.env import load_env

load_env()
pytestmark = pytest.mark.skipif(os.environ.get("RX_INTEGRATION") != "1", reason="set RX_INTEGRATION=1 to run against Supabase")

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
H = {"apikey": os.environ.get("SUPABASE_PUBLISHABLE_KEY", ""), "Content-Type": "application/json"}
QUESTION = "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?"


def totp(secret: str, at: float | None = None) -> str:
    """RFC 6238 code for the test authenticator (production verification is Supabase's)."""
    key = base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))
    digest = hmac.new(key, struct.pack(">Q", int((at or time.time()) // 30)), hashlib.sha1).digest()
    o = digest[-1] & 15
    return str((struct.unpack(">I", digest[o:o + 4])[0] & 0x7FFFFFFF) % 1_000_000).zfill(6)


def sign_in(email: str, password: str) -> dict:
    r = httpx.post(f"{URL}/auth/v1/token?grant_type=password", headers=H, json={"email": email, "password": password}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()


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
        s = sign_in(email, pw)
        return {"id": uid, "email": email, "password": pw, "token": s["access_token"], "refresh": s["refresh_token"]}

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


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------------------------------- authentication
def test_invalid_credentials_get_a_generic_error(ab):
    r = httpx.post(f"{URL}/auth/v1/token?grant_type=password", headers=H, json={"email": ab[0]["email"], "password": "wrong-password"}, timeout=30)
    unknown = httpx.post(f"{URL}/auth/v1/token?grant_type=password", headers=H, json={"email": "nobody-" + ab[0]["email"], "password": "x"}, timeout=30)
    assert r.status_code == unknown.status_code == 400
    assert r.json().get("error_code") == unknown.json().get("error_code") == "invalid_credentials"  # no account-existence oracle


def test_api_requires_a_valid_session(api):
    assert api.get("/api/conversations").status_code == 401
    assert api.get("/api/conversations", headers=bearer("not-a-jwt")).status_code == 401
    assert api.get("/api/health").status_code == 200  # the only public route; it carries no user data


def test_session_restores_with_refresh_token_and_dies_on_global_logout(admin, api):
    from services.auth import supabase as auth
    u = admin[1]()
    r = httpx.post(f"{URL}/auth/v1/token?grant_type=refresh_token", headers=H, json={"refresh_token": u["refresh"]}, timeout=30)
    assert r.status_code == 200
    fresh = r.json()
    assert api.get("/api/me", headers=bearer(fresh["access_token"])).json()["email"] == u["email"]
    out = httpx.post(f"{URL}/auth/v1/logout?scope=global", headers={**H, **bearer(fresh["access_token"])}, timeout=30)
    assert out.status_code == 204
    again = httpx.post(f"{URL}/auth/v1/token?grant_type=refresh_token", headers=H, json={"refresh_token": fresh["refresh_token"]}, timeout=30)
    assert again.status_code in (400, 401)
    auth.forget(fresh["access_token"])
    assert api.get("/api/me", headers=bearer(fresh["access_token"])).status_code == 401  # revoked session rejected by Supabase Auth


# ------------------------------------------------------------------------------------------- isolation and IDOR
def test_users_see_only_their_own_conversations_and_cannot_touch_others(api, ab):
    a, b = ab
    ca = api.post("/api/conversations", json={"title": "Chat A"}, headers=bearer(a["token"])).json()["id"]
    cb = api.post("/api/conversations", json={"title": "Chat B"}, headers=bearer(b["token"])).json()["id"]
    titles_a = [c["title"] for c in api.get("/api/conversations", headers=bearer(a["token"])).json()["conversations"]]
    titles_b = [c["title"] for c in api.get("/api/conversations", headers=bearer(b["token"])).json()["conversations"]]
    assert "Chat A" in titles_a and "Chat B" not in titles_a
    assert "Chat B" in titles_b and "Chat A" not in titles_b
    for method, path, body in (("get", f"/api/conversations/{cb}/messages", None), ("patch", f"/api/conversations/{cb}", {"title": "pwned"}),
                               ("delete", f"/api/conversations/{cb}", None)):
        r = getattr(api, method)(path, headers=bearer(a["token"]), **({"json": body} if body else {}))
        assert r.status_code == 404, (method, path, r.status_code)
    assert api.get(f"/api/conversations/{cb}/messages", headers=bearer(b["token"])).status_code == 200  # still B's, untouched
    assert api.get(f"/api/conversations/{ca}/messages", headers=bearer(b["token"])).status_code == 404
    renamed = api.patch(f"/api/conversations/{ca}", json={"title": "Chat A renamed"}, headers=bearer(a["token"]))
    assert renamed.status_code == 200
    assert api.get("/api/conversations?q=renamed", headers=bearer(a["token"])).json()["conversations"][0]["id"] == ca


def test_forged_user_id_in_bodies_is_rejected(api, ab):
    a, b = ab
    assert api.post("/api/conversations", json={"title": "x", "user_id": b["id"]}, headers=bearer(a["token"])).status_code == 422
    assert api.post("/api/research", json={"query": "q", "user_id": b["id"]}, headers=bearer(a["token"])).status_code == 422
    assert api.patch("/api/me", json={"display_name": "x", "id": b["id"]}, headers=bearer(a["token"])).status_code == 422


def test_rls_blocks_cross_user_access_even_with_handwritten_sql(admin, ab):
    from services.auth.supabase import user_tx, verify
    from services.storage.db import connect
    a, b = ab
    ua = verify(a["token"])
    with connect(autocommit=True) as conn:
        with user_tx(conn, ua) as c:  # "forget" the owner filter on purpose: RLS must still hide B's rows
            owners = {r[0] for r in c.execute("select user_id::text from rx.conversations").fetchall()}
            assert owners == {a["id"]}
            assert c.execute("update rx.conversations set title = 'pwned' where user_id = %s returning id", (b["id"],)).fetchall() == []
            assert c.execute("select count(*) from rx.research_runs where user_id = %s", (b["id"],)).fetchone()[0] == 0
        with pytest.raises(Exception):
            with user_tx(conn, ua) as c:
                c.execute("insert into rx.conversations (user_id, title) values (%s, 'forged')", (b["id"],))
        with pytest.raises(Exception):  # the shared corpus is backend-only, not readable as a user either
            with user_tx(conn, ua) as c:
                c.execute("select 1 from rx.evidence limit 1")


# ------------------------------------------------------------------------------------------- research runs, traces, cache
def test_research_runs_traces_and_answer_cache_are_private(api, ab):
    a, b = ab
    rb = api.post("/api/research", json={"query": QUESTION}, headers=bearer(b["token"])).json()
    assert rb["mode"] == "pipeline" and rb["research_run_id"].startswith("rx_") and not rb["cached"]
    trace = api.get(f"/api/research/{rb['research_run_id']}/trace", headers=bearer(b["token"]))
    assert trace.status_code == 200
    stages = {s["stage"]: s["status"] for s in trace.json()["stages"]}
    assert stages["bm25"] == stages["pgvector"] == stages["synthesis"] == "completed"
    assert api.get(f"/api/research/{rb['research_run_id']}/trace", headers=bearer(a["token"])).status_code == 404
    msgs = api.get(f"/api/conversations/{rb['conversation_id']}/messages", headers=bearer(b["token"])).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["research_run_id"] == rb["research_run_id"]
    again = api.post("/api/research", json={"query": QUESTION, "conversation_id": rb["conversation_id"]}, headers=bearer(b["token"])).json()
    assert again["cached"] and again["origin_research_run_id"] == rb["research_run_id"]
    ra = api.post("/api/research", json={"query": QUESTION}, headers=bearer(a["token"])).json()
    assert not ra["cached"] and ra.get("origin_research_run_id") is None  # B's cached answer never reaches A
    hit = next(s for s in ra["stages"] if s["stage"] == "cache_lookup")
    assert hit["detail"]["hit"] is False


# ------------------------------------------------------------------------------------------- private documents
def test_private_documents_never_cross_tenants(api, ab):
    a, b = ab
    text = ("Internal memo. The zorblaxian widget reached a retrieval efficiency of 97.3 percent on the Kestrel benchmark "
            "when trained with a learning rate of 0.0042.").encode()
    up = api.post("/api/documents", content=text, headers={**bearer(a["token"]), "Content-Type": "text/plain", "X-Filename": "zorblax-memo.txt"})
    assert up.status_code == 200, up.text
    doc = up.json()["id"]
    assert [d["id"] for d in api.get("/api/documents", headers=bearer(a["token"])).json()["documents"]] == [doc]
    assert api.get("/api/documents", headers=bearer(b["token"])).json()["documents"] == []
    assert api.delete(f"/api/documents/{doc}", headers=bearer(b["token"])).status_code == 404
    q = "What retrieval efficiency did the zorblaxian widget reach on the Kestrel benchmark?"
    ra = api.post("/api/research", json={"query": q}, headers=bearer(a["token"])).json()
    assert any(s["kind"] == "private" for s in ra["sources"]), "owner should retrieve their own document"
    rb = api.post("/api/research", json={"query": q}, headers=bearer(b["token"])).json()
    assert not any(s.get("kind") == "private" for s in rb["sources"])
    scope = next(s for s in rb["stages"] if s["stage"] == "hybrid_fusion")["detail"]["private_documents"]
    assert scope in ("no user scope", {"chunks_in_scope": 0, "lexical": 0, "dense": 0}) or scope.get("chunks_in_scope") == 0
    assert api.delete(f"/api/documents/{doc}", headers=bearer(a["token"])).status_code == 200


# ------------------------------------------------------------------------------------------- MFA (TOTP)
def test_totp_mfa_enrolment_challenge_enforcement_and_removal(admin, api):
    from services.auth.supabase import user_tx, verify
    from services.storage.db import connect
    u = admin[1]()
    hdr = {**H, **bearer(u["token"])}
    api.post("/api/conversations", json={"title": "before MFA"}, headers=bearer(u["token"]))
    enrol = httpx.post(f"{URL}/auth/v1/factors", headers=hdr, json={"factor_type": "totp", "friendly_name": "pytest"}, timeout=30).json()
    fid, secret = enrol["id"], enrol["totp"]["secret"]
    assert "<svg" in enrol["totp"]["qr_code"] and enrol["totp"]["uri"].startswith("otpauth://totp/")  # raw SVG via REST (supabase-js wraps it as a data URL)
    ch = httpx.post(f"{URL}/auth/v1/factors/{fid}/challenge", headers=hdr, timeout=30).json()["id"]
    wrong = str((int(totp(secret)) + 1) % 1_000_000).zfill(6)
    assert httpx.post(f"{URL}/auth/v1/factors/{fid}/verify", headers=hdr, json={"challenge_id": ch, "code": wrong}, timeout=30).status_code in (400, 422)
    ch = httpx.post(f"{URL}/auth/v1/factors/{fid}/challenge", headers=hdr, timeout=30).json()["id"]
    ok = httpx.post(f"{URL}/auth/v1/factors/{fid}/verify", headers=hdr, json={"challenge_id": ch, "code": totp(secret)}, timeout=30)
    assert ok.status_code == 200, ok.text
    aal2 = ok.json()["access_token"]
    aal1 = sign_in(u["email"], u["password"])["access_token"]  # password-only session after MFA is enabled
    r = api.get("/api/conversations", headers=bearer(aal1))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "mfa_required"
    assert [c["title"] for c in api.get("/api/conversations", headers=bearer(aal2)).json()["conversations"]] == ["before MFA"]
    with connect(autocommit=True) as conn:  # the database enforces the same rule (restrictive aal2 policies)
        with user_tx(conn, verify(aal2)) as c:
            assert c.execute("select count(*) from rx.conversations").fetchone()[0] == 1
        u1 = verify(aal1)
        with user_tx(conn, u1) as c:
            assert c.execute("select count(*) from rx.conversations").fetchone()[0] == 0
    assert httpx.delete(f"{URL}/auth/v1/factors/{fid}", headers={**H, **bearer(aal1)}, timeout=30).status_code in (401, 403, 422)
    assert httpx.delete(f"{URL}/auth/v1/factors/{fid}", headers={**H, **bearer(aal2)}, timeout=30).status_code == 200


# ------------------------------------------------------------------------------------------- account deletion
def test_account_deletion_removes_every_owned_record(admin, api):
    conn, make_user = admin
    u = make_user()
    api.post("/api/conversations", json={"title": "to be deleted"}, headers=bearer(u["token"]))
    up = api.post("/api/documents", content=b"A private note about nothing in particular, long enough to be a passage.",
                  headers={**bearer(u["token"]), "Content-Type": "text/plain", "X-Filename": "note.txt"})
    assert up.status_code == 200, up.text
    assert api.request("DELETE", "/api/account", json={"confirm": "yes"}, headers=bearer(u["token"])).status_code == 422
    r = api.request("DELETE", "/api/account", json={"confirm": "DELETE MY ACCOUNT"}, headers=bearer(u["token"]))
    assert r.status_code == 200 and r.json()["storage_objects_removed"] == 1
    left = {t: conn.execute(f"select count(*) from rx.{t} where user_id = %s", (u["id"],)).fetchone()[0]
            for t in ("conversations", "messages", "research_runs", "cache_entries", "user_documents", "user_chunks")}
    assert set(left.values()) == {0}, left
    assert conn.execute("select count(*) from auth.users where id = %s", (u["id"],)).fetchone()[0] == 0
    assert conn.execute("select count(*) from rx.profiles where id = %s", (u["id"],)).fetchone()[0] == 0
    assert conn.execute("select count(*) from storage.objects where name like %s", (u["id"] + "/%",)).fetchone()[0] == 0
