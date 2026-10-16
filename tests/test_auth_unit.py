import base64
import json
from contextlib import contextmanager

import httpx
import pytest

from services.auth import supabase as auth
from services.retrieval.scope import RetrievalScope


def token(sub="u-1", aal="aal1", role="authenticated", exp=4102444800):
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"{enc({'alg': 'ES256'})}.{enc({'sub': sub, 'aal': aal, 'role': role, 'exp': exp, 'iat': 1, 'session_id': 's-1'})}.sig"


def transport(status=200, user=None):
    def handle(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/auth/v1/user" and req.headers["apikey"] == "pk" and req.headers["authorization"].startswith("Bearer ")
        return httpx.Response(status, json=user if user is not None else {"id": "u-1", "email": "a@x.io", "factors": []})
    return httpx.MockTransport(handle)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "pk")
    auth._cache.clear()


def test_missing_and_rejected_tokens():
    with pytest.raises(auth.AuthError) as e:
        auth.verify(None)
    assert e.value.status == 401
    with pytest.raises(auth.AuthError) as e:
        auth.verify(token(), transport(status=403))  # revoked / expired session as judged by Supabase Auth
    assert e.value.code == "invalid_session"


def test_claims_must_match_the_user_supabase_returns():
    with pytest.raises(auth.AuthError):
        auth.verify(token(sub="someone-else"), transport())
    with pytest.raises(auth.AuthError):
        auth.verify(token(role="anon"), transport())


def test_mfa_state_and_assurance_level():
    verified = {"id": "u-1", "email": "a@x.io", "factors": [{"id": "f", "status": "verified", "factor_type": "totp"}]}
    u = auth.verify(token(aal="aal1"), transport(user=verified))
    assert u.mfa_enabled and u.needs_mfa
    auth._cache.clear()
    assert not auth.verify(token(aal="aal2"), transport(user=verified)).needs_mfa
    auth._cache.clear()
    assert not auth.verify(token(aal="aal1"), transport()).needs_mfa  # no verified factor: aal1 is enough


def test_verification_is_cached_briefly():
    t = token()
    auth.verify(t, transport())
    assert auth.verify(t, transport(status=500)).id == "u-1"  # served from the 30 s cache
    auth.forget(t)
    with pytest.raises(auth.AuthError):
        auth.verify(t, transport(status=500))


def test_user_tx_runs_as_authenticated_with_verified_claims():
    seen = []

    class Conn:
        @contextmanager
        def transaction(self):
            seen.append("BEGIN")
            yield
            seen.append("COMMIT")

        def execute(self, sql, params=None):
            seen.append((sql, params))

    u = auth.verify(token(), transport())
    with auth.user_tx(Conn(), u):
        pass
    assert seen[0] == "BEGIN" and seen[1][0] == "set local role authenticated"
    assert json.loads(seen[2][1][0])["sub"] == "u-1" and seen[-1] == "COMMIT"


def test_scope_has_no_default():
    with pytest.raises(TypeError):
        RetrievalScope()  # every retrieval call must state whose data it may read
    assert RetrievalScope(None).user_id is None
