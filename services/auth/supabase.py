"""Supabase Auth on the backend (docs/SECURITY.md).

The browser signs in with Supabase Auth (publishable key only) and calls the ResearchX API with its access token as
`Authorization: Bearer`. verify() asks Supabase Auth itself (GET /auth/v1/user) whether that token is valid, so
expired, forged and signed-out (revoked-session) tokens are rejected without a JWT library or a signing secret on our
side. The claims (sub, aal, session_id, iat) are read from the token only after Supabase has accepted it; acceptance is
cached for at most 30 s. user_tx() then runs SQL as role `authenticated` with exactly those claims, so PostgreSQL RLS
decides what each user can read or write. Identity never comes from request bodies, and no service-role key is used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx

CACHE_TTL_S = 30
_cache: OrderedDict[str, tuple[float, "AuthUser"]] = OrderedDict()
_lock = threading.Lock()


class AuthError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None
    aal: str                     # aal1 = password session; aal2 = MFA verified
    session_id: str | None
    issued_at: int
    mfa_enabled: bool            # the user has at least one verified MFA factor
    claims: dict = field(repr=False)
    token: str = field(repr=False)

    @property
    def needs_mfa(self) -> bool:
        return self.mfa_enabled and self.aal != "aal2"


def claims_of(token: str) -> dict:
    """Payload of a JWT. Not a verification: only call it on a token Supabase Auth has accepted."""
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError) as e:
        raise AuthError(401, "invalid_token", "Invalid access token.") from e


def verify(token: str | None, transport: httpx.BaseTransport | None = None) -> AuthUser:
    if not token:
        raise AuthError(401, "missing_token", "Sign in to continue.")
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    url, apikey = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not apikey:
        raise AuthError(503, "auth_unconfigured", "Authentication is not configured on the server.")
    try:
        with httpx.Client(transport=transport, timeout=10) as c:
            r = c.get(f"{url}/auth/v1/user", headers={"apikey": apikey, "Authorization": f"Bearer {token}"})
    except httpx.HTTPError as e:
        raise AuthError(503, "auth_unavailable", "Authentication service unavailable. Try again.") from e
    if r.status_code in (401, 403):
        raise AuthError(401, "invalid_session", "Your session has expired. Sign in again.")
    if r.status_code != 200:
        raise AuthError(503, "auth_unavailable", "Authentication service unavailable. Try again.")
    profile, claims = r.json(), claims_of(token)
    if claims.get("sub") != profile.get("id") or claims.get("role") != "authenticated":
        raise AuthError(401, "invalid_token", "Invalid access token.")
    user = AuthUser(id=profile["id"], email=profile.get("email"), aal=claims.get("aal", "aal1"), session_id=claims.get("session_id"),
                    issued_at=int(claims.get("iat", 0)), claims=claims, token=token,
                    mfa_enabled=any(f.get("status") == "verified" for f in profile.get("factors") or []))
    with _lock:
        _cache[key] = (min(now + CACHE_TTL_S, float(claims.get("exp", now))), user)
        while len(_cache) > 2000:
            _cache.popitem(last=False)
    return user


def forget(token: str) -> None:
    """Drop a cached verification (after sign-out or MFA changes)."""
    with _lock:
        _cache.pop(hashlib.sha256(token.encode()).hexdigest(), None)


def rls_claims(user: AuthUser) -> str:
    """The verified claims PostgreSQL sees as auth.jwt() / auth.uid()."""
    return json.dumps(user.claims)


@contextmanager
def user_tx(conn, user: AuthUser):
    """One transaction that runs as role `authenticated` with the user's verified claims: RLS applies to every
    statement. SET LOCAL ends with the transaction, so a shared connection never carries one user's role to the next."""
    with conn.transaction():
        conn.execute("set local role authenticated")
        conn.execute("select set_config('request.jwt.claims', %s, true)", (rls_claims(user),))
        yield conn
