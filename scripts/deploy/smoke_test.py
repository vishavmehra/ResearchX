"""Targeted smoke test of a deployed ResearchX: health, sign-in, a protected API call, one streamed research run with
citations and its Research Trace, cross-user isolation, and an abstention. Read-only apart from the runs it starts.

Credentials come from the environment and are never printed:
    RX_SMOKE_EMAIL / RX_SMOKE_PASSWORD       the account that runs the questions
    RX_SMOKE_EMAIL_2 / RX_SMOKE_PASSWORD_2   optional second account for the isolation check
    SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY  (or NEXT_PUBLIC_*), from .env when present
Usage:
    python scripts/deploy/smoke_test.py https://<web>.vercel.app [--question "..."] [--skip-abstention]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402

FLAGSHIP = "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?"
INVENTED = "What learning rate was used to train the ZorbleNet-9 model on the Quuxbench-3D dataset?"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}", flush=True)
    return ok


def sign_in(email: str, password: str) -> str:
    url = (os.environ.get("SUPABASE_URL") or os.environ["NEXT_PUBLIC_SUPABASE_URL"]).rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
    r = httpx.post(f"{url}/auth/v1/token?grant_type=password", headers={"apikey": key}, json={"email": email, "password": password},
                   timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def research(base: str, token: str, question: str) -> tuple[dict | None, int, float, float]:
    """Streamed run: (result, stage events seen, seconds to the first event, total seconds)."""
    t0, first, events, result = time.time(), None, 0, None
    with httpx.stream("POST", f"{base}/api/research", headers={"Authorization": f"Bearer {token}"},
                      json={"query": question, "stream": True}, timeout=httpx.Timeout(240, connect=30)) as r:
        r.raise_for_status()
        kind = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
                first = first or time.time() - t0
            elif line.startswith("data: ") and kind:
                if kind == "result":
                    result = json.loads(line[6:])
                elif kind == "error":
                    result = {"status": "ERROR"}
                else:
                    events += 1
    return result, events, first or 0.0, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("--question", default=FLAGSHIP)
    ap.add_argument("--skip-abstention", action="store_true")
    a = ap.parse_args()
    load_env()
    base = a.base.rstrip("/")

    r = httpx.get(f"{base}/api/health", timeout=90)
    h = r.json() if r.status_code == 200 else {}
    check("health through the web origin", r.status_code == 200 and h.get("database") == "ok",
          f"HTTP {r.status_code}, database {h.get('database')}, bm25 {h.get('bm25_artifact')}, {h.get('database_mb')} MB")
    r = httpx.get(f"{base}/", timeout=60, follow_redirects=False)
    check("signed-out page request goes to /login", r.status_code in (302, 303, 307) and "/login" in r.headers.get("location", ""),
          f"HTTP {r.status_code}")

    token = sign_in(os.environ["RX_SMOKE_EMAIL"], os.environ["RX_SMOKE_PASSWORD"])
    check("sign-in", bool(token))
    r = httpx.get(f"{base}/api/me", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    check("authenticated API call (/api/me)", r.status_code == 200 and r.json().get("email") == os.environ["RX_SMOKE_EMAIL"],
          f"HTTP {r.status_code}")
    r = httpx.get(f"{base}/api/me", timeout=60)
    check("API rejects a request without a token", r.status_code == 401, f"HTTP {r.status_code}")

    res, n, first, total = research(base, token, a.question)
    status = (res or {}).get("status")
    cited = len((res or {}).get("cited_sources") or [])
    check("research run completes with citations", status in ("ANSWERED", "PARTIAL", "CONFLICTING") and cited > 0,
          f"{status}, {cited} cited source(s), {len((res or {}).get('claims') or [])} claim(s), cached={(res or {}).get('cached')}, "
          f"{n} stage events, first event {first:.1f}s, total {total:.1f}s")
    run_id = (res or {}).get("research_run_id")
    if run_id:
        r = httpx.get(f"{base}/api/research/{run_id}/trace", headers={"Authorization": f"Bearer {token}"}, timeout=60)
        body = r.json() if r.status_code == 200 else {}
        steps = [s.get("stage") for s in body.get("stages", [])]
        check("Research Trace of the run", r.status_code == 200 and len(steps) > 0,
              f"HTTP {r.status_code}, {len(steps)} stage rows, cache_lookup {next((s.get('status') for s in body.get('stages', []) if s.get('stage') == 'cache_lookup'), '-')}")
        if os.environ.get("RX_SMOKE_EMAIL_2"):
            other = sign_in(os.environ["RX_SMOKE_EMAIL_2"], os.environ["RX_SMOKE_PASSWORD_2"])
            r = httpx.get(f"{base}/api/research/{run_id}/trace", headers={"Authorization": f"Bearer {other}"}, timeout=60)
            check("another user cannot read this run's trace", r.status_code in (403, 404), f"HTTP {r.status_code}")
            r = httpx.get(f"{base}/api/conversations", headers={"Authorization": f"Bearer {other}"}, timeout=60)
            body = r.json() if r.status_code == 200 else {}
            items = body.get("conversations", body.get("items", [])) if isinstance(body, dict) else body
            mine = {c.get("id") for c in items if isinstance(c, dict)}
            check("another user's conversation list excludes this run's chat",
                  r.status_code == 200 and (res or {}).get("conversation_id") not in mine, f"HTTP {r.status_code}")
    if not a.skip_abstention:
        res, n, first, total = research(base, token, INVENTED)
        check("invented entities lead to an abstention", (res or {}).get("status") == "ABSTAINED",
              f"{(res or {}).get('status')}, total {total:.1f}s")
    failed = [x for x in results if not x[1]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
