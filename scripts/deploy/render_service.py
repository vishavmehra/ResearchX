"""Create and operate the ResearchX API service on Render through the Render API, from the local environment.

Secrets are read from the environment (or .env, via services.env.load_env) and never printed. Render has no outbound
IPv6, while a Supabase project's direct database host is IPv6-only, so DATABASE_URL is rewritten to the project's
Supavisor session pooler (IPv4, session mode keeps per-connection role and settings) when --pooler-host is given.

Usage (RENDER_API_KEY must be set):
    python scripts/deploy/render_service.py create --pooler-host aws-0-ap-northeast-1.pooler.supabase.com [--plan free]
    python scripts/deploy/render_service.py status
    python scripts/deploy/render_service.py set-env RESEARCHX_CORS_ORIGINS https://<web>.vercel.app
    python scripts/deploy/render_service.py deploy
    python scripts/deploy/render_service.py logs [--limit 60]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402

API = "https://api.render.com/v1"
BUILD = "pip install uv && uv sync --frozen && uv run python scripts/retrieval/build_bm25.py --release 0.2.0"
START = "uv run uvicorn services.api.app:app --host 0.0.0.0 --port $PORT"
# API variables copied from the local environment when set (names only are ever printed)
PASSTHROUGH = ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "NEBIUS_API_KEY", "TAVILY_API_KEY", "S2_API_KEY", "CORE_API_KEY",
               "OPENALEX_API_KEY", "RESEARCHX_CONTACT_EMAIL", "UNPAYWALL_EMAIL", "RESEARCHX_CORS_ORIGINS")


def headers() -> dict:
    key = os.environ.get("RENDER_API_KEY")
    if not key:
        sys.exit("RENDER_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"}


def database_url(pooler_host: str | None) -> str:
    url = os.environ["DATABASE_URL"]
    if not pooler_host:
        return url
    u = urlsplit(url)
    ref = u.hostname.split(".")[1]  # db.<project-ref>.supabase.co
    return f"postgresql://postgres.{ref}:{quote(unquote(u.password or ''), safe='')}@{pooler_host}:5432/postgres?sslmode=require"


def find(name: str) -> dict | None:
    r = httpx.get(f"{API}/services", headers=headers(), params={"name": name, "limit": 20}, timeout=30)
    r.raise_for_status()
    return next((x["service"] for x in r.json() if x["service"]["name"] == name), None)


def describe(s: dict) -> None:
    d = s.get("serviceDetails", {})
    print(json.dumps({"id": s["id"], "name": s["name"], "url": d.get("url"), "plan": d.get("plan"), "region": d.get("region"),
                      "branch": s.get("branch"), "autoDeploy": s.get("autoDeploy"), "healthCheckPath": d.get("healthCheckPath"),
                      "suspended": s.get("suspended")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["create", "status", "set-env", "deploy", "logs"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--name", default="researchx-api")
    ap.add_argument("--repo", default="https://github.com/vishavmehra/ResearchX")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--plan", default="free", help="free, starter, standard, ... (a paid plan bills the Render workspace)")
    ap.add_argument("--region", default="singapore")
    ap.add_argument("--pooler-host", help="Supavisor host, e.g. aws-0-ap-northeast-1.pooler.supabase.com")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    load_env()
    H = headers()

    if a.command == "create":
        if (s := find(a.name)) is not None:
            print("service exists; not created again")
            describe(s)
            return
        owner = httpx.get(f"{API}/owners", headers=H, params={"limit": 20}, timeout=30).json()[0]["owner"]
        env_vars = [{"key": "DATABASE_URL", "value": database_url(a.pooler_host)}, {"key": "RESEARCHX_LAYA", "value": "0"}]
        for k in PASSTHROUGH:
            if os.environ.get(k):
                env_vars.append({"key": k, "value": os.environ[k]})
            else:
                print(f"not set locally, skipped: {k}")
        body = {"type": "web_service", "name": a.name, "ownerId": owner["id"], "repo": a.repo, "branch": a.branch, "autoDeploy": "no",
                "envVars": env_vars,
                "serviceDetails": {"runtime": "python", "plan": a.plan, "region": a.region, "healthCheckPath": "/api/health",
                                   "envSpecificDetails": {"buildCommand": BUILD, "startCommand": START}}}
        r = httpx.post(f"{API}/services", headers=H, json=body, timeout=60)
        if r.status_code >= 300:
            sys.exit(f"create failed: {r.status_code} {r.text[:500]}")
        describe(r.json().get("service", r.json()))
        return

    s = find(a.name)
    if s is None:
        sys.exit(f"no Render service named {a.name}")
    if a.command == "status":
        describe(s)
        r = httpx.get(f"{API}/services/{s['id']}/deploys", headers=H, params={"limit": 3}, timeout=30)
        for row in r.json():
            d = row["deploy"]
            print("deploy", d["id"], d["status"], (d.get("commit") or {}).get("id", "")[:7], d.get("finishedAt") or d.get("createdAt"))
    elif a.command == "set-env":
        key, value = a.args[0], a.args[1]
        r = httpx.put(f"{API}/services/{s['id']}/env-vars/{key}", headers=H, json={"value": value}, timeout=30)
        print("set", key, r.status_code, "(takes effect with the next deploy)")
    elif a.command == "deploy":
        r = httpx.post(f"{API}/services/{s['id']}/deploys", headers=H, json={"clearCache": "do_not_clear"}, timeout=30)
        print("deploy", r.status_code, r.json().get("id"), r.json().get("status"))
    elif a.command == "logs":
        r = httpx.get(f"{API}/logs", headers=H, params={"ownerId": s["ownerId"], "resource": s["id"], "limit": a.limit,
                                                         "direction": "backward"}, timeout=30)
        r.raise_for_status()
        secrets = [v for k, v in os.environ.items() if len(v) >= 12 and any(t in k for t in ("KEY", "TOKEN", "PASSWORD", "DATABASE_URL"))]
        for row in reversed(r.json().get("logs", [])):
            msg = row.get("message", "")
            for v in secrets:  # never echo a secret, even if a log line contains one
                msg = msg.replace(v, "<redacted>")
            sys.stdout.buffer.write(f"{row.get('timestamp', '')[11:19]} {msg[:240]}\n".encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
