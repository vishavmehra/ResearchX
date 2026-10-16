"""Verify the Tavily key and response shapes with minimal, credit-aware requests (architecture audit 2026-10-02).

GET /usage costs no credits (key validity, plan limits). --search adds ONE basic search (1 credit) to record the
real result shape needed for provenance (url, title, content, score, published date). Result content is
untrusted web text: only URL / title / score / field names are stored, never the content.
Writes data/manifests/tavily_capabilities.json. The key comes from TAVILY_API_KEY and is never printed or stored:
    python <scratchpad>/with_nebius.py scripts/research/probe_tavily.py [--search]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "manifests" / "tavily_capabilities.json"
BASE = "https://api.tavily.com/"
QUERY = "LoRA: Low-Rank Adaptation of Large Language Models arXiv 2106.09685"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", action="store_true", help="also run one basic search (1 credit)")
    args = ap.parse_args()
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        sys.exit("TAVILY_API_KEY not set")
    out = {"provider": "tavily", "base_url": BASE, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {key}"}, timeout=60) as c:
        r = c.get("usage")
        out["usage"] = {"http_status": r.status_code, **(r.json() if r.status_code == 200 else {"error": r.text[:200]})}
        if args.search and r.status_code == 200:
            t = time.time()
            s = c.post("search", json={"query": QUERY, "search_depth": "basic", "max_results": 3, "include_answer": False})
            body = s.json() if s.status_code == 200 else {}
            out["search"] = {"http_status": s.status_code, "query": QUERY, "search_depth": "basic", "documented_credits": 1,
                             "latency_s": round(time.time() - t, 2), "response_fields": sorted(body),
                             "result_fields": sorted({k for x in body.get("results", []) for k in x}),
                             "results": [{"url": x.get("url"), "title": x.get("title"), "score": x.get("score")} for x in body.get("results", [])],
                             "error": None if s.status_code == 200 else s.text[:200]}
            out["usage_after"] = c.get("usage").json()
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "usage_after"}, indent=1)[:2500])


if __name__ == "__main__":
    main()
