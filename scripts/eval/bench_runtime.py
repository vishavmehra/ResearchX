"""Runtime benchmark: research runs with the runtime graph, and Knowledge Map operations (live: Supabase, models, external APIs).

Scenarios
  A internal-only questions (the corpus answers; external research must not run)
  B questions with an evidence gap (external providers run, the runtime graph records what they found)
  C a large-context passage question (comparison intent: more sources, the larger synthesis task)
  D multi-experiment comparison (Knowledge Map diff and "closest experiments" over corpus experiments)
  E graph expansion (neighbourhood pages of corpus nodes, cold and warm) and runtime-graph build time
  F follow-up: the same questions again for the same user (answer cache, per-user provider cache)
Runs execute as a temporary user (deleted at the end, with everything it owns), so caches behave as in production.

    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval/bench_runtime.py [--out data/reports] [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import secrets
import time
import uuid
from pathlib import Path

import httpx

from services.env import load_env

load_env()

VERSION = "runtime-bench-0.1.0"
# the product's own example questions (A), configurations absent from the corpus (B), and a comparison question (C)
QUESTIONS = {
    "A": ["What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?", "What top-1 accuracy does ViT-B/16 reach on ImageNet?",
          "Which optimizer and weight decay were used to fine-tune BEiT on ADE20K?"],
    "B": ["What batch size was used to train Segmenter on ADE20K?", "What learning rate was used to train ConvNeXt-T on ImageNet-1K?",
          "What weight decay was used to train Swin-L on COCO?"],
    "C": ["Compare the ImageNet top-1 accuracy of vision transformers and convolutional networks reported in the corpus."],
}
# audit baselines (planning/AUDIT_BATCH2_2026-10-07.md; data/reports/grounding_eval_v0.2.0.md)
BASELINES = {"end_to_end_p50_ms_542_runs": 6500, "query_embedding_p50_ms": 690, "vector_sql_p50_ms": 175,
             "grounding_eval_median_ms_corpus_only": 11025}


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def stage(r: dict, name: str) -> dict:
    return next((s for s in r.get("stages") or [] if s["stage"] == name), {})


def summary(scenario: str, query: str, r: dict, wall_ms: int) -> dict:
    def lat(*names: str) -> int:
        return sum(stage(r, n).get("latency_ms") or 0 for n in names)
    usage, ext, g = r.get("usage") or {}, stage(r, "external_sources"), (r.get("graph") or {}).get("stats") or {}
    providers = (ext.get("detail") or {}).get("providers") or {}
    return {"scenario": scenario, "query": query, "status": r.get("status"), "evidence_status": r.get("evidence_status"), "total_ms": r.get("total_ms"),
            "wall_ms": wall_ms, "retrieval_ms": lat("bm25", "pgvector", "hybrid_fusion", "reranking"),
            "embedding_ms": (stage(r, "pgvector").get("detail") or {}).get("query_embedding_ms"),
            "external_ran": ext.get("status") not in (None, "not_required", "skipped"), "external_ms": lat("external_sources"), "tavily_ms": lat("tavily"),
            "providers": {k: v.get("status") for k, v in providers.items()}, "provider_cache_hits": sum(v.get("status") == "cache_hit" for v in providers.values()),
            "api_calls": sum(v for k, v in usage.items() if k.endswith("_calls")) + usage.get("tavily_credits", 0),
            "extraction_ms": lat("evidence_extraction"), "synthesis_ms": lat("synthesis"), "graph_ms": stage(r, "graph_enrichment").get("latency_ms"),
            "graph_nodes": g.get("nodes"), "graph_edges": g.get("edges"), "runtime_external_nodes": (g.get("by_origin") or {}).get("runtime_external", 0),
            "tokens": (usage.get("llm_input_tokens") or 0) + (usage.get("llm_output_tokens") or 0), "models": r.get("routing"),
            "answer_cache_hit": bool((stage(r, "cache_lookup").get("detail") or {}).get("hit")) or bool(r.get("cached")),
            "over_budget": sorted(s["stage"] for s in r.get("stages") or [] if (s.get("detail") or {}).get("over_budget")),
            "trace_persisted": r.get("trace_persisted")}


def temp_user():
    """A confirmed throwaway user, signed in through Supabase Auth; returns (AuthUser, cleanup)."""
    from services.auth import supabase as auth
    from services.storage.db import connect

    conn = connect(autocommit=True)
    uid, email, pw = str(uuid.uuid4()), f"rx-bench-{secrets.token_hex(4)}@mailinator.com", secrets.token_urlsafe(18) + "Aa1!"
    conn.execute("insert into auth.users (instance_id, id, aud, role, email, encrypted_password, email_confirmed_at, raw_app_meta_data, raw_user_meta_data, "
                 "created_at, updated_at, confirmation_token, recovery_token, email_change_token_new, email_change) values "
                 "('00000000-0000-0000-0000-000000000000', %s, 'authenticated', 'authenticated', %s, extensions.crypt(%s, extensions.gen_salt('bf')), now(), "
                 "'{\"provider\":\"email\",\"providers\":[\"email\"]}', '{}', now(), now(), '', '', '', '')", (uid, email, pw))
    conn.execute("insert into auth.identities (user_id, provider_id, identity_data, provider, last_sign_in_at, created_at, updated_at) "
                 "values (%s, %s, %s, 'email', now(), now(), now())", (uid, uid, json.dumps({"sub": uid, "email": email, "email_verified": True})))
    r = httpx.post(f"{os.environ['SUPABASE_URL'].rstrip('/')}/auth/v1/token?grant_type=password", timeout=30, json={"email": email, "password": pw},
                   headers={"apikey": os.environ["SUPABASE_PUBLISHABLE_KEY"], "Content-Type": "application/json"})
    r.raise_for_status()

    def cleanup():
        conn.execute("delete from auth.users where id = %s", (uid,))
        conn.close()
    return auth.verify(r.json()["access_token"]), cleanup


def timed(fn, *args) -> float:
    t = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reports")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    from services.graph.knowledge_map import GraphIndex
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator

    o, rng = Orchestrator(), random.Random(args.seed)
    user, cleanup = temp_user()
    runs: list[dict] = []

    def research(scenario: str, q: str) -> None:
        t = time.perf_counter()
        r = o.run(q, RetrievalScope(user))
        runs.append(summary(scenario, q, r, round((time.perf_counter() - t) * 1000)))
        print(scenario, runs[-1]["status"], runs[-1]["total_ms"], "ms", q[:60], flush=True)

    try:
        for scenario in ("A", "B", "C"):
            for q in QUESTIONS[scenario]:
                research(scenario, q)
        for q in QUESTIONS["A"][:1] + QUESTIONS["B"][:2]:  # F: follow-ups by the same user
            research("F", q)
        t = time.perf_counter()
        g = GraphIndex(o.q)
        index_ms = round((time.perf_counter() - t) * 1000)
        exps = sorted(x for x, e in g.exp.items() if any(link["edge"] == "USES_DATASET" for link in e["links"]))
        diff_ms, closest_ms = [], []
        for x in rng.sample(exps, 20):  # D: diff against another experiment on the same benchmark; rank the closest ones
            bench = next(link["entity_id"] for link in g.exp[x]["links"] if link["edge"] == "USES_DATASET")
            if other := next((y for y in g.ent_exps[bench] if y != x), None):
                diff_ms.append(timed(g.diff, x, other))
            closest_ms.append(timed(g.ask, x, "closest"))
        cold_ms, warm_ms = [], []
        for x in rng.sample(exps, 20):  # E: expansion pages
            cold_ms.append(timed(g.neighborhood, x))
            warm_ms.append(timed(g.neighborhood, x))
        entity_ms = [timed(g.neighborhood, e) for e in rng.sample(sorted(g.ent_exps), 20)]
    finally:
        cleanup()

    def agg(rows: list[dict], key: str) -> dict:
        xs = [r[key] for r in rows if r.get(key) is not None]
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "n": len(xs)}

    def ms(xs: list[float]) -> dict:
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "n": len(xs)}

    scenarios = {}
    for s in ("A", "B", "C", "F"):
        rows = [r for r in runs if r["scenario"] == s]
        scenarios[s] = {"runs": len(rows), "total_ms": agg(rows, "total_ms"), "retrieval_ms": agg(rows, "retrieval_ms"),
                        "external_ms": agg([r for r in rows if r["external_ran"]], "external_ms"), "graph_ms": agg(rows, "graph_ms"),
                        "external_ran": sum(r["external_ran"] for r in rows), "api_calls": sum(r["api_calls"] for r in rows),
                        "graph_nodes_added": sum(r["graph_nodes"] or 0 for r in rows), "graph_edges_added": sum(r["graph_edges"] or 0 for r in rows),
                        "runtime_external_nodes": sum(r["runtime_external_nodes"] for r in rows), "tokens": sum(r["tokens"] for r in rows),
                        "answer_cache_hit_rate": round(sum(r["answer_cache_hit"] for r in rows) / len(rows), 3) if rows else None,
                        "provider_cache_hits": sum(r["provider_cache_hits"] for r in rows), "trace_persisted": all(r["trace_persisted"] for r in rows)}
    ops = {"index_build_ms": index_ms, "D_diff_ms": ms(diff_ms), "D_closest_ms": ms(closest_ms), "E_experiment_page_cold_ms": ms(cold_ms),
           "E_experiment_page_warm_ms": ms(warm_ms), "E_entity_page_ms": ms(entity_ms)}
    report = {"version": VERSION, "pipeline_version": o.cfg["pipeline_version"], "seed": args.seed, "baselines": BASELINES, "scenarios": scenarios,
              "graph_operations": ops, "runs": runs, "models": runs[0]["models"] if runs else None}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_bench_v0.1.0.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    def fmt(d: dict) -> str:
        return f"{d['p50']:.0f} / {d['p90']:.0f} (n={d['n']})" if d.get("p50") is not None else "–"
    names = {"A": "A internal-only", "B": "B evidence gap (external)", "C": "C large context", "F": "F follow-up (same user)"}
    lines = [f"# Runtime benchmark {VERSION}", "", f"{o.cfg['pipeline_version']} · seed {args.seed} · live (Supabase, Nebius, external APIs) · "
             f"temporary user · models {json.dumps(report['models'])}", "",
             "| Scenario | Runs | Total ms p50 / p90 | Retrieval ms | External ms (when run) | Graph ms | External ran | API calls | "
             "Graph nodes / edges | Found outside corpus | Tokens | Answer-cache hit rate | Provider-cache hits |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in scenarios.items():
        lines.append(f"| {names[s]} | {v['runs']} | {fmt(v['total_ms'])} | {fmt(v['retrieval_ms'])} | {fmt(v['external_ms'])} | {fmt(v['graph_ms'])} | "
                     f"{v['external_ran']} | {v['api_calls']} | {v['graph_nodes_added']} / {v['graph_edges_added']} | {v['runtime_external_nodes']} | "
                     f"{v['tokens']} | {v['answer_cache_hit_rate']} | {v['provider_cache_hits']} |")
    lines += ["", "| Graph operation | p50 / p90 ms |", "|---|---|", f"| index build (once per API process) | {index_ms} |"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in ops.items() if k != "index_build_ms"]
    lines += ["", "Baselines (audit): " + ", ".join(f"{k} = {v}" for k, v in BASELINES.items()), "",
              "Per-run detail (stages over budget, provider outcomes) is in the JSON. Small samples: with three runs, p90 is the slowest run."]
    (out / "runtime_bench_v0.1.0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
