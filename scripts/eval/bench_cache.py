"""Cache hit-versus-miss benchmark (docs/ARCHITECTURE.md §68): latency and correctness of each cache layer.

  answer cache     (rx.cache_entries 'model', per user, RLS): each question asked twice by a temporary user; the second
                   run must come from the cache with the same status, answer and claims.
  query embedding  (in-process, per user) and reranker grades (in-process, per user): each question asked twice with the
                   answer cache unwritten, so the second run re-executes the pipeline; the second run must select the
                   same evidence, faster.
  external sources (rx.cache_entries 'api', per user, RLS): one evidence-gap question with external search on, asked
                   twice with the user's answer cache emptied in between; the second run must use the cached provider
                   results (no provider call) and accept the same sources.
The temporary user and every row it wrote are deleted at the end. Writes data/reports/cache_bench_v<release>.{json,md}.
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval/bench_cache.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.datasets.store import config  # noqa: E402
from services.env import load_env  # noqa: E402

VERSION = "cache-bench-0.1.0"
QUESTIONS = [
    "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?",
    "What top-1 accuracy does BEiT-L achieve on ImageNet-1K?",
    "Which optimizer and weight decay were used to fine-tune BEiT on ADE20K?",
    "Which papers evaluate state space models such as Mamba on the Long Range Arena benchmark?",
    "How does masked image modeling pre-training compare with contrastive pre-training for vision transformers?",
    "Compare the ImageNet top-1 accuracy of vision transformers and convolutional networks reported in the corpus.",
]
GAP_QUESTION = "Which benchmarks were used to evaluate Kolmogorov-Arnold Networks against MLPs?"


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def stage(r: dict, name: str) -> dict:
    return next((s for s in r["stages"] if s["stage"] == name), {})


def selected(r: dict) -> list:
    return (stage(r, "evidence_selection").get("detail") or {}).get("source_ids") or [s.get("chunk_id") or s.get("url") for s in r["sources"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="0.2.0")
    ap.add_argument("--write-limit-mb", type=float, default=None,
                    help="override cache.write_limit_mb for this run: the guard stops cache writes once the database passes it, "
                         "which would hide the hit path being measured (the temporary user's rows are deleted afterwards)")
    args = ap.parse_args()
    load_env()
    from scripts.eval.bench_runtime import temp_user
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator

    base = copy.deepcopy(config("runtime"))
    base["laya"]["enabled"] = False
    if args.write_limit_mb is not None:
        base["cache"]["write_limit_mb"] = args.write_limit_mb
    corpus_only = copy.deepcopy(base)
    corpus_only["external"]["enabled"] = False
    answers = Orchestrator(corpus_only)  # answer cache written as usual
    unwritten = copy.deepcopy(corpus_only)
    unwritten["cache"]["write_limit_mb"] = 0  # no answer (or provider) cache rows: the second run re-executes the pipeline
    components = Orchestrator(unwritten, gw=answers.gw, bm25=answers.bm25())
    external = Orchestrator(base, gw=answers.gw, bm25=answers.bm25())
    user, cleanup = temp_user()
    scope = RetrievalScope(user)
    out: dict = {"version": VERSION, "pipeline_version": base["pipeline_version"], "answer_cache": [], "component_caches": [], "external_cache": None}
    t0 = time.time()
    try:
        for q in QUESTIONS:  # first: these runs write no answer rows, so the answer cache cannot short-circuit them
            a, b = components.run(q, scope), components.run(q, scope)
            pa, pb = stage(a, "pgvector").get("detail") or {}, stage(b, "pgvector").get("detail") or {}
            ra, rb = stage(a, "reranking"), stage(b, "reranking")
            out["component_caches"].append({
                "query": q, "embedding_cache": [pa.get("embedding_cache"), pb.get("embedding_cache")],
                "embedding_ms": [pa.get("query_embedding_ms"), pb.get("query_embedding_ms")],
                "rerank_cache": [(ra.get("detail") or {}).get("cache"), (rb.get("detail") or {}).get("cache")],
                "rerank_ms": [ra.get("latency_ms"), rb.get("latency_ms")], "total_ms": [a["total_ms"], b["total_ms"]],
                "same_evidence": selected(a) == selected(b)})
            print("components", out["component_caches"][-1]["rerank_cache"], out["component_caches"][-1]["rerank_ms"], q[:60], flush=True)
        for q in QUESTIONS:
            a, b = answers.run(q, scope), answers.run(q, scope)
            hit = bool((stage(b, "cache_lookup").get("detail") or {}).get("hit"))
            out["answer_cache"].append({
                "query": q, "first_status": a["status"], "cacheable": a["status"] in ("ANSWERED", "PARTIAL", "CONFLICTING"), "hit": hit,
                "miss_ms": a["total_ms"], "hit_ms": b["total_ms"],
                "same_answer": hit and (a["status"], a["answer"], a["claims"]) == (b["status"], b["answer"], b["claims"])})
            print("answer", out["answer_cache"][-1]["first_status"], hit, a["total_ms"], b["total_ms"], q[:60], flush=True)
        a = external.run(GAP_QUESTION, scope)
        external.uq(user, "delete from rx.cache_entries where user_id = %s and kind = 'model'", (user.id,))  # as the user (RLS)
        b = external.run(GAP_QUESTION, scope)
        ea, eb = stage(a, "external_sources"), stage(b, "external_sources")
        prov = lambda s: {n: p.get("status") for n, p in ((s.get("detail") or {}).get("providers") or {}).items()}  # noqa: E731
        accepted = lambda s: [x.get("url") for x in ((stage(s, "source_validation").get("detail") or {}).get("selected") or [])]  # noqa: E731
        out["external_cache"] = {"query": GAP_QUESTION, "first_status": ea.get("status"), "second_status": eb.get("status"),
                                 "providers_first": prov(ea), "providers_second": prov(eb), "external_ms": [ea.get("latency_ms"), eb.get("latency_ms")],
                                 "total_ms": [a["total_ms"], b["total_ms"]], "same_sources": accepted(a) == accepted(b)}
        print("external", out["external_cache"], flush=True)
    finally:
        cleanup()
    ac = out["answer_cache"]
    hits = [x for x in ac if x["hit"]]
    cc = out["component_caches"]
    out["summary"] = {
        "answer_cache": {"cacheable": sum(x["cacheable"] for x in ac), "hits": len(hits), "of": len(ac),
                         "miss_ms_p50": pct([x["miss_ms"] for x in hits], 0.5), "hit_ms_p50": pct([x["hit_ms"] for x in hits], 0.5),
                         "same_answer": sum(x["same_answer"] for x in hits)},
        "embedding_cache": {"hits_on_repeat": sum(x["embedding_cache"][1] == "hit" for x in cc), "of": len(cc),
                            "miss_ms_p50": pct([x["embedding_ms"][0] for x in cc], 0.5), "hit_ms_p50": pct([x["embedding_ms"][1] for x in cc], 0.5)},
        "rerank_cache": {"hits_on_repeat": sum(x["rerank_cache"][1] == "hit" for x in cc), "of": len(cc),
                         "miss_ms_p50": pct([x["rerank_ms"][0] for x in cc], 0.5), "hit_ms_p50": pct([x["rerank_ms"][1] for x in cc], 0.5),
                         "same_evidence": sum(x["same_evidence"] for x in cc)},
        "pipeline_total_ms_p50": {"cold": pct([x["total_ms"][0] for x in cc], 0.5), "warm_components": pct([x["total_ms"][1] for x in cc], 0.5)},
        "wall_s": round(time.time() - t0, 1)}
    path = ROOT / "data" / "reports" / f"cache_bench_v{args.release}"  # with_suffix would eat the release's ".0"
    Path(f"{path}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    s, e = out["summary"], out["external_cache"] or {}
    guard = (f" · cache write guard raised to {args.write_limit_mb} MB for this run (configured {config('runtime')['cache']['write_limit_mb']} MB)"
             if args.write_limit_mb is not None else "")
    lines = [f"# Cache benchmark v{args.release}", "", f"{VERSION} · {base['pipeline_version']} · temporary user (deleted afterwards) · wall {s['wall_s']} s{guard}", "",
             "| Layer | Hits on repeat | Miss p50 (ms) | Hit p50 (ms) | Correct on hit |", "|---|---|---|---|---|",
             f"| Answer cache (per user, RLS) | {s['answer_cache']['hits']}/{s['answer_cache']['of']} ({s['answer_cache']['cacheable']} cacheable) | "
             f"{s['answer_cache']['miss_ms_p50']} | {s['answer_cache']['hit_ms_p50']} | {s['answer_cache']['same_answer']}/{s['answer_cache']['hits']} identical answers |",
             f"| Query embedding (in-process, per user) | {s['embedding_cache']['hits_on_repeat']}/{s['embedding_cache']['of']} | "
             f"{s['embedding_cache']['miss_ms_p50']} | {s['embedding_cache']['hit_ms_p50']} | same vector by construction |",
             f"| Reranker grades (in-process, per user) | {s['rerank_cache']['hits_on_repeat']}/{s['rerank_cache']['of']} | "
             f"{s['rerank_cache']['miss_ms_p50']} | {s['rerank_cache']['hit_ms_p50']} | {s['rerank_cache']['same_evidence']}/{s['rerank_cache']['of']} same evidence selected |",
             f"| External sources (per user, RLS) | {'yes' if any(v == 'cache_hit' for v in (e.get('providers_second') or {}).values()) else 'no'} | "
             f"{(e.get('external_ms') or [None])[0]} | {(e.get('external_ms') or [None, None])[1]} | {'same sources accepted' if e.get('same_sources') else 'sources differ'} |",
             "", f"Pipeline total p50 with in-process caches cold / warm (answer cache unwritten): {s['pipeline_total_ms_p50']['cold']} / "
             f"{s['pipeline_total_ms_p50']['warm_components']} ms. Abstained answers are never cached (by design).",
             f"External providers first / second run: {e.get('providers_first')} / {e.get('providers_second')}."]
    Path(f"{path}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
