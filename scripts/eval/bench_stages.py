"""Stage-latency benchmark for the runtime performance work (core-engine brief §2, §18): a fixed question set run as a
temporary user, every run cold (answer- and provider-cache writes off, lookups kept), per-stage p50 / p90 / max, the
slowest stage of each run, model-call tokens, and the time not spent in any stage (run setup, trace flush).

    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval/bench_stages.py --label baseline [--rounds 2] [--out data/reports]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from services.env import load_env

load_env()

VERSION = "stage-bench-0.1.0"
# (kind, question): field questions the corpus answers or lacks, passage questions with and without a gap, a comparison
QUESTIONS = [
    ("field_corpus", "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?"),
    ("field_corpus", "What top-1 accuracy does BEiT-L achieve on ImageNet-1K?"),
    ("field_corpus", "Which optimizer and weight decay were used to fine-tune BEiT on ADE20K?"),
    ("field_gap", "What top-1 accuracy does BEiT-large achieve on ImageNet-1K?"),
    ("field_gap", "What batch size was used to train Segmenter on ADE20K?"),
    ("field_gap", "What weight decay was used to train Swin-L on COCO?"),
    ("passage", "Which papers evaluate state space models such as Mamba on the Long Range Arena benchmark?"),
    ("passage", "How does masked image modeling pre-training compare with contrastive pre-training for vision transformers?"),
    ("passage_gap", "Which benchmarks were used to evaluate Kolmogorov-Arnold Networks against MLPs?"),
    ("comparison", "Compare the ImageNet top-1 accuracy of vision transformers and convolutional networks reported in the corpus."),
]


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="data/reports")
    args = ap.parse_args()
    from scripts.eval.bench_runtime import temp_user
    from services.datasets.store import config
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator

    cfg = copy.deepcopy(config("runtime"))
    cfg["cache"]["write_limit_mb"] = 0  # every run cold: nothing is written to the answer or provider cache
    o = Orchestrator(cfg=cfg)
    user, cleanup = temp_user()
    runs = []
    try:
        for rnd in range(args.rounds):
            for kind, q in QUESTIONS:
                t = time.perf_counter()
                r = o.run(q, RetrievalScope(user))
                wall = round((time.perf_counter() - t) * 1000)
                stages = {s["stage"]: s for s in r["stages"]}
                ran = {k: s["latency_ms"] for k, s in stages.items() if s["status"] in ("completed", "failed")}
                slowest = max(ran, key=ran.get) if ran else None
                rr = (stages.get("reranking") or {}).get("detail") or {}
                runs.append({"round": rnd, "kind": kind, "query": q, "status": r["status"], "evidence_status": r.get("evidence_status"),
                             "total_ms": r["total_ms"], "wall_ms": wall, "stages_ms": ran, "failed": [k for k, s in stages.items() if s["status"] == "failed"],
                             "slowest": slowest,  # tavily runs inside the external_sources phase: not summed twice
                             "unaccounted_ms": r["total_ms"] - sum(v for k, v in ran.items() if k != "tavily") if ran else None,
                             "embedding_ms": ((stages.get("pgvector") or {}).get("detail") or {}).get("query_embedding_ms"),
                             "rerank_tokens": [rr.get("input_tokens"), rr.get("output_tokens")], "usage": r.get("usage"),
                             "trace_persisted": r["trace_persisted"]})
                print(f"{kind:13s} {r['status']:10s} {r['total_ms']:6d} ms  slowest {slowest} {ran.get(slowest)} ms  {q[:60]}", flush=True)
    finally:
        cleanup()

    per_stage = defaultdict(list)
    for r in runs:
        for k, v in r["stages_ms"].items():
            per_stage[k].append(v)
    stage_stats = {k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "max": max(v), "budget_ms": (cfg.get("latency_budgets_ms") or {}).get(k)}
                   for k, v in sorted(per_stage.items(), key=lambda kv: -sum(kv[1]))}
    by_kind = {k: {"n": len(rs), "total_p50": pct([r["total_ms"] for r in rs], 0.5), "total_p90": pct([r["total_ms"] for r in rs], 0.9)}
               for k in dict.fromkeys(r["kind"] for r in runs) for rs in [[r for r in runs if r["kind"] == k]]}
    totals = [r["total_ms"] for r in runs]
    report = {"version": VERSION, "label": args.label, "pipeline_version": cfg["pipeline_version"], "rounds": args.rounds, "runs": runs,
              "total_ms": {"p50": pct(totals, 0.5), "p90": pct(totals, 0.9), "max": max(totals)}, "by_kind": by_kind, "stages": stage_stats,
              "unaccounted_ms": {"p50": pct([r["unaccounted_ms"] for r in runs if r["unaccounted_ms"] is not None], 0.5)},
              "embedding_ms": {"p50": pct([r["embedding_ms"] for r in runs if r["embedding_ms"]], 0.5),
                               "p90": pct([r["embedding_ms"] for r in runs if r["embedding_ms"]], 0.9)},
              "failed_stages": sorted({f for r in runs for f in r["failed"]}), "trace_persisted": all(r["trace_persisted"] for r in runs)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"stage_bench_{args.label}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"# Stage benchmark: {args.label}", "", (f"{VERSION} · {cfg['pipeline_version']} · {len(runs)} cold runs ({args.rounds} rounds x "
                                                      f"{len(QUESTIONS)} questions) · temporary user"), "",
             (f"Total ms p50 / p90 / max: {report['total_ms']['p50']} / {report['total_ms']['p90']} / {report['total_ms']['max']} · not in any stage "
              f"(setup, trace flush) p50 {report['unaccounted_ms']['p50']} ms · query embedding p50 / p90 {report['embedding_ms']['p50']} / "
              f"{report['embedding_ms']['p90']} ms · failed stages: {', '.join(report['failed_stages']) or 'none'}"), "",
             "| Question kind | Runs | Total p50 | Total p90 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['total_p50']} | {v['total_p90']} |" for k, v in by_kind.items()]
    lines += ["", "| Stage (by total time) | Runs | p50 ms | p90 ms | max ms | budget ms |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['p50']} | {v['p90']} | {v['max']} | {v['budget_ms'] or '–'} |" for k, v in stage_stats.items()]
    (out / f"stage_bench_{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
