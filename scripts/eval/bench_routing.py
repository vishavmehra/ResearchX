"""Fixed-versus-dynamic model routing benchmark (routing decision contract §7; docs/ARCHITECTURE.md §67).

The same questions run under four routing modes (config/runtime.yaml routing.mode): fixed_nano, fixed_super, static (the
rules before routing-0.1.0) and dynamic (routing-0.1.0). Field questions come from the grounding evaluation (TEST +
TEMPORAL_TEST: positives with gold values, invented-entity negatives, recorded contradictions) and are scored like it.
Passage questions (open and comparison) have no gold answer, so they are scored on structured-output validity, citation
validity and abstention only; their correctness is not measured (no LLM judge). External search and Laya are off and no
cache is shared (no user scope; one orchestrator per mode), so the modes differ only in model routing; the mode order
rotates per question. Cost is not estimated: the configuration holds no verified provider pricing; tokens are counted.
Writes data/reports/routing_bench_v<release>.{json,md}.
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval/bench_routing.py --positives 6 --negatives 2 --contradictions 4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.eval_grounding import build_cases, score_case  # noqa: E402
from services.datasets.store import config  # noqa: E402
from services.env import load_env  # noqa: E402

VERSION = "routing-bench-0.1.0"
MODES = ("fixed_nano", "fixed_super", "static", "dynamic")
PASSAGE = [
    "Which papers evaluate state space models such as Mamba on the Long Range Arena benchmark?",
    "Which benchmarks were used to evaluate Kolmogorov-Arnold Networks against MLPs?",
    "Why do vision transformers need large-scale pre-training data?",
    "What limitations of masked autoencoders do papers report?",
    "What data augmentation strategies are used to train vision transformers?",
    "Which methods make attention more efficient for long sequences?",
    "Which metrics are used to evaluate image captioning?",
    "How is knowledge distillation used to train smaller vision transformers?",
    "How does masked image modeling pre-training compare with contrastive pre-training for vision transformers?",
    "Compare the ImageNet top-1 accuracy of vision transformers and convolutional networks reported in the corpus.",
    "How does HRNetV2 compare with DeepLabV3+ on Cityscapes?",
    "Compare Swin Transformer and ViT on ADE20K semantic segmentation.",
    "Compare the pre-training objectives of BERT and RoBERTa.",
    "How does LoRA compare with full fine-tuning?",
    "Compare DeiT and ViT in their training data requirements.",
    "How do Mamba and Transformers compare on long-sequence modeling?",
]
GAP = [  # field questions whose values the corpus evidence lacks: runtime extraction (and its routing) decides
    "What top-1 accuracy does BEiT-large achieve on ImageNet-1K?",
    "What batch size was used to train Segmenter on ADE20K?",
    "What weight decay was used to train Swin-L on COCO?",
    "What learning rate and batch size were used to train FeatMark?",
]


def pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def routings(o, out: list) -> list:
    """Every routing trace (a decision with its attempts) in a run's stage details."""
    if isinstance(o, dict):
        if "routing_policy_version" in o:
            out.append(o)
        else:
            for v in o.values():
                routings(v, out)
    elif isinstance(o, list):
        for v in o:
            routings(v, out)
    return out


def measure(c: dict, mode: str, r: dict, db) -> dict:
    stages = {s["stage"]: s for s in r["stages"]}
    traces = [t for t in routings([s.get("detail") for s in r["stages"]], []) if t["task_type"] != "reranking"]
    attempts = [a for t in traces for a in t["attempts"] if "model" in a]
    row = {"mode": mode, "case": c["case"], "type": c["type"], "query": c["query"], "status": r["status"], "total_ms": r["total_ms"],
           "evidence_status": r.get("evidence_status"), "model_ms": sum(a["latency_ms"] for a in attempts) if attempts else None,
           "model_calls": len(attempts), "models": dict(Counter(a["model"] for a in attempts)),
           "outputs_valid": sum(a["outcome"] != "no usable output" for a in attempts),
           "fallback": any(t["fallback_used"] for t in traces), "no_route": any(t["selected_route"] is None for t in traces),
           "failed": r["mode"] == "fallback" or (stages.get("synthesis") or {}).get("status") == "failed",
           "tokens_in": (r.get("usage") or {}).get("llm_input_tokens", 0), "tokens_out": (r.get("usage") or {}).get("llm_output_tokens", 0),
           "reasons": sorted({t["selection_reason"].split(";")[0] for t in traces})}
    if c["case"] == "passage":
        cv = (stages.get("citation_validation") or {}).get("detail") or {}
        row |= {"answered": r["status"] in ("ANSWERED", "PARTIAL", "CONFLICTING"), "claims": cv.get("claims"), "claims_valid": cv.get("valid"),
                "schema_valid": (stages.get("synthesis") or {}).get("detail", {}).get("model_status") is not None
                if (stages.get("synthesis") or {}).get("status") != "not_required" else None}
    else:
        s = score_case(db, c, r)
        row |= {k: s.get(k) for k in ("answered", "exact_correct", "gold_in_values", "conflict_flagged", "claims", "claims_in_source", "claims_entailed")}
    return row


def summary(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["case"] == "positive"]
    neg = [r for r in rows if r["case"] == "negative"]
    con = [r for r in rows if r["case"] == "contradiction"]
    pas = [r for r in rows if r["case"] == "passage"]
    gap = [r for r in rows if r["case"] == "gap"]
    frac = lambda sel, f: {"value": round(sum(map(f, sel)) / len(sel), 3) if sel else None, "n": sum(map(f, sel)), "of": len(sel)}  # noqa: E731
    calls = sum(r["model_calls"] for r in rows)
    claims = sum(r.get("claims") or 0 for r in pas)
    field_claims = sum(r.get("claims") or 0 for r in pos + neg + con + gap)
    return {
        "runs": len(rows),
        "total_ms": {"p50": pct([r["total_ms"] for r in rows], 0.5), "p95": pct([r["total_ms"] for r in rows], 0.95)},
        "model_ms": {"p50": pct([r["model_ms"] for r in rows], 0.5), "p95": pct([r["model_ms"] for r in rows], 0.95)},
        "tokens": {"input": sum(r["tokens_in"] for r in rows), "output": sum(r["tokens_out"] for r in rows)},
        "model_calls": calls,
        "structured_output_validity": {"value": round(sum(r["outputs_valid"] for r in rows) / calls, 3) if calls else None,
                                       "n": sum(r["outputs_valid"] for r in rows), "of": calls},
        "fallback_rate": frac(rows, lambda r: r["fallback"]),
        "failure_rate": frac(rows, lambda r: r["failed"] or r["no_route"]),
        "field_exact_correct": frac(pos, lambda r: bool(r.get("exact_correct"))),
        "field_wrong_answers": frac([r for r in pos if r["answered"]], lambda r: not r.get("gold_in_values")),
        "field_false_abstention": frac(pos, lambda r: not r["answered"]),
        "negatives_answered": frac(neg, lambda r: bool(r["answered"])),
        "contradictions_flagged": frac(con, lambda r: bool(r.get("conflict_flagged"))),
        "field_citation_correctness": {"value": round(sum(r.get("claims_in_source") or 0 for r in pos + neg + con + gap) / field_claims, 3)
                                       if field_claims else None, "of": field_claims},
        "gap_answered": frac(gap, lambda r: bool(r["answered"])),
        "passage_answered": frac(pas, lambda r: r["answered"]),
        "passage_schema_valid": frac([r for r in pas if r.get("schema_valid") is not None], lambda r: r["schema_valid"]),
        "passage_citation_validity": {"value": round(sum(r.get("claims_valid") or 0 for r in pas) / claims, 3) if claims else None, "of": claims},
        "models": dict(sum((Counter(r["models"]) for r in rows), Counter())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="0.2.0")
    ap.add_argument("--positives", type=int, default=6, help="per type (protocol, result)")
    ap.add_argument("--negatives", type=int, default=2, help="per kind (invented model, invented dataset)")
    ap.add_argument("--contradictions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    load_env()
    from services.model_gateway.gateway import ModelGateway
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator, detect_intent
    from services.storage.db import connect

    db = connect(autocommit=True)
    cases = build_cases(db, args.release, args.positives, args.negatives, args.contradictions, args.seed)
    cases += [{"case": "passage", "type": "comparison" if detect_intent(q) in ("comparison", "contradiction") else "open", "query": q,
               "query_id": f"passage:{i}"} for i, q in enumerate(PASSAGE)]
    cases += [{"case": "gap", "type": "field value missing from corpus evidence", "query": q, "query_id": f"gap:{i}"} for i, q in enumerate(GAP)]
    base = copy.deepcopy(config("runtime"))
    base["external"]["enabled"] = False
    base["laya"]["enabled"] = False
    gw, orchs, bm25 = ModelGateway(), {}, None
    for m in MODES:
        cfg = copy.deepcopy(base)
        cfg["routing"]["mode"] = m
        orchs[m] = Orchestrator(cfg, gw=gw, bm25=bm25)
        bm25 = bm25 or orchs[m].bm25()
    jobs = [(c, MODES[(i + k) % len(MODES)]) for i, c in enumerate(cases) for k in range(len(MODES))]

    def one(job):
        c, m = job
        try:
            return c, m, orchs[m].run(c["query"], RetrievalScope(None)), None
        except Exception as e:  # recorded, never silently dropped
            return c, m, None, f"{type(e).__name__}: {e}"

    t0, rows, errors = time.time(), [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (c, m, r, err) in enumerate(pool.map(one, jobs), 1):
            if err:
                errors.append({"mode": m, "query": c["query"], "error": err})
            else:
                rows.append(measure(c, m, r, db))
            print(f"[{i}/{len(jobs)}] {m:<11} {c['case']:<13} {'ERROR' if err else rows[-1]['status']:<10} "
                  f"{'' if err else rows[-1]['total_ms']:>6} ms {c['query'][:70]}", flush=True)
    by_mode = {m: summary([r for r in rows if r["mode"] == m]) for m in MODES}
    report = {"version": VERSION, "pipeline_version": base["pipeline_version"], "release": args.release, "seed": args.seed,
              "cases": dict(Counter(c["case"] for c in cases)), "modes": by_mode, "errors": errors, "wall_s": round(time.time() - t0, 1),
              "rows": rows, "notes": "external search and Laya off; no cache shared between modes; cost not estimated (no verified pricing)"}
    out = ROOT / "data" / "reports"
    (out / f"routing_bench_v{args.release}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    show = lambda v: "–" if v is None else (f"{v['value']} ({v['n']}/{v['of']})" if "n" in v else f"{v['value']} (of {v['of']})")  # noqa: E731
    keys = [("field_exact_correct", "Field answers exactly correct"), ("field_wrong_answers", "Field wrong answers (when answered)"),
            ("field_false_abstention", "Field false abstentions"), ("negatives_answered", "Invented-entity questions answered"),
            ("contradictions_flagged", "Recorded contradictions flagged"),
            ("gap_answered", "Field questions beyond corpus evidence answered"), ("field_citation_correctness", "Field citation correctness"),
            ("passage_answered", "Passage questions answered"), ("passage_schema_valid", "Passage synthesis schema-valid"),
            ("passage_citation_validity", "Passage claims passing citation validation"),
            ("structured_output_validity", "Model outputs parsed and schema-valid"), ("fallback_rate", "Runs with a fallback model"),
            ("failure_rate", "Runs with a failed model stage")]
    lines = [f"# Routing benchmark v{args.release}", "",
             f"{VERSION} · {base['pipeline_version']} · seed {args.seed} · {len(rows)} runs ({report['cases']} x {len(MODES)} modes) · "
             f"{len(errors)} errors · wall {report['wall_s']} s · external search and Laya off · no shared cache", "",
             "| Metric | " + " | ".join(MODES) + " |", "|---|" + "---|" * len(MODES)]
    lines += [f"| {label} | " + " | ".join(show(by_mode[m][k]) for m in MODES) + " |" for k, label in keys]
    lines += ["| Total latency p50 / p95 (ms) | " + " | ".join(f"{by_mode[m]['total_ms']['p50']} / {by_mode[m]['total_ms']['p95']}" for m in MODES) + " |",
              "| Answer-model latency p50 / p95 (ms) | " + " | ".join(f"{by_mode[m]['model_ms']['p50']} / {by_mode[m]['model_ms']['p95']}" for m in MODES) + " |",
              "| LLM tokens in / out (answer models + reranker) | " + " | ".join(f"{by_mode[m]['tokens']['input']} / {by_mode[m]['tokens']['output']}" for m in MODES) + " |",
              "| Answer-model calls | " + " | ".join(str(by_mode[m]["model_calls"]) for m in MODES) + " |", "",
              "Models used (answer-model calls): " + "; ".join(f"{m}: {by_mode[m]['models']}" for m in MODES), "",
              "Passage questions have no gold answers: their correctness is not measured (no LLM judge). Cost is not estimated: "
              "config/models.yaml holds no verified provider pricing; token counts are reported instead."]
    (out / f"routing_bench_v{args.release}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
