"""Build the retrieval evaluation set of a release (Phase 7; docs/EVALUATION_SPEC.md Extension §51).

Relevance comes only from frozen, location-verified evidence: a deep-evidence record resolves to the chunk holding it
(<paper_id>#<anchor of source_location>). Queries come from VALIDATION (tuning) and TEST + TEMPORAL_TEST (reporting)
experiments, never TRAIN:
  result     "What <metric> does <model> achieve on <benchmark>?"  grade 2: chunk with the score evidence;
                                                                    grade 1: chunks with model/benchmark/metric evidence
  protocol   "What <field> was used to train <model> on <benchmark>?"  grade 2: chunk with that field's evidence;
                                                                    grade 1: chunks with the model evidence
  prior_art  prior_art_experiment_search task questions -> relevant papers = papers of the expected experiments
At most 2 queries per paper and type, seeded sampling, identical query texts kept once. Template queries name the
model and benchmark as users do, which favours lexical matching: report that bias next to the results. Usage:
    python scripts/eval/build_retrieval_eval.py --release 0.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "retrieval-eval-0.1.0"
SPLITS = ("VALIDATION", "TEST", "TEMPORAL_TEST")  # VALIDATION tunes; TEST + TEMPORAL_TEST report
PROTOCOL = {"learning_rate": "learning rate", "batch_size": "batch size", "optimizer": "optimizer", "weight_decay": "weight decay",
            "epochs": "number of epochs", "warmup_steps": "number of warmup steps", "dropout": "dropout rate",
            "learning_rate_schedule": "learning rate schedule"}
PER_SPLIT = {"result": 150, "protocol": 150}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def known(field: dict | None) -> bool:
    return isinstance(field, dict) and field.get("status") == "KNOWN" and bool(field.get("evidence_ids"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    data = ROOT / "data"
    bm25 = json.loads((data / "indexes" / "bm25" / f"v{args.release}.json").read_text(encoding="utf-8"))
    chunk_ids = set(bm25["ids"])
    loc = {}  # evidence id -> chunk id, only for evidence located in a parsed chunk
    for r in read_jsonl(data / "silver/evidence/deep_evidence.jsonl"):
        cid = f"{r['paper_id']}#{(r.get('source_location') or '').rsplit('#', 1)[-1]}"
        if cid in chunk_ids:
            loc[r["evidence_id"]] = cid

    def chunks(field: dict | None) -> list[str]:
        return sorted({loc[e] for e in (field or {}).get("evidence_ids", []) if e in loc}) if known(field) else []

    rng = random.Random(args.seed)
    exps = [e for e in read_jsonl(data / "silver/experiments/experiments.jsonl") if e.get("split") in SPLITS]
    rng.shuffle(exps)
    queries, seen, per_paper = [], set(), defaultdict(int)

    def add(qtype: str, split: str, text: str, source: dict, relevant: dict | None = None, papers: list | None = None):
        key = (split, qtype, text.lower())
        if key in seen:
            return
        seen.add(key)
        qid = "rq:" + hashlib.sha256(f"{VERSION}|{qtype}|{split}|{text}".encode()).hexdigest()[:16]
        queries.append({"query_id": qid, "split": split, "type": qtype, "query": text, "source": source,
                        **({"relevant_chunks": relevant} if relevant is not None else {"relevant_papers": papers})})

    count = defaultdict(int)
    for e in exps:
        model, bench = (e.get("model") or {}).get("model_name"), (e.get("evaluation") or {}).get("benchmark")
        if not known(model):
            continue
        split, pid = e["split"], e["paper_id"]
        for o in e.get("outcomes") or []:  # result lookups
            if count[(split, "result")] >= PER_SPLIT["result"] or per_paper[(pid, "result")] >= 2:
                break
            if known(bench) and known(o.get("metric")) and chunks(o.get("score")):
                rel = {c: 1 for f in (model, bench, o.get("metric")) for c in chunks(f)} | {c: 2 for c in chunks(o.get("score"))}
                add("result", split, f"What {o['metric']['value']} does {model['value']} achieve on {bench['value']}?",
                    {"experiment_id": e["experiment_id"], "fields": ["evaluation.score"]}, rel)
                count[(split, "result")] += 1
                per_paper[(pid, "result")] += 1
                break
        for name, label in PROTOCOL.items():  # protocol lookups
            f = (e.get("training") or {}).get(name)
            if count[(split, "protocol")] >= PER_SPLIT["protocol"] or per_paper[(pid, "protocol")] >= 2:
                break
            if chunks(f):
                on = f" on {bench['value']}" if known(bench) else ""
                rel = {c: 1 for c in chunks(model)} | {c: 2 for c in chunks(f)}
                add("protocol", split, f"What {label} was used to train {model['value']}{on}?",
                    {"experiment_id": e["experiment_id"], "fields": [f"training.{name}"]}, rel)
                count[(split, "protocol")] += 1
                per_paper[(pid, "protocol")] += 1
                break

    exp_paper = {e["experiment_id"]: e["paper_id"] for e in read_jsonl(data / "silver/experiments/experiments.jsonl")}
    for t in read_jsonl(data / "silver/tasks/research_tasks.jsonl"):
        m = re.fullmatch(r"What prior experiments exist on (.+) measured with (.+)\?", t["question"])
        if m and m.group(1).strip().lower() == m.group(2).strip().lower():
            continue  # degenerate silver task (benchmark == metric, e.g. "Benchmarks"), not a real question
        if t["task_type"] == "prior_art_experiment_search" and t["split"] in SPLITS and t.get("expected_behavior") == "ANSWER":
            papers = sorted({exp_paper[x.strip()] for x in t["expected_answer"].split(",") if x.strip() in exp_paper})
            if papers:
                add("prior_art", t["split"], t["question"], {"task_id": t["task_id"]}, papers=papers)

    out = data / "eval" / "retrieval" / f"v{args.release}"
    out.mkdir(parents=True, exist_ok=True)
    queries.sort(key=lambda q: (q["split"], q["type"], q["query_id"]))
    (out / "queries.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    counts = defaultdict(int)
    for q in queries:
        counts[f"{q['split']}/{q['type']}"] += 1
    meta = {"version": VERSION, "release": args.release, "seed": args.seed, "corpus_sha256": bm25["corpus_sha256"], "counts": dict(sorted(counts.items())),
            "relevance": "deep evidence located in parsed chunks (grade 2 decisive field, grade 1 supporting fields); prior_art: papers"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta["counts"]), "->", out)


if __name__ == "__main__":
    sys.exit(main())
