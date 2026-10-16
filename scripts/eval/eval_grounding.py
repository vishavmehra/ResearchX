"""Evaluate field-level evidence grounding end to end through the real pipeline (docs/EVALUATION_SPEC.md).

Cases (TEST + TEMPORAL_TEST only; VALIDATION tuned retrieval):
  positive       protocol / result questions of the retrieval eval set; gold value + gold evidence from the frozen experiment
  negative       the same questions with an invented model, or an invented dataset: the right behaviour is to abstain
  contradiction  score questions about experiments with a recorded corpus contradiction: the right behaviour is to report it
Metrics: false-abstention rate, sufficiency accuracy, parameter / numerical accuracy, wrong-answer rate, evidence recall,
configuration-matching accuracy, citation correctness, entailment accuracy, citation completeness, unsupported-claim rate,
contradiction accuracy, false-conflict rate. Citations are re-validated independently from the database. External search is
off by default (corpus grounding only, no web credits); --external turns it on. Usage:
    python scripts/eval/eval_grounding.py --release 0.2.0 --positives 40 --negatives 10 --workers 3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.datasets.store import config  # noqa: E402
from services.env import load_env  # noqa: E402
from services.evidence import grounding as G  # noqa: E402

VERSION = "grounding-eval-0.1.0"
ANSWERED = {"SUFFICIENT", "PARTIAL", "CONFLICTING_EVIDENCE", "AMBIGUOUS_EXPERIMENT_MATCH"}
ABSTAINED = {"INSUFFICIENT_EVIDENCE", "NO_RELEVANT_SOURCE_FOUND"}
SYLLABLES = ["zor", "quel", "vash", "trin", "pex", "lumo", "drak", "sil", "neb", "ka"]
EXCLUDED: dict[str, int] = {}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def number(v) -> float | None:
    """A reported value as a number: "82.3%" is 82.3, "73.9±3.1" is 73.9 (as the evidence's normalized value)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return G.number_of(re.sub(r"\s*±.*$", "", G.display(v)).rstrip("% ")) if v is not None else None


def same(a, b) -> bool:
    x, y = number(a), number(b)
    if x is not None and y is not None:
        return abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))
    return G.canon(str(a)) == G.canon(str(b))


def key_of(v) -> str:
    n = number(v)
    return repr(n) if n is not None else G.canon(str(v))


def gold_of(q: dict, rec: dict) -> dict | None:
    """Gold value + evidence ids of the field the query asks, from the frozen experiment record."""
    field = q["source"]["fields"][0]
    if field == "evaluation.score":
        for o in rec.get("outcomes") or []:
            metric = (o.get("metric") or {}).get("value")
            if metric and f"What {metric} does " in q["query"] and (o.get("score") or {}).get("value") is not None:
                return {"field": field, "value": o["score"]["value"], "evidence_ids": o["score"].get("evidence_ids") or []}
        return None
    f = (rec.get("training") or {}).get(field.split(".", 1)[1]) or {}
    return {"field": field, "value": f.get("value"), "evidence_ids": f.get("evidence_ids") or []} if f.get("value") is not None else None


def fake_name(seed: str) -> str:
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return "".join(SYLLABLES[(h >> (4 * i)) % len(SYLLABLES)] for i in range(3)).capitalize() + f"-{h % 90 + 10}"


def build_cases(db, release: str, n_pos: int, n_neg: int, n_con: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    qs = [q for q in read_jsonl(ROOT / "data" / "eval" / "retrieval" / f"v{release}" / "queries.jsonl")
          if q["split"] in ("TEST", "TEMPORAL_TEST") and q["type"] in ("protocol", "result")]
    recs = {x: (p, r) for x, p, r in db.execute("select id, paper_id, record from rx.experiments where id = any(%s)",
                                                (sorted({q["source"]["experiment_id"] for q in qs}),)).fetchall()}
    cases: list[dict] = []
    # silver-data noise: some result questions name a table header as the metric ("What NYC does X achieve on NYC?")
    metric_of = lambda q: (re.match(r"What (.+?) does ", q["query"]) or [None, ""])[1]  # noqa: E731
    malformed = [q for q in qs if q["type"] == "result" and not G.METRIC.search(metric_of(q))]
    EXCLUDED["malformed_result_questions"] = len(malformed)
    EXCLUDED["result_questions"] = sum(q["type"] == "result" for q in qs)
    qs = [q for q in qs if q not in malformed]
    for kind in ("protocol", "result"):
        pool = [q for q in qs if q["type"] == kind]
        rng.shuffle(pool)
        for q in pool:
            if sum(c["type"] == kind for c in cases) >= n_pos:
                break
            paper, rec = recs.get(q["source"]["experiment_id"], (None, {}))
            gold = gold_of(q, rec)
            if gold:
                cases.append({"case": "positive", "type": kind, "query": q["query"], "query_id": q["query_id"], "gold": gold,
                              "experiment_id": q["source"]["experiment_id"], "paper_id": paper,
                              "gold_chunks": sorted(c for c, g in q["relevant_chunks"].items() if g == 2),
                              "model": ((rec.get("model") or {}).get("model_name") or {}).get("value"),
                              "benchmark": ((rec.get("evaluation") or {}).get("benchmark") or {}).get("value")})
    protocol = [c for c in cases if c["type"] == "protocol" and c["model"] and c["benchmark"] and c["model"] in c["query"]]
    rng.shuffle(protocol)
    for i, c in enumerate(protocol[: 2 * n_neg]):  # half invented models, half invented datasets
        if i % 2 == 0:
            q, kind = c["query"].replace(c["model"], fake_name(c["query_id"] + "m"), 1), "invented model"
        else:
            q, kind = c["query"].replace(f" on {c['benchmark']}", f" on the {fake_name(c['query_id'] + 'd')} benchmark", 1), "invented dataset"
        if q != c["query"]:
            cases.append({"case": "negative", "type": kind, "query": q, "query_id": c["query_id"] + ":neg"})
    rows = db.execute("select c.id, c.subject_id, c.record, e.record from rx.contradictions c join rx.experiments e on e.id = c.subject_id "
                      "where c.field = 'evaluation.score' order by c.id limit %s", (n_con,)).fetchall()
    for cid, xid, crec, erec in rows:
        model = ((erec.get("model") or {}).get("model_name") or {}).get("value")
        bench = ((erec.get("evaluation") or {}).get("benchmark") or {}).get("value")
        metric = next(((o.get("metric") or {}).get("value") for o in erec.get("outcomes") or [] if (o.get("metric") or {}).get("value")), None)
        if model and bench and metric:
            cases.append({"case": "contradiction", "type": "recorded score contradiction", "query": f"What {metric} does {model} achieve on {bench}?",
                          "query_id": cid, "contradiction_id": cid, "experiment_id": xid, "values": [crec.get("value_a"), crec.get("value_b")]})
    return cases


def source_texts(db, result: dict) -> dict[str, str]:
    """Cited source text straight from the database (independent of what the run returned)."""
    chunks = [s["chunk_id"] for s in result["sources"] if s.get("kind") == "internal" and s.get("chunk_id")]
    records = [s["chunk_id"] for s in result["sources"] if s.get("kind") == "record"]
    text = dict(db.execute("select id, text from rx.chunks where id = any(%s)", (chunks,)).fetchall()) if chunks else {}
    if records:
        text |= {i: r.get("evidence_text") or "" for i, r in db.execute("select id, record from rx.evidence where id = any(%s)", (records,)).fetchall()}
    return {s["sid"]: text.get(s.get("chunk_id"), s.get("text") or "") for s in result["sources"]}


def score_case(db, c: dict, r: dict) -> dict:
    parsed = r.get("evidence_status") is not None  # the question was read as field request(s): field-level grounding ran
    st = r["evidence_status"] if parsed else ("FAILED" if r["mode"] == "fallback" else f"PASSAGE_{r['status']}")
    field = (r.get("fields") or [{}])[0]
    values = field.get("values") or []
    out = {k: c.get(k) for k in ("case", "type", "query", "query_id")} | {
        "research_run_id": r["research_run_id"], "status": r["status"], "evidence_status": st, "total_ms": r["total_ms"],
        "answer": r["answer"], "values": values, "close_match": field.get("close_match"), "request_parsed": parsed,
        "answered": st in ANSWERED or (not parsed and r["status"] in ("ANSWERED", "PARTIAL", "CONFLICTING"))}
    texts = source_texts(db, r) if r["claims"] else {}
    checks = []
    for cl in r["claims"]:
        src = texts.get(cl["sources"][0], "")
        in_source = len(G.canon(cl["quote"])) >= 2 and G.canon(cl["quote"]) in G.canon(src)
        # passage-path claims carry no field: the runtime already required their numbers inside the verbatim quote
        checks.append({"in_source": in_source, "entailed": G.validate(cl["field"], cl["value"], cl["quote"], src)[0] if cl.get("field") else in_source})
    out |= {"claims": len(checks), "claims_in_source": sum(x["in_source"] for x in checks), "claims_entailed": sum(x["entailed"] for x in checks)}
    out["complete_citations"] = (bool(values) and {key_of(v) for v in values} <= {key_of(cl.get("value")) for cl in r["claims"]}) if parsed \
        else bool(r["claims"]) and all(cl.get("sources") for cl in r["claims"])
    stages = {s["stage"]: s for s in r["stages"]}
    if c["case"] == "positive":
        gold = c["gold"]["value"]
        detail = (stages.get("experiment_extraction") or {}).get("detail") or {}
        papers = detail.get("papers") or {}
        cited_chunks = {s["chunk_id"] for s in r["sources"] if s.get("role") == "SUPPORTS"}
        normalized = {key_of(cl.get("value")): cl.get("value") for cl in r["claims"]}  # "73.9±3.1" is 73.9; "73.2%" is 73.2
        out |= {"gold": gold, "gold_in_values": any(same(v, gold) for v in list(values) + list(normalized.values())),
                "exact_correct": st in ("SUFFICIENT", "PARTIAL") and len(normalized) == 1 and same(next(iter(normalized.values())), gold),
                "numeric_gold": G.number_of(gold) is not None,
                "evidence_recalled": bool(set(c["gold"]["evidence_ids"]) & {cl.get("evidence_id") for cl in r["claims"]})
                or bool(cited_chunks & set(c["gold_chunks"])),
                "gold_paper_retrieved": c["paper_id"] in papers, "gold_paper_matched": papers.get(c["paper_id"]) in ("EXACT", "STRONG"),
                "gold_experiment_matched": any(m.get("experiment_id") == c["experiment_id"] for m in detail.get("matched") or []),
                "conflict_flagged": st == "CONFLICTING_EVIDENCE"}
    elif c["case"] == "contradiction":
        out |= {"conflict_flagged": st == "CONFLICTING_EVIDENCE",
                "both_values_reported": all(any(same(v, w) for v in values) for w in c["values"] if w is not None),
                "contradiction_attached": any(x["id"] == c["contradiction_id"] for x in r.get("contradictions") or [])}
    return out


def rate(rows: list[dict], key, where=lambda r: True) -> dict:
    sel = [r for r in rows if where(r)]
    n = sum(bool(key(r) if callable(key) else r.get(key)) for r in sel)
    return {"value": round(n / len(sel), 3) if sel else None, "n": n, "of": len(sel)}


def metrics(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["case"] == "positive"]
    neg = [r for r in rows if r["case"] == "negative"]
    con = [r for r in rows if r["case"] == "contradiction"]
    claims = sum(r["claims"] for r in rows)

    def claim_rate(key: str, invert=False) -> dict:
        n = sum(r[key] for r in rows)
        n = claims - n if invert else n
        return {"value": round(n / claims, 3) if claims else None, "n": n, "of": claims}

    return {
        "request_parsed": rate(pos, "request_parsed"),
        "false_abstention_rate": rate(pos, lambda r: not r["answered"]),
        "false_abstention_rate_parsed": rate(pos, lambda r: not r["answered"], lambda r: r["request_parsed"]),
        "sufficiency_accuracy": rate(rows, lambda r: r["answered"] == (r["case"] != "negative")),
        "parameter_accuracy": rate(pos, "exact_correct", lambda r: r["type"] == "protocol"),
        "parameter_accuracy_when_answered": rate(pos, "exact_correct", lambda r: r["type"] == "protocol" and r["answered"]),
        "parameter_gold_among_values": rate(pos, "gold_in_values", lambda r: r["type"] == "protocol"),
        "score_accuracy": rate(pos, "exact_correct", lambda r: r["type"] == "result"),
        "score_accuracy_when_answered": rate(pos, "exact_correct", lambda r: r["type"] == "result" and r["answered"]),
        "numerical_accuracy": rate(pos, "exact_correct", lambda r: r["numeric_gold"]),
        "wrong_answer_rate": rate(pos, lambda r: not r["gold_in_values"], lambda r: r["answered"] and r["request_parsed"]),
        "evidence_recall": rate(pos, "evidence_recalled"),
        "gold_paper_retrieved": rate(pos, "gold_paper_retrieved"),
        "configuration_matching_accuracy": rate(pos, "gold_paper_matched", lambda r: r["gold_paper_retrieved"]),
        "gold_experiment_matched": rate(pos, "gold_experiment_matched", lambda r: r["gold_paper_retrieved"]),
        "citation_correctness": claim_rate("claims_in_source"),
        "entailment_accuracy": claim_rate("claims_entailed"),
        "unsupported_claim_rate": claim_rate("claims_entailed", invert=True),
        "citation_completeness": rate(rows, "complete_citations", lambda r: r["answered"]),
        "negative_answered_rate": rate(neg, "answered"),
        "negative_answered_as_exact": rate(neg, lambda r: r["answered"] and not r["close_match"]),
        "contradiction_accuracy": rate(con, "conflict_flagged"),
        "contradiction_both_values_reported": rate(con, "both_values_reported"),
        "false_conflict_rate": rate(pos, "conflict_flagged"),
        "status_counts": {k: dict(Counter(r["evidence_status"] for r in rows if r["case"] == k)) for k in ("positive", "negative", "contradiction")},
        "latency_ms_median": sorted(r["total_ms"] for r in rows)[len(rows) // 2] if rows else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="0.2.0")
    ap.add_argument("--positives", type=int, default=40, help="per type (protocol, result)")
    ap.add_argument("--negatives", type=int, default=10, help="per kind (invented model, invented dataset)")
    ap.add_argument("--contradictions", type=int, default=24, help="recorded score contradictions to test (24 in v0.2.0)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--external", action="store_true", help="allow the external evidence-gap search (web credits)")
    args = ap.parse_args()
    load_env()
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator
    from services.storage.db import connect

    db = connect(autocommit=True)
    cases = build_cases(db, args.release, args.positives, args.negatives, args.contradictions, args.seed)
    cfg = copy.deepcopy(config("runtime"))
    cfg["external"]["enabled"] = args.external
    orch = Orchestrator(cfg)
    t0 = time.time()

    def one(c):
        try:
            return c, orch.run(c["query"], RetrievalScope(None)), None
        except Exception as e:  # recorded in the report, never silently dropped
            return c, None, f"{type(e).__name__}: {e}"

    rows, errors = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (c, r, err) in enumerate(pool.map(one, cases), 1):
            if err:
                errors.append({"query": c["query"], "error": err})
            else:
                rows.append(score_case(db, c, r))
            print(f"[{i}/{len(cases)}] {c['case']:<13} {('ERROR' if err else rows[-1]['evidence_status']):<26} {c['query'][:80]}", flush=True)
    m = metrics(rows)
    report = {"version": VERSION, "grounding_version": G.VERSION, "pipeline_version": cfg["pipeline_version"], "release": args.release,
              "seed": args.seed, "external_search": args.external, "cases": dict(Counter(c["case"] for c in cases)), "excluded": EXCLUDED,
              "errors": errors,
              "wall_s": round(time.time() - t0, 1), "metrics": m, "rows": rows}
    out = ROOT / "data" / "reports" / f"grounding_eval_v{args.release}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [f"# Grounding evaluation v{args.release}", "",
             f"{VERSION} · {G.VERSION} · {cfg['pipeline_version']} · seed {args.seed} · external search "
             f"{'on' if args.external else 'off (corpus only)'} · {len(rows)} runs {report['cases']} · {len(errors)} errors · wall {report['wall_s']} s", "",
             "| Metric | Value | n / of |", "|---|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v['value']} | {v['n']} / {v['of']} |" for k, v in m.items() if isinstance(v, dict) and "value" in v]
    lines += ["", f"Status counts: `{json.dumps(m['status_counts'])}` · median latency {m['latency_ms_median']} ms", "",
              f"Excluded from sampling: {EXCLUDED.get('malformed_result_questions', 0)} of {EXCLUDED.get('result_questions', 0)} result questions "
              "whose 'metric' the grounding metric lexicon does not recognise: mostly table-header noise in the silver data (dataset or "
              "model names, 'Overall'), plus about a dozen real metrics the lexicon lacks (ACC, EM, CIDEr, Recall@k), which take the "
              "passage path.", "",
              "Positives are answerable by construction (gold evidence located in the corpus); negatives (invented model or dataset) "
              "must abstain; recorded contradictions must be reported. Citation correctness: the quoted span is in the cited source "
              "text, re-read from the database. Entailment: the span states the field and value (deterministic validator). "
              "Completeness: every reported value has a citation. Parameter/score accuracy: a single answered value equal to the "
              "gold value; 'gold among values' also credits ambiguous or conflicting answers that list it."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in m.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
