"""Generalization and Transferability Analyzer: where else has a model been evaluated, and what changed?

Every result carries its configuration conditions (services/analysis/configs.py): the pre-training data and the
adaptation (fine-tuned, linear probe, zero-shot, ...) its records and paper state; a direct transfer lists the ones that
differ between its source and target rows, so a change of setup is never read as a change caused by the dataset.

The model's records are found exactly as in the claim auditor (services.analysis.claims.matched: every token of the name,
never another size or variant). The source setting is the benchmark the user names, else the one with the most records.
A direct transfer is one paper reporting the model on the source and on another benchmark (the same authors' setup);
a score change is shown only for the same metric name, and is descriptive: different benchmarks are not a controlled
measurement of transfer. Results from other papers are "evaluated elsewhere" (indirect). Untested combinations are
benchmarks of the model's tasks that other papers report results on and that no record of this model has: absence of a
record is never evidence that the model fails there. Distribution-shift statements are quoted from the papers' own
limitation and discussion sections, never generated.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from services.analysis import configs as C
from services.analysis import facts as F
from services.analysis.claims import SPLIT, _bench, _names, datasets_in, matched
from services.canonicalization.normalize import canonical_label
from services.evidence import grounding as G

VERSION = "transfer-0.1.0"
SECTIONS = r"limitation|discussion|conclusion|future work|broader impact"
SHIFT = r"shift|transfer|generali[sz]|out-of-distribution|domain|robust|unseen"
MAX_ROWS = 40


def _topics(index, xid: str) -> list[tuple[str, str]]:
    return [(link["entity_id"], link["raw"]) for link in index.exp[xid]["links"] if link["node_type"] == "Topic"]


def _protocol(models: dict, xid: str) -> str | None:
    e = models.get(xid)
    if e is None:
        return None
    f = F.fact("evaluation.evaluation_protocol", F.field_value(e, "evaluation.evaluation_protocol"))
    return (f["raw"] or f["value"]) if f["status"] in F.STATED else None


def _result(index, xid: str, r: dict, kind: str, protocol: str | None, bench: str, view: dict | None = None) -> dict:
    pid = index.exp[xid]["paper_id"]
    setup = {}
    for c in ("pretraining_data", "adaptation"):  # configuration conditions (services/analysis/configs.py)
        v = C.condition(view, c) if view else {"value": None, "status": F.UNKNOWN, "basis": None}
        setup[c] = {"value": v["value"], "status": v["status"], "basis": v["basis"]}
    return {"experiment": xid, "label": index.exp[xid]["label"], "paper": pid, "title": index.paper.get(pid, {}).get("title"),
            "year": index.paper.get(pid, {}).get("year"), "model": (_names(index, xid) or [None])[0], "match": kind, "benchmark": bench,
            "metric": r["metric"], "score": r["score"], "split": r["split"], "protocol": protocol, "setup": setup, "evidence_ids": r["evidence_ids"][:2]}


def _setup_difference(s: dict, t: dict) -> list[str]:
    """Configuration conditions both rows state with different values (pre-training data, adaptation)."""
    return [f"{c.replace('_', ' ')}: {s['setup'][c]['value']} vs {t['setup'][c]['value']}" for c in ("pretraining_data", "adaptation")
            if s["setup"][c]["value"] and t["setup"][c]["value"]
            and canonical_label(str(s["setup"][c]["value"])) != canonical_label(str(t["setup"][c]["value"]))]


def analyze(index, model: str, tr, source: str | None = None, target: str | None = None) -> dict:
    model = " ".join((model or "").split())
    if source is None and (named := datasets_in(index, model)):  # "ViT-L/16 on ImageNet" (the chat): the dataset is the source
        head = re.split(r"\s+(?:on|from)\s+(?:the\s+)?" + re.escape(named[0]), model, flags=re.I)[0]
        if head != model and G.tokens(head):
            model, source = head.strip(), named[0]
    if not G.tokens(model):
        raise ValueError("Name the model or method to analyse.")
    with tr.stage("experiment_retrieval", method="model-name token coverage over every corpus experiment (no similarity)") as st:
        exps = matched(index, [model])[0][0]  # (one {experiment: match kind} per mention, ambiguous count)
        by_bench: dict[str, dict] = {}
        for xid, kind in exps.items():
            b = _bench(index, xid)
            if b is None:
                continue
            slot = by_bench.setdefault(b["entity_id"], {"name": index.ent.get(b["entity_id"], {}).get("name") or b["raw"], "records": [],
                                                        "papers": set(), "tasks": Counter()})
            slot["records"].append((xid, kind))
            slot["papers"].add(index.exp[xid]["paper_id"])
            slot["tasks"].update(t for _, t in _topics(index, xid))
        st.detail.update(matched=len(exps), direct=sum(k == "direct" for k in exps.values()), benchmarks=len(by_bench))
    inputs = {"model": model, "source": source, "target": target}
    if not by_bench:
        return {"kind": "transfer", "version": VERSION, "model": model, "verdict": "NO_EVIDENCE", "source": None, "benchmarks": [], "transfers": [],
                "elsewhere": [], "untested": [], "limitations": [], "research_questions": [f"Where has {model} been evaluated?"],
                "summary": {"benchmarks": 0, "papers": 0, "transfers": 0, "measured": 0, "elsewhere": 0, "untested": 0}, "inputs": inputs,
                "notes": [f"No corpus experiment record names {model}. Absence from the corpus says nothing about the model."]}
    with tr.stage("field_facts", method="results per benchmark; evaluation protocol from the records") as st:
        models = index.experiments({x for s in by_bench.values() for x, _ in s["records"]})
        if source:  # the dataset itself ("ADE20K", "ADE20K val"); its shifted variants ("ImageNet-V2") stay transfer targets
            exact = [b for b, s in by_bench.items() if {t for t in G.tokens(s["name"]) if not SPLIT.fullmatch(t)} == G.tokens(source)]
            src = exact or [b for b, s in by_bench.items() if G.dataset_match(source, [s["name"]]) == 2]
            if not src:
                raise ValueError(f"No record of {model} on {source}. It is evaluated on: " + ", ".join(sorted(s["name"] for s in by_bench.values())[:12]))
        else:
            src = [max(by_bench, key=lambda b: (len(by_bench[b]["papers"]), len(by_bench[b]["records"])))]
        views = C.views(index, models)
        rows: dict[str, list] = defaultdict(list)
        for b, s in by_bench.items():
            for xid, kind in s["records"]:
                for r in index.exp[xid]["results"]:
                    if r["metric"] and G.number_of(r["score"]) is not None:
                        rows[b].append(_result(index, xid, r, kind, _protocol(models, xid), s["name"], views.get(xid)))
        st.detail.update(source=[by_bench[b]["name"] for b in src], results=sum(map(len, rows.values())))
    with tr.stage("comparison", method="same paper, source vs other benchmark; a score change only for the same metric name") as st:
        targets = [b for b in by_bench if b not in src and (not target or G.dataset_match(target, [by_bench[b]["name"]]) == 2)]
        src_by_paper = defaultdict(list)
        for b in src:
            for r in rows[b]:
                src_by_paper[r["paper"]].append(r)
        transfers, elsewhere = [], []
        for b in targets:
            for t in rows[b]:
                same = src_by_paper.get(t["paper"], [])
                if not same:
                    elsewhere.append(t)
                    continue
                s = next((s for s in same if canonical_label(s["metric"]) == canonical_label(t["metric"])), same[0])
                same_metric = canonical_label(s["metric"]) == canonical_label(t["metric"])
                # a change only between a dataset and a shifted variant of it (ImageNet -> ImageNet-V2): across unrelated datasets
                # (ADE20K -> Cityscapes) a difference of scores measures the datasets, not the transfer
                variant = bool(G.dataset_match(s["benchmark"], [t["benchmark"]]) or G.dataset_match(t["benchmark"], [s["benchmark"]]))
                measured = same_metric and variant
                transfers.append({"source": s, "target": t, "same_metric": same_metric, "shifted_variant": variant,
                                  "setup_differences": _setup_difference(s, t),
                                  "change": round(G.number_of(t["score"]) - G.number_of(s["score"]), 6) if measured else None,
                                  "note": ("Same paper, same metric, a variant of the source dataset: a descriptive change, not a controlled transfer measurement."
                                           if measured else "Same paper, different datasets: both scores are reported, their difference measures the datasets, not transfer."
                                           if same_metric else "Same paper, different metrics: the scores are not compared.")})
        st.detail.update(targets=len(targets), transfers=len(transfers), measured=sum(t["change"] is not None for t in transfers), elsewhere=len(elsewhere))
    with tr.stage("coverage", method="benchmarks of the model's tasks with results from other papers but no record of this model") as st:
        task_ids = {tid for xid in exps for tid, _ in _topics(index, xid)}
        cand: dict[str, dict] = {}
        for tid in task_ids:
            for xid in index.ent_exps.get(tid, []):
                b = _bench(index, xid)
                if b is None or b["entity_id"] in by_bench or xid in exps:
                    continue
                c = cand.setdefault(b["entity_id"], {"name": index.ent.get(b["entity_id"], {}).get("name") or b["raw"], "papers": set(),
                                                     "models": set(), "task": next((t for t_id, t in _topics(index, xid) if t_id == tid), None)})
                c["papers"].add(index.exp[xid]["paper_id"])
                c["models"].update(_names(index, xid)[:1])
        untested = sorted((c for c in cand.values() if len(c["papers"]) >= 2), key=lambda c: (-len(c["papers"]), c["name"].casefold()))[:12]
        st.detail.update(tasks=len(task_ids), untested=len(untested))
    with tr.stage("assessment", method="limitation / discussion passages of the model's papers that mention shift or transfer (quoted)") as st:
        pids = sorted({index.exp[x]["paper_id"] for x in exps})
        passages = index.q("select c.id, c.paper_id, c.section_title, c.text, c.unit_id, c.redistribution_allowed, p.arxiv_id from rx.chunks c "
                           "join rx.papers p on p.id = c.paper_id where c.paper_id = any(%s) and c.section_title ~* %s and c.text ~* %s "
                           "order by c.id limit 8", (pids, SECTIONS, SHIFT))
        limitations = [{"chunk_id": c, "paper": pid, "title": index.paper.get(pid, {}).get("title"), "section": sec,
                        "text": t[:900] if allowed is not False else None, "text_withheld": allowed is False,
                        "url": f"https://arxiv.org/html/{arxiv}#{u}" if arxiv and u else None} for c, pid, sec, t, u, allowed, arxiv in passages]
        st.detail.update(papers=len(pids), passages=len(limitations))
    source_rows = [r for b in src for r in rows[b]]
    shown = source_rows[:MAX_ROWS] + [t[k] for t in transfers[:MAX_ROWS] for k in ("source", "target")] + elsewhere[:MAX_ROWS]
    ids = [i for r in shown for i in r.get("evidence_ids", [])]
    ev = {e["id"]: e for e in index.evidence_detail(ids, limit=400)} if ids else {}
    for r in shown:
        if "evidence_ids" in r:  # a source row can also be a transfer's source: attach once
            r["evidence"] = [ev[i] for i in r.pop("evidence_ids") if i in ev]
    transfers.sort(key=lambda t: (t["change"] is None, not t["same_metric"], t["target"]["benchmark"].casefold()))
    return {"kind": "transfer", "version": VERSION, "model": model, "verdict": "OBSERVED" if transfers else "NO_DIRECT_TRANSFER",
            "source": {"benchmarks": [by_bench[b]["name"] for b in src], "results": source_rows[:MAX_ROWS],
                       "chosen": "named by you" if source else "the benchmark with the most records of this model"},
            "benchmarks": sorted(({"name": s["name"], "records": len(s["records"]), "papers": len(s["papers"]),
                                   "tasks": [t for t, _ in s["tasks"].most_common(2)]} for s in by_bench.values()), key=lambda s: (-s["papers"], -s["records"])),
            "transfers": transfers[:MAX_ROWS], "elsewhere": elsewhere[:MAX_ROWS],
            "untested": [{"name": c["name"], "task": c["task"], "papers": len(c["papers"]), "models": sorted(c["models"])[:5]} for c in untested],
            "limitations": limitations, "research_questions": [f"How does {model} perform on {c['name']}?" for c in untested[:3]],
            "summary": {"benchmarks": len(by_bench), "papers": len({index.exp[x]["paper_id"] for x in exps}), "transfers": len(transfers),
                        "measured": sum(t["change"] is not None for t in transfers), "elsewhere": len(elsewhere), "untested": len(untested)},
            "inputs": inputs,
            "notes": ["A direct transfer is one paper reporting the model on two benchmarks; a score change between benchmarks is descriptive, "
                      "not a controlled measurement of transfer.",
                      "Untested means no record of this model in the corpus; it is not evidence that the model fails there.",
                      "Shift and limitation statements are quoted from the papers; nothing about future performance is predicted."]}
