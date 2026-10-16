"""Paper-to-Paper Diff: two to four papers side by side, field by field, with a fairness verdict.

A paper's cell for a field aggregates what its experiment records state: one value, several values (VARIES: different
experiments of the paper state different settings), CONFLICTING (the sources disagree) or UNKNOWN (no record states it).
Training settings are shown per stage (services/analysis/configs.py): a pre-training learning rate is compared only with
pre-training learning rates, never with a fine-tuning one; a value whose stage is not stated has its own row. Rows compare
the cells across papers (SAME / DIFFERENT / PARTIAL / CONFLICTING / UNKNOWN). Limitations and conclusions are the papers'
own passages, quoted, never summarised. The fairness verdict is DERIVED by the comparison engine
(services/comparison/fields.py): the best-aligned experiment pair on a shared benchmark; UNKNOWN is never a match.
Papers may be runtime papers read at request time (services/analysis/runtime_papers.py, passed in through its Overlay):
the same code compares them, and each is marked with its origin and provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from services.analysis import configs as C
from services.analysis import facts as F
from services.canonicalization.normalize import canonical_label
from services.comparison.fields import LABELER_VERSION, compare
from services.evidence.grounding import TEXT_FIELDS, condition_value

VERSION = "paper-diff-0.2.0"  # 0.2.0: training settings per stage, configurations per paper, runtime papers
SECTIONS = (
    ("Research question", ("research.research_question", "research.hypothesis", "research.task", "research.domain")),
    ("Model and method", ("model.model_name", "model.architecture", "model.parameter_count", "model.base_model", "method.method_name",
                          "method.objective", "method.training_strategy")),
    ("Data", ("evaluation.benchmark", "dataset.dataset_name", "dataset.dataset_version", "dataset.preprocessing", "evaluation.dataset_split",
              "population.languages", "population.domains")),
    ("Training", ("training.optimizer", "training.learning_rate", "training.learning_rate_schedule", "training.batch_size", "training.epochs",
                  "training.steps", "training.warmup_steps", "training.weight_decay", "training.dropout", "training.seed", "training.precision")),
    ("Evaluation", ("method.baseline", "evaluation.evaluation_protocol", "evaluation.metric_definition", "evaluation.statistical_test")),
    ("Compute and code", ("compute.gpu_model", "compute.gpu_count", "compute.training_time", "artifacts.repository_url", "artifacts.checkpoint_url")),
)
TEXT_SECTIONS = r"limitation|conclusion|discussion|future work"
VERDICT = {"ALIGNED_EXPERIMENT": "COMPARABLE", "INSUFFICIENT_EVIDENCE": "CONDITIONALLY_COMPARABLE", "CONFLICTING_EVIDENCE": "CONDITIONALLY_COMPARABLE",
           "MATERIALLY_DIFFERENT": "NOT_COMPARABLE", "RELATED_CONTEXT": "NOT_COMPARABLE"}
RANK = {"ALIGNED_EXPERIMENT": 0, "INSUFFICIENT_EVIDENCE": 1, "CONFLICTING_EVIDENCE": 2, "MATERIALLY_DIFFERENT": 3, "RELATED_CONTEXT": 4}
MAX_PAPERS, MAX_VALUES, MAX_PAIRS = 4, 4, 48


def cell(exps: dict, path: str, paper_level: list | None = None) -> dict:
    """What a paper's experiment records state for one field; when none does, what the paper's own text states for it
    without assigning it to an experiment (paper-level evidence, `scope: paper`: it may not hold for every experiment)."""
    stated: dict[str, list] = defaultdict(list)
    conflicting = []
    for xid, e in exps.items():
        f = F.fact(path, F.field_value(e, path))
        if f["status"] in F.STATED:
            stated[canonical_label(str(f["raw"] if f["raw"] not in (None, "") else f["value"]))].append((xid, f))
        elif f["status"] == F.CONFLICTING:
            conflicting.append(xid)
    values = [{"value": fs[0][1]["raw"] if fs[0][1]["raw"] not in (None, "") else fs[0][1]["value"], "unit": fs[0][1]["unit"],
               "status": fs[0][1]["status"], "scope": "experiment", "experiments": [x for x, _ in fs][:3], "experiment_count": len(fs),
               "evidence_ids": fs[0][1]["evidence_ids"][:1], "key": key}
              for key, fs in sorted(stated.items(), key=lambda kv: -len(kv[1]))]
    if not values and not conflicting and paper_level:
        values = paper_level
    status = F.CONFLICTING if conflicting else F.UNKNOWN if not values else "VARIES" if len(values) > 1 else values[0]["status"]
    return {"status": status, "values": values[:MAX_VALUES], "more_values": max(0, len(values) - MAX_VALUES), "conflicting_experiments": conflicting[:3]}


def paper_level(index, pids: list[str], paths: list[str]) -> dict[tuple[str, str], list]:
    """Values the papers' own text states for these fields without an experiment assignment (rx.evidence, PAPER_
    locations, experiment_id null; plus request-time quotes of the same papers), grouped per (paper, field) and
    deduplicated, each with the training stage its statement names (C.stage_of; None when not stated)."""
    rows = index.q("select id, paper_id, field_name, record from rx.evidence where paper_id = any(%s) and experiment_id is null "
                   "and location_kind like 'PAPER_%%' and field_name = any(%s) order by id", (pids, paths))
    want = set(paths)
    rows = list(rows) + [(rid, r["paper_id"], r["field"], {"extracted_value": r.get("extracted_value"), "normalized_value": r.get("normalized_value"),
                                                           "evidence_text": r.get("text"), "section": r.get("section"), "applies_to": r.get("applies_to"),
                                                           "stage": r.get("stage")})
                         for rid, r in list(index.runtime_evidence.items()) if r.get("paper_id") in pids and r.get("experiment_id") is None and r.get("field") in want]
    out: dict[tuple[str, str], dict] = defaultdict(dict)
    for eid, pid, fld, rec in rows:
        raw = rec.get("extracted_value") if rec.get("extracted_value") not in (None, "") else rec.get("normalized_value")
        if raw in (None, ""):
            continue
        key = canonical_label(str(condition_value(fld, raw) or raw) if fld in TEXT_FIELDS else str(raw))
        stage = rec.get("stage") if "stage" in rec else (C.stage_of(rec.get("evidence_text"), raw, rec.get("section"))[0] if C.role(fld) == "training" else None)
        v = out[(pid, fld)].setdefault(key, {"value": raw, "unit": rec.get("unit"), "status": F.DERIVED if rec.get("derived") else F.REPORTED,
                                             "scope": "paper", "applies_to": rec.get("applies_to"), "experiments": [], "experiment_count": 0,
                                             "evidence_ids": [eid], "key": key, "mentions": 0, "stage": stage})
        v["mentions"] += 1
    return {k: sorted(v.values(), key=lambda x: -x["mentions"]) for k, v in out.items()}


STAGE_ORDER = ("pretraining", "finetuning", "unspecified", "evaluation")


def staged_cell(views: list[dict], path: str, stage: str, fallback: list | None = None) -> dict:
    """What one paper's configurations of `stage` state for a setting, across the experiments compared (each value with
    the number of experiments whose lineage has it); `fallback`: paper-level statements whose stage is not stated."""
    values: dict[str, dict] = {}
    for v in views:
        entry = C.stage_fields(v, stage).get(path)
        for val in (entry or {}).get("values", []):
            k = canonical_label(str(condition_value(path, val["raw"]) or val["raw"]) if path in TEXT_FIELDS else str(val["raw"]))
            slot = values.setdefault(k, {"value": val["raw"], "unit": val["unit"], "status": val["status"], "stage": stage, "basis": val["basis"],
                                         "scope": "experiment" if "record" in val["origins"] else "paper", "applies_to": val.get("scope"),
                                         "experiments": [], "experiment_count": 0, "evidence_ids": val["evidence_ids"][:1], "key": k})
            if v["experiment"] not in slot["experiments"]:
                slot["experiments"].append(v["experiment"])
                slot["experiment_count"] += 1
    vals = sorted(values.values(), key=lambda x: -x["experiment_count"])
    if not vals and fallback:
        vals = [{**x, "stage": stage} for x in fallback]
    for x in vals:
        x["experiments"] = x["experiments"][:3]
    status = F.UNKNOWN if not vals else "VARIES" if len(vals) > 1 else vals[0]["status"]
    return {"status": status, "values": vals[:MAX_VALUES], "more_values": max(0, len(vals) - MAX_VALUES), "conflicting_experiments": []}


def overview(index, pid: str) -> dict:
    """What a paper's experiment records cover: benchmarks, models, tasks and metrics (exact canonical entities, counts)."""
    groups: dict[str, Counter] = {"benchmarks": Counter(), "models": Counter(), "tasks": Counter(), "metrics": Counter()}
    kind = {"USES_DATASET": "benchmarks", "USES_MODEL": "models", "CONTAINS": "tasks"}
    for x in index.paper_exps.get(pid, []):
        e = index.exp.get(x, {})
        for link in e.get("links", ()):
            if link["edge"] in kind:
                groups[kind[link["edge"]]][link["entity_id"]] += 1
        for r in e.get("results", ()):
            if r["metric_id"]:
                groups["metrics"][r["metric_id"]] += 1
    return {k: [{"id": i, "name": index.ent.get(i, {}).get("name", i), "count": n} for i, n in c.most_common(8)] for k, c in groups.items()} | {
        "_ids": {k: set(c) for k, c in groups.items()}}


def shared_results(index, pids: list[str], limit: int = 12) -> list[dict]:
    """Reported results of the papers on the benchmark + metric pairs they share, side by side (values as reported)."""
    per: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in pids:
        for x in index.paper_exps.get(p, []):
            e = index.exp.get(x, {})
            dsets = [link["entity_id"] for link in e.get("links", ()) if link["edge"] == "USES_DATASET"]
            for r in e.get("results", ()):
                for d in dsets:
                    if r["metric_id"] and r["score"]:
                        per[(d, r["metric_id"])][p].append({"model": e["label"].split(" · ")[0], "score": r["score"], "split": r["split"],
                                                            "experiment": x, "result": r["id"]})
    rows = [{"benchmark": index.ent.get(d, {}).get("name", d), "metric": index.ent.get(m, {}).get("name", m),
             "cells": [v[p][:4] for p in pids], "more": [max(0, len(v[p]) - 4) for p in pids]}
            for (d, m), v in per.items() if all(v.get(p) for p in pids)]
    return sorted(rows, key=lambda r: -sum(len(c) for c in r["cells"]))[:limit]


def row_status(cells: list[dict]) -> str:
    if any(c["status"] == F.CONFLICTING for c in cells):
        return F.CONFLICTING
    known = [c for c in cells if c["values"]]
    if not known:
        return F.UNKNOWN
    if len(known) < len(cells):
        return "PARTIAL"
    keys = [{v["key"] for v in c["values"]} for c in cells]
    return "SAME" if all(len(k) == 1 for k in keys) and len(set().union(*keys)) == 1 else "DIFFERENT"


def _benchmarks(index, xids) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for x in xids:
        for link in index.exp.get(x, {}).get("links", ()):
            if link["edge"] == "USES_DATASET":
                out[link["entity_id"]].add(x)
    return out


def fairness(index, a: str, b: str, models: dict, views: dict | None = None) -> dict:
    """Can papers a and b be fairly compared? The best-aligned experiment pair on a benchmark both report (`views`: the
    configuration views, to compare that pair's training settings stage by stage)."""
    ba, bb = _benchmarks(index, index.paper_exps.get(a, [])), _benchmarks(index, index.paper_exps.get(b, []))
    shared = sorted(set(ba) & set(bb))
    names = [index.ent[e]["name"] for e in shared if e in index.ent]
    if not shared:
        return {"papers": [a, b], "verdict": "NOT_COMPARABLE", "shared_benchmarks": [],
                "why": "The papers report no benchmark in common, so their results describe different evaluation contexts.", "derived_by": LABELER_VERSION}
    best, tried = None, 0
    for ent in shared:
        for xa in sorted(ba[ent]):
            for xb in sorted(bb[ent]):
                if tried >= MAX_PAIRS or xa not in models or xb not in models:
                    continue
                tried += 1
                r = compare(models[xa], models[xb])
                lab = r["label"].value
                if best is None or RANK[lab] < RANK[best[0]]:
                    best = (lab, xa, xb, r)
    if best is None:
        return {"papers": [a, b], "verdict": "UNKNOWN", "shared_benchmarks": names, "why": "No validated experiment record to compare.",
                "derived_by": LABELER_VERSION}
    lab, xa, xb, r = best
    why = {"COMPARABLE": "An experiment pair on a shared benchmark has an identical or equivalent evaluation setup with every equivalence field known.",
           "CONDITIONALLY_COMPARABLE": f"They share a benchmark, but equivalence cannot be established: {r['rationale']}.",
           "NOT_COMPARABLE": f"They share a benchmark, but the setups differ: {r['rationale']}."}[VERDICT[lab]]
    out = {"papers": [a, b], "verdict": VERDICT[lab], "label": lab, "shared_benchmarks": names, "experiments": [xa, xb], "why": why,
           "missing": [F.label(m) for m in r["missing"]], "pairs_compared": tried, "derived_by": LABELER_VERSION}
    if views and xa in views and xb in views:  # how the training configurations of that pair compare, stage by stage
        rows = C.compare(views[xa], views[xb])
        out["stage_differences"] = [{k: row[k] for k in ("stage_label", "label", "value_a", "value_b", "status", "rationale")}
                                    for row in rows if row["status"] in ("DIFFERENT", "MATERIALLY_DIFFERENT")][:8]
        out["stage_not_compared"] = [{k: row[k] for k in ("stage_label", "label", "rationale")} for row in rows if "different stages" in row["rationale"]][:6]
    return out


def _passages(index, pid: str) -> list[dict]:
    if pid in getattr(index, "runtime_passages", {}):  # a runtime paper: the limitation / conclusion passages of the text it was read from
        return index.runtime_passages[pid]
    rows = index.q("select id, section_title, text, unit_id, redistribution_allowed from rx.chunks where paper_id = %s and section_title ~* %s "
                   "order by id limit 3", (pid, TEXT_SECTIONS))
    arxiv = (index.q("select arxiv_id from rx.papers where id = %s", (pid,)) or [[None]])[0][0]
    return [{"chunk_id": c, "section": s, "text": t[:900] if allowed is not False else None, "text_withheld": allowed is False,
             "url": f"https://arxiv.org/html/{arxiv}#{u}" if arxiv and u else None} for c, s, t, u, allowed in rows]


def _main_dataset(index, xid: str) -> str | None:
    return next((link["raw"] for link in index.exp.get(xid, {}).get("links", []) if link["edge"] == "USES_DATASET"), None)


GAP_FIELDS = ("evaluation.benchmark", "evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition",
              "training.learning_rate", "training.batch_size", "training.seed")


def _sections(pids: list[str], per: dict, views_of: dict, level: dict) -> list[dict]:
    """The rows: evaluation and identity fields from the experiment records; training settings per stage (one row per
    stage any paper states, plus the plain row when no stage states it)."""
    sections = []
    for name, paths in SECTIONS:
        rows = []
        for path in paths:
            if C.role(path) not in ("training", "staged_evaluation"):
                cells = [cell(per[p], path, level.get((p, path))) for p in pids]
                rows.append({"field": path, "label": F.label(path), "status": row_status(cells), "cells": cells})
                continue
            home = "unspecified" if C.role(path) == "training" else "evaluation"  # where a statement of no stated stage goes
            loose = {p: [v for v in level.get((p, path), []) if v.get("stage") is None] for p in pids}
            staged = {}
            for stage in STAGE_ORDER:
                cells = [staged_cell(views_of[p], path, stage, loose[p] if stage == home else None) for p in pids]
                if any(c["values"] for c in cells):
                    staged[stage] = cells
            if not staged:
                cells = [cell(per[p], path, level.get((p, path))) for p in pids]
                rows.append({"field": path, "label": F.label(path), "status": row_status(cells), "cells": cells})
            for stage, cells in staged.items():
                rows.append({"field": path, "stage": stage, "label": f"{F.label(path)} ({C.STAGE_LABEL[stage]})", "status": row_status(cells),
                             "cells": cells})
        sections.append({"name": name, "rows": rows})
    return sections


def _with_evidence(index, sections: list[dict], configurations: list[dict]) -> int:
    """Evidence records (location + quote, licence permitting) for every shown value: one batch."""
    vals = [v for s in sections for r in s["rows"] for c in r["cells"] for v in c["values"]]
    entries = [f for inv in configurations for stage in ("pretraining", "finetuning") for c in inv[stage] for f in c["fields"]]
    ids = [i for v in vals for i in v["evidence_ids"]] + [i for f in entries for val in f["values"] for i in val["evidence_ids"][:2]]
    found = {e["id"]: e for e in index.evidence_detail(ids, limit=len(ids))} if ids else {}
    for v in vals:
        v["evidence"] = [found[i] for i in v.pop("evidence_ids") if i in found]
        v.pop("key", None)
    for f in entries:
        for val in f["values"]:
            val["evidence"] = [found[i] for i in val["evidence_ids"][:2] if i in found]
    return len(found)


def _paper_entry(index, p: str, xs: dict, inv: dict) -> tuple[dict, dict]:
    d = index.node_detail(p)
    res = [{"experiment": x, "label": index.exp[x]["label"], **{k: r[k] for k in ("metric", "score", "split")}}
           for x in index.paper_exps.get(p, []) for r in index.exp.get(x, {}).get("results", [])][:12]
    ov = overview(index, p)
    ids = ov.pop("_ids")
    runtime = p in getattr(index, "runtime", {})
    return ({"id": p, "title": d.get("title"), "year": d.get("year"), "venue": d.get("venue"), "url": d.get("url"), "authors": (d.get("authors") or [])[:6],
             "abstract": d.get("abstract"), "experiments": len(xs), "results": res, "passages": _passages(index, p),
             "contradictions": len(d.get("contradictions") or []), "covers": ov, "origin": "runtime" if runtime else "corpus",
             "provenance": d.get("provenance") if runtime else None, "configurations": inv}, ids)


def _read_full_text(index, per: dict, tr, fill) -> None:
    """One (paper, main dataset) group per corpus paper, so every paper gets its full text read (runtime papers were
    read when they were loaded)."""
    main = {p: Counter(_main_dataset(index, x) for x in xs).most_common(1)[0][0] for p, xs in per.items() if xs}
    fill({x: m for p, xs in per.items() for x, m in xs.items() if not x.startswith("rt:") and _main_dataset(index, x) == main.get(p)}, tr)


def diff(index, paper_ids: list[str], tr, fill=None) -> dict:
    """Side-by-side comparison of 2-4 papers (corpus papers, or runtime papers when `index` is a runtime Overlay). `tr`
    records the stages (Research Trace); `fill(models, tr)` reads the full text for settings still unknown
    (services/analysis/enrich.py), for each paper's main dataset first."""
    pids = list(dict.fromkeys(paper_ids))
    if not 2 <= len(pids) <= MAX_PAPERS:
        raise ValueError(f"Compare between 2 and {MAX_PAPERS} different papers.")
    with tr.stage("experiment_retrieval", papers=pids) as st:
        if missing := [p for p in pids if p not in index.paper]:
            raise KeyError(missing[0])
        models = index.experiments([x for p in pids for x in index.paper_exps.get(p, [])])
        per = {p: {x: models[x] for x in index.paper_exps.get(p, []) if x in models} for p in pids}
        runtime = sorted(set(pids) & set(getattr(index, "runtime", {})))
        st.detail.update(experiments={p: len(v) for p, v in per.items()}, runtime_papers=runtime,
                         source="validated corpus records" + (" + runtime records read at request time" if runtime else ""))
    if fill:
        _read_full_text(index, per, tr, fill)
    with tr.stage("field_facts", method=f"field facts + configuration identity ({C.VERSION})") as st:
        views = C.views(index, models)
        views_of = {p: [views[x] for x in per[p] if x in views] for p in pids}
        level = paper_level(index, pids, [p for _, paths in SECTIONS for p in paths])
        sections = _sections(pids, per, views_of, level)
        invs = {p: C.inventory(views_of[p]) for p in pids}
        n_evidence = _with_evidence(index, sections, list(invs.values()))
        papers, ids = [], {}
        for p in pids:
            papers.append((entry := _paper_entry(index, p, per[p], invs[p]))[0])
            ids[p] = entry[1]
        shared = {k: [{"id": i, "name": index.ent.get(i, {}).get("name", i)} for i in sorted(set.intersection(*(ids[p][k] for p in pids)))][:12]
                  for k in ("benchmarks", "models", "tasks", "metrics")}
        cites = [{"citing": a, "cited": b} for a in pids for b in pids if a != b and any(c == b for c, _ in index.cites_out.get(a, ()))]
        statuses = [r["status"] for s in sections for r in s["rows"]]
        st.detail.update(rows=len(statuses), **{k.lower(): statuses.count(k) for k in ("SAME", "DIFFERENT", "PARTIAL", "CONFLICTING", "UNKNOWN")},
                         evidence_records=n_evidence, staged_rows=sum(1 for s in sections for r in s["rows"] if r.get("stage")),
                         configurations={p: {"pretraining": len(invs[p]["pretraining"]), "finetuning": len(invs[p]["finetuning"])} for p in pids},
                         paper_level_cells=sum(1 for s in sections for r in s["rows"] for c in r["cells"] if c["values"] and c["values"][0].get("scope") == "paper"))
    with tr.stage("comparison", method=f"comparison engine {LABELER_VERSION}, best-aligned experiment pair per paper pair; training settings stage by stage") as st:
        pairs = [fairness(index, a, b, models, views) for a, b in combinations(pids, 2)]
        st.detail.update(pairs=[{"papers": p["papers"], "verdict": p["verdict"], "label": p.get("label")} for p in pairs])
    gaps = [{"paper": p, "fields": [F.label(f) for f in GAP_FIELDS
                                    if not any(r["cells"][i]["values"] for s in sections for r in s["rows"] if r["field"] == f)]} for i, p in enumerate(pids)]
    verdicts = {p["verdict"] for p in pairs}
    overall = "COMPARABLE" if verdicts == {"COMPARABLE"} else "NOT_COMPARABLE" if verdicts == {"NOT_COMPARABLE"} else "CONDITIONALLY_COMPARABLE"
    return {"kind": "paper_diff", "version": VERSION, "papers": papers, "sections": sections, "fairness": pairs,
            "verdict": overall, "gaps": gaps, "shared": shared, "citations": cites, "shared_results": shared_results(index, pids),
            "summary": {"papers": len(pids), "rows": len(statuses), "runtime_papers": len(runtime),
                        **{k.lower(): statuses.count(k) for k in ("SAME", "DIFFERENT", "PARTIAL", "CONFLICTING", "UNKNOWN")}},
            "inputs": {"papers": pids},
            "notes": ["Cells show what the papers' experiment records state; VARIES means different experiments of one paper state different values.",
                      "Training settings are shown per stage (pre-training, fine-tuning / adaptation, stage not stated); values of different stages are "
                      "never compared with each other.",
                      "A value marked paper-level is stated in the paper's text but not assigned to an experiment: it may not hold for every experiment.",
                      "The fairness verdict is derived by the comparison engine from the evaluation setups, not from the scores.",
                      "Corpus records are silver-tier machine extractions unless marked reviewed; runtime papers are read at request time and unreviewed. "
                      "Open the evidence to check a value."]}


CONDITION_COLUMNS = ("evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version")


def inventory_view(index, pid: str, tr, fill=None) -> dict:
    """One paper's configurations: its pre-training and fine-tuning configurations (several of each when stated), the
    evaluated experiments with their split, protocol, metric definition, dataset version and seed, each with its status
    and evidence, and what stays unknown. Answers "does this paper report several configurations?" and "which seeds,
    splits and protocols were used?" without an experiment id."""
    with tr.stage("experiment_retrieval", papers=[pid]) as st:
        if pid not in index.paper:
            raise KeyError(pid)
        models = index.experiments(index.paper_exps.get(pid, []))
        st.detail.update(experiments=len(models), runtime=pid in getattr(index, "runtime", {}))
    if fill:
        _read_full_text(index, {pid: models}, tr, fill)
    with tr.stage("field_facts", method=f"configuration identity ({C.VERSION})") as st:
        views = C.views(index, models)
        inv = C.inventory(list(views.values()))
        rows = []
        for x, v in sorted(views.items(), key=lambda kv: index.exp.get(kv[0], {}).get("label", kv[0])):
            ev_fields = C.stage_fields(v, "evaluation")
            seed = next((C.stage_fields(v, s).get("training.seed") for s in ("finetuning", "unspecified", "pretraining") if C.stage_fields(v, s).get("training.seed")), None)
            row = {"experiment": x, "label": index.exp.get(x, {}).get("label", x), "results": index.exp.get(x, {}).get("results", [])[:4],
                   "conditions": {f: ev_fields.get(f) or {"field": f, "label": F.label(f), "status": F.UNKNOWN, "values": []} for f in CONDITION_COLUMNS},
                   "seed": seed or {"field": "training.seed", "label": F.label("training.seed"), "status": F.UNKNOWN, "values": []},
                   "lineage": [{k: r[k] for k in ("relation", "to", "basis", "status") if k in r} | ({"label": r["label"]} if "label" in r else {})
                               for r in v["relations"]]}
            rows.append(row)
        entries = [f for r in rows for f in list(r["conditions"].values()) + [r["seed"]]]
        entries += [f for stage in ("pretraining", "finetuning") for c in inv[stage] for f in c["fields"]]
        ids = [i for f in entries for val in f["values"] for i in val["evidence_ids"][:2]]
        found = {e["id"]: e for e in index.evidence_detail(ids, limit=len(ids))} if ids else {}
        for f in entries:
            for val in f["values"]:
                val["evidence"] = [found[i] for i in val["evidence_ids"][:2] if i in found]
        unknown = Counter(f["label"] for r in rows for f in list(r["conditions"].values()) + [r["seed"]] if f["status"] == F.UNKNOWN)
        st.detail.update(experiments=len(rows), pretraining=len(inv["pretraining"]), finetuning=len(inv["finetuning"]), evidence_records=len(found))
    d = index.node_detail(pid)
    runtime = pid in getattr(index, "runtime", {})
    return {"kind": "paper_configurations", "version": VERSION, "paper": {"id": pid, "title": d.get("title"), "year": d.get("year"), "url": d.get("url"),
            "origin": "runtime" if runtime else "corpus", "provenance": d.get("provenance") if runtime else None},
            "configurations": inv, "experiments": rows[:60], "more_experiments": max(0, len(rows) - 60),
            "unknown": [{"label": k, "experiments": n, "of": len(rows)} for k, n in unknown.most_common()],
            "summary": {"experiments": len(rows), "pretraining": len(inv["pretraining"]), "finetuning": len(inv["finetuning"]),
                        "multiple_pretraining": inv["multiple"]["pretraining"], "multiple_finetuning": inv["multiple"]["finetuning"]},
            "inputs": {"papers": [pid]},
            "notes": ["Configurations are separated by the stage each statement names (pre-training, fine-tuning / adaptation, evaluation); a value whose "
                      "stage is not stated is listed as such.",
                      "Links between configurations are derived by the stated rule (the statement names the evaluated dataset, or the paper states one "
                      "configuration of that stage); initialisation is as the record states it.",
                      "Unknown means not stated in the sources read (records, the paper's text statements and, when read, its full text)."]}
