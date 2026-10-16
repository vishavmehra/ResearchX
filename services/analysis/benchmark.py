"""Benchmark Reality Explorer: every reported result for one benchmark and metric, with comparability.

Results are author-reported values from the corpus records and, when asked for, from runtime papers read at request time
(services/analysis/runtime_papers.py, through its Overlay; marked runtime, unreviewed). None is reproduced by ResearchX.
Each result is classified against a reference set of conditions (a chosen result's, or the most common fully stated
split + protocol): COMPARABLE (every condition the reference states is stated and equal), CONDITIONALLY_COMPARABLE
(nothing differs, but some conditions are unstated, unverified or stated less specifically), INCOMPATIBLE (a stated
condition differs) or UNKNOWN (no condition stated, or no reference can be formed). Conditions are the evaluation split,
protocol, metric definition, dataset version and preprocessing, plus the configuration conditions (services/analysis/
configs.py): the pre-training data of the linked pre-training configuration and the adaptation (zero-shot, few-shot,
linear probe, kNN, frozen backbone, fine-tuned). Values are compared after normalisation ("val" = "validation set"); a
recorded value that is not a recognisable condition is UNVERIFIED and never establishes anything. Only COMPARABLE results
are ranked. Metrics are never rescaled and no interval or significance is invented; unstated conditions stay unknown.
"""

from __future__ import annotations

import re
from collections import Counter

from services.analysis import configs as C
from services.analysis import facts as F
from services.canonicalization.normalize import canonical_label, model_family
from services.evidence import grounding as G
from services.evidence.grounding import number_of

VERSION = "benchmark-0.2.0"  # 0.2.0: normalised conditions, pre-training data + adaptation, runtime papers, natural-language selection
CONDITIONS = ("split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version", "dataset.preprocessing",
              "pretraining_data", "adaptation")
KEY_CONDITIONS = ("split", "evaluation.evaluation_protocol")  # the reference is formed from these
VOCABULARY = ("split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version")  # recognisable values only
CLASSES = ("COMPARABLE", "CONDITIONALLY_COMPARABLE", "INCOMPATIBLE", "UNKNOWN")
UNVERIFIED = "UNVERIFIED"
CONDITION_LABEL = {"split": "evaluation split", "evaluation.evaluation_protocol": "evaluation protocol",
                   "evaluation.metric_definition": "metric definition", "dataset.dataset_version": "dataset version",
                   "dataset.preprocessing": "preprocessing", "pretraining_data": "pre-training data", "adaptation": "adaptation"}


def catalog(index, limit: int = 60) -> list[dict]:
    """Benchmarks with the most reported results, each with its metrics (for the selectors). Cached on the index."""
    if getattr(index, "_benchmark_catalog", None) is None:
        counts: Counter = Counter()
        papers: dict[tuple, set] = {}
        for xid, e in index.exp.items():
            for link in e["links"]:
                if link["edge"] != "USES_DATASET":
                    continue
                for r in e["results"]:
                    if r["metric_id"] and number_of(r["score"]) is not None:
                        counts[(link["entity_id"], r["metric_id"])] += 1
                        papers.setdefault((link["entity_id"], r["metric_id"]), set()).add(e["paper_id"])
        by_bench: dict[str, list] = {}
        for (bench, metric), n in counts.items():
            if bench in index.ent and metric in index.ent:
                by_bench.setdefault(bench, []).append({"id": metric, "name": index.ent[metric]["name"], "results": n,
                                                       "papers": len(papers[(bench, metric)])})
        rows = [{"id": b, "name": index.ent[b]["name"], "metrics": sorted(ms, key=lambda m: (-m["papers"], -m["results"])),
                 "papers": max(m["papers"] for m in ms)} for b, ms in by_bench.items()]
        index._benchmark_catalog = sorted(rows, key=lambda r: (-r["papers"], r["name"].casefold()))
    return index._benchmark_catalog[:limit]


def _condition(e, r: dict, path: str, view: dict | None = None) -> dict:
    if path in ("pretraining_data", "adaptation"):  # configuration conditions (services/analysis/configs.py)
        c = C.condition(view, path) if view else {"status": F.UNKNOWN, "value": None, "evidence_ids": [], "basis": None}
        return {"status": c["status"], "value": c["value"], "evidence_ids": c["evidence_ids"][:1], "basis": c["basis"]}
    if path == "split" and r["split"]:  # the outcome's own split first, else the experiment's evaluation split
        out = {"status": F.REPORTED, "value": r["split"], "evidence_ids": r["evidence_ids"][:1]}
    else:
        f = F.fact("evaluation.dataset_split" if path == "split" else path, F.field_value(e, "evaluation.dataset_split" if path == "split" else path))
        out = {"status": f["status"], "value": f["raw"] if f["raw"] not in (None, "") else f["value"], "evidence_ids": f["evidence_ids"][:1]}
        if f["derivation"]:
            out["basis"] = f["derivation"]
    if path in VOCABULARY and out["status"] in F.STATED and G.condition_key(path, out["value"]) is None:
        out.update(status=UNVERIFIED, note=f"the record's value is not a recognisable {CONDITION_LABEL[path]}")
    return out


def _key(path: str, c: dict) -> frozenset | None:
    """The comparable content of a stated condition (G.condition_key for split, protocol, metric definition, version)."""
    if c["status"] not in F.STATED or c["value"] in (None, ""):
        return None
    return G.condition_key(path, c["value"]) if path in VOCABULARY else frozenset([canonical_label(str(c["value"]))])


def _shown_key(k: frozenset | None) -> str | None:
    return " + ".join(sorted(k)) if k else None


def classify(conds: dict, ref: dict | None) -> tuple[str, str]:
    """COMPARABLE needs the split and the protocol stated, and equal, on both sides (plus every other condition the
    reference states); a condition either side leaves unstated, unverified or less specific can only make a result
    conditionally comparable; a stated condition that differs makes it incompatible."""
    if ref is None:
        return "UNKNOWN", "No result states its evaluation split, so no reference conditions can be formed."
    stated = {p: _key(p, c) for p, c in conds.items()}

    def rel(p):
        k, r = stated.get(p), ref.get(p)
        return None if not k or not r else "same" if k == r else "narrower" if (k < r or r < k) else "differs"
    differ = [p for p in CONDITIONS if rel(p) == "differs"]
    if differ:
        return "INCOMPATIBLE", "Differs from the reference in " + ", ".join(f"{CONDITION_LABEL[p]} ({conds[p]['value']})" for p in differ) + "."
    checked = [p for p in CONDITIONS if rel(p) == "same"]
    open_ = [p for p in CONDITIONS if (p in KEY_CONDITIONS or ref.get(p)) and rel(p) != "same"]
    if all(p in checked for p in KEY_CONDITIONS) and not [p for p in open_ if ref.get(p)]:
        return "COMPARABLE", "States the same " + ", ".join(CONDITION_LABEL[p] for p in checked) + " as the reference."
    if checked:
        why = [CONDITION_LABEL[p] + (" (stated less specifically)" if rel(p) == "narrower" else " (unverified value)"
                                     if conds.get(p, {}).get("status") == UNVERIFIED else "") for p in open_]
        return ("CONDITIONALLY_COMPARABLE", "Nothing stated differs from the reference (same " + ", ".join(CONDITION_LABEL[p] for p in checked)
                + "), but " + ", ".join(why) + " cannot be checked.")
    return "UNKNOWN", "States none of the reference conditions (" + ", ".join(CONDITION_LABEL[p] for p in CONDITIONS if ref.get(p)) + ")."


def reference_conditions(rows: list[dict]) -> tuple[dict | None, dict]:
    """The most common stated split + protocol shared by at least two results; else the most common split; a condition
    set only one result states is used last (then nothing else can be COMPARABLE to it)."""
    keys = [{c: _key(c, r["conditions"][c]) for c in KEY_CONDITIONS} for r in rows]
    full = Counter(tuple(k[c] for c in KEY_CONDITIONS) for k in keys if all(k[c] for c in KEY_CONDITIONS))
    split = Counter((k["split"], None) for k in keys if k["split"])
    for counter, need in ((full, 2), (split, 2), (full, 1), (split, 1)):
        if counter and counter.most_common(1)[0][1] >= need:
            top, support = counter.most_common(1)[0]
            ref = dict(zip(KEY_CONDITIONS, top, strict=True))
            return ref, {"kind": "most common stated conditions", "support": support,
                         "states": [CONDITION_LABEL[c] for c in KEY_CONDITIONS if ref[c]]}
    return None, {"kind": "none"}


def resolve(index, text: str) -> tuple[str, str]:
    """(benchmark id, metric id) named in a question ("Which ImageNet top-1 results are comparable ...?"): the longest
    catalog benchmark name written in the text, then the metric of that benchmark the text names (else its most reported
    one). KeyError when the text names no benchmark of the catalog."""
    t = G.canon(text)
    named = [b for b in catalog(index, limit=10_000) if (n := G.canon(b["name"])) and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", t)]
    if not named:
        raise KeyError("no benchmark named")
    bench = max(named, key=lambda b: (len(b["name"]), b["papers"]))
    metrics = [m for m in bench["metrics"] if G.tokens(m["name"]) and not G.tokens(m["name"]) <= G.tokens(bench["name"])]  # never the benchmark's own name
    said = [x.group(1) for x in G.METRIC.finditer(t)]  # "top-1" for "Top 1 Accuracy": one names a subset of the other's words
    metric = next((m for m in metrics if any(G.metric_ok(s, m["name"]) for s in said)), None) or next(
        (m for m in sorted(metrics, key=lambda m: -len(m["name"])) if re.search(rf"(?<![a-z0-9]){re.escape(G.canon(m['name']))}(?![a-z0-9])", t)), None)
    return bench["id"], (metric or (metrics or bench["metrics"])[0])["id"]


def explore(index, benchmark: str, metric: str, tr, reference: str | None = None, family: str | None = None,
            year_from: int | None = None, year_to: int | None = None) -> dict:
    with tr.stage("experiment_retrieval", benchmark=benchmark, metric=metric) as st:
        if benchmark not in index.ent or metric not in index.ent:
            raise KeyError(benchmark if benchmark not in index.ent else metric)
        found = [(x, r) for x in index.ent_exps.get(benchmark, []) for r in index.exp[x]["results"] if r["metric_id"] == metric]
        models = index.experiments({x for x, _ in found})
        st.detail.update(results=len(found), experiments=len(models), not_validated=len({x for x, _ in found} - set(models)))
    with tr.stage("field_facts", method=f"evaluation conditions + configuration conditions ({C.VERSION})") as st:
        views = C.views(index, models)
        rows = []
        for x, r in found:
            e = models.get(x)
            if e is None:
                continue
            pid = index.exp[x]["paper_id"]
            p = index.paper.get(pid, {})
            name = next((link["raw"] for link in index.exp[x]["links"] if link["edge"] == "USES_MODEL"), None) or index.exp[x]["label"]
            fam = model_family(name)[0] or canonical_label(name)
            runtime = index.exp[x].get("origin") == "runtime"
            row = {"result": r["id"], "experiment": x, "paper": pid, "paper_title": p.get("title"), "year": p.get("year"), "model": name,
                   "family": fam, "score": r["score"], "value": number_of(r["score"]), "conditions": {c: _condition(e, r, c, views.get(x)) for c in CONDITIONS},
                   "evidence_ids": r["evidence_ids"][:2], "quality_tier": e.quality_tier.value,
                   "origin": "runtime (read at request time, unreviewed)" if runtime else "corpus (author-reported)"}
            if runtime:
                rp = getattr(index, "runtime", {}).get(pid, {})
                row["runtime"] = {"provider": rp.get("provider"), "access": rp.get("access"), "url": rp.get("url"),
                                  "metric_as_written": r.get("metric_as_written") or r.get("metric"),
                                  "dataset_as_written": next((link.get("aligned_from") or link["raw"] for link in index.exp[x]["links"]
                                                              if link["edge"] == "USES_DATASET"), None)}
            rows.append(row)
        options = {"families": [f for f, _ in Counter(r["family"] for r in rows).most_common(40)],
                   "years": sorted({r["year"] for r in rows if r["year"]})}
        rows = [r for r in rows if (not family or r["family"] == family) and (year_from is None or (r["year"] or 0) >= year_from)
                and (year_to is None or (r["year"] or 9999) <= year_to)]
        st.detail.update(rows=len(rows), numeric=sum(r["value"] is not None for r in rows), runtime_rows=sum("runtime" in r for r in rows),
                         unverified=sum(c["status"] == UNVERIFIED for r in rows for c in r["conditions"].values()),
                         **{f"{c.split('.')[-1]}_stated": sum(_key(c, r["conditions"][c]) is not None for r in rows) for c in CONDITIONS})
    with tr.stage("comparability", method="stated conditions vs a reference after normalisation; UNKNOWN and UNVERIFIED never match") as st:
        ref_row = next((r for r in rows if r["result"] == reference), None) if reference else None
        if reference and ref_row is None:
            raise KeyError(reference)
        if ref_row is not None:
            ref = {c: _key(c, ref_row["conditions"][c]) for c in CONDITIONS}
            ref_source = {"kind": "chosen result", "result": reference, "states": [CONDITION_LABEL[c] for c in CONDITIONS if ref[c]]}
        else:
            ref, ref_source = reference_conditions(rows)
        for r in rows:
            r["class"], r["why"] = classify(r["conditions"], ref)
        ranked = sorted((r for r in rows if r["class"] == "COMPARABLE" and r["value"] is not None), key=lambda r: -r["value"])
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        st.detail.update(reference=ref_source, **{c.lower(): sum(r["class"] == c for r in rows) for c in CLASSES}, ranked=len(ranked))
    ids = [i for r in rows for i in r["evidence_ids"]]
    ev = {e["id"]: e for e in index.evidence_detail(ids, limit=len(ids))} if ids else {}
    for r in rows:
        r["evidence"] = [ev[i] for i in r.pop("evidence_ids") if i in ev]
        for c in r["conditions"].values():
            c.pop("evidence_ids", None)
    rows.sort(key=lambda r: (CLASSES.index(r["class"]), -(r["value"] if r["value"] is not None else float("-inf")), r["model"].casefold()))
    return {"kind": "benchmark", "version": VERSION, "benchmark": {"id": benchmark, "name": index.ent[benchmark]["name"]},
            "metric": {"id": metric, "name": index.ent[metric]["name"]},
            "reference": {"conditions": {CONDITION_LABEL[c]: _shown_key(v) for c, v in (ref or {}).items() if v}, **ref_source},
            "rows": rows, "options": options, "filters": {"family": family, "year_from": year_from, "year_to": year_to},
            "summary": {"results": len(rows), **{c.lower(): sum(r["class"] == c for r in rows) for c in CLASSES}, "ranked": len(ranked),
                        "runtime": sum("runtime" in r for r in rows)},
            "inputs": {"benchmark": benchmark, "metric": metric, "reference": reference},
            "notes": ["All values are author-reported in the papers; ResearchX has not reproduced any of them.",
                      "Only COMPARABLE results are ranked, by reported value; the metric's direction is not asserted.",
                      "Conditions include the pre-training data and the adaptation (fine-tuned, linear probe, zero-shot ...) of each result's "
                      "configuration; an unstated or unverified condition is never treated as equal.",
                      "Runtime results come from papers read at request time; they are unreviewed and not part of the corpus.",
                      "No confidence interval or significance is shown unless a paper reports one; metrics are never rescaled."]}
