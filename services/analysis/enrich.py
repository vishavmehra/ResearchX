"""Request-time full-text reading: before an analysis shows an experiment setting as not stated, the paper's full text is
read for it (recipe, paper diff, why results differ, claim checks).

Order of sources: the experiment record, then the paper's extracted text statements (GraphIndex._attach_paper_level),
then this: the passages and tables of the paper that name the setting (the stored full text, else the arXiv HTML read now,
services.runtime.reader) go to the schema-bound extractor (`extract`, the API passes orchestrator._extract on the
fulltext_extraction route plus its runtime checks). A setting that no passage even names is not asked for (that prompt
only invites a guess). A value is kept only when the deterministic checks pass: the quoted span is in the passage and
supports the value (G.validate), it is not a sweep of values, and neither the passage nor its section heading states it
for another of the paper's datasets; a passage that names no dataset is kept as a general setting of the paper and says
so. Evaluation conditions (split, protocol, metric definition, dataset version) are kept only when the passage names the
experiment's dataset, or the paper evaluates on one dataset. Every value takes the stage its passage names
(services/analysis/configs.stage_of): one value per (setting, stage) is attached as DERIVED with the quote; a pre-training
value, or several values for one stage, stays a paper-level statement that the configuration view shows (never resolved
into one). The quote is served by GraphIndex.evidence_detail like corpus evidence (ids "rt:..."), marked runtime. Nothing is
written to the corpus. Papers are read in parallel, and training settings and evaluation conditions in two parallel calls;
results, including "searched, nothing found", are cached in the process, so a repeated view costs no model call.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from services.analysis.configs import stage_of
from services.datasets.schemas import FieldStatus, FieldValue
from services.evidence import grounding as G
from services.runtime import reader

VERSION = "enrich-0.2.0"  # 0.2.0: seeds, splits, protocols, metric definitions, versions; values kept per (setting, stage)
FILLABLE = tuple(f for f in G.FIELDS if f != "evaluation.score")  # the settings the extractor and validator know
# evaluation conditions (the first three) and dataset versions: kept only when the passage names the experiment's dataset
CONDITIONS = ("evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version")
BATCHES = (tuple(f for f in FILLABLE if f not in CONDITIONS + ("training.seed", "model.checkpoint", "dataset.split")),
           CONDITIONS + ("training.seed", "model.checkpoint", "dataset.split"))
UNITS, PER_FIELD, CHARS, MAX_GROUPS, CACHE = 8, 2, 1500, 4, 4096  # passages per call, best passages per setting
_tried: OrderedDict[tuple, float] = OrderedDict()  # (paper, dataset, field) searched without a usable value
_lock = threading.Lock()


def missing(m) -> list[str]:
    """Fillable settings the experiment model leaves unknown."""
    out = []
    for f in FILLABLE:
        group, name = f.split(".", 1)
        fv = getattr(getattr(m, group, None), name, None)
        if fv is not None and fv.status is FieldStatus.UNKNOWN:
            out.append(f)
    return out


def _units(index, pid: str) -> tuple[list[dict], dict]:
    """The paper's passages and tables: stored full text, else its arXiv HTML read now. (units, source description)."""
    rows = index.q("select c.id, c.unit_type, c.unit_id, c.section_title, c.text, c.redistribution_allowed, p.arxiv_id from rx.chunks c "
                   "join rx.papers p on p.id = c.paper_id where c.paper_id = %s and c.unit_type in ('paragraph', 'table', 'caption')", (pid,))
    if rows:
        arxiv = rows[0][6]
        return ([{"chunk_id": cid, "unit_type": ut, "unit_id": uid, "section": sec, "text": text, "redistribution_allowed": allowed is not False,
                  "url": f"https://arxiv.org/html/{arxiv}#{uid}" if arxiv and uid else None} for cid, ut, uid, sec, text, allowed, _ in rows],
                {"source": "stored full text", "units": len(rows)})
    arxiv = (index.q("select arxiv_id from rx.papers where id = %s", (pid,)) or [[None]])[0][0]
    if not arxiv:
        return [], {"source": "none: no full text stored and no arXiv id"}
    doc = reader.read(arxiv, timeout=6)
    return ([{**u, "chunk_id": f"rt:{arxiv}#{u['unit_id']}", "redistribution_allowed": doc["redistribution_allowed"],
              "url": f"https://arxiv.org/html/{arxiv}#{u['unit_id']}"} for u in doc["units"]],
            {"source": "arXiv full text read now", "url": doc["url"], "license": doc["license"], "units": len(doc["units"])})


def _read_group(index, pid: str, dataset: str | None, g: dict, extract) -> dict:
    """One (paper, dataset) group: read the full text, extract the asked settings, keep the validated ones. Reads only
    (thread-safe); the caller attaches the values."""
    t, fields = time.perf_counter(), sorted(g["fields"])
    try:
        units, src = _units(index, pid)
    except Exception as e:  # an unreadable full text costs only this paper
        return {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:120]}", "fields": fields, "t": t}
    req = G.Request(fields=fields, model=g["model"], dataset=dataset)
    title = index.paper.get(pid, {}).get("title") or ""
    calls, asked = [], []  # a setting no passage of the paper even names is never asked for: that prompt only invites a guess
    for batch in BATCHES:  # training settings and evaluation conditions in separate calls: each output stays within its bound
        picked: dict[str, dict] = {}
        want = []
        for f in (f for f in fields if f in batch):
            best = reader.relevant(units, [f], req, PER_FIELD, CHARS)
            if best:
                want.append(f)
                for u in best:
                    picked.setdefault(u["chunk_id"], u)
        if want:
            calls.append((want, [{**u, "context": "\n".join(x for x in (title, u.get("section"), u["text"]) if x)} for u in list(picked.values())[:UNITS]]))
            asked += want
    # no dataset in the request: papers state most settings once for all experiments ("Training setting. We ..."), and a
    # model told "dataset = X" skips those; which dataset a value belongs to is decided deterministically below
    with ThreadPoolExecutor(max(1, len(calls))) as pool:
        facts = [f for got in pool.map(lambda c: extract(G.Request(fields=c[0], model=g["model"], dataset=None), c[0], c[1]), calls) for f in got]
    sources = list({s["chunk_id"]: s for _, ss in calls for s in ss}.values())
    by_id = {s["chunk_id"]: s for s in sources}
    known = sorted({link["raw"] for x in index.paper_exps.get(pid, []) for link in index.exp[x]["links"] if link["edge"] == "USES_DATASET"})
    kept: dict[tuple[str, str], list] = {}  # (field, stage) -> validated values; several distinct values are kept, never resolved
    for f in facts:
        s = by_id.get(f.chunk_id)
        if s is None:
            continue
        ok, _reason, value = G.validate(f.field, f.value if f.value is not None else f.raw, f.span, s["text"])
        if not ok or (f.field not in G.VERBATIM and G.value_set(f.span, value)):
            continue
        named = G.dataset_context(s["text"], f.span, dataset, known) if dataset else None
        heading_other = any(d != dataset and d.casefold() in (s.get("section") or "").casefold() for d in known)
        if named is False or (named is None and heading_other):  # another dataset's value (in the passage, or its section heading)
            continue
        if f.field in CONDITIONS and not (named or len(known) <= 1):  # an evaluation condition must name this dataset (or be the paper's only one)
            continue
        stage, basis = (("evaluation", "an evaluation condition") if f.field in CONDITIONS[:3] else stage_of(s["text"], f.raw, s.get("section")))
        if f.field == "dataset.dataset_version" and stage is None:
            stage, basis = "evaluation", "a dataset version stated beside this dataset"
        rows = kept.setdefault((f.field, stage or "unspecified"), [])
        if not any(G.canon(str(v)) == G.canon(str(value)) for _, _, v, _, _ in rows):
            rows.append((f, s, value, "for this dataset" if named else "a general setting: the passage names no dataset", basis))
    return {"status": "completed", "kept": kept, "fields": fields, "asked": asked, "sources": sources, "facts": facts, "src": src, "t": t}


def fill(index, models: dict, tr, extract) -> dict:
    """Read the full text for the settings `models` (experiment id -> Experiment) leave unknown and attach validated values
    in place. `extract(req, fields, sources) -> facts` is the shared extractor; None skips the reading."""
    if extract is None or not models:
        return {"status": "not run"}
    groups: dict[tuple[str, str | None], dict] = {}
    for xid, m in models.items():
        need = missing(m)
        if not need or not m.paper_id:
            continue
        links = index.exp.get(xid, {}).get("links", [])
        dsets = [link["raw"] for link in links if link["edge"] == "USES_DATASET"]
        key = (m.paper_id, dsets[0] if dsets else None)
        with _lock:
            need = [f for f in need if (key[0], key[1], f) not in _tried]
        if need:
            g = groups.setdefault(key, {"fields": set(), "experiments": [], "model": next((link["raw"] for link in links if link["edge"] == "USES_MODEL"), None)})
            g["fields"].update(need)
            g["experiments"].append(xid)
    if not groups:
        return {"status": "nothing to read: every setting known or already searched"}
    summary = {"groups": len(groups), "papers": [], "facts": 0, "attached": 0, "searched_fields": 0}
    with tr.stage("runtime_enrichment", method="full text read for settings still unknown; schema-bound extraction + deterministic validation") as st:
        todo = list(groups.items())[:MAX_GROUPS]
        with ThreadPoolExecutor(len(todo)) as pool:  # papers are read and extracted in parallel: the slowest one sets the latency
            futures = {key: pool.submit(_read_group, index, key[0], key[1], g, extract) for key, g in todo}
        for (pid, dataset), g in todo:
            r = futures[(pid, dataset)].result()
            row = {"paper": pid, "dataset": dataset, "fields": [G.FIELDS[f][0] for f in r["fields"]]}
            if r["status"] == "failed":
                summary["papers"].append({**row, "status": "failed", "error": r["error"]})
                continue
            for (field, stage), found in r["kept"].items():
                # one value per (setting, stage) is attached; a pre-training value or several values for one stage stay paper-level
                # statements (services/analysis/configs.py shows them in the paper's configurations), never resolved into one
                attach = stage != "pretraining" and len(found) == 1
                for f, s, value, scope, basis in found:
                    rid = "rt:" + hashlib.sha1(f"{pid}|{s['chunk_id']}|{field}|{value}".encode()).hexdigest()[:16]
                    index.runtime_evidence[rid] = {"id": rid, "field": field, "status": "RUNTIME_EXTRACTED", "location_kind": "PAPER_" + s["unit_type"].upper(),
                                                   "paper_id": pid, "experiment_id": None, "text": f.span if s["redistribution_allowed"] else None,
                                                   "text_withheld": not s["redistribution_allowed"], "extracted_value": f.raw, "normalized_value": value,
                                                   "source_location": s.get("url"), "section": s.get("section"),
                                                   "table": s["unit_id"] if s["unit_type"] == "table" else None, "cell": None,
                                                   "extraction_model": "runtime extraction at request time", "stage": stage, "stage_basis": basis,
                                                   "applies_to": dataset if scope == "for this dataset" else None}
                    if not attach:
                        continue
                    group, name = field.split(".", 1)
                    for xid in g["experiments"]:
                        fv = getattr(getattr(models[xid], group, None), name, None)
                        if fv is not None and fv.status is FieldStatus.UNKNOWN:
                            try:
                                setattr(getattr(models[xid], group), name, FieldValue(
                                    status=FieldStatus.KNOWN, raw_value=G.display(f.raw), value=value, evidence_ids=[rid], derived=True,
                                    derivation=f"found in the paper's full text at request time ({s.get('section') or s['unit_type']}; {scope}; "
                                               f"stage: {stage}); validated quote"))
                                summary["attached"] += 1
                            except (ValueError, TypeError):
                                continue
            kept_fields = {k for k, _ in r["kept"]}
            with _lock:  # searched: a setting without a usable value is not asked for again in this process
                for field in set(r["fields"]) - kept_fields:
                    _tried[(pid, dataset, field)] = time.time()
                while len(_tried) > CACHE:
                    _tried.popitem(last=False)
            summary["papers"].append({**row, "status": "completed", **r["src"], "passages": len(r["sources"]), "asked": [G.FIELDS[f][0] for f in r["asked"]],
                                      "not_named_in_text": [G.FIELDS[f][0] for f in r["fields"] if f not in r["asked"]], "facts": len(r["facts"]),
                                      "kept": sorted(f"{G.FIELDS[k][0]} ({st})" for k, st in r["kept"]),
                                      "latency_ms": round((time.perf_counter() - r["t"]) * 1000)})
            summary["facts"] += sum(len(v) for v in r["kept"].values())
            summary["searched_fields"] += len(r["fields"])
        st.detail.update(**summary)
    return {**summary, "status": "completed"}
