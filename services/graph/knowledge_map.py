"""Knowledge Map: the research graph, derived on demand from the loaded corpus tables (docs/ARCHITECTURE.md §64).

Nothing is stored and nothing is inferred from similarity. Nodes and edges are computed from rx.papers, rx.experiments,
rx.evidence, rx.citations, rx.study_links, rx.contradictions and rx.canonical_entities, with the identities the dataset
pipeline uses (model / method / dataset / metric / topic nodes are canonical entities, services.canonicalization.entities).
Every edge names its derivation (the record field or table row it comes from) and the evidence ids behind it.
Experiment-to-experiment edges come only from the deterministic comparison engine (services.comparison.fields.compare:
UNKNOWN is never a match). Study links that rest on title similarity alone are not drawn. Public corpus only: private
documents are never read here, so the graph cannot leak them.

Edge strength: EXACT = recorded verbatim in a source record (a field value, a citation, a table row); STRONG = aligned
evaluation setup per the comparison engine; RELATED = related context or a material difference; POSSIBLE = unreviewed
hypothesis or insufficient evidence; CONTRADICTORY = sources report different values for the same fact.
"""

from __future__ import annotations

import re
import threading
from collections import Counter, OrderedDict, defaultdict
from typing import Callable

from pydantic import ValidationError

from services.analysis.configs import stage_of
from services.canonicalization.entities import entity_id
from services.canonicalization.normalize import canonical_label, model_family
from services.comparison.fields import LABELER_VERSION, compare
from services.datasets.schemas import CanonicalEntityType as E
from services.datasets.schemas import Experiment, FieldStatus, FieldValue
from services.evidence.grounding import canon, condition_value, tokens
from services.storage.records import expand

VERSION = "knowledge-map-0.1.0"
EXPERIMENTS_FILE = "data/silver/experiments/experiments.jsonl"

# experiment field -> (entity type, node type, edge type); topic edges point Topic -> Experiment
ENTITY_FIELDS = {
    "model.model_name": (E.MODEL, "Model", "USES_MODEL"),
    "model.base_model": (E.MODEL, "Model", "USES_MODEL"),
    "method.method_name": (E.METHOD, "Method", "USES_METHOD"),
    "evaluation.benchmark": (E.BENCHMARK, "Dataset", "USES_DATASET"),
    "dataset.dataset_name": (E.DATASET, "Dataset", "USES_DATASET"),
    "research.task": (E.TASK, "Topic", "CONTAINS"),
}
ENTITY_NODE = {E.MODEL.value: "Model", E.METHOD.value: "Method", E.BENCHMARK.value: "Dataset", E.DATASET.value: "Dataset",
               E.TASK.value: "Topic", E.METRIC.value: "Metric"}
CONFIG_GROUPS = ("model", "method", "training", "retrieval", "prompting", "inference", "evaluation", "compute", "dataset")
NODE_TYPES = ("Topic", "Paper", "Study", "Experiment", "Configuration", "Model", "Method", "Dataset", "Metric", "Result", "Evidence")
EDGE_TYPES = ("REPORTS", "CONTAINS", "USES_MODEL", "USES_METHOD", "USES_DATASET", "HAS_CONFIGURATION", "PRODUCES_RESULT", "MEASURES",
              "SUPPORTED_BY", "CITES", "RELATED_TO", "REPLICATES", "EXTENDS", "DIFFERS_FROM", "CONTRADICTS")
# comparison-engine label -> (edge type, strength). SAME_EXPERIMENT is never produced automatically (human review only).
COMPARISON_EDGE = {"ALIGNED_EXPERIMENT": ("RELATED_TO", "STRONG"), "MATERIALLY_DIFFERENT": ("DIFFERS_FROM", "RELATED"),
                   "RELATED_CONTEXT": ("RELATED_TO", "RELATED"), "INSUFFICIENT_EVIDENCE": ("RELATED_TO", "POSSIBLE"),
                   "CONFLICTING_EVIDENCE": ("RELATED_TO", "CONTRADICTORY")}
LABEL_RANK = {"ALIGNED_EXPERIMENT": 0, "MATERIALLY_DIFFERENT": 1, "INSUFFICIENT_EVIDENCE": 2, "RELATED_CONTEXT": 3, "CONFLICTING_EVIDENCE": 4}
# study-link label -> (edge type, strength); every link in the release is an unreviewed candidate
STUDY_LINK_EDGE = {"REPLICATION": ("REPLICATES", "RELATED"), "EXTENSION": ("EXTENDS", "RELATED"), "FOLLOW_UP": ("RELATED_TO", "RELATED"),
                   "SAME_UNDERLYING_STUDY": ("RELATED_TO", "RELATED"), "RELATED_BUT_DISTINCT": ("RELATED_TO", "RELATED"),
                   "UNCERTAIN": ("RELATED_TO", "POSSIBLE")}
SIMILARITY_EVIDENCE = {"title_similarity"}
TYPE_ORDER = {t: i for i, t in enumerate(("Paper", "Study", "Topic", "Dataset", "Model", "Method", "Configuration", "Result", "Metric",
                                          "Experiment", "Evidence"))}
# paging order of a node's neighbours: recorded conflicts first, then the node's own records, then wider context
EDGE_RANK = {"CONTRADICTS": 0, "REPORTS": 1, "USES_DATASET": 2, "USES_MODEL": 2, "USES_METHOD": 2, "CONTAINS": 3, "HAS_CONFIGURATION": 4,
             "PRODUCES_RESULT": 5, "MEASURES": 6, "RELATED_TO": 7, "DIFFERS_FROM": 7, "REPLICATES": 7, "EXTENDS": 7, "CITES": 8, "SUPPORTED_BY": 9}
RELATED_K, RELATED_POOL, CACHE_SIZE = 6, 30, 2048
# settings a record may take from its paper's own text at read time (GraphIndex._attach_paper_level); never evaluation conditions
ATTACHABLE = ("training.learning_rate", "training.batch_size", "training.optimizer", "training.epochs", "training.steps", "training.weight_decay",
              "training.warmup_steps", "training.learning_rate_schedule", "training.dropout", "training.seed", "training.precision",
              "compute.gpu_model", "compute.gpu_count", "compute.training_time")
# evaluation conditions a record may take from its paper's text only when the statement names the record's dataset
LINKABLE = ("evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version")
DEFAULT = re.compile(r"\b(?:default|all experiments|every experiment|unless (?:otherwise )?(?:stated|specified))\b", re.I)


def _names_dataset(text: str | None, dataset: str | None) -> bool:
    """`text` writes the dataset's name as a whole name ("ADE20K" in "single-scale mIoU on ADE20K val")."""
    d = canon(dataset)
    return bool(d and text) and re.search(rf"(?<![a-z0-9]){re.escape(d)}(?![a-z0-9])", canon(text)) is not None


# ------------------------------------------------------------------------------------------- pure derivations
def _v(x) -> str:
    return getattr(x, "value", str(x))


def _known(fv) -> bool:
    return isinstance(fv, dict) and fv.get("status") == "KNOWN" and fv.get("value") not in (None, "")


def _raw(fv: dict) -> str:
    """Surface form exactly as services.canonicalization.entities.surface_forms reads it."""
    return str(fv.get("raw_value") if fv.get("raw_value") not in (None, "") else fv.get("value")).strip()


def _short(s, n: int = 120) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _sentence(s: str) -> str:
    s = (s or "").strip()
    return (s[:1].upper() + s[1:]).rstrip(".") + "." if s else ""


def entity_links(record: dict) -> list[dict]:
    """Entity-bearing KNOWN fields of an experiment record, each with the canonical entity id the dataset pipeline assigns
    to that surface form and the evidence ids behind the value."""
    out = []
    for path, (etype, ntype, edge) in ENTITY_FIELDS.items():
        group, name = path.split(".", 1)
        fv = (record.get(group) or {}).get(name)
        if _known(fv):
            raw = _raw(fv)
            eid = entity_id(etype, raw)
            if eid:
                out.append({"entity_id": eid, "entity_type": etype.value, "node_type": ntype, "edge": edge, "field": path, "raw": raw,
                            "evidence_ids": list(fv.get("evidence_ids") or [])})
    return out


def results_of(exp_id: str, record: dict) -> list[dict]:
    """One result per reported outcome (metric and/or score KNOWN); id res:<experiment id>#<outcome index>."""
    out = []
    for i, o in enumerate(record.get("outcomes") or []):
        metric, score, split = o.get("metric") or {}, o.get("score") or {}, o.get("dataset_split") or {}
        if not (_known(metric) or _known(score)):
            continue
        m = _raw(metric) if _known(metric) else None
        out.append({"id": f"res:{exp_id}#{i}", "index": i, "metric": m, "metric_id": entity_id(E.METRIC, m) if m else None,
                    "score": _raw(score) if _known(score) else None, "split": _raw(split) if _known(split) else None,
                    "evidence_ids": list(dict.fromkeys((score.get("evidence_ids") or []) + (metric.get("evidence_ids") or [])))})
    return out


def configuration_of(record: dict) -> list[dict]:
    """KNOWN configuration fields of an experiment, except the ones drawn as their own nodes."""
    out = []
    for group in CONFIG_GROUPS:
        for name, fv in (record.get(group) or {}).items():
            path = f"{group}.{name}"
            if path not in ENTITY_FIELDS and _known(fv):
                out.append({"field": path, "value": _raw(fv), "unit": fv.get("unit"), "evidence_ids": list(fv.get("evidence_ids") or [])})
    return out


def make_edge(etype: str, source: str, target: str, why: str, derivation: str, strength: str = "EXACT", evidence_ids=(),
              label: str | None = None, provenance: str | None = None) -> dict:
    return {"id": f"{etype}~{source}~{target}", "type": etype, "source": source, "target": target, "strength": strength,
            "label": label or etype.replace("_", " ").lower(), "why": why, "derivation": derivation,
            "evidence_ids": list(dict.fromkeys(evidence_ids))[:20], "provenance": provenance}


def comparison_edge(a: str, b: str, res: dict) -> dict:
    """Edge for one comparison-engine result; the label, rationale and differing / unknown fields say why."""
    label = _v(res["label"])
    etype, strength = COMPARISON_EDGE.get(label, ("RELATED_TO", "POSSIBLE"))
    st = {fc.field_name: _v(fc.comparison_status) for fc in res["field_comparisons"]}
    why = _sentence(res["rationale"])
    conflicting = [f for f, s in st.items() if s == "CONFLICTING"]
    differ = [f for f, s in st.items() if s in ("MATERIALLY_DIFFERENT", "DIFFERENT")]
    unknown = sum(s == "UNKNOWN" for s in st.values())
    if conflicting:
        why += f" Conflicting evidence on {', '.join(conflicting)}."
    if differ:
        why += f" Differs in {', '.join(differ[:6])}."
    if unknown:
        why += f" {unknown} compared field(s) unknown on at least one side (unknown is never treated as a match)."
    if res.get("uncertainty"):
        why += " " + _sentence(res["uncertainty"])
    return make_edge(etype, a, b, why, f"comparison engine {LABELER_VERSION}: {label}", strength, res.get("evidence") or [],
                     label.replace("_", " ").lower())


def study_link_edge(link: dict) -> dict | None:
    """None when the link rests on title similarity alone: similarity is not a relationship."""
    kinds = {e.get("kind") for e in link["evidence"] if isinstance(e, dict)}
    if not kinds - SIMILARITY_EVIDENCE:
        return None
    etype, strength = STUDY_LINK_EDGE.get(link["label"], ("RELATED_TO", "POSSIBLE"))
    signals = "; ".join(e["detail"] for e in link["evidence"] if isinstance(e, dict) and e.get("detail"))
    why = f"{_sentence(link['rationale'] or link['label'])} Signals: {signals}. Candidate link from the dataset, not reviewed."
    return make_edge(etype, link["a"], link["b"], why, f"rx.study_links {link['id']} ({link['label']})", strength,
                     label=link["label"].replace("_", " ").lower())


def _same_fields(res: dict) -> int:
    return sum(_v(fc.comparison_status) in ("IDENTICAL", "EQUIVALENT") for fc in res["field_comparisons"])


def _lead(label: str) -> str:
    """The leading name of a label ('Model-X (few-shot, k=5) · Bench' -> 'model-x'), used only to vary a page."""
    return re.split(r" \(| · |, ", label or "", maxsplit=1)[0].casefold()


def _interleave(ordered: list[str], key: Callable[[str], tuple]) -> list[str]:
    """Round-robin by key[1] inside each block of equal key[0], keeping block order: one page of a node's neighbours shows
    the variety of a block (different models, papers) instead of 40 variants of one name. Deterministic."""
    out: list[str] = []
    i = 0
    while i < len(ordered):
        block = key(ordered[i])[0]
        j = i
        while j < len(ordered) and key(ordered[j])[0] == block:
            j += 1
        lanes: dict[str, list[str]] = {}
        for n in ordered[i:j]:
            lanes.setdefault(key(n)[1], []).append(n)
        queues = list(lanes.values())
        while queues:
            out += [q.pop(0) for q in queues]
            queues = [q for q in queues if q]
        i = j
    return out


# ------------------------------------------------------------------------------------------- index
class GraphIndex:
    """In-process index of the public corpus graph, built once per process from the rx tables (read-only).
    `q(sql, params)` runs one statement on the corpus connection and returns its rows."""

    def __init__(self, q: Callable[..., list]):
        self.q = q
        self.paper = {pid: {"title": t, "year": y, "study_id": s} for pid, t, y, s in
                      q("select id, title, publication_year, study_id from rx.papers")}
        groups: dict[str, list] = defaultdict(list)
        for pid, p in self.paper.items():
            if p["study_id"]:
                groups[p["study_id"]].append(pid)
        self.study_first = {sid: sorted(pids)[0] for sid, pids in groups.items()}
        self.study_papers = {sid: sorted(pids) for sid, pids in groups.items() if len(pids) > 1}  # a one-paper study is that paper
        self.ent = {i: {"type": t, "name": n} for i, t, n in
                    q("select id, entity_type, preferred_name from rx.canonical_entities where entity_type = any(%s)", (list(ENTITY_NODE),))}
        self.exp: dict[str, dict] = {}
        self.ent_exps: dict[str, list] = defaultdict(list)
        self.metric_res: dict[str, list] = defaultdict(list)
        self.paper_exps: dict[str, list] = defaultdict(list)
        rows = q("select id, paper_id, depth_tier, jsonb_build_object("
                 "'model', jsonb_build_object('model_name', record->'model'->'model_name', 'base_model', record->'model'->'base_model'), "
                 "'method', jsonb_build_object('method_name', record->'method'->'method_name'), "
                 "'evaluation', jsonb_build_object('benchmark', record->'evaluation'->'benchmark'), "
                 "'dataset', jsonb_build_object('dataset_name', record->'dataset'->'dataset_name'), "
                 "'research', jsonb_build_object('task', record->'research'->'task'), "
                 "'outcomes', coalesce((select jsonb_agg(jsonb_build_object('metric', o->'metric', 'score', o->'score', "
                 "'dataset_split', o->'dataset_split') order by i) from jsonb_array_elements(case when jsonb_typeof(record->'outcomes') = 'array' "
                 "then record->'outcomes' else '[]'::jsonb end) with ordinality t(o, i)), '[]'::jsonb)) from rx.experiments")
        for xid, pid, depth, rec in rows:
            links, res = entity_links(rec), results_of(xid, rec)
            first = {}
            for link in links:
                first.setdefault(link["node_type"], link["raw"])
            # model · dataset · task: one paper often reports a model on many tasks of one benchmark, and the label must tell them apart
            name = " · ".join(dict.fromkeys(x for x in (first.get("Model") or first.get("Method"), first.get("Dataset"), first.get("Topic")) if x))
            self.exp[xid] = {"paper_id": pid, "depth": depth, "links": links, "results": res, "label": _short(name or f"Experiment {xid[-6:]}", 100)}
            self.paper_exps[pid].append(xid)
            for eid in dict.fromkeys(link["entity_id"] for link in links):
                self.ent_exps[eid].append(xid)
            for r in res:
                if r["metric_id"]:
                    self.metric_res[r["metric_id"]].append(r["id"])
        self.cites_out: dict[str, list] = defaultdict(list)
        self.cites_in: dict[str, list] = defaultdict(list)
        for a, b, src in q("select citing_paper_id, cited_paper_id, record->>'source_record_id' from rx.citations"):
            if a in self.paper and b in self.paper and a != b:
                self.cites_out[a].append((b, src))
                self.cites_in[b].append((a, src))
        self.links: dict[str, list] = defaultdict(list)
        self.links_total = self.links_similarity_only = 0
        for lid, ra, rb, rel, rec in q("select id, record_a, record_b, relationship, record from rx.study_links"):
            self.links_total += 1
            a, b = self._paper_of(ra), self._paper_of(rb)
            link = {"id": lid, "a": a, "b": b, "label": rel, "rationale": rec.get("rationale"), "evidence": rec.get("evidence") or []}
            edge = study_link_edge(link) if a and b and a != b else None
            if edge is None:
                self.links_similarity_only += 1
                continue
            link["edge"] = edge
            self.links[a].append(link)
            self.links[b].append(link)
        self.contra: dict[str, list] = defaultdict(list)
        for cid, subject, fld, rec in q("select id, subject_id, field, record from rx.contradictions"):
            self.contra[subject].append({"id": cid, "subject": subject, "field": fld, "claim": rec.get("claim"), "explanation": rec.get("explanation"),
                                         "value_a": rec.get("value_a"), "value_b": rec.get("value_b"), "source_a": rec.get("source_a"),
                                         "source_b": rec.get("source_b"), "evidence_a": rec.get("evidence_a") or [],
                                         "evidence_b": rec.get("evidence_b") or [], "conflict_type": rec.get("conflict_type")})
        self._consts: dict | None = None
        self._cache: OrderedDict[str, Experiment] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.runtime_evidence: dict[str, dict] = {}  # "rt:..." quotes found in full texts at request time (services/analysis/enrich.py)

    def _paper_of(self, record_id: str) -> str | None:
        return record_id if record_id in self.paper else self.study_first.get(record_id)

    # --------------------------------------------------------------------------------------- nodes
    def stats(self) -> dict:
        used = Counter(ENTITY_NODE[e["type"]] for i, e in self.ent.items() if i in self.ent_exps or i in self.metric_res)
        return {"papers": len(self.paper), "papers_with_experiments": len(self.paper_exps), "studies_grouping_papers": len(self.study_papers),
                "experiments": len(self.exp), "results": sum(len(e["results"]) for e in self.exp.values()), "entities": dict(used),
                "field_edges": sum(len(e["links"]) for e in self.exp.values()), "citations": sum(len(v) for v in self.cites_out.values()),
                "study_links": self.links_total - self.links_similarity_only, "study_links_not_drawn_similarity_only": self.links_similarity_only,
                "contradictions": sum(len(v) for v in self.contra.values()), "version": VERSION}

    def _result(self, rid: str) -> tuple[str, dict] | None:
        xid, _, idx = rid.removeprefix("res:").rpartition("#")
        r = next((r for r in self.exp.get(xid, {}).get("results", ()) if str(r["index"]) == idx), None)
        return (xid, r) if r else None

    def node(self, nid: str) -> dict | None:
        """Summary (type, label, small meta, flags) of any node except evidence (see _evidence_nodes); None if unknown."""
        kind = nid.split(":", 1)[0]
        if kind == "paper" and nid in self.paper:
            p = self.paper[nid]
            return {"id": nid, "type": "Paper", "label": _short(p["title"] or nid), "flags": ["conflict"] if self.contra.get(nid) else [],
                    "meta": {"year": p["year"], "experiments": len(self.paper_exps.get(nid, ()))}}
        if kind == "study" and nid in self.study_papers:
            first = self.paper[self.study_papers[nid][0]]
            return {"id": nid, "type": "Study", "label": _short(first["title"] or nid), "flags": [], "meta": {"papers": len(self.study_papers[nid])}}
        if kind == "exp" and nid in self.exp:
            e = self.exp[nid]
            return {"id": nid, "type": "Experiment", "label": e["label"], "flags": ["conflict"] if self.contra.get(nid) else [],
                    "meta": {"paper": _short(self.paper.get(e["paper_id"], {}).get("title"), 90), "depth": e["depth"], "results": len(e["results"])}}
        if kind == "ent" and nid in self.ent:
            ent = self.ent[nid]
            ntype = ENTITY_NODE[ent["type"]]
            n = len(self.metric_res.get(nid, ())) if ntype == "Metric" else len(self.ent_exps.get(nid, ()))
            return {"id": nid, "type": ntype, "label": _short(ent["name"]), "flags": [],
                    "meta": {"entity_type": ent["type"].lower(), "results" if ntype == "Metric" else "experiments": n}}
        if kind == "res" and (found := self._result(nid)):
            xid, r = found
            return {"id": nid, "type": "Result", "label": _short(f"{r['metric'] or 'metric unknown'} = {r['score'] or 'score unknown'}"),
                    "flags": ["conflict"] if any(c["field"] == "evaluation.score" for c in self.contra.get(xid, ())) else [],
                    "meta": {"experiment": self.exp[xid]["label"], "split": r["split"]}}
        if kind == "cfg" and (xid := nid.removeprefix("cfg:")) in self.exp:
            return {"id": nid, "type": "Configuration", "label": "Configuration", "flags": [], "meta": {"experiment": self.exp[xid]["label"]}}
        return None

    def _evidence_nodes(self, ids) -> dict[str, dict]:
        ids = [i for i in dict.fromkeys(ids) if i]
        if not ids:
            return {}
        rows = self.q("select id, field_name, location_kind, record->>'source_location', record->>'evidence_text', record->'redistribution_allowed' "
                      "from rx.evidence where id = any(%s)", (ids,))
        return {i: {"id": i, "type": "Evidence", "label": _short(f"{fld}: {txt}" if allowed is not False else f"{fld} ({loc})", 90), "flags": [],
                    "meta": {"field": fld, "location": (kind or "record").lower().replace("paper_", ""), "source": _short(loc, 160)}}
                for i, fld, kind, loc, txt, allowed in rows}

    def any_node(self, nid: str) -> dict | None:
        return self._evidence_nodes([nid]).get(nid) if nid.startswith("ev:") else self.node(nid)

    # --------------------------------------------------------------------------------------- full records
    def _constants(self) -> dict:
        if self._consts is None:
            rows = self.q("select constants from rx.load_checks where source_file = %s", (EXPERIMENTS_FILE,))
            self._consts = (rows[0][0] if rows else None) or {}
        return self._consts

    def _records(self, ids) -> dict[str, dict]:
        return dict(self.q("select id, record from rx.experiments where id = any(%s)", (list(ids),)))

    def configuration(self, xid: str) -> list[dict]:
        """Known configuration fields of a corpus experiment, from the cached validated model when there is one."""
        m = self.experiments([xid]).get(xid)
        return configuration_of(m.model_dump(mode="json") if m is not None else self._records([xid]).get(xid) or {})

    def experiments(self, ids) -> dict[str, Experiment]:
        """Full Experiment models rebuilt exactly as the loader verified them (records.expand + load constants)."""
        ids = list(dict.fromkeys(ids))
        with self._cache_lock:
            out = {i: self._cache[i] for i in ids if i in self._cache}
        missing = [i for i in ids if i not in out]
        if missing:
            rows = self.q("select id, paper_id, study_id, experiment_type, split, record, provenance_id from rx.experiments where id = any(%s)", (missing,))
            pids = sorted({r[-1] for r in rows if r[-1]})
            provs = dict(self.q("select id, provenance from rx.provenance where id = any(%s)", (pids,))) if pids else {}
            fresh = {}
            for xid, pid, sid, etype, split, rec, prov_id in rows:
                typed = {"experiment_id": xid, "paper_id": pid, "study_id": sid, "experiment_type": etype, "split": split}
                try:
                    fresh[xid] = Experiment.model_validate(expand(rec, typed, self._constants(), prov_id, provs))
                except ValidationError:  # a record that no longer validates is left out of comparisons, never guessed
                    continue
            self._attach_paper_level(fresh)
            out.update(fresh)
            with self._cache_lock:
                for xid in missing:
                    if xid in out:
                        self._cache[xid] = out[xid]
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        return out

    def _attach_paper_level(self, models: dict[str, Experiment]) -> None:
        """Settings a record leaves unknown but its paper's text states (rx.evidence, PAPER_ locations, no experiment id),
        attached as DERIVED values with the rule that applied, never as reported ones: the value the paper states for
        this experiment's dataset or model; else the only value it states for the setting; else its stated default.
        Ambiguous settings stay unknown. A value stated for pre-training is never attached: it belongs to the paper's
        pre-training configuration (services/analysis/configs.py), not to the evaluated experiment. Evaluation conditions
        (split, protocol, metric definition, dataset version) decide comparability, so they are linked only when the
        statement names this experiment's dataset and states one recognisable value for it (never "the only value" or a
        default). Read time only: records are not changed."""
        if not models:
            return
        pids = sorted({m.paper_id for m in models.values() if m.paper_id})
        rows = self.q("select id, paper_id, field_name, record from rx.evidence where paper_id = any(%s) and experiment_id is null "
                      "and location_kind like 'PAPER_%%' and field_name = any(%s) order by id", (pids, list(ATTACHABLE + LINKABLE))) if pids else []
        stated: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)  # (paper, field) -> canonical value -> value
        conditions: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for eid, pid, fld, rec in rows:
            raw = rec.get("extracted_value") if rec.get("extracted_value") not in (None, "") else rec.get("normalized_value")
            if raw in (None, ""):
                continue
            if fld in LINKABLE:
                if (canon_v := condition_value(fld, raw)) is not None:  # an unrecognisable value is never linked
                    conditions[(pid, fld)].append({"raw": str(raw), "value": canon_v, "applies_to": rec.get("applies_to"),
                                                   "text": rec.get("evidence_text"), "evidence_id": eid})
                continue
            if stage_of(rec.get("evidence_text"), raw, rec.get("section"))[0] == "pretraining":
                continue
            v = stated[(pid, fld)].setdefault(canonical_label(str(raw)), {"raw": str(raw), "value": rec.get("normalized_value", raw), "unit": rec.get("unit"),
                                                                          "applies_to": rec.get("applies_to"), "evidence_ids": []})
            v["evidence_ids"].append(eid)
        for xid, m in models.items():
            datasets = [link["raw"] for link in self.exp.get(xid, {}).get("links", []) if link["edge"] == "USES_DATASET"]
            for fld in LINKABLE:
                group, name = fld.split(".", 1)
                fv = getattr(getattr(m, group, None), name, None)
                aimed = [(c, d) for c in conditions.get((m.paper_id, fld), []) for d in datasets[:1]
                         if _names_dataset(c["applies_to"], d) or _names_dataset(c["text"], d)]
                if fv is None or fv.status is not FieldStatus.UNKNOWN or len({canonical_label(c["value"]) for c, _ in aimed}) != 1:
                    continue
                c, d = aimed[0]
                try:
                    setattr(getattr(m, group), name, FieldValue(status=FieldStatus.KNOWN, raw_value=c["raw"], value=c["value"],
                                                                evidence_ids=list(dict.fromkeys(x["evidence_id"] for x, _ in aimed))[:3], derived=True,
                                                                derivation=f"stated in the paper's text for {d}; linked to this experiment by its dataset "
                                                                           "name, not tied to the experiment record"))
                except (ValidationError, ValueError, TypeError):
                    continue
        for xid, m in models.items():
            names = [link["raw"] for link in self.exp.get(xid, {}).get("links", []) if link["edge"] in ("USES_DATASET", "USES_MODEL")]
            for fld in ATTACHABLE:
                group, name = fld.split(".", 1)
                fv = getattr(getattr(m, group, None), name, None)
                values = list(stated.get((m.paper_id, fld), {}).values())
                if fv is None or fv.status is not FieldStatus.UNKNOWN or not values:
                    continue
                aimed = [v for v in values if v["applies_to"] and any(len(tokens(n)) and tokens(n) <= tokens(v["applies_to"]) for n in names)]
                defaults = [v for v in values if v["applies_to"] and DEFAULT.search(v["applies_to"])]
                pick, why = ((aimed[0], f"stated in the paper's text for {aimed[0]['applies_to']}") if len(aimed) == 1 else
                             (values[0], "the only value the paper's text states for this setting") if len(values) == 1 and not aimed else
                             (defaults[0], f"the paper's stated default ({defaults[0]['applies_to']})") if len(defaults) == 1 and not aimed else (None, None))
                if pick is None:
                    continue
                try:
                    setattr(getattr(m, group), name, FieldValue(status=FieldStatus.KNOWN, raw_value=pick["raw"], value=pick["value"], unit=pick["unit"],
                                                                evidence_ids=pick["evidence_ids"][:3], derived=True,
                                                                derivation=why + "; not tied to this experiment record"))
                except (ValidationError, ValueError, TypeError):  # a value the schema rejects stays unknown
                    continue

    def related(self, xid: str, k: int = RELATED_K) -> list[tuple[str, dict]]:
        """Experiments that share a dataset/benchmark entity with `xid`, compared field by field with the comparison engine.
        Candidates are ranked by the number of shared entities (exact canonical ids, no similarity scores), the pairs by the
        engine's label and then by the number of identical fields."""
        e = self.exp.get(xid)
        if not e:
            return []
        mine = {link["entity_id"] for link in e["links"]}
        shared: dict[str, int] = {}
        for d in {link["entity_id"] for link in e["links"] if link["edge"] == "USES_DATASET"}:
            for x in self.ent_exps.get(d, ()):
                if x != xid and x not in shared:
                    shared[x] = len(mine & {link["entity_id"] for link in self.exp[x]["links"]})
        pool = sorted(shared, key=lambda x: (-shared[x], x))[:RELATED_POOL]
        if not pool:
            return []
        models = self.experiments([xid, *pool])
        if xid not in models:
            return []
        pairs = [(x, compare(models[xid], models[x])) for x in pool if x in models]
        pairs.sort(key=lambda t: (LABEL_RANK.get(_v(t[1]["label"]), 9), -_same_fields(t[1]), t[0]))
        return pairs[:k]

    # --------------------------------------------------------------------------------------- adjacency
    def _link_edge(self, xid: str, link: dict) -> dict:
        why = f"Field {link['field']} = “{link['raw']}” (KNOWN, {len(link['evidence_ids'])} evidence span(s))."
        derivation = f"rx.experiments.record.{link['field']}"
        if link["edge"] == "CONTAINS":
            return make_edge("CONTAINS", link["entity_id"], xid, why, derivation, evidence_ids=link["evidence_ids"], label="topic of")
        return make_edge(link["edge"], xid, link["entity_id"], why, derivation, evidence_ids=link["evidence_ids"],
                         label="evaluated on" if link["field"] == "evaluation.benchmark" else None)

    def _contradiction_pairs(self, subject: str) -> list[tuple[str, str, dict]]:
        return [(c["evidence_a"][0], c["evidence_b"][0], c) for c in self.contra.get(subject, ()) if c["evidence_a"] and c["evidence_b"]]

    @staticmethod
    def _contradicts(a: str, b: str, c: dict) -> dict:
        why = f"{_sentence(c['explanation'] or c['claim'])} Values: {c['value_a']!r} ({c['source_a']}) vs {c['value_b']!r} ({c['source_b']})."
        return make_edge("CONTRADICTS", a, b, why, f"rx.contradictions {c['id']} ({c['field']})", "CONTRADICTORY", [a, b], "contradicts")

    def _contradiction_support(self, subject: str) -> list[tuple[str, dict]]:
        out = []
        for a, b, c in self._contradiction_pairs(subject):
            for i, side in ((a, c["source_a"]), (b, c["source_b"])):
                out.append((i, make_edge("SUPPORTED_BY", subject, i, f"Evidence for {c['field']} from {side} (one side of a recorded contradiction).",
                                         f"rx.contradictions {c['id']}", evidence_ids=[i], label="evidence")))
        return out

    def _adjacent(self, nid: str, want_evidence: bool = False, with_related: bool = True) -> list[tuple[str, dict]]:
        """(neighbour id, edge) for every edge of `nid` the corpus records."""
        kind = nid.split(":", 1)[0]
        out: list[tuple[str, dict]] = []
        if kind == "paper" and nid in self.paper:
            out += [(x, make_edge("REPORTS", nid, x, "The experiment record was extracted from this paper.", "rx.experiments.paper_id"))
                    for x in self.paper_exps.get(nid, ())]
            out += [(b, make_edge("CITES", nid, b, f"Citation metadata (source record {src}).", "rx.citations", provenance=src))
                    for b, src in self.cites_out.get(nid, ())]
            out += [(a, make_edge("CITES", a, nid, f"Citation metadata (source record {src}).", "rx.citations", provenance=src))
                    for a, src in self.cites_in.get(nid, ())]
            out += [(link["b"] if link["a"] == nid else link["a"], link["edge"]) for link in self.links.get(nid, ())]
            sid = self.paper[nid]["study_id"]
            if sid in self.study_papers:
                out.append((sid, make_edge("CONTAINS", sid, nid, "The paper record belongs to this study.", "rx.papers.study_id")))
            out += self._contradiction_support(nid)
        elif kind == "study" and nid in self.study_papers:
            out += [(p, make_edge("CONTAINS", nid, p, "The paper record belongs to this study.", "rx.papers.study_id")) for p in self.study_papers[nid]]
        elif kind == "exp" and nid in self.exp:
            e = self.exp[nid]
            out.append((e["paper_id"], make_edge("REPORTS", e["paper_id"], nid, "The experiment record was extracted from this paper.",
                                                 "rx.experiments.paper_id")))
            out += [(link["entity_id"], self._link_edge(nid, link)) for link in e["links"]]
            out += [(r["id"], make_edge("PRODUCES_RESULT", nid, r["id"], f"Outcome {r['index']}: {r['metric'] or 'metric unknown'} = {r['score'] or 'score unknown'}.",
                                        f"rx.experiments.record.outcomes[{r['index']}]", evidence_ids=r["evidence_ids"])) for r in e["results"]]
            cfg = self.configuration(nid)
            if cfg:
                out.append((f"cfg:{nid}", make_edge("HAS_CONFIGURATION", nid, f"cfg:{nid}", f"{len(cfg)} known configuration field(s), each with evidence.",
                                                    "rx.experiments.record (configuration groups)", evidence_ids=[i for c in cfg for i in c["evidence_ids"]])))
            out += self._contradiction_support(nid)
            if with_related:
                for x, res in self.related(nid):
                    a, b = sorted((nid, x))
                    out.append((x, comparison_edge(a, b, res if a == nid else compare(*self._pair(a, b)))))
            if want_evidence:
                ids = [i for link in e["links"] for i in link["evidence_ids"]]
                out += [(i, make_edge("SUPPORTED_BY", nid, i, "Evidence for one of this experiment's fields.", "rx.experiments.record evidence_ids",
                                      evidence_ids=[i], label="evidence")) for i in dict.fromkeys(ids)]
        elif kind == "ent" and nid in self.ent:
            if ENTITY_NODE[self.ent[nid]["type"]] == "Metric":
                out += [(rid, make_edge("MEASURES", rid, nid, "The outcome reports this metric.", "rx.experiments.record.outcomes[].metric"))
                        for rid in self.metric_res.get(nid, ())]
            else:
                for x in self.ent_exps.get(nid, ()):
                    link = next(link for link in self.exp[x]["links"] if link["entity_id"] == nid)
                    out.append((x, self._link_edge(x, link)))
        elif kind == "res" and (found := self._result(nid)):
            xid, r = found
            out.append((xid, make_edge("PRODUCES_RESULT", xid, nid, f"Outcome {r['index']} of this experiment.", f"rx.experiments.record.outcomes[{r['index']}]",
                                       evidence_ids=r["evidence_ids"])))
            if r["metric_id"]:
                out.append((r["metric_id"], make_edge("MEASURES", nid, r["metric_id"], f"The outcome's metric field is “{r['metric']}”.",
                                                      "rx.experiments.record.outcomes[].metric")))
            out += [(i, make_edge("SUPPORTED_BY", nid, i, "Evidence for this outcome's score or metric.", "outcome evidence_ids", evidence_ids=[i], label="evidence"))
                    for i in r["evidence_ids"]]
        elif kind == "cfg" and (xid := nid.removeprefix("cfg:")) in self.exp:
            out.append((xid, make_edge("HAS_CONFIGURATION", xid, nid, "Configuration of this experiment.", "rx.experiments.record")))
            for c in self.configuration(xid):
                out += [(i, make_edge("SUPPORTED_BY", nid, i, f"Evidence for {c['field']} = “{c['value']}”.", f"rx.experiments.record.{c['field']}",
                                      evidence_ids=[i], label=c["field"].split(".", 1)[1].replace("_", " "))) for i in c["evidence_ids"]]
        elif kind == "ev":
            rows = self.q("select experiment_id, paper_id, field_name from rx.evidence where id = %s", (nid,))
            if rows:
                xid, pid, fld = rows[0]
                owner = xid if xid in self.exp else pid if pid in self.paper else None
                if owner:
                    out.append((owner, make_edge("SUPPORTED_BY", owner, nid, f"Evidence for {fld}.", "rx.evidence.experiment_id / paper_id",
                                                 evidence_ids=[nid], label="evidence")))
                for cs in self.contra.values():
                    for c in cs:
                        if c["evidence_a"] and c["evidence_b"] and nid in (c["evidence_a"][0], c["evidence_b"][0]):
                            a, b = c["evidence_a"][0], c["evidence_b"][0]
                            out.append((b if a == nid else a, self._contradicts(a, b, c)))
        return out

    def _pair(self, a: str, b: str) -> tuple[Experiment, Experiment]:
        m = self.experiments([a, b])
        if a not in m or b not in m:
            raise KeyError(a if a not in m else b)
        return m[a], m[b]

    def _contradiction_context(self, shown: set[str]) -> list[dict]:
        """CONTRADICTS edges between evidence nodes that are both on screen."""
        out = []
        for subject in [n for n in shown if n in self.contra]:
            out += [self._contradicts(a, b, c) for a, b, c in self._contradiction_pairs(subject) if a in shown and b in shown]
        return out

    def neighborhood(self, nid: str, node_types=None, edge_types=None, limit: int = 40, offset: int = 0) -> dict:
        """One expansion step: the neighbours of `nid` (filtered server-side, paged by neighbour) and the edges to them."""
        center = self.any_node(nid)
        if center is None:
            raise KeyError(nid)
        node_types, edge_types = set(node_types or ()), set(edge_types or ())
        with_related = (not edge_types or bool(edge_types & {"RELATED_TO", "DIFFERS_FROM"})) and (not node_types or "Experiment" in node_types)
        pairs = [(n, e) for n, e in self._adjacent(nid, "Evidence" in node_types, with_related) if not edge_types or e["type"] in edge_types]
        info = {n: self.node(n) for n, _ in pairs if not n.startswith("ev:")}
        info |= self._evidence_nodes([n for n, _ in pairs if n.startswith("ev:")])
        by_node: dict[str, dict[str, dict]] = defaultdict(dict)
        for n, e in pairs:
            if info.get(n) and (not node_types or info[n]["type"] in node_types):
                by_node[n][e["id"]] = e
        def rank(n: str) -> tuple:
            es = by_node[n].values()
            conflict = any(e["type"] == "CONTRADICTS" or e["derivation"].startswith("rx.contradictions") for e in es)
            return (0 if conflict else min(EDGE_RANK.get(e["type"], 9) for e in es) + 1, TYPE_ORDER.get(info[n]["type"], 99),
                    info[n]["label"].casefold(), n)

        order = _interleave(sorted(by_node, key=rank), lambda n: (rank(n)[:2], _lead(info[n]["label"])))
        page = order[offset: offset + limit]
        edges = {eid: e for n in page for eid, e in by_node[n].items()}
        edges |= {e["id"]: e for e in self._contradiction_context({nid, *page})}
        return {"center": center, "nodes": [center] + [info[n] for n in page], "edges": list(edges.values()), "total": len(order),
                "offset": offset, "limit": limit, "has_more": offset + limit < len(order),
                "counts": dict(Counter(info[n]["type"] for n in order)),
                "edge_counts": dict(Counter(e["type"] for n in order for e in by_node[n].values())), "version": VERSION}

    # --------------------------------------------------------------------------------------- search & landscape
    def search(self, text: str, types=None, limit: int = 20) -> list[dict]:
        t = " ".join((text or "").casefold().split())
        if len(t) < 2:
            return []
        types = set(types or ())
        hits: list[tuple[int, int, str]] = []
        for i, e in self.ent.items():
            ntype = ENTITY_NODE[e["type"]]
            used = len(self.metric_res.get(i, ())) if ntype == "Metric" else len(self.ent_exps.get(i, ()))
            name = (e["name"] or "").casefold()
            if used and (not types or ntype in types) and t in name:
                hits.append((0 if name == t else 1 if name.startswith(t) else 2, -used, i))
        if not types or "Paper" in types:
            for pid, p in self.paper.items():
                title = (p["title"] or "").casefold()
                if t in title or t == pid.casefold() or t == pid.casefold().split(":", 2)[-1]:
                    hits.append((0 if t == title or t in pid.casefold() else 1 if title.startswith(t) else 2, -len(self.paper_exps.get(pid, ())), pid))
        if (not types or "Experiment" in types) and t in self.exp:
            hits.append((0, 0, t))
        elif types and "Experiment" in types:  # by label ("OCR (HRNetV2-W48) · ADE20K"): pickers search experiments by name
            for xid, e in self.exp.items():
                label = e["label"].casefold()
                if t in label:
                    hits.append((1 if label.startswith(t) else 2, -len(e["results"]), xid))
        hits.sort()
        return [n for _, _, i in hits[:limit] if (n := self.node(i))]

    def experiments_of(self, nid: str) -> list[str]:
        kind = nid.split(":", 1)[0]
        if kind == "paper":
            return list(self.paper_exps.get(nid, ()))
        if kind == "study":
            return [x for p in self.study_papers.get(nid, ()) for x in self.paper_exps.get(p, ())]
        if kind == "ent" and nid in self.ent:
            if ENTITY_NODE[self.ent[nid]["type"]] == "Metric":
                return list(dict.fromkeys(r.removeprefix("res:").rpartition("#")[0] for r in self.metric_res.get(nid, ())))
            return list(self.ent_exps.get(nid, ()))
        if kind == "exp" and nid in self.exp:
            return [nid]
        return []

    def _cooccurring(self, exps, exclude: str | None, limit: int = 8) -> dict[str, list[dict]]:
        """Entities linked to the same experiment records, by node type, with the number of experiments for each."""
        groups: dict[str, Counter] = defaultdict(Counter)
        for x in exps:
            e = self.exp[x]
            for link in {link["entity_id"]: link for link in e["links"]}.values():
                if link["entity_id"] != exclude:
                    groups[link["node_type"]][link["entity_id"]] += 1
            for mid in {r["metric_id"] for r in e["results"] if r["metric_id"]} - {exclude}:
                groups["Metric"][mid] += 1
        return {k: [{**n, "count": c} for i, c in groups[k].most_common(limit) if (n := self.node(i))]
                for k in ("Topic", "Dataset", "Model", "Method", "Metric") if groups[k]}

    def landscape(self, root: str | None = None, limit: int = 10) -> dict:
        """A research area at a glance: the experiments of `root` (an entity, a paper, or the whole corpus) grouped by the
        topics, datasets, models, methods and metrics they are linked to. Every number is a count of experiment records;
        model families are derived from the name (a trailing size such as 7B), never merged otherwise."""
        if root and self.any_node(root) is None:
            raise KeyError(root)
        exps = self.experiments_of(root) if root else list(self.exp)
        papers = {self.exp[x]["paper_id"] for x in exps}
        families: Counter = Counter()
        for x in exps:
            for raw in {link["raw"] for link in self.exp[x]["links"] if link["node_type"] == "Model"}:
                fam, size = model_family(raw)
                if size and fam:
                    families[fam] += 1
        years = Counter(self.paper[p]["year"] for p in papers if self.paper.get(p, {}).get("year"))
        return {"root": self.any_node(root) if root else None, "experiments": len(exps), "papers": len(papers),
                "results": sum(len(self.exp[x]["results"]) for x in exps), "years": dict(sorted(years.items())),
                "groups": self._cooccurring(exps, root, limit), "model_families": [{"family": f, "count": n} for f, n in families.most_common(limit) if n > 1],
                "contradictions": sum(len(self.contra.get(s, ())) for s in {*exps, *papers}), "version": VERSION}

    # --------------------------------------------------------------------------------------- details
    def evidence_detail(self, ids, limit: int = 20) -> list[dict]:
        ids = [i for i in dict.fromkeys(ids) if i][: min(limit, 400)]
        if not ids:
            return []
        out = {i: {k: v for k, v in self.runtime_evidence[i].items() if not k.startswith("_")}  # request-time quotes, marked RUNTIME_EXTRACTED
               for i in ids if i in self.runtime_evidence}
        stored = [i for i in ids if i not in out]
        rows = self.q("select id, paper_id, experiment_id, field_name, status, location_kind, record from rx.evidence where id = any(%s)",
                      (stored,)) if stored else []
        for i, pid, xid, fld, status, kind, r in rows:
            allowed = r.get("redistribution_allowed") is not False
            out[i] = {"id": i, "field": fld, "status": status, "location_kind": kind, "paper_id": pid, "experiment_id": xid,
                      "text": r.get("evidence_text") if allowed else None, "text_withheld": not allowed,
                      "extracted_value": r.get("extracted_value"), "normalized_value": r.get("normalized_value"),
                      "source_location": r.get("source_location"), "section": r.get("section"), "table": r.get("table"), "cell": r.get("cell"),
                      "char_start": r.get("char_start"), "char_end": r.get("char_end"), "source_record_id": r.get("source_record_id"),
                      "source_version": r.get("source_version"), "retrieval_timestamp": r.get("retrieval_timestamp"),
                      "extraction_model": r.get("extraction_model")}
        return [out[i] for i in ids if i in out]

    def node_detail(self, nid: str) -> dict:
        base = self.any_node(nid)
        if base is None:
            raise KeyError(nid)
        kind = nid.split(":", 1)[0]
        d: dict = {"node": base, "questions": [{"key": k, "question": self.ASK[k].format(type=base["type"].lower())} for k in self.questions_for(base["type"])]}
        if kind == "paper":
            title, year, arxiv, doi, authors, abstract, venue = self.q(
                "select title, publication_year, arxiv_id, doi, record->'authors', record->>'abstract', record->>'venue' from rx.papers where id = %s", (nid,))[0]
            names = [a.get("name") if isinstance(a, dict) else str(a) for a in (authors or [])]
            exps = self.paper_exps.get(nid, [])
            d |= {"title": title, "year": year, "arxiv_id": arxiv, "doi": doi, "venue": venue, "authors": names[:12], "authors_total": len(names),
                  "abstract": _short(abstract, 1400), "url": f"https://arxiv.org/abs/{arxiv}" if arxiv else f"https://doi.org/{doi}" if doi else None,
                  "experiments": len(exps), "cites": len(self.cites_out.get(nid, ())), "cited_by": len(self.cites_in.get(nid, ())),
                  "study_links": [{"other": self.node(link["b"] if link["a"] == nid else link["a"]), "label": link["label"], "why": link["edge"]["why"],
                                   "strength": link["edge"]["strength"]} for link in self.links.get(nid, ())],
                  "contradictions": self.contra.get(nid, []), "related": self._cooccurring(exps, None)}
        elif kind == "study":
            d |= {"papers": [self.node(p) for p in self.study_papers[nid]]}
        elif kind == "exp":
            pid, sid, etype, depth, split, rec = self.q(
                "select paper_id, study_id, experiment_type, depth_tier, split, record from rx.experiments where id = %s", (nid,))[0]
            e = self.exp[nid]
            d |= {"paper": self.node(pid), "experiment_type": etype, "depth": depth, "split": split, "description": rec.get("description"),
                  "fields": [{"field": link["field"], "value": link["raw"], "node": link["entity_id"], "evidence": len(link["evidence_ids"])} for link in e["links"]],
                  "configuration": configuration_of(rec), "results": e["results"], "quality": rec.get("quality"),
                  "extraction": {"method": rec.get("extraction_method"), "model": rec.get("extraction_model")},
                  "source_record_ids": rec.get("source_record_ids") or [], "unresolved_fields": len(rec.get("unresolved_fields") or []),
                  "contradictions": self.contra.get(nid, [])}
        elif kind == "ent":
            exps = self.experiments_of(nid)
            forms = (self.q("select record->'surface_forms' from rx.canonical_entities where id = %s", (nid,)) or [[None]])[0][0] or []
            years = sorted(y for p in {self.exp[x]["paper_id"] for x in exps} if (y := self.paper.get(p, {}).get("year")))
            d |= {"name": self.ent[nid]["name"], "entity_type": self.ent[nid]["type"], "surface_forms": forms[:10], "experiments": len(exps),
                  "papers": len({self.exp[x]["paper_id"] for x in exps}), "results": len(self.metric_res.get(nid, ())),
                  "years": [years[0], years[-1]] if years else None, "related": self._cooccurring(exps, nid)}
        elif kind == "res":
            xid, r = self._result(nid)
            d |= {"experiment": self.node(xid), "paper": self.node(self.exp[xid]["paper_id"]), "metric": r["metric"], "metric_node": r["metric_id"],
                  "score": r["score"], "split": r["split"], "evidence": self.evidence_detail(r["evidence_ids"]),
                  "contradictions": [c for c in self.contra.get(xid, ()) if c["field"] == "evaluation.score"]}
        elif kind == "cfg":
            xid = nid.removeprefix("cfg:")
            d |= {"experiment": self.node(xid), "fields": self.configuration(xid)}
        elif kind == "ev":
            ev = self.evidence_detail([nid])
            d |= {"evidence": ev[0] if ev else None,
                  "contradictions": [c for cs in self.contra.values() for c in cs if nid in c["evidence_a"] + c["evidence_b"]]}
        return d

    def edge_detail(self, etype: str, source: str, target: str) -> dict:
        if etype not in EDGE_TYPES:
            raise ValueError("unknown edge type")
        if etype in ("RELATED_TO", "DIFFERS_FROM") and source.startswith("exp:") and target.startswith("exp:"):
            diff = self.diff(source, target)
            edge = comparison_edge(source, target, diff.pop("_raw"))
            if edge["type"] != etype:
                raise KeyError(etype)
            return {"edge": edge, "source": self.node(source), "target": self.node(target), "comparison": diff}
        eid = f"{etype}~{source}~{target}"
        for _, e in self._adjacent(source, True, False):
            if e["id"] == eid:
                extra = {}
                if etype == "CONTRADICTS":
                    extra["contradiction"] = next((c for cs in self.contra.values() for c in cs
                                                   if c["evidence_a"][:1] == [source] and c["evidence_b"][:1] == [target]), None)
                if source.startswith("paper:") and etype in ("RELATED_TO", "REPLICATES", "EXTENDS"):
                    extra["study_link"] = next(({k: v for k, v in link.items() if k != "edge"} for link in self.links.get(source, ())
                                                if link["edge"]["id"] == eid), None)
                return {"edge": e, "source": self.any_node(source), "target": self.any_node(target), **extra,
                        "evidence": self.evidence_detail(e["evidence_ids"][:10])}
        raise KeyError(eid)

    def diff(self, a: str, b: str, extra: dict[str, tuple[Experiment, dict]] | None = None) -> dict:
        """Field-by-field comparison of two experiments with the deterministic comparison engine. `extra` supplies
        experiments that are not corpus records (runtime experiments of a research run) as (model, node summary)."""
        extra = extra or {}
        corpus = self.experiments([x for x in (a, b) if x not in extra])
        models = {x: extra[x][0] if x in extra else corpus.get(x) for x in (a, b)}
        if missing := [x for x, m in models.items() if m is None]:
            raise KeyError(missing[0])
        res = compare(models[a], models[b])
        rows = [{"field": fc.field_name, "status": _v(fc.comparison_status), "value_a": fc.value_a, "value_b": fc.value_b,
                 "evidence_a": fc.evidence_a[:5], "evidence_b": fc.evidence_b[:5], "rationale": fc.rationale} for fc in res["field_comparisons"]]
        summary = {x: extra[x][1] if x in extra else self.node(x) for x in (a, b)}
        return {"a": summary[a], "b": summary[b], "label": _v(res["label"]), "rationale": res["rationale"], "uncertainty": res["uncertainty"],
                "missing": res["missing"], "fields": rows, "contradictions": self.contra.get(a, []) + self.contra.get(b, []),
                "labeler_version": LABELER_VERSION, "_raw": res}

    # --------------------------------------------------------------------------------------- Ask the Map
    ASK = {
        "experiments": "Which experiments are linked to this {type}?",
        "models": "Which models were evaluated here?",
        "datasets": "Which datasets or benchmarks were used?",
        "closest": "Which experiments are closest to this one?",
        "same_dataset": "Which experiments use the same dataset with a different model?",
        "contradictions": "Which results contradict each other here?",
        "replications": "Which studies replicate or extend this work?",
        "evidence": "What evidence supports this?",
    }
    QUESTIONS = {"Experiment": ("closest", "same_dataset", "evidence", "contradictions", "replications"),
                 "Paper": ("experiments", "models", "datasets", "contradictions", "replications"),
                 "Study": ("experiments",), "Dataset": ("experiments", "models"), "Model": ("experiments", "datasets"),
                 "Method": ("experiments", "datasets", "models"), "Topic": ("experiments", "datasets", "models"),
                 "Metric": ("experiments", "models", "datasets"), "Result": ("evidence", "contradictions"),
                 "Configuration": ("evidence",), "Evidence": ("contradictions",)}

    def questions_for(self, node_type: str) -> tuple[str, ...]:
        return self.QUESTIONS.get(node_type, ())

    def ask(self, nid: str, key: str) -> dict:
        """Deterministic answers from the graph about the selected node (no model call). Every item is a graph node or a
        recorded row; an empty answer says so instead of guessing."""
        node = self.any_node(nid)
        if node is None or key not in self.questions_for(node["type"]):
            raise KeyError(f"{key} for {nid}")
        out = {"key": key, "question": self.ASK[key].format(type=node["type"].lower()), "items": [], "node": node}
        exps = self.experiments_of(nid)
        if key == "experiments":
            items = sorted((self.node(x) for x in exps), key=lambda n: (n["label"].casefold(), n["id"]))
            return out | {"answer": f"{len(exps)} experiment record(s) in the corpus are linked to “{node['label']}”." + (" Showing 30." if len(exps) > 30 else ""),
                          "items": items[:30], "derivation": "experiment fields linked to this node"}
        if key in ("models", "datasets"):
            want = "Model" if key == "models" else "Dataset"
            items = self._cooccurring(exps, nid, 15).get(want, [])
            return out | {"answer": f"{len(items)} {want.lower()} node(s) appear in the {len(exps)} linked experiment record(s); counts are experiments."
                          if items else f"No {want.lower()} is recorded on the experiments linked to “{node['label']}”.",
                          "items": items, "derivation": "co-occurrence in the same experiment records"}
        if key == "closest":
            items = [{**self.node(x), "relation": _v(r["label"]), "why": _sentence(r["rationale"])} for x, r in self.related(nid, 10)]
            return out | {"answer": f"{len(items)} experiment(s) share a dataset or benchmark with this one; the comparison engine labels each pair "
                          "(an aligned setup is not the same experiment)." if items else "No experiment in the corpus shares a dataset or benchmark with this one.",
                          "items": items, "derivation": f"comparison engine {LABELER_VERSION} over experiments sharing a dataset or benchmark entity"}
        if key == "same_dataset":
            links = self.exp[nid]["links"]
            dsets = {link["entity_id"] for link in links if link["edge"] == "USES_DATASET"}
            models = {link["entity_id"] for link in links if link["edge"] == "USES_MODEL"}
            others = list(dict.fromkeys(x for d in sorted(dsets) for x in self.ent_exps.get(d, ())
                                        if x != nid and (xm := {link["entity_id"] for link in self.exp[x]["links"] if link["edge"] == "USES_MODEL"})
                                        and not xm & models))
            tasks = {link["entity_id"] for link in links if link["edge"] == "CONTAINS"}
            same_task = {x for x in others if tasks & {link["entity_id"] for link in self.exp[x]["links"] if link["edge"] == "CONTAINS"}}
            # the same task too comes first: those are the directly comparable results
            items = sorted((self.node(x) for x in others), key=lambda n: (n["id"] not in same_task, n["label"].casefold(), n["id"]))[:30]
            answer = (f"{len(others)} experiment(s) use the same dataset or benchmark with a different model"
                      + (f", {len(same_task)} of them on the same task." if tasks else ".") + (" Showing 30." if len(others) > 30 else "")
                      if others else "No other experiment records this dataset or benchmark with a different model." if dsets
                      else "This experiment records no dataset or benchmark.")
            return out | {"answer": answer, "items": items, "derivation": "same dataset/benchmark entity, different model entity (exact canonical ids)"}
        if key == "contradictions":
            if node["type"] == "Evidence":
                found = [c for cs in self.contra.values() for c in cs if nid in c["evidence_a"] + c["evidence_b"]]
            elif node["type"] == "Result":
                found = [c for c in self.contra.get(self._result(nid)[0], ()) if c["field"] == "evaluation.score"]
            else:
                found = [c for s in dict.fromkeys([nid, *exps]) for c in self.contra.get(s, ())]
            return out | {"answer": f"{len(found)} recorded contradiction(s): sources report different values for the same fact; neither side is chosen automatically."
                          if found else "No contradiction is recorded for this node in the corpus.", "contradictions": found,
                          "derivation": "rx.contradictions (both sides preserved)"}
        if key == "replications":
            pid = nid if node["type"] == "Paper" else self.exp[nid]["paper_id"]
            links = self.links.get(pid, [])
            reps = [link for link in links if link["edge"]["type"] in ("REPLICATES", "EXTENDS")]
            other = sorted({link["label"].replace("_", " ").lower() for link in links} - {link["label"].replace("_", " ").lower() for link in reps})
            answer = (f"{len(reps)} replication/extension link(s) recorded (unreviewed candidates)." if reps else
                      "No replication or extension of this work is recorded in the corpus." + (f" Other candidate link(s): {', '.join(other)}." if other else ""))
            return out | {"answer": answer, "items": [self.node(link["b"] if link["a"] == pid else link["a"]) for link in links],
                          "derivation": "rx.study_links (links resting on title similarity alone are not shown)"}
        if key == "evidence":
            if node["type"] == "Result":
                ids = self._result(nid)[1]["evidence_ids"]
            elif node["type"] == "Configuration":
                xid = nid.removeprefix("cfg:")
                ids = [i for c in self.configuration(xid) for i in c["evidence_ids"]]
            else:
                e = self.exp[nid]
                ids = [i for link in e["links"] for i in link["evidence_ids"]] + [i for r in e["results"] for i in r["evidence_ids"]]
            ev = self.evidence_detail(ids)
            return out | {"answer": f"{len(ev)} evidence span(s) with source locations" + (f" (first 20 of {len(set(ids))})." if len(set(ids)) > 20 else "."),
                          "evidence": ev, "derivation": "evidence ids recorded on the fields"}
        raise KeyError(key)


_INDEX: GraphIndex | None = None
_LOCK = threading.Lock()


def get_index(q: Callable[..., list]) -> GraphIndex:
    """The process-wide index, built on first use (one build at a time)."""
    global _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = GraphIndex(q)
        return _INDEX
