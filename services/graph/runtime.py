"""Runtime research graph (docs/ARCHITECTURE.md §65): what one research run found, kept apart from canonical knowledge.

Built at the end of a run from that run's own evidence objects: its sources (corpus passages, dataset records, external
results that passed the source gate, the user's private documents), the facts that passed deterministic validation and
the verified claims. Nothing is inferred from a title or from similarity: an external paper that contributed no
validated fact appears only as a retrieved, unverified source.

Every node and edge carries
  origin     canonical | runtime_external | private_document | user_input | derived
  status     validated | unverified | related | promoted | rejected
  provenance provider, URL, identifiers, retrieval time, content hash, research_run_id, evidence reference, method.
Identity is merged with the canonical graph only on reliable identifiers: DOI, arXiv id, corpus ids, and the canonical
entity key (services.canonicalization.entities: the same normalised name gives the same id). A matching title alone
keeps the nodes separate, linked by a POSSIBLE edge. The graph is stored with the run (rx.research_runs.answer, owner-only
RLS), never in the corpus; promotion of an external source is an explicit, reviewed step (rx.runtime_sources.state).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter
from typing import Callable

from services.canonicalization.entities import entity_id
from services.canonicalization.entities import key as entity_key
from services.canonicalization.normalize import normalize_doi, normalize_title, parse_arxiv_id
from services.datasets.schemas import FIELD_GROUPS, Experiment
from services.datasets.schemas import CanonicalEntityType as E
from services.graph.knowledge_map import ENTITY_NODE, make_edge

VERSION = "runtime-graph-0.1.0"
ORIGINS = ("canonical", "runtime_external", "private_document", "user_input", "derived")
STATUSES = ("validated", "unverified", "related", "promoted", "rejected")
RANK = {"rejected": 0, "unverified": 1, "related": 2, "validated": 3, "promoted": 4}
PROVENANCE_EDGES = ("ASKED_ABOUT", "RETRIEVED", "QUOTED_FROM")  # what the run did, not research relationships
SOURCE_CHECKS = ["https", "no_instruction_like_text", "not_duplicate", "min_content"]  # services.runtime.tools.validate_sources
SCORE = "evaluation.score"
TYPE_ORDER = {t: i for i, t in enumerate(("Run", "Topic", "Paper", "Source", "Document", "Experiment", "Model", "Method", "Dataset",
                                          "Configuration", "Result", "Metric", "Evidence"))}
MAX_NODES, MAX_EDGES = 250, 500


def _h(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _short(s, n: int = 140) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _arxiv(raw: str | None) -> str | None:
    p = parse_arxiv_id(raw) if raw else None
    return p[0] if p else None


class RunGraph:
    """Nodes and edges of one run, deduplicated by id: a node reached twice keeps its strongest status and the
    canonical origin if either path was canonical."""

    def __init__(self, run_id: str):
        self.run_id, self.nodes, self.edges = run_id, {}, {}

    def node(self, nid: str, ntype: str, label, origin: str, status: str, provenance: dict | None = None, **meta) -> str:
        meta = {k: v for k, v in meta.items() if v not in (None, [], {}, "")}
        n = self.nodes.get(nid)
        if n is None:
            self.nodes[nid] = {"id": nid, "type": ntype, "label": _short(label) or ntype, "origin": origin, "status": status,
                               "provenance": dict(provenance or {}), "meta": meta, "flags": []}
            return nid
        if RANK[status] > RANK[n["status"]]:
            n["status"] = status
        if origin == "canonical":
            n["origin"] = "canonical"
        for k, v in (provenance or {}).items():
            n["provenance"].setdefault(k, v)
        for k, v in meta.items():
            n["meta"].setdefault(k, v)
        return nid

    def edge(self, etype: str, source: str, target: str, why: str, derivation: str, origin: str, status: str, strength: str = "EXACT",
             evidence_ids=(), label: str | None = None, provenance: dict | None = None) -> None:
        e = make_edge(etype, source, target, why, derivation, strength, evidence_ids, label)
        e.update(origin=origin, status=status, provenance={"research_run_id": self.run_id, **(provenance or {})})
        old = self.edges.get(e["id"])
        if old is None or RANK[status] > RANK[old["status"]]:
            self.edges[e["id"]] = e

    def flag(self, nid: str | None, flag: str) -> None:
        if nid in self.nodes and flag not in self.nodes[nid]["flags"]:
            self.nodes[nid]["flags"].append(flag)


_TITLES: dict[str, tuple[str, str]] | None = None
_TITLES_LOCK = threading.Lock()


def _title_index(q: Callable[..., list]) -> dict[str, tuple[str, str]]:
    """normalised title -> corpus paper, read once per process (the corpus is a frozen release)."""
    global _TITLES
    with _TITLES_LOCK:
        if _TITLES is None:
            _TITLES = {}
            for pid, title in q("select id, title from rx.papers"):
                _TITLES.setdefault(normalize_title(title), (pid, title))
        return _TITLES


def _identities(q: Callable[..., list] | None, dois: set, arxivs: set, titles: set, ent_ids: set) -> dict:
    """One corpus lookup for every identifier of the run: DOI / arXiv id -> paper, entity id -> name; titles are matched
    against a cached index (a title match is only ever a candidate, never an identity)."""
    out: dict[str, dict] = {"doi": {}, "arxiv": {}, "title": {}, "entity": {}}
    if q is None:
        return out
    parts = [(sql, sorted(v)) for sql, v in (("select 'doi', lower(doi), id, title from rx.papers where lower(doi) = any(%s)", dois),
                                             ("select 'arxiv', arxiv_id, id, title from rx.papers where arxiv_id = any(%s)", arxivs),
                                             ("select 'entity', id, id, preferred_name from rx.canonical_entities where id = any(%s)", ent_ids)) if v]
    if parts:
        for kind, k, nid, name in q(" union all ".join(sql for sql, _ in parts), tuple(v for _, v in parts)):
            out[kind].setdefault(k, (nid, name))
    if titles:
        index = _title_index(q)
        out["title"] = {t: index[t] for t in titles if t in index}
    return out


def _entity(ident: dict, candidates: list[tuple[E, str | None]]) -> tuple[str, str, bool] | None:
    """(entity id, node type, canonical?) for a name: the first candidate type the corpus knows, else the first type."""
    ids = [(i, t) for t, raw in candidates if raw and (i := entity_id(t, raw))]
    if not ids:
        return None
    for i, t in ids:
        if i in ident["entity"]:
            return i, ENTITY_NODE[t.value], True
    return ids[0][0], ENTITY_NODE[ids[0][1].value], False


def _external_id(s: dict, ident: dict) -> tuple[str, bool, str | None]:
    """(node id, merged with a corpus paper?, corpus paper with the same title but no shared identifier)."""
    doi, arxiv = normalize_doi(s.get("doi")), _arxiv(s.get("arxiv_id"))
    for kind, k in (("doi", doi), ("arxiv", arxiv)):
        if k and k in ident[kind]:
            return ident[kind][k][0], True, None
    nid = f"rt:paper:doi:{doi}" if doi else f"rt:paper:arxiv:{arxiv}" if arxiv else f"rt:web:{_h(s.get('url'))}"
    same_title = ident["title"].get(normalize_title(s.get("title")))
    return nid, False, same_title[0] if same_title else None


def _external_provenance(s: dict) -> dict:
    return {"provider": s.get("provider"), "url": s.get("url"), "doi": normalize_doi(s.get("doi")), "arxiv_id": _arxiv(s.get("arxiv_id")),
            "source_id": s.get("source_id"), "retrieved_at": s.get("retrieved_at"), "content_sha256": s.get("content_sha256"),
            "reliability": s.get("reliability"), "source_validation": SOURCE_CHECKS, "also_found_by": s.get("also_found_by") or []}


def _external_node(g: RunGraph, s: dict, ident: dict, status: str) -> tuple[str, bool, bool]:
    """Add an external source; returns (node id, merged with a corpus paper, uncertain title-only match)."""
    nid, is_canon, same_title = _external_id(s, ident)
    prov = _external_provenance(s)
    if is_canon:
        g.node(nid, "Paper", s.get("title"), "canonical", status, {"runtime": prov})
        return nid, True, False
    ntype = "Paper" if (prov["doi"] or prov["arxiv_id"] or s.get("reliability") == "scholarly") else "Source"
    g.node(nid, ntype, s.get("title") or s.get("url"), "runtime_external", status, prov, year=s.get("year"), venue=s.get("venue"))
    if same_title:
        g.node(same_title, "Paper", same_title, "canonical", "related")
        g.edge("RELATED_TO", nid, same_title, "Same normalised title as a corpus paper but no shared DOI or arXiv id, so identity is not assumed.",
               "title comparison (candidate match only)", "derived", "related", "POSSIBLE", label="possibly the same paper")
    return nid, False, bool(same_title)


def build(run_id: str, result: dict, inputs: dict, q: Callable[..., list] | None = None) -> dict:
    """The runtime graph of one finished run. `inputs`: query, intent, request {model, dataset, metric}, entities (query
    entities from the corpus), external (accepted external results), facts (validated facts used by the answer),
    rejected (facts that failed validation), conflicts (fields with disagreeing validated values, as fact indices),
    result_ids (corpus evidence id -> result node id). `q` runs the identity lookup on the corpus."""
    t0 = time.perf_counter()
    g = RunGraph(run_id)
    sources = result.get("sources") or []
    by_sid = {s["sid"]: s for s in sources}
    by_chunk = {s["chunk_id"]: s for s in sources if s.get("chunk_id")}
    cited = set(result.get("cited_sources") or [])
    req = inputs.get("request") or {}
    facts = inputs.get("facts") or []
    def keys(s: dict) -> set:  # a work is used when any of its sources is (a full-text passage stands for its paper)
        return {k for k in (s.get("url"), normalize_doi(s.get("doi")), _arxiv(s.get("arxiv_id"))) if k}
    used_keys = set().union(*(keys(s) for s in sources if s.get("kind") == "external"))
    unused = [a for a in inputs.get("external") or [] if not keys(a) & used_keys]
    externals = [s for s in sources if s.get("kind") == "external"] + unused

    # ---- identity lookup (one round trip)
    ent_candidates = {"model": [(E.MODEL, req.get("model"))], "dataset": [(E.BENCHMARK, req.get("dataset")), (E.DATASET, req.get("dataset"))],
                      "metric": [(E.METRIC, req.get("metric"))]}
    ent_ids = {i for cands in ent_candidates.values() for t, raw in cands if raw and (i := entity_id(t, raw))}
    ident = _identities(q, {d for s in externals if (d := normalize_doi(s.get("doi")))}, {a for s in externals if (a := _arxiv(s.get("arxiv_id")))},
                        {t for s in externals if (t := normalize_title(s.get("title")))}, ent_ids)

    # ---- the run and what it asked about
    run = g.node(f"run:{run_id}", "Run", inputs.get("query") or run_id, "user_input", "validated", {"research_run_id": run_id, "scope": "owner only"},
                 intent=inputs.get("intent"), run_status=result.get("status"), evidence_status=result.get("evidence_status"))
    asked: dict[str, str] = {}
    for role, cands in ent_candidates.items():  # the parsed request only: raw n-gram hits on entity names are too noisy to draw
        raw = next((r for _, r in cands if r), None)
        hit = _entity(ident, cands) if raw else None
        if hit and (hit[2] or inputs.get("asked_unknown")):
            nid, ntype, canonical = hit
            asked[role] = g.node(nid, ntype, ident["entity"][nid][1] if canonical else raw, "canonical" if canonical else "user_input",
                                 "validated" if canonical else "unverified", {"asked_as": raw})
            g.edge("ASKED_ABOUT", run, nid, f"The question names the {role} “{raw}”" + ("." if canonical else "; the corpus has no entity with this name."),
                   "query understanding (request parser)", "user_input", "validated")

    # ---- sources the run retrieved
    node_of: dict[str, str] = {}  # sid / chunk id -> source node id
    merged = uncertain = 0
    for s in sources:
        status = "validated" if s["sid"] in cited or s.get("role") == "SUPPORTS" else "related"
        kind = s.get("kind")
        if kind in ("internal", "record") and s.get("paper_id"):
            nid = g.node(s["paper_id"], "Paper", s.get("title") or s["paper_id"], "canonical", status, year=s.get("year"))
            g.edge("RETRIEVED", run, nid, f"Retrieved as {s['sid']} ({'cited' if status == 'validated' else 'related context'}).", "research run sources",
                   "derived", status, provenance={"sid": s["sid"], "chunk_id": s.get("chunk_id"), "section": s.get("section")})
            for xid in (s.get("experiment_ids") or [])[:6]:
                g.node(xid, "Experiment", "Experiment", "canonical", "related")
                g.edge("REPORTS", nid, xid, "The experiment record was extracted from this paper.", "rx.experiments.paper_id", "canonical", "validated")
        elif kind == "external":
            nid, is_canon, unsure = _external_node(g, s, ident, status)
            merged, uncertain = merged + is_canon, uncertain + unsure
            g.edge("RETRIEVED", run, nid, f"Found by {s.get('provider')} and accepted by the source gate; " + ("its text supports this answer."
                   if status == "validated" else "shown as related context."), "external research (evidence gap)", "runtime_external", status,
                   provenance={"sid": s["sid"], "provider": s.get("provider"), "url": s.get("url"), "retrieved_at": s.get("retrieved_at")})
        elif kind == "private":
            nid = g.node(f"doc:{s.get('document_id')}", "Document", s.get("title") or "Private document", "private_document", status,
                         {"document_id": s.get("document_id"), "scope": "private: visible only to its owner"})
            g.edge("RETRIEVED", run, nid, f"Retrieved from your private documents as {s['sid']}.", "private document retrieval (owner-scoped)",
                   "private_document", status, provenance={"sid": s["sid"], "chunk_id": s.get("chunk_id")})
        else:
            continue
        node_of[s["sid"]] = nid
        if s.get("chunk_id"):
            node_of[s["chunk_id"]] = nid
    for a in unused:  # accepted by the gate but not used: discovered, not established
        nid, is_canon, unsure = _external_node(g, a, ident, "unverified")
        merged, uncertain = merged + is_canon, uncertain + unsure
        g.edge("RETRIEVED", run, nid, f"Found by {a.get('provider')} and accepted by the source gate, but nothing in it was used as evidence for this answer.",
               "external research (evidence gap)", "runtime_external", "unverified", provenance={"provider": a.get("provider"), "url": a.get("url")})
    by_title: dict[str, list[str]] = {}  # one work found by several providers without a shared identifier: linked, never merged
    for n in g.nodes.values():
        if n["origin"] == "runtime_external" and n["type"] in ("Paper", "Source") and (t := normalize_title(n["label"])):
            by_title.setdefault(t, []).append(n["id"])
    for ids in by_title.values():
        for other in sorted(ids)[1:]:
            uncertain += 1
            g.edge("RELATED_TO", other, sorted(ids)[0], "Same normalised title from another provider, but no shared DOI or arXiv id, so identity is not assumed.",
                   "title comparison (candidate match only)", "derived", "related", "POSSIBLE", label="possibly the same work")

    # ---- validated facts: evidence -> experiment / configuration / result
    ev_of: dict[int, str] = {}
    for i, f in enumerate(facts):
        s = by_chunk.get(f["chunk_id"]) or by_sid.get(f.get("sid"))
        snode = node_of.get(f["chunk_id"]) or (node_of.get(s["sid"]) if s else None)
        if not s or not snode:
            continue
        origin = {"external": "runtime_external", "private": "private_document"}.get(s.get("kind"), "canonical" if f.get("evidence_id") else "derived")
        label, value = f.get("label") or f["field"], f.get("value")
        if f.get("evidence_id"):
            ev = g.node(f["evidence_id"], "Evidence", f"{label}: {f.get('span')}", "canonical", "validated", {"sid": s["sid"], "locator": f.get("locator")},
                        field=f["field"], value=value)
        else:
            ev = g.node(f"rt:ev:{_h(f['chunk_id'], f.get('span'))}", "Evidence", f"{label}: {f.get('span')}", origin, "validated",
                        {"sid": s["sid"], "url": s.get("url"), "char_start": f.get("start"), "char_end": f.get("end"),
                         "extraction": f"{f.get('origin')} + deterministic validation"}, field=f["field"], value=value, quote=_short(f.get("span"), 300))
        ev_of[i] = ev
        g.edge("QUOTED_FROM", ev, snode, f"The span is quoted from {s['sid']}" + (f" ({s.get('provider')})." if s.get("provider") else "."),
               "run evidence (source span)", origin, "validated")
        why = f"{label} = {value}: the span names the field and states the value (deterministic validation)."
        if f.get("experiment_ids"):  # corpus experiments this evidence belongs to
            for xid in list(f["experiment_ids"])[:3]:
                g.node(xid, "Experiment", f.get("config_detail") or "Experiment", "canonical", "validated")
                if s.get("paper_id"):
                    g.edge("REPORTS", s["paper_id"], xid, "The experiment record was extracted from this paper.", "rx.experiments.paper_id", "canonical", "validated")
                subject = xid
                rid = (inputs.get("result_ids") or {}).get(f.get("evidence_id") or "")
                if f["field"] == SCORE and rid and rid.startswith(f"res:{xid}#"):
                    subject = g.node(rid, "Result", f"{f.get('metric') or 'score'} = {value}", "canonical", "validated")
                    g.edge("PRODUCES_RESULT", xid, rid, "Outcome of this experiment.", "rx.experiments.record.outcomes", "canonical", "validated")
                elif f["field"] != SCORE:
                    subject = g.node(f"cfg:{xid}", "Configuration", "Configuration", "canonical", "validated")
                    g.edge("HAS_CONFIGURATION", xid, subject, "Configuration of this experiment.", "rx.experiments.record", "canonical", "validated")
                g.edge("SUPPORTED_BY", subject, ev, why, "corpus evidence linked to the experiment", "canonical", "validated", evidence_ids=[ev])
            continue
        # a source outside the corpus's experiment records: one runtime experiment per (source, requested configuration)
        model, dataset, metric = req.get("model"), req.get("dataset"), f.get("metric") or req.get("metric")
        strength, config = ("EXACT" if f.get("config") == "EXACT" else "STRONG"), str(f.get("config")).lower()
        xid = g.node(f"rt:exp:{_h(snode, entity_key(model or ''), entity_key(dataset or ''))}", "Experiment",
                     " · ".join(x for x in (model, dataset) if x) or "Experiment", origin, "validated",
                     {"source": snode, "sid": s["sid"], "extraction": f.get("origin")}, configuration_match=f.get("config"))
        g.edge("REPORTS", snode, xid, f"{s['sid']} states the {label} for this configuration (configuration match {config}).",
               f"{f.get('origin')} + deterministic validation", origin, "validated", strength, [ev])
        for role, etype in (("model", "USES_MODEL"), ("dataset", "USES_DATASET")):
            if role in asked and (role == "model" or f.get("dataset_ok") is True):
                g.edge(etype, xid, asked[role], f"The source names the requested {role} for this configuration (configuration match {config}).",
                       "grounding configuration match", origin, "validated", strength, [ev])
        if f["field"] == SCORE:
            subject = g.node(f"rt:res:{_h(xid, metric or '', value)}", "Result", f"{metric or 'score'} = {value}", origin, "validated", {"sid": s["sid"]},
                             metric=metric, score=value, normalized=f.get("normalized"))
            g.edge("PRODUCES_RESULT", xid, subject, f"Reported by {s['sid']}.", f"{f.get('origin')} + deterministic validation", origin, "validated", strength, [ev])
            if "metric" in asked:
                g.edge("MEASURES", subject, asked["metric"], "The question's metric; the span reports this score.", "grounding metric match", origin,
                       "validated", strength, [ev])
        else:
            subject = g.node(f"rt:cfg:{xid.rsplit(':', 1)[-1]}", "Configuration", "Configuration", origin, "validated", {"sid": s["sid"]})
            fields = g.nodes[subject]["meta"].setdefault("fields", [])
            if not any(x["field"] == f["field"] and x["value"] == value for x in fields):
                fields.append({"field": f["field"], "label": label, "value": value, "normalized": f.get("normalized"), "evidence": ev})
            g.edge("HAS_CONFIGURATION", xid, subject, "Configuration stated by the source.", f"{f.get('origin')} + deterministic validation", origin,
                   "validated", strength, [ev])
        g.edge("SUPPORTED_BY", subject, ev, why, f"{f.get('origin')} + deterministic validation", origin, "validated", evidence_ids=[ev])

    # ---- disagreeing validated values: both kept, linked as a contradiction
    for c in inputs.get("conflicts") or []:
        firsts: dict[str, int] = {}
        for i in c["facts"]:
            if i in ev_of:
                firsts.setdefault(str(facts[i].get("value")), i)
        vals = list(firsts.items())[:3]
        for (va, ia), (vb, ib) in zip(vals, vals[1:], strict=False):  # consecutive pairs
            a, b = ev_of[ia], ev_of[ib]
            if a != b:
                g.edge("CONTRADICTS", a, b, f"Validated values disagree for {c['label']}: {va} ({facts[ia].get('sid')}) vs {vb} ({facts[ib].get('sid')}). "
                       "Both are kept; neither is chosen.", "field-level grounding (disagreeing validated values)", "derived", "validated", "CONTRADICTORY",
                       [a, b], "contradicts")
                g.flag(a, "conflict")
                g.flag(b, "conflict")
    for c in result.get("contradictions") or []:  # recorded corpus contradictions on the run's subjects
        g.flag(c.get("subject_id"), "conflict")

    # ---- verified claims of a passage answer: quote -> source
    if not facts:
        for c in result.get("claims") or []:
            for sid in (c.get("sources") or [])[:2]:
                s = by_sid.get(sid)
                if not s or sid not in node_of or not c.get("quote"):
                    continue
                origin = {"external": "runtime_external", "private": "private_document"}.get(s.get("kind"), "derived")
                ev = g.node(f"rt:ev:{_h(s.get('chunk_id') or s.get('url'), c['quote'])}", "Evidence", c["quote"], origin, "validated",
                            {"sid": sid, "url": s.get("url"), "extraction": "citation validation (verbatim quote, numbers in quote)"},
                            claim=_short(c.get("text"), 300), quote=_short(c["quote"], 300))
                g.edge("QUOTED_FROM", ev, node_of[sid], f"Verbatim quote from {sid}.", "citation validation", origin, "validated")
                g.edge("SUPPORTED_BY", run, ev, f"Verified claim of this answer: “{_short(c.get('text'), 160)}”", "citation validation", origin, "validated",
                       evidence_ids=[ev], label="answer claim")

    rejected = [{"field": f.get("field"), "value": f.get("value"), "reason": f.get("reason"),
                 "source": (by_chunk.get(f.get("chunk_id")) or {}).get("sid") or f.get("chunk_id"), "origin": f.get("origin"), "status": "rejected"}
                for f in (inputs.get("rejected") or [])][:20]
    nodes = sorted(g.nodes.values(), key=lambda n: (TYPE_ORDER.get(n["type"], 99), n["id"]))[:MAX_NODES]
    keep = {n["id"] for n in nodes}
    edges = sorted((e for e in g.edges.values() if e["source"] in keep and e["target"] in keep), key=lambda e: e["id"])[:MAX_EDGES]
    stats = {"nodes": len(nodes), "edges": len(edges), "runtime_nodes": sum(n["origin"] != "canonical" for n in nodes),
             "canonical_nodes": sum(n["origin"] == "canonical" for n in nodes), "by_origin": dict(Counter(n["origin"] for n in nodes)),
             "by_status": dict(Counter(n["status"] for n in nodes)), "by_type": dict(Counter(n["type"] for n in nodes)),
             "edges_by_type": dict(Counter(e["type"] for e in edges)), "merged_by_identifier": merged, "uncertain_identity": uncertain,
             "rejected_facts": len(rejected), "build_ms": round((time.perf_counter() - t0) * 1000)}
    return {"version": VERSION, "research_run_id": run_id, "nodes": nodes, "edges": edges, "rejected": rejected, "stats": stats}


def runtime_experiment(graph: dict, xid: str) -> Experiment:
    """A runtime experiment as an Experiment model (only its validated fields are KNOWN), so the comparison engine can
    diff it against corpus experiments exactly like two corpus records."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    x = nodes[xid]
    src = nodes.get(x["provenance"].get("source"), {})
    prov = src.get("provenance") or {}

    def fv(raw, value, evidence) -> dict:
        return {"status": "KNOWN", "raw_value": str(raw), "value": value if value is not None else raw, "evidence_ids": list(evidence) or [xid]}

    rec: dict = {"experiment_id": xid, "study_id": f"runtime:{graph['research_run_id']}", "paper_id": src.get("id") or xid,
                 "source_record_ids": [prov.get("url") or src.get("id") or xid],
                 "extraction_method": "LLM" if "runtime" in str(x["provenance"].get("extraction")) else "RULES",
                 "extraction_run_id": graph["research_run_id"], "quality_tier": "BRONZE", "outcomes": [],
                 "provenance": {"source_name": "RESEARCHX", "source_url": prov.get("url"), "retrieved_at": prov.get("retrieved_at"),
                                "content_hash": prov.get("content_sha256"), "processing_version": VERSION}}
    for e in graph["edges"]:
        if e["source"] != xid or e["target"] not in nodes:
            continue
        t = nodes[e["target"]]
        if e["type"] == "USES_MODEL":
            rec.setdefault("model", {})["model_name"] = fv(t["label"], t["label"], e["evidence_ids"])
        elif e["type"] == "USES_DATASET":
            rec.setdefault("evaluation", {})["benchmark"] = fv(t["label"], t["label"], e["evidence_ids"])
        elif e["type"] == "HAS_CONFIGURATION":
            for item in t["meta"].get("fields", []):
                group, name = item["field"].split(".", 1)
                if name in FIELD_GROUPS.get(group, ()):
                    rec.setdefault(group, {})[name] = fv(item["value"], item.get("normalized"), [item["evidence"]])
        elif e["type"] == "PRODUCES_RESULT":
            m = t["meta"]
            rec["outcomes"].append({"score": fv(m.get("score"), m.get("normalized"), e["evidence_ids"]),
                                    **({"metric": fv(m["metric"], m["metric"], e["evidence_ids"])} if m.get("metric") else {})})
    return Experiment.model_validate(rec)


RUN_QUESTIONS = {"discoveries": "What did this research run discover?", "evidence": "Which evidence supports this answer?",
                 "rejected": "Which extracted facts were rejected, and why?"}


def ask_run(graph: dict, key: str) -> dict:
    """Deterministic answers about one run's graph (no model call); every item is a node of that graph."""
    if key not in RUN_QUESTIONS:
        raise KeyError(key)
    nodes, out = graph["nodes"], {"key": key, "question": RUN_QUESTIONS[key], "items": [], "derivation": f"runtime graph {graph.get('version')}"}
    if key == "discoveries":
        ext = [n for n in nodes if n["origin"] == "runtime_external"]
        private = [n for n in nodes if n["origin"] == "private_document"]
        counts = Counter(n["status"] for n in ext)
        answer = (f"{len(ext)} item(s) found outside the corpus in this run (" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + ")"
                  if ext else "Nothing outside the corpus was found in this run")
        answer += f"; {len(private)} item(s) from your private documents." if private else "."
        items = sorted(ext + private, key=lambda n: (-RANK[n["status"]], TYPE_ORDER.get(n["type"], 99), n["label"]))[:40]
        return out | {"answer": answer + " Unverified items were retrieved but contributed no validated evidence.", "items": items}
    if key == "evidence":
        ev = [n for n in nodes if n["type"] == "Evidence" and n["status"] == "validated"]
        return out | {"answer": f"{len(ev)} validated evidence span(s) support this answer." if ev else "No validated evidence span supports this answer.",
                      "items": ev[:40]}
    rej = graph.get("rejected") or []
    return out | {"answer": f"{len(rej)} extracted fact(s) failed validation or matched a different configuration; none is used." if rej
                  else "No extracted fact was rejected.", "rejected": rej}


NODE_QUESTIONS = {"evidence": "What evidence supports this?", "closest": "Which corpus experiments are closest to this one?"}


def questions_for(node: dict) -> list[str]:
    return list(RUN_QUESTIONS) if node["type"] == "Run" else ["closest", "evidence"] if node["type"] == "Experiment" else ["evidence"]


def ask_node(graph: dict, nid: str, key: str, index=None) -> dict:
    """Deterministic answers about one runtime node: its validated evidence, or (runtime experiment) the corpus
    experiments sharing its dataset entity, each pair labelled by the comparison engine. `index` is the corpus GraphIndex."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    n = nodes.get(nid)
    if n is None or key not in questions_for(n) or key not in NODE_QUESTIONS:
        raise KeyError(f"{key} for {nid}")
    out = {"key": key, "question": NODE_QUESTIONS[key], "node": n, "items": []}
    if key == "evidence":
        near = {nid} | {e["target"] for e in graph["edges"] if e["source"] == nid and e["type"] in ("HAS_CONFIGURATION", "PRODUCES_RESULT")}
        ids = {e["target"] for e in graph["edges"] if e["source"] in near and e["type"] == "SUPPORTED_BY"}
        ids |= {e["source"] for e in graph["edges"] if e["target"] == nid and e["type"] == "QUOTED_FROM"}
        items = [nodes[i] for i in sorted(ids) if i in nodes]
        return out | {"answer": f"{len(items)} validated evidence span(s)." if items else "No validated evidence span is attached to this node in this run.",
                      "items": items, "derivation": "SUPPORTED_BY / QUOTED_FROM edges of this run's graph"}
    if index is None:
        raise KeyError("closest needs the corpus index")
    from services.graph.knowledge_map import LABEL_RANK, _same_fields, _v, compare  # one comparison engine for corpus and runtime

    model = runtime_experiment(graph, nid)
    dsets = [e["target"] for e in graph["edges"] if e["source"] == nid and e["type"] == "USES_DATASET"]
    pool = sorted({x for d in dsets for x in index.ent_exps.get(d, ())})[:30]
    corpus = index.experiments(pool)
    pairs = sorted(((x, compare(model, corpus[x])) for x in pool if x in corpus),
                   key=lambda t: (LABEL_RANK.get(_v(t[1]["label"]), 9), -_same_fields(t[1]), t[0]))[:10]
    items = [{**index.node(x), "relation": _v(r["label"]), "why": r["rationale"]} for x, r in pairs]
    return out | {"answer": f"{len(items)} corpus experiment(s) share its dataset; the comparison engine labels each pair (unknown is never a match)."
                  if items else "No corpus experiment shares this experiment's dataset.", "items": items,
                  "derivation": "comparison engine over corpus experiments with the same dataset entity"}


def apply_review_states(graph: dict, states: dict[tuple[str, str], str]) -> dict:
    """Overlay the reviewed state of runtime sources (rx.runtime_sources: CANDIDATE / PROMOTED / REJECTED) on a run graph."""
    for n in graph["nodes"]:
        p = n["provenance"]
        state = states.get((p.get("url"), p.get("content_sha256"))) if n["origin"] == "runtime_external" else None
        if state == "PROMOTED":
            n["status"] = "promoted"
        elif state == "REJECTED":
            n["status"] = "rejected"
        if state:
            n["meta"]["review_state"] = state
    return graph


def promotion_checks(graph: dict, nid: str) -> dict:
    """Whether a runtime node may be proposed for the canonical corpus, and why. Promotion itself happens only in the
    release pipeline after review; private documents are never proposed."""
    n = next((x for x in graph["nodes"] if x["id"] == nid), None)
    if n is None:
        raise KeyError(nid)
    if n["origin"] == "private_document":
        return {"eligible": False, "reason": "Private documents are never promoted to the shared corpus.", "checks": []}
    if n["origin"] == "canonical":
        return {"eligible": False, "reason": "Already canonical knowledge.", "checks": []}
    if n["type"] not in ("Paper", "Source") or n["origin"] != "runtime_external":
        return {"eligible": False, "reason": "Only external sources are proposed; their experiments and entities are re-extracted by the release pipeline.",
                "checks": []}
    p = n["provenance"]
    supported = any(e["type"] == "QUOTED_FROM" and e["target"] == nid and e["status"] == "validated" for e in graph["edges"])
    checks = [{"check": "source_validation", "ok": p.get("source_validation") == SOURCE_CHECKS, "detail": "passed the runtime source gate"},
              {"check": "provenance", "ok": all(p.get(k) for k in ("provider", "url", "retrieved_at", "content_sha256")),
               "detail": "provider, URL, retrieval time and content hash recorded"},
              {"check": "identifier", "ok": bool(p.get("doi") or p.get("arxiv_id")), "detail": "a DOI or arXiv id lets the release deduplicate it"},
              {"check": "not_in_corpus", "ok": True, "detail": "no corpus paper shares its DOI or arXiv id"},
              {"check": "validated_evidence", "ok": supported, "detail": "at least one span from it passed validation in this run"},
              {"check": "license", "ok": None, "detail": "checked by the release pipeline before ingestion (unknown means not redistributable)"}]
    ok = all(c["ok"] for c in checks if c["check"] in ("source_validation", "provenance", "identifier"))
    return {"eligible": ok, "reason": "Eligible for review." if ok else "Missing provenance or identifier: not eligible.", "checks": checks}
