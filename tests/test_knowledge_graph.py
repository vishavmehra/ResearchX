"""Knowledge Map unit tests (no network): the canonical corpus graph (services/graph/knowledge_map.py), the runtime research
graph of one run (services/graph/runtime.py) and the orchestrator's runtime-provider phase.

Every corpus here is synthetic and built inside the test, with invented names, so a passing test cannot depend on a
string hardcoded in the modules (the mutation test renames everything and checks that the graph follows the data).
"""

import copy
import re
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.canonicalization.entities import entity_id
from services.comparison.fields import compare
from services.datasets.schemas import CanonicalEntityType as E
from services.datasets.schemas import Experiment
from services.datasets.store import config
from services.graph import knowledge_map as km
from services.graph import runtime as rg
from services.runtime import tools
from services.runtime.orchestrator import Orchestrator, search_queries
from services.runtime.trace import Stage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _empty_title_index(monkeypatch):
    """The runtime builder caches corpus titles per process; tests start from an empty, private index."""
    monkeypatch.setattr(rg, "_TITLES", {})


# ------------------------------------------------------------------------------------------------ synthetic corpus
def fv(value, ev):
    return {"status": "KNOWN", "raw_value": str(value), "value": value, "evidence_ids": [ev]}


def exp_record(xid, pid, model, bench, task, metric, score, **extra):
    rec = {"experiment_id": xid, "study_id": f"study:{pid}", "paper_id": pid, "source_record_ids": [pid], "extraction_method": "RULES",
           "extraction_run_id": "test-run", "quality_tier": "SILVER", "provenance": {"source_name": "RESEARCHX", "processing_version": "test"},
           "model": {"model_name": fv(model, f"ev:{xid}m")}, "evaluation": {"benchmark": fv(bench, f"ev:{xid}b")},
           "research": {"task": fv(task, f"ev:{xid}t")},
           "outcomes": [{"metric": fv(metric, f"ev:{xid}om"), "score": fv(score, f"ev:{xid}os")}]}
    for path, value in extra.items():
        group, name = path.split("__")
        rec.setdefault(group, {})[name] = fv(value, f"ev:{xid}{name}")
    Experiment.model_validate(rec)  # every synthetic record is a valid Experiment
    return rec


class FakeCorpus:
    """Answers exactly the read statements GraphIndex issues, from in-memory rows; records every statement."""

    def __init__(self, papers, records, citations=(), links=(), contradictions=(), evidence=None):
        self.papers, self.records, self.citations, self.links = papers, records, list(citations), list(links)
        self.contradictions, self.evidence, self.sql = list(contradictions), evidence or {}, []
        names: dict[str, tuple] = {}
        for rec in records.values():
            for path, (etype, *_rest) in km.ENTITY_FIELDS.items():
                group, name = path.split(".", 1)
                v = (rec.get(group) or {}).get(name)
                if v:
                    names[entity_id(etype, v["raw_value"])] = (etype.value, v["raw_value"])
            for o in rec["outcomes"]:
                names[entity_id(E.METRIC, o["metric"]["raw_value"])] = ("METRIC", o["metric"]["raw_value"])
        self.entities = names

    @staticmethod
    def slim(r):
        return {"model": {"model_name": r["model"].get("model_name"), "base_model": r["model"].get("base_model")},
                "method": {"method_name": (r.get("method") or {}).get("method_name")}, "evaluation": {"benchmark": r["evaluation"].get("benchmark")},
                "dataset": {"dataset_name": (r.get("dataset") or {}).get("dataset_name")}, "research": {"task": r["research"].get("task")},
                "outcomes": [{"metric": o.get("metric"), "score": o.get("score"), "dataset_split": o.get("dataset_split")} for o in r["outcomes"]]}

    def __call__(self, sql, params=()):
        s = " ".join(sql.split())
        self.sql.append(s)
        if s.startswith("select id, title, publication_year, study_id from rx.papers"):
            return [(p["id"], p["title"], p["year"], p.get("study_id")) for p in self.papers]
        if s.startswith("select id, entity_type, preferred_name from rx.canonical_entities"):
            return [(i, t, n) for i, (t, n) in self.entities.items() if t in params[0]]
        if s.startswith("select id, paper_id, depth_tier, jsonb_build_object("):
            return [(x, r["paper_id"], "T2", self.slim(r)) for x, r in self.records.items()]
        if "from rx.citations" in s:
            return [(a, b, f"src:{a}") for a, b in self.citations]
        if "from rx.study_links" in s:
            return self.links
        if s.startswith("select id, subject_id, field, record from rx.contradictions"):
            return self.contradictions
        if "from rx.load_checks" in s:
            return [({},)]
        if s.startswith("select id, paper_id, study_id, experiment_type, split, record, provenance_id from rx.experiments"):
            return [(x, self.records[x]["paper_id"], self.records[x]["study_id"], "OTHER", "UNASSIGNED", self.records[x], None)
                    for x in params[0] if x in self.records]
        if s.startswith("select id, record from rx.experiments where id = any"):
            return [(x, self.records[x]) for x in params[0] if x in self.records]
        if s.startswith("select id, paper_id, field_name, record from rx.evidence where paper_id = any"):
            return []  # paper-level statements (GraphIndex._attach_paper_level): none in this corpus
        if s.startswith("select id, field_name, location_kind"):
            return [(i, self.evidence[i]["field"], "PAPER_TABLE_CELL", "loc", self.evidence[i]["text"], True) for i in params[0] if i in self.evidence]
        if s.startswith("select id, field_name, record from rx.evidence where id = any"):  # stage classification (services/analysis/configs.py)
            return [(i, self.evidence[i]["field"], {"evidence_text": self.evidence[i]["text"], "section": self.evidence[i].get("section")})
                    for i in params[0] if i in self.evidence]
        raise AssertionError(f"unexpected statement: {s[:120]}")


def corpus(names=None):
    """Two papers, five experiments on one benchmark with two models, one recorded score contradiction."""
    n = {"model_a": "Zeta-9", "model_b": "Orion-3", "bench": "Kappa-Bench", "task": "Glyph Sorting", "metric": "Q-Score", **(names or {})}
    papers = [{"id": "paper:arxiv:9901.00001", "title": f"{n['model_a']} for {n['task']}", "year": 2021},
              {"id": "paper:arxiv:9901.00002", "title": f"{n['model_b']} revisited", "year": 2023}]
    recs = {f"exp:a{i}": exp_record(f"exp:a{i}", papers[0]["id"], n["model_a"], n["bench"], n["task"], n["metric"], 70.0 + i,
                                    evaluation__dataset_split="test", evaluation__metric_definition="top-1",
                                    evaluation__evaluation_protocol="single crop") for i in range(3)}
    recs |= {f"exp:b{i}": exp_record(f"exp:b{i}", papers[1]["id"], n["model_b"], n["bench"], n["task"], n["metric"], 60.0 + i,
                                     training__learning_rate=0.01) for i in range(2)}
    contradictions = [("con:1", "exp:b0", "evaluation.score", {"claim": "score", "explanation": "two sources disagree", "value_a": 60, "value_b": 61,
                                                               "source_a": "paper", "source_b": "leaderboard", "evidence_a": ["ev:exp:b0os"],
                                                               "evidence_b": ["ev:other"]})]
    evidence = {"ev:exp:b0os": {"field": "evaluation.score", "text": "60.0"}, "ev:other": {"field": "evaluation.score", "text": "61.0"}}
    return FakeCorpus(papers, recs, citations=[(papers[1]["id"], papers[0]["id"])], contradictions=contradictions, evidence=evidence), n


# ------------------------------------------------------------------------------------------------ canonical graph
def test_canonical_nodes_use_the_pipeline_entity_ids_and_carry_provenance():
    q, n = corpus()
    g = km.GraphIndex(q)
    model = entity_id(E.MODEL, n["model_a"])
    assert g.node(model)["type"] == "Model" and g.node(model)["label"] == n["model_a"]
    nb = g.neighborhood("exp:a0")
    assert {"REPORTS", "USES_MODEL", "USES_DATASET", "CONTAINS", "PRODUCES_RESULT"} <= {e["type"] for e in nb["edges"]}
    uses = next(e for e in nb["edges"] if e["type"] == "USES_MODEL")
    assert uses["target"] == model and uses["evidence_ids"] == ["ev:exp:a0m"] and uses["derivation"] == "rx.experiments.record.model.model_name"
    assert all(e["why"] and e["derivation"] for e in nb["edges"])


def test_experiment_links_come_only_from_the_comparison_engine():
    q, _ = corpus()
    g = km.GraphIndex(q)
    pairs = [e for e in g.neighborhood("exp:a0")["edges"] if e["source"].startswith("exp:") and e["target"].startswith("exp:")]
    assert pairs, "experiments on the same benchmark are compared"
    for e in pairs:
        other = e["target"] if e["source"] == "exp:a0" else e["source"]
        label = compare(*g._pair(*sorted(("exp:a0", other))))["label"].value
        assert e["derivation"].startswith("comparison engine") and e["label"] == label.replace("_", " ").lower()
    models = g.experiments(["exp:a0", "exp:a1", "exp:b0"])
    assert compare(models["exp:a0"], models["exp:a1"])["label"].value == "ALIGNED_EXPERIMENT"  # all equivalence fields known
    assert compare(models["exp:a0"], models["exp:b0"])["label"].value != "ALIGNED_EXPERIMENT"  # an unknown field is never a match


def test_title_similarity_alone_never_draws_a_study_link():
    q, _ = corpus()
    a, b = q.papers[0]["id"], q.papers[1]["id"]
    q.links = [("sl:1", a, b, "RELATED_BUT_DISTINCT", {"rationale": "similar titles", "evidence": [{"kind": "title_similarity", "detail": "jaccard 0.8"}]}),
               ("sl:2", a, b, "UNCERTAIN", {"rationale": "possible follow-up", "evidence": [{"kind": "title_similarity", "detail": "x"},
                                                                                            {"kind": "citation", "detail": "one cites the other"}]})]
    g = km.GraphIndex(q)
    assert g.links_similarity_only == 1
    drawn = [e for e in g.neighborhood(a)["edges"] if e["derivation"].startswith("rx.study_links")]
    assert len(drawn) == 1 and drawn[0]["strength"] == "POSSIBLE" and "sl:2" in drawn[0]["derivation"]


def test_neighbourhoods_are_paged_and_never_load_the_corpus():
    q, n = corpus()
    for i in range(95):
        q.records[f"exp:z{i}"] = exp_record(f"exp:z{i}", q.papers[1]["id"], n["model_b"], n["bench"], n["task"], n["metric"], float(i))
    g = km.GraphIndex(q)
    bench = entity_id(E.BENCHMARK, n["bench"])
    before = len(q.sql)
    first = g.neighborhood(bench, limit=40)
    assert first["total"] == 100 and first["has_more"] and len(first["nodes"]) == 41
    assert {x["label"].split(" · ")[0] for x in first["nodes"][1:]} == {n["model_a"], n["model_b"]}, "one page shows the variety, not 40 variants"
    last = g.neighborhood(bench, limit=40, offset=80)
    assert len(last["nodes"]) == 21 and not last["has_more"]
    assert not any("from rx.experiments" in s and "any(" not in s for s in q.sql[before:]), "a page never re-reads the experiment table"


def test_mutation_the_graph_follows_the_data_not_the_code():
    renamed = {"model_a": "Quill-77", "model_b": "Basalt-2", "bench": "Rho-Set", "task": "Shard Ranking", "metric": "Lambda@3"}
    q1, n1 = corpus()
    q2, n2 = corpus(renamed)
    g1, g2 = km.GraphIndex(q1), km.GraphIndex(q2)
    assert g1.search(n1["bench"])[0]["label"] == n1["bench"] and not g1.search(n2["bench"])
    assert g2.search(n2["bench"])[0]["label"] == n2["bench"] and not g2.search(n1["bench"])
    ask = g2.ask(entity_id(E.BENCHMARK, n2["bench"]), "models")
    assert {i["label"] for i in ask["items"]} == {n2["model_a"], n2["model_b"]}
    assert g1.stats()["experiments"] == g2.stats()["experiments"] == 5


def test_ask_answers_come_from_records_and_say_when_nothing_is_recorded():
    q, _ = corpus()
    g = km.GraphIndex(q)
    assert {i["id"] for i in g.ask("exp:a0", "same_dataset")["items"]} == {"exp:b0", "exp:b1"}
    con = g.ask("exp:b0", "contradictions")
    assert len(con["contradictions"]) == 1 and con["contradictions"][0]["value_a"] == 60
    assert "No replication" in g.ask("exp:a0", "replications")["answer"]
    with pytest.raises(KeyError):
        g.ask("exp:a0", "models")  # not a question for an experiment


def test_landscape_counts_are_experiment_records():
    q, n = corpus()
    ls = km.GraphIndex(q).landscape(entity_id(E.BENCHMARK, n["bench"]))
    assert ls["experiments"] == 5 and ls["papers"] == 2 and ls["contradictions"] == 1
    assert {i["label"]: i["count"] for i in ls["groups"]["Model"]} == {n["model_a"]: 3, n["model_b"]: 2}


def test_graph_modules_hardcode_no_corpus_content_and_read_no_user_table():
    allowed = {"papers", "experiments", "evidence", "citations", "study_links", "contradictions", "canonical_entities", "load_checks", "provenance"}
    for path in ("services/graph/knowledge_map.py", "services/graph/runtime.py"):
        src = (ROOT / path).read_text(encoding="utf-8")
        tables = set(re.findall(r"(?:from|into|update|join)\s+rx\.(\w+)", src))
        assert tables <= allowed, f"{path} reads {tables - allowed}"
        assert not re.search(r"\b\d{4}\.\d{4,5}\b", src), f"{path} contains a paper id"
        assert not re.search(r"\b(?:exp|ev|ent):[0-9a-f]{16}\b", src), f"{path} contains a record id"


# ------------------------------------------------------------------------------------------------ runtime graph
RUN = "rx_01TESTRUN0000000000000000"
EXT = {"sid": "S1", "kind": "external", "chunk_id": "https://doi.org/10.9999/zeta", "provider": "openalex", "url": "https://doi.org/10.9999/zeta",
       "title": "Zeta-9 on Kappa-Bench, again", "doi": "10.9999/zeta", "arxiv_id": None, "year": 2024, "retrieved_at": "2026-10-08T05:00:00+00:00",
       "content_sha256": "ab" * 32, "reliability": "scholarly", "source_id": "https://openalex.org/W1", "also_found_by": [], "role": "SUPPORTS",
       "text": "We train Zeta-9 on Kappa-Bench with a learning rate of 0.02 and report a Q-Score of 81.5."}
UNUSED = {"provider": "tavily", "url": "https://blog.example.org/post", "title": "A blog post", "text": "x" * 60, "reliability": "web",
          "content_sha256": "cd" * 32, "retrieved_at": "2026-10-08T05:00:00+00:00", "also_found_by": []}


def fact(field, value, span, **kw):
    return {"field": field, "label": field.split(".")[-1].replace("_", " "), "value": value, "normalized": kw.pop("normalized", None), "span": span,
            "chunk_id": EXT["chunk_id"], "sid": "S1", "evidence_id": None, "experiment_ids": [], "origin": "runtime extraction", "config": "EXACT",
            "config_detail": "", "dataset_ok": True, "start": None, "end": None, "locator": None, "metric": kw.pop("metric", None), **kw}


def run_inputs(**over):
    base = {"query": "What learning rate was used to train Zeta-9 on Kappa-Bench?", "intent": "protocol", "asked_unknown": True,
            "request": {"model": "Zeta-9", "dataset": "Kappa-Bench", "metric": None}, "external": [EXT, UNUSED],
            "facts": [fact("training.learning_rate", "0.02", "a learning rate of 0.02", normalized=0.02)], "rejected": [], "conflicts": [], "result_ids": {}}
    return base | over


def ident_q(rows=()):
    log = []

    def q(sql, params=()):
        log.append(" ".join(sql.split()))
        return list(rows)
    q.log = log
    return q


def build(result_over=None, **inputs):
    result = {"status": "ANSWERED", "sources": [copy.deepcopy(EXT)], "cited_sources": ["S1"], "claims": [], "contradictions": [], **(result_over or {})}
    q = ident_q()
    return rg.build(RUN, result, run_inputs(**inputs), q), q


def test_runtime_api_paper_enters_the_current_graph_with_provenance():
    g, q = build()
    nodes = {n["id"]: n for n in g["nodes"]}
    paper = nodes["rt:paper:doi:10.9999/zeta"]
    assert paper["origin"] == "runtime_external" and paper["status"] == "validated"
    assert {k: paper["provenance"][k] for k in ("provider", "url", "doi", "retrieved_at", "content_sha256")} == {
        "provider": "openalex", "url": EXT["url"], "doi": "10.9999/zeta", "retrieved_at": EXT["retrieved_at"], "content_sha256": EXT["content_sha256"]}
    exp = next(n for n in g["nodes"] if n["type"] == "Experiment")
    edges = {(e["type"], e["source"], e["target"]): e for e in g["edges"]}
    assert ("REPORTS", paper["id"], exp["id"]) in edges
    uses = edges[("USES_MODEL", exp["id"], entity_id(E.MODEL, "Zeta-9"))]
    assert uses["status"] == "validated" and uses["evidence_ids"], "a model link carries the validated span"
    assert ("USES_DATASET", exp["id"], entity_id(E.BENCHMARK, "Kappa-Bench")) in edges
    assert all(s.startswith("select") for s in q.log), "building a run graph never writes"


def test_runtime_knowledge_is_never_canonical_and_is_scoped_to_its_run():
    g, _ = build()
    assert g["research_run_id"] == RUN and all(e["provenance"]["research_run_id"] == RUN for e in g["edges"])
    runtime = [n for n in g["nodes"] if n["id"].startswith("rt:")]
    assert runtime and all(n["origin"] != "canonical" for n in runtime)
    model = next(n for n in g["nodes"] if n["id"] == entity_id(E.MODEL, "Zeta-9"))
    assert model["origin"] == "user_input" and model["status"] == "unverified"  # named by the question, unknown to the corpus


def test_identity_merges_on_doi_and_never_on_title_alone(monkeypatch):
    result = {"status": "ANSWERED", "sources": [copy.deepcopy(EXT)], "cited_sources": ["S1"], "claims": [], "contradictions": []}
    merged = rg.build(RUN, result, run_inputs(), ident_q([("doi", "10.9999/zeta", "paper:arxiv:9901.00001", "Zeta-9 for Glyph Sorting")]))
    ids = {n["id"] for n in merged["nodes"]}
    assert "paper:arxiv:9901.00001" in ids and "rt:paper:doi:10.9999/zeta" not in ids and merged["stats"]["merged_by_identifier"] == 1
    corpus_paper = next(n for n in merged["nodes"] if n["id"] == "paper:arxiv:9901.00001")
    assert corpus_paper["origin"] == "canonical" and corpus_paper["provenance"]["runtime"]["provider"] == "openalex"

    monkeypatch.setattr(rg, "_TITLES", {rg.normalize_title(EXT["title"]): ("paper:arxiv:9901.00002", EXT["title"])})
    unsure = rg.build(RUN, result, run_inputs(), ident_q())
    link = next(e for e in unsure["edges"] if e["type"] == "RELATED_TO" and e["target"] == "paper:arxiv:9901.00002")
    assert link["strength"] == "POSSIBLE" and link["source"] == "rt:paper:doi:10.9999/zeta"


def test_a_retrieved_title_without_validated_facts_gets_no_research_relationship():
    g, _ = build(facts=[], result_over={"sources": [], "cited_sources": []}, external=[EXT, UNUSED])
    assert [e for e in g["edges"] if e["type"] not in rg.PROVENANCE_EDGES] == [], "retrieval alone creates no research relationship"
    assert {n["status"] for n in g["nodes"] if n["origin"] == "runtime_external"} == {"unverified"}


def test_dataset_link_requires_dataset_evidence_and_rejected_facts_stay_off_the_graph():
    g, _ = build(facts=[fact("training.learning_rate", "0.02", "a learning rate of 0.02", dataset_ok=None)],
                 rejected=[fact("training.batch_size", "32", "batch size 32", reason="different dataset")])
    assert not [e for e in g["edges"] if e["type"] == "USES_DATASET"]
    assert g["rejected"] and g["rejected"][0]["status"] == "rejected"
    assert not any(str(n["meta"].get("value")) == "32" for n in g["nodes"])


def test_disagreeing_validated_values_are_linked_as_a_contradiction():
    facts = [fact("training.learning_rate", "0.02", "a learning rate of 0.02"), fact("training.learning_rate", "0.01", "lr 0.01 for the large model")]
    g, _ = build(facts=facts, conflicts=[{"field": "training.learning_rate", "label": "learning rate", "facts": [0, 1]}])
    con = [e for e in g["edges"] if e["type"] == "CONTRADICTS"]
    assert len(con) == 1 and con[0]["strength"] == "CONTRADICTORY"
    assert {n["id"] for n in g["nodes"] if "conflict" in n["flags"]} == {con[0]["source"], con[0]["target"]}


def test_private_documents_stay_private_and_are_never_proposed():
    priv = {"sid": "S2", "kind": "private", "chunk_id": "private:7", "document_id": "11111111-2222-3333-4444-555555555555", "title": "My notes",
            "text": "notes", "role": "SUPPORTS"}
    g, _ = build(result_over={"sources": [copy.deepcopy(EXT), priv], "cited_sources": ["S1", "S2"]})
    doc = next(n for n in g["nodes"] if n["type"] == "Document")
    assert doc["origin"] == "private_document" and "private" in doc["provenance"]["scope"]
    assert rg.promotion_checks(g, doc["id"])["eligible"] is False
    assert rg.promotion_checks(g, "rt:paper:doi:10.9999/zeta")["eligible"] is True


def test_the_same_run_builds_the_same_graph():
    a, _ = build()
    b, _ = build()
    a["stats"].pop("build_ms")
    b["stats"].pop("build_ms")
    assert a == b


def test_a_runtime_experiment_can_be_diffed_against_corpus_experiments():
    g, _ = build(facts=[fact("training.learning_rate", "0.02", "a learning rate of 0.02", normalized=0.02),
                        fact("evaluation.score", "81.5", "a Q-Score of 81.5", normalized=81.5, metric="Q-Score")],
                 request={"model": "Zeta-9", "dataset": "Kappa-Bench", "metric": "Q-Score"})
    xid = next(n["id"] for n in g["nodes"] if n["type"] == "Experiment")
    runtime = rg.runtime_experiment(g, xid)
    assert runtime.model.model_name.value == "Zeta-9" and runtime.training.learning_rate.value == 0.02
    q, _ = corpus()
    idx = km.GraphIndex(q)
    diff = idx.diff(xid, "exp:b0", {xid: (runtime, next(n for n in g["nodes"] if n["id"] == xid))})
    lr = next(r for r in diff["fields"] if r["field"] == "training.learning_rate")
    assert (lr["value_a"], lr["value_b"], lr["status"]) == (0.02, 0.01, "DIFFERENT")
    closest = rg.ask_node(g, xid, "closest", idx)
    assert {i["id"] for i in closest["items"]} == set(q.records)  # every corpus experiment on the same benchmark, each labelled


def test_ask_the_run_lists_its_discoveries_and_review_states_overlay():
    g, _ = build()
    found = rg.ask_run(g, "discoveries")
    assert "rt:paper:doi:10.9999/zeta" in {n["id"] for n in found["items"]} and "outside the corpus" in found["answer"]
    rg.apply_review_states(g, {(EXT["url"], EXT["content_sha256"]): "PROMOTED"})
    assert next(n for n in g["nodes"] if n["id"] == "rt:paper:doi:10.9999/zeta")["status"] == "promoted"


def test_passage_claims_become_evidence_quoted_from_their_source():
    claim = {"text": "Zeta-9 reaches 81.5 Q-Score.", "sources": ["S1"], "quote": "report a Q-Score of 81.5"}
    g, _ = build(facts=[], result_over={"claims": [claim]})
    ev = next(n for n in g["nodes"] if n["type"] == "Evidence")
    edges = {(e["type"], e["source"], e["target"]) for e in g["edges"]}
    assert ("QUOTED_FROM", ev["id"], "rt:paper:doi:10.9999/zeta") in edges and ("SUPPORTED_BY", f"run:{RUN}", ev["id"]) in edges


# ------------------------------------------------------------------------------------------------ orchestrator: providers
class FakeTracer:
    research_run_id = RUN

    def __init__(self):
        self.rows, self.persisted = [], []

    @contextmanager
    def stage(self, name, **detail):
        st = Stage(name)
        st.detail.update(detail)
        try:
            yield st
        finally:
            self.rows.append(st)

    def record(self, name, status, latency_ms=0, error=None, **detail):
        st = Stage(name)
        st.status, st.detail, st.error = status, detail, error
        self.rows.append(st)

    def persist(self, sql, params):
        self.persisted.append((sql, params))

    def row(self, name):
        return next(r for r in self.rows if r.name == name)


def provider(rows=None, error=None, sleep=0.0):
    def search(q, limit, timeout):
        time.sleep(sleep)
        if error:
            raise RuntimeError(error)
        return [dict(r) for r in rows or []], {"calls": 1, "label": "not a number"}
    return search


def orchestrator(providers, **external):
    cfg = copy.deepcopy(config("runtime"))
    cfg["external"].update({"providers": {k: {"enabled": True, "limit": 5} for k in providers}, "timeout_s": 2, "deadline_s": 1.5, "grace_s": 0.3,
                            "routing": {}, **external})  # no routing table: every fake provider is asked (routing tests pass their own)
    fake = SimpleNamespace(PROVIDERS={k: (fn, None, "terms") for k, fn in providers.items()}, validate_sources=tools.validate_sources)
    o = Orchestrator(cfg=cfg, gw=SimpleNamespace(), connect_fn=lambda **_: None, search_tools=fake)
    o.q = lambda sql, params=(): []  # the shared runtime-source metadata insert
    return o


def row(i, provider_name="p"):
    return {"provider": provider_name, "url": f"https://doi.org/10.1/{i}", "title": f"T{i}", "text": "an abstract that is long enough to pass the gate",
            "doi": f"10.1/{i}"}


def test_external_failures_and_slow_providers_never_break_the_run():
    o = orchestrator({"good": provider([row(1, "good")]), "broken": provider(error="HTTP 429"), "slow": provider([row(2, "slow")], sleep=3)})
    tr, usage = FakeTracer(), Counter()
    t = time.perf_counter()
    accepted, _ = o._external(tr, "q", usage)
    assert time.perf_counter() - t < 1.4, "the grace period cuts a straggler off"
    assert [a["url"] for a in accepted] == ["https://doi.org/10.1/1"]
    p = tr.row("external_sources").detail["providers"]
    assert (p["good"]["status"], p["broken"]["status"], p["slow"]["status"]) == ("completed", "failed", "cut_off")
    assert usage == Counter(calls=1), "only numeric usage is counted"


def test_external_search_can_be_disabled_and_is_traced_as_skipped():
    o = orchestrator({"good": provider([row(1)])}, enabled=False)
    tr = FakeTracer()
    assert o._external(tr, "q", Counter()) == ([], [])
    assert {r.name: r.status for r in tr.rows} == {"external_sources": "skipped", "tavily": "skipped", "source_validation": "skipped"}


def test_duplicate_works_from_two_providers_are_one_source():
    o = orchestrator({"a": provider([row(1, "a")]), "b": provider([{**row(1, "b"), "url": "https://other.example.org/x"}])})
    accepted, rejected = o._external(FakeTracer(), "q", Counter())
    assert len(accepted) == 1 and accepted[0]["also_found_by"][0]["provider"] == "b"
    assert [r["reason"] for r in rejected] == ["duplicate_identifier"]


def test_provider_cache_is_per_user_and_unused_without_a_user():
    o = orchestrator({"good": provider([row(1)])})
    u1, u2 = SimpleNamespace(user=SimpleNamespace(id="u1")), SimpleNamespace(user=SimpleNamespace(id="u2"))
    assert o._external_key(u1, "good", "q", 5) != o._external_key(u2, "good", "q", 5)
    assert o._external_key(u1, "good", "q", 5) == o._external_key(u1, "good", "q", 5)
    o.uq = lambda *a, **k: pytest.fail("a run without a user must never touch the cache")
    tr = FakeTracer()
    o._external(tr, "q", Counter(), SimpleNamespace(user=None))
    assert not tr.persisted


def test_graph_failure_never_changes_a_valid_answer(monkeypatch):
    o = orchestrator({})

    def boom(*_a, **_k):
        raise RuntimeError("boom")
    monkeypatch.setattr(rg, "build", boom)
    result, tr = {"status": "ANSWERED", "answer": "x [S1]", "sources": []}, FakeTracer()
    o._graph(tr, result, {})
    assert "graph" not in result and result["answer"] == "x [S1]"
    st = tr.row("graph_enrichment")
    assert st.status == "failed" and "boom" in st.error


def test_graph_enrichment_is_traced_with_its_counts():
    o = orchestrator({})
    o.q = ident_q()
    result, tr = {"status": "ANSWERED", "sources": [copy.deepcopy(EXT)], "cited_sources": ["S1"], "claims": [], "contradictions": []}, FakeTracer()
    o._graph(tr, result, run_inputs())
    st = tr.row("graph_enrichment")
    assert st.status == "completed" and st.detail["nodes"] == len(result["graph"]["nodes"]) and "build_ms" in st.detail


def test_unused_candidate_values_say_why_and_where():
    from services.evidence import grounding as G

    cands = {"p#1": {"title": "Paper A", "paper_id": "paper:arxiv:9901.00001", "section": "Training", "arxiv_id": "9901.00001", "unit_id": "S4",
                     "unit_type": "paragraph"}}
    facts = [G.Fact(field="training.learning_rate", raw="0.1", span="lr 0.1", chunk_id="p#1", valid=False, reason="no_field_cue"),
             G.Fact(field="training.learning_rate", raw="0.2", span="learning rate 0.2", chunk_id="p#1", valid=True, config="RELATED", dataset_ok=True),
             G.Fact(field="training.learning_rate", raw="0.3", span="learning rate 0.3", chunk_id="p#1", valid=True, config="EXACT", dataset_ok=False),
             G.Fact(field="training.learning_rate", raw="0.4", span="learning rate 0.4", chunk_id="p#1", valid=True, config="EXACT", dataset_ok=True)]
    out = Orchestrator._not_used(facts, cands, {"training.learning_rate": "learning rate"})
    assert [(r["value"], r["reason"]) for r in out] == [("0.3", "different_dataset"), ("0.2", "configuration_related"), ("0.1", "no_field_cue")]
    assert out[0]["label"] == "learning rate" and out[0]["source"]["url"] == "https://arxiv.org/html/9901.00001#S4"


def test_scholarly_search_terms_are_keywords_of_the_question():
    req = SimpleNamespace(fields=["training.learning_rate"], model="Zeta-9", dataset="Kappa-Bench", metric=None)
    assert search_queries("What learning rate was used to train Zeta-9 on Kappa-Bench?", req)[0] == "Zeta-9 Kappa-Bench"
    free = SimpleNamespace(fields=[], model=None, dataset=None, metric=None)
    assert search_queries("Which papers evaluate vision transformers on dense prediction?", free)[0] == "vision transformers dense prediction"
