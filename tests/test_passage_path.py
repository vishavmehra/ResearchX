"""Passage path (no field-level request) end to end through Orchestrator._pipeline with offline fakes: retrieval, rerank,
external search and synthesis are stubbed at their seams; the decisions between them are real (no network)."""

import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from services.datasets.store import config
from services.runtime import orchestrator as orch
from services.runtime import tools
from tests.test_knowledge_graph import FakeTracer

QUESTION = "Compare Zeta-9 and Omega-3 on Kappa-Bench"


class PipeTracer(FakeTracer):
    def retrieval(self, *_a, **_k):
        pass

    def rows_named(self, name):
        return [r for r in self.rows if r.name == name]


def chunk(cid, text):
    return {"chunk_id": cid, "paper_id": f"paper:{cid}", "unit_type": "paragraph", "unit_id": "S1.p1", "section": "Results", "text": text,
            "title": f"Paper {cid}", "arxiv_id": None, "year": 2026, "redistribution_allowed": True, "context": text}


CANDS = {"c1": chunk("c1", "Zeta-9 is trained on Kappa-Bench with a cosine schedule."),
         "c2": chunk("c2", "Omega-3 uses Kappa-Bench as one of four evaluation suites.")}
EXTERNAL = {"provider": "semantic_scholar", "url": "https://arxiv.org/abs/9903.00001", "title": "Zeta-9 versus Omega-3", "reliability": "scholarly",
            "text": "On Kappa-Bench, Zeta-9 reaches 71.2 while Omega-3 reaches 69.8.", "content_sha256": "x", "retrieved_at": "2026-10-10T00:00:00Z"}


def pipeline(monkeypatch, synth_results, external=(EXTERNAL,), enabled=True, grades=None):
    """An orchestrator whose stages are stubbed at their seams; returns (orchestrator, calls)."""
    cfg = copy.deepcopy(config("runtime"))
    cfg["external"]["enabled"] = enabled
    calls = {"external": [], "synth_sources": []}
    gw = SimpleNamespace(embed=lambda *a, **k: None, resolve_model=lambda task: (None, f"fake-{task}"))
    o = orch.Orchestrator(cfg=cfg, gw=gw, connect_fn=lambda **_: None, search_tools=SimpleNamespace(PROVIDERS={}, validate_sources=tools.validate_sources))
    bm = SimpleNamespace(search=lambda q, k: [("c1", 5.0), ("c2", 4.0)], k1=1.2, b=0.75)
    o.bm25 = lambda: bm
    o.q = lambda sql, params=(): [("c1", 0.1), ("c2", 0.2)] if "chunk_embeddings" in sql else []
    o._entities = lambda normalized: []
    o._query_embedding = lambda *a: ([[1.0, 0.0, 0.0]], SimpleNamespace(input_tokens=1, latency_s=0.01, status="ok", error=None), False)
    o._private = lambda *a: ({}, [], [])
    o._candidates = lambda ids: {c: dict(CANDS[c]) for c in ids if c in CANDS}
    o._prefetch = lambda papers, fields: {"records": {}, "links": {}}
    o._evidence_links = lambda head, cands, rows=None: {}
    o._paper_experiments = lambda papers, records=None: ({}, {})
    o._rare_terms = lambda q: []
    o._finish = lambda tr, scope, ckey, routing, synth_task, result, inputs: result
    graded = grades or {"c1": 2, "c2": 2}  # two passages graded 2: sufficient by grades
    monkeypatch.setattr(orch, "rerank", lambda gw, q, items, task="reranking": ([c for c, _ in items], {"ok": True, "grades": graded, "model": "fake-rerank"}))

    def fake_external(tr, query, usage, scope, queries):
        calls["external"].append(queries)
        tr.record("external_sources", "completed", queries=queries)
        return [dict(a) for a in external], []
    o._external = fake_external
    results = iter(synth_results)

    def fake_synthesize(tr, query, sources, contradictions, synth_task, usage, *_routing, **_retry):
        calls["synth_sources"].append([s["sid"] for s in sources])
        tr.record("synthesis", "completed")
        out = next(results)
        return out(sources) if callable(out) else out
    o._synthesize = fake_synthesize
    return o, calls


def cite_external(sources):
    sid = next(s["sid"] for s in sources if s.get("kind") == "external")
    claim = {"text": "Zeta-9 reaches 71.2 and Omega-3 69.8 on Kappa-Bench.", "sources": [sid], "quote": "Zeta-9 reaches 71.2 while Omega-3 reaches 69.8"}
    return f"{claim['text']} [{sid}]", [claim], [], "ANSWERED"


def run(o):
    tr = PipeTracer()
    return o._pipeline(tr, QUESTION, orch.norm(QUESTION), Counter(), {}, SimpleNamespace(user=None)), tr


def test_passages_graded_sufficient_that_do_not_answer_get_a_second_look_outside_the_corpus(monkeypatch):
    o, calls = pipeline(monkeypatch, [("", [], [], "INSUFFICIENT"), cite_external])
    result, tr = run(o)
    assert len(calls["external"]) == 1 and calls["external"][0][0] == "Zeta-9 Omega-3 Kappa-Bench"
    assert calls["synth_sources"] == [["S1", "S2"], ["S1", "S2", "S3"]], "the second synthesis reads the external source too"
    assert result["status"] == "ANSWERED" and result["cited_sources"] == ["S3"]
    gaps = tr.rows_named("evidence_gap")
    assert [g.status for g in gaps] == ["not_required", "completed"] and gaps[1].detail["second_look"] is True
    assert "insufficient" in gaps[1].detail["reason"]


def test_claims_that_all_fail_validation_also_get_a_second_look(monkeypatch):
    o, calls = pipeline(monkeypatch, [("", [], [{"reason": "quote_not_found"}], "ANSWERED"), cite_external])
    result, tr = run(o)
    assert len(calls["external"]) == 1 and result["status"] == "ANSWERED"
    assert "citation validation" in tr.rows_named("evidence_gap")[1].detail["reason"]


@pytest.mark.parametrize("case", ["model unavailable", "external disabled", "nothing found outside"])
def test_no_second_synthesis_without_a_reason_or_without_new_sources(monkeypatch, case):
    extractive = ("quoted passages", [{"text": "t", "sources": ["S1"], "quote": "t"}], [], None)
    first = extractive if case == "model unavailable" else ("", [], [], "INSUFFICIENT")
    o, calls = pipeline(monkeypatch, [first], external=() if case == "nothing found outside" else (EXTERNAL,), enabled=case != "external disabled")
    result, tr = run(o)
    assert len(calls["synth_sources"]) == 1, "one synthesis call only"
    assert len(calls["external"]) == (1 if case == "nothing found outside" else 0)
    if case != "model unavailable":
        assert result["status"] == "ABSTAINED"


def test_a_gap_found_by_grades_searches_once_and_never_twice(monkeypatch):
    o, calls = pipeline(monkeypatch, [("", [], [], "INSUFFICIENT")], grades={"c1": 1, "c2": 0})
    result, tr = run(o)
    assert len(calls["external"]) == 1 and len(calls["synth_sources"]) == 1
    assert [g.status for g in tr.rows_named("evidence_gap")] == ["completed"] and result["status"] == "ABSTAINED"
