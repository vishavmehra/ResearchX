"""End-to-end evidence-grounding regressions through the real pipeline (corpus in Supabase, BM25 artifact, embeddings,
reranker). Public scope only: no user, no answer cache, so every run executes every stage.

Opt-in (network + database + model calls):   RX_INTEGRATION=1 uv run pytest -q tests/test_grounding_integration.py
The insufficient and no-source cases trigger the external evidence-gap search (one Tavily credit each when configured).
"""

import os

import pytest

from services.env import load_env

load_env()
pytestmark = pytest.mark.skipif(os.environ.get("RX_INTEGRATION") != "1", reason="set RX_INTEGRATION=1 to run the live pipeline")


@pytest.fixture(scope="module")
def run():
    from services.retrieval.scope import RetrievalScope
    from services.runtime.orchestrator import Orchestrator

    o = Orchestrator()
    return lambda q: o.run(q, RetrievalScope(None))


def stage(result, name):
    return next(s for s in result["stages"] if s["stage"] == name)


def test_exact_parameters_answered_from_s1_with_spans(run):
    r = run("What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?")
    assert (r["status"], r["evidence_status"], r["mode"]) == ("ANSWERED", "SUFFICIENT", "pipeline")
    assert "- Learning rate: 0.02 [S1]" in r["answer"] and "- Batch size: 16 [S1]" in r["answer"]
    s1 = r["sources"][0]
    assert s1["sid"] == "S1" and s1["role"] == "SUPPORTS" and s1["match"] == "EXACT" and s1["paper_id"] == "paper:arxiv:1909.11065"
    assert {sp["text"] for sp in s1["spans"]} == {"initial learning rate as 0.02", "batch size as 16"}
    assert all(s1["text"][sp["start"]:sp["end"]] == sp["text"] for sp in s1["spans"])  # exact offsets into the cited passage
    assert all(s["role"] == "RELATED" for s in r["sources"][1:])  # other passages are context, never support
    assert r["cited_sources"] == ["S1"] and all(c["sources"] == ["S1"] and c["verified"] for c in r["claims"])
    assert stage(r, "evidence_assessment")["detail"]["status"] == "SUFFICIENT"
    assert stage(r, "citation_validation")["detail"]["verified"] == 2 and stage(r, "citation_validation")["detail"]["offsets_verified"] == 2
    assert stage(r, "experiment_extraction")["detail"]["papers"]["paper:arxiv:1909.11065"] == "EXACT"
    assert r["trace_persisted"]


def test_parameter_the_corpus_does_not_state_abstains(run):
    r = run("What dropout rate was used to train HRNetV2 + OCR on ADE20K?")
    assert (r["status"], r["evidence_status"]) == ("ABSTAINED", "INSUFFICIENT_EVIDENCE")
    assert r["answer"].startswith("Insufficient evidence found.") and not r["claims"]
    assert not any(s["role"] == "SUPPORTS" for s in r["sources"])


def test_unknown_model_and_dataset_find_no_relevant_source(run):
    r = run("What learning rate was used to train QuasarFlux-9 on the Zentrix-77 benchmark?")
    assert (r["status"], r["evidence_status"]) == ("ABSTAINED", "NO_RELEVANT_SOURCE_FOUND")
    assert r["request"] == {"model": "QuasarFlux-9", "dataset": "Zentrix-77", "fields": ["learning rate"]}


def test_paper_table_and_leaderboard_disagreement_is_reported(run):
    r = run("What smoothed BLEU-4 score does RoBERTa achieve on CodeSearchNet JavaScript code documentation generation?")
    assert (r["status"], r["evidence_status"]) == ("CONFLICTING", "CONFLICTING_EVIDENCE")
    values = {s["supports"][0]["value"]: s for s in r["sources"] if s["role"] == "SUPPORTS"}
    assert set(values) == {"11.90", "5.72"}
    assert values["11.90"]["kind"] == "internal" and values["11.90"]["unit_type"] == "table"
    assert values["5.72"]["kind"] == "record" and values["5.72"]["provider"] == "Papers With Code archive"
    assert any(c["id"] == "con:3743469c07e972db" for c in r["contradictions"])  # the corpus's recorded contradiction
