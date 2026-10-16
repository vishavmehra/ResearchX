"""Scholarly search query construction and the per-provider query variants of the external phase (brief §4; no network).

Cases from the brief: natural-language, technical, dataset, metric and comparison questions; a provider that finds
nothing (retries with the next variant); a provider that fails. Names are invented so nothing can match a hardcoded string.
"""

import json
import time
from collections import Counter
from types import SimpleNamespace

from services.evidence import grounding as G
from services.runtime.orchestrator import search_queries
from tests.test_knowledge_graph import FakeTracer, orchestrator, provider, row

FREE = G.Request(fields=[])


def by_query(answers: dict, calls: list, fail=(), error="HTTP 500"):
    """A provider that answers per query string, records every query it receives, and raises `error` for the queries in
    `fail` (a list is consumed: each entry fails one call)."""
    def search(q, limit, timeout):
        calls.append((q, time.monotonic()))
        if q in fail:
            if isinstance(fail, list):
                fail.remove(q)
            raise RuntimeError(error)
        return [dict(r) for r in answers.get(q, [])], {"calls": 1}
    return search


# ------------------------------------------------------------------------------------------------ query construction
def test_natural_language_question_keeps_its_content_words_without_question_wording():
    q = "Which papers evaluate vision transformers on dense prediction?"
    assert search_queries(q, FREE) == ["vision transformers dense prediction"]
    ents = [{"type": "TASK", "name": "Dense Prediction"}]
    assert search_queries(q, FREE, ents) == ["vision transformers dense prediction", "dense prediction"], "a multi-word corpus entity is a variant"


def test_technical_question_searches_model_and_dataset_then_the_model_alone():
    req = G.Request(fields=["training.learning_rate"], model="Zeta-9", dataset="Kappa-Bench")
    queries = search_queries("What learning rate was used to train Zeta-9 on Kappa-Bench?", req)
    assert queries == ["Zeta-9 Kappa-Bench", "Zeta-9"]
    assert not any("learning" in q for q in queries), "parameter names are never search terms"
    assert search_queries("?", G.Request(fields=["training.learning_rate"], model="the Zeta-9 model"))[0] == "Zeta-9"


def test_dataset_and_metric_questions():
    assert search_queries("What batch size is used on Kappa-Bench?", G.Request(fields=["training.batch_size"], dataset="Kappa-Bench")) == ["Kappa-Bench"]
    score = G.Request(fields=["evaluation.score"], dataset="Kappa-Bench", metric="top-1 accuracy")
    assert search_queries("Which top-1 accuracy is reported on Kappa-Bench?", score) == ["Kappa-Bench top-1 accuracy"], "no model: the metric narrows"
    score.model = "Zeta-9"
    assert search_queries("What top-1 accuracy does Zeta-9 reach on Kappa-Bench?", score) == ["Zeta-9 Kappa-Bench", "Zeta-9"]


def test_comparison_question_keeps_every_compared_name():
    queries = search_queries("Compare Zeta-9 and Omega-3 on Kappa-Bench top-1 accuracy", FREE)
    assert queries == ["Zeta-9 Omega-3 Kappa-Bench top-1 accuracy", "Zeta-9 Omega-3 Kappa-Bench"], "metric names are left out of the broader query"
    assert all("Zeta-9" in q and "Omega-3" in q for q in queries)
    assert search_queries("Which benchmarks were used to evaluate Zeta-9 against Omega-3?", FREE) == ["Zeta-9 Omega-3"]


def test_noisy_entities_and_empty_keywords():
    assert search_queries("Do zeta networks need warmup?", FREE, [{"type": "MODEL", "name": "zeta"}, {"type": "BENCHMARK", "name": "No"}]) == [
        "zeta networks need warmup"], "one-word corpus names are too noisy to be a variant"
    assert search_queries("What is it?", FREE) == ["What is it?"], "no keywords: the question as asked"


# ------------------------------------------------------------------------------------------------ provider routing by need
def test_providers_are_routed_by_the_missing_evidence_and_the_others_are_skipped_and_traced():
    calls = {n: [] for n in "abc"}
    o = orchestrator({n: by_query({"Zeta-9": [row(i, n)]}, calls[n]) for i, n in enumerate("abc", 1)},
                     routing={"field_values": ["c", "a"], "critique": ["b"]})
    tr = FakeTracer()
    accepted, _ = o._external(tr, "What learning rate was used to train Zeta-9?", Counter(), None, ["Zeta-9"], need="field_values")
    assert calls["a"] and calls["c"] and not calls["b"], "a provider not routed to the need is never asked"
    d = tr.row("external_sources").detail
    assert d["routing"] == {"need": "field_values", "providers": ["c", "a"], "why": "field_values evidence"}
    assert d["providers"]["b"] == {"status": "skipped", "reason": "not routed for field_values evidence"} and len(accepted) == 2


def test_critique_questions_add_the_critique_providers_and_unknown_needs_ask_everyone():
    from services.runtime.orchestrator import route_providers
    x = {"providers": {"a": {}, "b": {}, "c": {}}, "routing": {"passage": ["a"], "critique": ["b"]}}
    assert route_providers("passage", "What are the limitations of Zeta-9?", x)[0] == ["b", "a"]
    assert route_providers("passage", "What does Zeta-9 do?", x)[0] == ["a"]
    assert route_providers("survey", "anything", x) == (["a", "b", "c"], "no routing for survey: every configured provider")


# ------------------------------------------------------------------------------------------------ variants in the external phase
def test_zero_result_provider_retries_with_the_next_variant_and_the_trace_says_so():
    a_calls, b_calls = [], []
    o = orchestrator({"a": by_query({"Zeta-9": [row(1, "a")]}, a_calls), "b": by_query({"Zeta-9 Kappa-Bench": [row(2, "b")]}, b_calls)},
                     retry_min_s=0.1)
    tr = FakeTracer()
    accepted, _ = o._external(tr, "What learning rate was used to train Zeta-9 on Kappa-Bench?", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert [q for q, _ in a_calls] == ["Zeta-9 Kappa-Bench", "Zeta-9"] and [q for q, _ in b_calls] == ["Zeta-9 Kappa-Bench"]
    assert {a["url"] for a in accepted} == {"https://doi.org/10.1/1", "https://doi.org/10.1/2"}
    d = tr.row("external_sources").detail
    assert d["question"].startswith("What learning rate") and d["queries"] == ["Zeta-9 Kappa-Bench", "Zeta-9"]
    assert (d["providers"]["a"]["query"], d["providers"]["a"]["results"]) == ("Zeta-9", 1)
    assert [(t["query"], t["results"]) for t in d["providers"]["a"]["queries_tried"]] == [("Zeta-9 Kappa-Bench", 0), ("Zeta-9", 1)]
    assert "queries_tried" not in d["providers"]["b"]
    assert {s["url"] for s in tr.row("source_validation").detail["selected"]} == {a["url"] for a in accepted}


def test_failed_provider_is_not_retried_and_the_others_still_answer():
    calls = []
    o = orchestrator({"broken": by_query({}, calls, fail={"Zeta-9 Kappa-Bench"}), "good": by_query({"Zeta-9 Kappa-Bench": [row(1, "good")]}, [])},
                     retry_min_s=0.1, rate_limit_backoff_s=0.1, grace_s=1.0)  # a retry would fit: only the error kind prevents it
    tr = FakeTracer()
    accepted, _ = o._external(tr, "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    p = tr.row("external_sources").detail["providers"]
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench"], "an error is not 'nothing found': no retry"
    assert p["broken"]["status"] == "failed" and "500" in p["broken"]["error"] and len(accepted) == 1


def test_rate_limited_provider_tries_the_same_query_once_more_after_a_pause():
    calls = []
    o = orchestrator({"a": by_query({"Zeta-9 Kappa-Bench": [row(1, "a")]}, calls, fail=["Zeta-9 Kappa-Bench"], error="HTTP 429")},
                     retry_min_s=0.1, rate_limit_backoff_s=0.3)
    tr = FakeTracer()
    accepted, _ = o._external(tr, "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    a = tr.row("external_sources").detail["providers"]["a"]
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench"] * 2 and calls[1][1] - calls[0][1] >= 0.3
    assert (a["status"], a["results"], [t["status"] for t in a["queries_tried"]]) == ("completed", 1, ["failed", "completed"]) and len(accepted) == 1

    calls.clear()  # rate limited twice: one retry only, then the failure is reported
    o.tools.PROVIDERS["a"] = (by_query({}, calls, fail={"Zeta-9 Kappa-Bench"}, error="HTTP 429"), None, "terms")
    tr = FakeTracer()
    o._external(tr, "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    a = tr.row("external_sources").detail["providers"]["a"]
    assert len(calls) == 2 and a["status"] == "failed" and len(a["queries_tried"]) == 2


def test_failed_retry_keeps_the_completed_search_and_is_not_cached():
    user = SimpleNamespace(user=SimpleNamespace(id="u1"))
    o = orchestrator({"a": by_query({}, [], fail={"Zeta-9"})}, retry_min_s=0.1)
    o.uq, o._db_mb = (lambda *a, **k: []), (lambda: 0.0)
    tr = FakeTracer()
    o._external(tr, "q", Counter(), user, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    a = tr.row("external_sources").detail["providers"]["a"]
    assert (a["status"], a["results"], a["query"]) == ("completed", 0, "Zeta-9 Kappa-Bench")
    assert a["queries_tried"][1]["status"] == "failed" and "500" in a["queries_tried"][1]["error"]
    assert not tr.persisted, "a search whose retry failed is not conclusive: never cached"


def test_a_retry_the_shrunken_window_cannot_fit_is_dropped_at_once_and_never_sent():
    calls = []
    o = orchestrator({"a": by_query({"Zeta-9": [row(1, "a")]}, calls), "b": provider([row(2, "b")], sleep=0.2)}, retry_min_s=0.1)
    ready = time.monotonic() + 0.8  # fits the deadline when scheduled; b's answer at ~0.2 s shrinks the window to ~0.5 s
    o.tools.ready_at = lambda name: ready if name == "a" else 0.0
    tr = FakeTracer()
    t = time.monotonic()
    o._external(tr, "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert time.monotonic() - t < 0.45, "a retry that can no longer finish never holds the phase open"
    time.sleep(max(0.0, ready - time.monotonic()) + 0.3)
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench"], "the scheduled retry never reached the provider"
    a = tr.row("external_sources").detail["providers"]["a"]
    assert (a["status"], a["results"], a["queries_tried"][1]["status"]) == ("completed", 0, "not_sent")


def test_retry_waits_for_the_provider_rate_limit_and_is_skipped_without_time():
    calls = []
    o = orchestrator({"a": by_query({"Zeta-9": [row(1, "a")]}, calls)}, retry_min_s=0.1)
    ready = time.monotonic() + 0.4
    o.tools.ready_at = lambda name: ready if name == "a" else 0.0
    tr = FakeTracer()
    o._external(tr, "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench", "Zeta-9"] and calls[1][1] >= ready
    assert tr.row("external_sources").detail["providers"]["a"]["latency_ms"] < 300, "the rate-limit wait is not provider latency"

    calls.clear()
    o.tools.ready_at = lambda name: time.monotonic() + 60  # the phase closes long before the provider accepts a call
    o._external(FakeTracer(), "q", Counter(), None, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench"]


def test_cache_is_keyed_on_every_variant_and_replays_the_query_used():
    user = SimpleNamespace(user=SimpleNamespace(id="u1"))
    calls = []
    o = orchestrator({"a": by_query({"Zeta-9 Kappa-Bench": [row(1, "a")]}, calls)})
    o._db_mb = lambda: 0.0
    o.uq = lambda *a, **k: []
    tr = FakeTracer()
    o._external(tr, "q", Counter(), user, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    key, value = tr.persisted[0][1][0], json.loads(tr.persisted[0][1][3])
    assert key == o._external_key(user, "a", "Zeta-9 Kappa-Bench | Zeta-9", 5) != o._external_key(user, "a", "Zeta-9 Kappa-Bench", 5)
    assert value["query"] == "Zeta-9 Kappa-Bench" and len(value["rows"]) == 1

    calls.clear()
    o.uq = lambda *a, **k: [(key, {"query": "Zeta-9", "rows": [row(7, "a")]})]
    tr = FakeTracer()
    accepted, _ = o._external(tr, "q", Counter(), user, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert calls == [] and [a["url"] for a in accepted] == ["https://doi.org/10.1/7"]
    assert tr.row("external_sources").detail["providers"]["a"] == {"status": "cache_hit", "results": 1, "query": "Zeta-9"}

    o.uq = lambda *a, **k: [(key, [row(8, "a")])]  # a value in another shape is ignored, never trusted
    o._external(FakeTracer(), "q", Counter(), user, ["Zeta-9 Kappa-Bench", "Zeta-9"])
    assert [q for q, _ in calls] == ["Zeta-9 Kappa-Bench"]
