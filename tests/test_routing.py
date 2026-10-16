"""Model routing policy (services/model_gateway/router.py), gateway health, in-process caches and the routed synthesis."""

from __future__ import annotations

import copy
import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from services.datasets.store import config
from services.model_gateway import router
from services.model_gateway.gateway import ModelCall, ModelGateway
from services.runtime import memo
from services.runtime.orchestrator import Orchestrator
from services.runtime.trace import Stage

TASKS = {t: {"provider": "nebius", "candidates": [f"m-{t}"], "typical_ms": 2000}
         for t in ("reranking", "synthesis", "synthesis_complex", "field_extraction", "fulltext_extraction")}
TASKS["embedding"] = {"provider": "nebius", "candidates": ["qwen"], "dimensions": 1024, "embedding_version": "emb-0.1.0"}


class FakeGW:
    """Gateway stand-in: scripted outputs per route, a configuration, health switches."""

    def __init__(self, outputs: dict | None = None, unhealthy=(), private_ok=True, tasks=None):
        self.cfg = {"providers": {"nebius": {"private_data": private_ok}}, "tasks": copy.deepcopy(tasks or TASKS)}
        self.outputs, self.unhealthy, self.calls = outputs or {}, set(unhealthy), []

    def available(self, task):
        return True

    def healthy(self, task):
        return task not in self.unhealthy

    def resolve_model(self, task):
        return "nebius", f"m-{task}"

    def complete_json(self, task, system, user, schema, prompt_version):
        self.calls.append(task)
        out = self.outputs.get(task)
        out = out.pop(0) if isinstance(out, list) and out else out if not isinstance(out, list) else None
        return out, SimpleNamespace(model_resolved=f"m-{task}", status="OK" if out is not None else "HTTP_ERROR", error=None if out is not None else "down",
                                    input_tokens=10, output_tokens=5, latency_s=0.01)


def ctx(**kw):
    return router.RoutingContext(**{"task_type": "synthesis", "input_tokens": 500, "latency_budget_ms": 20000, **kw})


# ------------------------------------------------------------------------------------------------ decisions
def test_routine_synthesis_starts_on_nano_with_super_as_fallback():
    d = router.decide(ctx(context_required=4), FakeGW())
    assert (d.selected_route, d.fallback_route, d.selected_model) == ("synthesis", "synthesis_complex", "m-synthesis")
    assert d.selection_reason.startswith("Nano") and d.routing_policy_version == router.POLICY_VERSION


def test_comparison_escalates_to_super_but_recorded_contradictions_do_not():
    d = router.decide(ctx(complexity="high"), FakeGW())
    assert (d.selected_route, d.fallback_route) == ("synthesis_complex", "synthesis") and d.selection_reason.startswith("Super")
    d = router.decide(ctx(verification_required=True), FakeGW())
    assert d.selected_route == "synthesis" and "checked deterministically" in d.selection_reason


def test_length_alone_never_escalates():
    d = router.decide(ctx(input_tokens=9000), FakeGW())
    assert d.selected_route == "synthesis" and "length alone does not escalate" in d.selection_reason


def test_static_and_fixed_modes_reproduce_the_baselines():
    gw = FakeGW()
    assert [router.decide(ctx(complexity=c), gw, "static").selected_route for c in ("low", "high")] == ["synthesis", "synthesis_complex"]
    assert router.decide(ctx(complexity="high"), gw, "static").fallback_route is None
    assert router.decide(ctx(complexity="high"), gw, "fixed_nano").selected_route == "synthesis"
    assert router.decide(ctx(), gw, "fixed_super").selected_route == "synthesis_complex"
    with pytest.raises(ValueError):
        router.decide(ctx(), gw, "cheapest")


def test_extraction_routes():
    gw = FakeGW()
    full = router.decide(ctx(task_type="fulltext_extraction"), gw)
    assert (full.selected_route, full.fallback_route) == ("fulltext_extraction", "field_extraction")
    stored = router.decide(ctx(task_type="extraction"), gw)
    assert (stored.selected_route, stored.fallback_route) == ("field_extraction", "fulltext_extraction")
    assert router.decide(ctx(task_type="extraction"), gw, "static").fallback_route is None
    rr = router.decide(ctx(task_type="reranking"), gw)
    assert (rr.selected_route, rr.fallback_route) == ("reranking", None) and "RRF" in rr.fallback_strategy


def test_unhealthy_route_hands_over_to_its_fallback():
    d = router.decide(ctx(complexity="high"), FakeGW(unhealthy={"synthesis_complex"}))
    assert (d.selected_route, d.fallback_route) == ("synthesis", None) and "circuit open" in d.selection_reason


def test_no_usable_route_names_the_deterministic_last_resort():
    d = router.decide(ctx(task_type="reranking"), FakeGW(unhealthy={"reranking"}))
    assert d.selected_route is None and "hybrid (RRF) order kept" in d.selection_reason


def test_private_context_never_reaches_a_provider_not_approved_for_it():
    d = router.decide(ctx(privacy_scope="user-private"), FakeGW(private_ok=False))
    assert d.selected_route is None and "not approved for private documents" in d.selection_reason
    assert router.decide(ctx(privacy_scope="user-private"), FakeGW()).selected_route == "synthesis"


def test_embedding_route_is_never_generative():
    assert router.usable(FakeGW(), "embedding") == (False, "not a generative route")


def test_context_rejects_user_supplied_model_names_and_bad_values():
    with pytest.raises(ValidationError):
        ctx(model="gpt-x")
    with pytest.raises(ValidationError):
        ctx(complexity="extreme")
    with pytest.raises(ValidationError):
        ctx(latency_budget_ms=0)


def test_latency_budget_swaps_synthesis_tiers_when_only_one_fits():
    tasks = copy.deepcopy(TASKS)
    tasks["synthesis_complex"]["typical_ms"] = 30000
    d = router.decide(ctx(complexity="high"), FakeGW(tasks=tasks))
    assert d.selected_route == "synthesis" and "over the 20000 ms budget" in d.selection_reason


# ------------------------------------------------------------------------------------------------ execution
def _call(route):
    return SimpleNamespace(model_resolved=f"m-{route}", status="OK", error=None, input_tokens=1, output_tokens=1)


def test_execute_falls_back_once_and_keeps_the_best_result():
    d = router.decide(ctx(), FakeGW())
    seen = []

    def attempt(route):
        seen.append(route)
        return route, _call(route), {"synthesis": 1, "synthesis_complex": 0}[route]

    best, attempts = router.execute(FakeGW(), d, attempt)
    assert seen == ["synthesis", "synthesis_complex"] and best == "synthesis"  # parsed-but-rejected beats nothing
    assert [a["event"] for a in attempts] == ["selected", "fallback"]
    assert router.trace(d, attempts)["fallback_used"] is True


def test_execute_stops_at_an_accepted_output():
    d = router.decide(ctx(), FakeGW())
    best, attempts = router.execute(FakeGW(), d, lambda r: (r, _call(r), 2))
    assert best == "synthesis" and len(attempts) == 1 and attempts[0]["outcome"] == "accepted"


def test_execute_skips_a_fallback_that_cannot_fit_the_budget():
    tasks = copy.deepcopy(TASKS)
    tasks["synthesis_complex"]["typical_ms"] = 19999
    gw = FakeGW(tasks=tasks)
    d = router.decide(ctx(), gw)

    def slow(route):
        time.sleep(0.01)
        return None, _call(route), 0

    _, attempts = router.execute(gw, d, slow)
    assert attempts[-1]["event"] == "fallback skipped" and "latency budget" in attempts[-1]["reason"]


# ------------------------------------------------------------------------------------------------ gateway health
def test_gateway_health_opens_after_repeated_failures_and_closes_on_success():
    gw = ModelGateway(cfg={"providers": {}, "tasks": {}}, env={}, log_path=None)
    fail = ModelCall(call_id="x", task="synthesis", provider="p", model_requested=[], model_resolved=None, prompt_version="v",
                     request_sha256="h", started_at="t", status="HTTP_ERROR")
    for _ in range(3):
        gw._log(fail)
    assert not gw.healthy("synthesis") and gw.healthy("reranking")
    fail.status = "OK"
    gw._log(fail)
    assert gw.healthy("synthesis")


# ------------------------------------------------------------------------------------------------ caches
def test_ttl_cache_expires_and_evicts_least_recently_used():
    c = memo.TTLCache(2, 60)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b, the least recently used
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    short = memo.TTLCache(4, 0.01)
    short.put("k", "v")
    time.sleep(0.02)
    assert short.get("k") is None and short.misses == 1


def test_query_embedding_cache_is_per_user():
    calls = []
    gw = FakeGW()
    gw.embed = lambda task, texts, query: (calls.append(texts) or [[0.1, 0.2]], SimpleNamespace(status="OK", error=None, input_tokens=3, latency_s=0.1))
    o = Orchestrator(cfg=copy.deepcopy(config("runtime")), gw=gw, connect_fn=None)
    first = o._embed_query("vit learning rate", "user-a")
    again = o._embed_query("vit learning rate", "user-a")
    other = o._embed_query("vit learning rate", "user-b")
    assert len(calls) == 2 and getattr(again[1], "cache_hit", False) and not getattr(other[1], "cache_hit", False)
    assert first[0] == again[0]


# ------------------------------------------------------------------------------------------------ routed synthesis
class Tr:
    def __init__(self):
        self.rows: dict[str, Stage] = {}

    @contextmanager
    def stage(self, name, **detail):
        st = Stage(name)
        st.detail.update(detail)
        yield st
        self.rows[name] = st

    def record(self, name, status, latency_ms=0, error=None, **detail):
        st = Stage(name)
        st.status, st.detail = status, detail
        self.rows[name] = st


SOURCE = {"sid": "S1", "title": "Zeta", "text": "We train Zeta with a learning rate of 0.001 for 90 epochs.", "chunk_id": "c1", "paper_id": "p1"}
GOOD = {"status": "ANSWERED", "claims": [{"text": "Zeta uses a learning rate of 0.001.", "sources": ["S1"],
                                         "quote": "We train Zeta with a learning rate of 0.001"}]}


def synth(outputs, synth_task="synthesis"):
    gw, tr, routing = FakeGW(outputs), Tr(), {}
    o = Orchestrator(cfg=copy.deepcopy(config("runtime")), gw=gw, connect_fn=None)
    out = o._synthesize(tr, "What learning rate does Zeta use?", [SOURCE], [], synth_task, Counter(), routing, True)
    return out, gw.calls, tr.rows["synthesis"].detail["routing"], routing


def test_invalid_nano_output_escalates_to_super():
    (answer, valid, _, status), calls, rt, routing = synth({"synthesis": [{"status": "MAYBE"}], "synthesis_complex": [GOOD]})
    assert calls == ["synthesis", "synthesis_complex"] and status == "ANSWERED" and len(valid) == 1
    assert rt["fallback_used"] and rt["model_used"] == "m-synthesis_complex" and routing["decisions"][0]["fallback_used"]


def test_unsupported_claims_escalate_but_abstention_does_not():
    bad = {"status": "ANSWERED", "claims": [{"text": "Zeta uses 0.5.", "sources": ["S1"], "quote": "a learning rate of 0.5"}]}
    _, calls, _, _ = synth({"synthesis": [bad], "synthesis_complex": [GOOD]})
    assert calls == ["synthesis", "synthesis_complex"]
    (_, valid, _, status), calls, rt, _ = synth({"synthesis": [{"status": "INSUFFICIENT", "claims": []}], "synthesis_complex": [GOOD]})
    assert calls == ["synthesis"] and status == "INSUFFICIENT" and not valid and not rt["fallback_used"]


def test_comparison_question_starts_on_super():
    _, calls, rt, _ = synth({"synthesis_complex": [GOOD]}, synth_task="synthesis_complex")
    assert calls == ["synthesis_complex"] and rt["selected_route"] == "synthesis_complex"
