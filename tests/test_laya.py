"""Laya decision layer (services/laya/decider.py) with a fake laya-serve: no checkpoint, no network. The run's rules stay
authoritative: shadow mode changes nothing, advise mode only picks the synthesis task, and only past the gate."""

from collections import Counter
from concurrent.futures import Future
from types import SimpleNamespace

import httpx
import pytest

from services.laya import decider
from services.runtime import orchestrator as orch
from tests.test_passage_path import PipeTracer, pipeline

GENERAL = "Tell me about Zeta-9 and Omega-3 on Kappa-Bench"  # no rule intent: detect_intent -> general


class FakeService:
    """Stands in for laya-serve's /v1/systemone (decider._post)."""

    def __init__(self, choice="comparison", gate="passed", injection=0.01, fail=None):
        self.choice, self.gate, self.injection, self.fail, self.calls = choice, gate, injection, fail, []

    def __call__(self, url, body, timeout):
        self.calls.append((url, body, timeout))
        if self.fail:
            raise self.fail
        probs = dict.fromkeys(body["questions"]["workflow"]["criteria"], 0.01) | {self.choice: 0.97}
        return {"routing": {"model": "english", "repo": "convaiinnovations/laya"}, "answers": {
            "workflow": {"choice": self.choice, "answer_confidence": 0.97, "abstention": self.gate, "probabilities": probs},
            "external_needed": {"noul": 0.2}, "injection": {"noul": self.injection}, "depth": {"score": 1.1}}}


@pytest.fixture
def router(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(decider, "_post", fake)
    monkeypatch.delenv("RESEARCHX_LAYA_URL", raising=False)
    return fake


def done(value) -> Future:
    f = Future()
    f.set_result(value)
    return f


def cfg(mode="shadow", enabled=True):
    return {"laya": {"enabled": enabled, "mode": mode, "model": "english", "min_confidence": 0.8, "budget_ms": 200}}


def test_decision_shape_and_inputs(router):
    d = decider.decide(GENERAL, cfg())
    assert d["status"] == "completed" and d["workflow"]["choice"] == "comparison" and d["workflow"]["gate"] == "passed"
    assert d["injection"] == 0.01 and d["depth"] == 1.1 and d["checkpoint"] == "convaiinnovations/laya"
    url, body, timeout = router.calls[0]
    assert url == "http://127.0.0.1:8090" and body["state"] == {"request": GENERAL} and sorted(body["questions"]) == sorted(decider.QUESTIONS)
    assert body["model"] == "english" and body["min_confidence"] == 0.8 and timeout == 4.0


def test_laya_failures_and_absence_never_raise(monkeypatch, router):
    router.fail = httpx.ConnectError("connection refused")
    d = decider.decide(GENERAL, cfg())
    assert d["status"] == "failed" and "unreachable" in d["error"]
    router.fail = httpx.ReadTimeout("slow")
    assert decider.decide(GENERAL, cfg())["status"] == "failed"
    router.fail = KeyError("answers")  # a malformed answer is a failure, never a decision
    assert decider.decide(GENERAL, cfg())["status"] == "failed"
    assert decider.decide(GENERAL, cfg(enabled=False))["status"] == "skipped"
    monkeypatch.setenv("RESEARCHX_LAYA", "0")
    assert "disabled" in decider.decide(GENERAL, cfg())["reason"]


def test_agreement_with_the_rules_and_the_injection_guard(router):
    d = decider.decide(GENERAL, cfg())
    assert decider.agreement(d, "comparison", [], False) == {"rule_workflow": "comparison", "workflow_agrees": True, "rule_injection": False,
                                                            "injection_agrees": True}
    assert decider.agreement(d, "general", ["training.learning_rate"], True)["workflow_agrees"] is False
    assert decider.agreement(d, "general", [], False)["workflow_agrees"] is None, "no rule counterpart: no agreement claimed"


@pytest.mark.parametrize("mode, choice, gate, intent, expected", [
    ("advise", "comparison", "passed", "general", "synthesis_complex"),
    ("advise", "explanation", "passed", "general", "synthesis_complex"),
    ("advise", "comparison", "abstained", "general", None),  # below the gate: the rules' choice stands
    ("advise", "lookup", "passed", "general", None),
    ("advise", "comparison", "passed", "protocol", None),  # the rules found an intent: they decide
    ("shadow", "comparison", "passed", "general", None),
])
def test_advice_applies_only_past_the_gate_when_the_rules_found_no_intent(mode, choice, gate, intent, expected):
    d = {"status": "completed", "workflow": {"choice": choice, "gate": gate}}
    assert decider.advise_synthesis(d, intent, cfg(mode)) == expected


def run_pipeline(monkeypatch, mode, decision):
    o, calls = pipeline(monkeypatch, [("ok", [{"text": "t", "sources": ["S1"], "quote": "t"}], [], "ANSWERED")])
    o.cfg["laya"] = cfg(mode)["laya"]
    tasks = []
    inner = o._synthesize
    o._synthesize = lambda tr, q, s, c, task, u, *rest, **kw: (tasks.append(task), inner(tr, q, s, c, task, u, *rest, **kw))[1]
    lay = {"future": done(decision), "recorded": False}
    tr = PipeTracer()
    o._pipeline(tr, GENERAL, orch.norm(GENERAL), Counter(), {}, SimpleNamespace(user=None), lay)
    return o, tr, lay, tasks


def test_advise_mode_picks_the_larger_synthesis_model_and_traces_it(monkeypatch, router):
    o, tr, lay, tasks = run_pipeline(monkeypatch, "advise", decider.decide(GENERAL, cfg()))
    row = next(r for r in tr.rows if r.name == "laya_decision")
    assert tasks == ["synthesis_complex"] and lay["recorded"] and row.status == "completed"
    assert row.detail["applied"] == "synthesis task -> synthesis_complex" and row.detail["mode"] == "advise"


def test_shadow_mode_changes_nothing_and_is_traced_after_the_run(monkeypatch, router):
    o, tr, lay, tasks = run_pipeline(monkeypatch, "shadow", decider.decide(GENERAL, cfg()))
    assert tasks == ["synthesis"], "the rules chose synthesis; Laya's comparison was not applied"
    assert not lay["recorded"] and not [r for r in tr.rows if r.name == "laya_decision"]
    o._laya_record(tr, GENERAL, lay, o._laya_wait(lay, 0.05, o.cfg))
    row = next(r for r in tr.rows if r.name == "laya_decision")
    assert row.status == "completed" and row.detail["applied"] is None and row.detail["agreement"]["workflow_agrees"] is None


def test_a_late_laya_answer_is_skipped_not_waited_for(monkeypatch):
    o, _ = pipeline(monkeypatch, [])
    pending = Future()  # never completes
    d = o._laya_wait({"future": pending, "recorded": False}, 0.01, o.cfg)
    assert d["status"] == "skipped" and "had not answered" in d["reason"]
    assert o._laya_wait({"future": None, "recorded": False}, 0, o.cfg)["status"] == "skipped", "no future: no model is loaded to explain"
