"""Structured-output hardening (core-engine brief §3): every model output goes parse -> schema -> semantic -> provenance,
and malformed output never crashes the pipeline, never becomes a factual answer and is traced. Offline."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest

from services.evidence import grounding as G
from services.graph import runtime as rg
from services.model_gateway import structured
from services.model_gateway.gateway import ModelGateway
from services.runtime import reader
from tests.test_runtime_evidence import (
    ABSTRACT,
    LR_Q,
    FakeDB,
    FakeTracer,
    ask,
    external,
    orchestrator,
)

FACT = {"source": "R1", "field": "training.learning_rate", "value": "3e-4", "span": "a learning rate of 3e-4"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(rg, "_TITLES", {})
    monkeypatch.setattr(reader, "read", lambda *a, **k: pytest.fail("unexpected full-text read (network)"))


# ------------------------------------------------------------------------------------------------ schema: field extraction
@pytest.mark.parametrize("data, status, kept", [
    ({"facts": [FACT]}, "OK", 1),                                                       # valid
    ({"facts": "none"}, "INVALID", 0),                                                   # wrong shape (the Batch 2 case)
    ({"facts": [{k: v for k, v in FACT.items() if k != "span"}]}, "INVALID", 0),         # missing field
    ({"facts": [{**FACT, "span": 42}]}, "INVALID", 0),                                   # wrong type
    ({"facts": [{**FACT, "source": "source one"}]}, "INVALID", 0),                       # malformed id
    ({"facts": [{**FACT, "confidence": 0.9}]}, "INVALID", 0),                            # unexpected item field
    ({"facts": [FACT], "notes": "trust me"}, "INVALID", 0),                              # unexpected top-level key
    ({"facts": []}, "EMPTY", 0),                                                         # empty output
    ({}, "INVALID", 0),                                                                  # nothing at all
    ({"facts": [FACT, {"source": "R2"}]}, "PARTIAL", 1),                                 # partial output
    (["facts"], "INVALID", 0),                                                           # not an object
])
def test_extraction_output_schema(data, status, kept):
    items, rep = structured.items(data, "facts", structured.ExtractedFact, 8)
    assert (rep["status"], len(items)) == (status, kept), rep
    assert status in ("OK", "EMPTY") or rep["errors"], "every failure says why"


def test_a_bare_number_is_the_value_as_written_but_other_types_are_not():
    items, _ = structured.items({"facts": [{**FACT, "value": 0.0003}]}, "facts", structured.ExtractedFact)
    assert items[0].value == "0.0003"
    assert structured.items({"facts": [{**FACT, "value": True}]}, "facts", structured.ExtractedFact)[1]["status"] == "INVALID"
    _, rep = structured.items({"facts": [FACT] * 10}, "facts", structured.ExtractedFact, 8)
    assert rep["status"] == "PARTIAL" and rep["dropped"] == 2, "items past the limit are dropped and reported"


# ------------------------------------------------------------------------------------------------ schema: synthesis
CLAIM = {"text": "The learning rate is 0.02.", "sources": ["S1"], "quote": "learning rate as 0.02"}


@pytest.mark.parametrize("data, status, claims", [
    ({"status": "ANSWERED", "claims": [CLAIM]}, "ANSWERED", 1),
    ({"status": "MAYBE", "claims": [CLAIM]}, None, 0),                                   # invalid enum
    ({"status": "ANSWERED", "claims": "see above"}, None, 0),                            # wrong shape
    ({"status": "ANSWERED", "claims": [CLAIM], "confidence": "high"}, None, 0),          # unexpected key
    ({}, None, 0),                                                                       # empty output
    ({"status": "ANSWERED", "claims": [{"text": "x", "sources": [], "quote": "y"}]}, "ANSWERED", 0),  # all claims malformed
    ({"status": "PARTIAL", "claims": [CLAIM, {"text": "no quote", "sources": ["S1"]}]}, "PARTIAL", 1),  # partial
    ({"status": "INSUFFICIENT", "claims": []}, "INSUFFICIENT", 0),
])
def test_synthesis_output_schema(data, status, claims):
    st, cl, rep = structured.synthesis(data)
    assert (st, len(cl)) == (status, claims), rep


# ------------------------------------------------------------------------------------------------ parse: the gateway
def gateway(content: str) -> ModelGateway:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}})
    cfg = {"providers": {"nebius": {"base_url": "https://mock.local/v1/", "api_key_env": "K", "list_models": False}},
           "tasks": {"t": {"provider": "nebius", "candidates": ["m"], "structured_output": "json_schema", "max_retries": 1}}}
    return ModelGateway(cfg, transport=httpx.MockTransport(handler), env={"K": "k"}, log_path=None)


@pytest.mark.parametrize("content, status", [
    ('{"facts": []}', "OK"),
    ("not json at all", "INVALID_JSON"),                                                 # malformed JSON
    ("", "INVALID_JSON"),                                                                # empty output
    ('["facts"]', "INVALID_SHAPE"),                                                      # a list, not an object
    ('"done"', "INVALID_SHAPE"),
])
def test_gateway_parse_step(content, status):
    data, call = gateway(content).complete_json("t", "s", "u", {"type": "object"}, "v")
    assert call.status == status and (data is not None) == (status == "OK")


def test_truncated_output_keeps_its_complete_leading_items():
    data, call = gateway('{"facts": [' + json.dumps(FACT) + ', {"source": "R2", "fie').complete_json("t", "s", "u", {}, "v")
    assert call.status == "OK" and structured.items(data, "facts", structured.ExtractedFact)[1]["status"] == "OK"


# ------------------------------------------------------------------------------------------------ the pipeline never crashes
class ScriptedModel:
    """Returns the given outputs in turn (None: the provider failed)."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def complete_json(self, task, system, user, schema, prompt_version):
        out = self.outputs.pop(0) if self.outputs else None
        return out, SimpleNamespace(input_tokens=1, output_tokens=1, model_resolved="fake", status="OK" if out is not None else "HTTP_ERROR",
                                    latency_s=0.0, error=None if out is not None else "down")

    def resolve_model(self, task):
        return None, "fake"


@pytest.mark.parametrize("output", [{"facts": "none"}, {"facts": [{"source": 1}]}, {"facts": [{**FACT, "source": "R9"}]},
                                    {"facts": [{**FACT, "field": "training.dropout"}]}, None])
def test_malformed_extraction_never_crashes_and_never_answers(output):
    row = external(1, "Zeta-9: scaling small models", ABSTRACT + " We use a learning rate of 3e-4 on Kappa-Bench.")
    result, tr, _ = ask(orchestrator(FakeDB(), ScriptedModel(output), [row]), LR_Q)
    assert result["status"] == "ABSTAINED" and not result["claims"], "no fact without a valid output"
    step = tr.row("external_extraction").detail["extraction"][0]["validation"]
    assert step["status"] in ("INVALID", "UNAVAILABLE") and step["errors"], "the failure is traced with its reason"


def test_a_valid_extraction_still_answers():
    row = external(1, "Zeta-9: scaling small models", ABSTRACT + " We use a learning rate of 3e-4 on Kappa-Bench.")
    out = {"facts": [{**FACT, "span": "We use a learning rate of 3e-4 on Kappa-Bench"}]}
    result, tr, _ = ask(orchestrator(FakeDB(), ScriptedModel(out), [row]), LR_Q)
    assert result["status"] == "ANSWERED" and tr.row("external_extraction").detail["extraction"][0]["validation"]["status"] == "OK"


# ------------------------------------------------------------------------------------------------ synthesis: schema then provenance
SOURCES = [{"sid": "S1", "title": "OCR", "text": "We set the initial learning rate as 0.02 and the batch size as 16."},
           {"sid": "S2", "title": "Other", "text": "A different paper about crop sizes."}]


def synthesize(output):
    o = orchestrator(FakeDB(), ScriptedModel(output))
    tr = FakeTracer()
    return o._synthesize(tr, "What learning rate?", SOURCES, [], "synthesis", Counter()), tr


def test_malformed_synthesis_becomes_quoted_evidence_never_a_model_answer():
    for bad in ({"status": "MAYBE", "claims": [CLAIM]}, {"status": "ANSWERED", "claims": "x"}, None):
        (answer, valid, _, status), tr = synthesize(bad)
        assert status is None and answer.startswith("The synthesis model")
        assert all(any(c["quote"] in s["text"] for s in SOURCES) for c in valid), "only verbatim passages, no model claim"
        assert tr.row("synthesis").status == "failed" and tr.row("synthesis").detail["validation"]["errors"]


def test_schema_valid_claims_still_face_citation_validation():
    out = {"status": "ANSWERED", "claims": [
        {"text": "The learning rate is 0.02.", "sources": ["S1", "S2"], "quote": "initial learning rate as 0.02"},  # S2 does not hold it
        {"text": "The learning rate is 0.03.", "sources": ["S1"], "quote": "initial learning rate as 0.03"},        # not in the source
        {"text": "Batch size 16.", "sources": ["S7"], "quote": "batch size as 16"}]}                              # unknown source
    (answer, valid, rejected, _), tr = synthesize(out)
    assert [c["sources"] for c in valid] == [["S1"]], "a citation whose source lacks the quote is dropped"
    assert [r["reason"] for r in rejected] == ["quote_not_found", "no_valid_source"] and "0.03" not in answer
    assert tr.row("citation_validation").detail["valid"] == 1


def test_grounding_rejects_a_schema_valid_value_its_span_does_not_support():
    ok, reason, _ = G.validate("training.learning_rate", "0.03", "initial learning rate as 0.02", SOURCES[0]["text"])
    assert not ok and reason == "value_not_in_span"
