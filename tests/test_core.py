"""Core invariants: normalization, schema rules, HTTP cache/retry/redaction, source normalization, merging."""

import json
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

from services.canonicalization.normalize import (
    canonical_unit,
    comparable_units,
    convert_duration,
    normalize_doi,
    parse_arxiv_id,
    parse_number,
    reconstruct_abstract,
)
from services.datasets.schemas import EvidenceAssertion, FieldStatus, FieldValue, Provenance, SourceRecord
from services.datasets.store import ROOT
from services.retrieval import http as http_mod
from services.retrieval.corpus import build_papers
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources import openalex


def test_numbers_normalize_consistently():
    # spec §13/§42: 0.00002 and 2e-5 normalize to the same numeric representation
    a, b, c, d = (parse_number(s)[0] for s in ("2e-5", "0.00002", r"2\times 10^{-5}", "2E-05"))
    assert a == b == c == d == 2e-05
    assert parse_number("1e^{-4}")[0] == 1e-4
    assert parse_number("524,288")[0] == 524288
    assert parse_number("2^{19}")[0] == 524288
    assert parse_number("10k") == (10000.0, "k_suffix")
    assert parse_number("about ten") is None


def test_units_are_never_merged_or_inferred():
    assert canonical_unit("epochs") == "epochs" and canonical_unit("steps") == "steps"
    assert not comparable_units("epochs", "steps")  # "10 epochs" != "10 steps" (spec §13)
    assert not comparable_units("steps", "iterations")
    assert convert_duration(2, "hours", "minutes")[0] == 120
    with pytest.raises(ValueError):
        convert_duration(10, "epochs", "steps")
    assert canonical_unit("fortnights") is None  # unknown units are not guessed


def test_identifiers():
    assert normalize_doi("https://doi.org/10.1109/TPAMI.2015.2448083") == "10.1109/tpami.2015.2448083"
    assert parse_arxiv_id("http://arxiv.org/abs/2106.09685v2") == ("2106.09685", "v2")
    assert parse_arxiv_id("10.48550/arXiv.1706.03762") == ("1706.03762", None)
    assert parse_arxiv_id("https://arxiv.org/pdf/1403.6015.pdf") == ("1403.6015", None)


def test_field_value_invariants():
    with pytest.raises(ValidationError):
        FieldValue(status=FieldStatus.KNOWN, value=1e-4)  # KNOWN requires evidence
    with pytest.raises(ValidationError):
        FieldValue(status=FieldStatus.UNKNOWN, value=3)  # UNKNOWN never carries a value
    with pytest.raises(ValidationError):
        FieldValue(status=FieldStatus.KNOWN, value=3, evidence_ids=["ev:1"], derived=True)  # derivation required
    with pytest.raises(ValidationError):
        FieldValue(status=FieldStatus.CONFLICTING, evidence_ids=["ev:1"])  # both sides required
    assert FieldValue().status is FieldStatus.UNKNOWN


def test_evidence_field_names_are_canonical():
    prov = Provenance(source_name="ARXIV", processing_version="x")
    kw = dict(evidence_id="ev:1", source_record_id="arxiv:1v1", evidence_text="lr 1e-4", extraction_model="m", extraction_run_id="r",
              quality_tier="SILVER", provenance=prov)
    EvidenceAssertion(field_name="training.learning_rate", **kw)
    with pytest.raises(ValidationError):
        EvidenceAssertion(field_name="training.magic_number", **kw)


def test_exported_json_schemas_are_in_sync():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/data/export_schemas.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_http_retry_cache_and_secret_redaction(tmp_path, monkeypatch):
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path == "/missing":
            return httpx.Response(404)
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    client = CachedHttp("test", min_interval=0, transport=httpx.MockTransport(handler), cache_root=tmp_path)
    r1 = client.get("https://example.org/works", params={"api_key": "SECRET-123", "q": "lora"})
    assert r1.json() == {"ok": True} and calls["n"] == 2  # retried after 429
    r2 = client.get("https://example.org/works", params={"api_key": "SECRET-123", "q": "lora"})
    assert r2.from_cache and calls["n"] == 2  # replayed from cache: no new request
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert "SECRET-123" not in f.read_text(encoding="utf-8", errors="ignore")
    with pytest.raises(FetchError):
        client.get("https://example.org/missing")


def _work(**over):
    work = {
        "id": "https://openalex.org/W3168867926",
        "doi": "https://doi.org/10.48550/arxiv.2106.09685",
        "display_name": "LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency",
        "publication_date": "2021-06-17", "publication_year": 2021, "updated_date": "2026-09-29",
        "ids": {"mag": "3168867926"},
        "authorships": [{"author": {"display_name": "J. Edward Hu", "id": "A1"}, "raw_author_name": "Hu, Edward J."}],
        "primary_location": {"landing_page_url": "http://arxiv.org/abs/2106.09685", "source": {"id": openalex.ARXIV_SOURCE_ID, "display_name": "arXiv"}},
        "locations": [
            {"landing_page_url": "http://arxiv.org/abs/2106.09685", "license": None, "source": {"id": openalex.ARXIV_SOURCE_ID, "display_name": "arXiv"}},
            {"landing_page_url": "https://doi.org/10.5281/zenodo.21850639", "license": "cc-by", "source": {"id": "S4306400562", "display_name": "Zenodo"}},
        ],
        "abstract_inverted_index": {"We": [0], "propose": [1], "LoRA": [2]},
        "referenced_works": [], "topics": [], "open_access": {},
    }
    work.update(over)
    return work


def test_openalex_license_only_from_arxiv_location():
    rec = openalex.normalize(_work(), {"retrieved_at": "2026-09-30T00:00:00+00:00", "body_path": "data/raw/x.body"}, [])
    assert rec.arxiv_id == "2106.09685"
    assert rec.license.content_license is None  # Zenodo's cc-by must not leak onto the paper
    assert any(l["source"] == "Zenodo" for l in rec.extra["non_arxiv_locations"])
    assert rec.abstract == "We propose LoRA"


def _rec(srid, source, arxiv, title, year=2021):
    return SourceRecord(source_record_id=srid, source_name=source, source_identifier=srid, title=title, arxiv_id=arxiv,
                        publication_year=year, quality_tier="BRONZE",
                        provenance=Provenance(source_name=source, processing_version="x"))


def test_merge_is_identifier_only_and_preserves_conflicts():
    recs = [
        _rec("openalex:W1", "OPENALEX", "2106.09685", "HISTORIAE, a wrongly merged title", 2025),
        _rec("arxiv:2106.09685v2", "ARXIV", "2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", 2021),
        _rec("openalex:W2", "OPENALEX", "2106.09999", "LoRA: Low-Rank Adaptation of Large Language Models", 2021),
    ]
    papers = {p.paper_id: p for p in build_papers(recs)}
    assert len(papers) == 2  # identical titles with different arXiv ids are NOT merged
    p = papers["paper:arxiv:2106.09685"]
    assert p.title.startswith("LoRA")  # documented precedence: arXiv title
    fields = {c["field"] for c in p.metadata_conflicts}
    assert {"title", "publication_year"} <= fields  # disagreement preserved, not overwritten
    assert p.publication_year == 2021  # arXiv v1 date precedence


def test_reconstruct_abstract_order():
    assert reconstruct_abstract({"b": [1], "a": [0], "c": [2, 3]}) == "a b c c"
    assert reconstruct_abstract(None) is None
