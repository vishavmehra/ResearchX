"""Parsing, licence detection, rule extraction, table cross-check, evidence verification, LLM guardrails."""

import json

import httpx

from services.datasets import manifests, store
from services.datasets.schemas import LicenseInfo, Paper, Provenance, Relevance
from services.evidence.assertions import verify
from services.extraction import experiments, llm, rules
from services.extraction.parse import parse_html
from services.model_gateway.gateway import ModelGateway
from services.retrieval.sources.arxiv import html_license

HTML = r"""<html><head><title>T</title></head><body>
<a id="license-tr" href="https://info.arxiv.org/help/license/index.html#licenses-available">License: arXiv.org perpetual non-exclusive license</a>
<h1 class="ltx_title ltx_title_document">Tiny Paper</h1>
<div class="ltx_abstract"><h6>Abstract</h6><p>We study adapters.</p></div>
<section id="S2" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">2 </span>Related Work</h2>
<div id="S2.p1" class="ltx_para"><p class="ltx_p">Prior work used a learning rate of 0.1 for 90 epochs.</p></div></section>
<section id="S4" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">4 </span>Experimental Setup</h2>
<div id="S4.p1" class="ltx_para"><p class="ltx_p">We fine-tune with the AdamW optimizer, a learning rate of <math alttext="2\times 10^{-4}" display="inline"><mn>2</mn></math> and a batch size of 32 sequences for 3 epochs. The learning rate is decayed at epoch 8. We quantize to INT4 on one A100 GPU.</p></div>
<figure id="S4.T1" class="ltx_table"><figcaption class="ltx_caption">Table 1: Accuracy on GLUE dev.</figcaption>
<table class="ltx_tabular"><thead><tr><th>Model</th><th>SST-2</th><th>RTE</th></tr></thead>
<tbody><tr><td>BERT-base</td><td><math alttext="\mathbf{93.5}"><mn>93.5</mn></math></td><td>66.4</td></tr>
<tr><td>TinyAdapter-7B</td><td>92.1</td><td>70.21</td></tr></tbody></table></figure>
</section></body></html>"""

META = {"paper_id": "paper:arxiv:9999.00001", "source_record_id": "arxiv:9999.00001v1", "source_url": "https://export.arxiv.org/html/9999.00001v1",
        "source_path": "data/fulltext/restricted/9999.00001/source.html", "source_sha256": "0" * 64}


def _doc():
    return parse_html(HTML, document_meta=META, license=LicenseInfo(content_license="ARXIV-NONEXCLUSIVE-1.0", redistribution_allowed=False))


def _paper():
    return Paper(paper_id=META["paper_id"], study_id="study:arxiv:9999.00001", title="Tiny Paper", normalized_title="tiny paper",
                 arxiv_id="9999.00001", source_record_ids=[META["source_record_id"]], merge_evidence=[], quality_tier="BRONZE",
                 relevance=Relevance(is_relevant=True, rule_version="t"), provenance=Provenance(source_name="RESEARCHX", processing_version="x"))


def test_parse_structure_math_and_tables():
    doc = _doc()
    assert doc.title == "Tiny Paper" and doc.abstract == "We study adapters."
    assert {s.section_id: s.section_type for s in doc.sections} == {"S2": "RELATED_WORK", "S4": "EXPERIMENTS"}
    para = next(p for p in doc.paragraphs if p.paragraph_id == "S4.p1")
    assert r"2\times 10^{-4}" in para.text  # TeX alttext preserved as text
    table = doc.tables[0]
    assert table.table_id == "S4.T1" and table.rows[0] == ["Model", "SST-2", "RTE"] and table.header_rows == 1


def test_licence_detection_and_storage_class():
    url, label = html_license(HTML)
    lic = manifests.match_license(url) or manifests.match_license(label)
    assert lic == "ARXIV-NONEXCLUSIVE-1.0"
    assert manifests.fulltext_class(lic) == "restricted"  # arXiv default licence: never redistributed
    assert manifests.fulltext_class(manifests.match_license("CC BY 4.0")) == "permitted"
    assert manifests.fulltext_class(manifests.match_license("CC BY-NC-SA 4.0")) == "restricted"
    assert manifests.fulltext_class(None) == "restricted"  # unknown => restricted


def test_rules_extract_stated_values_with_spans_and_skip_false_positives():
    doc = _doc()
    evs = rules.extract_document(doc, _paper(), "run-test", None)
    got = {(e.field_name, json.dumps(e.normalized_value)) for e in evs}
    assert ("training.learning_rate", "0.0002") in got
    assert ("training.batch_size", "32") in got and ("training.epochs", "3") in got
    assert ("training.optimizer", '"AdamW"') in got and ("compute.gpu_model", '"A100"') in got
    assert ("training.epochs", "8") not in got  # "decayed at epoch 8" is not a number of epochs
    assert not any(e.normalized_value == "T4" for e in evs)  # "INT4" is not a T4 GPU
    assert not any(e.paragraph == "S2.p1" for e in evs)  # related work describes other papers
    bs = next(e for e in evs if e.field_name == "training.batch_size")
    assert bs.unit == "sequences" and bs.experiment_id is None  # study-level evidence, unit from text
    for e in evs:
        para = next(p for p in doc.paragraphs if p.paragraph_id == e.paragraph)
        assert para.text[e.char_start:e.char_end] == e.evidence_text  # verbatim span


def test_evidence_verification_detects_tampering():
    doc = _doc()
    ev = rules.extract_document(doc, _paper(), "run-test", None)[0]
    assert verify(ev, {doc.paper_id: doc}, {}) == (True, "offset_text_mismatch")
    bad = ev.model_copy(update={"evidence_text": ev.evidence_text.replace("AdamW", "SGD").replace("3 epochs", "30 epochs")})
    assert verify(bad, {doc.paper_id: doc}, {})[0] is False


def test_table_crosscheck_statuses():
    doc = _doc()
    row = {"task": "Natural Language Inference", "dataset": "RTE", "model_name": "TinyAdapter-7B", "row_index": 0}
    assert experiments.crosscheck(doc, row, "Accuracy", "70.21")["status"] == "FOUND_ALIGNED"
    assert experiments.crosscheck(doc, row, "Accuracy", "70.2")["status"] == "ROUNDING_CONSISTENT"
    assert experiments.crosscheck(doc, row, "Accuracy", "75.0")["status"] == "CONFLICT_CANDIDATE"
    other_size = {**row, "model_name": "TinyAdapter-13B"}  # size tokens clash: never aligned with the 7B row
    assert experiments.crosscheck(doc, other_size, "Accuracy", "75.0")["status"] == "NOT_FOUND"
    assert experiments.cell_number(r"\mathbf{93.5}") == experiments.cell_number("93.5 ± 0.2")


def _gateway(content: dict, env: dict | None = None) -> ModelGateway:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    cfg = {"providers": {"nebius": {"base_url": "https://mock.local/v1/", "api_key_env": "NEBIUS_API_KEY", "list_models": False}},
           "tasks": {"experiment_extraction": {"provider": "nebius", "candidates": ["nvidia/test-nemotron"], "structured_output": "json_schema"}}}
    return ModelGateway(cfg, transport=httpx.MockTransport(handler), env={"NEBIUS_API_KEY": "test-key"} if env is None else env, log_path=None)


def test_llm_output_is_accepted_only_with_verbatim_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS", tmp_path)  # keep failure records out of the real dataset
    content = {"experiments": [{
        "local_id": "e1", "experiment_type": "MAIN", "description": "TinyAdapter fine-tuning",
        "fields": [
            {"field_name": "training.learning_rate", "value": "2e-4", "location_id": "S4.p1", "quote": r"a learning rate of 2\times 10^{-4}"},
            {"field_name": "training.weight_decay", "value": "0.01", "location_id": "S4.p1", "quote": "weight decay of 0.01"},  # hallucinated
            {"field_name": "training.warmup_steps", "value": "500", "location_id": "S9.p9", "quote": "500 warmup steps"},  # unknown location
            {"field_name": "training.magic", "value": "1", "location_id": "S4.p1", "quote": "AdamW"},  # not in schema
        ],
        "outcomes": [{"metric": "RTE", "score": "70.21", "location_id": "S4.T1", "quote": "70.21"}],
    }]}
    run = store.Run("test_llm")
    exps, evs = llm.extract(_doc(), _paper(), _gateway(content), run)
    assert len(exps) == 1
    e = exps[0]
    assert e.training.learning_rate.status.value == "KNOWN" and e.training.learning_rate.value == 2e-4
    assert e.training.weight_decay.status.value == "UNKNOWN"  # rejected: quote not in source
    assert e.training.warmup_steps.status.value == "UNKNOWN"
    assert len(e.outcomes) == 1 and e.outcomes[0].score.value == 70.21
    assert run.failures == 3
    failures = [json.loads(l) for l in (tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {f["category"] for f in failures} >= {"hallucination"}
    assert all("test-key" not in json.dumps(f) for f in failures)


def test_gateway_without_key_is_unavailable_not_crashing():
    gw = _gateway({}, env={})
    assert not gw.available("experiment_extraction")
    parsed, call = gw.complete_json("experiment_extraction", "s", "u", {}, "v1")
    assert parsed is None and call.status == "UNAVAILABLE"
