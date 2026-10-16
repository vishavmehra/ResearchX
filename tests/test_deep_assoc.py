"""Experiment assembly, association statuses, scope rules and depth tiers (no model calls)."""

from services.datasets.depth import assess
from services.datasets.schemas import (
    AssociationStatus,
    EvidenceScope,
    Experiment,
    ExperimentType,
    ExtractionMethod,
    FieldStatus,
    FieldValue,
    LicenseInfo,
    MergeEvidence,
    OutcomeObservation,
    Paper,
    Provenance,
)
from services.datasets.store import Run
from services.extraction import associate, deep
from services.extraction.parse import parse_html

HTML = """<html><body>
<section id="S4" class="ltx_section"><h2 class="ltx_title ltx_title_section">4 Experiments</h2>
 <div class="ltx_para" id="S4.p1"><p class="ltx_p">All models are trained with a learning rate of 2e-5 for 3 epochs.</p></div>
 <div class="ltx_para" id="S4.p2"><p class="ltx_p">For ablations we use a learning rate of 1e-4 instead.</p></div>
 <figure class="ltx_table" id="S4.T1"><figcaption>Table 1: Results on the GLUE dev set.</figcaption>
  <table class="ltx_tabular">
   <thead><tr><th class="ltx_td ltx_th ltx_th_column">Model</th><th class="ltx_td ltx_th ltx_th_column">MNLI</th><th class="ltx_td ltx_th ltx_th_column">SST-2</th></tr></thead>
   <tbody><tr><th class="ltx_td ltx_th ltx_th_row">BERT</th><td>84.6</td><td>93.5</td></tr>
          <tr><th class="ltx_td ltx_th ltx_th_row">Ours</th><td>86.1</td><td>94.0</td></tr></tbody>
  </table></figure>
</section></body></html>"""
META = {"paper_id": "paper:arxiv:9999.00001", "source_record_id": "arxiv:9999.00001v1", "source_url": "https://export.arxiv.org/html/9999.00001v1",
        "source_path": "x", "source_sha256": "0" * 64}


class MemRun(Run):
    def __init__(self):
        super().__init__("test")
        self.mem = []

    def fail(self, *a, **k):
        self.failures += 1
        self.mem.append(a)


def setup():
    doc = parse_html(HTML, document_meta=META, license=LicenseInfo())
    paper = Paper(paper_id=META["paper_id"], study_id="study:arxiv:9999.00001", title="T", normalized_title="t", arxiv_id="9999.00001",
                  source_record_ids=[META["source_record_id"]], merge_evidence=[MergeEvidence(source_record_id=META["source_record_id"], matched_on="self", value="x")],
                  provenance=Provenance(source_name="RESEARCHX", processing_version="test"))
    return doc, paper, MemRun()


def spec_for_ours(doc, paper, run):
    texts = deep.location_texts(doc)
    a = deep.accept(doc, paper, texts, run, field_name="model.model_name", value="Ours", location_id="S4.T1/r2c0", quote="Ours",
                    model_id="m", extractor="t", scope=EvidenceScope.EXPERIMENT)
    sp = deep.ExperimentSpec("e1", ExperimentType.MAIN, "Ours on GLUE")
    sp.identity["model.model_name"] = a
    sp.rows, sp.columns = [("S4.T1", "ours")], [("S4.T1", "mnli")]
    return sp


def setting(doc, paper, run, loc, quote, value, scope=EvidenceScope.PAPER, applies_to=None):
    return deep.accept(doc, paper, deep.location_texts(doc), run, field_name="training.learning_rate", value=value, location_id=loc, quote=quote,
                       model_id="m", extractor="t", scope=scope, applies_to=applies_to).evidence


def test_new_experiment_gets_cell_outcome_split_and_paper_setting():
    doc, paper, run = setup()
    lr = setting(doc, paper, run, "S4.p1", "learning rate of 2e-5", "2e-5")
    out = associate.assemble(paper, doc, [], [spec_for_ours(doc, paper, run)], [lr], run)
    (e,) = out.experiments
    assert e.extraction_method is ExtractionMethod.LLM
    assert [a.status for a in out.associations] == [AssociationStatus.LIKELY]
    (o,) = e.outcomes
    assert (o.metric.value, o.score.value) == ("MNLI", 86.1)  # value from the cell, never from the model
    assert e.training.learning_rate.status is FieldStatus.KNOWN and e.training.learning_rate.value == 2e-5
    depth, quality = assess(e, {ev.evidence_id: ev.model_dump(mode="json") for ev in out.evidence}, ["LIKELY"], [])
    assert depth.tier.value == "shallow_result" and depth.technical_fields_known == 1
    assert quality.experiment_attribution_quality.value == "MEDIUM" and quality.evidence_quality.value == "HIGH"


def test_competing_values_stay_unknown():
    doc, paper, run = setup()
    a = setting(doc, paper, run, "S4.p1", "learning rate of 2e-5", "2e-5")
    b = setting(doc, paper, run, "S4.p2", "learning rate of 1e-4", "1e-4")
    out = associate.assemble(paper, doc, [], [spec_for_ours(doc, paper, run)], [a, b], run)
    (e,) = out.experiments
    assert e.training.learning_rate.status is FieldStatus.UNKNOWN
    assert "training.learning_rate" in e.unresolved_fields


def test_percent_fraction_impossible_values_and_duplicate_specs():
    html = HTML.replace("All models are trained with a learning rate of 2e-5 for 3 epochs.", "We use a warmup ratio of 5% and a learning rate of 0.")
    doc = parse_html(html, document_meta=META, license=LicenseInfo())
    _, paper, run = setup()
    texts = deep.location_texts(doc)
    w = deep.accept(doc, paper, texts, run, field_name="training.warmup_ratio", value="5%", location_id="S4.p1", quote="warmup ratio of 5%",
                    model_id="m", extractor="t", scope=EvidenceScope.PAPER).evidence
    lr = deep.accept(doc, paper, texts, run, field_name="training.learning_rate", value="0", location_id="S4.p1", quote="learning rate of 0",
                     model_id="m", extractor="t", scope=EvidenceScope.PAPER).evidence
    out = associate.assemble(paper, doc, [], [spec_for_ours(doc, paper, run), spec_for_ours(doc, paper, run)], [w, lr], run)
    (e,) = out.experiments  # the repeated identical spec is one experiment
    assert e.training.warmup_ratio.value == 0.05 and e.training.warmup_ratio.normalization_rule == "percent_to_fraction"
    assert e.training.learning_rate.status is FieldStatus.UNKNOWN and out.counts["settings_invalid_value"] == 1


def test_claim_cell_verified_and_conflict():
    doc, paper, run = setup()
    claim = Experiment(
        experiment_id="exp:claim", study_id=paper.study_id, paper_id=paper.paper_id, source_record_ids=["pwc:9999.00001"],
        experiment_type=ExperimentType.EVALUATION, extraction_method=ExtractionMethod.SOURCE_IMPORT, extraction_run_id="r",
        provenance=Provenance(source_name="RESEARCHX", processing_version="test"),
        outcomes=[OutcomeObservation(metric=FieldValue(status=FieldStatus.KNOWN, raw_value="MNLI", value="MNLI", evidence_ids=["ev:m"]),
                                     score=FieldValue(status=FieldStatus.KNOWN, raw_value="84.6", value=84.6, evidence_ids=["ev:s"]))],
        cross_check={"outcomes": [{"status": "FOUND_ALIGNED", "location": {"table_id": "S4.T1", "cell": "84.6", "row_label": "BERT", "row": 1}}]},
    )
    out = associate.assemble(paper, doc, [claim], [], [], run)
    (a,) = out.associations
    assert a.status is AssociationStatus.VERIFIED and out.results[0].cell == "S4.T1/r1c1"
    assert any(i.startswith("ev:") for i in claim.outcomes[0].score.evidence_ids if i not in ("ev:s",))
