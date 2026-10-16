"""Field-level comparison labels, review/promotion policy and validation failure modes."""

import json
from datetime import UTC, datetime, timedelta

from services.comparison.fields import compare, compare_field
from services.datasets.review import _outcome
from services.datasets.schemas import (
    ComparisonLabel,
    Experiment,
    FieldComparisonStatus as S,
    FieldStatus,
    FieldValue,
    OutcomeObservation,
    Provenance,
    ReviewDecision,
    ReviewDecisionValue as D,
    ReviewerRole,
    ReviewerType,
)
from services.datasets.validate import validate


def known(v, ev="ev:x", unit=None):
    return FieldValue(status=FieldStatus.KNOWN, raw_value=str(v), value=v, unit=unit, evidence_ids=[ev])


def exp(eid, task="QA", bench="SQuAD", metrics=("F1",), model="BERT", **equiv) -> Experiment:
    e = Experiment(experiment_id=eid, study_id="study:s", paper_id=f"paper:{eid}", source_record_ids=["pwc:1"], extraction_method="SOURCE_IMPORT",
                   extraction_run_id="r", provenance=Provenance(source_name="RESEARCHX", processing_version="x"))
    e.research.task = known(task)
    e.evaluation.benchmark = known(bench)
    e.model.model_name = known(model)
    for k, v in equiv.items():
        setattr(e.evaluation, k, known(v))
    e.outcomes = [OutcomeObservation(metric=known(m), score=known(90.0)) for m in metrics]
    return e


def test_unknown_is_never_a_match():
    fc = compare_field("training.learning_rate", known(1e-4), FieldValue())
    assert fc.comparison_status is S.UNKNOWN
    assert compare_field("evaluation.benchmark", known("SQuAD 1.1"), known("squad-1.1")).comparison_status is S.EQUIVALENT
    assert compare_field("training.steps", known(10, unit="epochs"), known(10, unit="steps")).comparison_status is S.UNKNOWN
    assert compare_field("compute.training_time", known(2, unit="hours"), known(120, unit="minutes")).comparison_status is S.EQUIVALENT


def test_label_ladder():
    a = exp("a")
    assert compare(a, exp("b", model="RoBERTa"))["label"] is ComparisonLabel.INSUFFICIENT_EVIDENCE  # split/protocol unknown
    assert compare(a, exp("c", bench="TriviaQA"))["label"] is ComparisonLabel.RELATED_CONTEXT
    assert compare(a, exp("d", metrics=("EM",)))["label"] is ComparisonLabel.MATERIALLY_DIFFERENT  # same benchmark, disjoint metrics
    full = dict(dataset_split="dev", metric_definition="token F1", evaluation_protocol="official script")
    aligned = compare(exp("e", **full), exp("f", model="RoBERTa", **full))
    assert aligned["label"] is ComparisonLabel.ALIGNED_EXPERIMENT
    same = compare(exp("g", **full), exp("h", **full))
    assert same["label"] is ComparisonLabel.ALIGNED_EXPERIMENT and "SAME_EXPERIMENT" in same["uncertainty"]  # never auto-assigned
    differ = compare(exp("i", **full), exp("j", **{**full, "dataset_split": "test"}))
    assert differ["label"] is ComparisonLabel.MATERIALLY_DIFFERENT


T0 = datetime(2026, 9, 30, tzinfo=UTC)


def dec(i, reviewer, decision, rtype=ReviewerType.HUMAN, role=ReviewerRole.ANNOTATOR, **kw):
    return ReviewDecision(decision_id=f"dec:{i}", target_type="x", target_id="t", reviewer_id=reviewer, reviewer_type=rtype, reviewer_role=role,
                          decision=decision, guideline_version="g", timestamp=T0 + timedelta(minutes=i), **kw)


def test_promotion_policy():
    assert _outcome("experiment", [dec(1, "agent", D.ACCEPT, ReviewerType.AGENT)])[0] == "PENDING"  # agents never promote
    assert _outcome("experiment", [dec(1, "alice", D.ACCEPT)])[0] == "PROMOTE"
    assert _outcome("comparison", [dec(1, "alice", D.ACCEPT)])[0] == "PENDING"  # high impact needs 2
    assert _outcome("comparison", [dec(1, "alice", D.ACCEPT), dec(2, "bob", D.ACCEPT)])[0] == "PROMOTE"
    assert _outcome("comparison", [dec(1, "alice", D.ACCEPT), dec(2, "bob", D.REJECT)])[0] == "DISPUTED"  # disagreement preserved
    adjudicated = [dec(1, "alice", D.ACCEPT), dec(2, "bob", D.REJECT), dec(3, "carol", D.REJECT, role=ReviewerRole.ADJUDICATOR)]
    assert _outcome("comparison", adjudicated)[0] == "REJECTED"
    assert _outcome("experiment", [dec(1, "alice", D.UNSURE)])[0] == "PENDING"


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_validation_fails_loudly_on_orphans_and_unreviewed_gold(tmp_path):
    prov = {"source_name": "RESEARCHX", "processing_version": "x"}
    _write(tmp_path / "normalized/papers/papers.jsonl", [{
        "record_type": "paper", "paper_id": "paper:1", "study_id": "study:missing", "title": "T", "normalized_title": "t",
        "source_record_ids": ["openalex:W404"], "merge_evidence": [], "quality_tier": "BRONZE", "provenance": prov}])
    e = exp("gold1").model_dump(mode="json")
    e["quality_tier"] = "GOLD"
    _write(tmp_path / "gold/experiments/experiments.jsonl", [e])
    report = validate(data_root=tmp_path, reverify_evidence=False)
    checks = {c["check"] for c in report.critical}
    assert {"orphan_reference", "gold_integrity"} <= checks
    assert not report.summary()["passed"]
