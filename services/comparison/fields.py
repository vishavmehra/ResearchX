"""Deterministic field-level experiment comparison (DATASET_SPEC §15, §16, §40, §42).

Versioned rules. UNKNOWN is never a match. Similarity never produces SAME_EXPERIMENT (reserved
for human review). The evaluation setup is compared before outcomes; score similarity is
recorded per metric but never used as evidence of equivalence (spec §40).

Evaluation conditions (split, protocol, metric definition, dataset version) are compared after normalisation
(services.evidence.grounding.condition_compare): "val" equals "validation set"; a narrower form of the other side, or a
value that is not a recognisable condition, is UNKNOWN (0.2.0; the frozen release 0.2.0 labels were made with 0.1.0).

Label ladder (silver-labeler-0.1.0), documented in data/annotations/GUIDELINES.md:
  CONFLICTING_EVIDENCE  a compared field has conflicting evidence
  RELATED_CONTEXT       different benchmarks (or different tasks when a benchmark is unknown)
  MATERIALLY_DIFFERENT  same benchmark, but task / metric set / split / protocol / metric definition differ
  INSUFFICIENT_EVIDENCE same benchmark and metric, but fields needed to establish equivalence are UNKNOWN
  ALIGNED_EXPERIMENT    evaluation setup identical/equivalent with every equivalence field KNOWN
"""

from __future__ import annotations

from typing import Any

from services.canonicalization.normalize import canonical_label, comparable_units, convert_duration
from services.datasets.schemas import ComparisonLabel, Experiment, FieldComparison, FieldComparisonStatus as S, FieldStatus, FieldValue
from services.evidence.grounding import CONDITION_FIELDS, condition_compare

LABELER_VERSION = "silver-labeler-0.2.0"  # 0.2.0: evaluation conditions compared after normalisation (val = validation); unrecognisable = UNKNOWN
CORE_FIELDS = ("research.task", "evaluation.benchmark")
EQUIVALENCE_FIELDS = ("evaluation.dataset_split", "evaluation.metric_definition", "evaluation.evaluation_protocol")
INTERVENTION_FIELDS = (
    "model.model_name", "model.checkpoint", "model.parameter_count", "method.method_name",
    "training.learning_rate", "training.batch_size", "training.epochs", "training.steps",
)
MATERIAL_FIELDS = frozenset(CORE_FIELDS + EQUIVALENCE_FIELDS + ("model.model_name", "model.checkpoint", "method.method_name"))
SAME = {S.IDENTICAL, S.EQUIVALENT}
DIFF = {S.MATERIALLY_DIFFERENT, S.DIFFERENT}


def field_value(exp: Experiment, path: str) -> FieldValue:
    group, name = path.split(".", 1)
    return getattr(getattr(exp, group), name)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_field(path: str, fa: FieldValue, fb: FieldValue) -> FieldComparison:
    def fc(status: S, why: str) -> FieldComparison:
        return FieldComparison(field_name=path, value_a=fa.value, value_b=fb.value, comparison_status=status,
                               evidence_a=list(fa.evidence_ids), evidence_b=list(fb.evidence_ids), rationale=why)

    if FieldStatus.CONFLICTING in (fa.status, fb.status):
        return fc(S.CONFLICTING, "conflicting evidence on at least one side")
    if fa.status is FieldStatus.NOT_APPLICABLE and fb.status is FieldStatus.NOT_APPLICABLE:
        return fc(S.NOT_APPLICABLE, "not applicable to either experiment")
    if fa.status is not FieldStatus.KNOWN or fb.status is not FieldStatus.KNOWN:
        sides = "+".join(s for s, f in (("a", fa), ("b", fb)) if f.status is not FieldStatus.KNOWN)
        return fc(S.UNKNOWN, f"no evidence on side {sides}; UNKNOWN is never treated as a match")
    differ = S.MATERIALLY_DIFFERENT if path in MATERIAL_FIELDS else S.DIFFERENT
    va, vb = fa.value, fb.value
    if path in CONDITION_FIELDS and not (_is_num(va) and _is_num(vb)):  # splits, protocols, metric definitions: normalised, never by spelling
        status, why = condition_compare(path, fa.raw_value or va, fb.raw_value or vb)
        return fc(S.IDENTICAL if status == "SAME" and va == vb else S.EQUIVALENT if status == "SAME" else S.UNKNOWN if status == "UNKNOWN" else differ, why)
    if _is_num(va) and _is_num(vb):
        if fa.unit != fb.unit:
            if not (fa.unit and fb.unit and comparable_units(fa.unit, fb.unit)):
                return fc(S.UNKNOWN, f"incomparable or unstated units ({fa.unit} vs {fb.unit}); no conversion applied")
            vb, rule = convert_duration(vb, fb.unit, fa.unit)
            return fc(S.EQUIVALENT, f"equal after explicit {rule}") if vb == va else fc(differ, f"{va} {fa.unit} != {vb} {fa.unit} after {rule}")
        return fc(S.IDENTICAL, "numerically equal") if va == vb else fc(differ, f"{va} != {vb}")
    if va == vb:
        return fc(S.IDENTICAL, "identical values")
    if canonical_label(str(va)) == canonical_label(str(vb)):
        return fc(S.EQUIVALENT, "equal after canonical_label (case/punctuation) normalization")
    return fc(differ, "values differ")


def _outcomes(exp: Experiment) -> dict[str, Any]:
    return {canonical_label(str(o.metric.value)): o for o in exp.outcomes if o.metric.status is FieldStatus.KNOWN}


def compare_outcomes(a: Experiment, b: Experiment) -> tuple[FieldComparison, list[FieldComparison]]:
    oa, ob = _outcomes(a), _outcomes(b)
    ids_a = [i for o in oa.values() for i in o.metric.evidence_ids]
    ids_b = [i for o in ob.values() for i in o.metric.evidence_ids]
    if not oa or not ob:
        status, why = S.UNKNOWN, "no reported metric on at least one side"
    elif oa.keys() == ob.keys():
        status, why = S.IDENTICAL, "same metric set"
    elif oa.keys() & ob.keys():
        status, why = S.DIFFERENT, f"partially overlapping metrics; shared: {sorted(oa.keys() & ob.keys())}"
    else:
        status, why = S.MATERIALLY_DIFFERENT, "no shared metric: outcomes are not comparable"
    metric_fc = FieldComparison(field_name="evaluation.metric", value_a=sorted(oa), value_b=sorted(ob), comparison_status=status,
                                evidence_a=ids_a, evidence_b=ids_b, rationale=why)
    scores = []
    for m in sorted(oa.keys() & ob.keys()):
        fc = compare_field(f"evaluation.score[{m}]", oa[m].score, ob[m].score)
        fc.rationale += " (outcome recorded only; never evidence of equivalence)"
        scores.append(fc)
    return metric_fc, scores


def compare(a: Experiment, b: Experiment) -> dict[str, Any]:
    """Field comparisons + Silver label for an experiment pair."""
    fcs = {p: compare_field(p, field_value(a, p), field_value(b, p)) for p in CORE_FIELDS + EQUIVALENCE_FIELDS + INTERVENTION_FIELDS}
    metric_fc, score_fcs = compare_outcomes(a, b)
    st = {p: fc.comparison_status for p, fc in fcs.items()}
    uncertainty = None
    missing: list[str] = []
    if S.CONFLICTING in st.values() or metric_fc.comparison_status is S.CONFLICTING:
        label, why = ComparisonLabel.CONFLICTING_EVIDENCE, "at least one compared field has conflicting evidence"
    elif st["evaluation.benchmark"] in DIFF:
        label, why = ComparisonLabel.RELATED_CONTEXT, "different benchmarks: different evaluation context"
    elif st["evaluation.benchmark"] is S.UNKNOWN:
        if st["research.task"] in DIFF:
            label, why = ComparisonLabel.RELATED_CONTEXT, "different tasks; benchmark unknown"
        else:
            missing = ["evaluation.benchmark"]
            label, why = ComparisonLabel.INSUFFICIENT_EVIDENCE, "benchmark unknown on at least one side"
    elif st["research.task"] in DIFF:
        label, why = ComparisonLabel.MATERIALLY_DIFFERENT, "same benchmark, different task"
    elif metric_fc.comparison_status is S.MATERIALLY_DIFFERENT:
        label, why = ComparisonLabel.MATERIALLY_DIFFERENT, "same benchmark, no shared metric"
    elif any(st[f] in DIFF for f in EQUIVALENCE_FIELDS):
        differing = [f for f in EQUIVALENCE_FIELDS if st[f] in DIFF]
        label, why = ComparisonLabel.MATERIALLY_DIFFERENT, f"same benchmark, different {differing}"
    else:
        missing = [f for f in EQUIVALENCE_FIELDS if st[f] is S.UNKNOWN]
        if st["research.task"] is S.UNKNOWN:
            missing.append("research.task")
        if metric_fc.comparison_status is S.UNKNOWN:
            missing.append("evaluation.metric")
        if missing:
            label, why = ComparisonLabel.INSUFFICIENT_EVIDENCE, f"same benchmark and metric, but equivalence cannot be established: {missing} UNKNOWN"
        else:
            label, why = ComparisonLabel.ALIGNED_EXPERIMENT, "evaluation setup identical/equivalent with all equivalence fields known"
            if st["model.model_name"] in SAME:
                uncertainty = "intervention also matches: possible SAME_EXPERIMENT, which requires human verification"
    if label is not ComparisonLabel.ALIGNED_EXPERIMENT and label is not ComparisonLabel.CONFLICTING_EVIDENCE:
        uncertainty = uncertainty or (f"missing: {missing}" if missing else None)
    field_comparisons = list(fcs.values()) + [metric_fc] + score_fcs
    evidence = sorted({i for fc in field_comparisons for i in fc.evidence_a + fc.evidence_b})
    return {"field_comparisons": field_comparisons, "label": label, "rationale": why, "uncertainty": uncertainty,
            "missing": missing, "evidence": evidence, "metric_status": metric_fc.comparison_status}
