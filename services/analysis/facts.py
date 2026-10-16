"""Field facts shared by every analysis feature: one experiment field = one fact with a status and its provenance.

Nothing is inferred. A field the record does not state is UNKNOWN; a value whose sources disagree is CONFLICTING (both
sides kept); a value the dataset pipeline computed (unit conversion, a paper-level setting applied to the experiment) is
DERIVED and says how; only a value a source states is REPORTED. Evidence comes from rx.evidence with its location
(paper section, table cell, passage) and the verbatim text unless the source's licence withholds it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from services.datasets.schemas import Experiment, FieldStatus, FieldValue

REPORTED, DERIVED, CONFLICTING, UNKNOWN, NOT_APPLICABLE = "REPORTED", "DERIVED", "CONFLICTING", "UNKNOWN", "NOT_APPLICABLE"
STATED = (REPORTED, DERIVED)
LABELS = {  # field path -> label shown to people (the path itself is kept for machines)
    "evaluation.benchmark": "benchmark", "evaluation.dataset_split": "evaluation split", "evaluation.metric_definition": "metric definition",
    "evaluation.evaluation_protocol": "evaluation protocol", "model.model_name": "model", "method.method_name": "method",
    "method.baseline": "baselines", "training.seed": "random seed", "compute.gpu_model": "GPU", "compute.gpu_count": "GPU count",
    "artifacts.repository_url": "code repository", "artifacts.checkpoint_url": "checkpoint", "research.task": "task",
    "dataset.dataset_name": "dataset", "dataset.dataset_version": "dataset version", "model.parameter_count": "parameter count",
    "training.learning_rate": "learning rate", "training.learning_rate_schedule": "learning-rate schedule",
}


def label(path: str) -> str:
    return LABELS.get(path) or path.split(".", 1)[-1].replace("_", " ")


def status_of(fv: FieldValue) -> str:
    if fv.status is FieldStatus.CONFLICTING:
        return CONFLICTING
    if fv.status is FieldStatus.KNOWN:
        return DERIVED if fv.derived else REPORTED
    return NOT_APPLICABLE if fv.status is FieldStatus.NOT_APPLICABLE else UNKNOWN


def fact(path: str, fv: FieldValue) -> dict[str, Any]:
    return {"field": path, "label": label(path), "status": status_of(fv), "value": fv.value, "raw": fv.raw_value, "unit": fv.unit,
            "derivation": fv.derivation, "evidence_ids": list(fv.evidence_ids)}


def field_value(exp: Experiment, path: str) -> FieldValue:
    group, name = path.split(".", 1)
    return getattr(getattr(exp, group), name)


def facts(exp: Experiment, paths: Iterable[str]) -> list[dict[str, Any]]:
    return [fact(p, field_value(exp, p)) for p in paths]


def results(exp: Experiment) -> list[dict[str, Any]]:
    """Reported outcomes: metric, score and split as facts (uncertainty / interval only when the source states them)."""
    out = []
    for i, o in enumerate(exp.outcomes):
        if o.metric.status is FieldStatus.UNKNOWN and o.score.status is FieldStatus.UNKNOWN:
            continue
        row = {"index": i, "metric": fact("evaluation.metric", o.metric), "score": fact("evaluation.score", o.score),
               "split": fact("evaluation.dataset_split", o.dataset_split)}
        for extra in ("uncertainty", "confidence_interval", "significance"):
            fv = getattr(o, extra)
            if fv.status is not FieldStatus.UNKNOWN:
                row[extra] = fact(f"evaluation.{extra}", fv)
        out.append(row)
    return out


def counts(items: Iterable[dict]) -> dict[str, int]:
    out = dict.fromkeys((REPORTED, DERIVED, CONFLICTING, UNKNOWN, NOT_APPLICABLE), 0)
    for f in items:
        out[f["status"]] += 1
    return out


def with_evidence(items: list[dict], evidence_detail, per_fact: int = 2) -> list[dict]:
    """Attach up to `per_fact` evidence records (location + verbatim text) to every fact; `evidence_detail(ids, limit)` is
    GraphIndex.evidence_detail. One query for the whole list."""
    ids = [i for f in items for i in f["evidence_ids"][:per_fact]]
    found = {e["id"]: e for e in evidence_detail(ids, limit=len(ids))} if ids else {}
    for f in items:
        f["evidence"] = [found[i] for i in f["evidence_ids"][:per_fact] if i in found]
    return items
