"""Experiment technical depth + scientific-quality dimensions (docs/DATASET_DEEPENING.md §2-3).

Deterministic and evidence-based: a field counts only when KNOWN (which already requires evidence). Model
confidence plays no role. Identity fields (task, model name, benchmark, dataset name) never count as depth.
"""

from __future__ import annotations

from typing import Any

from services.datasets.schemas import (
    FIELD_GROUPS,
    DepthAssessment,
    DepthTier,
    DimensionCoverage,
    Experiment,
    FieldStatus,
    QualityDimensions,
    QualityLevel,
)

DEPTH_VERSION = "depth-0.1.0"
QUALITY_VERSION = "quality-0.1.0"
IDENTITY = {"research.task", "research.domain", "evaluation.benchmark", "model.model_name", "model.model_family", "dataset.dataset_name", "artifacts.paper_url"}
EVAL_PROTOCOL = ["evaluation.dataset_split", "evaluation.metric_definition", "evaluation.statistical_test", "evaluation.evaluation_population",
                 "evaluation.evaluation_protocol", "evaluation.aggregation_method"]
OUTCOME_DEPTH = ["uncertainty", "confidence_interval", "significance", "p_value", "dataset_split"]
DIMENSIONS: dict[str, list[str]] = {
    "dataset": [f"dataset.{n}" for n in FIELD_GROUPS["dataset"]],
    "model": [f"model.{n}" for n in FIELD_GROUPS["model"]],
    "method": [f"method.{n}" for n in FIELD_GROUPS["method"]],
    "training": [f"training.{n}" for n in FIELD_GROUPS["training"]],
    "inference": [f"{g}.{n}" for g in ("inference", "retrieval", "prompting") for n in FIELD_GROUPS[g]],
    "evaluation": EVAL_PROTOCOL + [f"outcome.{n}" for n in OUTCOME_DEPTH],
    "compute": [f"compute.{n}" for n in FIELD_GROUPS["compute"]],
    "artifacts": [f"artifacts.{n}" for n in FIELD_GROUPS["artifacts"]],
}
DIMENSIONS = {d: [p for p in paths if p not in IDENTITY] for d, paths in DIMENSIONS.items()}
TECHNICAL = sorted({p for paths in DIMENSIONS.values() for p in paths})
PAPER_TEXT = {"PAPER_SENTENCE", "PAPER_PARAGRAPH", "PAPER_TABLE_CELL", "PAPER_TABLE", "PAPER_CAPTION", "PAPER_FOOTNOTE"}
ARTIFACT = {"ARTIFACT_FILE", "MODEL_CARD", "RUN_RECORD", "BENCHMARK_SUBMISSION"}


def evidence_class(ev: dict[str, Any] | None) -> str:
    """PAPER_TEXT / ARTIFACT / CLAIM / NONE for one evidence row (handles 0.1.0 rows without location_kind)."""
    if not ev:
        return "NONE"
    kind = ev.get("location_kind")
    if kind in PAPER_TEXT:
        return "PAPER_TEXT"
    if kind in ARTIFACT or ev.get("artifact_id"):
        return "ARTIFACT"
    if kind is None and (ev.get("paragraph") or ((ev.get("table") or "") and not str(ev.get("table")).startswith("leaderboard:"))):
        return "PAPER_TEXT"  # 0.1.0 paragraph spans and table cross-check evidence
    return "CLAIM"


def _known_fields(exp: Experiment) -> dict[str, list[str]]:
    """path -> evidence ids for KNOWN configuration fields and outcome-level depth fields."""
    out: dict[str, list[str]] = {}
    for path, fv in exp.iter_fields():
        if fv.status is FieldStatus.KNOWN:
            out[path] = fv.evidence_ids
    for o in exp.outcomes:
        for n in OUTCOME_DEPTH:
            fv = getattr(o, n)
            if fv.status is FieldStatus.KNOWN:
                out.setdefault(f"outcome.{n}", []).extend(fv.evidence_ids)
    return out


def assess(exp: Experiment, evidence: dict[str, dict[str, Any]], assoc_statuses: list[str], artifact_relations: list[str]) -> tuple[DepthAssessment, QualityDimensions]:
    known = _known_fields(exp)
    tech = {p: ids for p, ids in known.items() if p in TECHNICAL}
    dims = {d: DimensionCoverage(known=sum(1 for p in paths if p in tech), total=len(paths)) for d, paths in DIMENSIONS.items()}
    dims_hit = sum(1 for c in dims.values() if c.known)
    classes = {p: {evidence_class(evidence.get(i)) for i in ids} for p, ids in known.items()}
    for o in exp.outcomes:  # outcome metric/score evidence also informs evidence quality
        for n in ("metric", "score"):
            fv = getattr(o, n)
            if fv.status is FieldStatus.KNOWN:
                classes[f"outcome.{n}#{id(o)}"] = {evidence_class(evidence.get(i)) for i in fv.evidence_ids}
    paper_text = sum(1 for p in tech if "PAPER_TEXT" in classes.get(p, set()))
    artifact = sum(1 for p in tech if "ARTIFACT" in classes.get(p, set()))
    human = sum(1 for p in tech if tech[p] and all((evidence.get(i) or {}).get("status") == "HUMAN_REVIEWED" for i in tech[p]))
    eval_protocol = any(p in tech for p in EVAL_PROTOCOL) or "outcome.dataset_split" in tech
    trained = any(p.startswith("training.") for p in tech)
    train_or_inf = trained or any(p.startswith(("inference.", "retrieval.", "prompting.")) for p in tech)
    has_result = any(o.metric.status is FieldStatus.KNOWN and o.score.status is FieldStatus.KNOWN for o in exp.outcomes)
    deep = len(tech) >= 8 and dims_hit >= 4 and eval_protocol and train_or_inf
    artifact_ok = ("artifacts.repository_url" in tech and "artifacts.commit" in tech) or "artifacts.checkpoint_url" in tech or "model.checkpoint" in tech
    config_ok = (all(p in tech for p in ("training.optimizer", "training.learning_rate", "training.batch_size"))
                 and ("training.epochs" in tech or "training.steps" in tech)) if trained else \
        ("inference.decoding_strategy" in tech or "inference.temperature" in tech)
    reproducible = deep and artifact_ok and eval_protocol and config_ok
    if reproducible and tech and human == len(tech):
        tier = DepthTier.fully_verified_experiment
    elif reproducible:
        tier = DepthTier.reproducible_configuration
    elif deep:
        tier = DepthTier.deep_configuration
    elif len(tech) >= 3 and dims_hit >= 2:
        tier = DepthTier.partial_configuration
    elif has_result:
        tier = DepthTier.shallow_result
    else:
        tier = DepthTier.metadata_only
    depth = DepthAssessment(tier=tier, dimensions=dims, technical_fields_known=len(tech), paper_text_fields=paper_text,
                            artifact_fields=artifact, human_reviewed_fields=human, depth_version=DEPTH_VERSION)

    all_classes = [c for cs in classes.values() for c in cs]
    if classes and human and human == len(tech):
        evq = QualityLevel.HUMAN_VERIFIED
    elif classes and all("PAPER_TEXT" in cs for cs in classes.values()):
        evq = QualityLevel.HIGH
    elif "PAPER_TEXT" in all_classes or "ARTIFACT" in all_classes:
        evq = QualityLevel.MEDIUM
    elif all_classes:
        evq = QualityLevel.LOW
    else:
        evq = QualityLevel.NONE
    share = len(tech) / len(TECHNICAL)
    completeness = QualityLevel.NONE if not tech else QualityLevel.LOW if share < 0.10 else QualityLevel.MEDIUM if share < 0.30 else QualityLevel.HIGH
    if "VERIFIED" in assoc_statuses:
        attribution = QualityLevel.HIGH
    elif "CONFLICTING" in assoc_statuses:
        attribution = QualityLevel.CONFLICTING
    elif "LIKELY" in assoc_statuses:
        attribution = QualityLevel.MEDIUM
    elif "AMBIGUOUS" in assoc_statuses:
        attribution = QualityLevel.LOW
    else:
        attribution = QualityLevel.UNKNOWN
    if "CONFLICTING" in artifact_relations:
        art_q = QualityLevel.CONFLICTING
    elif "CONFIRMED_BY_BOTH" in artifact_relations:
        art_q = QualityLevel.HIGH
    elif "PRESENT_IN_REPOSITORY" in artifact_relations:
        art_q = QualityLevel.MEDIUM
    elif "RELATED_BUT_UNCONFIRMED" in artifact_relations:
        art_q = QualityLevel.LOW
    else:
        art_q = QualityLevel.NONE
    quality = QualityDimensions(evidence_quality=evq, configuration_completeness=completeness, experiment_attribution_quality=attribution,
                                artifact_verification_quality=art_q, quality_version=QUALITY_VERSION)
    return depth, quality
