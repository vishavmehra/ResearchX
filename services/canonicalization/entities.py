"""Canonical entities (deepening §37): one id per concept name, every original surface form kept.

Normalization is deliberately conservative - case, unicode width/dash variants, whitespace and punctuation only.
"ImageNet" and "ImageNet-21k" stay different entities; "Acc." and "Accuracy" stay different entities.
No synonym tables, no embedding merges: a wrong merge silently corrupts every comparison built on it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from services.datasets.schemas import CanonicalEntity, CanonicalEntityType, Experiment, FieldStatus, Provenance
from services.datasets.store import code_version, stable_id

ENTITY_VERSION = "entities-0.1.0"
RULE = "nfkc+casefold; keep [a-z0-9] and dots between digits"

# entity type -> experiment field paths whose KNOWN raw values are surface forms of that entity type
FIELD_SOURCES: dict[CanonicalEntityType, tuple[str, ...]] = {
    CanonicalEntityType.DATASET: ("dataset.dataset_name",),
    CanonicalEntityType.MODEL: ("model.model_name", "model.base_model"),
    CanonicalEntityType.BENCHMARK: ("evaluation.benchmark",),
    CanonicalEntityType.METRIC: ("outcome.metric",),
    CanonicalEntityType.OPTIMIZER: ("training.optimizer",),
    CanonicalEntityType.SCHEDULER: ("training.learning_rate_schedule",),
    CanonicalEntityType.HARDWARE: ("compute.gpu_model", "compute.accelerator"),
    CanonicalEntityType.METHOD: ("method.method_name",),
    CanonicalEntityType.TASK: ("research.task",),
}


def key(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"[^a-z0-9.]+", "", s)
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", s)


def surface_forms(e: Experiment) -> Iterable[tuple[CanonicalEntityType, str, str, list[str]]]:
    """(entity type, field path, original name, evidence ids) for every KNOWN entity-bearing field of one experiment."""
    for etype, paths in FIELD_SOURCES.items():
        for fp in paths:
            if fp == "outcome.metric":
                fvs = [o.metric for o in e.outcomes]
            else:
                g, n = fp.split(".", 1)
                fvs = [getattr(getattr(e, g), n)]
            for fv in fvs:
                if fv.status is FieldStatus.KNOWN and fv.value not in (None, ""):
                    raw = str(fv.raw_value if fv.raw_value not in (None, "") else fv.value).strip()
                    if key(raw):
                        yield etype, fp, raw, list(fv.evidence_ids)


def entity_id(etype: CanonicalEntityType, name: str) -> str | None:
    k = key(name)
    return stable_id("ent", etype.value, k) if k else None


def build(exps: list[Experiment]) -> list[CanonicalEntity]:
    forms: dict[tuple[CanonicalEntityType, str], Counter] = defaultdict(Counter)
    for e in exps:
        for etype, _, raw, _ in surface_forms(e):
            forms[(etype, key(raw))][raw] += 1
    prov = Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=ENTITY_VERSION)
    out = []
    for (etype, k), counts in sorted(forms.items(), key=lambda t: (t[0][0].value, t[0][1])):
        ranked = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
        out.append(CanonicalEntity(
            canonical_id=stable_id("ent", etype.value, k), entity_type=etype, normalized_name=k, preferred_name=ranked[0][0],
            surface_forms=[{"original_name": n, "count": c} for n, c in ranked], normalization_rule=RULE,
            normalization_version=ENTITY_VERSION, provenance=prov))
    return out
