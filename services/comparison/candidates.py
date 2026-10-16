"""Comparison candidates and hard negatives (DATASET_SPEC §17, §40, §41).

Generators (benchmark overlap, citation links, same paper, same model, same model family,
same benchmark with different metrics) only PROPOSE pairs; they are stored as candidate
features. Every label comes from services.comparison.fields (explicit field-level evidence).
Sampling is deterministic (hash order) so rebuilds are reproducible.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Iterable

from services.canonicalization.normalize import canonical_label, jaccard, model_family, title_tokens
from services.comparison.fields import LABELER_VERSION, compare, field_value
from services.datasets.schemas import Comparison, ComparisonLabel, Experiment, FieldStatus, Provenance, Split
from services.datasets.store import code_version, stable_id

MAX_EXPERIMENTS_PER_BENCHMARK = 60
MAX_PAIRS_PER_BENCHMARK = 40
MAX_HARD_NEGATIVES_PER_GROUP = 4


def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _value(exp: Experiment, path: str) -> str:
    fv = field_value(exp, path)
    return canonical_label(str(fv.value)) if fv.status is FieldStatus.KNOWN else ""


def _metrics(exp: Experiment) -> frozenset[str]:
    return frozenset(canonical_label(str(o.metric.value)) for o in exp.outcomes if o.metric.status is FieldStatus.KNOWN)


def features(a: Experiment, b: Experiment, cites: set[tuple[str, str]]) -> dict[str, float]:
    return {
        "description_jaccard": round(jaccard(title_tokens(a.description), title_tokens(b.description)), 4),
        "same_paper": float(a.paper_id == b.paper_id),
        "citation_link": float((a.paper_id, b.paper_id) in cites or (b.paper_id, a.paper_id) in cites),
        "shared_metrics": float(len(_metrics(a) & _metrics(b))),
    }


def make_comparison(a: Experiment, b: Experiment, methods: list[str], feats: dict[str, float], hard_negative_type: str | None = None) -> Comparison:
    if a.experiment_id > b.experiment_id:
        a, b = b, a
    res = compare(a, b)
    label = res["label"]
    if hard_negative_type is None and "benchmark_overlap" in methods:
        if label is ComparisonLabel.INSUFFICIENT_EVIDENCE:
            hard_negative_type = "SIMILAR_EXPERIMENT_INSUFFICIENT_EVIDENCE"  # spec §17 example 14
        elif label is ComparisonLabel.MATERIALLY_DIFFERENT and res["metric_status"].value == "MATERIALLY_DIFFERENT":
            hard_negative_type = "SAME_BENCHMARK_DIFFERENT_METRIC"  # spec §17 example 9
    is_hn = hard_negative_type is not None and label not in (ComparisonLabel.SAME_EXPERIMENT, ComparisonLabel.ALIGNED_EXPERIMENT)
    return Comparison(
        comparison_id=stable_id("cmp", a.experiment_id, b.experiment_id),
        experiment_a=a.experiment_id,
        experiment_b=b.experiment_id,
        relationship_label=label,
        field_comparisons=res["field_comparisons"],
        evidence=res["evidence"],
        rationale=res["rationale"],
        uncertainty=res["uncertainty"],
        is_hard_negative=is_hn,
        hard_negative_type=hard_negative_type if is_hn else None,
        candidate_generation={"methods": methods, "features": feats},
        labeler_version=LABELER_VERSION,
        split=a.split if a.split == b.split else Split.CROSS_SPLIT,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=LABELER_VERSION,
                              parent_record_ids=[a.experiment_id, b.experiment_id]),
    )


def comparison_candidates(exps: list[Experiment], cites: set[tuple[str, str]]) -> list[Comparison]:
    """Cross-paper pairs sharing a benchmark (+ citation-link flag), capped per benchmark."""
    by_bench: dict[str, list[Experiment]] = defaultdict(list)
    for e in exps:
        if b := _value(e, "evaluation.benchmark"):
            by_bench[b].append(e)
    out: dict[str, Comparison] = {}
    for bench, group in sorted(by_bench.items()):
        group = sorted(group, key=lambda e: _h(bench, e.experiment_id))[:MAX_EXPERIMENTS_PER_BENCHMARK]
        pairs = [(x, y) for x, y in combinations(group, 2) if x.paper_id != y.paper_id]
        scored = []
        for x, y in pairs:
            f = features(x, y, cites)
            scored.append((-f["citation_link"], -min(f["shared_metrics"], 1), _h(x.experiment_id, y.experiment_id), x, y, f))
        for *_, x, y, f in sorted(scored)[:MAX_PAIRS_PER_BENCHMARK]:
            methods = ["benchmark_overlap"] + (["citation_link"] if f["citation_link"] else [])
            c = make_comparison(x, y, methods, f)
            out[c.comparison_id] = c
    return [out[k] for k in sorted(out)]


def _take(pairs: Iterable[tuple[Experiment, Experiment]], key: str, n: int) -> list[tuple[Experiment, Experiment]]:
    return sorted(pairs, key=lambda p: _h(key, p[0].experiment_id, p[1].experiment_id))[:n]


def hard_negatives(exps: list[Experiment], cites: set[tuple[str, str]], exclude: set[str]) -> list[Comparison]:
    """Targeted hard-negative constructions (spec §17), each tagged with its construction method."""
    constructions: list[tuple[str, list[str], list[tuple[Experiment, Experiment]]]] = []

    by_paper: dict[str, list[Experiment]] = defaultdict(list)
    for e in exps:
        by_paper[e.paper_id].append(e)
    for pid, group in sorted(by_paper.items()):  # spec §17 ex.4: same paper, different experiment
        pairs = [(x, y) for x, y in combinations(group, 2) if _value(x, "evaluation.benchmark") != _value(y, "evaluation.benchmark")]
        constructions.append(("SAME_PAPER_DIFFERENT_EXPERIMENT", ["same_paper"], _take(pairs, pid, MAX_HARD_NEGATIVES_PER_GROUP)))

    by_model: dict[str, list[Experiment]] = defaultdict(list)
    by_family: dict[str, list[Experiment]] = defaultdict(list)
    for e in exps:
        if m := _value(e, "model.model_name"):
            by_model[m].append(e)
            fam, size = model_family(m)
            if size and fam:
                by_family[fam].append(e)
    for m, group in sorted(by_model.items()):
        cross = [(x, y) for x, y in combinations(group, 2) if x.paper_id != y.paper_id]
        diff_bench = [(x, y) for x, y in cross if _value(x, "evaluation.benchmark") != _value(y, "evaluation.benchmark")]
        same_bench = [(x, y) for x, y in cross if _value(x, "evaluation.benchmark") == _value(y, "evaluation.benchmark")]
        # spec §17 ex.7/12: same model, different setup; ex.13/14: same model+benchmark reported by different papers
        constructions.append(("SAME_MODEL_DIFFERENT_BENCHMARK", ["same_model_name"], _take(diff_bench, m, MAX_HARD_NEGATIVES_PER_GROUP)))
        constructions.append(("SAME_MODEL_SAME_BENCHMARK_DIFFERENT_PAPER", ["same_model_name", "benchmark_overlap"], _take(same_bench, m, 2)))
    for fam, group in sorted(by_family.items()):  # spec §17 ex.2: same model family, different checkpoint
        pairs = [(x, y) for x, y in combinations(group, 2)
                 if model_family(_value(x, "model.model_name"))[1] != model_family(_value(y, "model.model_name"))[1]
                 and _value(x, "evaluation.benchmark") == _value(y, "evaluation.benchmark")]
        constructions.append(("SAME_FAMILY_DIFFERENT_CHECKPOINT", ["same_model_family", "benchmark_overlap"], _take(pairs, fam, MAX_HARD_NEGATIVES_PER_GROUP)))

    by_bench: dict[str, list[Experiment]] = defaultdict(list)
    for e in exps:
        if b := _value(e, "evaluation.benchmark"):
            by_bench[b].append(e)
    for bench, group in sorted(by_bench.items()):  # spec §17 ex.9: same benchmark, different metric
        group = sorted(group, key=lambda e: _h(bench, e.experiment_id))[:MAX_EXPERIMENTS_PER_BENCHMARK]
        pairs = [(x, y) for x, y in combinations(group, 2) if _metrics(x) and _metrics(y) and not (_metrics(x) & _metrics(y))]
        constructions.append(("SAME_BENCHMARK_DIFFERENT_METRIC", ["benchmark_overlap", "disjoint_metrics"], _take(pairs, bench, MAX_HARD_NEGATIVES_PER_GROUP)))

    out: dict[str, Comparison] = {}
    for hn_type, methods, pairs in constructions:
        for x, y in pairs:
            cid = stable_id("cmp", *sorted((x.experiment_id, y.experiment_id)))
            if cid in exclude or cid in out:
                continue
            out[cid] = make_comparison(x, y, methods, features(x, y, cites), hard_negative_type=hn_type)
    return [out[k] for k in sorted(out)]
