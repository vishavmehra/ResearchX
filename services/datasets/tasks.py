"""Research tasks and hard research cases (deepening §24-25) built over real records.

Wording is templated (`synthetic_text=True`); every task points at the real records and evidence it came from, and
its expected behaviour follows deterministic rules. Nothing here is Gold: tasks stay Silver until a human reviews
them. Each type is capped and round-robined across papers, so the set stays small and diverse instead of large.
A type with no supporting records yields nothing (true duplicates, for example, need human SAME_EXPERIMENT labels).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from services.canonicalization import entities
from services.datasets.schemas import (
    CanonicalEntityType,
    ExpectedBehavior,
    Experiment,
    ExperimentType,
    FieldStatus,
    HardCase,
    HardCaseType,
    Provenance,
    ResearchTask,
    ResearchTaskType,
    Split,
)
from services.datasets.store import code_version, stable_id

GENERATOR_VERSION = "tasks-0.1.0"
CAP = 200  # per task / hard-case type
PROTOCOL = ("evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition")
HYPER = ("training.learning_rate", "training.batch_size", "training.epochs", "training.steps", "training.optimizer",
         "training.weight_decay", "training.warmup_steps", "training.learning_rate_schedule")
DIFF = ("MATERIALLY_DIFFERENT", "DIFFERENT", "CONFLICTING")
SAME = ("IDENTICAL", "EQUIVALENT")
T, B = ResearchTaskType, ExpectedBehavior


def _fields(e: Experiment) -> dict:
    return dict(e.iter_fields())


def _known(e: Experiment, path: str):
    fv = _fields(e).get(path)
    return fv.value if fv is not None and fv.status is FieldStatus.KNOWN else None


def _raw(e: Experiment, path: str) -> str | None:
    fv = _fields(e).get(path)
    if fv is None or fv.status is not FieldStatus.KNOWN:
        return None
    return str(fv.raw_value if fv.raw_value not in (None, "") else fv.value)


def _metric(e: Experiment) -> str | None:
    return next((str(o.metric.raw_value or o.metric.value) for o in e.outcomes if o.metric.status is FieldStatus.KNOWN), None)


def describe(e: Experiment) -> str:
    model = _raw(e, "model.model_name") or "an unnamed model"
    data = _raw(e, "dataset.dataset_name") or _raw(e, "evaluation.benchmark")
    metric = _metric(e)
    return f"{model}{' on ' + data if data else ''}{' (' + metric + ')' if metric else ''}"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def lexical_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return round(len(ta & tb) / len(ta | tb), 4) if ta | tb else 0.0


def _split(exps: Iterable[Experiment]) -> Split:
    splits = {e.split for e in exps}
    return splits.pop() if len(splits) == 1 else Split.CROSS_SPLIT


def _round_robin(items: list[tuple[str, object]], cap: int = CAP) -> list:
    """Interleave items grouped by their key (paper) so no single paper dominates; deterministic order."""
    groups: dict[str, list] = defaultdict(list)
    for k, it in items:
        groups[k].append(it)
    queues = [groups[k] for k in sorted(groups)]
    out: list = []
    while queues and len(out) < cap:
        queues = [q for q in queues if q]
        for q in queues:
            if len(out) >= cap:
                break
            out.append(q.pop(0))
    return out


class Builder:
    def __init__(self, exps: list[Experiment], has_text: set[str]):
        self.exps = exps
        self.by_id = {e.experiment_id: e for e in exps}
        self.has_text = has_text
        self.prov = Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=GENERATOR_VERSION)

    def task(self, ttype: ResearchTaskType, question: str, context: list[str], behavior: ExpectedBehavior, rationale: str, *,
             answer: str | None = None, evidence: Iterable[str] = (), decisive: Iterable[str] = (), proposed: dict | None = None,
             members: Iterable[Experiment] = ()) -> ResearchTask:
        members = list(members)
        return ResearchTask(task_id=stable_id("task", ttype.value, context, sorted(set(decisive))), task_type=ttype, question=question, context_ids=context,
                            evidence_ids=sorted(set(evidence)), decisive_fields=sorted(set(decisive)), proposed_experiment=proposed or {},
                            expected_behavior=behavior, expected_answer=answer, rationale=rationale, generator_version=GENERATOR_VERSION,
                            split=_split(members) if members else Split.UNASSIGNED, provenance=self.prov)

    # -- comparisons: benchmark / protocol / configuration / duplicate detection ------------------------------
    def comparison_tasks(self, comps: list[dict]) -> list[ResearchTask]:
        pools: dict[ResearchTaskType, list] = defaultdict(list)
        for c in comps:
            a, b = self.by_id.get(c["experiment_a"]), self.by_id.get(c["experiment_b"])
            if a is None or b is None:
                continue
            fcs = c.get("field_comparisons", [])
            status = {f["field_name"]: f["comparison_status"] for f in fcs}
            diff = sorted(f for f, s in status.items() if s in DIFF)
            unknown_protocol = sorted(f for f in PROTOCOL if status.get(f) == "UNKNOWN")
            evs = [i for f in fcs for i in f.get("evidence_a", []) + f.get("evidence_b", [])]
            ctx = [c["comparison_id"], a.experiment_id, b.experiment_id]
            label = c["relationship_label"]
            if a.paper_id == b.paper_id and label in ("MATERIALLY_DIFFERENT", "RELATED_CONTEXT") and diff:
                pools[T.duplicate_experiment_detection].append((a.paper_id, self.task(
                    T.duplicate_experiment_detection, f"Is experiment '{describe(a)}' a duplicate of experiment '{describe(b)}' reported in the same paper?",
                    ctx, B.ANSWER, f"Silver comparison {c['comparison_id']} ({label}); fields differ: {', '.join(diff)}.",
                    answer=f"No - {label}: differs in {', '.join(diff)}", evidence=evs, decisive=diff, members=(a, b))))
                continue
            if status.get("evaluation.benchmark") not in SAME:
                continue
            bench = _raw(a, "evaluation.benchmark") or "the same benchmark"
            q = f"Can the results of '{describe(a)}' and '{describe(b)}' on {bench} be compared directly?"
            if label == "INSUFFICIENT_EVIDENCE" and unknown_protocol:
                full = a.paper_id in self.has_text and b.paper_id in self.has_text
                pools[T.benchmark_comparison].append((a.paper_id, self.task(
                    T.benchmark_comparison, q, ctx, B.ABSTAIN if full else B.SEARCH_MORE,
                    f"Comparability depends on {', '.join(unknown_protocol)}, which no extracted evidence states"
                    + (" although both papers' full text was parsed." if full else "; at least one paper has no parsed full text."),
                    answer="Cannot be determined from the available evidence", evidence=evs, decisive=unknown_protocol, members=(a, b))))
                continue
            known_protocol = [f for f in PROTOCOL if status.get(f) not in (None, "UNKNOWN")]
            known_hyper = [f for f in HYPER if status.get(f) not in (None, "UNKNOWN")]
            ttype = (T.dataset_protocol_comparison if known_protocol else T.model_configuration_comparison if known_hyper else T.benchmark_comparison)
            pools[ttype].append((a.paper_id, self.task(
                ttype, q, ctx, B.ANSWER, f"Silver comparison {c['comparison_id']} labelled {label} by the field-level ladder.",
                answer=f"{label}" + (f": differs in {', '.join(diff)}" if diff else ""), evidence=evs,
                decisive=diff or known_protocol or known_hyper, members=(a, b))))
        return [t for pool in pools.values() for t in _round_robin(pool)]

    # -- ablations within one paper ---------------------------------------------------------------------------
    def ablation_tasks(self) -> list[ResearchTask]:
        by_paper: dict[str, list[Experiment]] = defaultdict(list)
        for e in self.exps:
            by_paper[e.paper_id].append(e)
        pool = []
        for pid, es in sorted(by_paper.items()):
            mains = sorted((e for e in es if e.experiment_type is ExperimentType.MAIN), key=lambda e: e.experiment_id)
            for ab in sorted((e for e in es if e.experiment_type is ExperimentType.ABLATION), key=lambda e: e.experiment_id):
                if not mains:
                    break
                m, fa, fm = mains[0], _fields(ab), _fields(mains[0])
                diff = sorted(p for p, fv in fa.items() if fv.status is FieldStatus.KNOWN and p in fm and fm[p].status is FieldStatus.KNOWN
                              and str(fv.value).lower() != str(fm[p].value).lower())
                if diff:
                    evs = [i for p in diff for i in fa[p].evidence_ids + fm[p].evidence_ids]
                    pool.append((pid, self.task(T.ablation_comparison, f"Which configuration fields does ablation '{describe(ab)}' change relative to the main experiment '{describe(m)}'?",
                                                [ab.experiment_id, m.experiment_id], B.ANSWER, "Both experiments come from the same paper; differing fields are KNOWN on both sides.",
                                                answer=", ".join(diff), evidence=evs, decisive=diff, members=(ab, m))))
        return _round_robin(pool)

    # -- study links and contradictions -------------------------------------------------------------------------
    def link_tasks(self, links: list[dict], papers: dict[str, dict]) -> list[ResearchTask]:
        pool = []
        for lk in links:
            a, b = papers.get(lk["record_a"]), papers.get(lk["record_b"])
            if a is None or b is None:
                continue
            uncertain = lk["relationship"] == "UNCERTAIN"
            t = self.task(T.study_linking, f"Do '{a.get('title')}' and '{b.get('title')}' report the same underlying study?",
                          [lk["link_id"], lk["record_a"], lk["record_b"]], B.ESCALATE if uncertain else B.ANSWER,
                          "SAME_UNDERLYING_STUDY needs human review; " + ("candidate evidence is inconclusive." if uncertain else "Silver link label from candidate evidence."),
                          answer=lk["relationship"])
            t.split = Split(lk.get("split", "UNASSIGNED"))
            pool.append((lk["record_a"], t))
        return _round_robin(pool)

    def contradiction_tasks(self, cons: list[dict]) -> list[ResearchTask]:
        pool = []
        for k in cons:
            t = self.task(T.contradiction_investigation, f"What is the {k['field']} of {k['subject_id']}?", [k["contradiction_id"], k["subject_id"]], B.ANSWER,
                          "Sources disagree; the expected answer reports both values with their sources and picks no winner.",
                          answer=f"Conflicting: {k['source_a']} reports {k.get('value_a')!r}; {k['source_b']} reports {k.get('value_b')!r}",
                          evidence=k.get("evidence_a", []) + k.get("evidence_b", []), decisive=[k["field"]])
            t.split = Split(k.get("split", "UNASSIGNED"))
            pool.append((k["subject_id"], t))
        return _round_robin(pool)

    # -- evidence verification (positives from checked evidence, negatives from unfound leaderboard claims) ----
    def evidence_tasks(self, evidence: dict[str, dict]) -> list[ResearchTask]:
        pos, neg = [], []
        for e in self.exps:
            for path, fv in _fields(e).items():
                if fv.status is not FieldStatus.KNOWN or not path.startswith(("training.", "dataset.", "evaluation.", "compute.", "inference.")):
                    continue
                checked = [i for i in fv.evidence_ids if (evidence.get(i) or {}).get("status") == "SOURCE_CHECKED"
                           and (evidence.get(i) or {}).get("location_kind") != "ARTIFACT_FILE"]
                if checked:
                    pos.append((e.paper_id, self.task(T.evidence_verification, f"Does the paper state {path} = {fv.raw_value or fv.value} for '{describe(e)}'?",
                                                      [e.experiment_id], B.ANSWER, "The value is re-found verbatim at the cited location (SOURCE_CHECKED).",
                                                      answer="Yes", evidence=checked, decisive=[path], members=(e,))))
                    break  # one field per experiment keeps the set diverse
            for o in (e.cross_check or {}).get("outcomes", []):
                if o.get("status") == "NOT_FOUND" and e.outcomes:
                    full = e.paper_id in self.has_text
                    neg.append((e.paper_id, self.task(
                        T.evidence_verification, f"A leaderboard lists '{describe(e)}' with score {e.outcomes[0].score.raw_value}. Does the paper's text report this value?",
                        [e.experiment_id], B.ABSTAIN if full else B.SEARCH_MORE,
                        "Papers With Code claim not found at any parsed location of the paper" + ("." if full else "; full text not parsed."),
                        answer="Not found in the parsed paper text", evidence=e.outcomes[0].score.evidence_ids, decisive=["evaluation.score"], members=(e,))))
                    break
        return _round_robin(pos) + _round_robin(neg)

    # -- missing protocol and preflight review ------------------------------------------------------------------
    def protocol_tasks(self) -> list[ResearchTask]:
        missing, preflight = [], []
        known_protocol: dict[str, list[Experiment]] = defaultdict(list)
        for e in self.exps:
            bench = entities.entity_id(CanonicalEntityType.BENCHMARK, _raw(e, "evaluation.benchmark") or "")
            if bench and _known(e, "evaluation.dataset_split") is not None:
                known_protocol[bench].append(e)
        for e in self.exps:
            if not e.outcomes or e.paper_id not in self.has_text:
                continue
            gaps = [p for p in PROTOCOL if _known(e, p) is None]
            if "evaluation.dataset_split" in gaps:
                missing.append((e.paper_id, self.task(T.missing_protocol_search, f"Which data split was '{describe(e)}' evaluated on?", [e.experiment_id], B.ABSTAIN,
                                                      "No extracted evidence states the split although the full text was parsed; human review may still find it.",
                                                      answer="Not stated in the extracted evidence", decisive=["evaluation.dataset_split"], members=(e,))))
            bench_raw = _raw(e, "evaluation.benchmark")
            if bench_raw and gaps and _raw(e, "model.model_name"):
                prior = [x for x in known_protocol.get(entities.entity_id(CanonicalEntityType.BENCHMARK, bench_raw) or "", []) if x.paper_id != e.paper_id]
                proposed = {"model.model_name": _raw(e, "model.model_name"), "evaluation.benchmark": bench_raw}
                if prior:
                    p = sorted(prior, key=lambda x: x.experiment_id)[0]
                    preflight.append((e.paper_id, self.task(T.preflight_experiment_review, f"I plan to evaluate {proposed['model.model_name']} on {bench_raw}. What must be fixed before results are comparable with prior work?",
                                                            [e.experiment_id, p.experiment_id], B.ANSWER, "A prior experiment on the same benchmark states its split; align with it.",
                                                            answer=f"Use the protocol of {p.experiment_id} (split {_raw(p, 'evaluation.dataset_split')})",
                                                            evidence=_fields(p)["evaluation.dataset_split"].evidence_ids, decisive=gaps, proposed=proposed, members=(e, p))))
                else:
                    preflight.append((e.paper_id, self.task(T.preflight_experiment_review, f"I plan to evaluate {proposed['model.model_name']} on {bench_raw}. Is the plan specified well enough to compare with prior work?",
                                                            [e.experiment_id], B.ASK_USER, "No experiment in the corpus states the protocol for this benchmark; the user must specify it.",
                                                            answer=f"Specify: {', '.join(gaps)}", decisive=gaps, proposed=proposed, members=(e,))))
        return _round_robin(missing) + _round_robin(preflight)

    # -- search tasks over canonical entities (prior art, extension, replication, novelty) -----------------------
    def search_tasks(self) -> list[ResearchTask]:
        keyed = []
        for e in self.exps:
            m = entities.entity_id(CanonicalEntityType.MODEL, _raw(e, "model.model_name") or "")
            d = (entities.entity_id(CanonicalEntityType.DATASET, _raw(e, "dataset.dataset_name") or "")
                 or entities.entity_id(CanonicalEntityType.BENCHMARK, _raw(e, "evaluation.benchmark") or ""))
            k = entities.entity_id(CanonicalEntityType.METRIC, _metric(e) or "")
            if m and d:
                keyed.append((e, m, d, k))
        by_mdk, by_dk, by_md = defaultdict(list), defaultdict(list), defaultdict(list)
        for e, m, d, k in keyed:
            by_md[(m, d)].append(e)
            if k:
                by_mdk[(m, d, k)].append(e)
                by_dk[(d, k)].append(e)
        pools: dict[ResearchTaskType, list] = defaultdict(list)

        def others(group: list[Experiment], e: Experiment) -> list[str]:
            return sorted({x.experiment_id for x in group if x.paper_id != e.paper_id})[:10]

        for e, m, d, k in keyed:
            proposed = {p: _raw(e, p) for p in ("model.model_name", "dataset.dataset_name", "evaluation.benchmark") if _raw(e, p)}
            if k and (hits := others(by_mdk[(m, d, k)], e)):
                pools[T.replication_discovery].append((e.paper_id, self.task(
                    T.replication_discovery, f"Which other papers report '{describe(e)}' with the same model, data and metric?", [e.experiment_id, *hits], B.ANSWER,
                    "Same canonical model, dataset/benchmark and metric names in different papers; replication status needs human review.",
                    answer=", ".join(hits), proposed=proposed, members=(e,))))
            if k and (hits := others(by_dk[(d, k)], e)):
                pools[T.prior_art_experiment_search].append((e.paper_id, self.task(
                    T.prior_art_experiment_search, f"What prior experiments exist on {proposed.get('dataset.dataset_name') or proposed.get('evaluation.benchmark')} measured with {_metric(e)}?",
                    [e.experiment_id, *hits], B.ANSWER, "Same canonical dataset/benchmark and metric in other papers.", answer=", ".join(hits), proposed=proposed, members=(e,))))
            ext = sorted({x.experiment_id for x in by_md[(m, d)] if x.paper_id != e.paper_id and _metric(x) != _metric(e)})[:10]
            if ext:
                pools[T.experiment_extension_search].append((e.paper_id, self.task(
                    T.experiment_extension_search, f"Which experiments extend '{describe(e)}' by measuring the same model and data with a different metric?",
                    [e.experiment_id, *ext], B.ANSWER, "Same canonical model and dataset/benchmark, different metric, other papers.", answer=", ".join(ext), proposed=proposed, members=(e,))))
        datas = sorted({(d, e.experiment_id) for e, _, d, _ in keyed})
        seen_md = set(by_md)
        for e, m, d, _ in keyed:
            other = next(((d2, eid) for d2, eid in datas if d2 != d and (m, d2) not in seen_md), None)
            if other:
                f = self.by_id[other[1]]
                data_name = _raw(f, "dataset.dataset_name") or _raw(f, "evaluation.benchmark")
                proposed = {"model.model_name": _raw(e, "model.model_name"), "dataset.dataset_name": data_name}
                pools[T.research_novelty_assessment].append((e.paper_id, self.task(
                    T.research_novelty_assessment, f"Has anyone evaluated {proposed['model.model_name']} on {data_name}?", [e.experiment_id, f.experiment_id], B.SEARCH_MORE,
                    "No experiment in the internal corpus pairs this model with this data; novelty cannot be claimed without external search.",
                    answer="No matching experiment in the internal corpus", proposed=proposed, members=(e, f))))
        return [t for pool in pools.values() for t in _round_robin(pool)]

    # -- hard cases -------------------------------------------------------------------------------------------------
    def hard_cases(self, comps: list[dict]) -> list[HardCase]:
        pools: dict[HardCaseType, list] = defaultdict(list)
        for c in comps:
            a, b = self.by_id.get(c["experiment_a"]), self.by_id.get(c["experiment_b"])
            if a is None or b is None:
                continue
            fcs = c.get("field_comparisons", [])
            status = {f["field_name"]: f["comparison_status"] for f in fcs}
            same = sorted(f for f, s in status.items() if s in SAME)
            diff = sorted(f for f, s in status.items() if s in DIFF)
            unknown = sorted(f for f, s in status.items() if s == "UNKNOWN")
            hn = c.get("hard_negative_type")
            if hn == "SAME_FAMILY_DIFFERENT_CHECKPOINT":
                ctype = HardCaseType.SAME_MODEL_DIFFERENT_CHECKPOINT
            elif hn == "SAME_PAPER_DIFFERENT_EXPERIMENT":
                ctype = HardCaseType.SAME_TITLE_DIFFERENT_EXPERIMENT
            elif "evaluation.dataset_split" in diff and "evaluation.benchmark" in same:
                ctype = HardCaseType.SAME_DATASET_DIFFERENT_SPLIT
            elif "evaluation.evaluation_protocol" in diff and "evaluation.benchmark" in same:
                ctype = HardCaseType.SAME_BENCHMARK_DIFFERENT_PROTOCOL
            elif set(diff) & set(HYPER) and {"evaluation.benchmark", "model.model_name"} <= set(same):
                ctype = HardCaseType.SAME_SETUP_DIFFERENT_HYPERPARAMETERS
            elif "method.objective" in diff and "model.model_name" in same:
                ctype = HardCaseType.SAME_MODEL_DIFFERENT_OBJECTIVE
            elif c["relationship_label"] == "INSUFFICIENT_EVIDENCE" and any(f in unknown for f in PROTOCOL):
                ctype = HardCaseType.MISSING_CRITICAL_PROTOCOL
            else:
                continue
            da, db = describe(a), describe(b)
            pools[ctype].append((a.paper_id, HardCase(
                case_id=stable_id("hard", ctype.value, c["comparison_id"]), case_type=ctype, record_kind="experiment", record_a=a.experiment_id,
                record_b=b.experiment_id, lexical_similarity=lexical_similarity(da, db), scientific_relation=c["relationship_label"],
                matching_fields=same, differing_fields=diff, unknown_fields=unknown,
                rationale=f"From comparison {c['comparison_id']} ({c['relationship_label']}); '{da}' vs '{db}'.",
                evidence_ids=sorted({i for f in fcs for i in f.get("evidence_a", []) + f.get("evidence_b", [])}),
                generator_version=GENERATOR_VERSION, split=_split((a, b)), provenance=self.prov)))
        return [h for pool in pools.values() for h in _round_robin(pool, 300)]
