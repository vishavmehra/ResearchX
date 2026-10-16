"""Canonical entities, knowledge graph provenance and research-task behaviour rules (no network)."""

from services.canonicalization import entities
from services.datasets import graph, tasks
from services.datasets.schemas import (
    ExpectedBehavior,
    Experiment,
    ExperimentType,
    ExtractionMethod,
    FieldStatus,
    FieldValue,
    GraphEdgeType,
    HardCaseType,
    OutcomeObservation,
    Provenance,
    ResearchTaskType,
)


def known(v, ev="ev:x"):
    return FieldValue(status=FieldStatus.KNOWN, raw_value=str(v), value=v, evidence_ids=[ev])


def exp(eid: str, paper: str, model: str, dataset: str, metric: str = "Accuracy") -> Experiment:
    e = Experiment(experiment_id=eid, study_id=f"study:{paper}", paper_id=f"paper:{paper}", source_record_ids=[], experiment_type=ExperimentType.MAIN,
                   extraction_method=ExtractionMethod.LLM, extraction_run_id="r", provenance=Provenance(source_name="RESEARCHX", processing_version="t"),
                   outcomes=[OutcomeObservation(metric=known(metric, f"ev:{eid}:m"), score=known(80.0, f"ev:{eid}:s"))])
    e.model.model_name = known(model, f"ev:{eid}:model")
    e.dataset.dataset_name = known(dataset, f"ev:{eid}:data")
    e.evaluation.benchmark = known(dataset, f"ev:{eid}:bench")
    return e


def comparison(cid, a, b, label, statuses):
    return {"comparison_id": cid, "experiment_a": a, "experiment_b": b, "relationship_label": label,
            "field_comparisons": [{"field_name": f, "comparison_status": s, "evidence_a": [], "evidence_b": []} for f, s in statuses.items()]}


def test_entity_keys_are_conservative():
    assert entities.key("BERT-base") == entities.key("bert base")
    assert entities.key("ImageNet") != entities.key("ImageNet-21k")
    assert entities.key("Qwen1.5-7B") == "qwen1.57b" and entities.key("Acc.") != entities.key("Accuracy")
    ents = entities.build([exp("e1", "p1", "BERT-base", "GLUE"), exp("e2", "p2", "bert base", "GLUE")])
    (bert,) = [x for x in ents if x.entity_type.value == "MODEL"]
    assert bert.preferred_name == "BERT-base" and sum(s["count"] for s in bert.surface_forms) == 2


def test_graph_edges_carry_provenance_and_skip_unknown_associations():
    e = exp("e1", "p1", "BERT-base", "GLUE")
    results = {"res:1": {"result_id": "res:1", "metric": "Accuracy", "raw_value": "80.0"}, "res:2": {"result_id": "res:2", "metric": "F1", "raw_value": "1"}}
    assocs = [{"association_id": "as:1", "result_id": "res:1", "experiment_id": "e1", "status": "LIKELY", "evidence_ids": ["ev:cell"]},
              {"association_id": "as:2", "result_id": "res:2", "status": "UNKNOWN"}]
    g = graph.build({}, {}, [e], results, assocs, {}, [], [])
    by_type = {}
    for edge in g.edges.values():
        by_type.setdefault(edge.edge_type, []).append(edge)
    (uses,) = by_type[GraphEdgeType.EXPERIMENT_USES_DATASET]
    assert "ev:e1:data" in uses.derived_from
    (prod,) = by_type[GraphEdgeType.EXPERIMENT_PRODUCES_RESULT]  # UNKNOWN association produced no edge
    assert prod.derived_from == ["as:1", "ev:cell"]


def test_task_behaviour_rules():
    a, b, c = exp("e1", "p1", "BERT", "GLUE"), exp("e2", "p2", "RoBERTa", "GLUE"), exp("e3", "p1", "BERT-large", "GLUE")
    proto_unknown = {"evaluation.benchmark": "IDENTICAL", "evaluation.dataset_split": "UNKNOWN", "model.model_name": "MATERIALLY_DIFFERENT"}
    comps = [comparison("cmp:1", "e1", "e2", "INSUFFICIENT_EVIDENCE", proto_unknown),
             comparison("cmp:2", "e1", "e3", "MATERIALLY_DIFFERENT", {"model.model_name": "MATERIALLY_DIFFERENT"})]
    with_text = tasks.Builder([a, b, c], {"paper:p1", "paper:p2"}).comparison_tasks(comps)
    by_type = {t.task_type: t for t in with_text}
    assert by_type[ResearchTaskType.benchmark_comparison].expected_behavior is ExpectedBehavior.ABSTAIN
    assert by_type[ResearchTaskType.duplicate_experiment_detection].expected_answer.startswith("No")
    no_text = tasks.Builder([a, b, c], {"paper:p1"}).comparison_tasks(comps[:1])
    assert no_text[0].expected_behavior is ExpectedBehavior.SEARCH_MORE  # one paper never parsed: search, do not abstain
    (case,) = tasks.Builder([a, b, c], set()).hard_cases(comps[:1])
    assert case.case_type is HardCaseType.MISSING_CRITICAL_PROTOCOL and 0 <= case.lexical_similarity <= 1


def test_round_robin_caps_and_diversifies():
    items = [("p1", i) for i in range(5)] + [("p2", 10 + i) for i in range(5)]
    assert tasks._round_robin(items, cap=4) == [0, 10, 1, 11]
