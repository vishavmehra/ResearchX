"""Configuration-level experiment identity (services/analysis/configs.py) and its use by the analyses, over the synthetic
corpus of tests/test_analysis.py: pre-training, fine-tuning and evaluation kept apart, several configurations of each
in one paper, stage-wise comparison, linked evaluation conditions, configuration conditions in the Benchmark Reality
Explorer, and backward compatibility of all eight analyses. Invented names only; no network."""

from services.analysis import benchmark as bench
from services.analysis import claims, configs as C, discrepancy, papers, planner, recipe, review, transfer
from services.canonicalization.entities import entity_id
from services.datasets.schemas import CanonicalEntityType as E
from tests.test_analysis import build
from tests.test_knowledge_graph import FakeTracer, exp_record

P1, P2, P3 = "paper:arxiv:9901.00001", "paper:arxiv:9901.00002", "paper:arxiv:9901.00003"


def pe(i, pid, field, value, text, section=None, applies_to=None):
    return (f"ev:pe{i}", pid, field, {"extracted_value": value, "evidence_text": text, "section": section, "applies_to": applies_to})


def staged_corpus():
    """Paper 2 pre-trains on two datasets and fine-tunes on two; its two experiments start from different pre-trainings."""
    extra = {"exp:b0": exp_record("exp:b0", P2, "Orion-3", "Kappa-Bench", "Glyph Sorting", "Q-Score", 60.0, training__learning_rate=0.01,
                                  model__pretrained_source="ImageNet-1K"),
             "exp:b1": exp_record("exp:b1", P2, "Orion-3", "Kappa-Bench", "Glyph Sorting", "Q-Score", 61.0, training__learning_rate=0.01,
                                  model__pretrained_source="ImageNet-21K")}
    rows = [pe(1, P2, "training.learning_rate", "1.5e-4", "We pre-train on ImageNet-21K with a learning rate of 1.5e-4.", "3.1 Pre-training"),
            pe(2, P2, "training.epochs", "90", "We pre-train on ImageNet-21K for 90 epochs."),
            pe(3, P2, "training.epochs", "800", "Pre-training on ImageNet-1K runs for 800 epochs."),
            pe(4, P2, "training.batch_size", "1024", "We fine-tune on Kappa-Bench with a batch size of 1024.", "4 Fine-tuning", "Kappa-Bench"),
            pe(5, P2, "training.batch_size", "512", "We fine-tune on Omega-Set with a batch size of 512.", "4 Fine-tuning", "Omega-Set"),
            pe(6, P2, "training.seed", "0", "All fine-tuning runs use random seed 0.", "4 Fine-tuning")]
    return build(extra, paper_evidence=rows)


def test_one_paper_holds_several_pretraining_and_finetuning_configurations_without_conflating_them():
    idx, _, _ = staged_corpus()
    models = idx.experiments(["exp:b0", "exp:b1"])
    views = C.views(idx, models)
    inv = C.inventory(list(views.values()))
    assert inv["multiple"] == {"pretraining": True, "finetuning": True}
    pre = {c["scope"]: {f["field"]: [v["raw"] for v in f["values"]] for f in c["fields"]} for c in inv["pretraining"]}
    assert pre["ImageNet-21K"]["training.epochs"] == ["90"] and pre["ImageNet-1K"]["training.epochs"] == ["800"], "each pre-training keeps its own epochs"
    assert pre["ImageNet-21K"]["training.learning_rate"] == ["1.5e-4"] and "training.learning_rate" not in pre["ImageNet-1K"]
    kappa = next(c for c in inv["finetuning"] if c["scope"] == "Kappa-Bench")
    assert {f["field"]: f["values"][0]["raw"] for f in kappa["fields"]} == {"training.batch_size": "1024", "training.seed": "0"}, \
        "the dataset-free seed statement holds for the fine-tuning on Kappa-Bench; Omega-Set's batch size does not"
    assert any("Omega-Set" in u["label"] for u in inv["not_linked"]), "the other fine-tuning configuration is listed, not merged"
    rel = {x: {r["relation"]: r for r in v["relations"]} for x, v in views.items()}
    assert rel["exp:b0"]["starts from"]["to"].endswith(":pretraining:imagenet1k") and rel["exp:b1"]["starts from"]["to"].endswith(":pretraining:imagenet21k")
    assert rel["exp:b0"]["evaluated after"]["to"].endswith(":finetuning:kappabench") and "names Kappa-Bench" in rel["exp:b0"]["evaluated after"]["basis"]
    assert all(r["status"] == "DERIVED" for v in views.values() for r in v["relations"] if r["relation"] in ("starts from", "evaluated after"))


def test_pretraining_values_are_never_attached_to_the_evaluated_experiment():
    idx, _, _ = staged_corpus()
    e = idx.experiments(["exp:b0"])["exp:b0"]
    assert e.training.epochs.status.value == "UNKNOWN", "800 and 90 are pre-training epochs: not this evaluation's training length"
    assert e.training.batch_size.raw_value == "1024" and "Kappa-Bench" in e.training.batch_size.derivation, "the fine-tuning value aimed at its dataset"
    assert e.training.learning_rate.value == 0.01, "the record's own value is never replaced"


def test_paper_diff_rows_compare_training_settings_stage_by_stage():
    idx, _, _ = staged_corpus()
    d = papers.diff(idx, [P1, P2], FakeTracer())
    rows = {(r["field"], r.get("stage")): r for s in d["sections"] for r in s["rows"]}
    epochs = rows[("training.epochs", "pretraining")]
    assert epochs["label"] == "epochs (pre-training)" and epochs["status"] == "PARTIAL"
    assert {v["value"] for v in epochs["cells"][1]["values"]} == {"90", "800"} and epochs["cells"][1]["status"] == "VARIES"
    assert ("training.epochs", None) not in rows and ("training.epochs", "unspecified") not in rows
    assert rows[("training.learning_rate", "pretraining")]["cells"][1]["values"][0]["value"] == "1.5e-4"
    assert rows[("training.learning_rate", "unspecified")]["cells"][1]["values"][0]["value"] in ("0.01", 0.01), "the records' own unstaged value"
    p2 = d["papers"][1]
    assert p2["configurations"]["multiple"]["pretraining"] and len(p2["configurations"]["pretraining"]) == 2


def test_discrepancy_never_calls_a_pretraining_value_a_difference_from_a_finetuning_one():
    text = {"ev:exp:a0learning_rate": {"evidence_text": "We fine-tune with a learning rate of 0.001.", "section": "4 Fine-tuning"},
            "ev:exp:b0learning_rate": {"evidence_text": "During pre-training we use a learning rate of 0.01.", "section": "3 Pre-training"}}
    extra = {"exp:a0": exp_record("exp:a0", P1, "Zeta-9", "Kappa-Bench", "Glyph Sorting", "Q-Score", 70.0, evaluation__dataset_split="test",
                                  evaluation__metric_definition="top-1", evaluation__evaluation_protocol="single crop", training__learning_rate=0.001)}
    idx, _, _ = build(extra, evidence_text=text)
    r = discrepancy.investigate(idx, "exp:a0", "exp:b0", FakeTracer())
    lr = [u for u in r["unknown"] if u["field"] == "training.learning_rate"]
    assert lr and all(u["stage_mismatch"] for u in lr), "fine-tuning 0.001 vs pre-training 0.01: not compared"
    assert not any(h["field"] == "training.learning_rate" for h in r["hypotheses"])
    assert any("different training stages" in f for f in r["findings"])
    text["ev:exp:b0learning_rate"] = {"evidence_text": "We fine-tune with a learning rate of 0.01.", "section": "4 Fine-tuning"}
    idx, _, _ = build(extra, evidence_text=text)
    r = discrepancy.investigate(idx, "exp:a0", "exp:b0", FakeTracer())
    h = [h for h in r["hypotheses"] if h["field"] == "training.learning_rate"]
    assert h and h[0]["factor"] == "learning rate (fine-tuning / adaptation)" and "may contribute" in h[0]["statement"]


def test_paper_level_conditions_are_linked_only_when_they_name_the_experiments_dataset():
    rows = [("ev:c1", P2, "evaluation.dataset_split", {"extracted_value": "validation set", "evidence_text": "All results are reported on the Kappa-Bench validation set."}),
            ("ev:c2", P2, "evaluation.evaluation_protocol", {"extracted_value": "linear probe", "evidence_text": "linear probe"}),
            ("ev:c3", P2, "evaluation.metric_definition", {"extracted_value": "#Params(GB)", "evidence_text": "Kappa-Bench #Params(GB)"}),
            ("ev:c4", P2, "dataset.dataset_version", {"extracted_value": "Kappa-Bench v2", "evidence_text": "We use Kappa-Bench v2 throughout."})]
    idx, _, _ = build(paper_evidence=rows)
    e = idx.experiments(["exp:b0"])["exp:b0"]
    assert (e.evaluation.dataset_split.raw_value, e.evaluation.dataset_split.value, e.evaluation.dataset_split.derived) == ("validation set", "validation", True)
    assert "linked to this experiment by its dataset name" in e.evaluation.dataset_split.derivation
    assert e.evaluation.evaluation_protocol.status.value == "UNKNOWN", "names no dataset: stays paper-level"
    assert e.evaluation.metric_definition.status.value == "UNKNOWN", "not a recognisable metric definition: never linked"
    assert e.dataset.dataset_version.raw_value == "Kappa-Bench v2"
    two = rows + [("ev:c5", P2, "evaluation.evaluation_protocol", {"extracted_value": "single-scale", "evidence_text": "Kappa-Bench, single-scale testing"}),
                  ("ev:c6", P2, "evaluation.evaluation_protocol", {"extracted_value": "multi-scale", "evidence_text": "Kappa-Bench, multi-scale testing"})]
    idx, _, _ = build(paper_evidence=two)
    assert idx.experiments(["exp:b0"])["exp:b0"].evaluation.evaluation_protocol.status.value == "UNKNOWN", "two protocols for one dataset: ambiguous"


def test_benchmark_conditions_include_pretraining_data_and_never_match_unverified_values():
    def rec(xid, model, score, **kw):
        return exp_record(xid, P3, model, "Kappa-Bench", "Glyph Sorting", "Q-Score", score, evaluation__evaluation_protocol="single crop", **kw)
    extra = {"exp:c0": rec("exp:c0", "Lyra-5", 90.0, evaluation__dataset_split="test", model__pretrained_source="JFT-300M"),
             "exp:c1": rec("exp:c1", "Lyra-6", 91.0, evaluation__dataset_split="test", model__pretrained_source="ImageNet-21K"),
             "exp:c2": rec("exp:c2", "Lyra-7", 92.0, evaluation__dataset_split="#Params(GB)")}
    idx, _, n = build(extra)
    b, m = entity_id(E.BENCHMARK, n["bench"]), entity_id(E.METRIC, n["metric"])
    r = bench.explore(idx, b, m, FakeTracer())
    rows = {row["experiment"]: row for row in r["rows"]}
    assert rows["exp:c0"]["conditions"]["pretraining_data"]["value"] == "JFT-300M"
    assert rows["exp:c2"]["conditions"]["split"]["status"] == "UNVERIFIED" and rows["exp:c2"]["class"] != "COMPARABLE"
    ref = rows["exp:c0"]["result"]
    r2 = {row["experiment"]: row for row in bench.explore(idx, b, m, FakeTracer(), reference=ref)["rows"]}
    assert r2["exp:c1"]["class"] == "INCOMPATIBLE" and "pre-training data" in r2["exp:c1"]["why"], "same split and protocol, other pre-training data"
    assert r2["exp:a0"]["class"] == "CONDITIONALLY_COMPARABLE" and "pre-training data" in r2["exp:a0"]["why"], "unstated pre-training data cannot match"
    assert "rank" not in r2["exp:c1"] and "rank" not in r2["exp:a0"]


def test_split_spellings_are_equivalent_for_comparability():
    extra = {"exp:c0": exp_record("exp:c0", P3, "Lyra-5", "Kappa-Bench", "Glyph Sorting", "Q-Score", 80.0, evaluation__dataset_split="val",
                                  evaluation__evaluation_protocol="single-crop"),
             "exp:c1": exp_record("exp:c1", P3, "Lyra-6", "Kappa-Bench", "Glyph Sorting", "Q-Score", 81.0, evaluation__dataset_split="validation set",
                                  evaluation__evaluation_protocol="single crop")}
    idx, _, n = build(extra)
    b, m = entity_id(E.BENCHMARK, n["bench"]), entity_id(E.METRIC, n["metric"])
    ref = next(row["result"] for row in bench.explore(idx, b, m, FakeTracer())["rows"] if row["experiment"] == "exp:c0")
    rows = {row["experiment"]: row for row in bench.explore(idx, b, m, FakeTracer(), reference=ref)["rows"]}
    assert rows["exp:c1"]["class"] == "COMPARABLE" and rows["exp:a0"]["class"] == "INCOMPATIBLE", "val = validation set; test differs"


def test_recipe_tags_each_value_with_its_stage_and_lists_the_lineage():
    idx, _, _ = staged_corpus()
    r = recipe.build(idx, "exp:b0", "0.2.0", FakeTracer())
    stages = {c["stage"]: c for c in r["configurations"]}
    assert set(stages) == {"pretraining", "finetuning"} and stages["finetuning"]["link"]
    assert {f["field"] for f in stages["pretraining"]["fields"]} >= {"training.epochs", "model.pretrained_source"}
    assert "optimizer" in r["missing_by_stage"]["pre-training"] and "random seed" not in r["missing_by_stage"]["fine-tuning / adaptation"]
    facts = {f["field"]: f for s in r["sections"] for f in s["facts"]}
    assert facts["model.pretrained_source"]["stage"] == "pretraining"
    assert "## Configurations in this experiment's lineage" in r["markdown"] and r["export"]["configurations"]


def test_one_paper_inventory_answers_which_seeds_splits_and_protocols_were_used():
    idx, _, _ = staged_corpus()
    r = papers.inventory_view(idx, P2, FakeTracer())
    assert r["kind"] == "paper_configurations" and r["summary"]["multiple_pretraining"] and r["summary"]["multiple_finetuning"]
    row = next(x for x in r["experiments"] if x["experiment"] == "exp:b0")
    assert row["seed"]["values"][0]["raw"] == "0" and row["seed"]["values"][0]["evidence"]
    assert row["conditions"]["evaluation.dataset_split"]["status"] == "UNKNOWN"
    assert {u["label"] for u in r["unknown"]} >= {"evaluation split", "evaluation protocol"}
    a = papers.inventory_view(idx, P1, FakeTracer())
    assert a["experiments"][0]["conditions"]["evaluation.dataset_split"]["values"][0]["raw"] == "test"


def test_all_eight_analyses_still_run_on_existing_records():
    idx, _, n = build()
    tr = FakeTracer()
    b, m = entity_id(E.BENCHMARK, n["bench"]), entity_id(E.METRIC, n["metric"])
    kinds = [papers.diff(idx, [P1, P2], tr)["kind"], discrepancy.investigate(idx, "exp:a0", "exp:b0", tr)["kind"],
             recipe.build(idx, "exp:a0", "0.2.0", tr)["kind"], planner.plan(idx, "exp:a0", tr)["kind"],
             claims.audit(idx, f"{n['model_a']} outperforms {n['model_b']} on {n['bench']}", tr)["kind"],
             transfer.analyze(idx, n["model_a"], tr)["kind"], review.audit(idx, f"We study {n['model_a']} on {n['bench']}.", tr)["kind"],
             bench.explore(idx, b, m, tr)["kind"]]
    assert kinds == ["paper_diff", "discrepancy", "recipe", "plan", "claim", "transfer", "review", "benchmark"]
