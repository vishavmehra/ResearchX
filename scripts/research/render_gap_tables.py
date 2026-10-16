"""Fill the generated tables inside data/reports/dataset_gap_analysis.md.

The narrative of the gap report is written by hand; every number in it comes from here:
- field coverage per experiment group, read from an audit JSON (default data/reports/dataset_audit.json)
- the deepening-program field list (§6) mapped onto the schema, checked against the live schema module
Blocks are delimited by <!-- BEGIN generated:NAME --> / <!-- END generated:NAME -->.

Usage: uv run python scripts/research/render_gap_tables.py [--audit data/reports/dataset_audit.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.datasets import schemas  # noqa: E402
from services.datasets.store import DATA, ROOT, atomic_write_bytes, utf8_stdio  # noqa: E402

REPORT = DATA / "reports" / "dataset_gap_analysis.md"

# Deepening program §6 field -> schema field path (planned path for fields added in schema 0.2.0).
# Paths that start with "experiment." are top-level Experiment attributes.
SPEC6: dict[str, list[tuple[str, str]]] = {
    "Identity": [("experiment_id", "experiment.experiment_id"), ("study_id", "experiment.study_id"), ("source_record_ids", "experiment.source_record_ids"),
                 ("experiment_type", "experiment.experiment_type"), ("parent_experiment_id", "experiment.parent_experiment_id")],
    "Research question": [(f, f"research.{f}") for f in ("research_question", "hypothesis", "task", "subtask", "domain")],
    "Dataset": [("dataset_name", "dataset.dataset_name"), ("dataset_version", "dataset.dataset_version"), ("source", "dataset.dataset_source"),
                ("split", "dataset.split"), ("train_split", "dataset.train_data"), ("validation_split", "dataset.validation_data"),
                ("test_split", "dataset.test_data"), ("sampling", "dataset.sampling_method"), ("sample_count", "dataset.sample_count"),
                ("filtering", "dataset.filtering"), ("preprocessing", "dataset.preprocessing"), ("augmentation", "dataset.augmentation"),
                ("deduplication", "dataset.deduplication"), ("contamination_controls", "dataset.data_leakage_controls"),
                ("train/test construction", "dataset.split_construction")],
    "Model": [("model_family", "model.model_family"), ("model_name", "model.model_name"), ("version", "model.model_version"),
              ("checkpoint", "model.checkpoint"), ("parameter_count", "model.parameter_count"), ("active_parameter_count", "model.active_parameter_count"),
              ("architecture", "model.architecture"), ("initialization", "model.initialization"), ("pretrained_model", "model.base_model"),
              ("tokenizer", "model.tokenizer"), ("context_length", "model.context_length")],
    "Method": [("method_name", "method.method_name"), ("algorithm", "method.algorithm"), ("objective", "method.objective"),
               ("loss", "method.loss_function"), ("baseline", "method.baseline"), ("ablation", "method.ablation"), ("modifications", "method.modifications")],
    "Training": [("optimizer", "training.optimizer"), ("learning_rate", "training.learning_rate"), ("scheduler", "training.learning_rate_schedule"),
                 ("batch_size", "training.batch_size"), ("gradient_accumulation", "training.gradient_accumulation"), ("epochs", "training.epochs"),
                 ("steps", "training.steps"), ("warmup", "training.warmup_steps"), ("weight_decay", "training.weight_decay"),
                 ("dropout", "training.dropout"), ("seed", "training.seed"), ("precision", "training.precision"), ("quantization", "training.quantization"),
                 ("gradient_clipping", "training.gradient_clipping"), ("LoRA", "training.lora"), ("QLoRA", "training.qlora"), ("DoRA", "training.dora"),
                 ("rank", "training.peft_rank"), ("alpha", "training.peft_alpha"), ("target_modules", "training.peft_target_modules")],
    "Retrieval / RAG": [("retriever", "retrieval.retriever"), ("retrieval_model", "retrieval.retrieval_model"), ("embedding_model", "retrieval.embedding_model"),
                        ("top_k", "retrieval.retrieval_top_k"), ("chunk_size", "retrieval.chunk_size"), ("chunk_overlap", "retrieval.chunk_overlap"),
                        ("index_type", "retrieval.index_type"), ("reranker", "retrieval.reranker"), ("reranking", "retrieval.reranking"),
                        ("corpus", "retrieval.retrieval_corpus"), ("retrieval_prompt", "retrieval.retrieval_prompt"), ("retrieval_strategy", "retrieval.retrieval_strategy")],
    "Prompting / agent": [("system_prompt", "inference.system_prompt"), ("prompt_template", "prompting.prompt_template"),
                          ("few_shot_examples", "prompting.few_shot_examples"), ("chain_of_thought_usage", "prompting.chain_of_thought"),
                          ("tool_usage", "prompting.tool_usage"), ("agent_framework", "prompting.agent_framework"), ("planner", "prompting.planner"),
                          ("memory", "prompting.memory"), ("number_of_steps", "prompting.number_of_steps"), ("stopping_condition", "prompting.stopping_condition")],
    "Inference": [("temperature", "inference.temperature"), ("top_p", "inference.top_p"), ("top_k", "inference.top_k"), ("max_tokens", "inference.max_tokens"),
                  ("decoding_strategy", "inference.decoding_strategy"), ("context_window", "inference.context_length"), ("batching", "inference.batching"),
                  ("hardware", "inference.hardware")],
    "Evaluation": [("benchmark", "evaluation.benchmark"), ("metric", "outcome.metric"), ("metric_definition", "evaluation.metric_definition"),
                   ("score", "outcome.score"), ("uncertainty", "outcome.uncertainty"), ("confidence_interval", "outcome.confidence_interval"),
                   ("statistical_test", "evaluation.statistical_test"), ("p_value", "outcome.p_value"), ("significance", "outcome.significance"),
                   ("evaluation_protocol", "evaluation.evaluation_protocol"), ("evaluation_population", "evaluation.evaluation_population"),
                   ("aggregation_method", "evaluation.aggregation_method")],
    "Compute": [(f, f"compute.{f}") for f in ("gpu_model", "gpu_count", "cpu", "memory", "training_time", "inference_time", "flops", "energy", "cost")],
    "Artifacts": [("code_repository", "artifacts.repository_url"), ("commit", "artifacts.commit"), ("release", "artifacts.release"),
                  ("checkpoint", "artifacts.checkpoint_url"), ("model_card", "artifacts.model_card_url"), ("dataset_repository", "artifacts.dataset_url"),
                  ("configuration_file", "artifacts.configuration_file"), ("supplementary_material", "artifacts.supplementary_material_url"),
                  ("benchmark_submission", "artifacts.benchmark_submission_url"), ("experiment_logs", "artifacts.experiment_logs_url")],
}


def schema_has(path: str) -> bool:
    group, name = path.split(".", 1)
    if group == "experiment":
        return name in schemas.Experiment.model_fields
    if group == "outcome":
        return name in schemas.OUTCOME_FIELDS
    return name in schemas.FIELD_GROUPS.get(group, ())


def coverage_table(audit: dict) -> str:
    cov = audit["experiments"]["field_coverage"]
    n_exp, n_out = audit["experiments"]["total"], audit["experiments"]["outcomes"]
    lines = [f"Source: `{audit['audit_version']}` audit generated {audit['generated_at']} over {n_exp} experiments and {n_out} outcomes.", ""]
    groups: dict[str, list[str]] = {}
    for path in cov:
        groups.setdefault(path.split(".", 1)[0], []).append(path)
    lines += ["| Group | Fields | Fields ever KNOWN | Known fields (count, % of records, of which with paper-text evidence) |", "|---|---|---|---|"]
    for g, paths in groups.items():
        known = [p for p in paths if cov[p]["known"]]
        detail = "; ".join(f"{p.split('.', 1)[1]} {cov[p]['known']} ({cov[p]['known_pct']}%, text {cov[p]['known_with_paper_text_evidence']})" for p in known) or "none"
        lines.append(f"| {g} | {len(paths)} | {len(known)} | {detail} |")
    return "\n".join(lines)


def schema_gap_table() -> str:
    lines = ["| §6 group | Field | Schema path | In current schema |", "|---|---|---|---|"]
    missing = present = 0
    for group, fields in SPEC6.items():
        for label, path in fields:
            ok = schema_has(path)
            present += ok
            missing += not ok
            lines.append(f"| {group} | {label} | `{path}` | {'yes' if ok else '**MISSING**'} |")
    lines.insert(0, f"Schema version `{schemas.SCHEMA_VERSION}`: {present} of {present + missing} deepening-program fields present, {missing} missing.\n")
    return "\n".join(lines)


def fill(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN generated:{name} -->).*?(<!-- END generated:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker generated:{name} not found in {REPORT}")
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text)


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(DATA / "reports" / "dataset_audit.json"))
    args = ap.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    text = REPORT.read_text(encoding="utf-8")
    text = fill(text, "field-coverage", coverage_table(audit))
    text = fill(text, "schema-gap", schema_gap_table())
    atomic_write_bytes(REPORT, text.encode("utf-8"))
    print(f"filled generated tables in {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
