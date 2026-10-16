"""Reproducibility Recipe Builder: one experiment's configuration as a reproduction specification.

Every field keeps its status (REPORTED / DERIVED / CONFLICTING / UNKNOWN, services/analysis/facts.py) and its evidence.
No default is ever filled in: a setting the sources do not state stays UNKNOWN and is listed as missing. The export is
validated against RecipeExport before it is returned, so a malformed record cannot become a recipe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from services.analysis import configs as C
from services.analysis import facts as F
from services.datasets.schemas import FIELD_GROUPS

VERSION = "recipe-0.2.0"  # 0.2.0: each value's training stage, the pre-training / fine-tuning configurations of the lineage
SECTIONS = (
    ("Model", ("model.model_name", "model.model_version", "model.model_family", "model.architecture", "model.parameter_count", "model.checkpoint",
               "model.base_model", "model.pretrained_source", "model.initialization", "model.tokenizer", "model.context_length")),
    ("Dataset", ("evaluation.benchmark", "dataset.dataset_name", "dataset.dataset_version", "dataset.split", "dataset.train_data",
                 "dataset.validation_data", "dataset.test_data", "dataset.sample_count", "dataset.split_construction", "dataset.dataset_source")),
    ("Preprocessing", ("dataset.preprocessing", "dataset.filtering", "dataset.augmentation", "dataset.deduplication", "dataset.data_leakage_controls",
                       "dataset.sampling_method")),
    ("Method", ("method.method_name", "method.algorithm", "method.objective", "method.loss_function", "method.training_strategy", "method.modifications")),
    ("Training", ("training.optimizer", "training.learning_rate", "training.learning_rate_schedule", "training.batch_size", "training.gradient_accumulation",
                  "training.epochs", "training.steps", "training.warmup_steps", "training.warmup_ratio", "training.weight_decay", "training.dropout",
                  "training.gradient_clipping", "training.precision", "training.quantization", "training.lora", "training.peft_rank",
                  "training.peft_alpha", "training.peft_target_modules")),
    ("Randomness", ("training.seed",)),
    ("Evaluation", ("evaluation.dataset_split", "evaluation.metric_definition", "evaluation.evaluation_protocol", "evaluation.statistical_test",
                    "evaluation.aggregation_method", "evaluation.evaluation_population")),
    ("Baselines", ("method.baseline", "method.ablation")),
    ("Inference", ("inference.decoding_strategy", "inference.temperature", "inference.top_p", "inference.max_tokens", "inference.context_length",
                   "inference.batching")),
    ("Hardware and software", ("compute.accelerator", "compute.gpu_model", "compute.gpu_count", "compute.memory", "compute.training_time",
                               "inference.hardware", "artifacts.configuration_file")),
    ("Code and checkpoints", ("artifacts.repository_url", "artifacts.commit", "artifacts.release", "artifacts.checkpoint_url",
                              "artifacts.model_card_url", "artifacts.dataset_url", "artifacts.experiment_logs_url")),
)
OPTIONAL = (("Retrieval", "retrieval"), ("Prompting", "prompting"))  # shown only when the experiment states one of their fields
# Without these a reproduction cannot even start; each one the sources leave UNKNOWN is listed as missing.
CRITICAL = ("model.model_name", "evaluation.benchmark", "evaluation.dataset_split", "evaluation.metric_definition", "training.optimizer",
            "training.learning_rate", "training.batch_size", "training.seed", "artifacts.repository_url")
STAGE_CRITICAL = ("training.optimizer", "training.learning_rate", "training.batch_size", "training.seed")  # per linked training stage
Status = Literal["REPORTED", "DERIVED", "CONFLICTING", "UNKNOWN", "NOT_APPLICABLE"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRef(_Strict):
    id: str
    location: str | None
    section: str | None = None
    table: str | None = None
    cell: str | None = None
    text: str | None  # None when the source's licence withholds redistribution


class RecipeField(_Strict):
    field: str
    label: str
    status: Status
    value: Any = None
    unit: str | None = None
    derivation: str | None = None
    stage: str | None = None  # pretraining | finetuning | evaluation | unspecified (services/analysis/configs.py)
    evidence: list[EvidenceRef]


class RecipeConfiguration(_Strict):
    id: str
    stage: str
    label: str
    link: str | None  # why this configuration belongs to the experiment's lineage (a derived rule); None for the experiment's own
    fields: list[dict[str, Any]]


class RecipeResult(_Strict):
    metric: RecipeField
    score: RecipeField
    split: RecipeField


class RecipeExport(_Strict):
    format: Literal["researchx-recipe"] = "researchx-recipe"
    version: str
    generated_at: str
    corpus_release: str
    experiment_id: str
    paper_id: str
    paper_title: str | None
    paper_url: str | None
    quality_tier: str
    reviewer_status: str
    extraction_method: str
    fields: list[RecipeField]
    results: list[RecipeResult]
    configurations: list[RecipeConfiguration] = []
    missing: list[str]
    missing_by_stage: dict[str, list[str]] = {}
    conflicting: list[str]


def _ref(e: dict) -> dict:
    return {"id": e["id"], "location": e.get("source_location"), "section": e.get("section"), "table": e.get("table"), "cell": e.get("cell"),
            "text": e.get("text")}


def _field(f: dict) -> dict:
    return {k: f.get(k) for k in ("field", "label", "status", "value", "unit", "derivation", "stage")} | {"evidence": [_ref(e) for e in f.get("evidence", [])]}


def _shown(f: dict) -> str:
    v = f["raw"] if f["raw"] not in (None, "") else f["value"]
    return f"{v}{f' {f['unit']}' if f['unit'] and f['unit'] not in str(v) else ''}"


def markdown(r: dict) -> str:
    """The recipe as a readable checklist: [x] for a stated value (with its status and source), [ ] for what is missing."""
    lines = [f"# Reproduction recipe: {r['experiment']['label']}", "", f"Paper: {r['paper']['title'] or r['paper']['id']}"
             + (f" ({r['paper']['url']})" if r['paper'].get('url') else ""),
             f"Source: ResearchX corpus {r['provenance']['corpus_release']}, {r['provenance']['quality_tier'].lower()} record, "
             f"{r['provenance']['reviewer_status'].lower()}. Values are as the sources state them; nothing is filled in.", ""]
    for s in r["sections"]:
        lines.append(f"## {s['name']}")
        for f in s["facts"]:
            if f["status"] in F.STATED:
                where = next((e.get("table") or e.get("section") or e.get("source_location") for e in f.get("evidence", []) if e), None)
                lines.append(f"- [x] {f['label']}: {_shown(f)} ({f['status'].lower()}{f'; {where}' if where else ''})")
            elif f["status"] == F.CONFLICTING:
                lines.append(f"- [ ] {f['label']}: sources disagree; check every cited source")
            elif f["status"] == F.UNKNOWN:
                lines.append(f"- [ ] {f['label']}: not stated in the sources read")
        lines.append("")
    if r["results"]:
        lines.append("## Reported results")
        lines += [f"- {x['metric']['value'] or 'metric unknown'} = {x['score']['raw'] or x['score']['value'] or 'score unknown'}"
                  + (f" ({x['split']['value']})" if x['split']['value'] else "") for x in r["results"]]
        lines.append("")
    staged = [c for c in r.get("configurations", []) if c["stage"] in ("pretraining", "finetuning")]
    if staged:
        lines.append("## Configurations in this experiment's lineage")
        for c in staged:
            lines.append(f"### {c['label']}" + (f" (linked: {c['link']})" if c.get("link") else ""))
            for f in c["fields"]:
                vals = "; ".join(str(v["raw"]) for v in f["values"])
                lines.append(f"- [x] {f['label']}: {vals} ({f['status'].lower()})" if f["status"] != "VARIES" else
                             f"- [ ] {f['label']}: several values stated ({vals}); which one applies is not stated")
        lines.append("")
    if r["missing"]:
        lines.append("## Missing before a reproduction can start")
        lines += [f"- {m}" for m in r["missing"]]
        lines += [f"- {m} ({stage})" for stage, ms in r.get("missing_by_stage", {}).items() for m in ms]
    return "\n".join(lines).rstrip() + "\n"


def build(index, xid: str, release: str, tr, fill=None) -> dict:
    """Recipe of corpus experiment `xid`. `index` is the Knowledge Map GraphIndex (validated records + evidence); `tr`
    records the stages (RunTracer: the Research Trace); `fill(models, tr)` reads the full text for settings still unknown
    (services/analysis/enrich.py)."""
    with tr.stage("experiment_retrieval", experiment=xid) as st:
        exp = index.experiments([xid]).get(xid)
        if exp is None:
            raise KeyError(xid)
        st.detail.update(paper=exp.paper_id, quality_tier=exp.quality_tier.value, reviewer_status=exp.reviewer_status.value,
                         source="validated corpus record (GraphIndex.experiments)")
    if fill and not xid.startswith("rt:"):
        fill({xid: exp}, tr)
    with tr.stage("field_facts", method=f"field facts + configuration identity ({C.VERSION})") as st:
        view = C.views(index, {xid: exp})[xid]
        stage = {f["field"]: c["stage"] for c in view["configurations"] for f in c["fields"]
                 if any(xid in v["experiments"] for v in f["values"])}  # the stage of each value the record itself states
        sections, every = [], []
        for name, paths in SECTIONS:
            fs = F.facts(exp, paths)
            sections.append({"name": name, "facts": fs})
            every += fs
        for name, group in OPTIONAL:
            fs = F.facts(exp, [f"{group}.{n}" for n in FIELD_GROUPS[group]])
            if any(f["status"] != F.UNKNOWN for f in fs):
                sections.append({"name": name, "facts": fs})
                every += fs
        for f in every:
            f["stage"] = stage.get(f["field"]) if f["status"] != F.UNKNOWN else None
        res = F.results(exp)
        links = {r["to"]: r["basis"] for r in view["relations"] if r["relation"] in ("evaluated after", "starts from")}
        configurations = [{"id": c["id"], "stage": c["stage"], "label": c["label"], "link": links.get(c["id"]),
                           "fields": [{"field": f["field"], "label": f["label"], "status": f["status"],
                                       "values": [{"raw": v["raw"], "value": v["value"], "status": v["status"], "evidence_ids": v["evidence_ids"][:2],
                                                   "basis": v["basis"]} for v in f["values"]]} for f in c["fields"]]}
                          for c in view["configurations"] if c["stage"] in ("pretraining", "finetuning")]
        F.with_evidence(every + [x[k] for x in res for k in ("metric", "score", "split")], index.evidence_detail)
        ids = [i for c in configurations for f in c["fields"] for v in f["values"] for i in v["evidence_ids"]]
        found = {e["id"]: e for e in index.evidence_detail(ids, limit=len(ids))} if ids else {}
        for c in configurations:
            for f in c["fields"]:
                for v in f["values"]:
                    v["evidence"] = [found[i] for i in v["evidence_ids"] if i in found]
        st.detail.update(fields=len(every), results=len(res), **{k.lower(): v for k, v in F.counts(every).items()},
                         evidence_records=sum(len(f["evidence"]) for f in every), configurations=len(configurations))
    by_path = {f["field"]: f for f in every}
    missing = [F.label(p) for p in CRITICAL if by_path[p]["status"] == F.UNKNOWN]
    missing_by_stage = {C.STAGE_LABEL[c["stage"]]: [F.label(p) for p in STAGE_CRITICAL if p not in {f["field"] for f in c["fields"]}]
                        for c in configurations}
    conflicting = [f["label"] for f in every if f["status"] == F.CONFLICTING]
    detail = index.node_detail(exp.paper_id) if exp.paper_id in index.paper else {}
    paper = {"id": exp.paper_id, "title": detail.get("title"), "url": detail.get("url"), "year": detail.get("year")}
    prov = {"corpus_release": release, "quality_tier": exp.quality_tier.value, "reviewer_status": exp.reviewer_status.value,
            "extraction_method": exp.extraction_method.value, "extraction_model": exp.extraction_model}
    export = RecipeExport.model_validate({
        "version": VERSION, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "corpus_release": release, "experiment_id": xid,
        "paper_id": exp.paper_id, "paper_title": paper["title"], "paper_url": paper["url"], "quality_tier": prov["quality_tier"],
        "reviewer_status": prov["reviewer_status"], "extraction_method": prov["extraction_method"],
        "fields": [_field(f) for f in every], "results": [{k: _field(x[k]) for k in ("metric", "score", "split")} for x in res],
        "configurations": [{**c, "fields": [{**f, "values": [{"raw": v["raw"], "status": v["status"], "evidence": [_ref(e) for e in v["evidence"]]}
                                                             for v in f["values"]]} for f in c["fields"]]} for c in configurations],
        "missing": missing, "missing_by_stage": missing_by_stage, "conflicting": conflicting}).model_dump(mode="json")
    out = {"kind": "recipe", "version": VERSION, "experiment": index.node(xid), "paper": paper, "sections": sections, "results": res,
           "summary": F.counts(every), "missing": missing, "missing_by_stage": missing_by_stage, "conflicting": conflicting, "provenance": prov,
           "configurations": configurations, "relations": view["relations"], "unlinked": view["unlinked"], "export": export}
    out["markdown"] = markdown(out)
    return out


def options(index, pid: str) -> list[dict]:
    """The paper's experiments, most completely described first (stated configuration fields), to choose a recipe from."""
    exps = index.experiments(index.paper_exps.get(pid, []))
    rows = []
    for xid, e in exps.items():
        stated = sum(1 for _, fv in e.iter_fields() if F.status_of(fv) in F.STATED)
        rows.append({**(index.node(xid) or {"id": xid, "label": xid}), "stated_fields": stated, "results": len(F.results(e))})
    return sorted(rows, key=lambda r: (-r["stated_fields"], r["label"]))
