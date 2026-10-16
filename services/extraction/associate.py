"""Experiment assembly and table-first experiment-result association (docs/DATASET_DEEPENING.md §5, §8).

Statuses (deepening §19), deterministic:
- VERIFIED: two independent sources agree - a PwC claim value equals (or is rounding/scale consistent with) the
  paper table cell at a row/column aligned with the claim's model and dataset (v0.1.0 cross-check FOUND_ALIGNED).
- CONFLICTING: aligned cell, different value beyond rounding (cross-check CONFLICT_CANDIDATE).
- LIKELY: a model-identified experiment maps to the cell's row and column, labels verified in the table, one
  source only.
- AMBIGUOUS: several experiments claim the same table row.
- UNKNOWN: not emitted for unmatched cells (they stay in the parsed document); never forced onto an experiment.
Settings attach by scope: PAPER -> every non-baseline experiment of the paper; MODEL -> experiments whose model/method
matches applies_to; TABLE -> experiments associated with that table; EXPERIMENT -> experiments whose benchmark or
description matches applies_to. Two different values competing for one field leave it UNKNOWN (listed in
unresolved_fields); a value that disagrees with an existing KNOWN value makes the field CONFLICTING.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from services.canonicalization.normalize import NORMALIZATION_VERSION, parse_number
from services.datasets.schemas import (
    GROUP_MODELS,
    AssociationStatus,
    EvidenceAssertion,
    EvidenceScope,
    EvidenceStatus,
    Experiment,
    ExperimentResultAssociation,
    ExperimentType,
    ExtractionMethod,
    FieldStatus,
    FieldValue,
    LocationKind,
    OutcomeObservation,
    Paper,
    ParsedDocument,
    Provenance,
    ResultRecord,
    ResultSourceKind,
    Table,
    TableCell,
)
from services.datasets.store import Run, code_version, stable_id
from services.datasets.validate import POSITIVE_FIELDS, UNIT_INTERVAL_FIELDS  # same numeric rules as release validation
from services.evidence.assertions import make
from services.extraction.deep import ExperimentSpec, _norm_label
from services.extraction.experiments import _coverage, _size_clash, _toks

ASSOC_VERSION = "associate-0.1.0"
SPLIT_WORD = re.compile(r"\b(dev|development|test|validation|val|held[- ]out)\b(?:\s*set)?", re.I)
NON_TRAINED = {ExperimentType.BASELINE}


@dataclass
class PaperOutput:
    experiments: list[Experiment] = field(default_factory=list)  # all experiments of the paper after assembly
    results: list[ResultRecord] = field(default_factory=list)
    associations: list[ExperimentResultAssociation] = field(default_factory=list)
    evidence: list[EvidenceAssertion] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def names_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    if _norm_label(a) == _norm_label(b):
        return True
    ta, tb = _toks(a), _toks(b)
    return _coverage(ta, tb) and not _size_clash(ta, tb)


def _known(exp: Experiment, path: str) -> FieldValue:
    g, n = path.split(".", 1)
    return getattr(getattr(exp, g), n)


def _set(exp: Experiment, path: str, fv: FieldValue) -> None:
    g, n = path.split(".", 1)
    setattr(getattr(exp, g), n, fv)


def _cell_evidence(doc: ParsedDocument, paper: Paper, t: Table, c: TableCell, field_name: str, value: Any, run: Run,
                   experiment_id: str | None, extractor: str) -> EvidenceAssertion:
    num = parse_number(str(value)) if value is not None and not isinstance(value, str) else None
    version = doc.source_record_id.rsplit(paper.arxiv_id or "", 1)[-1] or None
    sections = {s.section_id: s for s in doc.sections}
    return make(
        source_name="ARXIV", source_url=doc.source_url, content_hash=doc.source_sha256, source_record_id=doc.source_record_id,
        paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=experiment_id, field_name=field_name,
        extracted_value=c.text, normalized_value=value, normalization_rule=num[1] if num else None,
        normalization_version=NORMALIZATION_VERSION if num else None, evidence_text=c.text,
        source_location=f"{doc.source_url}#{t.table_id}", section=t.section_id, table=t.table_id, cell=c.cell_id,
        location_kind=LocationKind.PAPER_TABLE_CELL,
        is_appendix=bool(sections.get(t.section_id or "") and sections[t.section_id].is_appendix),
        scope=EvidenceScope.EXPERIMENT, source_version=version, retrieval_timestamp=doc.parsed_at,
        extraction_model=extractor, extraction_run_id=run.run_id, status=EvidenceStatus.MACHINE_EXTRACTED,
        redistribution_allowed=doc.license.redistribution_allowed,
    )


def _result(doc: ParsedDocument, paper: Paper, t: Table, c: TableCell, evidence_ids: list[str]) -> ResultRecord:
    split = next((m.group(1) for p in c.col_header_path for m in [SPLIT_WORD.search(p)] if m), None)
    return ResultRecord(
        result_id=stable_id("res", doc.source_record_id, c.cell_id), paper_id=paper.paper_id, study_id=paper.study_id,
        source_kind=ResultSourceKind.PAPER_TABLE, source_record_id=doc.source_record_id,
        model=" > ".join(c.row_header_path) or None, metric=c.col_header_path[-1] if c.col_header_path else None,
        dataset=" > ".join(c.col_header_path[:-1]) or None, dataset_split=split, raw_value=c.text, value=c.value, unit=c.unit,
        uncertainty=c.uncertainty, table=t.table_id, cell=c.cell_id, row_header_path=c.row_header_path, col_header_path=c.col_header_path,
        caption=t.caption[:500] or None, source_location=f"{doc.source_url}#{t.table_id}", evidence_ids=evidence_ids,
        extraction_version=ASSOC_VERSION, redistribution_allowed=doc.license.redistribution_allowed, split=paper.split,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=ASSOC_VERSION,
                              parent_record_ids=[doc.source_record_id]),
    )


def _assoc(result: ResultRecord, status: AssociationStatus, rationale: str, exp_id: str | None, candidates: list[str],
           matched: list[str], mismatched: list[str], evidence_ids: list[str]) -> ExperimentResultAssociation:
    return ExperimentResultAssociation(
        association_id=stable_id("assoc", result.result_id, exp_id, status.value), result_id=result.result_id,
        experiment_id=exp_id if status in (AssociationStatus.VERIFIED, AssociationStatus.LIKELY, AssociationStatus.CONFLICTING) else None,
        status=status, candidate_experiment_ids=sorted(set(candidates)), matched_fields=matched, mismatched_fields=mismatched,
        rationale=rationale, evidence_ids=evidence_ids, method_version=ASSOC_VERSION,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=ASSOC_VERSION,
                              parent_record_ids=[result.result_id]),
    )


def _find_cell(t: Table, loc: dict[str, Any]) -> TableCell | None:
    """Map a v0.1.0 cross-check location (legacy rows view) to the 0.2.0 cell id."""
    cands = [c for c in t.cells if c.spanned_from is None and c.text == loc.get("cell")]
    if len(cands) > 1:
        lab = _norm_label(loc.get("row_label") or "")
        narrowed = [c for c in cands if lab and any(_norm_label(x) == lab for x in c.row_header_path)]
        cands = narrowed or [c for c in cands if c.row == loc.get("row")] or cands
    return cands[0] if len(cands) == 1 else None


def associate_claims(exp: Experiment, doc: ParsedDocument, paper: Paper, run: Run, out: PaperOutput) -> None:
    """PwC claim vs paper table: VERIFIED / CONFLICTING cell associations from the v0.1.0 cross-check."""
    tables = {t.table_id: t for t in doc.tables}
    for i, cc in enumerate((exp.cross_check or {}).get("outcomes", [])):
        status = cc.get("status")
        loc = cc.get("location") or {}
        t = tables.get(loc.get("table_id"))
        if status not in ("FOUND_ALIGNED", "ROUNDING_CONSISTENT", "SCALE_CONSISTENT", "CONFLICT_CANDIDATE") or t is None:
            out.counts[f"claim_{status or 'none'}"] += 1
            continue
        c = _find_cell(t, loc)
        if c is None:
            out.counts["claim_cell_unmapped"] += 1
            continue
        ev = _cell_evidence(doc, paper, t, c, "evaluation.score", c.value, run, exp.experiment_id, f"{ASSOC_VERSION}/claim-cell")
        res = _result(doc, paper, t, c, [ev.evidence_id])
        claim_ids = exp.outcomes[i].score.evidence_ids if i < len(exp.outcomes) else []
        if status == "CONFLICT_CANDIDATE":
            a = _assoc(res, AssociationStatus.CONFLICTING, f"PwC claim value differs from paper cell {c.cell_id} ({cc.get('paper_value')})",
                       exp.experiment_id, [exp.experiment_id], ["research.task", "model.model_name", "evaluation.benchmark"], ["evaluation.score"],
                       [ev.evidence_id, *claim_ids])
        else:
            a = _assoc(res, AssociationStatus.VERIFIED, f"PwC claim and paper cell {c.cell_id} agree ({status})", exp.experiment_id,
                       [exp.experiment_id], ["model.model_name", "evaluation.benchmark", "evaluation.score"], [], [ev.evidence_id, *claim_ids])
            if i < len(exp.outcomes) and ev.evidence_id not in exp.outcomes[i].score.evidence_ids:
                exp.outcomes[i].score.evidence_ids.append(ev.evidence_id)
        out.evidence.append(ev)
        out.results.append(res)
        out.associations.append(a)
        out.counts[f"assoc_{a.status.value}"] += 1


def _new_experiment(sp: ExperimentSpec, paper: Paper, doc: ParsedDocument, run: Run) -> Experiment:
    eid = stable_id("exp", paper.paper_id, "llm", sp.local_id, sp.description, *(a.raw for a in sp.identity.values()))
    groups: dict[str, dict[str, FieldValue]] = defaultdict(dict)
    for path, a in sp.identity.items():
        g, n = path.split(".", 1)
        groups[g][n] = FieldValue(status=FieldStatus.KNOWN, raw_value=a.raw, value=a.raw, evidence_ids=[a.evidence.evidence_id])
    return Experiment(
        experiment_id=eid, study_id=paper.study_id, paper_id=paper.paper_id, source_record_ids=[doc.source_record_id],
        experiment_type=sp.experiment_type, description=sp.description, **{g: GROUP_MODELS[g](**f) for g, f in groups.items()},
        extraction_method=ExtractionMethod.LLM, extraction_model=sp.model_id, extraction_run_id=run.run_id, split=paper.split,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=ASSOC_VERSION,
                              parent_record_ids=[doc.source_record_id]),
    )


def _header_cell(t: Table, c: TableCell) -> TableCell | None:
    by_id = {x.cell_id: x for x in t.cells}
    for x in sorted((x for x in t.cells if x.col == c.col and x.row < t.header_rows), key=lambda x: -x.row):
        origin = by_id.get(x.spanned_from) if x.spanned_from else x
        if origin is not None and origin.text:
            return origin
    return None


def _add_outcome(exp: Experiment, doc: ParsedDocument, paper: Paper, t: Table, c: TableCell, score_ev: EvidenceAssertion,
                 run: Run, out: PaperOutput) -> None:
    hdr = _header_cell(t, c)
    if hdr is None or c.value is None:
        return
    if any(o.metric.value == hdr.text and o.score.value == c.value for o in exp.outcomes):
        return
    m_ev = _cell_evidence(doc, paper, t, hdr, "evaluation.metric", hdr.text, run, exp.experiment_id, f"{ASSOC_VERSION}/header")
    out.evidence.append(m_ev)
    kwargs: dict[str, FieldValue] = {
        "metric": FieldValue(status=FieldStatus.KNOWN, raw_value=hdr.text, value=hdr.text, evidence_ids=[m_ev.evidence_id]),
        "score": FieldValue(status=FieldStatus.KNOWN, raw_value=c.text, value=c.value, unit="percent" if c.unit == "percent" else None,
                            evidence_ids=[score_ev.evidence_id]),
    }
    if c.uncertainty is not None:  # own evidence for its own field (the score evidence is for evaluation.score)
        u_ev = _cell_evidence(doc, paper, t, c, "evaluation.uncertainty", c.uncertainty, run, exp.experiment_id, f"{ASSOC_VERSION}/cell-uncertainty")
        out.evidence.append(u_ev)
        kwargs["uncertainty"] = FieldValue(status=FieldStatus.KNOWN, raw_value=c.text, value=c.uncertainty, evidence_ids=[u_ev.evidence_id])
    split_src = next(((x, m) for x in [*(h for h in t.cells if h.is_header and h.col == c.col and h.spanned_from is None)]
                      for m in [SPLIT_WORD.search(x.text)] if m), None)
    if split_src:
        s_ev = _cell_evidence(doc, paper, t, split_src[0], "evaluation.dataset_split", split_src[1].group(0), run, exp.experiment_id, f"{ASSOC_VERSION}/header")
        out.evidence.append(s_ev)
        kwargs["dataset_split"] = FieldValue(status=FieldStatus.KNOWN, raw_value=split_src[1].group(0), value=split_src[1].group(1).lower(),
                                             evidence_ids=[s_ev.evidence_id])
    exp.outcomes.append(OutcomeObservation(**kwargs))


def assemble(paper: Paper, doc: ParsedDocument, pwc_exps: list[Experiment], specs: list[ExperimentSpec],
             settings: list[EvidenceAssertion], run: Run) -> PaperOutput:
    out = PaperOutput()
    for e in pwc_exps:
        associate_claims(e, doc, paper, run, out)
    out.evidence.extend(settings)
    for sp in specs:
        out.evidence.extend(a.evidence for a in sp.identity.values())

    # merge model-identified experiments into existing claims when model (and stated benchmark) match
    targets: list[tuple[ExperimentSpec, Experiment]] = []
    new: list[Experiment] = []
    new_by_id: dict[str, Experiment] = {}
    for sp in specs:
        model = sp.identity.get("model.model_name") or sp.identity.get("method.method_name")
        bench = sp.identity.get("evaluation.benchmark")
        matches = [e for e in pwc_exps if names_match(model.raw if model else None, _known(e, "model.model_name").raw_value)
                   and (bench is None or names_match(bench.raw, _known(e, "evaluation.benchmark").raw_value))]
        if matches:
            for e in matches:
                for path, a in sp.identity.items():
                    if _known(e, path).status is FieldStatus.UNKNOWN:
                        _set(e, path, FieldValue(status=FieldStatus.KNOWN, raw_value=a.raw, value=a.raw, evidence_ids=[a.evidence.evidence_id]))
                targets.append((sp, e))
            out.counts["specs_merged_into_claims"] += 1
        else:
            e = _new_experiment(sp, paper, doc, run)
            if e.experiment_id in new_by_id:  # the model repeated an identical spec: one experiment, not two
                e = new_by_id[e.experiment_id]
                out.counts["specs_duplicate"] += 1
            else:
                new_by_id[e.experiment_id] = e
                new.append(e)
                out.counts["experiments_new_llm"] += 1
            targets.append((sp, e))

    # cells for each experiment's table rows x columns
    tables = {t.table_id: t for t in doc.tables}
    claimed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sp, e in targets:
        for key in sp.rows:
            claimed[key].add(e.experiment_id)
    done: set[tuple[str, str]] = set()
    for sp, e in targets:
        for tid, row in sp.rows:
            t = tables.get(tid)
            if t is None:
                continue
            cols = {c for (tt, c) in sp.columns if tt == tid}
            owners = claimed[(tid, row)]
            for c in t.cells:
                if c.is_header or c.spanned_from or c.value is None or (c.cell_id, e.experiment_id) in done:
                    continue
                if not any(_norm_label(x) == row for x in c.row_header_path):
                    continue
                if cols and not any(_norm_label(x) in cols for x in c.col_header_path):
                    continue
                done.add((c.cell_id, e.experiment_id))
                if len(owners) > 1:
                    if (c.cell_id, "ambiguous") in done:
                        continue
                    done.add((c.cell_id, "ambiguous"))
                    ev = _cell_evidence(doc, paper, t, c, "evaluation.score", c.value, run, None, f"{ASSOC_VERSION}/spec-cell")
                    res = _result(doc, paper, t, c, [ev.evidence_id])
                    a = _assoc(res, AssociationStatus.AMBIGUOUS, f"row '{row}' of {tid} is claimed by {len(owners)} experiments", None,
                               sorted(owners), [], [], [ev.evidence_id])
                else:
                    ev = _cell_evidence(doc, paper, t, c, "evaluation.score", c.value, run, e.experiment_id, f"{ASSOC_VERSION}/spec-cell")
                    res = _result(doc, paper, t, c, [ev.evidence_id])
                    a = _assoc(res, AssociationStatus.LIKELY, f"model-identified experiment maps to row '{row}' and column of {tid}; labels verified",
                               e.experiment_id, [e.experiment_id], ["model.model_name"], [], [ev.evidence_id])
                    if e.extraction_method is ExtractionMethod.LLM:
                        _add_outcome(e, doc, paper, t, c, ev, run, out)
                out.evidence.append(ev)
                out.results.append(res)
                out.associations.append(a)
                out.counts[f"assoc_{a.status.value}"] += 1

    # settings by scope
    table_of: dict[str, set[str]] = defaultdict(set)
    for a in out.associations:
        if a.experiment_id:
            table_of[a.experiment_id].add(a.result_id)
    result_table = {r.result_id: r.table for r in out.results}
    all_exps = pwc_exps + new
    for e in all_exps:
        tables_of_e = {result_table[r] for r in table_of.get(e.experiment_id, set())}
        wanted: dict[str, list[EvidenceAssertion]] = defaultdict(list)
        for s in settings:
            if s.scope is None:
                continue
            if s.scope is EvidenceScope.PAPER:
                ok = e.experiment_type not in NON_TRAINED
            elif s.scope is EvidenceScope.MODEL:
                ok = any(names_match(s.applies_to, _known(e, p).raw_value) for p in ("model.model_name", "method.method_name", "model.base_model"))
            elif s.scope is EvidenceScope.TABLE:
                ok = bool(s.applies_to) and s.applies_to.split("/")[0] in tables_of_e
            else:
                ok = bool(s.applies_to) and (names_match(s.applies_to, _known(e, "evaluation.benchmark").raw_value)
                                             or names_match(s.applies_to, e.description))
            if ok:
                wanted[s.field_name].append(s)
        for path, evs in wanted.items():
            values = {str(ev.normalized_value).lower() for ev in evs}
            current = _known(e, path)
            if len(values) > 1:
                if path not in e.unresolved_fields:
                    e.unresolved_fields.append(path)
                out.counts["settings_competing_values"] += 1
                continue
            ids = sorted({ev.evidence_id for ev in evs})
            if current.status is FieldStatus.KNOWN:
                if str(current.value).lower() in values:
                    current.evidence_ids = sorted(set(current.evidence_ids) | set(ids))
                else:
                    _set(e, path, FieldValue(status=FieldStatus.CONFLICTING, evidence_ids=sorted(set(current.evidence_ids) | set(ids))))
                    out.counts["settings_conflicting_existing"] += 1
                continue
            if current.status is not FieldStatus.UNKNOWN:
                continue
            ev0 = evs[0]
            pct = str(ev0.extracted_value).strip().endswith("%")
            value, rule, unit = ev0.normalized_value, ev0.normalization_rule, "percent" if pct else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if path in UNIT_INTERVAL_FIELDS and pct:  # "5%" warmup ratio / dropout -> 0.05
                    value, rule, unit = value / 100, "percent_to_fraction", None
                if (path in UNIT_INTERVAL_FIELDS and not 0 <= value <= 1) or (path in POSITIVE_FIELDS and value <= 0):
                    out.counts["settings_invalid_value"] += 1  # impossible as stated: stays UNKNOWN, never guessed
                    continue
            _set(e, path, FieldValue(status=FieldStatus.KNOWN, raw_value=ev0.extracted_value, value=value, unit=unit,
                                     normalization_rule=rule, normalization_version=ev0.normalization_version, evidence_ids=ids))
            out.counts["settings_attached"] += 1
    out.experiments = all_exps
    return out
