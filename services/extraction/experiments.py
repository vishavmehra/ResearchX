"""Experiment candidates from Papers With Code leaderboard claims, cross-checked against the
paper's own tables (DATASET_SPEC §37, §40, §44).

- One Experiment per leaderboard row = one experimental condition (model x benchmark x task),
  with one OutcomeObservation per reported metric. extraction_method = SOURCE_IMPORT, tier SILVER.
- Every KNOWN field cites a PwC evidence assertion: the PwC record is the source of the claim.
- Cross-check: when the paper's parsed tables contain the same number in a cell whose row/column
  labels align with the model and metric/benchmark, a paper evidence assertion is added to the
  score. A strictly aligned cell with a DIFFERENT number is a NUMERIC_CONFLICT candidate for
  human review; nothing is resolved automatically. Value-only matches are recorded, not used.
- Official code links become study-level evidence (artifacts.repository_url, experiment_id=None).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

from services.canonicalization.normalize import NORMALIZATION_VERSION, canonical_label, parse_number
from services.datasets.schemas import (
    EvaluationConfiguration,
    EvidenceAssertion,
    EvidenceStatus,
    Experiment,
    ExperimentType,
    ExtractionMethod,
    FieldStatus,
    FieldValue,
    ModelConfiguration,
    OutcomeObservation,
    Paper,
    ParsedDocument,
    Provenance,
    ResearchQuestion,
    SourceRecord,
)
from services.datasets.store import Run, code_version, stable_id
from services.evidence.assertions import leaderboard_locator, make

IMPORT_VERSION = "pwc-import-0.2.0"
CROSSCHECK_VERSION = "table-crosscheck-0.3.0"
GENERIC = frozenset({"model", "models", "ours", "our", "the", "with", "and", "of", "for", "on", "in", "method", "baseline", "net", "network",
                     "score", "dataset", "benchmark", "set", "test", "dev", "split",
                     # evaluation-setting words in PwC model names ("(0-shot)", "(few-shot, k=5)", "(fine-tuned)")
                     "shot", "zero", "few", "one", "fine", "tuned", "finetuned", "k"})
# PwC stores model size as a leaderboard "metric" (e.g. "Parameters (Billions)"): it is model configuration,
# not an outcome. Routed to model.parameter_count with the unit stated in the metric name (explicit conversion).
PARAM_METRIC = re.compile(r"^\s*(?:#\s*|number of\s+)?param(?:eter)?s?\s*(?:\((?P<unit>billions?|millions?|b|m)\))?\s*$", re.I)
PARAM_MULT = {"billion": 10**9, "billions": 10**9, "b": 10**9, "million": 10**6, "millions": 10**6, "m": 10**6}
_TEX_WRAP = re.compile(r"\\(?:mathbf|textbf|bm|boldsymbol|underline|textit|mathit|text|mathrm|emph|textsc)\s*\{([^{}]*)\}")


def cell_number(cell: str) -> Decimal | None:
    """Numeric value of a table cell, ignoring formatting (bold, ±std, footnote marks). None if not numeric."""
    s = cell
    for _ in range(3):
        s = _TEX_WRAP.sub(r"\1", s)
    s = re.split(r"±|\\pm|\(|\[| \+", s)[0]
    s = re.sub(r"\\[A-Za-z]+", " ", s).replace("{", "").replace("}", "").replace("$", "")
    s = s.strip().rstrip("%*†‡§¶").strip().replace("−", "-")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _toks(s: str | None) -> frozenset[str]:
    return frozenset(t for t in canonical_label(s).split() if (len(t) > 1 or t.isdigit()) and t not in GENERIC)


def _prefix_overlap(metric: frozenset[str], header: frozenset[str]) -> bool:
    return any(h == m or (len(h) >= 3 and m.startswith(h)) or (len(m) >= 3 and h.startswith(m)) for m in metric for h in header)


def _coverage(tokens: frozenset[str], where: frozenset[str]) -> bool:
    """At least half of the label tokens must appear, including one non-numeric token.
    'CIFAR-10-LT' vs 'ResNet-10' (only '10') fails; 'T5-XXL 11B' vs 'T5-Large' (only 't5') fails."""
    hit = tokens & where
    return bool(tokens) and len(hit) / len(tokens) >= 0.5 and any(not t.isdigit() for t in hit)


_SIZE_TOKEN = re.compile(r"^\d+(?:\.\d+)?[bmk]$")


def _size_clash(model: frozenset[str], where: frozenset[str]) -> bool:
    """'T5-3B' must not align with a 'T5-11B' row: differing size tokens veto a model match."""
    sa, sb = {t for t in model if _SIZE_TOKEN.match(t)}, {t for t in where if _SIZE_TOKEN.match(t)}
    return bool(sa and sb and not sa & sb)


def _consistent(target: Decimal, num: Decimal) -> str | None:
    """Deterministic non-conflict explanations for unequal numbers (recorded, never silently merged)."""
    lower_precision = target if target.as_tuple().exponent >= num.as_tuple().exponent else num
    other = num if lower_precision is target else target
    try:
        if other.quantize(Decimal(1).scaleb(lower_precision.as_tuple().exponent)) == lower_precision:
            return "ROUNDING_CONSISTENT"
    except InvalidOperation:
        pass
    if target * 100 == num or num * 100 == target:
        return "SCALE_CONSISTENT"  # fraction vs percent representation
    return None


def crosscheck(doc: ParsedDocument, r: dict[str, Any], metric: str, raw: str) -> dict[str, Any]:
    target = cell_number(raw)
    if target is None:
        return {"status": "PWC_VALUE_NOT_NUMERIC"}
    mt, kt, dt = _toks(r["model_name"]), _toks(metric), _toks(r["dataset"])
    exact: list[tuple[bool, bool, dict]] = []
    others: list[tuple[Decimal, dict]] = []
    for tb in doc.tables:
        if len(tb.rows) < 2:
            continue
        n_head = max(tb.header_rows, 1)
        header = tb.rows[:n_head]
        caption = _toks(tb.caption)
        header_all = frozenset(t for h in header for c in h for t in _toks(c))
        for i, row in enumerate(tb.rows[:n_head]):  # values sitting in (mis-detected) header rows still count as found
            for j, cell in enumerate(row):
                if cell_number(cell) == target:
                    exact.append((False, False, {"table_id": tb.table_id, "section_id": tb.section_id, "row": i, "col": j,
                                                 "cell": cell, "row_label": (row[0] if row else "")[:80], "col_label": "(header row)"}))
        for i, row in enumerate(tb.rows[n_head:], start=n_head):
            row_toks = _toks(row[0]) if row else frozenset()
            for j, cell in enumerate(row[1:], start=1):
                num = cell_number(cell)
                if num is None:
                    continue
                col_label = " ".join(h[j] for h in header if j < len(h))
                col = _toks(col_label)
                model_cov = _coverage(mt, row_toks | col) and not _size_clash(mt, row_toks | col)
                row_model, col_model = model_cov and bool(mt & row_toks), model_cov and bool(mt & col)
                col_metric, row_metric = _prefix_overlap(kt, col), _prefix_overlap(kt, row_toks)
                ds = _coverage(dt, caption | col | row_toks)
                ds_axis_col, ds_axis_row = _coverage(dt, col), _coverage(dt, row_toks)
                metric_stated = col_metric or row_metric or _prefix_overlap(kt, caption | header_all)
                aligned = (row_model and (col_metric or ds_axis_col or ds)) or (col_model and (row_metric or ds_axis_row or ds))
                # conflict candidates need model axis + dataset coverage + the metric stated in the table itself
                strict = ds and metric_stated and ((row_model and (col_metric or ds_axis_col)) or (col_model and (row_metric or ds_axis_row)))
                loc = {"table_id": tb.table_id, "section_id": tb.section_id, "row": i, "col": j, "cell": cell,
                       "row_label": row[0][:80], "col_label": col_label[:80]}
                if num == target:
                    exact.append((aligned, strict, loc))
                elif strict:
                    others.append((num, loc))
    aligned_hits = [loc for a, _, loc in exact if a]
    if aligned_hits:
        strict_hits = [loc for _, s, loc in exact if s]
        return {"status": "FOUND_ALIGNED", "location": (strict_hits or aligned_hits)[0], "n_candidates": len(aligned_hits)}
    if exact:
        return {"status": "VALUE_FOUND_UNALIGNED", "locations": [loc for _, _, loc in exact[:3]], "n_candidates": len(exact)}
    if len(others) == 1:
        num, loc = others[0]
        why = _consistent(target, num)
        if why:
            return {"status": why, "location": loc, "paper_value": str(num)}
        return {"status": "CONFLICT_CANDIDATE", "location": loc, "paper_value": str(num)}
    return {"status": "NOT_FOUND", "strict_cells_with_other_values": len(others)}


def _known(raw: str, value: Any, evidence_ids: list[str], **kw: Any) -> FieldValue:
    return FieldValue(status=FieldStatus.KNOWN, raw_value=raw, value=value, evidence_ids=evidence_ids, **kw)


def build(paper: Paper, rec: SourceRecord, doc: ParsedDocument | None, run: Run) -> tuple[list[Experiment], list[EvidenceAssertion], Counter]:
    exps: list[Experiment] = []
    evs: list[EvidenceAssertion] = []
    stats: Counter = Counter()
    doc_version = doc.source_record_id.rsplit(paper.arxiv_id or "", 1)[-1] if doc else None

    for link in rec.extra.get("code_links", []):
        if link.get("is_official") and link.get("repo_url"):
            evs.append(
                make(
                    source_name=rec.source_name, source_url=rec.source_url, content_hash=rec.provenance.content_hash,
                    source_record_id=rec.source_record_id, paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=None,
                    field_name="artifacts.repository_url", extracted_value=link["repo_url"], normalized_value=link["repo_url"],
                    evidence_text=f"Papers With Code links-between-paper-and-code: official repository {link['repo_url']} (framework: {link.get('framework')})",
                    source_location=f"pwc-archive links-between-paper-and-code [{rec.source_version}] repo_url={link['repo_url']}",
                    source_version=rec.source_version, retrieval_timestamp=rec.provenance.retrieved_at,
                    extraction_model=f"import:{IMPORT_VERSION}", extraction_run_id=run.run_id,
                    status=EvidenceStatus.MACHINE_EXTRACTED, redistribution_allowed=rec.license.redistribution_allowed,
                )
            )
            stats["official_repository_links"] += 1

    seen: set[str] = set()
    for r in rec.extra.get("evaluation_results", []):
        if not (r.get("task") and r.get("dataset") and r.get("model_name") and r.get("metrics")):
            stats["invalid_pwc_rows_skipped"] += 1
            continue
        eid = stable_id("exp", paper.paper_id, "pwc", r["task"], r["dataset"], r["model_name"], sorted(r["metrics"].items()))
        if eid in seen:
            stats["duplicate_pwc_rows_skipped"] += 1
            continue
        seen.add(eid)
        claim = (
            f'Papers With Code leaderboard "{r["task"]}" / "{r["dataset"]}" (row {r["row_index"]}): '
            f'model "{r["model_name"]}" reports {json.dumps(r["metrics"], ensure_ascii=False, sort_keys=True)}'
        )

        def pev(field: str, value: str, normalized: Any = None, unit: str | None = None, rule: str | None = None) -> EvidenceAssertion:
            e = make(
                source_name=rec.source_name, source_url=rec.source_url, content_hash=rec.provenance.content_hash,
                source_record_id=rec.source_record_id, paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=eid,
                field_name=field, extracted_value=value, normalized_value=value if normalized is None else normalized, unit=unit,
                normalization_rule=rule, normalization_version=NORMALIZATION_VERSION if rule else None,
                evidence_text=claim,
                source_location=f"pwc-archive evaluation-tables [{rec.source_version}] task={r['task']!r} dataset={r['dataset']!r} row={r['row_index']}",
                table=leaderboard_locator(r), source_version=rec.source_version, retrieval_timestamp=rec.provenance.retrieved_at,
                extraction_model=f"import:{IMPORT_VERSION}", extraction_run_id=run.run_id,
                status=EvidenceStatus.MACHINE_EXTRACTED, redistribution_allowed=rec.license.redistribution_allowed,
            )
            evs.append(e)
            return e

        task_ev, bench_ev, model_ev = pev("research.task", r["task"]), pev("evaluation.benchmark", r["dataset"]), pev("model.model_name", r["model_name"])
        outcomes, checks, used_paper = [], [], False
        param_fv = FieldValue()
        for metric, raw in sorted(r["metrics"].items()):
            raw = str(raw)
            if pm := PARAM_METRIC.match(metric):
                parsed = parse_number(raw.strip())
                unit_word = (pm.group("unit") or "").lower()
                mult = PARAM_MULT.get(unit_word)
                value = parsed[0] * mult if (parsed and mult) else (parsed[0] if parsed else raw)
                rule = f"pwc_parameter_metric:{unit_word or 'unit_unstated'}" + (f"(x{mult:.0e})" if mult else "")
                p_ev = pev("model.parameter_count", raw, value, "parameters" if mult else None, rule)
                param_fv = _known(raw, value, [p_ev.evidence_id], unit="parameters" if mult else None,
                                  normalization_rule=rule, normalization_version=NORMALIZATION_VERSION)
                stats["pwc_parameter_metrics_routed_to_model"] += 1
                continue
            parsed = parse_number(raw.strip().rstrip("%").strip())
            unit = "percent" if raw.strip().endswith("%") else None
            m_ev = pev("evaluation.metric", metric)
            s_ev = pev("evaluation.score", raw, parsed[0] if parsed else None, unit, parsed[1] if parsed else None)
            score_ids = [s_ev.evidence_id]
            cc = crosscheck(doc, r, metric, raw) if doc else {"status": "NO_FULLTEXT"}
            stats[f"crosscheck_{cc['status']}"] += 1
            if doc and cc["status"] in ("FOUND_ALIGNED", "CONFLICT_CANDIDATE"):
                loc = cc["location"]
                p_ev = make(
                    source_name="ARXIV", source_url=doc.source_url, content_hash=doc.source_sha256,
                    source_record_id=doc.source_record_id, paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=eid,
                    field_name="evaluation.score", extracted_value=loc["cell"], normalized_value=float(cell_number(loc["cell"])),
                    normalization_rule="table_cell_number", normalization_version=NORMALIZATION_VERSION,
                    evidence_text=loc["cell"],
                    source_location=f'{doc.source_url}#{loc["table_id"]} row {loc["row"]} ("{loc["row_label"]}") column {loc["col"]} ("{loc["col_label"]}")',
                    section=loc["section_id"], table=loc["table_id"], source_version=doc_version or None,
                    retrieval_timestamp=doc.parsed_at, extraction_model=f"crosscheck:{CROSSCHECK_VERSION}", extraction_run_id=run.run_id,
                    status=EvidenceStatus.MACHINE_EXTRACTED, redistribution_allowed=doc.license.redistribution_allowed,
                )
                evs.append(p_ev)
                used_paper = True
                cc = {**cc, "paper_evidence_id": p_ev.evidence_id}
                if cc["status"] == "FOUND_ALIGNED":
                    score_ids.append(p_ev.evidence_id)
            checks.append({"metric": metric, **cc})
            outcomes.append(
                OutcomeObservation(
                    metric=_known(metric, metric, [m_ev.evidence_id]),
                    score=_known(
                        raw, parsed[0] if parsed else raw, score_ids, unit=unit,
                        normalization_rule=parsed[1] if parsed else None,
                        normalization_version=NORMALIZATION_VERSION if parsed else None,
                    ),
                )
            )
        parents = [rec.source_record_id] + ([doc.source_record_id] if used_paper else [])
        exps.append(
            Experiment(
                experiment_id=eid,
                study_id=paper.study_id,
                paper_id=paper.paper_id,
                source_record_ids=parents,
                experiment_type=ExperimentType.EVALUATION,
                description=f'{r["model_name"]} on {r["dataset"]} ({r["task"]})',
                research=ResearchQuestion(task=_known(r["task"], r["task"], [task_ev.evidence_id])),
                evaluation=EvaluationConfiguration(benchmark=_known(r["dataset"], r["dataset"], [bench_ev.evidence_id])),
                model=ModelConfiguration(model_name=_known(r["model_name"], r["model_name"], [model_ev.evidence_id]), parameter_count=param_fv),
                outcomes=outcomes,
                extraction_method=ExtractionMethod.SOURCE_IMPORT,
                extraction_model=f"import:{IMPORT_VERSION}",
                extraction_run_id=run.run_id,
                cross_check={"version": CROSSCHECK_VERSION, "outcomes": checks},
                split=paper.split,
                provenance=Provenance(
                    source_name="RESEARCHX", processing_version=code_version(), extractor_version=IMPORT_VERSION, parent_record_ids=parents
                ),
            )
        )
    return exps, evs, stats
