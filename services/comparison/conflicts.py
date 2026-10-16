"""Contradiction and abstention datasets (DATASET_SPEC §18, §19, §44).

Contradictions preserve both claims with evidence; nothing is resolved automatically.
- Metadata conflicts between sources for the same paper (title, year, author count, DOI).
- Numeric conflicts: a PwC leaderboard value vs a strictly aligned cell in the paper's table.
Abstention cases are derived from comparison outcomes (and numeric conflicts): the correct
behavior is sometimes to SEARCH_MORE, ABSTAIN or ESCALATE instead of answering.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from services.canonicalization.normalize import jaccard, title_tokens
from services.datasets.schemas import (
    AbstentionCase,
    Comparison,
    ComparisonLabel,
    ConflictType,
    Contradiction,
    EvidenceAssertion,
    EvidenceClass,
    EvidenceStatus,
    ExpectedBehavior,
    Experiment,
    FieldComparisonStatus,
    Paper,
    Provenance,
    SourceRecord,
    Split,
)
from services.datasets.store import code_version, stable_id
from services.evidence.assertions import make

CONFLICT_VERSION = "conflicts-0.1.0"
ABSTENTION_VERSION = "abstention-0.1.0"
META = {  # conflict field -> (paper field path, SourceRecord attribute, conflict type)
    "title": ("paper.title", "title", ConflictType.STUDY_IDENTITY_CONFLICT),
    "publication_year": ("paper.publication_year", "publication_year", ConflictType.TIMELINE_CONFLICT),
    "author_count": ("paper.authors", "authors", ConflictType.OTHER),
    "doi": ("paper.doi", "doi", ConflictType.STUDY_IDENTITY_CONFLICT),
}
EXPLAIN = {
    "title": "a wrongly merged record, a title change between versions, or a venue retitle",
    "publication_year": "different date semantics (preprint v1 vs later publication/indexing date) or a metadata error",
    "author_count": "an author list changed between versions or incomplete source metadata",
    "doi": "preprint and published versions carrying different DOIs, or a wrongly merged record",
}


def _prov(parents: list[str], version: str) -> Provenance:
    return Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=version, parent_record_ids=parents)


def _meta_evidence(rec: SourceRecord, paper: Paper, field: str, run_id: str) -> EvidenceAssertion:
    path, attr, _ = META[field]
    raw = getattr(rec, attr)
    value = str(len(raw)) if attr == "authors" else str(raw)
    stated = f"lists {value} authors" if attr == "authors" else f'states {attr} = "{value}"'
    return make(
        source_name=rec.source_name, source_url=rec.source_url, content_hash=rec.provenance.content_hash,
        source_record_id=rec.source_record_id, paper_id=paper.paper_id, study_id=paper.study_id, experiment_id=None,
        field_name=path, extracted_value=value, normalized_value=value,
        evidence_text=f"{rec.source_name} record {rec.source_record_id} {stated}",
        source_location=f"{rec.source_name} metadata record {rec.source_record_id}, field {attr}",
        source_version=rec.source_version, retrieval_timestamp=rec.provenance.retrieved_at,
        extraction_model=f"import:{CONFLICT_VERSION}", extraction_run_id=run_id,
        status=EvidenceStatus.MACHINE_EXTRACTED, redistribution_allowed=None,
    )


def metadata_contradictions(papers: list[Paper], records: dict[str, SourceRecord], run_id: str) -> tuple[list[Contradiction], list[EvidenceAssertion]]:
    contras: dict[str, Contradiction] = {}
    evidence: dict[str, EvidenceAssertion] = {}
    for p in papers:
        for c in p.metadata_conflicts:
            field = c["field"]
            if field not in META:
                continue
            values = c["values"]
            chosen = p.field_sources.get("authors" if field == "author_count" else field)
            ref = chosen if chosen in values else sorted(values)[0]
            for srid, v in sorted(values.items()):
                if srid == ref or str(v) == str(values[ref]):
                    continue
                if field == "title" and jaccard(title_tokens(str(v)), title_tokens(str(values[ref]))) >= 0.7:
                    continue  # only genuinely diverging pairs (same threshold that raised the conflict)
                ra, rb = records.get(ref), records.get(srid)
                if not ra or not rb:
                    continue
                ea, eb = _meta_evidence(ra, p, field, run_id), _meta_evidence(rb, p, field, run_id)
                evidence[ea.evidence_id], evidence[eb.evidence_id] = ea, eb
                k = Contradiction(
                    contradiction_id=stable_id("con", p.paper_id, field, ref, srid),
                    subject_id=p.paper_id,
                    claim=f"{field} of {p.paper_id}",
                    field=META[field][0],
                    source_a=ref,
                    source_b=srid,
                    value_a=values[ref],
                    value_b=v,
                    conflict_type=META[field][2],
                    evidence_a=[ea.evidence_id],
                    evidence_b=[eb.evidence_id],
                    explanation=(f"{ra.source_name} reports {values[ref]!r} while {rb.source_name} reports {v!r}. "
                                 f"Possible explanations include {EXPLAIN[field]}. Not resolved automatically."),
                    split=p.split,
                    provenance=_prov([ref, srid], CONFLICT_VERSION),
                )
                contras[k.contradiction_id] = k
    return [contras[k] for k in sorted(contras)], [evidence[k] for k in sorted(evidence)]


def numeric_contradictions(exps: list[Experiment], evidence: dict[str, EvidenceAssertion]) -> list[Contradiction]:
    out = []
    for e in exps:
        for check in (e.cross_check or {}).get("outcomes", []):
            if check.get("status") != "CONFLICT_CANDIDATE":
                continue
            outcome = next((o for o in e.outcomes if o.metric.value == check["metric"]), None)
            paper_ev = evidence.get(check.get("paper_evidence_id", ""))
            if outcome is None or paper_ev is None:
                continue
            pwc_ids = [i for i in outcome.score.evidence_ids if i in evidence and evidence[i].source_record_id.startswith("pwc:")]
            if not pwc_ids:
                continue
            loc = check["location"]
            out.append(
                Contradiction(
                    contradiction_id=stable_id("con", e.experiment_id, check["metric"], paper_ev.evidence_id),
                    subject_id=e.experiment_id,
                    claim=f"{check['metric']} score of {e.description}",
                    field="evaluation.score",
                    source_a=evidence[pwc_ids[0]].source_record_id,
                    source_b=paper_ev.source_record_id,
                    value_a=outcome.score.raw_value,
                    value_b=loc["cell"],
                    conflict_type=ConflictType.NUMERIC_CONFLICT,
                    evidence_a=pwc_ids,
                    evidence_b=[paper_ev.evidence_id],
                    explanation=(f"Papers With Code lists {outcome.score.raw_value} while table {loc['table_id']} of the paper shows "
                                 f"{loc['cell']!r} at the aligned cell (row {loc['row_label']!r}, column {loc['col_label']!r}). "
                                 "Possible explanations: a different split/setting row, rounding, or a leaderboard transcription "
                                 "error. The row/column alignment is heuristic; not resolved automatically."),
                    split=e.split,
                    provenance=_prov([e.experiment_id, paper_ev.evidence_id], CONFLICT_VERSION),
                )
            )
    return sorted(out, key=lambda k: k.contradiction_id)


def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def abstention_cases(comparisons: list[Comparison], exps: dict[str, Experiment], parsed_papers: set[str],
                     numeric: list[Contradiction], cap_per_class: int = 400) -> list[AbstentionCase]:
    """Balanced SUFFICIENT / INSUFFICIENT / CONFLICTING cases (spec §19). Question text is templated."""
    by_class: dict[str, list[AbstentionCase]] = defaultdict(list)
    template = "same-configuration-v1"
    for c in comparisons:
        a, b = exps[c.experiment_a], exps[c.experiment_b]
        missing = sorted({fc.field_name for fc in c.field_comparisons if fc.comparison_status is FieldComparisonStatus.UNKNOWN
                          and fc.field_name.split("[")[0] in ("evaluation.dataset_split", "evaluation.metric_definition",
                                                              "evaluation.evaluation_protocol", "evaluation.benchmark", "research.task")})
        conflicting = [fc.field_name for fc in c.field_comparisons if fc.comparison_status is FieldComparisonStatus.CONFLICTING]
        differing = [fc.field_name for fc in c.field_comparisons if fc.comparison_status is FieldComparisonStatus.MATERIALLY_DIFFERENT]
        if c.relationship_label is ComparisonLabel.CONFLICTING_EVIDENCE:
            cls, beh, why = EvidenceClass.CONFLICTING_EVIDENCE, ExpectedBehavior.ESCALATE, f"conflicting evidence on {conflicting}"
        elif c.relationship_label is ComparisonLabel.INSUFFICIENT_EVIDENCE:
            unread = not (a.paper_id in parsed_papers and b.paper_id in parsed_papers)
            cls = EvidenceClass.INSUFFICIENT_EVIDENCE
            beh = ExpectedBehavior.SEARCH_MORE if unread else ExpectedBehavior.ABSTAIN
            why = (f"{missing} unknown; full text of at least one paper not yet read" if unread
                   else f"{missing} unknown even after reading both parsed papers")
        else:
            cls, beh = EvidenceClass.SUFFICIENT_EVIDENCE, ExpectedBehavior.ANSWER
            why = f"answerable from evidence: {c.relationship_label.value}" + (f"; materially different fields {differing}" if differing else "")
        by_class[cls.value].append(
            AbstentionCase(
                case_id=stable_id("abs", c.comparison_id),
                question=f"Has the experiment \"{a.description}\" ({a.paper_id}) already been tested in the same configuration as \"{b.description}\" ({b.paper_id})?",
                question_template=template,
                synthetic_question=True,
                context_ids=[c.comparison_id, a.experiment_id, b.experiment_id],
                evidence_class=cls,
                expected_behavior=beh,
                missing_information=missing,
                conflicting_fields=conflicting,
                evidence_ids=c.evidence,
                rationale=why,
                split=c.split,
                provenance=_prov([c.comparison_id], ABSTENTION_VERSION),
            )
        )
    for k in numeric:
        e = exps.get(k.subject_id)
        if not e:
            continue
        by_class[EvidenceClass.CONFLICTING_EVIDENCE.value].append(
            AbstentionCase(
                case_id=stable_id("abs", k.contradiction_id),
                question=f"What {k.claim.split(' score of ')[0]} score does \"{e.description}\" report?",
                question_template="reported-score-v1",
                synthetic_question=True,
                context_ids=[k.contradiction_id, e.experiment_id],
                evidence_class=EvidenceClass.CONFLICTING_EVIDENCE,
                expected_behavior=ExpectedBehavior.ESCALATE,
                conflicting_fields=["evaluation.score"],
                evidence_ids=k.evidence_a + k.evidence_b,
                rationale="two sources disagree on the reported value; do not pick one without resolving evidence",
                split=k.split,
                provenance=_prov([k.contradiction_id], ABSTENTION_VERSION),
            )
        )
    out = []
    for cls, cases in sorted(by_class.items()):
        out += sorted(cases, key=lambda x: _h(cls, x.case_id))[:cap_per_class]
    return sorted(out, key=lambda x: x.case_id)
