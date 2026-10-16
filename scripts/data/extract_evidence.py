"""Stage 7: evidence extraction + verification (DATASET_SPEC §11, §12, §34 stage 7).

1. Rule-based STUDY-LEVEL evidence from parsed full text (explicitly stated hyperparameters,
   compute) -> silver/evidence/rules_evidence.jsonl.
2. Verification of ALL Silver evidence: every assertion is re-found at its stated location
   (parsed full text or stored source record). Pass -> SOURCE_CHECKED; fail -> REJECTED with a
   failure record. Experiment fields left citing only rejected evidence become UNKNOWN.
"""

import argparse
from collections import Counter

from services.datasets.schemas import EvidenceAssertion, EvidenceStatus, Experiment, FailureCategory, FieldStatus, FieldValue, Paper, SourceRecord
from services.datasets.store import DATA, path, read_jsonl, read_models, run_stage, write_jsonl
from services.evidence.assertions import load_parsed_docs, verify
from services.extraction import rules
from services.retrieval.fulltext import load_index


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("extract_evidence") as run:
        papers = {p.paper_id: p for p in read_models(path("papers"), Paper)}
        docs = load_parsed_docs()
        index = load_index()
        rule_evs = []
        for pid, doc in sorted(docs.items()):
            paper = papers.get(pid)
            if paper is None:
                continue
            ts = index.get(paper.arxiv_id or "", {}).get("fulltext", {}).get("retrieval_timestamp")
            rule_evs += rules.extract_document(doc, paper, run.run_id, ts)
        rule_evs.sort(key=lambda e: e.evidence_id)
        run.output(write_jsonl(path("rules_evidence"), rule_evs))
        run.counts["rules_evidence"] = len(rule_evs)

        records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        artifacts = {a["artifact_id"]: a for a in read_jsonl(path("artifacts"))}  # artifact evidence is re-read from the cached pinned file
        status: dict[str, EvidenceStatus] = {}
        verdicts = Counter()
        for f in sorted((DATA / "silver" / "evidence").glob("*.jsonl")):
            evs = read_models(f, EvidenceAssertion)
            for ev in evs:
                if ev.status is not EvidenceStatus.HUMAN_REVIEWED:
                    ok, reason = verify(ev, docs, records, artifacts)
                    ev.status = EvidenceStatus.SOURCE_CHECKED if ok else EvidenceStatus.REJECTED
                    verdicts[f"{ev.status.value}:{reason if not ok else 'ok'}"] += 1
                    if not ok:
                        run.fail(FailureCategory.extraction_failure, f"{ev.evidence_id} {ev.field_name}", "evidence re-found at stated location", reason, severity="LOW")
                status[ev.evidence_id] = ev.status
            run.output(write_jsonl(f, evs))

        exps = read_models(path("experiments"), Experiment)
        downgraded = 0
        for e in exps:
            slots = [(getattr(e, g), n) for g in ("research", "dataset", "population", "model", "method", "training", "inference", "evaluation", "compute", "artifacts") for n in type(getattr(e, g)).model_fields]
            slots += [(o, n) for o in e.outcomes for n in type(o).model_fields]
            for holder, name in slots:
                fv: FieldValue = getattr(holder, name)
                if fv.status is not FieldStatus.KNOWN:
                    continue
                keep = [i for i in fv.evidence_ids if status.get(i) is not EvidenceStatus.REJECTED]
                if not keep:
                    setattr(holder, name, FieldValue())
                    downgraded += 1
                elif len(keep) < len(fv.evidence_ids):
                    fv.evidence_ids = keep
        run.output(write_jsonl(path("experiments"), exps))
        run.counts.update(evidence_verified=len(status), fields_downgraded_to_unknown=downgraded, **verdicts)


if __name__ == "__main__":
    main()
