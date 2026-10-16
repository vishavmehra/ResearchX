"""Contradiction candidates with both claims preserved (DATASET_SPEC §18, §44).

Metadata conflicts across sources and PwC-vs-paper numeric conflicts. The source-record claims
behind metadata conflicts are stored as evidence and immediately re-verified.
"""

import argparse
from collections import Counter

from services.comparison.conflicts import metadata_contradictions, numeric_contradictions
from services.datasets.schemas import EvidenceAssertion, EvidenceStatus, Experiment, FailureCategory, Paper, SourceRecord
from services.datasets.store import path, read_models, run_stage, write_jsonl
from services.evidence.assertions import verify


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_contradictions") as run:
        papers = read_models(path("papers"), Paper)
        records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        meta, meta_evs = metadata_contradictions(papers, records, run.run_id)
        for ev in meta_evs:
            ok, reason = verify(ev, {}, records)
            ev.status = EvidenceStatus.SOURCE_CHECKED if ok else EvidenceStatus.REJECTED
            if not ok:
                run.fail(FailureCategory.dataset_error, ev.evidence_id, "metadata value re-found in source record", reason)
        evidence = {e.evidence_id: e for e in read_models(path("experiment_evidence"), EvidenceAssertion)}
        numeric = numeric_contradictions(read_models(path("experiments"), Experiment), evidence)
        contras = sorted(meta + numeric, key=lambda k: k.contradiction_id)
        run.output(write_jsonl(path("metadata_evidence"), meta_evs))
        run.output(write_jsonl(path("contradictions"), contras))
        run.counts.update(contradictions=len(contras), metadata_evidence=len(meta_evs),
                          **{f"type_{k}": v for k, v in Counter(k.conflict_type.value for k in contras).items()})


if __name__ == "__main__":
    main()
