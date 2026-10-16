"""Study-link candidates between papers (DATASET_SPEC §14). Silver; SAME_* labels need review."""

import argparse
from collections import Counter

from services.comparison.links import build_links
from services.datasets.schemas import CitationEdge, Paper, SourceRecord
from services.datasets.store import path, read_models, run_stage, write_jsonl


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_study_links") as run:
        papers = read_models(path("papers"), Paper)
        records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        cites = {(c.citing_paper_id, c.cited_paper_id) for c in read_models(path("citations"), CitationEdge)}
        links = build_links(papers, records, cites)
        run.output(write_jsonl(path("study_links"), links))
        run.counts.update(study_links=len(links), hard_negatives=sum(l.is_hard_negative for l in links),
                          **{f"label_{k}": v for k, v in Counter(l.relationship.value for l in links).items()})


if __name__ == "__main__":
    main()
