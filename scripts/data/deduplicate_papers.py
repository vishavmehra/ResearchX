"""Stage 2: identifier-only deduplication into Papers + provisional Studies (DATASET_SPEC §29).

Also assigns study-level splits (spec §28) and intra-corpus citation edges.
Merges never use title similarity; every merge lists its matching identifier.
"""

import argparse
from collections import Counter

from services.datasets.schemas import SourceRecord
from services.datasets.store import path, read_models, run_stage, write_jsonl
from services.retrieval.corpus import assign_splits, build_citations, build_papers, build_studies
from services.retrieval.fulltext import apply_fulltext_index


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("deduplicate_papers") as run:
        records = read_models(path("source_records"), SourceRecord)
        papers = build_papers(records)
        apply_fulltext_index(papers)
        assign_splits(papers)
        studies = build_studies(papers)
        citations = build_citations(papers, records)
        for key, rows in (("papers", papers), ("studies", studies), ("citations", citations)):
            run.output(write_jsonl(path(key), rows))
        per_source = Counter()
        for p in papers:
            per_source.update(s.split(":", 1)[0] for s in p.source_record_ids)
        dup_within_source = sum(
            1 for p in papers if len({s.split(':', 1)[0] for s in p.source_record_ids}) < len(p.source_record_ids)
        )
        run.counts.update(
            source_records=len(records),
            papers=len(papers),
            papers_with_duplicate_records_in_one_source=dup_within_source,
            papers_with_metadata_conflicts=sum(1 for p in papers if p.metadata_conflicts),
            citation_edges=len(citations),
        )


if __name__ == "__main__":
    main()
