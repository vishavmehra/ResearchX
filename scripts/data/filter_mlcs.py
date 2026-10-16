"""Stage 3: deterministic ML/CS relevance filter (DATASET_SPEC §34). Rules: config/domain.yaml."""

import argparse
from collections import Counter, defaultdict

from services.datasets.schemas import Paper, SourceRecord
from services.datasets.store import path, read_models, run_stage, write_jsonl
from services.retrieval.corpus import assess_relevance


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("filter_mlcs") as run:
        papers = read_models(path("papers"), Paper)
        by_id = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        areas = Counter()
        reasons = defaultdict(int)
        for p in papers:
            p.relevance = assess_relevance(p, [by_id[s] for s in p.source_record_ids if s in by_id])
            areas.update(p.relevance.areas if p.relevance.is_relevant else [])
            if not p.relevance.is_relevant:
                reasons["no_area_term" if not p.relevance.areas else "field_check_failed"] += 1
        run.output(write_jsonl(path("papers"), papers))
        run.counts.update(
            papers=len(papers),
            relevant=sum(p.relevance.is_relevant for p in papers),
            **{f"excluded_{k}": v for k, v in reasons.items()},
        )
        run.note(f"relevant papers per area: {dict(areas)}")


if __name__ == "__main__":
    main()
