"""Normalize every acquired source's cached raw payloads into SourceRecords (DATASET_SPEC §4.2).

Pure function of data/raw/**: deterministic, no network. Discovery sources define the corpus
(OpenAlex); enrichment sources (S2, arXiv, PwC) only contribute records for corpus papers.
"""

import argparse

from services.canonicalization.normalize import normalize_title
from services.datasets.store import path, run_stage, write_jsonl
from services.retrieval.sources import acquired_adapters

DISCOVERY_SOURCES = {"OPENALEX", "ARXIV_OAI_PMH"}  # OAI-PMH contributes only promoted Layer 0 candidates
TRUSTED_TITLE_SOURCES = ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("normalize_metadata") as run:
        records = []
        context: dict = {"arxiv_ids": set(), "titles": {}}
        for adapter in acquired_adapters():  # ordered: OpenAlex, S2, arXiv, PwC
            recs = adapter.normalize_all(context)
            for r in recs:
                if r.source_name in DISCOVERY_SOURCES and r.arxiv_id:
                    context["arxiv_ids"].add(r.arxiv_id)
                if r.source_name in TRUSTED_TITLE_SOURCES and r.arxiv_id and r.title:
                    context["titles"].setdefault(normalize_title(r.title), r.arxiv_id)
            run.counts[adapter.SOURCE] = len(recs)
            records.extend(recs)
        records.sort(key=lambda r: r.source_record_id)
        ids = [r.source_record_id for r in records]
        if len(ids) != len(set(ids)):
            raise SystemExit("duplicate source_record_id produced by adapters")
        run.output(write_jsonl(path("source_records"), records))
        run.counts["source_records"] = len(records)


if __name__ == "__main__":
    main()
