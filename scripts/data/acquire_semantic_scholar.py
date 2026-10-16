"""Semantic Scholar enrichment for every corpus paper (DATASET_SPEC §6.2). Cached + resumable."""

import argparse

from services.datasets.store import run_stage
from services.retrieval.sources import semantic_scholar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    with run_stage("acquire_semantic_scholar", vars(args)) as run:
        semantic_scholar.acquire(run, refresh=args.refresh)


if __name__ == "__main__":
    main()
