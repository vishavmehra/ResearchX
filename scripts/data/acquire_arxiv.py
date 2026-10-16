"""arXiv metadata + versions for every corpus paper with an arXiv id (DATASET_SPEC §6.3).

Respects the arXiv API ToU (one request every 3 s, single connection). Cached + resumable.
"""

import argparse

from services.datasets.store import run_stage
from services.retrieval.sources import arxiv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    with run_stage("acquire_arxiv", vars(args)) as run:
        arxiv.acquire(run, refresh=args.refresh)


if __name__ == "__main__":
    main()
