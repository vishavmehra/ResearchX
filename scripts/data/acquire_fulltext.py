"""Stage 5: selective full-text acquisition for relevant papers (DATASET_SPEC §34, §36).

Licence-classified storage: data/fulltext/{permitted,restricted}/<arxiv_id>/. Resumable.
"""

import argparse

from services.datasets.store import run_stage
from services.retrieval import fulltext


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, help="override config fulltext.sample_size")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    with run_stage("acquire_fulltext", vars(args)) as run:
        fulltext.acquire(run, sample_size=args.sample_size, refresh=args.refresh)


if __name__ == "__main__":
    main()
