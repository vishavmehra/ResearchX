"""Papers With Code archive acquisition (DATASET_SPEC §6.4): pinned, checksum-verified parquet.

Downloads pwc-archive/links-between-paper-and-code (~41 MB) and pwc-archive/evaluation-tables
(~138 MB). Already-verified files are skipped, so the stage is resumable.
"""

import argparse

from services.datasets.store import run_stage
from services.retrieval.sources import papers_with_code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-resolve the dataset revision")
    args = ap.parse_args()
    with run_stage("acquire_papers_with_code", vars(args)) as run:
        papers_with_code.acquire(run, refresh=args.refresh)


if __name__ == "__main__":
    main()
