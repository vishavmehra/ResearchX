"""Stage 1: OpenAlex candidate metadata discovery (DATASET_SPEC §6.1, §34).

Profiles live in config/pipeline.yaml (smoke | initial). Every page is cached under
data/raw/openalex/, so re-runs replay the cache and cost nothing.
"""

import argparse

from services.datasets.store import run_stage
from services.retrieval.sources import openalex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="smoke", help="corpus profile in config/pipeline.yaml")
    ap.add_argument("--areas", nargs="*", help="restrict to these domain areas (config/domain.yaml)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-query (costs API credits)")
    ap.add_argument("--mode", default="auto", choices=["auto", "search", "topic_fallback"])
    args = ap.parse_args()
    with run_stage("acquire_openalex", vars(args)) as run:
        openalex.acquire(run, profile=args.profile, areas=args.areas, refresh=args.refresh, mode=args.mode)


if __name__ == "__main__":
    main()
