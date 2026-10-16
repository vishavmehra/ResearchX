"""Harvest arXiv OAI-PMH (arXivRaw) metadata for the configured ML sets and windows (Layer 0 discovery).

One request every >= 3 s (arXiv legacy-API terms). Every page is cached (gzip) and replayable, so an
interrupted harvest resumes from the cache. --sets limits the harvest to some sets (comma-separated).
"""

import argparse

from services.datasets.store import run_stage
from services.retrieval.sources import arxiv_oai


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sets", default="", help="comma-separated OAI setSpecs (default: config)")
    args = ap.parse_args()
    with run_stage("acquire_arxiv_oai", vars(args)) as run:
        arxiv_oai.acquire(run, refresh=args.refresh, sets=[s for s in args.sets.split(",") if s] or None)


if __name__ == "__main__":
    main()
