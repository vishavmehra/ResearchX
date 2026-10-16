"""Build the Layer 0 discovery corpus from the arXiv OAI-PMH harvest and promote candidates for full text.

Writes data/normalized/discovery/candidates.jsonl and data/manifests/selection/main_corpus.json.
Pure function of the cached harvest + the current corpus ids (no network).
"""

import argparse
import json

from services.datasets.store import run_stage
from services.retrieval import discovery


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_discovery_corpus") as run:
        rows, selection = discovery.build()
        run.output(discovery.write(rows, selection))
        run.counts.update(selection["counts"])
        print(json.dumps(selection["counts"], indent=1))


if __name__ == "__main__":
    main()
