"""Regenerate Gold from the append-only decision log under the review policy (DATASET_SPEC §25, §26).

Gold = pure function of recorded HUMAN decisions; every promotion is written to
data/annotations/promotions/promotions.jsonl. AGENT decisions never promote.
"""

import argparse
import json

from services.datasets.review import agreement_report, promote
from services.datasets.store import DATA, atomic_write_bytes, rel, run_stage


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("promote_gold") as run:
        run.counts.update(promote())
        report = agreement_report()  # kappa / annotation time from human annotators only; empty until humans review
        out = DATA / "reports" / "review_agreement.json"
        atomic_write_bytes(out, json.dumps(report, indent=1, sort_keys=True).encode())
        run.note(f"agreement report: {rel(out)}")
        run.counts["human_annotator_decisions"] = report["human_annotator_decisions"]


if __name__ == "__main__":
    main()
