"""Targeted hard negatives tagged by construction method (DATASET_SPEC §17).

Pairs already present in comparisons.jsonl are skipped (one record per pair per tier).
"""

import argparse
from collections import Counter

from services.comparison.candidates import hard_negatives
from services.datasets.schemas import CitationEdge, Experiment
from services.datasets.store import path, read_jsonl, read_models, run_stage, write_jsonl


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_hard_negatives") as run:
        exps = read_models(path("experiments"), Experiment)
        cites = {(c.citing_paper_id, c.cited_paper_id) for c in read_models(path("citations"), CitationEdge)}
        existing = {c["comparison_id"] for c in read_jsonl(path("comparisons"))}
        hns = hard_negatives(exps, cites, existing)
        run.output(write_jsonl(path("hard_negatives"), hns))
        run.counts.update(pairs=len(hns), tagged_hard_negatives=sum(c.is_hard_negative for c in hns),
                          **{f"type_{k}": v for k, v in Counter(c.hard_negative_type for c in hns).items()},
                          **{f"label_{k}": v for k, v in Counter(c.relationship_label.value for c in hns).items()})


if __name__ == "__main__":
    main()
