"""Experiment comparison candidates with field-level Silver labels (DATASET_SPEC §15, §16, §40)."""

import argparse
from collections import Counter

from services.comparison.candidates import comparison_candidates
from services.datasets.schemas import CitationEdge, Experiment
from services.datasets.store import path, read_models, run_stage, write_jsonl


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_comparison_candidates") as run:
        exps = read_models(path("experiments"), Experiment)
        cites = {(c.citing_paper_id, c.cited_paper_id) for c in read_models(path("citations"), CitationEdge)}
        comps = comparison_candidates(exps, cites)
        run.output(write_jsonl(path("comparisons"), comps))
        run.counts.update(comparisons=len(comps), hard_negatives=sum(c.is_hard_negative for c in comps),
                          **{f"label_{k}": v for k, v in Counter(c.relationship_label.value for c in comps).items()})


if __name__ == "__main__":
    main()
