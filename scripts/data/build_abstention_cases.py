"""Abstention cases: SUFFICIENT / INSUFFICIENT / CONFLICTING evidence (DATASET_SPEC §19)."""

import argparse
from collections import Counter

from services.comparison.conflicts import abstention_cases
from services.datasets.schemas import Comparison, ConflictType, Contradiction, Experiment
from services.datasets.store import path, read_models, run_stage, write_jsonl
from services.evidence.assertions import load_parsed_docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap-per-class", type=int, default=400)
    args = ap.parse_args()
    with run_stage("build_abstention_cases", vars(args)) as run:
        comps = read_models(path("comparisons"), Comparison) + read_models(path("hard_negatives"), Comparison)
        exps = {e.experiment_id: e for e in read_models(path("experiments"), Experiment)}
        numeric = [k for k in read_models(path("contradictions"), Contradiction) if k.conflict_type is ConflictType.NUMERIC_CONFLICT]
        cases = abstention_cases(comps, exps, set(load_parsed_docs()), numeric, args.cap_per_class)
        run.output(write_jsonl(path("abstention"), cases))
        run.counts.update(cases=len(cases),
                          **{f"class_{k}": v for k, v in Counter(c.evidence_class.value for c in cases).items()},
                          **{f"behavior_{k}": v for k, v in Counter(c.expected_behavior.value for c in cases).items()})


if __name__ == "__main__":
    main()
