"""Compute the technical depth tier and scientific-quality dimensions of every experiment (depth-0.1.0).

Runs after all experiment producers (claims, deep extraction, artifacts, OpenML, MLPerf). Deterministic.
"""

import argparse
from collections import Counter, defaultdict

from services.datasets.depth import assess
from services.datasets.schemas import Experiment
from services.datasets.store import DATA, path, read_jsonl, read_models, run_stage, write_jsonl


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("assess_depth") as run:
        exps = read_models(path("experiments"), Experiment)
        evidence = {e["evidence_id"]: e for f in sorted((DATA / "silver" / "evidence").glob("*.jsonl")) for e in read_jsonl(f)}
        assoc: dict[str, list[str]] = defaultdict(list)
        for a in read_jsonl(path("associations")):
            for eid in [a.get("experiment_id"), *a.get("candidate_experiment_ids", [])]:
                if eid:
                    assoc[eid].append(a["status"] if a.get("experiment_id") == eid else "AMBIGUOUS")
        # artifact relations reach an experiment directly, through the paper evidence it cites, or (no paper value) per paper
        art_exp: dict[str, list[str]] = defaultdict(list)
        art_ev: dict[str, list[str]] = defaultdict(list)
        art_paper: dict[str, list[str]] = defaultdict(list)
        for r in read_jsonl(path("artifact_evidence")):
            if r.get("experiment_id"):
                art_exp[r["experiment_id"]].append(r["relation"])
            for i in r.get("paper_evidence_ids", []):
                art_ev[i].append(r["relation"])
            if not r.get("paper_evidence_ids"):
                art_paper[r["paper_id"]].append(r["relation"])
        tiers: Counter = Counter()
        for e in exps:
            rels = art_exp.get(e.experiment_id, []) + art_paper.get(e.paper_id, [])
            rels += [x for _, fv in e.iter_fields() for i in fv.evidence_ids for x in art_ev.get(i, [])]
            e.depth, e.quality = assess(e, evidence, assoc.get(e.experiment_id, []), rels)
            tiers[e.depth.tier.value] += 1
            tiers[f"evidence_{e.quality.evidence_quality.value}"] += 1
            tiers[f"attribution_{e.quality.experiment_attribution_quality.value}"] += 1
        run.output(write_jsonl(path("experiments"), exps))
        run.counts.update(dict(tiers), experiments=len(exps))
        print(dict(sorted(tiers.items())))


if __name__ == "__main__":
    main()
