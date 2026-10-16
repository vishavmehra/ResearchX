"""Research tasks (14 types) and hard research cases over real records (deepening §24-25). Deterministic; Silver only."""

import argparse
from collections import Counter

from services.datasets.schemas import Experiment
from services.datasets.store import DATA, path, read_jsonl, read_models, run_stage, write_jsonl
from services.datasets.tasks import Builder
from services.retrieval.fulltext import load_index


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_research_tasks") as run:
        exps = read_models(path("experiments"), Experiment)
        has_text = {f"paper:arxiv:{a}" for a, row in load_index().items() if row.get("fulltext", {}).get("parsed_path")}
        comps = read_jsonl(path("comparisons")) + read_jsonl(path("hard_negatives"))
        evidence = {e["evidence_id"]: e for f in sorted((DATA / "silver" / "evidence").glob("*.jsonl")) for e in read_jsonl(f)}
        b = Builder(exps, has_text)
        tasks = (b.comparison_tasks(comps) + b.ablation_tasks() + b.link_tasks(read_jsonl(path("study_links")), {p["paper_id"]: p for p in read_jsonl(path("papers"))})
                 + b.contradiction_tasks(read_jsonl(path("contradictions"))) + b.evidence_tasks(evidence) + b.protocol_tasks() + b.search_tasks())
        tasks = list({t.task_id: t for t in tasks}.values())
        cases = list({h.case_id: h for h in b.hard_cases(comps)}.values())
        run.output(write_jsonl(path("research_tasks"), sorted(tasks, key=lambda t: t.task_id)))
        run.output(write_jsonl(path("hard_cases"), sorted(cases, key=lambda h: h.case_id)))
        run.counts.update(research_tasks=len(tasks), hard_cases=len(cases),
                          **{f"task_{k}": v for k, v in Counter(t.task_type.value for t in tasks).items()},
                          **{f"behavior_{k}": v for k, v in Counter(t.expected_behavior.value for t in tasks).items()},
                          **{f"hard_{k}": v for k, v in Counter(h.case_type.value for h in cases).items()})


if __name__ == "__main__":
    main()
