"""Reproducible end-to-end dataset build (DATASET_SPEC §55): runs every stage script in order.

    uv run python scripts/data/run_pipeline.py --profile initial
    uv run python scripts/data/run_pipeline.py --from normalize_metadata     # offline replay from cached raw data

Every acquisition stage is cached and resumable, so re-running replays cached raw payloads without
new downloads. The build stops at the first failing stage (validation failures stop it: spec §46).
Review/promotion (build_review_queue, record_review, promote_gold) is a separate human loop.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES: list[tuple[str, list[str]]] = [
    ("export_schemas", []),
    ("acquire_openalex", ["--profile", "{profile}"]),
    ("acquire_arxiv_oai", []),
    ("build_discovery_corpus", []),
    ("normalize_metadata", []),
    ("deduplicate_papers", []),
    ("acquire_semantic_scholar", []),
    ("acquire_arxiv", []),
    ("acquire_papers_with_code", []),
    ("normalize_metadata", []),
    ("deduplicate_papers", []),
    ("filter_mlcs", []),
    ("acquire_fulltext", ["--sample-size", "{fulltext}"]),
    ("parse_fulltext", []),
    ("extract_experiments", []),
    ("extract_evidence", []),
    ("extract_deep", []),
    ("link_artifacts", []),
    ("assess_depth", []),
    ("build_study_links", []),
    ("build_comparison_candidates", []),
    ("build_hard_negatives", []),
    ("build_contradictions", []),
    ("build_abstention_cases", []),
    ("build_knowledge_graph", []),
    ("build_research_tasks", []),
    ("promote_gold", []),
    ("validate_dataset", []),
    ("detect_leakage", []),
    ("check_licenses", []),
    ("generate_statistics", []),
    ("build_manifest", []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--fulltext", default="0", help="full-text sample size (0 = whole selection-0.1.0 queue)")
    ap.add_argument("--from", dest="start", help="first stage to run (first occurrence)")
    ap.add_argument("--to", dest="stop", help="last stage to run (last occurrence)")
    args = ap.parse_args()
    names = [s for s, _ in STAGES]
    i0 = names.index(args.start) if args.start else 0
    i1 = len(names) - 1 - names[::-1].index(args.stop) if args.stop else len(names) - 1
    for name, extra in STAGES[i0 : i1 + 1]:
        cmd = [sys.executable, str(HERE / f"{name}.py"), *[a.format(profile=args.profile, fulltext=args.fulltext) for a in extra]]
        print(f"\n=== {name} {' '.join(cmd[2:])}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"stage {name} failed with exit code {rc}; stopping", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
