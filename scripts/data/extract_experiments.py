"""Stage 6: experiment extraction -> Silver (DATASET_SPEC §34 stage 6, §37, §38).

1. Papers With Code leaderboard claims for relevant papers -> experiment candidates
   (SOURCE_IMPORT), cross-checked against the paper's parsed tables where full text exists.
2. Model-assisted extraction (NVIDIA Nemotron via Nebius, through the model gateway) for parsed
   papers when NEBIUS_API_KEY is set. Without the key the step is skipped and recorded.
Nothing here writes Gold.
"""

import argparse
from collections import Counter

from services.datasets.schemas import Paper, SourceRecord
from services.datasets.store import path, read_models, run_stage, write_jsonl
from services.evidence.assertions import load_parsed_docs
from services.extraction import experiments, llm
from services.model_gateway.gateway import ModelGateway


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-v1", action="store_true", help="also run the superseded v0.1.0 chunked LLM extraction")
    ap.add_argument("--llm-limit", type=int, default=10, help="max parsed papers sent to the model (with --llm-v1)")
    args = ap.parse_args()
    with run_stage("extract_experiments", vars(args)) as run:
        papers = [p for p in read_models(path("papers"), Paper) if p.relevance and p.relevance.is_relevant]
        records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        docs = load_parsed_docs()
        exps, evs, stats = [], [], Counter()
        for p in papers:
            rec = records.get(f"pwc:{p.arxiv_id}") if p.arxiv_id else None
            if rec:
                e, v, s = experiments.build(p, rec, docs.get(p.paper_id), run)
                exps += e
                evs += v
                stats.update(s)
                stats["papers_with_pwc_experiments"] += bool(e)

        gateway = ModelGateway()
        parsed_relevant = [p for p in papers if p.paper_id in docs]
        if not args.llm_v1:
            run.note("v0.1.0 LLM extraction not run (superseded by extract_deep)")
        elif not gateway.available(llm.TASK):
            run.note(f"LLM extraction skipped: NEBIUS_API_KEY not set ({len(parsed_relevant)} parsed papers pending)")
            stats["llm_skipped_no_api_key"] = len(parsed_relevant)
        else:
            for p in parsed_relevant[: args.llm_limit]:
                e, v = llm.extract(docs[p.paper_id], p, gateway, run)
                exps += e
                evs += v
                stats["llm_papers"] += 1
                stats["llm_experiments"] += len(e)

        exps.sort(key=lambda e: e.experiment_id)
        evs = sorted({e.evidence_id: e for e in evs}.values(), key=lambda e: e.evidence_id)
        run.output(write_jsonl(path("experiments"), exps))
        run.output(write_jsonl(path("experiment_evidence"), evs))
        run.counts.update(
            relevant_papers=len(papers),
            parsed_relevant_papers=len(parsed_relevant),
            experiments=len(exps),
            outcomes=sum(len(e.outcomes) for e in exps),
            evidence=len(evs),
            **stats,
        )


if __name__ == "__main__":
    main()
