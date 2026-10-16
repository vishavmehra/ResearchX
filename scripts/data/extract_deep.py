"""Deep extraction (llm-extract-0.2.0) + table-first experiment-result association for every parsed paper.

Model calls go through the gateway (NEBIUS_API_KEY from the environment only) and are cached; a re-run without the
key replays cached outputs and records cache misses as failures. Rewrites silver/experiments/experiments.jsonl
(claims enriched + new model-identified experiments) and writes deep_evidence, results and associations.
"""

import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.datasets.schemas import Experiment, Paper
from services.datasets.store import config, path, read_jsonl, read_models, run_stage, write_jsonl
from services.evidence.assertions import load_parsed_docs
from services.extraction import associate, deep
from services.model_gateway.gateway import ModelGateway


class Budget:
    def __init__(self, limit: int):
        self.limit, self.used, self.lock = limit, 0, threading.Lock()

    def take(self, n: int = 2) -> bool:
        with self.lock:
            if self.used + n > self.limit:
                return False
            self.used += n
            return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="process at most N parsed papers (0 = all)")
    args = ap.parse_args()
    cfg = config("pipeline")["deep_extraction"]
    with run_stage("extract_deep", vars(args)) as run:
        papers = {p.paper_id: p for p in read_models(path("papers"), Paper)}
        docs = load_parsed_docs()
        exps = read_models(path("experiments"), Experiment)
        by_paper: dict[str, list[Experiment]] = {}
        for e in exps:
            by_paper.setdefault(e.paper_id, []).append(e)
        # structured (HTML) documents only: PDF parses have no tables/sections and invite invented locations
        todo = sorted(pid for pid in docs if pid in papers and docs[pid].format.value == "HTML")
        if args.limit:
            todo = todo[: args.limit]
        gw = ModelGateway()
        budget = Budget(cfg["max_live_calls"])
        cached_before = {p.name for p in deep.CACHE.glob("*.json")} if deep.CACHE.exists() else set()

        def work(pid: str):
            doc, paper = docs[pid], papers[pid]
            if not budget.take():
                return pid, [], []
            return pid, deep.extract_settings(doc, paper, gw, run), deep.identify_experiments(doc, paper, gw, run)

        outputs: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            futures = [pool.submit(work, pid) for pid in todo]
            for i, f in enumerate(as_completed(futures), 1):
                pid, settings, specs = f.result()
                outputs[pid] = (settings, specs)
                if i % 25 == 0:
                    print(f"  deep extraction {i}/{len(todo)}", flush=True)

        counts: Counter = Counter()
        all_exps: list[Experiment] = []
        results, assocs, evidence = [], [], []
        for pid, group in sorted(by_paper.items()):
            if pid not in outputs:
                all_exps.extend(group)
        for pid in todo:
            settings, specs = outputs[pid]
            out = associate.assemble(papers[pid], docs[pid], by_paper.get(pid, []), specs, settings, run)
            all_exps.extend(out.experiments)
            results += out.results
            assocs += out.associations
            evidence += out.evidence
            counts.update(out.counts)
            counts["papers_processed"] += 1
            counts["settings_accepted"] += len(settings)
            counts["specs_accepted"] += len(specs)
        # an identical assertion another stage already stored (same content-derived id) is cited there, not written twice
        others = {r["evidence_id"] for k in ("experiment_evidence", "rules_evidence", "metadata_evidence") for r in read_jsonl(path(k))}
        uniq = {e.evidence_id: e for e in evidence if e.evidence_id not in others}
        counts["evidence_already_stored_elsewhere"] = len({e.evidence_id for e in evidence} & others)
        all_exps.sort(key=lambda e: e.experiment_id)
        run.output(write_jsonl(path("experiments"), all_exps))
        run.output(write_jsonl(path("deep_evidence"), [uniq[k] for k in sorted(uniq)]))
        run.output(write_jsonl(path("results"), sorted({r.result_id: r for r in results}.values(), key=lambda r: r.result_id)))
        run.output(write_jsonl(path("associations"), sorted({a.association_id: a for a in assocs}.values(), key=lambda a: a.association_id)))
        cached_after = {p.name for p in deep.CACHE.glob("*.json")} if deep.CACHE.exists() else set()
        counts["llm_new_cache_entries"] = len(cached_after - cached_before)
        counts["llm_budget_used"] = budget.used
        counts["experiments_total"] = len(all_exps)
        run.counts.update({k: int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
