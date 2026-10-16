"""Canonical entities + paper-centred knowledge graph (deepening §37-38). Deterministic; derived from records."""

import argparse
from collections import Counter

from services.canonicalization import entities
from services.datasets import graph
from services.datasets.schemas import Experiment
from services.datasets.store import path, read_jsonl, read_models, run_stage, write_jsonl


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("build_knowledge_graph") as run:
        exps = read_models(path("experiments"), Experiment)
        ents = entities.build(exps)
        run.output(write_jsonl(path("canonical_entities"), ents))
        g = graph.build(
            papers={p["paper_id"]: p for p in read_jsonl(path("papers"))},
            source_records={s["source_record_id"]: s for s in read_jsonl(path("source_records"))},
            exps=exps,
            results={r["result_id"]: r for r in read_jsonl(path("results"))},
            associations=read_jsonl(path("associations")),
            artifacts={a["artifact_id"]: a for a in read_jsonl(path("artifacts"))},
            citations=read_jsonl(path("citations")),
            contradictions=read_jsonl(path("contradictions")),
        )
        run.output(write_jsonl(path("graph_nodes"), [g.nodes[k] for k in sorted(g.nodes)]))
        run.output(write_jsonl(path("graph_edges"), [g.edges[k] for k in sorted(g.edges)]))
        run.counts.update(entities=len(ents), nodes=len(g.nodes), edges=len(g.edges),
                          **{f"entity_{k}": v for k, v in Counter(e.entity_type.value for e in ents).items()},
                          **{f"edge_{k}": v for k, v in Counter(e.edge_type.value for e in g.edges.values()).items()})


if __name__ == "__main__":
    main()
