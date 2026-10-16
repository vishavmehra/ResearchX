"""Paper-centred knowledge graph (deepening §38), derived from records on every build - never edited by hand.

Every edge names the records / evidence it was derived from. Only relations the records support are emitted:
experiment -> result only for VERIFIED / LIKELY associations; paper -> code only for official repositories.
"""

from __future__ import annotations

from services.canonicalization import entities
from services.datasets.schemas import (
    ArtifactType,
    AssociationStatus,
    CanonicalEntityType,
    Experiment,
    GraphEdge,
    GraphEdgeType,
    GraphNode,
    GraphNodeType,
)
from services.datasets.store import stable_id

GRAPH_VERSION = "graph-0.1.0"
ENTITY_NODE = {CanonicalEntityType.DATASET: GraphNodeType.Dataset, CanonicalEntityType.MODEL: GraphNodeType.Model,
               CanonicalEntityType.BENCHMARK: GraphNodeType.Benchmark, CanonicalEntityType.METHOD: GraphNodeType.Method,
               CanonicalEntityType.METRIC: GraphNodeType.Metric}
FIELD_EDGE = {"dataset.dataset_name": GraphEdgeType.EXPERIMENT_USES_DATASET, "model.model_name": GraphEdgeType.EXPERIMENT_USES_MODEL,
              "model.base_model": GraphEdgeType.EXPERIMENT_USES_MODEL, "method.method_name": GraphEdgeType.EXPERIMENT_USES_METHOD,
              "evaluation.benchmark": GraphEdgeType.EXPERIMENT_EVALUATES_ON}


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    def node(self, ntype: GraphNodeType, ref: str, label: str | None = None) -> str:
        nid = f"{ntype.value}:{ref}"
        if nid not in self.nodes:
            self.nodes[nid] = GraphNode(node_id=nid, node_type=ntype, ref_id=ref, label=(label or None) and label[:200], graph_version=GRAPH_VERSION)
        return nid

    def edge(self, etype: GraphEdgeType, source: str, target: str, derived_from: list[str]) -> None:
        eid = stable_id("edge", etype.value, source, target)
        if eid in self.edges:  # same relation derived twice: keep one edge, union the provenance
            old = self.edges[eid]
            old.derived_from = sorted(set(old.derived_from) | set(derived_from))
        else:
            self.edges[eid] = GraphEdge(edge_id=eid, edge_type=etype, source=source, target=target,
                                        derived_from=sorted(set(derived_from)), graph_version=GRAPH_VERSION)


def build(papers: dict, source_records: dict, exps: list[Experiment], results: dict, associations: list[dict],
          artifacts: dict, citations: list[dict], contradictions: list[dict]) -> Graph:
    """Inputs are record dicts (experiments as models). Returns the graph; caller writes nodes/edges."""
    g = Graph()
    for pid, p in papers.items():
        pn = g.node(GraphNodeType.Paper, pid, p.get("title"))
        g.edge(GraphEdgeType.PAPER_DESCRIBES_STUDY, pn, g.node(GraphNodeType.Study, p["study_id"]), [pid])
        for sid in p.get("source_record_ids", []):
            if sid in source_records:
                g.edge(GraphEdgeType.PAPER_HAS_VERSION, pn, g.node(GraphNodeType.SourceRecord, sid, source_records[sid].get("source_name")), [pid, sid])
    for e in exps:
        en = g.node(GraphNodeType.Experiment, e.experiment_id, e.description)
        g.edge(GraphEdgeType.STUDY_HAS_EXPERIMENT, g.node(GraphNodeType.Study, e.study_id), en, [e.experiment_id])
        for sid in e.source_record_ids:
            if sid in source_records:
                g.edge(GraphEdgeType.SOURCE_REPORTS_EXPERIMENT, g.node(GraphNodeType.SourceRecord, sid), en, [e.experiment_id])
        for etype, fp, raw, ev_ids in entities.surface_forms(e):
            if fp in FIELD_EDGE and etype in ENTITY_NODE:
                g.edge(FIELD_EDGE[fp], en, g.node(ENTITY_NODE[etype], entities.entity_id(etype, raw), raw), [e.experiment_id, *ev_ids])
    for a in associations:
        if a["status"] in (AssociationStatus.VERIFIED, AssociationStatus.LIKELY) and a.get("experiment_id") and a["result_id"] in results:
            r = results[a["result_id"]]
            rn = g.node(GraphNodeType.Result, r["result_id"], f"{r.get('metric') or '?'} = {r['raw_value']}")
            g.edge(GraphEdgeType.EXPERIMENT_PRODUCES_RESULT, g.node(GraphNodeType.Experiment, a["experiment_id"]), rn,
                   [a["association_id"], *a.get("evidence_ids", [])])
            mid = entities.entity_id(CanonicalEntityType.METRIC, r.get("metric") or "")
            if mid:
                g.edge(GraphEdgeType.RESULT_USES_METRIC, rn, g.node(GraphNodeType.Metric, mid, r["metric"]), [r["result_id"]])
    for aid, a in artifacts.items():
        if a["artifact_type"] == ArtifactType.CODE_REPOSITORY:
            rn = g.node(GraphNodeType.CodeRepository, aid, a.get("repository"))
            if a.get("official"):
                for pid in a.get("linked_paper_ids", []):
                    if pid in papers:
                        g.edge(GraphEdgeType.PAPER_LINKS_CODE, g.node(GraphNodeType.Paper, pid), rn, [aid])
        elif a.get("parent_artifact_id") in artifacts:
            g.edge(GraphEdgeType.ARTIFACT_CONTAINS_CONFIG, g.node(GraphNodeType.CodeRepository, a["parent_artifact_id"]),
                   g.node(GraphNodeType.Artifact, aid, a.get("path")), [aid])
    for c in citations:
        if c["citing_paper_id"] in papers and c["cited_paper_id"] in papers:
            g.edge(GraphEdgeType.PAPER_CITES_PAPER, g.node(GraphNodeType.Paper, c["citing_paper_id"]), g.node(GraphNodeType.Paper, c["cited_paper_id"]),
                   [c["source_record_id"]])
    for k in contradictions:
        if k["source_a"] in source_records and k["source_b"] in source_records:
            g.edge(GraphEdgeType.SOURCE_CONFLICTS_WITH_SOURCE, g.node(GraphNodeType.SourceRecord, k["source_a"]),
                   g.node(GraphNodeType.SourceRecord, k["source_b"]), [k["contradiction_id"]])
    return g
