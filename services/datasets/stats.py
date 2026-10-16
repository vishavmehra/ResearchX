"""Dataset statistics (DATASET_SPEC §47). Computed from files on disk; never hand-entered."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from services.datasets.schemas import FIELD_GROUPS, OUTCOME_FIELDS
from services.datasets.store import DATA, atomic_write_bytes, now_utc, path, read_jsonl


def _dist(values) -> dict[str, int]:
    return dict(sorted(Counter(v if v is not None else "null" for v in values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _tier_files(kind: str) -> dict[str, list[dict]]:
    return {tier: [r for f in sorted((DATA / tier / kind).glob("*.jsonl")) for r in read_jsonl(f)] for tier in ("silver", "gold")}


def generate() -> dict[str, Any]:
    src = read_jsonl(path("source_records"))
    papers = read_jsonl(path("papers"))
    studies = read_jsonl(path("studies"))
    citations = read_jsonl(path("citations"))
    tiers = {k: _tier_files(k) for k in ("experiments", "evidence", "study_links", "comparisons", "contradictions", "abstention", "laya", "trajectories")}
    exps = tiers["experiments"]["silver"] + tiers["experiments"]["gold"]
    evs = tiers["evidence"]["silver"] + tiers["evidence"]["gold"]
    links = tiers["study_links"]["silver"] + tiers["study_links"]["gold"]
    comps = tiers["comparisons"]["silver"] + tiers["comparisons"]["gold"]
    contras = tiers["contradictions"]["silver"] + tiers["contradictions"]["gold"]
    abst = tiers["abstention"]["silver"] + tiers["abstention"]["gold"]
    laya = tiers["laya"]["silver"] + tiers["laya"]["gold"]
    traj = tiers["trajectories"]["silver"] + tiers["trajectories"]["gold"]
    relevant = [p for p in papers if (p.get("relevance") or {}).get("is_relevant")]

    missing_fields: dict[str, float] = {}
    if exps:
        for group, names in FIELD_GROUPS.items():
            for name in names:
                known = sum(1 for e in exps if (e.get(group, {}).get(name) or {}).get("status") == "KNOWN")
                missing_fields[f"{group}.{name}"] = round(1 - known / len(exps), 4)
        outcomes = [o for e in exps for o in e.get("outcomes", [])]
        for name in OUTCOME_FIELDS:
            known = sum(1 for o in outcomes if (o.get(name) or {}).get("status") == "KNOWN")
            missing_fields[f"outcome.{name}"] = round(1 - known / len(outcomes), 4) if outcomes else 1.0

    stats: dict[str, Any] = {
        "generated_at": now_utc().isoformat(),
        "counts": {
            "number_of_source_records": len(src),
            "number_of_papers": len(papers),
            "number_of_relevant_papers": len(relevant),
            "number_of_studies": len(studies),
            "number_of_citation_edges": len(citations),
            "number_of_fulltext_documents": sum(1 for p in papers if (p.get("fulltext") or {}).get("full_text_available")),
            "number_of_experiments": len(exps),
            "number_of_outcome_observations": sum(len(e.get("outcomes", [])) for e in exps),
            "number_of_evidence_assertions": len(evs),
            "number_of_comparisons": len(comps),
            "number_of_study_links": len(links),
            "number_of_same_study_pairs": sum(1 for l in links if l["relationship"] in ("SAME_UNDERLYING_STUDY", "SAME_STUDY_DIFFERENT_RECORD")),
            "number_of_hard_negatives": sum(1 for r in comps + links if r.get("is_hard_negative")),
            "number_of_contradictions": len(contras),
            "number_of_abstention_cases": len(abst),
            "number_of_laya_decisions": len(laya),
            "number_of_trajectories": len({t["trajectory_id"] for t in traj}),
        },
        "tier_distribution": {
            kind: {tier: len(rows) for tier, rows in by_tier.items()} for kind, by_tier in tiers.items()
        } | {"normalized_corpus": {"BRONZE": len(papers)}},
        "source_distribution": _dist(s["source_name"] for s in src),
        "papers_by_source_count": _dist(len(p["source_record_ids"]) for p in papers),
        "domain_distribution": _dist(a for p in relevant for a in p["relevance"]["areas"]),
        "temporal_distribution": dict(sorted(Counter(str(p.get("publication_year")) for p in papers).items())),
        "split_distribution": {
            "papers": _dist(p.get("split", "UNASSIGNED") for p in papers),
            "experiments": _dist(e.get("split", "UNASSIGNED") for e in exps),
            "comparisons": _dist(c.get("split", "UNASSIGNED") for c in comps),
        },
        "license_distribution": {
            "paper_content_license": _dist((p.get("license") or {}).get("content_license") for p in papers),
            "fulltext_storage": _dist((p.get("fulltext") or {}).get("license_status") for p in papers if p.get("fulltext")),
        },
        "evidence_status_distribution": _dist(v.get("status", "MACHINE_EXTRACTED") for v in evs),
        "evidence_by_extractor": _dist(v["extraction_model"] for v in evs),
        "evidence_by_field": _dist(v["field_name"] for v in evs),
        "experiment_extraction_method": _dist(e["extraction_method"] for e in exps),
        "label_distribution": {
            "comparisons": _dist(c["relationship_label"] for c in comps),
            "hard_negative_types": _dist(c.get("hard_negative_type") or c.get("construction_method") for c in comps + links if c.get("is_hard_negative")),
            "study_links": _dist(l["relationship"] for l in links),
            "contradiction_types": _dist(c["conflict_type"] for c in contras),
            "abstention_classes": _dist(a["evidence_class"] for a in abst),
            "abstention_behaviors": _dist(a["expected_behavior"] for a in abst),
        },
        "paper_metadata_missing": {
            f: round(sum(1 for p in papers if not p.get(f)) / len(papers), 4) if papers else None
            for f in ("abstract", "doi", "arxiv_id", "venue", "publication_date", "authors")
        },
        "missing_field_distribution": missing_fields,
        "metadata_conflicts": _dist(c["field"] for p in papers for c in p.get("metadata_conflicts", [])),
    }
    # schema 0.2.0 record types (deepening): depth, associations, artifacts, entities, graph, tasks
    v2 = {k: read_jsonl(path(k)) for k in ("discovery", "results", "associations", "artifacts", "artifact_evidence", "canonical_entities",
                                           "graph_nodes", "graph_edges", "research_tasks", "hard_cases")}
    stats["counts"].update({f"number_of_{k}": len(v) for k, v in v2.items()})
    known = sorted((e.get("depth") or {}).get("technical_fields_known", 0) for e in exps)
    stats["depth_distribution"] = _dist((e.get("depth") or {}).get("tier", "UNASSESSED") for e in exps)
    stats["technical_fields_known_per_experiment"] = (
        {"min": known[0], "median": known[len(known) // 2], "p90": known[int(len(known) * 0.9)], "max": known[-1]} if known else {})
    stats["quality_distribution"] = {d: _dist((e.get("quality") or {}).get(d) for e in exps)
                                     for d in ("evidence_quality", "experiment_attribution_quality", "configuration_completeness")}
    stats["association_status_distribution"] = _dist(a["status"] for a in v2["associations"])
    stats["artifact_relation_distribution"] = _dist(r["relation"] for r in v2["artifact_evidence"])
    stats["artifact_type_distribution"] = _dist(a["artifact_type"] for a in v2["artifacts"])
    stats["entity_type_distribution"] = _dist(e["entity_type"] for e in v2["canonical_entities"])
    stats["graph_edge_distribution"] = _dist(e["edge_type"] for e in v2["graph_edges"])
    stats["research_task_distribution"] = {"task_type": _dist(t["task_type"] for t in v2["research_tasks"]),
                                           "expected_behavior": _dist(t["expected_behavior"] for t in v2["research_tasks"])}
    stats["hard_case_distribution"] = _dist(h["case_type"] for h in v2["hard_cases"])
    return stats


def write(stats: dict[str, Any], out_dir: Path = DATA / "reports") -> tuple[Path, Path]:
    j = out_dir / "statistics.json"
    atomic_write_bytes(j, json.dumps(stats, indent=1, sort_keys=False, default=str).encode())
    lines = ["# ResearchX dataset statistics", "", f"Generated: {stats['generated_at']}", "", "| Count | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats["counts"].items()]
    for section in ("source_distribution", "domain_distribution", "evidence_status_distribution", "label_distribution", "split_distribution", "license_distribution",
                    "depth_distribution", "technical_fields_known_per_experiment", "quality_distribution", "association_status_distribution",
                    "artifact_relation_distribution", "entity_type_distribution", "graph_edge_distribution", "research_task_distribution",
                    "hard_case_distribution"):
        if section in stats:
            lines += ["", f"## {section}", "", "```json", json.dumps(stats[section], indent=1), "```"]
    md = out_dir / "statistics.md"
    atomic_write_bytes(md, ("\n".join(lines) + "\n").encode())
    return j, md
