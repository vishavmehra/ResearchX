"""Render data/reports/source_research.md and docs/DATASET_SOURCE_MAP.md from the source registry.

data/manifests/source_discovery.yaml is the single source of truth; both documents are regenerated
from it (and from the live probe log data/reports/source_probes.json), so they cannot drift apart.

Usage: uv run python scripts/research/render_source_reports.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.datasets.store import DATA, ROOT, atomic_write_bytes, load_yaml, utf8_stdio  # noqa: E402

REGISTRY = DATA / "manifests" / "source_discovery.yaml"
PROBES = DATA / "reports" / "source_probes.json"
RESEARCH_MD = DATA / "reports" / "source_research.md"
MAP_MD = ROOT / "docs" / "DATASET_SOURCE_MAP.md"

DIMS = ["discovery", "fulltext", "experiment_depth", "configuration_depth", "evidence", "artifact", "study_link", "comparison",
        "retrieval", "temporal", "coverage", "license_quality", "api_quality", "maintenance", "duplication_risk"]
SHORT = {"HIGH": "H", "MEDIUM": "M", "LOW": "L", "NONE": "-", "UNKNOWN": "?"}
REC_ORDER = ["USE_NOW", "USE_LATER", "EVALUATION_ONLY", "DO_NOT_USE", "NEEDS_MANUAL_LICENSE_REVIEW"]
REC_TITLE = {"USE_NOW": "USE NOW", "USE_LATER": "USE LATER", "EVALUATION_ONLY": "EVALUATION ONLY", "DO_NOT_USE": "DO NOT USE",
             "NEEDS_MANUAL_LICENSE_REVIEW": "NEEDS MANUAL LICENSE REVIEW"}

# Which sources can supply each edge of the paper-centred graph (deepening program §15). Analysis, kept
# here beside the registry it summarises; source ids must exist in the registry (checked at render time).
EDGE_SUPPLY = [
    ("PAPER -> PAPER VERSION", ["ARXIV", "ARXIV_OAI_PMH", "OPENREVIEW", "CROSSREF", "DATACITE"], "arXiv vN history; OpenReview revisions; Crossref/DataCite is-version-of / has-preprint"),
    ("PAPER -> AUTHOR", ["OPENALEX", "SEMANTIC_SCHOLAR", "CROSSREF", "ARXIV"], "ORCID where present; author strings otherwise (no identity merge on names alone)"),
    ("PAPER -> STUDY", ["CROSSREF", "OPENREVIEW", "DBLP", "ACL_ANTHOLOGY", "ARXIV"], "explicit preprint->published relations, venue versions, journal_ref/DOI on arXiv"),
    ("PAPER -> CODE REPOSITORY", ["PAPERS_WITH_CODE_ARCHIVE", "HUGGINGFACE_PAPERS", "GITHUB", "ARXIV"], "PwC official links (frozen 2025-07), HF Papers links, URLs in arXiv comments / paper text"),
    ("CODE -> COMMIT / RELEASE", ["GITHUB", "ZENODO", "SOFTWARE_HERITAGE"], "commit sha + tags; Zenodo version DOIs; SWHIDs"),
    ("PAPER -> MODEL", ["HUGGINGFACE_PAPERS", "HUGGINGFACE_HUB", "EPOCH_AI"], "HF arXiv tags (many merely cite the paper); Epoch model rows link papers"),
    ("MODEL -> CHECKPOINT", ["HUGGINGFACE_HUB"], "repo revision sha, base_model tags"),
    ("EXPERIMENT -> DATASET", ["ARXIV_HTML", "HUGGINGFACE_DATASETS", "PAPERS_WITH_CODE_ARCHIVE", "OPENML", "DATACITE"], "paper text/tables first; catalogs canonicalize names and splits"),
    ("EXPERIMENT -> BENCHMARK", ["ARXIV_HTML", "PAPERS_WITH_CODE_ARCHIVE", "MLCOMMONS_MLPERF", "HELM", "MTEB"], "benchmark name + version + split"),
    ("STUDY -> EXPERIMENT (configuration)", ["ARXIV_HTML", "GITHUB", "HUGGINGFACE_HUB", "OPENML", "MLCOMMONS_MLPERF"], "paper claims vs repository/card evidence kept as separate relations"),
    ("EXPERIMENT -> RESULT", ["ARXIV_HTML", "PAPERS_WITH_CODE_ARCHIVE", "HUGGINGFACE_HUB", "EVERY_EVAL_EVER", "MLCOMMONS_MLPERF", "OPENML"], "table cell (paper) > model-index / leaderboard claim"),
    ("PAPER -> SUPPLEMENTARY DATA", ["ZENODO", "DATACITE", "FIGSHARE", "OPENAIRE_GRAPH"], "isSupplementTo / IsSupplementedBy relations"),
    ("PAPER -> CITING PAPER", ["OPENALEX", "SEMANTIC_SCHOLAR", "OPENCITATIONS", "CROSSREF", "UNARXIVE"], "citation edges; unarXive adds citation contexts"),
]


def load():
    reg = load_yaml(REGISTRY)
    probes = json.loads(PROBES.read_text(encoding="utf-8")) if PROBES.exists() else {}
    ids = {s["source_id"] for s in reg["sources"]}
    for edge, sources, _ in EDGE_SUPPLY:
        missing = [s for s in sources if s not in ids]
        if missing:
            raise SystemExit(f"EDGE_SUPPLY {edge!r} names unknown sources {missing}")
    return reg, probes


def probe_summary(s: dict, probes: dict) -> str:
    refs = [v.split(":", 1)[1] for v in s.get("verification", []) if v.startswith("probe:")]
    if not refs:
        return "no live probe (web/docs research only)"
    parts = []
    for r in refs:
        p = probes.get(r)
        parts.append(f"{r}={p.get('status') if p else 'MISSING'}")
    return ", ".join(parts)


def flags(d: dict) -> str:
    on = [k for k, v in d.items() if v is True]
    part = [k for k, v in d.items() if v == "partial"]
    return ", ".join(on + [f"{k} (partial)" for k in part]) or "none"


def bullet_block(s: dict, probes: dict) -> list[str]:
    a = s["assessment"]
    return [
        f"### {s['name']} (`{s['source_id']}`)",
        "",
        "| Item | Assessment |",
        "|---|---|",
        f"| URL / API | {s['url']} / `{s.get('api_url', 'n/a')}` |",
        f"| Data type | {s['source_type']}; data: {flags(s['data'])}; links: {flags(s['links'])} |",
        f"| Estimated size | {s['estimated_scale']} |",
        f"| Update cadence | {s['update_frequency']} |",
        f"| Coverage | {s['domain']} ({s['temporal_coverage']}) |",
        f"| Technical depth / experiment depth | {s['technical_depth']} / {a['experiment_depth']} (configuration {a['configuration_depth']}) |",
        f"| Full-text access / artifact access | {a['fulltext']} / {a['artifact']} |",
        f"| Licence | {s['license']} |",
        f"| Training permission | {s['training_allowed']} |",
        f"| Redistribution permission | {s['redistribution_allowed']} |",
        f"| Rate limits | {s['rate_limit']} |",
        f"| Identifier quality | {s['entity_linkability']} |",
        f"| Study-link / experiment-link / evidence usefulness | {a['study_link']} / {a['comparison']} / {a['evidence']} |",
        f"| ResearchX priority / recommendation | **{s['priority']} / {REC_TITLE[s['recommendation']]}** |",
        f"| Live probes | {probe_summary(s, probes)} |",
        f"| Status | {s['status']} |",
        f"| Risks | {s['risks']} |",
        "",
    ]


def render_research(reg: dict, probes: dict) -> str:
    S = reg["sources"]
    ok = sum(1 for p in probes.values() if p.get("ok"))
    failed = sorted((k, p.get("status") or p.get("error")) for k, p in probes.items() if not p.get("ok"))
    L = [
        "# ResearchX source research report",
        "",
        f"Assessed {reg['assessed_at']}. Rendered from `data/manifests/source_discovery.yaml` "
        "(single source of truth) by `scripts/research/render_source_reports.py`. Do not edit by hand.",
        "",
        "## Method",
        "",
        "- Every \"API works\" statement rests on a live, unauthenticated request recorded in "
        "`data/reports/source_probes.json` (status, latency, rate-limit headers, compact observation).",
        "- Licences were read from terms pages, dataset cards, README files and LICENSE files. A code repository's "
        "licence is never assumed to cover its data. Anything not read is UNVERIFIED, which means: never train on it, "
        "never redistribute it.",
        "- Each source is assessed on 15 separate dimensions. There is no combined score.",
        "- No source was bulk-downloaded during this research. Probes fetched single records or headers only.",
        "",
        f"Live probes: **{ok} of {len(probes)} succeeded**. Failures (kept as findings, not hidden):",
        "",
    ]
    L += [f"- `{k}`: {v}" for k, v in failed] + [""]
    L += ["## Headline findings", ""]
    L += [
        "- arXiv's new OAI-PMH endpoint returns licence-tagged metadata in bulk (1300 cs records per 2-3 days in one probe). It is the cheapest licence-clean route to the 20K-50K candidate target and fills the licence field that is unknown for 1,478 of 1,521 v0.1.0 papers.",
        "- arXiv HTML (LaTeXML) is the main depth source: section, paragraph, table-cell and footnote ids for configuration and results.",
        "- OpenML runs and MLPerf result logs are the only open sources found with fully specified, machine-readable experiment configurations; they enable real ALIGNED/SAME_EXPERIMENT comparisons that paper leaderboards cannot.",
        "- Crossref exposes 576,156 works with explicit has-preprint relations: the strongest open evidence for SAME_STUDY_DIFFERENT_RECORD links.",
        "- Papers with Code is frozen (sunset 2025-07-24); Hugging Face Papers/Hub replace its paper-code-model links, but many linked repositories merely cite a paper and must stay RELATED_BUT_UNCONFIRMED until verified.",
        "- HELM is in maintenance mode since 2026-06-01 (verified in its README); MLPerf training_results_v6.1 does not exist yet (404); OpenReview venue listings need login (403) while search works; DBLP's API is behind an anti-bot challenge but its XML dump downloads.",
        "- Licence traps found: ACL-OCL card tag says MIT but its body says CC BY-NC 4.0; MetaLead, HyperPIE, ArxivDIGESTables and PeerRead publish no licence; CORE requires a licence for API use; Open LLM Leaderboard datasets state no licence.",
        "",
        "## Recommendation lists",
        "",
    ]
    for rec in REC_ORDER:
        group = sorted((s for s in S if s["recommendation"] == rec), key=lambda s: (s["priority"], s["source_id"]))
        L += [f"### {REC_TITLE[rec]} ({len(group)})", ""]
        L += [f"- **{s['source_id']}** ({s['priority']}): {s['why_researchx_needs_it']}" for s in group] + [""]
    L += ["## Structured assessment matrix", "",
          "H = high, M = medium, L = low, - = none, ? = unknown. For duplication risk, H is bad.", "",
          "| Source | Rec | P | " + " | ".join(d.replace("_", " ") for d in DIMS) + " |",
          "|---|---|---|" + "---|" * len(DIMS)]
    for s in sorted(S, key=lambda s: (s["priority"], s["source_id"])):
        L.append(f"| {s['source_id']} | {REC_TITLE[s['recommendation']]} | {s['priority']} | "
                 + " | ".join(SHORT[s["assessment"][d]] for d in DIMS) + " |")
    L += ["", "## Licence and usage matrix", "",
          "| Source | Licence | Training | Evaluation | Redistribution | Attribution |", "|---|---|---|---|---|---|"]
    for s in sorted(S, key=lambda s: s["source_id"]):
        L.append(f"| {s['source_id']} | {s['license']} | {s['training_allowed']} | {s['evaluation_allowed']} | "
                 f"{s['redistribution_allowed']} | {s['attribution_required']} |")
    L += ["", "## Exact API / URL list", "", "| Source | Landing URL | API / download endpoint | Authentication |", "|---|---|---|---|"]
    for s in sorted(S, key=lambda s: s["source_id"]):
        L.append(f"| {s['source_id']} | {s['url']} | `{s.get('api_url', 'n/a')}` | {s['authentication_required']} |")
    L += ["", "## Per-source assessment", ""]
    for s in sorted(S, key=lambda s: (s["priority"], s["source_id"])):
        L += bullet_block(s, probes)
    return "\n".join(L) + "\n"


def render_map(reg: dict) -> str:
    S = reg["sources"]
    sections = reg["map_sections"]
    primary = {s["source_id"]: s["map_sections"][0] for s in S}
    L = [
        "# ResearchX dataset source map",
        "",
        f"Generated from `data/manifests/source_discovery.yaml` (assessed {reg['assessed_at']}) by "
        "`scripts/research/render_source_reports.py`. Evidence for each claim (live probes, licence texts) is listed "
        "in `data/reports/source_research.md`. Sources appear in full in their primary section and as cross-references "
        "elsewhere.",
        "",
        "## How sources feed the dataset layers",
        "",
        "| Layer | Main sources |",
        "|---|---|",
        "| L0 scholarly discovery | ARXIV_OAI_PMH, OPENALEX, ARXIV, SEMANTIC_SCHOLAR |",
        "| L1 normalized scholarly corpus | ARXIV, OPENALEX, CROSSREF, SEMANTIC_SCHOLAR, ACL_ANTHOLOGY |",
        "| L2 full-text corpus | ARXIV_HTML (structured); ACL_ANTHOLOGY 2016+; COMMON_PILE_ARXIV / S2ORC_PES2O later |",
        "| L3 document structure | ARXIV_HTML; DOCLING_TOOL / GROBID_TOOL for PDFs (later) |",
        "| L4 deep experiments | ARXIV_HTML + LLM/rule extraction; OPENML; MLCOMMONS_MLPERF |",
        "| L5 evidence + attribution | ARXIV_HTML spans and cells; artifact files; source records |",
        "| L6 artifact/config linkage | GITHUB, HUGGINGFACE_HUB, HUGGINGFACE_PAPERS, PAPERS_WITH_CODE_ARCHIVE, ZENODO |",
        "| L7 study linking | CROSSREF relations, ARXIV versions, OPENREVIEW, DBLP, ACL_ANTHOLOGY |",
        "| L8 comparison / contradictions / abstention | built from L4-L7; OPENML and MLPERF give fully specified pairs |",
        "| Evaluation-only | QASPER, LITSEARCH, BRIGHT, BEIR, SCITAB, RESEARCHQA, agent benchmarks |",
        "",
        "## Paper-centred graph: which sources can supply each edge",
        "",
        "| Edge | Sources | Notes |",
        "|---|---|---|",
    ]
    L += [f"| {e} | {', '.join(srcs)} | {note} |" for e, srcs, note in EDGE_SUPPLY]
    L += [""]
    for i, (key, title) in enumerate(sections.items(), 1):
        members = sorted((s for s in S if key in s["map_sections"]), key=lambda s: (s["priority"], s["source_id"]))
        L += [f"## {i}. {title}", ""]
        if not members:
            L += ["No source assessed for this section.", ""]
            continue
        for s in members:
            if primary[s["source_id"]] != key:
                L += [f"- **{s['name']}** (`{s['source_id']}`, {s['priority']}): see section "
                      f"{list(sections).index(primary[s['source_id']]) + 1} ({sections[primary[s['source_id']]]})."]
                continue
            L += [
                f"### {s['name']} (`{s['source_id']}`) - {s['priority']}, {REC_TITLE[s['recommendation']]}",
                "",
                f"- What we get: {s['what_we_get']}",
                f"- Why ResearchX needs it: {s['why_researchx_needs_it']}",
                f"- How to acquire: {s['acquisition_method']}",
                f"- Licence: {s['license']}",
                f"- API: {s.get('api_url', 'n/a')} (available: {s['api_available']}; bulk: {s['bulk_available']})",
                f"- Rate limits: {s['rate_limit']}",
                f"- Versioning: {s['versioning']}",
                f"- Expected technical depth: {s['technical_depth']} (configuration depth {s['assessment']['configuration_depth']})",
                f"- Entity linking: {s['entity_linkability']}",
                f"- Deduplication strategy: {s['dedup_strategy']}",
                f"- Training / evaluation usage: {s['usage']} (training: {s['training_allowed']}; evaluation: {s['evaluation_allowed']})",
                f"- Risks: {s['risks']}",
                f"- Recommended priority: {s['priority']}",
                "",
            ]
        L += [""]
    counts = Counter(s["recommendation"] for s in S)
    L += ["## Totals", "", f"{len(S)} sources assessed: " + ", ".join(f"{REC_TITLE[r]} {counts.get(r, 0)}" for r in REC_ORDER) + ".", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    utf8_stdio()
    reg, probes = load()
    atomic_write_bytes(RESEARCH_MD, render_research(reg, probes).encode("utf-8"))
    atomic_write_bytes(MAP_MD, render_map(reg).encode("utf-8"))
    print(f"wrote {RESEARCH_MD.relative_to(ROOT)} and {MAP_MD.relative_to(ROOT)} for {len(reg['sources'])} sources")


if __name__ == "__main__":
    main()
