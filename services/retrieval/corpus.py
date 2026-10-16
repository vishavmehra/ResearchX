"""Corpus building: identifier-only merge, relevance filter, splits, citations (spec §28, §29).

Merge policy (auditable, never on title similarity):
- A Paper is keyed by exactly one primary identifier: arXiv id (unversioned) if any source
  gives one, else DOI, else the source record itself.
- Records merge only when they share that primary key. Cross-identifier claims (e.g. S2 says
  arXiv X == DOI Y) become study-link candidates, not merges (built in services.comparison).
- Field values are chosen by documented precedence; disagreeing values are preserved in
  Paper.metadata_conflicts and become contradiction candidates.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from services.canonicalization.normalize import jaccard, normalize_title, parse_arxiv_id, title_tokens
from services.datasets.schemas import (
    CitationEdge,
    LicenseInfo,
    MergeEvidence,
    Paper,
    Provenance,
    RecordKind,
    Relevance,
    SourceRecord,
    Split,
    Study,
)
from services.datasets.store import code_version, config

MERGE_VERSION = "merge-0.1.0"
RELEVANCE_VERSION = "relevance-0.1.0"
TITLE_CONFLICT_JACCARD = 0.7

# Field precedence (first available wins). Documented in data/README.md.
PRECEDENCE: dict[str, tuple[str, ...]] = {
    "title": ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR", "OPENALEX", "PAPERS_WITH_CODE"),
    "abstract": ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR", "OPENALEX", "PAPERS_WITH_CODE"),
    "authors": ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR", "OPENALEX"),
    # arXiv v1 date first: earliest public availability is what temporal splits need (spec §52),
    # and OpenAlex dates were observed wrong for famous papers (1706.03762 dated 2025).
    "publication_date": ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR", "OPENALEX"),
    "publication_year": ("ARXIV", "ARXIV_OAI_PMH", "SEMANTIC_SCHOLAR", "OPENALEX"),
    "venue": ("SEMANTIC_SCHOLAR", "OPENALEX", "ARXIV"),
    "doi": ("OPENALEX", "SEMANTIC_SCHOLAR", "ARXIV", "ARXIV_OAI_PMH"),
}
ARXIV_SOURCES = ("ARXIV", "ARXIV_OAI_PMH")


def primary_key(rec: SourceRecord) -> tuple[str, str]:
    if rec.arxiv_id:
        return "arxiv_id", rec.arxiv_id
    if rec.doi and rec.doi.startswith("10.48550/arxiv.") and (p := parse_arxiv_id(rec.doi)):
        return "arxiv_id", p[0]
    if rec.doi:
        return "doi", rec.doi
    return "self", rec.source_record_id


def paper_id_for(kind: str, value: str) -> str:
    return {"arxiv_id": f"paper:arxiv:{value}", "doi": f"paper:doi:{value}"}.get(kind, f"paper:{value}")


def _pick(recs: list[SourceRecord], field: str) -> tuple[Any, str | None]:
    order = {s: i for i, s in enumerate(PRECEDENCE[field])}
    for rec in sorted(recs, key=lambda r: (order.get(r.source_name, 99), r.source_record_id)):
        value = getattr(rec, field)
        if value not in (None, "", []):
            return value, rec.source_record_id
    return None, None


def _conflicts(recs: list[SourceRecord]) -> list[dict[str, Any]]:
    """Preserve disagreeing metadata across sources (spec §44: never silently overwrite)."""
    out: list[dict[str, Any]] = []
    meta = [r for r in recs if r.record_kind is RecordKind.PAPER_METADATA]
    titles = {r.source_record_id: r.title for r in meta if r.title}
    toks = {k: title_tokens(v) for k, v in titles.items()}
    keys = sorted(toks)
    if any(jaccard(toks[a], toks[b]) < TITLE_CONFLICT_JACCARD for i, a in enumerate(keys) for b in keys[i + 1:]):
        out.append({"field": "title", "values": titles, "rule": f"title_token_jaccard<{TITLE_CONFLICT_JACCARD}"})
    years = {r.source_record_id: r.publication_year for r in meta if r.publication_year}
    if len(set(years.values())) > 1:
        out.append({"field": "publication_year", "values": years, "rule": "distinct_years"})
    counts = {r.source_record_id: len(r.authors) for r in meta if r.authors}
    if counts and max(counts.values()) - min(counts.values()) > 1:
        out.append({"field": "author_count", "values": counts, "rule": "author_count_diff>1"})
    dois = {r.source_record_id: r.doi for r in meta if r.doi and not r.doi.startswith("10.48550/")}
    if len(set(dois.values())) > 1:
        out.append({"field": "doi", "values": dois, "rule": "distinct_non_arxiv_dois"})
    return out


def build_papers(records: list[SourceRecord]) -> list[Paper]:
    groups: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    for rec in records:
        groups[primary_key(rec)].append(rec)
    papers = []
    for (kind, value), recs in sorted(groups.items()):
        recs.sort(key=lambda r: r.source_record_id)
        pid = paper_id_for(kind, value)
        fields, sources = {}, {}
        for f in PRECEDENCE:
            fields[f], src = _pick(recs, f)
            if src:
                sources[f] = src
        identifiers: dict[str, str] = {}
        for rec in recs:
            for k, v in rec.identifiers.items():
                identifiers.setdefault(k, v)
        arxiv_rec = next((r for r in recs if r.source_name == "ARXIV"), None)
        oai_rec = next((r for r in recs if r.source_name == "ARXIV_OAI_PMH"), None)
        # licence: full-text header licence (on the ARXIV record once fetched) > arXiv OAI-PMH licence field
        lic_rec = arxiv_rec if arxiv_rec and arxiv_rec.license.content_license else (oai_rec or arxiv_rec)
        title = fields["title"] or ""
        papers.append(
            Paper(
                paper_id=pid,
                study_id=pid.replace("paper:", "study:", 1),
                title=title,
                normalized_title=normalize_title(title),
                abstract=fields["abstract"],
                authors=fields["authors"] or [],
                publication_date=fields["publication_date"],
                publication_year=fields["publication_year"],
                venue=fields["venue"],
                doi=fields["doi"],
                arxiv_id=value if kind == "arxiv_id" else None,
                identifiers=identifiers,
                source_record_ids=[r.source_record_id for r in recs],
                field_sources=sources,
                merge_evidence=[
                    MergeEvidence(source_record_id=r.source_record_id, matched_on=kind if kind != "self" else "self", value=value)
                    for r in recs
                ],
                metadata_conflicts=_conflicts(recs),
                license=lic_rec.license if lic_rec else LicenseInfo(),
                fulltext=arxiv_rec.fulltext if arxiv_rec else None,
                provenance=Provenance(
                    source_name="RESEARCHX",
                    processing_version=code_version(),
                    extractor_version=MERGE_VERSION,
                    parent_record_ids=[r.source_record_id for r in recs],
                ),
            )
        )
    return papers


def _bucket(study_id: str, seed: str) -> float:
    return int(hashlib.sha256(f"{seed}:{study_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(papers: list[Paper]) -> None:
    """Split at the study level (spec §28): all records of one study share a split."""
    cfg = config("pipeline")["splits"]
    cutoff = cfg["temporal_cutoff"]
    for p in papers:
        if p.publication_date and p.publication_date >= cutoff:
            p.split = Split.TEMPORAL_TEST
            continue
        b = _bucket(p.study_id, cfg["seed"])
        p.split = Split.TRAIN if b < cfg["train"] else Split.VALIDATION if b < cfg["train"] + cfg["validation"] else Split.TEST


def build_studies(papers: list[Paper]) -> list[Study]:
    """Provisional 1:1 studies. Linking multiple papers to one study requires verified study links."""
    return [
        Study(
            study_id=p.study_id,
            title=p.title,
            paper_ids=[p.paper_id],
            source_record_ids=p.source_record_ids,
            split=p.split,
            provenance=Provenance(
                source_name="RESEARCHX",
                processing_version=code_version(),
                extractor_version=MERGE_VERSION,
                parent_record_ids=[p.paper_id],
            ),
        )
        for p in papers
    ]


def build_citations(papers: list[Paper], records: list[SourceRecord]) -> list[CitationEdge]:
    """Intra-corpus citation edges from OpenAlex referenced_works and Semantic Scholar reference ids (arXiv / DOI).
    The first source to produce a (citing, cited) pair keeps it; OpenAlex records are processed first."""
    by_ref = {p.identifiers["openalex"]: p.paper_id for p in papers if "openalex" in p.identifiers}
    by_ref |= {f"arxiv:{p.arxiv_id}": p.paper_id for p in papers if p.arxiv_id}
    by_ref |= {f"doi:{p.doi}": p.paper_id for p in papers if p.doi}
    paper_of = {srid: p.paper_id for p in papers for srid in p.source_record_ids}
    edges = {}
    for rec in sorted(records, key=lambda r: r.source_name != "OPENALEX"):
        if rec.source_name not in ("OPENALEX", "SEMANTIC_SCHOLAR"):
            continue
        citing = paper_of.get(rec.source_record_id)
        for ref in rec.references:
            cited = by_ref.get(ref)
            if citing and cited and cited != citing and (citing, cited) not in edges:
                edges[(citing, cited)] = CitationEdge(
                    citing_paper_id=citing,
                    cited_paper_id=cited,
                    source_record_id=rec.source_record_id,
                    provenance=Provenance(
                        source_name="RESEARCHX",
                        processing_version=code_version(),
                        extractor_version=MERGE_VERSION,
                        parent_record_ids=[rec.source_record_id],
                    ),
                )
    return [edges[k] for k in sorted(edges)]


def _compile_terms(area: dict) -> list[re.Pattern]:
    return [re.compile(t["pattern"], 0 if t.get("case_sensitive") else re.I) for t in area["terms"]]


def match_areas(text: str) -> dict[str, list[str]]:
    """Priority-area term hits in a text (config/domain.yaml); candidate-selection signal only."""
    matched: dict[str, list[str]] = {}
    for area_id, area in config("domain")["areas"].items():
        hits = sorted({m.group(0) for pat in _compile_terms(area) for m in [pat.search(text)] if m})
        if hits:
            matched[area_id] = hits
    return matched


def assess_relevance(paper: Paper, records: list[SourceRecord]) -> Relevance:
    """Deterministic ML/CS relevance (spec §34 stage 3). Keyword areas + field check, versioned."""
    domain = config("domain")
    matched = match_areas(f"{paper.title}\n{paper.abstract or ''}")
    fc = domain["field_check"]
    field_hit = None
    for rec in records:
        for t in rec.topics:
            if t.get("is_primary") and t.get("field_id") in fc["openalex_field_ids"]:
                field_hit = f"openalex_primary_field={t.get('field')}"
        for cat in rec.extra.get("categories", []) if rec.source_name in ARXIV_SOURCES else []:
            if any(cat.startswith(p) for p in fc["arxiv_category_prefixes"]):
                field_hit = field_hit or f"arxiv_category={cat}"
        for fos in rec.extra.get("fields_of_study", []) if rec.source_name == "SEMANTIC_SCHOLAR" else []:
            if fos in fc["s2_fields_of_study"]:
                field_hit = field_hit or f"s2_field={fos}"
    return Relevance(
        is_relevant=bool(matched) and field_hit is not None,
        areas=sorted(matched),
        matched_terms=matched,
        field_check=field_hit,
        rule_version=f"{RELEVANCE_VERSION}/{domain['domain_id']}-{domain['version']}",
    )
