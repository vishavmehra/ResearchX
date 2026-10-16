"""Study-link candidates (DATASET_SPEC §4.6, §14, §29). Candidate generation is not Gold labeling.

Evidence-based rules (link-rules-0.1.0):
- External identifier: a Semantic Scholar / arXiv record of paper A lists a DOI owned by paper B
  -> SAME_STUDY_DIFFERENT_RECORD (identifier evidence; still unreviewed).
- Near-identical title + shared authors -> UNCERTAIN (hypothesis: same study, different record).
- Similar title + shared authors -> UNCERTAIN (hypothesis: follow-up / extension).
- Similar title + NO shared authors -> RELATED_BUT_DISTINCT (hard negative: similar title, different study).
Title similarity alone never yields a SAME_* label (spec §2, §29). SAME_UNDERLYING_STUDY needs review.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from services.canonicalization.normalize import jaccard, normalize_title, title_tokens
from services.datasets.schemas import Paper, Provenance, SourceRecord, Split, StudyLink, StudyLinkLabel
from services.datasets.store import code_version, stable_id

LINK_VERSION = "link-rules-0.1.0"
NEAR_IDENTICAL = 0.85
SIMILAR = 0.5
MAX_TOKEN_DF = 40  # blocking: only rare shared title tokens generate candidate pairs


def author_keys(p: Paper) -> frozenset[str]:
    return frozenset(normalize_title(a.name).split()[-1] for a in p.authors if normalize_title(a.name))


def author_overlap(a: Paper, b: Paper) -> float:
    ka, kb = author_keys(a), author_keys(b)
    return len(ka & kb) / min(len(ka), len(kb)) if ka and kb else 0.0


def _link(a: Paper, b: Paper, label: StudyLinkLabel, methods: list[str], evidence: list[dict], rationale: str,
          feats: dict[str, float], hard_negative: str | None = None) -> StudyLink:
    if a.paper_id > b.paper_id:
        a, b = b, a
    return StudyLink(
        link_id=stable_id("link", a.paper_id, b.paper_id),
        record_a=a.paper_id,
        record_b=b.paper_id,
        relationship=label,
        candidate_methods=methods,
        evidence=evidence,
        rationale=rationale,
        is_hard_negative=hard_negative is not None,
        construction_method=hard_negative,
        features=feats,
        labeler_version=LINK_VERSION,
        split=a.split if a.split == b.split else Split.CROSS_SPLIT,
        provenance=Provenance(source_name="RESEARCHX", processing_version=code_version(), extractor_version=LINK_VERSION,
                              parent_record_ids=[a.paper_id, b.paper_id]),
    )


def build_links(papers: list[Paper], records: dict[str, SourceRecord], cites: set[tuple[str, str]]) -> list[StudyLink]:
    by_id = {p.paper_id: p for p in papers}
    paper_of_record = {s: p.paper_id for p in papers for s in p.source_record_ids}
    doi_owner = {p.doi: p.paper_id for p in papers if p.doi and not p.doi.startswith("10.48550/")}
    out: dict[str, StudyLink] = {}

    # 1. cross-identifier claims (S2 DOI, arXiv journal DOI) pointing at another paper
    for srid, rec in sorted(records.items()):
        pid = paper_of_record.get(srid)
        for key in ("doi", "journal_doi"):
            doi = rec.identifiers.get(key)
            other = doi_owner.get(doi or "")
            if pid and other and other != pid:
                a, b = by_id[pid], by_id[other]
                link = _link(a, b, StudyLinkLabel.SAME_STUDY_DIFFERENT_RECORD, ["external_identifier"],
                             [{"kind": "external_identifier", "detail": f"{rec.source_name} record {srid} of {pid} lists DOI {doi}, owned by {other}", "source_record_id": srid}],
                             "Another source's record for paper A carries paper B's DOI (identifier evidence, unreviewed).",
                             {"author_overlap": author_overlap(a, b)})
                out[link.link_id] = link

    # 2. title/author similarity with rare-token blocking
    toks = {p.paper_id: title_tokens(p.title) for p in papers}
    df: dict[str, set[str]] = defaultdict(set)
    for pid, ts in toks.items():
        for t in ts:
            df[t].add(pid)
    cand: set[tuple[str, str]] = set()
    for t, pids in df.items():
        if 1 < len(pids) <= MAX_TOKEN_DF:
            cand.update(combinations(sorted(pids), 2))
    for x, y in sorted(cand):
        j = jaccard(toks[x], toks[y])
        if j < SIMILAR:
            continue
        a, b = by_id[x], by_id[y]
        ov = author_overlap(a, b)
        cited = (x, y) in cites or (y, x) in cites
        gap = abs((a.publication_year or 0) - (b.publication_year or 0)) if a.publication_year and b.publication_year else None
        feats = {"title_jaccard": round(j, 4), "author_overlap": round(ov, 4), "citation_link": float(cited)}
        if gap is not None:
            feats["year_gap"] = float(gap)
        ev = [{"kind": "title_similarity", "detail": f"title token Jaccard {j:.2f}: {a.title!r} vs {b.title!r}", "source_record_id": None}]
        if ov:
            ev.append({"kind": "shared_authors", "detail": f"author surname overlap {ov:.2f}: {sorted(author_keys(a) & author_keys(b))[:6]}", "source_record_id": None})
        if cited:
            ev.append({"kind": "citation", "detail": "one paper cites the other (OpenAlex referenced_works)", "source_record_id": None})
        if ov == 0:
            link = _link(a, b, StudyLinkLabel.RELATED_BUT_DISTINCT, ["title_similarity"], ev,
                         "Similar titles but no shared authors: treated as distinct studies (hard negative).", feats,
                         hard_negative="SIMILAR_TITLE_DIFFERENT_STUDY")
        elif j >= NEAR_IDENTICAL and ov >= 0.5:
            link = _link(a, b, StudyLinkLabel.UNCERTAIN, ["title_similarity", "shared_authors"], ev,
                         "Hypothesis: same study, different record (near-identical title, shared authors). Needs review.", feats)
        elif ov >= 0.3:
            link = _link(a, b, StudyLinkLabel.UNCERTAIN, ["title_similarity", "shared_authors"] + (["citation"] if cited else []), ev,
                         "Hypothesis: follow-up or extension (similar title, shared authors). Needs review.", feats)
        else:
            continue
        out.setdefault(link.link_id, link)
    return [out[k] for k in sorted(out)]
