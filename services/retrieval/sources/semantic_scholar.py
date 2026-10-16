"""Semantic Scholar adapter: cross-source enrichment and identifiers (DATASET_SPEC §6.2).

Uses POST /graph/v1/paper/batch (<=500 ids per call). Unauthenticated calls share a public
pool and are frequently throttled (HTTP 429); CachedHttp backs off and every successful batch
is cached, so an interrupted run resumes without re-requesting finished batches.
Licence: S2 API License Agreement (no redistribution of API data; attribution required).
"""

from __future__ import annotations

import json
import os
from typing import Any

from services.canonicalization.normalize import clean_text, normalize_doi, parse_arxiv_id
from services.datasets.schemas import Author, FailureCategory, LicenseInfo, Paper, Provenance, SourceRecord
from services.datasets.store import Run, code_version, config, path, read_models, sha256_bytes
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources import load_acquisition, read_cached, save_acquisition

SOURCE = "SEMANTIC_SCHOLAR"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
NORMALIZER_VERSION = "s2-normalizer-0.2.0"  # 0.2.0: reference identifiers
REF_FIELDS = "externalIds,references.externalIds"
FIELDS = ",".join(
    [
        "paperId", "corpusId", "externalIds", "url", "title", "abstract", "venue", "year", "publicationDate",
        "authors", "citationCount", "referenceCount", "isOpenAccess", "openAccessPdf", "fieldsOfStudy",
        "s2FieldsOfStudy", "publicationTypes", "journal",
    ]
)


def make_client() -> tuple[CachedHttp, dict[str, Any]]:
    cfg = config("pipeline")["sources"][SOURCE]
    key = os.environ.get(cfg["api_key_env"])
    http = CachedHttp(
        "semantic_scholar",
        min_interval=cfg["min_interval_s"],
        max_retries=cfg["max_retries"],
        secret_headers={"x-api-key": key} if key else None,
    )
    return http, cfg


def corpus_request_ids() -> list[str]:
    papers = read_models(path("papers"), Paper)
    return sorted({f"ARXIV:{p.arxiv_id}" if p.arxiv_id else f"DOI:{p.doi}" for p in papers if p.arxiv_id or p.doi})


def acquire(run: Run, refresh: bool = False) -> None:
    http, cfg = make_client()
    ids = corpus_request_ids()
    requests = []
    for i in range(0, len(ids), cfg["batch_size"]):
        chunk = ids[i : i + cfg["batch_size"]]
        try:
            resp = http.post(BATCH_URL, {"ids": chunk}, params={"fields": FIELDS}, refresh=refresh)
        except FetchError as e:
            run.fail(
                FailureCategory.retrieval_failure,
                f"S2 batch ids[{i}:{i + len(chunk)}]",
                "paper metadata for every requested id",
                str(e),
                root_cause="Semantic Scholar throttling/availability (no API key)",
            )
            continue
        data = resp.json()
        requests.append(
            {
                "ids": chunk,
                "body_path": resp.meta["body_path"],
                "sha256": resp.meta["sha256"],
                "retrieved_at": resp.meta["retrieved_at"],
                "from_cache": resp.from_cache,
                "found": sum(1 for x in data if x),
            }
        )
    # reference lists (identifiers only) -> intra-corpus citation edges for papers OpenAlex does not cover
    ref_requests = []
    step = cfg.get("reference_batch_size", 50)
    for i in range(0, len(ids), step):
        chunk = ids[i : i + step]
        try:
            resp = http.post(BATCH_URL, {"ids": chunk}, params={"fields": REF_FIELDS}, refresh=refresh)
        except FetchError as e:
            run.fail(FailureCategory.retrieval_failure, f"S2 reference batch ids[{i}:{i + len(chunk)}]", "reference identifiers for every requested id",
                     str(e), root_cause="Semantic Scholar throttling/availability")
            continue
        ref_requests.append({"ids": chunk, "body_path": resp.meta["body_path"], "sha256": resp.meta["sha256"],
                             "retrieved_at": resp.meta["retrieved_at"], "from_cache": resp.from_cache})
    save_acquisition("semantic_scholar", {"run_id": run.run_id, "fields": FIELDS, "requests": requests,
                                          "reference_fields": REF_FIELDS, "reference_requests": ref_requests})
    run.counts.update(
        requested_ids=len(ids),
        batches_ok=len(requests),
        reference_batches_ok=len(ref_requests),
        found=sum(r["found"] for r in requests),
        live_requests=http.requests_made,
        cache_hits=http.cache_hits,
    )


def normalize(paper: dict, requested_id: str, req: dict[str, Any]) -> SourceRecord:
    ext = paper.get("externalIds") or {}
    kind, value = requested_id.split(":", 1)
    parsed = parse_arxiv_id(ext.get("ArXiv")) if ext.get("ArXiv") else None
    s2_arxiv = parsed[0] if parsed else None
    extra: dict[str, Any] = {"requested_id": requested_id}
    arxiv_id = s2_arxiv
    if kind == "ARXIV":
        if s2_arxiv and s2_arxiv != value:
            extra["arxiv_id_mismatch"] = {"requested": value, "s2_external_id": s2_arxiv}
        arxiv_id = value  # S2 resolved this record from the requested arXiv id
    doi = normalize_doi(ext.get("DOI"))
    pid = paper["paperId"]
    identifiers = {
        k: str(v)
        for k, v in {
            "s2": pid, "corpus_id": paper.get("corpusId"), "doi": doi, "arxiv": arxiv_id,
            "dblp": ext.get("DBLP"), "acl": ext.get("ACL"), "mag": ext.get("MAG"), "pubmed": ext.get("PubMed"),
        }.items()
        if v
    }
    journal = paper.get("journal") or {}
    extra.update(
        {
            k: v
            for k, v in {
                "fields_of_study": paper.get("fieldsOfStudy"),
                "publication_types": paper.get("publicationTypes"),
                "open_access_pdf": paper.get("openAccessPdf"),
                "reference_count": paper.get("referenceCount"),
                "external_ids": ext,
            }.items()
            if v
        }
    )
    return SourceRecord(
        source_record_id=f"s2:{pid}",
        source_name=SOURCE,
        source_identifier=pid,
        source_url=paper.get("url"),
        source_version=f"graph/v1@{req['retrieved_at'][:10]}",
        title=clean_text(paper.get("title")) or None,
        abstract=clean_text(paper.get("abstract")) or None,
        authors=[Author(name=a.get("name") or "UNKNOWN", source_author_id=a.get("authorId")) for a in paper.get("authors") or []],
        publication_date=paper.get("publicationDate"),
        publication_year=paper.get("year"),
        venue=paper.get("venue") or journal.get("name") or None,
        doi=doi,
        arxiv_id=arxiv_id,
        identifiers=identifiers,
        urls={"s2": paper["url"]} if paper.get("url") else {},
        cited_by_count=paper.get("citationCount"),
        topics=[{"source": SOURCE, "name": f.get("category"), "origin": f.get("source")} for f in paper.get("s2FieldsOfStudy") or []],
        open_access={"is_oa": paper.get("isOpenAccess")},
        license=LicenseInfo(metadata_license="S2-API-LICENSE", license_source="s2_api_license_agreement"),
        extra=extra,
        provenance=Provenance(
            source_name=SOURCE,
            source_url=paper.get("url"),
            source_identifier=pid,
            source_version=f"graph/v1@{req['retrieved_at'][:10]}",
            retrieved_at=req["retrieved_at"],
            content_hash=sha256_bytes(json.dumps(paper, sort_keys=True).encode()),
            processing_version=code_version(),
            extractor_version=NORMALIZER_VERSION,
            raw_path=req["body_path"],
        ),
    )


def reference_ids(paper: dict) -> list[str]:
    """Referenced works as 'arxiv:<id>' / 'doi:<doi>' (both when available); works without either are dropped."""
    out: list[str] = []
    for ref in paper.get("references") or []:
        ext = (ref or {}).get("externalIds") or {}
        parsed = parse_arxiv_id(ext.get("ArXiv")) if ext.get("ArXiv") else None
        if parsed:
            out.append(f"arxiv:{parsed[0]}")
        if doi := normalize_doi(ext.get("DOI")):
            out.append(f"doi:{doi}")
    return sorted(set(out))


def normalize_all(context: dict[str, Any] | None = None) -> list[SourceRecord]:
    acq = load_acquisition("semantic_scholar")
    refs: dict[str, list[str]] = {}
    for req in (acq or {}).get("reference_requests", []):
        for requested_id, paper in zip(req["ids"], json.loads(read_cached(req))):
            if paper:
                refs[requested_id] = reference_ids(paper)
    out: dict[str, SourceRecord] = {}
    for req in acq["requests"] if acq else []:
        for requested_id, paper in zip(req["ids"], json.loads(read_cached(req))):
            if paper and paper.get("paperId"):
                rec = normalize(paper, requested_id, req)
                rec.references = refs.get(requested_id, [])
                out.setdefault(rec.source_record_id, rec)
    return [out[k] for k in sorted(out)]
