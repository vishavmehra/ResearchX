"""OpenAlex adapter: candidate discovery + scholarly metadata (DATASET_SPEC §6.1).

Metadata is CC0. Billing (verified 2026-09-30): list+filter $0.0001/call, search $0.001/call;
free allowance $0.10/day without a key, $1/day with a free key (OPENALEX_API_KEY).
Every page is cached, so re-runs cost nothing.

License rule: a work's content license is taken ONLY from its arXiv location. Other locations
(e.g. a Zenodo upload wrongly merged into W3168867926) must never lend their license to the paper.
"""

from __future__ import annotations

import json
import os
from typing import Any

from services.canonicalization.normalize import clean_text, normalize_doi, parse_arxiv_id, reconstruct_abstract
from services.datasets.schemas import Author, FailureCategory, LicenseInfo, Provenance, SourceRecord
from services.datasets.store import Run, code_version, config, sha256_bytes
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources import load_acquisition, read_cached, save_acquisition

SOURCE = "OPENALEX"
WORKS_URL = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "https://openalex.org/S4306400194"
NORMALIZER_VERSION = "openalex-normalizer-0.1.0"
SELECT = ",".join(
    [
        "id", "doi", "title", "display_name", "publication_date", "publication_year", "type", "ids",
        "authorships", "primary_location", "locations", "best_oa_location", "open_access",
        "primary_topic", "topics", "referenced_works", "cited_by_count", "updated_date",
        "language", "indexed_in", "abstract_inverted_index",
    ]
)


def make_client() -> tuple[CachedHttp, dict[str, str]]:
    cfg = config("pipeline")["sources"][SOURCE]
    base: dict[str, str] = {"select": SELECT}
    if key := os.environ.get(cfg["api_key_env"]):
        base["api_key"] = key  # redacted from cache keys/meta by CachedHttp
    if mailto := os.environ.get(config("pipeline")["contact_email_env"]):
        base["mailto"] = mailto
    return CachedHttp("openalex", min_interval=cfg["min_interval_s"], max_retries=cfg["max_retries"]), base


def search_works(
    http: CachedHttp, base: dict[str, str], filter_str: str, sort: str, limit: int, run: Run, refresh: bool = False
) -> tuple[list[dict], list[dict]]:
    """Cursor-paginate /works until `limit` works; stop early if the daily budget runs low."""
    min_usd = config("pipeline")["sources"][SOURCE]["min_remaining_usd"]
    per_page = min(200, limit)
    works: list[dict] = []
    pages: list[dict] = []
    cursor: str | None = "*"
    while cursor and len(works) < limit:
        params = {**base, "filter": filter_str, "sort": sort, "per-page": str(per_page), "cursor": cursor}
        resp = http.get(WORKS_URL, params=params, refresh=refresh)
        data = resp.json()
        pages.append(
            {
                "body_path": resp.meta["body_path"],
                "sha256": resp.meta["sha256"],
                "retrieved_at": resp.meta["retrieved_at"],
                "from_cache": resp.from_cache,
                "cost_usd": data.get("meta", {}).get("cost_usd"),
                "count": data.get("meta", {}).get("count"),
            }
        )
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        remaining = resp.meta.get("rate_limit", {}).get("x-ratelimit-remaining-usd")
        if not resp.from_cache and remaining is not None and float(remaining) < min_usd:
            run.note(f"OpenAlex budget guard: remaining ${remaining} < ${min_usd}; stopping pagination")
            break
    return works[:limit], pages


def acquire(run: Run, profile: str, areas: list[str] | None = None, refresh: bool = False, mode: str = "auto") -> None:
    """Discovery modes:
    - search: per-area title_and_abstract.search queries (targeted; billed as search).
    - topic_fallback: filter-only query over curated ML/CS OpenAlex topics; the deterministic
      relevance filter (config/domain.yaml terms) then selects priority-area papers. Used when
      OpenAlex pauses anonymous search (observed 2026-09-30: HTTP 503 "Anonymous search is paused").
    auto = search, switching to topic_fallback after the first search failure (circuit breaker).
    """
    domain = config("domain")
    prof = config("pipeline")["corpus"][profile]
    areas = areas or prof.get("areas")
    http, base = make_client()
    search_http = CachedHttp("openalex", min_interval=http.min_interval, max_retries=2)
    queries: list[dict] = []

    def record(area: str, qmode: str, filt: str, sort: str, limit: int, works: list[dict], pages: list[dict]) -> None:
        queries.append(
            {"area": area, "mode": qmode, "filter": filt, "sort": sort, "limit": limit, "pages": pages, "work_ids": [w["id"] for w in works]}
        )
        run.note(f"{area} [{qmode} {sort}] -> {len(works)} works ({sum(not p['from_cache'] for p in pages)} live requests)")

    use_search = mode in ("auto", "search")
    if use_search:
        for area_id, area in domain["areas"].items():
            if areas and area_id not in areas:
                continue
            for sort, limit in (("cited_by_count:desc", prof["per_area_most_cited"]), ("publication_date:desc", prof["per_area_recent"])):
                if not limit or not use_search:
                    continue
                filt = f"{domain['openalex']['base_filter']},title_and_abstract.search:{area['openalex_query']}"
                try:
                    works, pages = search_works(search_http, base, filt, sort, limit, run, refresh)
                except FetchError as e:
                    run.fail(
                        FailureCategory.retrieval_failure,
                        f"OpenAlex search area={area_id} sort={sort}",
                        "search results",
                        str(e),
                        severity="HIGH",
                        root_cause="OpenAlex search endpoint unavailable (anonymous search paused under load)",
                    )
                    if mode == "search":
                        raise
                    run.note("search unavailable -> switching to topic_fallback discovery")
                    use_search = False
                    continue
                record(area_id, "search", filt, sort, limit, works, pages)
    if not use_search or mode == "topic_fallback":
        topics = "|".join(t["id"] for t in domain["openalex"]["fallback_topics"])
        filt = f"{domain['openalex']['base_filter']},primary_topic.id:{topics}"
        for sort, limit in (("cited_by_count:desc", prof["fallback_most_cited"]), ("publication_date:desc", prof["fallback_recent"])):
            if limit:
                works, pages = search_works(http, base, filt, sort, limit, run, refresh)
                record("ALL_FALLBACK_TOPICS", "topic_fallback", filt, sort, limit, works, pages)
    save_acquisition("openalex", {"run_id": run.run_id, "profile": profile, "queries": queries})
    run.counts.update(
        queries=len(queries),
        works_listed=sum(len(q["work_ids"]) for q in queries),
        live_requests=http.requests_made + search_http.requests_made,
        cache_hits=http.cache_hits + search_http.cache_hits,
    )


def _arxiv_from_work(work: dict) -> tuple[str | None, str | None, list[str]]:
    found: dict[str, str | None] = {}
    doi = normalize_doi(work.get("doi"))
    if doi and doi.startswith("10.48550/arxiv."):
        if p := parse_arxiv_id(doi):
            found[p[0]] = p[1]
    for loc in work.get("locations") or []:
        if (loc.get("source") or {}).get("id") != ARXIV_SOURCE_ID:
            continue
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if url and (p := parse_arxiv_id(url)):
                found.setdefault(p[0], p[1])
                found[p[0]] = found[p[0]] or p[1]
    ids = sorted(found)
    if len(ids) == 1:
        return ids[0], found[ids[0]], ids
    return None, None, ids  # zero or ambiguous: never pick one silently


def normalize(work: dict, page: dict[str, Any], found_by: list[dict]) -> SourceRecord:
    wid = work["id"].rsplit("/", 1)[-1]
    doi = normalize_doi(work.get("doi"))
    arxiv_id, arxiv_version, arxiv_candidates = _arxiv_from_work(work)
    primary = work.get("primary_location") or {}
    arxiv_loc = next((l for l in work.get("locations") or [] if (l.get("source") or {}).get("id") == ARXIV_SOURCE_ID), None)
    others = [
        {"source": (l.get("source") or {}).get("display_name"), "url": l.get("landing_page_url"), "license": l.get("license")}
        for l in work.get("locations") or []
        if (l.get("source") or {}).get("id") != ARXIV_SOURCE_ID
    ]
    ids = work.get("ids") or {}
    topics = []
    primary_topic_id = (work.get("primary_topic") or {}).get("id")
    for t in work.get("topics") or []:
        topics.append(
            {
                "source": SOURCE, "id": t.get("id"), "name": t.get("display_name"), "score": t.get("score"),
                "field_id": (t.get("field") or {}).get("id"), "field": (t.get("field") or {}).get("display_name"),
                "subfield": (t.get("subfield") or {}).get("display_name"), "is_primary": t.get("id") == primary_topic_id,
            }
        )
    return SourceRecord(
        source_record_id=f"openalex:{wid}",
        source_name=SOURCE,
        source_identifier=wid,
        source_url=work["id"],
        source_version=work.get("updated_date"),
        title=clean_text(work.get("display_name") or work.get("title")) or None,
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=[
            Author(
                name=(a.get("author") or {}).get("display_name") or a.get("raw_author_name") or "UNKNOWN",
                raw_name=a.get("raw_author_name"),
                orcid=(a.get("author") or {}).get("orcid"),
                source_author_id=(a.get("author") or {}).get("id"),
            )
            for a in work.get("authorships") or []
        ],
        publication_date=work.get("publication_date"),
        publication_year=work.get("publication_year"),
        venue=(primary.get("source") or {}).get("display_name"),
        doi=doi,
        arxiv_id=arxiv_id,
        arxiv_version=arxiv_version,
        identifiers={
            k: v
            for k, v in {"openalex": wid, "doi": doi, "arxiv": arxiv_id, "mag": ids.get("mag"), "pmid": ids.get("pmid")}.items()
            if v
        },
        urls={
            k: v
            for k, v in {
                "landing": primary.get("landing_page_url"),
                "pdf": primary.get("pdf_url"),
                "arxiv_abs": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            }.items()
            if v
        },
        references=[r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []],
        cited_by_count=work.get("cited_by_count"),
        topics=topics,
        open_access=work.get("open_access") or {},
        license=LicenseInfo(metadata_license="CC0-1.0", license_source="openalex_data_license"),
        extra={
            k: v
            for k, v in {
                "type": work.get("type"),
                "indexed_in": work.get("indexed_in"),
                "language": work.get("language"),
                "arxiv_id_candidates": arxiv_candidates if len(arxiv_candidates) > 1 else None,
                "arxiv_location_license": (arxiv_loc or {}).get("license"),
                "non_arxiv_locations": others or None,
                "found_by": found_by,
            }.items()
            if v is not None
        },
        provenance=Provenance(
            source_name=SOURCE,
            source_url=work["id"],
            source_identifier=wid,
            source_version=work.get("updated_date"),
            retrieved_at=page["retrieved_at"],
            content_hash=sha256_bytes(json.dumps(work, sort_keys=True).encode()),
            processing_version=code_version(),
            extractor_version=NORMALIZER_VERSION,
            raw_path=page["body_path"],
        ),
    )


def normalize_all(context: dict[str, Any] | None = None) -> list[SourceRecord]:
    acq = load_acquisition("openalex")
    if not acq:
        return []
    raw: dict[str, tuple[dict, dict]] = {}
    found_by: dict[str, list[dict]] = {}
    for q in acq["queries"]:
        wanted = set(q["work_ids"])
        for page in q["pages"]:
            for w in json.loads(read_cached(page)).get("results", []):
                if w["id"] in wanted:
                    raw.setdefault(w["id"], (w, page))
                    fb = {"area": q["area"], "mode": q.get("mode", "search"), "sort": q["sort"]}
                    if fb not in found_by.setdefault(w["id"], []):
                        found_by[w["id"]].append(fb)
    return [normalize(w, page, found_by[wid]) for wid, (w, page) in sorted(raw.items())]
