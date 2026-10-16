"""arXiv OAI-PMH adapter (metadataPrefix=arXivRaw): Layer 0 discovery with a licence per paper.

Why: the cheapest licence-clean route to the 20K-50K candidate target (source assessment, 2026-09-30).
Metadata is CC0; the legacy-API terms (<= 1 request / 3 s, one connection) cover OAI-PMH.

Harvest = configured sets x monthly datestamp windows, each followed through resumption tokens. The tokens
are deterministic query strings (verb/from/until/set/skip), so every page is cacheable and replayable.
Pages are stored gzip-compressed in data/raw/arxiv_oai/ (sha256 over the raw XML).

normalize_all() returns records only for papers in the main corpus (discovery ids from OpenAlex + ids promoted
by the selection manifest); the full harvest feeds the compact Layer 0 file built by services.retrieval.discovery.
"""

from __future__ import annotations

import calendar
import json
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Iterator
from urllib.parse import unquote

from lxml import etree

from services.canonicalization.normalize import arxiv_doi, clean_text, normalize_doi
from services.datasets.manifests import license_flags, match_license
from services.datasets.schemas import Author, FailureCategory, LicenseInfo, Provenance, SourceRecord
from services.datasets.store import DATA, Run, code_version, config, sha256_bytes
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources import load_acquisition, read_cached, save_acquisition

SOURCE = "ARXIV_OAI_PMH"
CACHE_DIR = "arxiv_oai"
NORMALIZER_VERSION = "arxiv-oai-normalizer-0.1.0"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
RAW = "{http://arxiv.org/OAI/arXivRaw/}"
SELECTION = DATA / "manifests" / "selection" / "main_corpus.json"


def make_client() -> tuple[CachedHttp, dict[str, Any]]:
    cfg = config("pipeline")["sources"][SOURCE]
    return CachedHttp(CACHE_DIR, min_interval=cfg["min_interval_s"], max_retries=cfg["max_retries"], timeout=240, compress=True), cfg


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Inclusive monthly [from, until] date windows covering start..end (ISO dates)."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out, y, m = [], d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        first = max(date(y, m, 1), d0)
        last = min(date(y, m, calendar.monthrange(y, m)[1]), d1)
        out.append((first.isoformat(), last.isoformat()))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def parse_page(xml: bytes) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """-> (entries, resumption token or None, OAI error code or None)."""
    root = etree.fromstring(xml)
    err = root.find(f"{OAI}error")
    if err is not None:
        return [], None, err.get("code")
    entries = []
    for rec in root.iter(f"{OAI}record"):
        header = rec.find(f"{OAI}header")
        datestamp = header.findtext(f"{OAI}datestamp") if header is not None else None
        sets = [s.text for s in header.findall(f"{OAI}setSpec")] if header is not None else []
        if header is not None and header.get("status") == "deleted":
            ident = header.findtext(f"{OAI}identifier") or ""
            entries.append({"deleted": True, "arxiv_id": ident.rsplit(":", 1)[-1], "datestamp": datestamp})
            continue
        raw = rec.find(f"{OAI}metadata/{RAW}arXivRaw")
        if raw is None:
            continue
        versions = []
        for v in raw.findall(f"{RAW}version"):
            try:
                when = parsedate_to_datetime(v.findtext(f"{RAW}date") or "").date().isoformat()
            except (TypeError, ValueError):
                when = None
            versions.append({"version": v.get("version"), "date": when})
        entries.append(
            {
                "arxiv_id": clean_text(raw.findtext(f"{RAW}id")),
                "datestamp": datestamp,
                "sets": sets,
                "versions": versions,
                "title": clean_text(raw.findtext(f"{RAW}title")),
                "authors": clean_text(raw.findtext(f"{RAW}authors")),
                "categories": (raw.findtext(f"{RAW}categories") or "").split(),
                "comments": clean_text(raw.findtext(f"{RAW}comments")) or None,
                "journal_ref": clean_text(raw.findtext(f"{RAW}journal-ref")) or None,
                "doi": normalize_doi(raw.findtext(f"{RAW}doi")),
                "license": clean_text(raw.findtext(f"{RAW}license")) or None,
                "abstract": clean_text(raw.findtext(f"{RAW}abstract")) or None,
            }
        )
    tok = root.find(f".//{OAI}resumptionToken")
    token = tok.text.strip() if tok is not None and tok.text and tok.text.strip() else None
    return entries, token, None


def acquire(run: Run, refresh: bool = False, sets: list[str] | None = None) -> None:
    http, cfg = make_client()
    sets = sets or cfg["sets"]
    windows = month_windows(cfg["from"], cfg["until"])
    pages: list[dict[str, Any]] = []
    for s in sets:
        for frm, until in windows:
            params: dict[str, str] = {"verb": "ListRecords", "metadataPrefix": cfg["metadata_prefix"], "set": s, "from": frm, "until": until}
            for page_no in range(10_000):
                try:
                    resp = http.get(cfg["base_url"], params=params, refresh=refresh)
                    entries, token, error = parse_page(resp.body)
                    if error == "badResumptionToken" and "resumptionToken" in params:  # some servers want it decoded
                        params = {**params, "resumptionToken": unquote(params["resumptionToken"])}
                        resp = http.get(cfg["base_url"], params=params, refresh=refresh)
                        entries, token, error = parse_page(resp.body)
                except (FetchError, etree.XMLSyntaxError) as e:
                    run.fail(FailureCategory.retrieval_failure, f"OAI {s} {frm}..{until} page {page_no}", "ListRecords page", str(e)[:300])
                    break
                if error and error != "noRecordsMatch":
                    run.fail(FailureCategory.retrieval_failure, f"OAI {s} {frm}..{until} page {page_no}", "ListRecords page", f"OAI error {error}")
                pages.append({"set": s, "from": frm, "until": until, "page": page_no, "body_path": resp.meta["body_path"], "sha256": resp.meta["sha256"],
                              "retrieved_at": resp.meta["retrieved_at"], "from_cache": resp.from_cache, "records": len(entries), "oai_error": error})
                if not token or error:
                    break
                params = {"verb": "ListRecords", "resumptionToken": token}  # verbatim, as the OAI-PMH spec requires
            run.counts["pages"] = len(pages)
            print(f"  {s} {frm[:7]}: {sum(p['records'] for p in pages if p['set'] == s and p['from'] == frm)} records", flush=True)
        save_acquisition(CACHE_DIR, {"run_id": run.run_id, "sets": sets, "windows": [list(w) for w in windows], "pages": pages})
    run.counts.update(pages=len(pages), records_seen=sum(p["records"] for p in pages), live_requests=http.requests_made, cache_hits=http.cache_hits)


def iter_entries() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Every harvested (entry, page), latest datestamp per arXiv id winning; deleted records dropped."""
    acq = load_acquisition(CACHE_DIR)
    best: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for page in acq["pages"] if acq else []:
        entries, _, _ = parse_page(read_cached(page))
        for e in entries:
            aid = e.get("arxiv_id")
            if not aid:
                continue
            prev = best.get(aid)
            sets = sorted(set(e.get("sets") or []) | set((prev[0].get("sets") if prev else None) or []))
            if prev is None or (e.get("datestamp") or "") > (prev[0].get("datestamp") or ""):
                best[aid] = ({**e, "sets": sets}, page)
            else:
                prev[0]["sets"] = sets
    for aid in sorted(best):
        e, page = best[aid]
        if not e.get("deleted"):
            yield e, page


_AUTHOR_SPLIT = re.compile(r",\s*|\s+and\s+")


def split_authors(raw: str | None) -> list[str]:
    """arXivRaw authors are one string ('A, B and C'); affiliations in parentheses are dropped."""
    if not raw:
        return []
    raw = re.sub(r"\([^)]*\)", "", raw)
    return [a.strip() for a in _AUTHOR_SPLIT.split(raw) if a.strip()]


def to_source_record(e: dict[str, Any], page: dict[str, Any]) -> SourceRecord:
    aid = e["arxiv_id"]
    versions = e.get("versions") or []
    latest = versions[-1]["version"] if versions else None
    v1_date = versions[0]["date"] if versions else None
    lic_id = match_license(e.get("license"))
    flags = license_flags(lic_id)
    cats = e.get("categories") or []
    return SourceRecord(
        source_record_id=f"arxivoai:{aid}",
        source_name=SOURCE,
        source_identifier=aid,
        source_url=f"https://arxiv.org/abs/{aid}",
        source_version=latest,
        title=e.get("title") or None,
        abstract=e.get("abstract"),
        authors=[Author(name=n) for n in split_authors(e.get("authors"))],
        publication_date=v1_date,
        publication_year=int(v1_date[:4]) if v1_date else None,
        doi=arxiv_doi(aid),
        arxiv_id=aid,
        arxiv_version=latest,
        identifiers={"arxiv": aid, **({"journal_doi": e["doi"]} if e.get("doi") else {})},
        urls={"abs": f"https://arxiv.org/abs/{aid}", "pdf": f"https://arxiv.org/pdf/{aid}", "html": f"https://arxiv.org/html/{aid}{latest or ''}"},
        topics=[{"source": SOURCE, "name": c, "is_primary": i == 0} for i, c in enumerate(cats)],
        license=LicenseInfo(
            content_license=lic_id,
            metadata_license="CC0-1.0",
            license_source=f"arxiv_oai_pmh:{e.get('license')}" if e.get("license") else None,
            redistribution_allowed=flags["redistribution_allowed"],
            training_allowed=flags["training_allowed"],
        ),
        extra={k: v for k, v in {
            "categories": cats, "primary_category": cats[0] if cats else None, "comments": e.get("comments"),
            "journal_ref": e.get("journal_ref"), "versions": versions, "oai_datestamp": e.get("datestamp"), "oai_sets": e.get("sets"),
        }.items() if v},
        provenance=Provenance(
            source_name=SOURCE,
            source_url=f"https://arxiv.org/abs/{aid}",
            source_identifier=aid,
            source_version=latest,
            retrieved_at=datetime.fromisoformat(page["retrieved_at"]),
            content_hash=sha256_bytes(json.dumps(e, sort_keys=True).encode()),
            processing_version=code_version(),
            extractor_version=NORMALIZER_VERSION,
            raw_path=page["body_path"],
        ),
    )


def promoted_ids() -> set[str]:
    if not SELECTION.exists():
        return set()
    return {s["arxiv_id"] for s in json.loads(SELECTION.read_text(encoding="utf-8"))["promoted"]}


def normalize_all(context: dict[str, Any] | None = None) -> list[SourceRecord]:
    """Records for main-corpus papers only: OpenAlex discovery ids (context) + promoted Layer 0 candidates."""
    wanted = set((context or {}).get("arxiv_ids", set())) | promoted_ids()
    return [to_source_record(e, p) for e, p in iter_entries() if e["arxiv_id"] in wanted]
