"""arXiv adapter: preprint metadata, versions and licence detection (DATASET_SPEC §6.3).

Metadata (Atom API, CC0) via export.arxiv.org, one request every 3 s (arXiv API ToU).
The record distinguishes arXiv id, version, source date (v1 'published', latest 'updated')
and retrieval date. Content licences are per paper and are read from each paper's HTML
header during full-text acquisition (services.retrieval.fulltext).
"""

from __future__ import annotations

import json
import re
from typing import Any

from lxml import etree

from services.canonicalization.normalize import arxiv_doi, clean_text, normalize_doi, parse_arxiv_id
from services.datasets.schemas import Author, FailureCategory, LicenseInfo, Paper, Provenance, SourceRecord
from services.datasets.store import Run, code_version, config, path, read_models, sha256_bytes
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources import load_acquisition, read_cached, save_acquisition

SOURCE = "ARXIV"
API_URL = "https://export.arxiv.org/api/query"
NORMALIZER_VERSION = "arxiv-normalizer-0.1.0"
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"


def make_client() -> tuple[CachedHttp, dict[str, Any]]:
    cfg = config("pipeline")["sources"][SOURCE]
    return CachedHttp("arxiv", min_interval=cfg["min_interval_s"], max_retries=cfg["max_retries"]), cfg


def acquire(run: Run, refresh: bool = False) -> None:
    http, cfg = make_client()
    ids = sorted({p.arxiv_id for p in read_models(path("papers"), Paper) if p.arxiv_id})
    requests = []
    for i in range(0, len(ids), cfg["batch_size"]):
        chunk = ids[i : i + cfg["batch_size"]]
        params = {"id_list": ",".join(chunk), "max_results": str(len(chunk))}
        try:
            resp = http.get(API_URL, params=params, refresh=refresh)
        except FetchError as e:
            run.fail(FailureCategory.retrieval_failure, f"arXiv id_list[{i}:{i + len(chunk)}]", "Atom entries", str(e))
            continue
        entries = parse_feed(resp.body)
        got = {e["arxiv_id"] for e in entries if "arxiv_id" in e}
        for missing in sorted(set(chunk) - got):
            run.fail(FailureCategory.retrieval_failure, f"arXiv id {missing}", "one Atom entry", "id not returned by arXiv API", severity="LOW")
        requests.append(
            {"ids": chunk, "body_path": resp.meta["body_path"], "sha256": resp.meta["sha256"], "retrieved_at": resp.meta["retrieved_at"], "from_cache": resp.from_cache, "entries": len(got)}
        )
    save_acquisition("arxiv", {"run_id": run.run_id, "requests": requests})
    run.counts.update(requested_ids=len(ids), entries=sum(r["entries"] for r in requests), live_requests=http.requests_made, cache_hits=http.cache_hits)


def parse_feed(xml: bytes) -> list[dict[str, Any]]:
    root = etree.fromstring(xml)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        eid = e.findtext(f"{ATOM}id") or ""
        parsed = parse_arxiv_id(eid)
        if "/api/errors" in eid or not parsed:
            out.append({"error": clean_text(e.findtext(f"{ATOM}summary"))})
            continue
        prim = e.find(f"{ARX}primary_category")
        out.append(
            {
                "arxiv_id": parsed[0],
                "version": parsed[1],
                "abs_url": eid,
                "title": clean_text(e.findtext(f"{ATOM}title")),
                "summary": clean_text(e.findtext(f"{ATOM}summary")),
                "published": e.findtext(f"{ATOM}published"),
                "updated": e.findtext(f"{ATOM}updated"),
                "authors": [clean_text(a.findtext(f"{ATOM}name")) for a in e.findall(f"{ATOM}author")],
                "primary_category": prim.get("term") if prim is not None else None,
                "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
                "comment": clean_text(e.findtext(f"{ARX}comment")) or None,
                "journal_ref": clean_text(e.findtext(f"{ARX}journal_ref")) or None,
                "journal_doi": normalize_doi(e.findtext(f"{ARX}doi")),
            }
        )
    return out


def normalize(entry: dict[str, Any], req: dict[str, Any]) -> SourceRecord:
    aid, ver = entry["arxiv_id"], entry["version"] or ""
    published = entry.get("published") or ""
    return SourceRecord(
        source_record_id=f"arxiv:{aid}{ver}",
        source_name=SOURCE,
        source_identifier=f"{aid}{ver}",
        source_url=f"https://arxiv.org/abs/{aid}{ver}",
        source_version=ver or None,
        title=entry["title"] or None,
        abstract=entry["summary"] or None,
        authors=[Author(name=n) for n in entry["authors"] if n],
        publication_date=published[:10] or None,  # v1 submission date: earliest public availability
        publication_year=int(published[:4]) if published[:4].isdigit() else None,
        doi=arxiv_doi(aid),
        arxiv_id=aid,
        arxiv_version=ver or None,
        identifiers={"arxiv": aid, **({"journal_doi": entry["journal_doi"]} if entry.get("journal_doi") else {})},
        urls={"abs": f"https://arxiv.org/abs/{aid}{ver}", "pdf": f"https://arxiv.org/pdf/{aid}{ver}", "html": f"https://arxiv.org/html/{aid}{ver}"},
        topics=[{"source": SOURCE, "name": c, "is_primary": c == entry.get("primary_category")} for c in entry["categories"]],
        license=LicenseInfo(metadata_license="CC0-1.0", license_source="arxiv_metadata_cc0"),
        extra={
            k: v
            for k, v in {
                "categories": entry["categories"],
                "primary_category": entry.get("primary_category"),
                "comment": entry.get("comment"),
                "journal_ref": entry.get("journal_ref"),
                "updated": entry.get("updated"),
                "latest_version": ver or None,
            }.items()
            if v
        },
        provenance=Provenance(
            source_name=SOURCE,
            source_url=f"https://arxiv.org/abs/{aid}{ver}",
            source_identifier=f"{aid}{ver}",
            source_version=ver or None,
            retrieved_at=req["retrieved_at"],
            content_hash=sha256_bytes(json.dumps(entry, sort_keys=True).encode()),
            processing_version=code_version(),
            extractor_version=NORMALIZER_VERSION,
            raw_path=req["body_path"],
        ),
    )


def normalize_all(context: dict[str, Any] | None = None) -> list[SourceRecord]:
    acq = load_acquisition("arxiv")
    out: dict[str, SourceRecord] = {}
    for req in acq["requests"] if acq else []:
        for entry in parse_feed(read_cached(req)):
            if "arxiv_id" in entry:
                rec = normalize(entry, req)
                out.setdefault(rec.source_record_id, rec)
    return [out[k] for k in sorted(out)]


# arXiv HTML header: <a id="license-tr" href="...">License: arXiv.org perpetual non-exclusive license</a>
_LICENSE_RE = re.compile(r'(?:id|class)="license-tr"[^>]*href="([^"]*)"[^>]*>\s*(?:License:\s*)?([^<]+)<', re.I)


def html_license(html: str) -> tuple[str | None, str | None]:
    """Read the licence statement arXiv prints in each HTML paper header -> (url, label)."""
    m = _LICENSE_RE.search(html)
    if m:
        return m.group(1).strip(), clean_text(m.group(2))
    m = re.search(r'href="((?:https?:)?//creativecommons\.org/[^"]+|https?://arxiv\.org/licenses/[^"]+)"', html)
    return (m.group(1), None) if m else (None, None)
