"""Selective, license-aware full-text acquisition (DATASET_SPEC §31, §36).

- Source: arXiv HTML (LaTeXML) from export.arxiv.org, the host arXiv designates for programmatic
  access; PDF fallback when no HTML exists. One request every 3 s (arXiv ToU).
- Licence: read from each paper's own HTML/abs page header, mapped via data/manifests/licenses.yaml.
  CC0 / CC BY / CC BY-SA -> data/fulltext/permitted/; everything else (incl. the default arXiv
  non-exclusive licence) -> data/fulltext/restricted/ (internal research use only, never
  redistributed). Both directories are git-ignored.
- The full-text index (data/normalized/sources/fulltext_index.jsonl) records what was fetched,
  from where, when, under which licence, and where it is stored; deduplication re-applies it.
"""

from __future__ import annotations

import json
import math
from typing import Any

from services.datasets import manifests
from services.datasets.schemas import FailureCategory, FulltextFormat, FulltextIndexEntry, FulltextInfo, LicenseInfo, Paper, SourceRecord
from services.datasets.store import DATA, Run, atomic_write_bytes, config, now_utc, path, read_jsonl, read_models, rel, sha256_bytes, write_jsonl
from services.retrieval.http import CachedHttp, FetchError
from services.retrieval.sources.arxiv import html_license

FULLTEXT_INDEX = DATA / "normalized" / "sources" / "fulltext_index.jsonl"
HTML_URL = "https://export.arxiv.org/html/{id}{v}"
PDF_URL = "https://export.arxiv.org/pdf/{id}{v}"
ABS_URL = "https://export.arxiv.org/abs/{id}{v}"


def load_index() -> dict[str, dict]:
    return {row["arxiv_id"]: row for row in read_jsonl(FULLTEXT_INDEX)}


def write_index(index: dict[str, dict]) -> dict:
    rows = [{**index[k], "record_type": "fulltext_index"} for k in sorted(index)]
    return write_jsonl(FULLTEXT_INDEX, [FulltextIndexEntry.model_validate(r) for r in rows])


def apply_fulltext_index(papers: list[Paper]) -> int:
    index = load_index()
    applied = 0
    for p in papers:
        row = index.get(p.arxiv_id or "")
        if row:
            p.fulltext = FulltextInfo.model_validate(row["fulltext"])
            p.license = LicenseInfo.model_validate(row["license"])
            applied += 1
    return applied


def storage_dir(klass: str, arxiv_id: str):
    return DATA / "fulltext" / klass / arxiv_id.replace("/", "_")


def fulltext_priority(p: Paper, records: dict[str, SourceRecord]) -> float:
    """selection-0.1.0: cross-checkable PwC results, code link, priority-area breadth, recency (HTML likelier)."""
    pwc = records.get(f"pwc:{p.arxiv_id}")
    n_results = len(pwc.extra.get("evaluation_results", [])) if pwc else 0
    has_code = bool(pwc and pwc.extra.get("code_links"))
    areas = len(p.relevance.areas) if p.relevance else 0
    return 3.0 * (n_results > 0) + math.log1p(n_results) + 2.0 * has_code + min(areas, 4) + 0.5 * ((p.publication_date or "") >= "2024-01-01")


def select_candidates(papers: list[Paper], records: dict[str, SourceRecord], n: int | None = None) -> list[Paper]:
    """Full-text queue: promoted Layer 0 candidates first, then the top relevant corpus papers (deterministic)."""
    from services.retrieval.sources.arxiv_oai import promoted_ids

    promoted = promoted_ids()
    budget = config("pipeline")["discovery"]["fulltext_existing_relevant"]
    pool = [p for p in papers if p.arxiv_id and p.relevance and p.relevance.is_relevant]
    key = lambda p: (-fulltext_priority(p, records), p.paper_id)  # noqa: E731
    first = sorted((p for p in pool if p.arxiv_id in promoted), key=key)
    rest = sorted((p for p in pool if p.arxiv_id not in promoted), key=key)[:budget]
    queue = first + rest
    return queue[:n] if n else queue


def _latest_version(p: Paper, records: dict[str, SourceRecord]) -> str:
    for sid in p.source_record_ids:
        rec = records.get(sid)
        if rec and rec.source_name == "ARXIV" and rec.arxiv_version:
            return rec.arxiv_version
    return ""


def acquire(run: Run, sample_size: int | None = None, refresh: bool = False) -> None:
    n = sample_size or None  # None = the whole selection-0.1.0 queue
    pdf_fallback = bool(config("pipeline")["fulltext"].get("pdf_fallback", False))
    papers = read_models(path("papers"), Paper)
    records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
    http = CachedHttp("arxiv_fulltext", min_interval=config("pipeline")["sources"]["ARXIV"]["min_interval_s"], max_retries=4)
    index = load_index()
    queue = select_candidates(papers, records, n)
    run.counts["queue"] = len(queue)
    for i, p in enumerate(queue):
        prior = index.get(p.arxiv_id)
        if prior and not refresh and (DATA.parent / prior["fulltext"]["path"]).exists():
            run.counts["skipped_existing"] = run.counts.get("skipped_existing", 0) + 1
            continue
        v = _latest_version(p, records)
        try:
            row = _fetch_one(http, p, v, pdf_fallback)
        except FetchError as e:
            no_html = e.status == 404 and not pdf_fallback
            run.counts["no_html" if no_html else "fetch_failed"] = run.counts.get("no_html" if no_html else "fetch_failed", 0) + 1
            run.fail(FailureCategory.retrieval_failure, f"full text {p.arxiv_id}{v}", "HTML full text" if no_html else "HTML or PDF full text",
                     "no arXiv HTML (PDF path deferred)" if no_html else str(e), severity="LOW" if no_html else "MEDIUM")
            continue
        if i % 25 == 0:
            print(f"  full text {i + 1}/{len(queue)}", flush=True)
            write_index({**index, p.arxiv_id: row})  # checkpoint: an interruption keeps what was fetched
        index[p.arxiv_id] = row
        key = f"{row['fulltext']['license_status'].lower()}_{row['fulltext']['format'].lower()}"
        run.counts[key] = run.counts.get(key, 0) + 1
    run.output(write_index(index))
    applied = apply_fulltext_index(papers)
    run.output(write_jsonl(path("papers"), papers))
    run.counts.update(indexed=len(index), papers_with_fulltext=applied, live_requests=http.requests_made)


def _fetch_one(http: CachedHttp, p: Paper, v: str, pdf_fallback: bool = True) -> dict[str, Any]:
    fmt, url = FulltextFormat.HTML, HTML_URL.format(id=p.arxiv_id, v=v)
    try:
        resp = http.get(url, cache=False)
        lic_url, lic_label = html_license(resp.text())
    except FetchError as e:
        if e.status != 404 or not pdf_fallback:
            raise
        fmt, url = FulltextFormat.PDF, PDF_URL.format(id=p.arxiv_id, v=v)
        resp = http.get(url, cache=False)
        lic_url, lic_label = html_license(http.get(ABS_URL.format(id=p.arxiv_id, v=v), cache=False).text())
    lic_id = manifests.match_license(lic_url) or manifests.match_license(lic_label)
    klass = manifests.fulltext_class(lic_id)
    flags = manifests.license_flags(lic_id)
    out_dir = storage_dir(klass, p.arxiv_id)
    src = out_dir / ("source.html" if fmt is FulltextFormat.HTML else "source.pdf")
    atomic_write_bytes(src, resp.body)
    retrieved = now_utc()
    prov = {
        "arxiv_id": p.arxiv_id,
        "version": v or None,
        "url": url,
        "retrieved_at": retrieved.isoformat(),
        "sha256": sha256_bytes(resp.body),
        "bytes": len(resp.body),
        "format": fmt.value,
        "license_statement": lic_label,
        "license_url": lic_url,
        "content_license": lic_id,
        "storage_class": klass,
    }
    atomic_write_bytes(out_dir / "provenance.json", json.dumps(prov, indent=1).encode())
    return {
        "arxiv_id": p.arxiv_id,
        "fulltext": FulltextInfo(
            full_text_available=True,
            full_text_source="ARXIV",
            format=fmt,
            retrieval_url=url,
            retrieval_timestamp=retrieved,
            license_status="PERMITTED" if klass == "permitted" else "RESTRICTED",
            path=rel(src),
            sha256=prov["sha256"],
        ).model_dump(mode="json"),
        "license": LicenseInfo(
            content_license=lic_id,
            metadata_license="CC0-1.0",
            license_source="arxiv_html_header" if fmt is FulltextFormat.HTML else "arxiv_abs_page",
            redistribution_allowed=flags["redistribution_allowed"],
            training_allowed=flags["training_allowed"],
        ).model_dump(mode="json"),
    }


