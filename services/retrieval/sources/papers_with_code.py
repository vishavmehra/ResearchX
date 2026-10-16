"""Papers With Code adapter (DATASET_SPEC §6.4) via the Hugging Face `pwc-archive` snapshot.

The live service is discontinued (paperswithcode.com/api/v1 redirects to Hugging Face, verified
2026-09-30). The archive is CC BY-SA 4.0 and frozen ("will not be updated"). Acquisition pins
each dataset to its Hugging Face commit sha and verifies every parquet file against the
LFS sha256 published by the Hub, so the download is reproducible and tamper-evident.

Normalization keeps only corpus papers: one SourceRecord per PwC paper page with its code
links and leaderboard results (community-curated claims, later cross-checked against paper text).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import httpx
import pyarrow.parquet as pq

from services.canonicalization.normalize import clean_text, normalize_title, parse_arxiv_id
from services.datasets.schemas import FailureCategory, LicenseInfo, Provenance, RecordKind, SourceRecord
from services.datasets.store import DATA, ROOT, Run, code_version, config, now_utc, rel, sha256_bytes
from services.retrieval.http import USER_AGENT, CachedHttp
from services.retrieval.sources import load_acquisition, save_acquisition

SOURCE = "PAPERS_WITH_CODE"
NORMALIZER_VERSION = "pwc-normalizer-0.1.0"
HF_API = "https://huggingface.co/api/datasets/{repo}"
HF_TREE = "https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
RAW = DATA / "raw" / "papers_with_code"


def _download(url: str, dest, expected_sha256: str) -> None:
    """Stream to a temp file, verify sha256, then atomically move into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                h.update(chunk)
                f.write(chunk)
    if h.hexdigest() != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {url}")
    tmp.replace(dest)


def acquire(run: Run, refresh: bool = False) -> None:
    cfg = config("pipeline")["sources"][SOURCE]
    http = CachedHttp("papers_with_code/api", min_interval=cfg["min_interval_s"], max_retries=cfg["max_retries"])
    manifest: dict[str, Any] = {"run_id": run.run_id, "datasets": {}}
    for key, repo in cfg["hf_datasets"].items():
        info = http.get(HF_API.format(repo=repo), refresh=refresh).json()
        rev = info["sha"]
        tree = http.get(HF_TREE.format(repo=repo, rev=rev)).json()
        files = []
        for f in tree:
            if f.get("type") != "file" or not f["path"].endswith(".parquet"):
                continue
            sha = (f.get("lfs") or {}).get("oid")
            dest = RAW / key / f["path"].rsplit("/", 1)[-1]
            url = HF_FILE.format(repo=repo, rev=rev, path=f["path"])
            if not (dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha):
                try:
                    _download(url, dest, sha)
                except (httpx.HTTPError, ValueError) as e:
                    run.fail(FailureCategory.retrieval_failure, url, "verified parquet file", str(e))
                    continue
            files.append({"path": rel(dest), "sha256": sha, "bytes": f.get("size"), "url": url})
        manifest["datasets"][key] = {
            "repo": repo,
            "revision": rev,
            "last_modified": info.get("lastModified"),
            "license": (info.get("cardData") or {}).get("license"),
            "retrieved_at": now_utc().isoformat(),
            "files": files,
        }
        run.note(f"{repo}@{rev[:10]}: {len(files)} parquet files verified")
    save_acquisition("papers_with_code", manifest)
    run.counts.update(datasets=len(manifest["datasets"]), files=sum(len(d["files"]) for d in manifest["datasets"].values()))


def _read(files: list[dict], columns: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(ROOT / f["path"], columns=columns).to_pylist())
    return rows


def _slug(paper_url: str | None) -> str | None:
    return paper_url.rstrip("/").rsplit("/", 1)[-1] if paper_url else None


def flatten_evaluations(tasks: list[dict]) -> list[dict]:
    """evaluation-tables rows: task -> datasets -> sota rows (-> subtasks, recursive)."""
    out: list[dict] = []

    def walk(task: dict, parents: list[str]) -> None:
        name = task.get("task")
        for ds in task.get("datasets") or []:
            sota = ds.get("sota") or {}
            for i, row in enumerate(sota.get("rows") or []):
                metrics = row.get("metrics") or {}
                if isinstance(metrics, str):
                    metrics = json.loads(metrics)
                out.append(
                    {
                        "task": name,
                        "parent_tasks": parents,
                        "dataset": ds.get("dataset"),
                        "row_index": i,
                        "model_name": row.get("model_name"),
                        "metrics": {k: v for k, v in (metrics.items() if isinstance(metrics, dict) else []) if v not in (None, "")},
                        "paper_title": row.get("paper_title"),
                        "paper_url": row.get("paper_url"),
                        "paper_date": row.get("paper_date"),
                        "uses_additional_data": row.get("uses_additional_data"),
                    }
                )
        for sub in task.get("subtasks") or []:
            walk(sub, parents + [name])

    for t in tasks:
        walk(t, [])
    return out


def normalize_all(context: dict[str, Any] | None = None) -> list[SourceRecord]:
    acq = load_acquisition("papers_with_code")
    if not acq or not context:
        return []
    corpus = context["arxiv_ids"]
    titles = context["titles"]
    ds_links, ds_eval = acq["datasets"]["links"], acq["datasets"]["evaluation_tables"]

    # One PwC record per corpus arXiv paper: code links + leaderboard rows linked by identifier
    # (arXiv id in the links table or in the result's paper URL); exact normalized title match
    # against trusted arXiv/S2 titles is the last resort and is recorded per result.
    papers: dict[str, dict[str, Any]] = {}

    def entry(arxiv_id: str, title: str | None) -> dict[str, Any]:
        return papers.setdefault(arxiv_id, {"title": title, "pwc_urls": set(), "links": [], "results": []})

    for row in _read(ds_links["files"]):
        parsed = parse_arxiv_id(row.get("paper_arxiv_id") or "")
        if not parsed or parsed[0] not in corpus:
            continue
        p = entry(parsed[0], row.get("paper_title"))
        p["pwc_urls"].add(row["paper_url"])
        p["links"].append({k: row.get(k) for k in ("repo_url", "is_official", "mentioned_in_paper", "mentioned_in_github", "framework")})

    for ev in flatten_evaluations(_read(ds_eval["files"])):
        url = ev["paper_url"] or ""
        parsed = parse_arxiv_id(url) if "arxiv.org" in url else None
        arxiv_id, method = (parsed[0], "result_paper_url_arxiv_id") if parsed else (titles.get(normalize_title(ev["paper_title"])), "exact_normalized_title")
        if not arxiv_id or arxiv_id not in corpus:
            continue
        entry(arxiv_id, ev["paper_title"])["results"].append({**ev, "arxiv_link_method": method})

    retrieved = ds_eval["retrieved_at"]
    version = f"links@{ds_links['revision'][:12]};evaluation-tables@{ds_eval['revision'][:12]}"
    out = []
    for arxiv_id, p in sorted(papers.items()):
        p["results"].sort(key=lambda r: (r["task"] or "", r["dataset"] or "", r["row_index"]))
        content = {"arxiv_id": arxiv_id, "links": p["links"], "results": p["results"]}
        pwc_url = sorted(p["pwc_urls"])[0] if p["pwc_urls"] else None
        # papers known only through leaderboard rows have no PwC page: cite the pinned archive dataset
        archive_url = f"https://huggingface.co/datasets/{ds_eval['repo']}/tree/{ds_eval['revision']}"
        out.append(
            SourceRecord(
                source_record_id=f"pwc:{arxiv_id}",
                source_name=SOURCE,
                source_identifier=arxiv_id,
                source_url=pwc_url or archive_url,
                source_version=version,
                record_kind=RecordKind.PAPER_METADATA,
                title=clean_text(p["title"]) or None,
                arxiv_id=arxiv_id,
                identifiers={"arxiv": arxiv_id, **({"pwc": _slug(pwc_url)} if pwc_url else {})},
                urls={"pwc": pwc_url} if pwc_url else {},
                license=LicenseInfo(content_license="CC-BY-SA-4.0", metadata_license="CC-BY-SA-4.0", license_source="hf_dataset_card", redistribution_allowed=True, training_allowed=True),
                extra={"code_links": p["links"], "evaluation_results": p["results"], "pwc_urls": sorted(p["pwc_urls"])},
                provenance=Provenance(
                    source_name=SOURCE,
                    source_url=pwc_url or archive_url,
                    source_identifier=arxiv_id,
                    source_version=version,
                    retrieved_at=retrieved,
                    content_hash=sha256_bytes(json.dumps(content, sort_keys=True, default=str).encode()),
                    processing_version=code_version(),
                    extractor_version=NORMALIZER_VERSION,
                    raw_path=ds_eval["files"][0]["path"] if ds_eval["files"] else None,
                ),
            )
        )
    return out
