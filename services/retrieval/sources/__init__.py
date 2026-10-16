"""Source adapters. Each module exposes SOURCE, acquire(...) and normalize_all() -> [SourceRecord].

Adding a source = new module + registry entry in data/manifests/sources.yaml + ADAPTERS entry.
Acquisition manifests (data/raw/<source>/acquisition.json) list the cached payloads a stage
produced, so normalization is a pure function of cached raw data.
"""

from __future__ import annotations

import json
from typing import Any

from services.datasets.store import DATA, ROOT, atomic_write_bytes, sha256_bytes


def acquisition_path(source_dir: str):
    return DATA / "raw" / source_dir / "acquisition.json"


def save_acquisition(source_dir: str, payload: dict[str, Any]) -> None:
    atomic_write_bytes(acquisition_path(source_dir), json.dumps(payload, indent=1, sort_keys=True).encode())


def load_acquisition(source_dir: str) -> dict[str, Any] | None:
    p = acquisition_path(source_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_cached(page: dict[str, Any]) -> bytes:
    """Read a cached payload listed in an acquisition manifest, verifying its checksum (gzip-aware)."""
    from services.retrieval.http import read_body

    body = read_body(ROOT / page["body_path"])
    if sha256_bytes(body) != page["sha256"]:
        raise ValueError(f"checksum mismatch for {page['body_path']}")
    return body


# source registry id -> module name (== raw cache directory name). Order matters for normalization:
# discovery sources (OpenAlex, arXiv OAI-PMH) first, then enrichment sources.
ADAPTER_MODULES = {
    "OPENALEX": "openalex",
    "ARXIV_OAI_PMH": "arxiv_oai",
    "SEMANTIC_SCHOLAR": "semantic_scholar",
    "ARXIV": "arxiv",
    "PAPERS_WITH_CODE": "papers_with_code",
}


def acquired_adapters():
    """Adapters whose acquisition manifest exists (lazy import: only what was acquired)."""
    import importlib

    return [
        importlib.import_module(f"services.retrieval.sources.{mod}")
        for mod in ADAPTER_MODULES.values()
        if acquisition_path(mod).exists()
    ]
