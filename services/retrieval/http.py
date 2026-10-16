"""Cached, rate-limited, retrying HTTP client with provenance (DATASET_SPEC §33).

- Cache: every successful response body + a .meta.json sidecar under data/raw/<source>/.
  Re-running a stage replays the cache (resumable, no repeated downloads, reproducible).
- Retries: exponential backoff with jitter on timeouts/429/5xx; honors Retry-After.
- Rate limit: minimum interval between requests per client (arXiv ToU: 1 request / 3 s).
- Secrets: api keys and auth headers never reach cache keys, meta files, logs, or errors.
- Writes are atomic, so an interrupted download never leaves a corrupt cache entry.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from services.datasets.store import DATA, atomic_write_bytes, now_utc, rel, sha256_bytes

USER_AGENT = "ResearchX-dataset-pipeline/0.1 (research data acquisition)"
SECRET_PARAMS = frozenset({"api_key", "apikey", "key", "token", "access_token"})
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
KEEP_HEADERS = ("content-type", "etag", "last-modified", "retry-after")


class FetchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Response:
    status: int
    body: bytes
    meta: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def redact_params(params: dict[str, Any] | None) -> dict[str, Any]:
    return {k: ("<redacted>" if k.lower() in SECRET_PARAMS else v) for k, v in sorted((params or {}).items())}


class CachedHttp:
    def __init__(
        self,
        source_dir: str,
        *,
        min_interval: float = 1.0,
        max_retries: int = 6,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        cache_root: Path | None = None,
        max_backoff: float = 120.0,
        compress: bool = False,
    ):
        self.cache_dir = (cache_root or DATA / "raw") / source_dir
        # compress=True stores bodies as deterministic gzip (mtime=0); sha256 is always over the raw bytes.
        self.compress = compress
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self._last = 0.0
        self.requests_made = 0
        self.cache_hits = 0
        # secret_headers are sent but excluded from cache keys and meta.
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, **(headers or {}), **(secret_headers or {})},
            transport=transport,
        )

    def _key(self, method: str, url: str, params: dict | None, body: Any) -> str:
        payload = json.dumps([method, url, redact_params(params), body], sort_keys=True, default=str)
        return sha256_bytes(payload.encode())[:32]

    def _wait(self) -> None:
        delay = self.min_interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        cache: bool = True,
        refresh: bool = False,
    ) -> Response:
        key = self._key(method, url, params, json_body)
        body_path = self.cache_dir / (f"{key}.body.gz" if self.compress else f"{key}.body")
        meta_path = self.cache_dir / f"{key}.meta.json"
        if cache and not refresh and body_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            body = read_body(body_path)
            if sha256_bytes(body) == meta.get("sha256"):  # verify integrity on replay
                self.cache_hits += 1
                meta["body_path"] = rel(body_path)
                return Response(meta["status"], body, meta, from_cache=True)

        last_error = ""
        for attempt in range(self.max_retries + 1):
            self._wait()
            self.requests_made += 1
            try:
                r = self.client.request(method, url, params=params, json=json_body)
            except httpx.TransportError as e:  # timeouts, connection resets
                last_error = f"{type(e).__name__}"
                self._backoff(attempt, None)
                continue
            if r.status_code in RETRY_STATUS:
                last_error = f"HTTP {r.status_code}"
                if attempt < self.max_retries:
                    self._backoff(attempt, r.headers.get("retry-after"))
                continue
            if r.status_code >= 400:
                raise FetchError(f"{method} {url} -> HTTP {r.status_code}", r.status_code)
            meta = {
                "method": method,
                "url": url,
                "final_url": str(r.url.copy_with(query=None)),
                "params": redact_params(params),
                "request_body_sha256": sha256_bytes(json.dumps(json_body, sort_keys=True).encode()) if json_body is not None else None,
                "status": r.status_code,
                "retrieved_at": now_utc().isoformat(),
                "sha256": sha256_bytes(r.content),
                "bytes": len(r.content),
                "headers": {h: r.headers[h] for h in KEEP_HEADERS if h in r.headers},
                "rate_limit": {h.lower(): v for h, v in r.headers.items() if h.lower().startswith("x-ratelimit")},
                "attempts": attempt + 1,
            }
            if cache:
                meta["compressed"] = self.compress
                atomic_write_bytes(body_path, gzip.compress(r.content, mtime=0) if self.compress else r.content)
                atomic_write_bytes(meta_path, json.dumps(meta, indent=1, sort_keys=True).encode())
                meta["body_path"] = rel(body_path)
            return Response(r.status_code, r.content, meta)
        raise FetchError(f"{method} {url} failed after {self.max_retries + 1} attempts ({last_error})")

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        wait = min(self.max_backoff, 2.0 * (2**attempt)) + random.uniform(0, 1.0)
        if retry_after:
            try:
                wait = max(wait, min(float(retry_after), self.max_backoff))
            except ValueError:
                pass
        time.sleep(wait)

    def get(self, url: str, **kw: Any) -> Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, json_body: Any, **kw: Any) -> Response:
        return self.request("POST", url, json_body=json_body, **kw)

    def body_path_for(self, method: str, url: str, params: dict | None = None, json_body: Any = None) -> str:
        suffix = ".body.gz" if self.compress else ".body"
        return rel(self.cache_dir / f"{self._key(method, url, params, json_body)}{suffix}")


def read_body(p: Path) -> bytes:
    """Raw payload bytes of a cache entry (transparently gunzips *.gz bodies)."""
    data = p.read_bytes()
    return gzip.decompress(data) if p.name.endswith(".gz") else data
