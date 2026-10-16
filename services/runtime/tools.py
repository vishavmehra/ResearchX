"""External research tools for evidence gaps (Phases 10-12; runtime provider registry 2026-10-07).

Scholarly search: Semantic Scholar, OpenAlex, arXiv, CORE, OpenReview. Web search: Tavily. Unpaywall is a DOI lookup
(open-access location and licence) used when a runtime source is proposed for promotion. Results are RUNTIME sources:
candidates for one answer, never truth. validate_sources() keeps only results that pass deterministic checks (https
URL, no instruction-like text, not a duplicate URL or identifier) and grades the domain's reliability; rejected results
are counted in the trace and never persisted. Keys come from the environment only.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx

from services.canonicalization.normalize import normalize_doi, parse_arxiv_id, reconstruct_abstract

TAVILY_URL = "https://api.tavily.com/search"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"
CORE_URL = "https://api.core.ac.uk/v3/search/works/"
OPENREVIEW_URL = "https://api2.openreview.net/notes/search"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/"
INJECTION = re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions|disregard (?:the )?(?:system|previous)|"
                       r"you are now|system prompt|<\s*/?\s*(?:script|iframe)", re.I)
SCHOLARLY = ("arxiv.org", "openreview.net", "aclanthology.org", "semanticscholar.org", "doi.org", "neurips.cc", "mlr.press",
             "thecvf.com", "acm.org", "ieee.org", "springer.com", "nature.com", "jmlr.org", "aaai.org", "ijcai.org", "openalex.org",
             "core.ac.uk")
TECHNICAL = ("github.com", "huggingface.co", "paperswithcode.com", "pytorch.org", "tensorflow.org")
ARXIV_INTERVAL_S = 3.0  # arXiv API terms of use: at most one request every 3 seconds
_arxiv_lock, _arxiv_last = threading.Lock(), [0.0]


def _timeout(total: float) -> httpx.Timeout:
    return httpx.Timeout(total, connect=min(3.0, total))  # an unreachable host fails fast instead of holding the run


def tavily_search(query: str, max_results: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    """One basic search (1 credit). Returns (results, usage)."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    r = httpx.post(TAVILY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=_timeout(timeout),
                   json={"query": query[:400], "search_depth": "basic", "max_results": max_results, "include_answer": False})
    r.raise_for_status()
    rows = [{"provider": "tavily", "url": x.get("url"), "title": x.get("title"), "text": x.get("content") or "", "score": x.get("score")}
            for x in r.json().get("results", [])]
    return rows, {"tavily_credits": 1, "tavily_search_depth": "basic"}


def s2_search(query: str, limit: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    key = os.environ.get("S2_API_KEY")
    r = httpx.get(S2_SEARCH_URL, params={"query": query[:300], "limit": limit, "fields": "title,abstract,year,externalIds,url,venue"},
                  headers={"x-api-key": key} if key else None, timeout=_timeout(timeout))
    r.raise_for_status()
    rows = []
    for p in r.json().get("data") or []:
        ids = p.get("externalIds") or {}
        url = f"https://arxiv.org/abs/{ids['ArXiv']}" if ids.get("ArXiv") else p.get("url")
        rows.append({"provider": "semantic_scholar", "url": url, "title": p.get("title"), "text": p.get("abstract") or "",
                     "year": p.get("year"), "venue": p.get("venue"), "arxiv_id": ids.get("ArXiv"), "doi": normalize_doi(ids.get("DOI")),
                     "source_id": f"s2:{p.get('paperId')}" if p.get("paperId") else None})
    return rows, {"s2_calls": 1}


S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/"


def s2_paper(identifier: str, timeout: float = 8) -> dict | None:
    """One work by identifier ("arXiv:2111.06377", "DOI:10.1/x"): exact lookup, no search. None when S2 has no such work."""
    key = os.environ.get("S2_API_KEY")
    r = httpx.get(S2_PAPER_URL + quote(identifier, safe=":/"), params={"fields": "title,abstract,year,externalIds,url,venue,authors"},
                  headers={"x-api-key": key} if key else None, timeout=_timeout(timeout))
    if r.status_code == 404:
        return None
    r.raise_for_status()
    p = r.json()
    ids = p.get("externalIds") or {}
    return {"provider": "semantic_scholar", "url": f"https://arxiv.org/abs/{ids['ArXiv']}" if ids.get("ArXiv") else p.get("url"),
            "title": p.get("title"), "text": p.get("abstract") or "", "year": p.get("year"), "venue": p.get("venue"), "arxiv_id": ids.get("ArXiv"),
            "doi": normalize_doi(ids.get("DOI")), "source_id": f"s2:{p.get('paperId')}" if p.get("paperId") else None,
            "authors": [a.get("name") for a in (p.get("authors") or [])[:12] if a.get("name")]}


def openalex_search(query: str, limit: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    """OpenAlex /works search (metadata CC0). An API key and contact e-mail are used when configured."""
    params = {"search": query[:300], "per-page": limit, "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location"}
    if key := os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = key
    if mailto := os.environ.get("RESEARCHX_CONTACT_EMAIL"):
        params["mailto"] = mailto
    r = httpx.get(OPENALEX_URL, params=params, timeout=_timeout(timeout))
    r.raise_for_status()
    rows = []
    for w in r.json().get("results") or []:
        doi = normalize_doi(w.get("doi"))
        landing = (w.get("primary_location") or {}).get("landing_page_url") or ""
        arx = parse_arxiv_id(doi) if doi and doi.startswith("10.48550/arxiv.") else parse_arxiv_id(landing) if "arxiv.org" in landing else None
        arxiv_id = arx[0] if arx else None
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://doi.org/{doi}" if doi else w.get("id")
        rows.append({"provider": "openalex", "url": url, "title": w.get("title"), "text": reconstruct_abstract(w.get("abstract_inverted_index")) or "",
                     "year": w.get("publication_year"), "arxiv_id": arxiv_id, "doi": doi, "source_id": w.get("id")})
    return rows, {"openalex_calls": 1}


def arxiv_search(query: str, limit: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    """arXiv Atom API, all terms required. Requests closer than 3 s apart are refused (traced), never delayed."""
    from services.retrieval.sources.arxiv import parse_feed  # lxml-based parser shared with the dataset adapter

    terms = [t for t in re.findall(r"[\w.+-]+", query) if len(t) > 1][:6]
    if not terms:
        return [], {"arxiv_calls": 0}
    with _arxiv_lock:
        if time.monotonic() - _arxiv_last[0] < ARXIV_INTERVAL_S:
            raise RuntimeError("arXiv rate limit: one request every 3 s")
        _arxiv_last[0] = time.monotonic()
    r = httpx.get(ARXIV_URL, params={"search_query": " AND ".join(f"all:{t}" for t in terms), "max_results": limit}, timeout=_timeout(timeout))
    r.raise_for_status()
    rows = [{"provider": "arxiv", "url": f"https://arxiv.org/abs/{e['arxiv_id']}", "title": e.get("title"), "text": e.get("summary") or "",
             "year": int(e["published"][:4]) if (e.get("published") or "")[:4].isdigit() else None, "arxiv_id": e["arxiv_id"],
             "doi": e.get("journal_doi"), "source_id": f"arxiv:{e['arxiv_id']}"} for e in parse_feed(r.content) if e.get("arxiv_id")]
    return rows, {"arxiv_calls": 1}


def ready_at(provider: str) -> float:
    """Monotonic time from which `provider` accepts another request: 3 s after the last arXiv call, else at once (0)."""
    return _arxiv_last[0] + ARXIV_INTERVAL_S if provider == "arxiv" else 0.0


def core_search(query: str, limit: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    """CORE v3 works search. Only abstracts are used: full texts need per-work licence checks."""
    key = os.environ.get("CORE_API_KEY")
    if not key:
        raise RuntimeError("CORE_API_KEY is not set")
    r = httpx.get(CORE_URL, params={"q": query[:300], "limit": limit}, headers={"Authorization": f"Bearer {key}"}, timeout=_timeout(timeout),
                  follow_redirects=True)
    r.raise_for_status()
    rows = []
    for w in r.json().get("results") or []:
        arxiv_id, doi = w.get("arxivId"), normalize_doi(w.get("doi"))
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://doi.org/{doi}" if doi else f"https://core.ac.uk/works/{w.get('id')}"
        rows.append({"provider": "core", "url": url, "title": w.get("title"), "text": w.get("abstract") or "", "year": w.get("yearPublished"),
                     "arxiv_id": arxiv_id, "doi": doi, "source_id": f"core:{w.get('id')}"})
    return rows, {"core_calls": 1}


def openreview_search(query: str, limit: int = 5, timeout: float = 30) -> tuple[list[dict], dict]:
    """OpenReview API v2 term search over public forum notes."""
    r = httpx.get(OPENREVIEW_URL, params={"term": query[:200], "type": "terms", "content": "all", "group": "all", "source": "forum", "limit": limit},
                  timeout=_timeout(timeout))
    r.raise_for_status()
    rows = []
    for n in r.json().get("notes") or []:
        c = n.get("content") or {}
        val = {k: c[k].get("value") if isinstance(c.get(k), dict) else c.get(k) for k in ("title", "abstract", "venue")}  # API v2 wraps values
        year = datetime.fromtimestamp(n["cdate"] / 1000, timezone.utc).year if isinstance(n.get("cdate"), (int, float)) else None
        rows.append({"provider": "openreview", "url": f"https://openreview.net/forum?id={n.get('forum') or n.get('id')}", "title": val["title"],
                     "text": val["abstract"] or "", "year": year, "venue": val["venue"], "source_id": f"openreview:{n.get('id')}"})
    return rows, {"openreview_calls": 1}


def unpaywall_lookup(doi: str, timeout: float = 12) -> dict:
    """Open-access location and licence of one DOI (Unpaywall requires a contact e-mail)."""
    email = os.environ.get("UNPAYWALL_EMAIL")
    if not email:
        raise RuntimeError("UNPAYWALL_EMAIL is not set")
    r = httpx.get(UNPAYWALL_URL + quote(doi, safe="/"), params={"email": email}, timeout=_timeout(timeout))
    r.raise_for_status()
    j = r.json()
    best = j.get("best_oa_location") or {}
    return {"doi": doi, "is_oa": j.get("is_oa"), "license": best.get("license"), "oa_url": best.get("url_for_pdf") or best.get("url"),
            "host_type": best.get("host_type")}


# provider -> (search function, required environment variable, query form). Scholarly APIs get keyword terms; the web
# search gets the question as asked.
PROVIDERS = {
    "semantic_scholar": (s2_search, None, "terms"),
    "openalex": (openalex_search, None, "terms"),
    "arxiv": (arxiv_search, None, "terms"),
    "core": (core_search, "CORE_API_KEY", "terms"),
    "openreview": (openreview_search, None, "terms"),
    "tavily": (tavily_search, "TAVILY_API_KEY", "question"),
}


def reliability(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if any(host == d or host.endswith("." + d) for d in SCHOLARLY):
        return "scholarly"
    if any(host == d or host.endswith("." + d) for d in TECHNICAL):
        return "technical"
    return "web"


def validate_sources(results: list[dict], known_urls: set[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    """Deterministic gate for runtime sources. Returns (accepted, rejected-with-reason). A result whose DOI or arXiv id was
    already accepted from another provider is the same work: it is not added twice, its provider is recorded instead."""
    accepted, rejected, seen, by_id = [], [], set(known_urls), {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in results:
        url, text = (r.get("url") or "").strip(), (r.get("text") or "").strip()
        ids = [k for k in (("doi", normalize_doi(r.get("doi"))), ("arxiv", r.get("arxiv_id"))) if k[1]]
        same = next((by_id[k] for k in ids if k in by_id), None)
        reason = ("not_https" if not url.startswith("https://") else "duplicate" if url in seen else "duplicate_identifier" if same is not None
                  else "empty_content" if len(text) < 40 else "instruction_like_text" if INJECTION.search(text) or INJECTION.search(r.get("title") or "")
                  else None)
        if reason:
            if same is not None and reason == "duplicate_identifier":
                same["also_found_by"].append({"provider": r.get("provider"), "url": url, "source_id": r.get("source_id")})
            rejected.append({"url": url, "provider": r.get("provider"), "reason": reason,  # an injection-like title is never echoed
                             "title": None if reason == "instruction_like_text" else (r.get("title") or "")[:300] or None})
            continue
        seen.add(url)
        a = {**r, "url": url, "text": text[:1500], "doi": normalize_doi(r.get("doi")), "reliability": reliability(url),
             "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "retrieved_at": now, "also_found_by": []}
        accepted.append(a)
        for k in ids:
            by_id.setdefault(k, a)
    return accepted, rejected
