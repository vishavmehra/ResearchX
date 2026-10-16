"""Runtime full-text reading for evidence gaps (core-engine brief §1: external result -> relevant extraction).

When a requested field is still unknown after the abstracts, the best-matching external paper that has an arXiv id is
read from its arXiv HTML (LaTeXML) and parsed by the corpus parser (services/extraction/parse.py). Only the passages and
tables that name a requested field (its cue) or the asked metric go on to extraction, so the extraction prompt stays
small. One paper per run under a strict timeout. Parsed text is cached in memory (public paper text, no user linkage)
and never written to the corpus: what it yields is runtime evidence, not canonical evidence.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime

import httpx

from services.datasets import manifests
from services.datasets.schemas import LicenseInfo
from services.evidence import grounding as G
from services.extraction.parse import parse_html
from services.retrieval.sources.arxiv import html_license

HTML_URLS = ("https://arxiv.org/html/{id}", "https://ar5iv.labs.arxiv.org/html/{id}")  # ar5iv: papers arXiv has no HTML for
USER_AGENT = "ResearchX/0.1 (evidence-gap reader; one paper per question)"
CACHE_SIZE = 32
SETUP = re.compile(r"implementation|training|experiment(?:al)?\s+(?:setup|settings?|details)|set-?up|hyper-?param|details|optimi[sz]ation|"
                   r"evaluation|protocol|fine-?tun|pre-?train|appendix|supplementary", re.I)  # appendices hold most protocols and seeds
_cache: OrderedDict[str, dict] = OrderedDict()
_lock = threading.Lock()


def table_text(t) -> str:
    """Caption, then one line per row with cells joined by ' | ': the corpus chunk format (scripts/db/load_release.py),
    which the table-cell validators read."""
    body = "\n".join(" | ".join(str(c) for c in row) for row in t.rows)
    return (t.caption or "") + ("\n" + body if body else "")


def read(arxiv_id: str, timeout: float = 4.0) -> dict:
    """Parsed units of one arXiv paper: url, retrieved_at, sha256 of the HTML, licence id, whether it may be
    redistributed, and units [{unit_type, unit_id, section, text, table_type}]. Raises when the paper cannot be read
    within `timeout` seconds or has no LaTeXML HTML."""
    with _lock:
        if arxiv_id in _cache:
            _cache.move_to_end(arxiv_id)
            return _cache[arxiv_id]
    deadline, error, r = time.monotonic() + timeout, None, None
    for base in HTML_URLS:
        left = deadline - time.monotonic()
        if left <= 0.2:
            break
        url = base.format(id=arxiv_id)
        try:
            r = httpx.get(url, timeout=httpx.Timeout(left, connect=min(3.0, left)), follow_redirects=True, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as e:
            r, error = None, f"{type(e).__name__} from {url}"
            continue
        if r.status_code == 200 and "ltx_document" in r.text:
            break
        error = f"HTTP {r.status_code} from {url}" if r.status_code != 200 else f"no LaTeXML document at {url}"
        r = None
    if r is None:
        raise RuntimeError(error or f"no time left within {timeout} s")
    sha = hashlib.sha256(r.content).hexdigest()
    doc = parse_html(r.text, document_meta={"paper_id": f"arxiv:{arxiv_id}", "source_record_id": str(r.url), "source_url": str(r.url),
                                            "source_path": "", "source_sha256": sha}, license=LicenseInfo())
    lic_url, lic_label = html_license(r.text)
    lic = manifests.match_license(lic_url) or manifests.match_license(lic_label)
    sections = {s.section_id: s.title for s in doc.sections}
    units = [{"unit_type": "paragraph", "unit_id": p.paragraph_id, "section": sections.get(p.section_id), "text": p.text, "table_type": None}
             for p in doc.paragraphs]
    units += [{"unit_type": "table", "unit_id": t.table_id, "section": sections.get(t.section_id), "text": table_text(t), "table_type": t.table_type}
              for t in doc.tables]
    out = {"url": str(r.url), "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"), "sha256": sha, "license": lic,
           "redistribution_allowed": manifests.license_flags(lic)["redistribution_allowed"] is True, "units": units,
           "title": doc.title, "abstract": doc.abstract}
    with _lock:
        _cache[arxiv_id] = out
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def relevant(units: list[dict], fields: list[str], req: G.Request, limit: int = 4, chars: int = 1500) -> list[dict]:
    """The units most likely to state the requested values. A unit qualifies only by naming a requested field (its cue)
    or, for a score, a metric that fits the asked one; naming the asked dataset or model, and being a hyperparameter or
    results table, ranks it higher. Text is cut to `chars` (extraction reads, and validation checks, the same text)."""
    cues = [G.SELECT_CUES[f] for f in fields if f in G.SELECT_CUES]
    score_asked = "evaluation.score" in fields
    dataset, model = G.tokens(req.dataset), G.tokens(req.model)
    ranked = []
    for i, u in enumerate(units):
        text = G.canon(u["text"])
        hits = sum(bool(rx.search(text)) for rx in cues)
        if score_asked and any(G.metric_ok(m.group(1), req.metric) for m in G.METRIC.finditer(text)):
            hits += 1
        if not hits:
            continue
        toks = G.tokens(u["text"])
        rank = (2 * hits + 2 * bool(dataset and dataset <= toks) + bool(model and model & toks)
                + 2 * (u.get("table_type") in ("HYPERPARAMETERS", "RESULTS")) + (u["unit_type"] == "table")
                + 2 * bool(SETUP.search(u.get("section") or "")))  # settings live in implementation / training / setup sections
        ranked.append((-rank, i, u))
    return [{**u, "text": u["text"][:chars]} for _, _, u in sorted(ranked, key=lambda x: (x[0], x[1]))[:limit]]
