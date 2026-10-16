"""Runtime-discovered papers in analyses (docs/ARCHITECTURE.md §69): from what a researcher types to validated records.

A reference is what people write: a title, a DOI, an arXiv id or URL, a Semantic Scholar or ACL Anthology URL, or a corpus
paper. Resolution never rests on similarity alone:
  1. the corpus paper with the same DOI or arXiv id; a typed title equal (normalised) to a corpus title, flagged as such;
  2. otherwise, only when the caller opted in, the approved scholarly providers: an exact identifier lookup (Semantic
     Scholar by DOI / arXiv id, or the arXiv full text itself), else a title search through the shared provider phase
     (orchestrator._external, routing need `papers`: Semantic Scholar, OpenAlex, arXiv; per-user cache, deadline, and the
     deterministic source gate that drops instruction-like text). Results are deduplicated on DOI / arXiv id; a typed
     title resolves only when a result's title matches it closely (token Jaccard >= 0.8) and no other result does as well,
     otherwise the candidates are returned for the user to pick. A found work whose identifier is in the corpus is that
     corpus paper (canonical), never a duplicate.
The paper is read: its arXiv full text when it has one (licence and redistribution recorded; quotes withheld when the
licence does not allow redistribution), else its abstract alone, and every setting not found is "not stated in the
accessible text". Settings and evaluation conditions go through the shared schema-bound extractor plus its runtime checks
(the caller's `extract`), result rows through `extract_results`; a row is kept only when the deterministic checks pass (the
span is in the passage and holds the value, the metric is named beside it or heads its table column, the dataset is named,
no instruction-like text). Kept rows become runtime Experiment records with configuration-level ids (paper, model,
dataset, split), never title-based; their quotes are served beside corpus evidence as RUNTIME_EXTRACTED; training settings
keep the stage their passage names (services/analysis/configs.py). Nothing is written to the corpus: bundles live in the
process, per user, for a few hours. Promotion stays the reviewed step of the Knowledge Map. Only what the user typed (a
reference or a topic) reaches a provider, and only when they asked for papers outside the corpus; private documents are
never read here.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import ChainMap, Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from services.analysis.configs import stage_of
from services.analysis.enrich import BATCHES, CONDITIONS
from services.canonicalization.entities import key as entity_key
from services.canonicalization.normalize import jaccard, normalize_doi, normalize_title, parse_arxiv_id, title_tokens
from services.datasets.schemas import Experiment, FieldStatus, FieldValue
from services.evidence import grounding as G
from services.graph.knowledge_map import entity_links, results_of
from services.runtime import memo, reader, tools

VERSION = "runtime-papers-0.1.0"
MAX_REFS, MAX_DISCOVER, REF_CHARS, TOPIC_CHARS = 4, 3, 300, 200
TITLE_MATCH = 0.8
UNITS, PER_FIELD, CHARS, RESULT_UNITS, RESULT_CHARS = 8, 2, 1500, 6, 2400
TEXT_SECTIONS = re.compile(r"limitation|conclusion|discussion|future work", re.I)
DOI_RX = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]*[^\s\"'<>.,;)\]])", re.I)
S2_URL = re.compile(r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})", re.I)
ACL_URL = re.compile(r"aclanthology\.org/([\w.-]+?)(?:\.pdf)?/?$", re.I)
SPLIT_WORDS = {"val", "validation", "test", "dev", "set", "split"}
_cache = memo.TTLCache(256, 6 * 3600)  # (user, paper key, version) -> bundle: public paper text, kept per user like provider results
_mine: dict[str, dict[str, str]] = defaultdict(dict)  # user -> paper key -> cache key: the runtime papers a user has read (bounded)
MAX_PER_USER = 32


# ---------------------------------------------------------------------------------------------- references
def parse_reference(text: str) -> dict:
    """{kind: corpus | arxiv | doi | s2 | acl | url | title, value, raw}. Raises ValueError for an empty or overlong one."""
    t = " ".join((text or "").split())
    if not t:
        raise ValueError("An empty reference.")
    if len(t) > REF_CHARS:
        raise ValueError(f"A reference is a title, DOI, arXiv id or URL of at most {REF_CHARS} characters, not a document.")
    low = t.casefold()
    if re.fullmatch(r"paper:[\w.:/-]+", t):
        return {"kind": "corpus", "value": t, "raw": t}
    if "arxiv" in low or re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", low):
        if a := parse_arxiv_id(t):
            return {"kind": "arxiv", "value": a[0], "raw": t}
    if (m := DOI_RX.search(t)) and (low.startswith(("10.", "doi:")) or "doi.org/" in low):
        d = normalize_doi(m.group(1))
        if d.startswith("10.48550/arxiv.") and (a := parse_arxiv_id(d)):
            return {"kind": "arxiv", "value": a[0], "raw": t}
        return {"kind": "doi", "value": d, "raw": t}
    if m := S2_URL.search(t):
        return {"kind": "s2", "value": m.group(1), "raw": t}
    if (m := ACL_URL.search(t)) and "aclanthology.org" in low:
        return {"kind": "acl", "value": m.group(1), "raw": t}
    if re.match(r"https?://", low):
        return {"kind": "url", "value": t, "raw": t}
    return {"kind": "title", "value": t, "raw": t}


def split_references(text: str) -> list[str]:
    """Several references in one line: separated by new lines, semicolons, " vs ", " versus " or " compared with/to "."""
    parts = re.split(r"\s*(?:\n|;|\bvs\.?(?=\s)|\bversus\b|\bcompared (?:with|to)\b)\s*", text or "", flags=re.I)
    return [p.strip(" ,.\"'“”") for p in parts if p and p.strip(" ,.\"'“”")]


TOPIC = re.compile(r"\b(?:find|show|list|get)?\s*(?:the\s+)?(?:(?:most\s+)?recent|latest|new|some)?\s*(?:papers?|works?|studies|publications)\s+"
                   r"(?:on|about|for|using|that use|proposing)\s+(?P<topic>.+?)(?:\s*(?:,\s*)?and\s+compare\b.*)?[?.!]?$", re.I)
LEAD = re.compile(r"^\s*(?:please\s+)?(?:compare|diff|contrast)\s+(?:the\s+)?(?:(?:these|those|two|three|four|the following)\s+)?(?:papers?)?\s*[:,-]?\s*", re.I)
TAIL = re.compile(r"[,;]?\s*(?:including|even if|also|incl\.)\b[^\n]*\bcorpus\b.*$|\s*\b(?:side by side|in detail)\b.*$", re.I)


def papers_from_text(text: str) -> dict:
    """A chat request -> {"refs": [...]} or {"topic": "..."}: "Compare <A> vs <B>, including papers not in the corpus",
    "Find recent papers on <method> and compare their experimental settings", or one title / DOI / arXiv id."""
    t = " ".join((text or "").split())
    if m := TOPIC.search(t):
        topic = re.sub(r"\b(?:their|its)\s+(?:experimental\s+)?(?:settings|configurations|setups)\b.*$", "", m.group("topic"), flags=re.I).strip(" ,.?")
        if topic:
            return {"topic": topic}
    body = TAIL.sub("", LEAD.sub("", t)).strip(" .?")
    refs = split_references(body)
    if len(refs) == 1 and " and " in refs[0] and not re.search(r"arxiv|10\.\d|https?://", refs[0], re.I):
        both = [p.strip(" ,") for p in refs[0].split(" and ")]  # "Compare A and B": two titles when each side is a name of its own
        if len(both) == 2 and all(len(p) >= 3 for p in both):
            return {"refs": both, "split_on_and": True}
    return {"refs": refs}


def _title_index(index) -> dict[str, str]:
    if getattr(index, "_rt_titles", None) is None:
        titles = {}
        for pid, p in index.paper.items():
            titles.setdefault(normalize_title(p.get("title")), pid)
        index._rt_titles = titles
    return index._rt_titles


def corpus_match(index, ref: dict) -> tuple[str | None, str | None]:
    """(corpus paper id, how it matched) on identifiers; a typed title only when it equals a corpus title (normalised)."""
    if ref["kind"] == "corpus":
        return (ref["value"], "corpus id") if ref["value"] in index.paper else (None, None)
    if ref["kind"] == "arxiv":
        pid = f"paper:arxiv:{ref['value']}"
        if pid in index.paper:
            return pid, "arXiv id"
        rows = index.q("select id from rx.papers where arxiv_id = %s limit 1", (ref["value"],))
        return (rows[0][0], "arXiv id") if rows else (None, None)
    if ref["kind"] == "doi":
        rows = index.q("select id from rx.papers where lower(doi) = %s limit 1", (ref["value"],))
        return (rows[0][0], "DOI") if rows else (None, None)
    if ref["kind"] == "title":
        pid = _title_index(index).get(normalize_title(ref["value"]))
        return (pid, "exact title (as typed)") if pid else (None, None)
    return None, None


def _identifier_in_corpus(index, cand: dict) -> str | None:
    for kind, value in (("arxiv", cand.get("arxiv_id")), ("doi", normalize_doi(cand.get("doi")))):
        if value:
            pid, _ = corpus_match(index, {"kind": kind, "value": value})
            if pid:
                return pid
    return None


def _candidate(c: dict, score: float | None = None) -> dict:
    return ({k: c.get(k) for k in ("title", "year", "venue", "url", "doi", "arxiv_id", "provider", "authors")}
            | ({"title_match": round(score, 2)} if score is not None else {})
            | {"also_found_by": [a.get("provider") for a in c.get("also_found_by") or [] if isinstance(a, dict)]})


def _dedupe(found: list[dict]) -> list[dict]:
    """One candidate per work: DOI / arXiv id (the source gate already merged those), then a same normalised title from
    another provider is folded into the one with an arXiv id and recorded as also found (same title, no shared id)."""
    out: dict[str, dict] = {}
    for c in found:
        k = normalize_title(c.get("title"))
        if not k:
            continue
        if k in out:
            keep, other = (c, out[k]) if c.get("arxiv_id") and not out[k].get("arxiv_id") else (out[k], c)
            keep = {**keep, "also_found_by": list(keep.get("also_found_by") or []) + [
                {"provider": other.get("provider"), "url": other.get("url"), "same_title_only": True}]}
            out[k] = keep
        else:
            out[k] = c
    return list(out.values())


def _lookup(lookup, identifier: str) -> dict | None:
    if lookup is None:
        return None
    try:
        return lookup(identifier)
    except Exception:  # a failed lookup falls back to search or to the full text itself
        return None


def _search(search, tr, query: str) -> tuple[list[dict], str | None]:
    """Provider results for one query, or ([], why) when the provider phase itself failed: a failure costs only this
    reference (the provider phase already survives slow, failing and rate-limited providers one by one)."""
    try:
        return _dedupe(search(tr, query) or []), None
    except Exception as e:
        return [], f"the scholarly providers failed ({type(e).__name__})"


def resolve(index, refs: list[str], tr, search=None, lookup=None) -> list[dict]:
    """Each typed reference -> {input, kind: corpus | runtime | ambiguous | not_found | unsupported, paper_id (corpus),
    source (a provider record to read), matched_by, candidates, why}. `search(tr, query)` and `lookup(identifier)` are the
    provider calls; without them (no opt-in) only the corpus is consulted."""
    out, parsed = [], []
    with tr.stage("query_understanding", references=len(refs), external=search is not None) as st:
        for raw in refs[:MAX_REFS]:
            try:
                parsed.append(parse_reference(raw))
            except ValueError as e:
                out.append({"input": str(raw)[:80], "kind": "unsupported", "why": str(e)})
        st.detail.update(kinds=dict(Counter(p["kind"] for p in parsed)), dropped=max(0, len(refs) - MAX_REFS))
    todo = []
    for ref in parsed:
        pid, how = corpus_match(index, ref)
        thin = _thin(index, pid) if pid and (search or lookup) else None
        if thin:  # in the corpus without experiment records (abstract only): read at request time like a runtime paper
            out.append({"input": ref["raw"], "kind": "runtime", "paper_id": pid, "matched_by": f"{how}; in the corpus without experiment records, read now",
                        "source": thin})
        elif pid:
            out.append({"input": ref["raw"], "kind": "corpus", "paper_id": pid, "matched_by": how, "why": f"in the ResearchX corpus (matched by {how})"})
        elif search is None and lookup is None:
            out.append({"input": ref["raw"], "kind": "not_found", "why": "not in the corpus; turn on papers outside the corpus to look it up"})
        elif ref["kind"] == "url":
            out.append({"input": ref["raw"], "kind": "unsupported", "why": "only arXiv, DOI, Semantic Scholar and ACL Anthology URLs are resolved; give "
                        "the paper's title, DOI or arXiv id"})
        else:
            todo.append(ref)

    def one(ref: dict) -> dict:
        if ref["kind"] == "arxiv":
            meta = _lookup(lookup, f"arXiv:{ref['value']}") or {}
            return {"input": ref["raw"], "kind": "runtime", "matched_by": "arXiv id",
                    "source": {**meta, "arxiv_id": ref["value"], "url": f"https://arxiv.org/abs/{ref['value']}", "provider": meta.get("provider") or "arxiv"}}
        if ref["kind"] in ("doi", "s2", "acl"):
            meta = _lookup(lookup, {"doi": f"DOI:{ref['value']}", "s2": ref["value"], "acl": f"ACL:{ref['value']}"}[ref["kind"]])
            if meta:
                return {"input": ref["raw"], "kind": "runtime", "source": meta,
                        "matched_by": {"doi": "DOI", "s2": "Semantic Scholar id", "acl": "ACL Anthology id"}[ref["kind"]]}
            if ref["kind"] != "doi" or search is None:
                return {"input": ref["raw"], "kind": "not_found", "why": "the identifier lookup found no such work"}
        found, failed = _search(search, tr, ref["value"]) if search else ([], None)
        if failed:
            return {"input": ref["raw"], "kind": "not_found", "why": failed + "; try again, or give the DOI or arXiv id"}
        if ref["kind"] == "doi":
            hit = next((c for c in found if normalize_doi(c.get("doi")) == ref["value"]), None)
            return ({"input": ref["raw"], "kind": "runtime", "matched_by": "DOI", "source": hit} if hit else
                    {"input": ref["raw"], "kind": "not_found", "why": "no provider returned a work with this DOI"})
        want = title_tokens(ref["value"])
        scored = sorted(((jaccard(want, title_tokens(c.get("title"))), c) for c in found), key=lambda x: -x[0])
        if scored and scored[0][0] >= TITLE_MATCH and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            return {"input": ref["raw"], "kind": "runtime", "matched_by": f"title search (title match {scored[0][0]:.2f})", "source": scored[0][1],
                    "candidates": [_candidate(c, s) for s, c in scored[1:4]]}
        if scored:
            return {"input": ref["raw"], "kind": "ambiguous", "why": "no result's title matches the reference closely enough to choose it for you",
                    "candidates": [_candidate(c, s) for s, c in scored[:5]]}
        return {"input": ref["raw"], "kind": "not_found", "why": "the scholarly providers returned no work for this reference"}

    if todo:
        with ThreadPoolExecutor(min(len(todo), MAX_REFS)) as pool:
            done = list(pool.map(one, todo))
        for r in done:  # a found work that the corpus holds is the corpus paper (read now if the corpus has no experiment records for it)
            pid = _identifier_in_corpus(index, r["source"]) if r["kind"] == "runtime" else None
            if pid and _thin(index, pid):
                out.append({**r, "paper_id": pid, "source": {**r["source"], "corpus_paper": pid},
                            "matched_by": f"{r['matched_by']}; in the corpus without experiment records, read now"})
            elif pid:
                out.append({**r, "kind": "corpus", "paper_id": pid, "source": None, "matched_by": f"{r['matched_by']}; its identifier is in the corpus",
                            "why": "found by the providers and already in the ResearchX corpus"})
            else:
                out.append(r)
    return out


def _thin(index, pid: str) -> dict | None:
    """A corpus paper the corpus holds no experiment record for, as a source to read now (None when it has records or no
    arXiv id: there is then nothing to read)."""
    if index.paper_exps.get(pid):
        return None
    arxiv = pid.split("paper:arxiv:", 1)[1] if pid.startswith("paper:arxiv:") else (index.q("select arxiv_id from rx.papers where id = %s", (pid,)) or [[None]])[0][0]
    p = index.paper.get(pid, {})
    return {"arxiv_id": arxiv, "title": p.get("title"), "year": p.get("year"), "provider": "arxiv", "corpus_paper": pid} if arxiv else None


def discover(index, topic: str, tr, search=None, limit: int = MAX_DISCOVER, include_corpus: bool = True) -> dict:
    """Papers on a topic ("find recent papers on X and compare them"): corpus papers whose titles contain every topic term
    and, when opted in, the providers' results. The most recent are proposed first, outside the corpus only works with a
    readable arXiv full text; every other result stays a candidate the user can pick instead. `include_corpus=False`:
    only works outside the corpus are proposed (the corpus is already in the analysis)."""
    t = " ".join((topic or "").split())
    if not 3 <= len(t) <= TOPIC_CHARS:
        raise ValueError(f"Describe the topic in 3 to {TOPIC_CHARS} characters.")
    terms = sorted(w for w in title_tokens(t) if len(w) > 2)
    corpus = [(p.get("year") or 0, pid) for pid, p in index.paper.items()
              if include_corpus and terms and all(w in title_tokens(p.get("title")) for w in terms)]
    ext = []
    found, failed = _search(search, tr, t) if search else ([], None)
    for c in found:
        pid = _identifier_in_corpus(index, c)
        if pid and index.paper_exps.get(pid):  # in the corpus with experiment records
            if include_corpus:
                corpus.append((index.paper.get(pid, {}).get("year") or 0, pid))
        else:  # outside the corpus, or in it without experiment records: read now
            ext.append({**c, "corpus_paper": pid} if pid else c)
    ranked = list(dict.fromkeys(p for _, p in sorted(set(corpus), reverse=True)))
    ext.sort(key=lambda c: (not c.get("arxiv_id"), -(c.get("year") or 0)))
    picks = [{"kind": "corpus", "paper_id": p} for p in ranked[:limit]]
    picks += [{"kind": "runtime", "source": c} for c in [c for c in ext if c.get("arxiv_id")][: max(0, limit - len(picks))]]
    return {"topic": t, "terms": terms, "picked": picks, "provider_error": failed,
            "candidates": [{"paper_id": p, "title": index.paper.get(p, {}).get("title"), "year": index.paper.get(p, {}).get("year"), "in_corpus": True}
                           for p in ranked[:10]] + [{**_candidate(c), "in_corpus": False} for c in ext[:10]],
            "why": "most recent first; outside the corpus only works with an arXiv full text are read, the others are listed"}


# ---------------------------------------------------------------------------------------------- reading + extraction
def paper_key(src: dict) -> str:
    arxiv = (parse_arxiv_id(src.get("arxiv_id")) or [None])[0] if src.get("arxiv_id") else None
    doi = normalize_doi(src.get("doi"))
    return f"rt:paper:arxiv:{arxiv}" if arxiv else f"rt:paper:doi:{doi}" if doi else "rt:paper:web:" + hashlib.sha1(str(src.get("url")).encode()).hexdigest()[:16]


def _h(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _named(text: str | None, name: str | None) -> bool:
    c = G.canon(name)
    return bool(c) and re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", G.canon(text)) is not None


def _fv(raw, value, ids: list[str]) -> dict:
    return {"status": "KNOWN", "raw_value": str(raw), "value": value if value not in (None, "") else raw, "evidence_ids": list(ids)}


def load(index, src: dict, user_key: str, read, extract, extract_results) -> dict:
    """Read one runtime paper and extract its validated records (cached per user). `read(arxiv_id)` -> reader document;
    `extract(req, fields, sources)` -> validated facts (orchestrator._extract + _check_fact); `extract_results(sources)` ->
    (rows, report)."""
    key = paper_key(src)
    ck = memo.key(user_key, key, VERSION)
    if (hit := _cache.get(ck)) is not None:
        _remember(user_key, key, ck)
        index.runtime_evidence.update(hit["evidence"])
        return {**hit, "cached": True}
    t0 = time.perf_counter()
    arxiv = (parse_arxiv_id(src.get("arxiv_id")) or [None])[0] if src.get("arxiv_id") else None
    meta = {"key": key, "title": src.get("title"), "year": src.get("year") or (2000 + int(arxiv[:2]) if arxiv and arxiv[:2].isdigit() else None),
            "venue": src.get("venue"), "authors": src.get("authors") or [], "doi": normalize_doi(src.get("doi")), "arxiv_id": arxiv,
            "url": src.get("url") or (f"https://arxiv.org/abs/{arxiv}" if arxiv else None), "provider": src.get("provider"),
            "also_found_by": [a.get("provider") for a in src.get("also_found_by") or [] if isinstance(a, dict)], "abstract": src.get("text") or None,
            "retrieved_at": src.get("retrieved_at"), "content_sha256": src.get("content_sha256"), "license": None, "redistribution_allowed": False,
            "corpus_paper": src.get("corpus_paper")}
    units, access, error = [], "abstract only", None
    if arxiv:
        try:
            doc = read(arxiv)
            units = [{**u, "chunk_id": f"rt:{arxiv}#{u['unit_id']}", "url": f"https://arxiv.org/html/{arxiv}#{u['unit_id']}"} for u in doc["units"]]
            meta.update(title=meta["title"] or doc.get("title"), abstract=meta["abstract"] or doc.get("abstract"), license=doc["license"],
                        redistribution_allowed=doc["redistribution_allowed"], retrieved_at=doc["retrieved_at"], content_sha256=doc["sha256"])
            access = "full text (arXiv HTML)"
        except Exception as e:  # an unreadable full text leaves the abstract
            error = f"full text not readable: {type(e).__name__}: {str(e)[:120]}"
    if not units and meta["abstract"]:
        units = [{"unit_type": "abstract", "unit_id": None, "section": "Abstract", "text": meta["abstract"], "table_type": None,
                  "chunk_id": f"{key}#abstract", "url": meta["url"]}]
    for u in units:
        u["redistribution_allowed"] = bool(meta["redistribution_allowed"]) and u["unit_type"] != "abstract"
        u["context"] = "\n".join(x for x in (meta["title"], u.get("section"), u["text"]) if x)
    bundle = extract_records(meta, units, extract, extract_results)
    bundle.update(access=access, error=error, latency_ms=round((time.perf_counter() - t0) * 1000), version=VERSION,
                  passages=[{"chunk_id": u["chunk_id"], "section": u["section"], "text": u["text"][:900] if u["redistribution_allowed"] else None,
                             "text_withheld": not u["redistribution_allowed"], "url": u["url"]}
                            for u in units if u["unit_type"] == "paragraph" and TEXT_SECTIONS.search(u.get("section") or "")][:3])
    _cache.put(ck, bundle)
    _remember(user_key, key, ck)
    index.runtime_evidence.update(bundle["evidence"])
    return bundle


def _remember(user_key: str, key: str, ck: str) -> None:
    mine = _mine[user_key]
    mine.pop(key, None)
    mine[key] = ck
    while len(mine) > MAX_PER_USER:
        mine.pop(next(iter(mine)))


def user_overlay(index, user_key: str, keys=None) -> "Overlay":
    """The runtime papers this user has read (still cached), over the corpus index: lets a runtime experiment shown in an
    earlier result (e.g. a Benchmark Reality Explorer row) be investigated or turned into a recipe by its id."""
    bundles = [b for k, ck in list(_mine.get(user_key, {}).items()) if (keys is None or k in keys) and (b := _cache.get(ck)) is not None]
    for b in bundles:
        index.runtime_evidence.update(b["evidence"])
    return Overlay(index, bundles)


def _row_problem(r: dict, u: dict | None) -> str | None:
    if u is None:
        return "source_not_found"
    if tools.INJECTION.search(u["text"] or ""):
        return "instruction_like_source"
    if not (_named(u["text"], r["dataset"]) or _named(u.get("section"), r["dataset"])):
        return "dataset_not_named"
    return None


def _row_split(r: dict, u: dict) -> str | None:
    """The split of a result row: written in the row itself ("Vega-2 (val) | 85.0") while the passage gives it split context,
    else the only evaluation split the passage names; otherwise not stated."""
    if not r.get("split"):
        return None
    want = G.canonical_split(r["split"])
    in_row = {G.canonical_split(m.group(1)) for m in G.SPLIT.finditer(G.canon(r["span"]))} - G.TRAIN_SPLITS - {"train"}
    if want and in_row == {want} and (G.SPLIT_NOUN.search(G.canon(u["text"])) or G.UNAMBIGUOUS_SPLIT.search(want)):
        return want
    ok, _, split = G.validate("evaluation.dataset_split", r["split"], u["text"], u["text"])
    named = {G.canonical_split(m.group(1)) for m in G.SPLIT.finditer(G.canon(u["text"]))} - G.TRAIN_SPLITS - {"train"}
    return split if ok and named <= {split} else None


def extract_records(meta: dict, units: list[dict], extract, extract_results) -> dict:
    """Settings, conditions and result rows of one paper, each validated; runtime Experiment records with their evidence."""
    result_units = reader.relevant(units, ["evaluation.score"], G.Request(fields=["evaluation.score"]), RESULT_UNITS, RESULT_CHARS)
    result_units = [{**u, "context": "\n".join(x for x in (meta["title"], u.get("section"), u["text"]) if x)} for u in result_units]
    calls = []
    for batch in BATCHES:
        picked, want = {}, []
        for f in batch:
            best = reader.relevant(units, [f], G.Request(fields=[f]), PER_FIELD, CHARS)
            if best:
                want.append(f)
                for u in best:
                    picked.setdefault(u["chunk_id"], {**u, "context": "\n".join(x for x in (meta["title"], u.get("section"), u["text"]) if x)})
        if want:
            calls.append((want, list(picked.values())[:UNITS]))
    with ThreadPoolExecutor(1 + len(calls)) as pool:
        res_future = pool.submit(extract_results, result_units) if result_units else None
        fact_futures = [pool.submit(extract, G.Request(fields=w), w, ss) for w, ss in calls]
        rows, rep = res_future.result() if res_future else ([], {"status": "EMPTY", "errors": [], "dropped": 0})
        facts = [f for fu in fact_futures for f in fu.result()]
    by_id = {u["chunk_id"]: u for u in units}
    by_id.update({u["chunk_id"]: u for _, ss in calls for u in ss})
    by_id.update({u["chunk_id"]: u for u in result_units})
    kept, rejected = [], Counter()
    for r in rows:
        u = by_id.get(r["chunk_id"])
        if why := _row_problem(r, u):
            rejected[why] += 1
            continue
        ok, reason, value = G.validate_score(r["value"], r["span"], u["text"], r["metric"], r["model"], r["dataset"])
        if not ok:
            rejected[reason] += 1
            continue
        line = G.row_of(u["text"], value, r["model"])
        kept.append({**r, "value_n": value, "span": line[1] if line else r["span"], "split_v": _row_split(r, u), "unit": u})
    datasets = list(dict.fromkeys(r["dataset"] for r in kept))
    evidence: dict[str, dict] = {}

    def ev(field, raw, value, u, span, xid, stage=None, basis=None, applies_to=None) -> str:
        rid = "rt:" + _h(meta["key"], u["chunk_id"], field, raw, xid or "")
        evidence[rid] = {"id": rid, "field": field, "status": "RUNTIME_EXTRACTED", "location_kind": "PAPER_" + str(u["unit_type"]).upper(),
                         "paper_id": meta["key"], "experiment_id": xid, "text": span if u["redistribution_allowed"] else None,
                         "text_withheld": not u["redistribution_allowed"], "extracted_value": raw, "normalized_value": value,
                         "source_location": u.get("url"), "section": u.get("section"), "table": u["unit_id"] if u["unit_type"] == "table" else None,
                         "cell": None, "extraction_model": "runtime extraction at request time", "stage": stage, "stage_basis": basis,
                         "applies_to": applies_to, "origin": "runtime"}
        return rid

    groups: dict[tuple, list] = defaultdict(list)
    for r in kept:  # configuration-level identity: one evaluated configuration per (model, dataset, split)
        groups[(entity_key(r["model"]), entity_key(r["dataset"]), r["split_v"] or "", "")].append(r)
    for k in [k for k, rs in groups.items() if max(Counter(G.canon(r["metric"]) for r in rs).values()) > 1]:
        for r in groups.pop(k):  # one metric twice: two rows (variants, settings) the extraction could not tell apart; never one record
            groups[k[:3] + (G.canon(r["span"]),)].append(r)
    experiments: dict[str, Experiment] = {}
    variant: dict[str, str] = {}  # experiment -> the table row that tells it apart from a namesake
    for (mk, dk, split, row), rs in groups.items():
        first = rs[0]
        xid = "rt:exp:" + _h(meta["key"], mk, dk, split, row)
        if row:
            variant[xid] = G.display(first["span"])[:60]
        outcomes, ids = [], []
        for r in rs:
            rid = ev("evaluation.score", G.display(r["value"]), r["value_n"], r["unit"], r["span"], xid)
            ids.append(rid)
            outcomes.append({"metric": _fv(r["metric"], r["metric"], [rid]), "score": _fv(G.display(r["value"]), r["value_n"], [rid]),
                             **({"dataset_split": _fv(r["split"], split, [rid])} if split else {})})
        rec = {"experiment_id": xid, "study_id": f"runtime:{meta['key']}", "paper_id": meta["key"], "source_record_ids": [meta["url"] or meta["key"]],
               "experiment_type": "EVALUATION", "extraction_method": "LLM", "extraction_model": "runtime extraction at request time",
               "extraction_run_id": VERSION, "quality_tier": "BRONZE", "outcomes": outcomes,
               "model": {"model_name": _fv(first["model"], first["model"], ids[:1])},
               "evaluation": {"benchmark": _fv(first["dataset"], first["dataset"], ids[:1]), **({"dataset_split": _fv(first["split"], split, ids[:1])} if split else {})},
               "provenance": {"source_name": "RESEARCHX", "source_url": meta["url"], "retrieved_at": meta["retrieved_at"],
                              "content_hash": meta["content_sha256"], "processing_version": VERSION}}
        experiments[xid] = Experiment.model_validate(rec)
    statements, settings = [], Counter()
    for f in facts:  # settings and conditions, each with its passage's dataset and stage
        u = by_id.get(f.chunk_id)
        if u is None:
            rejected["setting_source_not_found"] += 1
            continue
        ok, reason, value = G.validate(f.field, f.value if f.value is not None else f.raw, f.span, u["text"])  # whatever `extract` checked
        if not ok or tools.INJECTION.search(u["text"] or ""):
            rejected[f"setting_{reason if not ok else 'instruction_like_source'}"] += 1
            continue
        f.value = value
        named = [d for d in datasets if _named(u["text"], d)]
        applies = named[0] if len(named) == 1 else None
        if f.field in CONDITIONS[:3]:
            stage, basis = "evaluation", "an evaluation condition"
        else:
            stage, basis = stage_of(u["text"], f.raw, u.get("section"))
            stage = stage or ("evaluation" if f.field == "dataset.dataset_version" else "unspecified")
        if f.field in CONDITIONS and not applies and len(datasets) > 1:
            targets = []  # a condition that names none of several datasets stays a paper-level statement
        else:
            targets = [x for x, m in experiments.items() if not applies or _named(m.evaluation.benchmark.raw_value, applies)]
        settings[f.field] += 1
        statements.append({"field": f.field, "raw": f.raw, "value": f.value, "stage": stage, "basis": basis, "applies_to": applies, "unit": u,
                           "span": f.span, "targets": targets})
    by_field: dict[tuple, list] = defaultdict(list)
    for s in statements:
        by_field[(s["field"], s["stage"], s["applies_to"] or "")].append(s)
    for (field, stage, _a), ss in by_field.items():
        distinct = {G.canon(str(s["value"])) for s in ss}
        for s in ss:
            rid = ev(field, s["raw"], s["value"], s["unit"], s["span"], None, stage, s["basis"], s["applies_to"])
            if stage == "pretraining" or len(distinct) > 1:
                continue  # a pre-training value, or several values for one stage: paper-level statements the configuration view shows
            group, name = field.split(".", 1)
            for xid in s["targets"]:
                fv = getattr(getattr(experiments[xid], group), name)
                if fv.status is FieldStatus.UNKNOWN:
                    setattr(getattr(experiments[xid], group), name, FieldValue(
                        status=FieldStatus.KNOWN, raw_value=G.display(s["raw"]), value=s["value"], evidence_ids=[rid], derived=True,
                        derivation=f"found in the paper's full text at request time ({s['unit'].get('section') or s['unit']['unit_type']}; "
                                   f"{'stated for ' + s['applies_to'] if s['applies_to'] else 'the passage names no dataset'}; stage: {stage}); validated quote"))
    entries = {}
    for xid, m in experiments.items():
        rec = m.model_dump(mode="json")
        label = " · ".join(x for x in (m.model.model_name.raw_value, m.evaluation.benchmark.raw_value, m.evaluation.dataset_split.raw_value,
                                       f"row “{variant[xid]}”" if xid in variant else None) if x)
        entries[xid] = {"paper_id": meta["key"], "depth": None, "links": entity_links(rec), "results": results_of(xid, rec), "label": label[:100],
                        "origin": "runtime"}
    return {**meta, "experiments": experiments, "entries": entries, "evidence": evidence, "status": "completed",
            "extraction": {"result_rows": len(rows), "results_kept": len(kept), "results_rejected": dict(rejected), "settings_found": dict(settings),
                           "experiments": len(experiments), "datasets": datasets, "result_report": {k: rep.get(k) for k in ("status", "errors", "model")},
                           "units_read": len({u["chunk_id"] for _, ss in calls for u in ss} | {u["chunk_id"] for u in result_units})}}


def load_many(index, sources: list[dict], user_key: str, read, extract, extract_results, tr) -> list[dict]:
    """Read several runtime papers in parallel under one trace stage (experiment_extraction); failures cost only their paper."""
    if not sources:
        return []
    with tr.stage("experiment_extraction", method="runtime papers: full text (else abstract) -> schema-bound extraction -> deterministic validation",
                  papers=len(sources)) as st:
        with ThreadPoolExecutor(min(len(sources), MAX_REFS)) as pool:
            futures = [pool.submit(_safe_load, index, s, user_key, read, extract, extract_results) for s in sources]
        out = [f.result() for f in futures]
        st.detail.update(read=[{"paper": p.get("key"), "title": (p.get("title") or "")[:100], "access": p.get("access"), "status": p.get("status"),
                                "experiments": len(p.get("entries") or {}), "latency_ms": p.get("latency_ms"), "cached": bool(p.get("cached")),
                                "error": p.get("error"), **{k: v for k, v in (p.get("extraction") or {}).items() if k != "result_report"}} for p in out])
    return out


def _safe_load(index, src, user_key, read, extract, extract_results) -> dict:
    try:
        return load(index, src, user_key, read, extract, extract_results)
    except Exception as e:  # one unreadable paper costs only that paper
        return {"key": paper_key(src), "title": src.get("title"), "status": "failed", "error": f"{type(e).__name__}: {str(e)[:160]}"}


# ---------------------------------------------------------------------------------------------- analyses over corpus + runtime
class Overlay:
    """The corpus GraphIndex plus this request's runtime papers: the same analysis code reads both. Corpus lookups go to
    the index; runtime papers and experiments come from their bundles (marked `origin: runtime`). `align`: a requested
    benchmark and metric (id, name) that runtime names join when they name the same thing (see _aligned_*)."""

    def __init__(self, index, papers: list[dict], align: dict | None = None):
        self._index = index
        papers = [p for p in papers if p.get("status") == "completed"]
        self.runtime = {p["key"]: p for p in papers}
        self._models = {x: m for p in papers for x, m in p["experiments"].items()}
        entries = {x: dict(e) for p in papers for x, e in p["entries"].items()}
        extra_ent: dict[str, dict] = {}
        ent_exps: dict[str, list] = {}
        for x, e in entries.items():
            e["links"] = list(e["links"]) + _aligned_links(e, align)
            e["results"] = [_aligned_result(r, align) for r in e["results"]]
            for link in e["links"]:
                if link["entity_id"] not in index.ent:
                    extra_ent[link["entity_id"]] = {"type": link["entity_type"], "name": link["raw"]}
                bucket = ent_exps.setdefault(link["entity_id"], list(index.ent_exps.get(link["entity_id"], ())))
                if x not in bucket:
                    bucket.append(x)
            for r in e["results"]:
                if r["metric_id"] and r["metric_id"] not in index.ent:
                    extra_ent[r["metric_id"]] = {"type": "METRIC", "name": r["metric"]}
        self.exp = ChainMap(entries, index.exp)
        self.paper = ChainMap({p["key"]: {"title": p["title"], "year": p["year"], "study_id": None} for p in papers}, index.paper)
        self.paper_exps = ChainMap({p["key"]: list(p["entries"]) for p in papers}, index.paper_exps)
        self.ent = ChainMap(extra_ent, index.ent)
        self.ent_exps = ChainMap(ent_exps, index.ent_exps)
        self.runtime_passages = {p["key"]: p.get("passages") or [] for p in papers}

    def __getattr__(self, name):
        return getattr(self._index, name)

    def experiments(self, ids) -> dict:
        ids = list(dict.fromkeys(ids))
        corpus = [i for i in ids if i not in self._models]
        out = self._index.experiments(corpus) if corpus else {}
        out.update({i: self._models[i].model_copy(deep=True) for i in ids if i in self._models})
        return out

    def node(self, nid: str) -> dict | None:
        if nid in self.runtime:
            p = self.runtime[nid]
            return {"id": nid, "type": "Paper", "label": p["title"] or nid, "flags": ["runtime"],
                    "meta": {"year": p["year"], "experiments": len(p["entries"]), "origin": "runtime"}}
        if nid in self._models:
            e = self.exp[nid]
            return {"id": nid, "type": "Experiment", "label": e["label"], "flags": ["runtime"],
                    "meta": {"paper": self.runtime[e["paper_id"]]["title"], "results": len(e["results"]), "origin": "runtime"}}
        return self._index.node(nid)

    def node_detail(self, nid: str) -> dict:
        if nid in self.runtime:
            p = self.runtime[nid]
            return {"node": self.node(nid), "title": p["title"], "year": p["year"], "venue": p["venue"], "url": p["url"], "authors": p["authors"],
                    "abstract": p["abstract"], "contradictions": [], "origin": "runtime", "provenance": provenance(p)}
        return self._index.node_detail(nid)


def provenance(p: dict) -> dict:
    """What the UI shows for a runtime paper: where it came from, what could be read, what was kept."""
    return {k: p.get(k) for k in ("key", "provider", "also_found_by", "url", "doi", "arxiv_id", "retrieved_at", "content_sha256", "license",
                                  "redistribution_allowed", "access", "error", "latency_ms", "cached", "corpus_paper")} | {
        "extraction": {k: v for k, v in (p.get("extraction") or {}).items() if k != "result_report"},
        "status": ("runtime: in the corpus without experiment records; read at request time, unreviewed" if p.get("corpus_paper")
                   else "runtime: read at request time, unreviewed, not in the ResearchX corpus")}


def _aligned_links(e: dict, align: dict | None) -> list[dict]:
    """A runtime dataset name that is the requested benchmark up to a split word ("ImageNet val" for ImageNet) joins it."""
    if not align or not align.get("benchmark"):
        return []
    bid, name = align["benchmark"]
    return [{**link, "entity_id": bid, "aligned_from": link["raw"]} for link in e["links"]
            if link["edge"] == "USES_DATASET" and link["entity_id"] != bid and G.tokens(link["raw"]) - SPLIT_WORDS == G.tokens(name)]


def _aligned_result(r: dict, align: dict | None) -> dict:
    """A runtime metric written differently from the requested one joins it when one names a subset of the other's words
    (G.metric_ok: "accuracy" for "top-1 accuracy", never top-5 for top-1); the paper's own wording is kept beside it."""
    if not align or not align.get("metric") or not r.get("metric"):
        return r
    mid, name = align["metric"]
    if r["metric_id"] != mid and G.tokens(r["metric"]) and G.metric_ok(r["metric"], name):
        return {**r, "metric_id": mid, "metric_as_written": r["metric"]}
    return r


def summary(papers: list[dict], resolutions: list[dict] | None = None, discovery: dict | None = None) -> dict:
    return {"papers": [provenance(p) | {"title": p.get("title"), "experiments": len(p.get("entries") or {}), "status": p.get("status")} for p in papers],
            "resolutions": resolutions or [], "discovery": discovery,
            "notes": ["Runtime papers are read at request time and are not part of the ResearchX corpus; nothing is promoted automatically.",
                      "Their values are machine extractions from the accessible text, each kept only after deterministic validation; a setting not "
                      "found is not stated in the accessible text (an abstract-only paper states little)."]}
