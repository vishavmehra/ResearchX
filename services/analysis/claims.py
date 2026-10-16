"""Claim-to-Evidence Auditor: what do the corpus records say about a stated claim?

The claim is read deterministically into one of three forms the records can test:
- comparison: "<A> outperforms <B> [on <dataset>] [in <metric>]" (also "is worse than", which swaps the sides)
- reported value: "<A> achieves <number> [<metric>] on <dataset>"
- setting: "<A> was trained with <setting> <value> [on <dataset>]"
Evidence is the corpus records, never another model's opinion: the experiments whose model or method names contain every
token of the claim's model (services.evidence.grounding.coverage; another size or variant never matches). A record whose
names add only size/variant markers is direct evidence; one that adds other components (a modified model) is indirect.

A result pair counts for or against a comparison only on the same benchmark entity and the same metric, and only when it
is a same-paper comparison without a stated difference in conditions or the benchmark comparability engine finds the
conditions COMPARABLE (services.analysis.benchmark); every other pair is inconclusive with the reason. Whether higher or
lower is better comes from the metric's name; when the name does not say, the pair is inconclusive. Different metrics
are never converted into each other. The verdict comes from direct evidence when there is any; a general claim
("across benchmarks", no dataset) is checked for evidence that covers too few benchmarks or papers.
"""

from __future__ import annotations

import re
from collections import defaultdict

from services.analysis import benchmark as B
from services.analysis import configs as C
from services.analysis import facts as F
from services.analysis.papers import paper_level
from services.canonicalization.normalize import canonical_label
from services.evidence import grounding as G
from services.graph.knowledge_map import ENTITY_NODE

VERSION = "claims-0.1.0"
GUIDE = ("Write the claim in a form the records can test: '<model> outperforms <model> on <dataset> [in <metric>]', "
         "'<model> achieves <value> <metric> on <dataset>' or '<model> was trained with <setting> <value> [on <dataset>]'.")
BETTER = r"outperforms?|beats?|surpass(?:es)?|exceeds?|is better than|performs better than|does better than|is superior to|improves (?:on|over|upon)"
WORSE = r"underperforms?|is worse than|performs worse than|trails|lags behind|falls short of|is inferior to"
COMPARE = re.compile(rf"^(?P<a>.+?)\s+(?:\w+ly\s+)?(?P<verb>{BETTER}|{WORSE})\s+(?P<rest>.+)$", re.I)
VALUE = re.compile(r"^(?P<a>.+?)\s+(?:achieves?|reaches?|gets?|obtains?|scores?|attains?|reports?|yields?|has)\s+(?:an?\s+)?"
                   r"(?:(?P<m1>[^\d]+?)\s+(?:of|=)\s+)?(?P<v>\d+(?:\.\d+)?)\s*(?P<pct>%|percent)?\s*(?P<m2>[^\d]*?)\s+(?:on|in|for)\s+(?P<d>.+)$", re.I)
SETTING_VERB = re.compile(r"\s+(?:was|were|is|are)\s+(?:trained|fine-?tuned|pre-?trained|optimi[sz]ed|run)\b|\s+(?:uses|used|trains|employs|employed)\b", re.I)
CUT = re.compile(r"\s+(?:on|in|at|by|with|across|for|when|under|using|over|in terms of)\s+", re.I)
GENERAL = re.compile(r"\b(?:all|every|any|always|consistently|across|in general|generally|universally|regardless)\b", re.I)
VAGUE = re.compile(r"\b(?:all|other|others|existing|previous|prior|baselines?|sota|state[- ]of[- ]the[- ]art|competitors?)\b", re.I)
SPLIT = re.compile(r"\b(validation|val|test|dev)\b", re.I)
LOWER_BETTER = re.compile(r"\b(?:error|err|perplexity|ppl|fid|kid|wer|cer|ter|loss|mse|mae|rmse|nll|bpc|bpd|latency|eer|lpips|miss rate)\b", re.I)
HIGHER_BETTER = re.compile(r"\b(?:acc|accuracy|top-?\d|m?iou|m?ap(?:@[\d.:]+)?|ap\d*|f1|f-?score|bleu(?:-\d)?|rouge(?:-\w+)?|meteor|cider|spice|auc|auroc|"
                           r"em|exact match|pass@\d+|recall|precision|psnr|ssim|ndcg(?:@\d+)?|mrr|hit rate|win rate|success rate|pq|dice|kappa|"
                           r"correlation|spearman|pearson)\b", re.I)
SETTING_VALUE = r"\s*(?:of|=|as|is|to|:|at)?\s*(?:about\s+|around\s+)?(?P<v>\d+(?:\.\d+)?(?:e-?\d+)?\s*[kKmM]?)\b"
MAX_ITEMS = 30


def direction(metric: str | None) -> str | None:
    """'higher' or 'lower' when the metric's name says which is better ('top-1 error' is lower), else None."""
    if not metric:
        return None
    if LOWER_BETTER.search(metric):
        return "lower"
    return "higher" if HIGHER_BETTER.search(metric) else None


def _key(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").casefold())


def _clean(s: str) -> str:
    s = re.sub(r"^(?:the|a|an)\s+", "", s.strip(" ,;:"), flags=re.I)
    return re.sub(r"'s$", "", s).strip(" ,;:")


GENERIC_NAMES = G.GENERIC_DATASETS | {"model", "models", "method", "methods", "transformer", "transformers", "attention", "baseline", "baselines",
                                      "network", "networks", "ours", "base", "large", "small", "classification", "detection", "segmentation",
                                      "arxiv", "github", "openreview", "paper", "papers", "table", "figure", "appendix", "section"}
NAMED = re.compile(r"[a-z][A-Z]|[A-Z]{2}|\d")  # BEiT, OCR, ADE20K: a name, not a word of the sentence


def names_in(index, text: str, node_types=("Dataset",), min_experiments: int = 1) -> list[tuple[str, str]]:
    """(entity id, name) of corpus entities of these node types, linked to at least `min_experiments` experiment records,
    whose name appears in `text` (longest first, no name inside a longer one); punctuation and spacing variants match
    ("ADE-20K" ~ "ADE20K"). Generic words the entity table also lists ("Dataset", "Transformer") never count."""
    tk, found = _key(text), []
    for i, e in index.ent.items():
        name = (e["name"] or "").strip()
        if (ENTITY_NODE.get(e["type"]) not in node_types or len(index.ent_exps.get(i, ())) < min_experiments or len(_key(name)) < 3
                or not re.search(r"[A-Za-z]", name) or G.ekey(name) in GENERIC_NAMES or name.casefold() in GENERIC_NAMES or _key(name) not in tk):
            continue
        plain = re.fullmatch(r"[A-Za-z]+", name) and not NAMED.search(name)  # "Aggregation": counts only as written ("context aggregation" no)
        if re.search(rf"(?<!\w){G._loose(name)}(?!\w)", text, 0 if plain else re.I):
            found.append((i, name))
    found.sort(key=lambda x: len(x[1]), reverse=True)
    return [(i, n) for k, (i, n) in enumerate(found) if not any(_key(n) in _key(m) for _, m in found[:k])]


def datasets_in(index, text: str) -> list[str]:
    return [n for _, n in names_in(index, text, ("Dataset",))]


def _dataset(index, segment: str) -> str | None:
    if names := datasets_in(index, segment):
        return names[0]
    m = re.search(r"\b(?:on|in)\s+(?:the\s+)?(?P<d>[A-Z0-9][\w.+/-]*(?:\s+[A-Z0-9][\w.+/-]*)*)", segment)
    return m.group("d") if m and not G.NOT_A_DATASET.search(m.group("d")) else None


def parse(index, claim: str) -> dict:
    """The claim's form and parts; ValueError (with the accepted forms) when it is none of them."""
    text = " ".join((claim or "").split()).strip().rstrip(".")
    if len(text) < 8:
        raise ValueError(GUIDE)
    general = bool(GENERAL.search(text))
    if m := COMPARE.match(text):
        a, rest = _clean(m.group("a")), m.group("rest")
        head = CUT.split(rest, maxsplit=1)[0]
        b = _clean(head)
        if re.fullmatch(WORSE, m.group("verb"), re.I):
            a, b = b, a
        if not G.tokens(a) or not G.tokens(b) or VAGUE.search(b):
            raise ValueError("Name the model or method the claim compares against (not 'all baselines' or 'prior work'). " + GUIDE)
        tail = rest[len(head):]
        metric = G.METRIC.search(tail)
        dataset = _dataset(index, tail)
        return {"form": "comparison", "a": a, "b": b, "dataset": dataset, "metric": metric.group(1) if metric else None,
                "split": (s.group(1).lower() if (s := SPLIT.search(tail)) else None), "general": general or not dataset}
    if m := VALUE.match(text):
        metric = (m.group("m1") or m.group("m2") or "").strip(" ,") or None
        if metric and (hit := G.METRIC.search(metric)):
            metric = hit.group(1)
        return {"form": "value", "a": _clean(m.group("a")), "value": m.group("v"), "percent": bool(m.group("pct")), "metric": metric,
                "dataset": _dataset(index, m.group("d")) or _clean(m.group("d")), "split": (s.group(1).lower() if (s := SPLIT.search(m.group("d"))) else None),
                "general": general}
    if v := SETTING_VERB.search(text):
        settings = []
        for pat, path in G.QUERY_FIELDS:
            if hit := re.search(rf"(?:{pat}){SETTING_VALUE}", text[v.end():], re.I):
                settings.append({"field": path, "label": G.FIELDS[path][0], "value": hit.group("v").replace(" ", "")})
        for path, pat in G.CATEGORICAL.items():
            if path not in {s["field"] for s in settings} and (hit := re.search(pat, text[v.end():], re.I)):
                settings.append({"field": path, "label": G.FIELDS[path][0], "value": hit.group(0)})
        a = _clean(text[: v.start()])
        if settings and G.tokens(a):
            return {"form": "setting", "a": a, "settings": settings, "dataset": _dataset(index, text[v.end():]), "general": general}
    raise ValueError(GUIDE)


# ------------------------------------------------------------------------------------------------ record matching
def _names(index, xid: str) -> list[str]:
    return [link["raw"] for link in index.exp[xid]["links"] if link["node_type"] in ("Model", "Method")]


def _name_tokens(index, xid: str) -> set[str]:
    cache = index.__dict__.setdefault("_claim_name_tokens", {})  # per process, like the benchmark catalog
    if xid not in cache:
        cache[xid] = set().union(*(G.tokens(n) for n in _names(index, xid)))
    return cache[xid]


def match(mention: str, names: list[str]) -> str | None:
    """'direct': every token of the mention is in the names and they add only size/variant markers; 'variant': they add
    other components (a modified model); None: not this model (missing tokens, or another size/variant)."""
    if not names or (G.coverage(mention, names, {}) or 0) < 0.999 or G.variant_conflict(mention, names):
        return None
    extra = set().union(*(G.tokens(n) for n in names)) - G.tokens(mention)
    return "direct" if all(G.VARIANT.fullmatch(t) for t in extra) else "variant"


def matched(index, mentions: list[str]) -> tuple[list[dict[str, str]], int]:
    """Experiments per mention ({experiment id: 'direct' | 'variant'}); a record fitting two mentions goes to the more
    specific one (its tokens include the other's), and is skipped as ambiguous otherwise."""
    want = [G.tokens(m) for m in mentions]
    out: list[dict[str, str]] = [{} for _ in mentions]
    ambiguous = 0
    for xid in index.exp:
        have = _name_tokens(index, xid)
        hits = [(k, kind) for k, m in enumerate(mentions) if want[k] & have and (kind := match(m, _names(index, xid)))]
        if len(hits) > 1:
            best = [h for h in hits if all(want[h[0]] >= want[o[0]] for o in hits)]
            if len(best) != 1:
                ambiguous += 1
                continue
            hits = best
        for k, kind in hits:
            out[k][xid] = kind
    return out, ambiguous


def _bench(index, xid: str) -> dict | None:
    links = [link for link in index.exp[xid]["links"] if link["edge"] == "USES_DATASET"]
    return next((link for link in links if link["field"] == "evaluation.benchmark"), links[0] if links else None)


def _on_dataset(index, xid: str, dataset: str | None) -> tuple[bool, bool]:
    """(on the dataset or a variant of it, on exactly that dataset); a split word in the recorded name ("ADE20K val") is
    not another dataset. No dataset asked: every record qualifies."""
    if not dataset:
        return True, True
    names = [link["raw"] for link in index.exp[xid]["links"] if link["edge"] == "USES_DATASET"]
    want = G.tokens(dataset)
    exact = any({t for t in G.tokens(n) if not SPLIT.fullmatch(t)} == want for n in names)
    return G.dataset_match(dataset, names) == 2, exact


def _pair_reason(ca: dict, cb: dict, cls: str, why: str) -> str:
    """Why a pair of results cannot be compared, in terms of the two results (not of a reference)."""
    if cls == "INCOMPATIBLE":
        return why
    unstated = [B.CONDITION_LABEL[c] for c in B.KEY_CONDITIONS if not (B._key(c, ca[c]) and B._key(c, cb[c]))]
    both = [B.CONDITION_LABEL[c] for c in B.KEY_CONDITIONS if not B._key(c, ca[c]) and not B._key(c, cb[c])]
    if both and len(both) == len(unstated):
        return f"Neither result states its {' or '.join(both)}, so the setups cannot be checked."
    return f"The {' and '.join(unstated)} {'is' if len(unstated) == 1 else 'are'} not stated for both results, so the setups cannot be checked."


def _metric_ok(asked: str | None, metric: str | None) -> bool:
    return bool(metric) and (not asked or G.tokens(asked) <= G.tokens(metric))


def _side(index, xid: str, r: dict, kind: str) -> dict:
    pid = index.exp[xid]["paper_id"]
    return {"experiment": xid, "label": index.exp[xid]["label"], "paper": pid, "title": index.paper.get(pid, {}).get("title"),
            "year": index.paper.get(pid, {}).get("year"), "model": (_names(index, xid) or [None])[0], "match": kind,
            "score": r["score"], "metric": r["metric"], "split": r["split"], "evidence_ids": r["evidence_ids"][:2]}


# ------------------------------------------------------------------------------------------------ the three forms
def _comparisons(index, p: dict, found: list[dict[str, str]], st) -> list[dict]:
    groups: dict[tuple, list[list]] = defaultdict(lambda: [[], []])
    exact: dict[str, bool] = {}
    for side, exps in enumerate(found):
        for xid, kind in exps.items():
            ok, exact[xid] = _on_dataset(index, xid, p["dataset"])
            bench = _bench(index, xid)
            if not ok or bench is None:
                continue
            for r in index.exp[xid]["results"]:
                if r["metric_id"] and G.number_of(r["score"]) is not None and _metric_ok(p["metric"], r["metric"]):
                    groups[(bench["entity_id"], r["metric_id"])][side].append((xid, r, kind))
    pairs = [(k, ra, rb) for k, (sa, sb) in groups.items() for ra in sa for rb in sb if ra[0] != rb[0]]
    models = index.experiments({x for _, ra, rb in pairs for x in (ra[0], rb[0])})
    views = C.views(index, models)  # pre-training data and adaptation are conditions of the pair too
    items = []
    for (bench, metric), (xa, ra, ka), (xb, rb, kb) in pairs:
        ea, eb = models.get(xa), models.get(xb)
        if ea is None or eb is None:
            continue
        ca = {c: B._condition(ea, ra, c, views.get(xa)) for c in B.CONDITIONS}
        cb = {c: B._condition(eb, rb, c, views.get(xb)) for c in B.CONDITIONS}
        cls, why = B.classify(cb, {c: B._key(c, v) for c, v in ca.items()})
        same_paper = index.exp[xa]["paper_id"] == index.exp[xb]["paper_id"]
        va, vb = G.number_of(ra["score"]), G.number_of(rb["score"])
        dirn = direction(index.ent.get(metric, {}).get("name") or ra["metric"])
        if not (cls == "COMPARABLE" or (same_paper and cls != "INCOMPATIBLE")):
            why = _pair_reason(ca, cb, cls, why)
            stance, reason = "INCONCLUSIVE", ("Same paper, but " if same_paper else "Different papers. ") + (why[0].lower() if same_paper else why[0]) + why[1:]
        elif dirn is None:
            stance, reason = "INCONCLUSIVE", "The metric's name does not say whether higher or lower is better."
        else:
            stance = "SUPPORTS" if (va > vb if dirn == "higher" else va < vb) else "CONTRADICTS"
            basis = "the same paper's own comparison" if same_paper else "results with comparable stated conditions"
            reason = (f"{p['a']} {'scores higher' if va > vb else 'scores lower' if va < vb else 'ties'} ({ra['score']} vs {rb['score']}, "
                      f"{dirn} is better) in {basis}.")
        items.append({"stance": stance, "direct": ka == kb == "direct" and exact.get(xa, True) and exact.get(xb, True),
                      "benchmark": index.ent.get(bench, {}).get("name"), "metric": index.ent.get(metric, {}).get("name"), "same_paper": same_paper,
                      "comparability": cls, "direction": dirn, "difference": round(va - vb, 6), "reason": reason,
                      "a": _side(index, xa, ra, ka), "b": _side(index, xb, rb, kb)})
    coverage = [{"mention": m, "records": len({x for g in groups.values() for x, _, _ in g[side]}),
                 "results": sum(len(g[side]) for g in groups.values())} for side, m in enumerate((p["a"], p["b"]))]
    st.detail.update(groups=len(groups), pairs=len(items), same_paper=sum(i["same_paper"] for i in items), coverage=coverage,
                     **{c.lower(): sum(i["comparability"] == c for i in items) for c in B.CLASSES})
    return items, coverage


def _decimals(v: str) -> int:
    return len(v.split(".", 1)[1]) if "." in v else 0


def _values(index, p: dict, found: list[dict[str, str]], st) -> list[dict]:
    claimed, d = float(p["value"]), _decimals(p["value"])
    items = []
    for xid, kind in found[0].items():
        ok, exact = _on_dataset(index, xid, p["dataset"])
        if not ok:
            continue
        for r in index.exp[xid]["results"]:
            v = G.number_of(r["score"])
            if v is None or not _metric_ok(p["metric"], r["metric"]):
                continue
            scaled = v * 100 if p["percent"] and v <= 1 < claimed else v  # "87.8%" vs a recorded 0.878
            note = " (recorded as a fraction, read as a percentage)" if scaled != v else ""
            if abs(scaled - claimed) <= 0.5 * 10 ** -d + 1e-9:
                stance, reason = "SUPPORTS", f"Reports {r['score']}{note} for {r['metric']}: the claimed value at its stated precision."
            elif p["split"] and r["split"] and p["split"][:3] != canonical_label(r["split"])[:3]:
                stance, reason = "INCONCLUSIVE", f"Reports {r['score']} on the {r['split']} split; the claim is about the {p['split']} split."
            elif not p["metric"]:
                stance, reason = "INCONCLUSIVE", f"Reports {r['score']} for {r['metric']}; the claim names no metric, so another value is no contradiction."
            else:
                stance, reason = "CONTRADICTS", f"Reports {r['score']}{note} for {r['metric']}, not {p['value']} (the setup may differ: see its conditions)."
            items.append({"stance": stance, "direct": kind == "direct" and exact, "reason": reason, "claimed": p["value"],
                          "benchmark": (_bench(index, xid) or {}).get("raw"), "metric": r["metric"], "a": _side(index, xid, r, kind)})
    coverage = [{"mention": p["a"], "records": len({i["a"]["experiment"] for i in items}), "results": len(items)}]
    st.detail.update(results=len(items))
    return items, coverage


def _same_setting(path: str, claimed: str, stated) -> bool:
    a, b = G.number_of(claimed), G.number_of(stated)
    if a is not None and b is not None and path not in G.CATEGORICAL:
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return canonical_label(str(claimed)) == canonical_label(str(stated))


def _settings(index, p: dict, found: list[dict[str, str]], st) -> list[dict]:
    on = {xid: kind for xid, kind in found[0].items() if _on_dataset(index, xid, p["dataset"])[0]}
    models = index.experiments(set(on))
    items, papers = [], set()
    for xid, kind in on.items():
        e = models.get(xid)
        if e is None:
            continue
        papers.add(index.exp[xid]["paper_id"])
        for s in p["settings"]:
            f = F.fact(s["field"], F.field_value(e, s["field"]))
            if f["status"] not in F.STATED:
                continue
            stated = f["raw"] if f["raw"] not in (None, "") else f["value"]
            items.append({"stance": "SUPPORTS" if _same_setting(s["field"], s["value"], stated) else "CONTRADICTS",
                          "direct": kind == "direct" and _on_dataset(index, xid, p["dataset"])[1], "field": s["field"], "label": s["label"],
                          "claimed": s["value"], "stated": stated, "scope": "experiment", "evidence_ids": f["evidence_ids"][:2],
                          "reason": f"The record of this experiment states {s['label']} = {stated}.",
                          "a": _side(index, xid, {"score": None, "metric": None, "split": None, "evidence_ids": []}, kind)})
    level = paper_level(index, sorted(papers), [s["field"] for s in p["settings"]]) if papers else {}
    for (pid, path), values in level.items():
        s = next(s for s in p["settings"] if s["field"] == path)
        for v in values[:3]:
            names_dataset = bool(p["dataset"]) and G.dataset_match(p["dataset"], [v.get("applies_to")]) == 2
            stance = "SUPPORTS" if _same_setting(path, s["value"], v["value"]) else "CONTRADICTS" if names_dataset else "INCONCLUSIVE"
            items.append({"stance": stance, "direct": False, "field": path, "label": s["label"], "claimed": s["value"], "stated": v["value"],
                          "scope": "paper", "applies_to": v.get("applies_to"), "evidence_ids": v["evidence_ids"][:2],
                          "reason": f"The paper's text states {s['label']} = {v['value']}" + (f" for {v['applies_to']}" if v.get("applies_to") else "")
                                    + "; not assigned to an experiment record.",
                          "a": {"paper": pid, "title": index.paper.get(pid, {}).get("title"), "year": index.paper.get(pid, {}).get("year")}})
    st.detail.update(experiments=len(on), facts=len(items))
    return items, [{"mention": p["a"], "records": len(on), "results": len(items)}]


# ------------------------------------------------------------------------------------------------ assessment
def _verdict(items: list[dict]) -> str:
    s, c = (sum(i["stance"] == k for i in items) for k in ("SUPPORTS", "CONTRADICTS"))
    return "MIXED" if s and c else "SUPPORTED" if s else "CONTRADICTED" if c else "INSUFFICIENT_EVIDENCE"


WHY = {"SUPPORTED": "Every assessable record agrees with the claim.", "CONTRADICTED": "Every assessable record disagrees with the claim.",
       "MIXED": "Assessable records point both ways; see where each holds.",
       "INSUFFICIENT_EVIDENCE": "Records exist, but none can be assessed against the claim (conditions unstated or different, or the metric's direction unknown).",
       "NO_EVIDENCE": "No corpus record matches the claim's model(s) and dataset."}


def _questions(p: dict) -> list[str]:
    """Questions for the full research pipeline (corpus + live providers) that would gather the missing evidence."""
    on = f" on {p['dataset']}" if p.get("dataset") else ""
    if p["form"] == "setting":
        return [f"What {' and '.join(s['label'] for s in p['settings'])} were used to train {p['a']}{on}?"]
    metric = p.get("metric") or "score"
    if p["form"] == "value":
        return [f"What {metric} does {p['a']} achieve{on}?"]
    return [f"What {metric} does {m} achieve{on}?" for m in (p["a"], p["b"])] if on else []


def audit(index, claim: str, tr, fill=None) -> dict:
    """`fill(models, tr)`: for a setting claim, read the full texts for the claimed settings the records leave unknown
    (services/analysis/enrich.py) before they are compared with the claim."""
    with tr.stage("analysis_request", method="deterministic claim reading (comparison / reported value / setting)") as st:
        p = parse(index, claim)
        st.detail.update(form=p["form"], model=p["a"], other=p.get("b"), dataset=p.get("dataset"), metric=p.get("metric"), general=p["general"])
    with tr.stage("experiment_retrieval", method="model-name token coverage over every corpus experiment (no similarity)") as st:
        found, ambiguous = matched(index, [p["a"], p["b"]] if p["form"] == "comparison" else [p["a"]])
        st.detail.update(matched=[len(f) for f in found], direct=[sum(k == "direct" for k in f.values()) for f in found], ambiguous=ambiguous)
    if fill and p["form"] == "setting":  # the models are cached, so the values found here are the ones _settings compares
        on = [x for x, k in sorted(found[0].items(), key=lambda xk: xk[1] != "direct") if _on_dataset(index, x, p["dataset"])[0]][:8]
        fill(index.experiments(on), tr)
    with tr.stage("comparability" if p["form"] == "comparison" else "field_facts") as st:
        items, coverage = {"comparison": _comparisons, "value": _values, "setting": _settings}[p["form"]](index, p, found, st)
    with tr.stage("assessment", method="direct evidence first; inconclusive pairs never count") as st:
        direct = [i for i in items if i["direct"]]
        assessable = [i for i in items if i["stance"] != "INCONCLUSIVE"]
        basis = "direct" if any(i in assessable for i in direct) else "indirect" if assessable else "none"
        pool = direct if basis == "direct" else items
        missing = [c["mention"] for c in coverage if not c["records"]]
        verdict = "NO_EVIDENCE" if not items else _verdict(pool)
        why = WHY[verdict]
        if verdict == "NO_EVIDENCE" and missing:
            on = f" on {p['dataset']}" if p.get("dataset") else ""
            why = (f"No corpus record reports {' or '.join(missing)}{on}"
                   + (f" ({'; '.join(f'{c['mention']}: {c['results']} result(s)' for c in coverage if c['records'])})" if len(missing) < len(coverage) else "")
                   + ". Absence from the corpus is not evidence against the claim.")
        counted = [i for i in pool if i["stance"] != "INCONCLUSIVE"]
        benches = sorted({i.get("benchmark") for i in counted if i.get("benchmark")})
        paper_ids = sorted({x for i in counted for x in (i["a"].get("paper"), (i.get("b") or {}).get("paper")) if x})
        flags = []
        if p["general"] and counted and len(benches) <= 1:
            flags.append(f"The claim is stated generally, but the assessable evidence covers {len(benches)} benchmark(s).")
        if p["general"] and verdict == "MIXED" and len(benches) > 1:
            held = sorted({i.get("benchmark") for i in counted if i["stance"] == "SUPPORTS"} - {None})
            failed = sorted({i.get("benchmark") for i in counted if i["stance"] == "CONTRADICTS"} - {None})
            flags.append(f"The outcome depends on the benchmark: it holds on {', '.join(held) or 'none'}, not on {', '.join(failed) or 'none'}.")
        if counted and len(paper_ids) == 1:
            flags.append("All assessable evidence comes from a single paper.")
        metrics = {i.get("metric") for i in counted if i.get("metric")}
        if p["form"] != "setting" and not p.get("metric") and len(metrics) > 1:
            flags.append(f"The claim names no metric; the evidence uses {len(metrics)} different metrics, each judged on its own.")
        reasons = []
        if basis == "indirect":
            reasons.append("only indirect evidence (variants of the model, dataset variants or paper-level statements)")
        if counted and len(paper_ids) == 1:
            reasons.append("a single paper")
        if any(i.get("comparability", "COMPARABLE") != "COMPARABLE" for i in counted):
            reasons.append("evaluation conditions partly unstated (same-paper comparisons)")
        reasons.append("machine-extracted records that no human has reviewed")
        st.detail.update(verdict=verdict, basis=basis, supporting=sum(i["stance"] == "SUPPORTS" for i in items),
                         contradicting=sum(i["stance"] == "CONTRADICTS" for i in items), inconclusive=sum(i["stance"] == "INCONCLUSIVE" for i in items),
                         flags=len(flags))
    ids = [i for it in items for i in it.get("evidence_ids", []) + it["a"].get("evidence_ids", []) + (it.get("b") or {}).get("evidence_ids", [])]
    ev = {e["id"]: e for e in index.evidence_detail(ids, limit=400)} if ids else {}
    for it in items:
        for side in (it, it["a"], it.get("b") or {}):
            if "evidence_ids" in side:
                side["evidence"] = [ev[i] for i in side.pop("evidence_ids") if i in ev]
    order = {"SUPPORTS": 0, "CONTRADICTS": 1, "INCONCLUSIVE": 2}
    items.sort(key=lambda i: (order[i["stance"]], not i["direct"], not i.get("same_paper", False), str(i.get("benchmark"))))
    by = defaultdict(list)
    for it in items:
        by[it["stance"]].append(it)
    text = " ".join(claim.split())
    return {"kind": "claim", "version": VERSION, "claim": text, "parsed": p, "verdict": verdict, "verdict_why": why, "basis": basis, "coverage": coverage,
            "supporting": by["SUPPORTS"][:MAX_ITEMS], "contradicting": by["CONTRADICTS"][:MAX_ITEMS], "inconclusive": by["INCONCLUSIVE"][:MAX_ITEMS],
            "flags": flags, "confidence": {"level": "LOW" if len(reasons) > 1 else "MODERATE", "reasons": reasons},
            "summary": {"verdict": verdict, "basis": basis, "supporting": len(by["SUPPORTS"]), "contradicting": len(by["CONTRADICTS"]),
                        "inconclusive": len(by["INCONCLUSIVE"]), "direct": len(direct), "benchmarks": benches, "papers": len(paper_ids)},
            "inputs": {"claim": text}, "research_questions": _questions(p),
            "scope": {"experiments_searched": len(index.exp), "papers_in_corpus": len(index.paper), "matched": [len(f) for f in found],
                      "ambiguous_records_skipped": ambiguous},
            "notes": ["Evidence is the corpus records only; no model's opinion is used as evidence.",
                      "A pair counts only on the same benchmark and metric, within one paper or under comparable stated conditions.",
                      "Whether higher or lower is better comes from the metric's name; scores of different metrics are never converted.",
                      "Absence of records is not evidence against the claim: the corpus is a sample of the literature."]}
