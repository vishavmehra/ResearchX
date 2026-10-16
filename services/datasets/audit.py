"""Dataset audit: layer inventory, field coverage, evidence granularity, source-level quality and KPIs.

Everything is computed from files on disk. Metrics that need human Gold (precision, agreement,
association accuracy) are reported as NOT_MEASURABLE with the reason, never estimated.
Used for the v0.1.0 baseline audit and again for every later release (deepening program §30-31).
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from services.canonicalization.normalize import normalize_title
from services.datasets.schemas import FIELD_GROUPS, OUTCOME_FIELDS
from services.datasets.store import DATA, ROOT, atomic_write_bytes, now_utc, path, read_jsonl

AUDIT_VERSION = "audit-0.1.0"
# Identity-like fields: knowing them says *what* was run, not *how*. Excluded from "technical depth".
IDENTITY_FIELDS = frozenset(
    {"research.task", "research.domain", "evaluation.benchmark", "model.model_name", "model.model_family", "dataset.dataset_name", "artifacts.paper_url"}
)
CONFIG_PATHS = [f"{g}.{f}" for g, names in FIELD_GROUPS.items() for f in names]
TECHNICAL_PATHS = [p for p in CONFIG_PATHS if p not in IDENTITY_FIELDS]
DEEP_THRESHOLD = 5  # >= 5 technical fields KNOWN = "deep" for the baseline KPI (formal depth score: v0.2.0)
GITHUB = re.compile(r"github\.com/[\w.-]+/[\w.-]+", re.I)
NOT_MEASURABLE = "NOT_MEASURABLE"


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 2) if d else None


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def _dist(values) -> dict[str, int]:
    return dict(sorted(Counter(v if v is not None else "null" for v in values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def locator_class(ev: dict) -> str:
    """Granularity of the location an evidence assertion points to."""
    loc = ev.get("source_location") or ""
    table = ev.get("table") or ""
    if ev.get("paragraph"):
        return "paper_paragraph_span" if ev.get("char_start") is not None else "paper_paragraph"
    if table.startswith("leaderboard:"):
        return "leaderboard_claim"
    if table:
        return "paper_table_cell" if re.search(r"row|cell|col", loc, re.I) else "paper_table"
    if "links-between-paper-and-code" in loc:
        return "code_link_claim"
    if "metadata record" in loc:
        return "source_metadata"
    return "other"


PAPER_TEXT = frozenset({"paper_paragraph_span", "paper_paragraph", "paper_table_cell", "paper_table"})


def _field_items(exp: dict):
    """(path, FieldValue dict) for configuration fields and outcome fields of one experiment."""
    for g, names in FIELD_GROUPS.items():
        section = exp.get(g) or {}
        for n in names:
            fv = section.get(n)
            if fv:
                yield f"{g}.{n}", fv
    for o in exp.get("outcomes", []):
        for n in OUTCOME_FIELDS:
            fv = o.get(n)
            if fv:
                yield f"outcome.{n}", fv


def load_parsed_docs(papers: list[dict]) -> dict[str, dict]:
    docs = {}
    for p in papers:
        pp = (p.get("fulltext") or {}).get("parsed_path")
        if pp and (ROOT / pp).exists():
            docs[p["paper_id"]] = json.loads((ROOT / pp).read_text(encoding="utf-8"))
    return docs


def _silver(kind: str) -> list[dict]:
    return [r for f in sorted((DATA / "silver" / kind).glob("*.jsonl")) for r in read_jsonl(f)]


def audit() -> dict[str, Any]:
    src = read_jsonl(path("source_records"))
    papers = read_jsonl(path("papers"))
    studies = read_jsonl(path("studies"))
    citations = read_jsonl(path("citations"))
    exps = _silver("experiments")
    evs = _silver("evidence")
    links = _silver("study_links")
    comps = read_jsonl(path("comparisons")) + read_jsonl(path("hard_negatives"))  # both files hold comparison records
    hard = [c for c in comps if c.get("is_hard_negative")]
    contras = _silver("contradictions")
    abst = _silver("abstention")
    decisions = read_jsonl(path("decisions"))
    promotions = read_jsonl(path("promotions"))
    gold_files = sorted((DATA / "gold").rglob("*.jsonl"))
    gold = sum(len(read_jsonl(f)) for f in gold_files)
    docs = load_parsed_docs(papers)

    rec_source = {r["source_record_id"]: r["source_name"] for r in src}
    rec_paper = {rid: p["paper_id"] for p in papers for rid in p["source_record_ids"]}
    relevant = {p["paper_id"] for p in papers if (p.get("relevance") or {}).get("is_relevant")}
    ev_by_id = {e["evidence_id"]: e for e in evs}
    ev_class = {e["evidence_id"]: locator_class(e) for e in evs}

    # ------------------------------------------------------------------ experiments
    field_known: Counter = Counter()
    field_conflicting: Counter = Counter()
    field_paper_text: Counter = Counter()
    tech_per_exp: list[int] = []
    verified_per_exp: list[int] = []
    exp_by_paper: Counter = Counter()
    for e in exps:
        exp_by_paper[e["paper_id"]] += 1
        tech = verified = 0
        for p_, fv in _field_items(e):
            st = fv.get("status")
            if st == "KNOWN":
                field_known[p_] += 1
                in_text = any(ev_class.get(i) in PAPER_TEXT for i in fv.get("evidence_ids", []))
                field_paper_text[p_] += in_text
                verified += in_text
                tech += p_ in TECHNICAL_PATHS
            elif st == "CONFLICTING":
                field_conflicting[p_] += 1
        tech_per_exp.append(tech)
        verified_per_exp.append(verified)
    n_exp = len(exps)
    n_out = sum(len(e.get("outcomes", [])) for e in exps)
    coverage = {}
    for p_ in CONFIG_PATHS + [f"outcome.{n}" for n in OUTCOME_FIELDS]:
        denom = n_out if p_.startswith("outcome.") else n_exp
        coverage[p_] = {
            "known": field_known[p_],
            "known_pct": _pct(field_known[p_], denom),
            "known_with_paper_text_evidence": field_paper_text[p_],
            "conflicting": field_conflicting[p_],
        }
    group_cov = {}
    for g, names in FIELD_GROUPS.items():
        fields = [f"{g}.{n}" for n in names]
        filled = [f for f in fields if field_known[f]]
        group_cov[g] = {"fields": len(fields), "fields_with_any_known": len(filled), "fields_never_known": [f for f in fields if not field_known[f]]}
    cross = Counter(o.get("status") for e in exps for o in (e.get("cross_check") or {}).get("outcomes", []))

    # ------------------------------------------------------------------ documents (Layer 2/3)
    tables = [t for d in docs.values() for t in d.get("tables", [])]
    cells = sum(len(r) for t in tables for r in t.get("rows", []))
    doc_stats = {
        "parsed_documents": len(docs),
        "by_format": _dist((p.get("fulltext") or {}).get("format") for p in papers if p["paper_id"] in docs),
        "by_license_class": _dist((p.get("fulltext") or {}).get("license_status") for p in papers if p["paper_id"] in docs),
        "sections": sum(len(d.get("sections", [])) for d in docs.values()),
        "section_types": _dist(s["section_type"] for d in docs.values() for s in d.get("sections", [])),
        "paragraphs": sum(len(d.get("paragraphs", [])) for d in docs.values()),
        "tables": len(tables),
        "tables_with_rows": sum(1 for t in tables if t.get("rows")),
        "tables_with_caption": sum(1 for t in tables if (t.get("caption") or "").strip()),
        "tables_with_header_rows": sum(1 for t in tables if t.get("header_rows")),
        "table_cells": cells,
        "sentences": 0,  # not represented in schema 0.1.0
        "figures": 0,
        "footnotes": 0,
        "documents_with_zero_sections": sum(1 for d in docs.values() if not d.get("sections")),
        "relevant_papers_with_parsed_fulltext_pct": _pct(len(relevant & set(docs)), len(relevant)),
    }

    # ------------------------------------------------------------------ evidence (Layer 5)
    ev_papers = Counter(e.get("paper_id") for e in evs if e.get("paper_id"))
    evidence_stats = {
        "total": len(evs),
        "by_status": _dist(e.get("status", "MACHINE_EXTRACTED") for e in evs),
        "by_extractor": _dist(e["extraction_model"] for e in evs),
        "by_locator_granularity": _dist(ev_class.values()),
        "top_fields": dict(list(_dist(e["field_name"] for e in evs).items())[:25]),
        "redistribution_allowed": _dist(e.get("redistribution_allowed") for e in evs),
        "papers_with_evidence": len(ev_papers),
        "papers_with_paper_text_evidence": len({e.get("paper_id") for e in evs if ev_class[e["evidence_id"]] in PAPER_TEXT}),
    }

    # ------------------------------------------------------------------ relationship datasets (Layers 7-8)
    comp_field_status: dict[str, Counter] = defaultdict(Counter)
    for c in comps:
        for f in c.get("field_comparisons", []):
            comp_field_status[re.sub(r"\[.*\]", "", f["field_name"])][f["comparison_status"]] += 1
    blocking = Counter(m for a in abst for m in a.get("missing_information", []))
    relationships = {
        "study_links": {"total": len(links), "by_relationship": _dist(l["relationship"] for l in links), "by_candidate_method": _dist(m for l in links for m in l.get("candidate_methods", [])), "hard_negatives": sum(1 for l in links if l.get("is_hard_negative"))},
        "comparisons": {"total": len(comps), "by_label": _dist(c["relationship_label"] for c in comps)},
        "hard_negatives": {"total": len(hard) + sum(1 for l in links if l.get("is_hard_negative")), "comparison_hard_negatives": len(hard), "comparison_hard_negatives_by_type": _dist(h.get("hard_negative_type") for h in hard), "comparison_hard_negative_labels": _dist(h["relationship_label"] for h in hard)},
        "comparison_field_status": {k: dict(v) for k, v in sorted(comp_field_status.items())},
        "contradictions": {"total": len(contras), "by_type": _dist(c["conflict_type"] for c in contras), "by_field": _dist(c["field"] for c in contras)},
        "abstention": {"total": len(abst), "by_class": _dist(a["evidence_class"] for a in abst), "by_behavior": _dist(a["expected_behavior"] for a in abst), "most_common_missing_information": dict(blocking.most_common(10))},
    }

    # ------------------------------------------------------------------ source-level quality (§30)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in src:
        by_source[r["source_name"]].append(r)
    paper_by_id = {p["paper_id"]: p for p in papers}
    conflict_records = Counter()
    for p in papers:
        for c in p.get("metadata_conflicts", []):
            for rid in (c.get("values") or {}):
                conflict_records[rid] += 1
    per_paper_source = Counter((rec_paper.get(r["source_record_id"]), r["source_name"]) for r in src)
    ev_by_source = Counter(rec_source.get(e["source_record_id"]) for e in evs)
    exp_sources: dict[str, list[int]] = defaultdict(list)
    for e, tech in zip(exps, tech_per_exp):
        for s in {rec_source.get(rid) for rid in e.get("source_record_ids", [])}:
            exp_sources[s].append(tech)
    link_sources = Counter()
    for l in links:
        for s in {rec_source.get(it.get("source_record_id")) for it in l.get("evidence", []) if it.get("source_record_id")}:
            link_sources[s] += 1
    papers_with_exp = set(exp_by_paper)
    papers_with_text_ev = {e.get("paper_id") for e in evs if ev_class[e["evidence_id"]] in PAPER_TEXT}
    source_quality = {}
    for s, recs in sorted(by_source.items()):
        n = len(recs)
        pids = [rec_paper.get(r["source_record_id"]) for r in recs]
        agree = {"title": [0, 0], "publication_year": [0, 0], "doi": [0, 0]}
        for r, pid in zip(recs, pids):
            p = paper_by_id.get(pid)
            if not p:
                continue
            for f in agree:
                rv, pv = r.get(f), p.get(f)
                if rv is None or pv is None:
                    continue
                agree[f][1] += 1
                agree[f][0] += (normalize_title(rv) == normalize_title(pv)) if f == "title" else (rv == pv)
        missing = {
            f: _pct(sum(1 for r in recs if not r.get(f)), n)
            for f in ("title", "abstract", "authors", "publication_date", "venue", "doi", "arxiv_id", "references")
        }
        missing["cited_by_count"] = _pct(sum(1 for r in recs if r.get("cited_by_count") is None), n)
        missing["content_license"] = _pct(sum(1 for r in recs if not (r.get("license") or {}).get("content_license")), n)
        github = sum(1 for r in recs if GITHUB.search(json.dumps([r.get("urls"), r.get("extra"), r.get("abstract")], ensure_ascii=False)))
        source_quality[s] = {
            "records": n,
            "papers": len({p for p in pids if p}),
            "metadata_precision": {"status": NOT_MEASURABLE, "reason": "needs human-verified Gold metadata"},
            "cross_source_agreement_pct": {f: _pct(a, d) for f, (a, d) in agree.items()},
            "duplicate_rate_pct": _pct(sum(c - 1 for (pid, ss), c in per_paper_source.items() if ss == s and pid and c > 1), n),
            "overlap_with_other_sources_pct": _pct(sum(1 for pid in pids if pid and len(paper_by_id[pid]["source_record_ids"]) > 1), n),
            "missingness_pct": missing,
            "evidence_density": round(ev_by_source[s] / n, 3) if n else None,
            "experiment_density": round(len(exp_sources[s]) / n, 3) if n else None,
            "configuration_density_mean_technical_fields": _mean(exp_sources[s]),
            "artifact_linkage_rate_pct": _pct(github, n),
            "source_conflict_rate_pct": _pct(sum(1 for r in recs if conflict_records[r["source_record_id"]]), n),
            "study_link_yield": round(link_sources[s] / n, 4) if n else None,
            "useful_paper_yield_pct": {
                "relevant": _pct(sum(1 for p in pids if p in relevant), n),
                "with_experiments": _pct(sum(1 for p in pids if p in papers_with_exp), n),
                "with_parsed_fulltext": _pct(sum(1 for p in pids if p in docs), n),
                "with_paper_text_evidence": _pct(sum(1 for p in pids if p in papers_with_text_ev), n),
            },
        }

    # ------------------------------------------------------------------ KPIs (§31)
    repo_papers = {e.get("paper_id") for e in evs if e["field_name"] == "artifacts.repository_url"}
    seeds = sum(1 for e in exps if ((e.get("training") or {}).get("seed") or {}).get("status") == "KNOWN")
    reproducible = sum(
        1 for e in exps
        if e.get("paper_id") in repo_papers
        and any(((e.get("training") or {}).get(f) or {}).get("status") == "KNOWN" for f in FIELD_GROUPS["training"])
        and ((e.get("training") or {}).get("seed") or {}).get("status") == "KNOWN"
    )
    n_papers, n_rel = len(papers), len(relevant)
    kpis = {
        "evidence_assertions_per_paper": round(len(evs) / n_papers, 3) if n_papers else None,
        "evidence_assertions_per_relevant_paper": round(sum(v for k, v in ev_papers.items() if k in relevant) / n_rel, 3) if n_rel else None,
        "deep_experiments_per_paper_with_experiments": round(sum(1 for t in tech_per_exp if t >= DEEP_THRESHOLD) / len(papers_with_exp), 3) if papers_with_exp else None,
        "deep_experiment_definition": f">= {DEEP_THRESHOLD} technical configuration fields KNOWN",
        "technical_fields_filled_per_experiment": {"mean": _mean(tech_per_exp), "median": _median(tech_per_exp), "max": max(tech_per_exp, default=0), "of_possible": len(TECHNICAL_PATHS)},
        "technical_fields_histogram": dict(sorted(Counter(tech_per_exp).items())),
        "verified_fields_per_experiment": {"mean": _mean(verified_per_exp), "median": _median(verified_per_exp), "definition": "KNOWN fields with SOURCE_CHECKED evidence located in the paper's own text (paragraph/table)"},
        "experiment_result_association_accuracy": {"status": NOT_MEASURABLE, "reason": "no human-verified associations; automatic table cross-check statuses reported instead", "cross_check_statuses": dict(cross)},
        "study_link_precision": {"status": NOT_MEASURABLE, "reason": "no human-verified study links"},
        "comparison_label_agreement": {"status": NOT_MEASURABLE, "reason": "no double-annotated comparisons"},
        "hard_negative_yield": {"per_comparison": round(len(hard) / len(comps), 3) if comps else None, "study_link_share": round(relationships["study_links"]["hard_negatives"] / len(links), 3) if links else None},
        "contradiction_yield": {"per_100_papers": round(100 * len(contras) / n_papers, 2) if n_papers else None, "numeric_or_configuration": sum(1 for c in contras if c["conflict_type"] in ("NUMERIC_CONFLICT", "CONFIGURATION_CONFLICT"))},
        "abstention_quality": {"status": NOT_MEASURABLE, "reason": "no human-verified abstention labels", "class_distribution": relationships["abstention"]["by_class"]},
        "artifact_link_yield_pct": _pct(len(repo_papers & relevant), n_rel),
        "reproducibility_coverage_pct": _pct(reproducible, n_exp),
        "reproducibility_definition": "experiment whose paper links a repository AND has >= 1 training field AND seed KNOWN",
        "experiments_with_seed": seeds,
        "fulltext_coverage_relevant_pct": doc_stats["relevant_papers_with_parsed_fulltext_pct"],
        "table_structure_coverage_pct": _pct(doc_stats["tables_with_rows"], doc_stats["tables"]),
        "gold_records": gold,
    }

    # ------------------------------------------------------------------ layers
    layers = [
        ("L0 scholarly discovery corpus", len(src), "source records"),
        ("L1 normalized scholarly corpus", len(papers), f"papers ({n_rel} relevant), {len(studies)} studies, {len(citations)} citation edges"),
        ("L2 full-text research corpus", len(docs), "parsed full texts"),
        ("L3 scientific document structure", doc_stats["paragraphs"] + doc_stats["tables"], f"{doc_stats['sections']} sections, {doc_stats['paragraphs']} paragraphs, {doc_stats['tables']} tables ({doc_stats['tables_with_rows']} with rows), 0 sentences/figures/footnotes"),
        ("L4 deep experiment extraction", sum(1 for t in tech_per_exp if t >= DEEP_THRESHOLD), f"deep experiments of {n_exp} Silver experiment conditions"),
        ("L5 evidence + attribution", len(evs), f"evidence assertions ({evidence_stats['papers_with_paper_text_evidence']} papers with paper-text evidence)"),
        ("L6 artifact/code/config linkage", len(repo_papers), "papers with a repository link; 0 artifact/config records"),
        ("L7 study linking", len(links), f"study links ({relationships['study_links']['by_relationship']})"),
        ("L8 experiment comparison", len(comps), f"comparison records, of which {len(hard)} are hard negatives"),
        ("contradictions", len(contras), "Silver contradictions"),
        ("abstention", len(abst), "Silver abstention cases"),
        ("human Gold", gold, f"{len(decisions)} review decisions ({_dist(d['reviewer_type'] for d in decisions)}), {len(promotions)} promotions"),
        ("Laya decisions / trajectories", 0, "not built (by design until the data deepening phase)"),
        ("research tasks", 0, "not built"),
        ("frozen evaluation", sum(1 for _ in (DATA / "eval").rglob("*.jsonl")), "files under data/eval"),
    ]

    years = Counter(p.get("publication_year") for p in papers)
    retrieved = sorted(r["provenance"]["retrieved_at"] for r in src if (r.get("provenance") or {}).get("retrieved_at"))
    result = {
        "audit_version": AUDIT_VERSION,
        "generated_at": now_utc().isoformat(),
        "layers": [{"layer": a, "count": b, "detail": c} for a, b, c in layers],
        "documents": doc_stats,
        "experiments": {
            "total": n_exp,
            "outcomes": n_out,
            "by_extraction_method": _dist(e["extraction_method"] for e in exps),
            "by_type": _dist(e.get("experiment_type") for e in exps),
            "papers_with_experiments": len(papers_with_exp),
            "experiments_per_paper": {"mean": _mean(list(exp_by_paper.values())), "median": _median(list(exp_by_paper.values())), "max": max(exp_by_paper.values(), default=0)},
            "experiments_on_papers_with_fulltext": sum(1 for e in exps if e["paper_id"] in docs),
            "group_coverage": group_cov,
            "field_coverage": coverage,
            "table_cross_check": dict(cross),
        },
        "evidence": evidence_stats,
        "relationships": relationships,
        "source_quality": source_quality,
        "kpis": kpis,
        "temporal": {
            "publication_year": {str(k): v for k, v in sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
            "source_retrieval_window": [retrieved[0], retrieved[-1]] if retrieved else None,
            "papers_missing_year": years.get(None, 0),
        },
        "licensing": {
            "paper_content_license": _dist((p.get("license") or {}).get("content_license") for p in papers),
            "fulltext_license_class": doc_stats["by_license_class"],
        },
        "review": {"decisions": len(decisions), "by_reviewer_type": _dist(d["reviewer_type"] for d in decisions), "by_decision": _dist(d["decision"] for d in decisions), "promotions": len(promotions), "gold_records": gold},
    }
    return result


def _table(rows: list[list[Any]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return out


def to_markdown(a: dict[str, Any], title: str) -> str:
    L = [f"# {title}", "", f"Generated {a['generated_at']} by `{a['audit_version']}` from the files on disk. "
         "Values marked NOT_MEASURABLE need human-verified Gold and are deliberately not estimated.", ""]
    L += ["## 1. Layer inventory", ""] + _table([[x["layer"], x["count"], x["detail"]] for x in a["layers"]], ["Layer", "Count", "Detail"])
    d = a["documents"]
    L += ["", "## 2. Full text and document structure (Layers 2-3)", ""]
    L += _table([[k, v] for k, v in d.items() if not isinstance(v, dict)], ["Measure", "Value"])
    L += ["", f"Section types: `{json.dumps(d['section_types'])}`", ""]
    e = a["experiments"]
    L += ["## 3. Experiment layer (Layer 4)", "", f"{e['total']} experiment conditions, {e['outcomes']} outcomes, {e['papers_with_experiments']} papers. "
          f"Extraction: `{json.dumps(e['by_extraction_method'])}`. On papers with parsed full text: {e['experiments_on_papers_with_fulltext']}.", ""]
    L += _table([[g, v["fields"], v["fields_with_any_known"], ", ".join(x.split(".", 1)[1] for x in v["fields_never_known"])] for g, v in e["group_coverage"].items()],
                ["Group", "Fields", "Fields ever KNOWN", "Never KNOWN"])
    L += ["", "Fields with any KNOWN value:", ""]
    L += _table([[k, v["known"], v["known_pct"], v["known_with_paper_text_evidence"], v["conflicting"]] for k, v in e["field_coverage"].items() if v["known"] or v["conflicting"]],
                ["Field", "KNOWN", "% of experiments/outcomes", "KNOWN with paper-text evidence", "CONFLICTING"])
    L += ["", f"Table cross-check statuses: `{json.dumps(e['table_cross_check'])}`", ""]
    v = a["evidence"]
    L += ["## 4. Evidence (Layer 5)", "", f"{v['total']} assertions; status `{json.dumps(v['by_status'])}`.", ""]
    L += _table([[k, n] for k, n in v["by_locator_granularity"].items()], ["Location granularity", "Assertions"])
    L += [""] + _table([[k, n] for k, n in v["by_extractor"].items()], ["Extractor", "Assertions"])
    r = a["relationships"]
    L += ["", "## 5. Relationship datasets (Layers 7-8)", ""]
    L += [f"- Study links: {r['study_links']['total']} `{json.dumps(r['study_links']['by_relationship'])}`; methods `{json.dumps(r['study_links']['by_candidate_method'])}`",
          f"- Comparisons: {r['comparisons']['total']} `{json.dumps(r['comparisons']['by_label'])}`",
          f"- Hard negatives: {r['hard_negatives']['total']} (comparison types `{json.dumps(r['hard_negatives']['comparison_hard_negatives_by_type'])}`)",
          f"- Contradictions: {r['contradictions']['total']} `{json.dumps(r['contradictions']['by_type'])}`",
          f"- Abstention: {r['abstention']['total']} `{json.dumps(r['abstention']['by_class'])}`; most common missing information `{json.dumps(r['abstention']['most_common_missing_information'])}`", ""]
    L += ["Comparison field status (why pairs cannot be ALIGNED):", ""]
    L += _table([[f, s.get("IDENTICAL", 0) + s.get("EQUIVALENT", 0), s.get("MATERIALLY_DIFFERENT", 0) + s.get("DIFFERENT", 0), s.get("UNKNOWN", 0)] for f, s in r["comparison_field_status"].items()],
                ["Field", "Identical/equivalent", "Different", "UNKNOWN"])
    L += ["", "## 6. Source-level quality", ""]
    sq = a["source_quality"]
    rows = []
    for s, q in sq.items():
        rows.append([s, q["records"], q["duplicate_rate_pct"], q["overlap_with_other_sources_pct"], q["source_conflict_rate_pct"], q["evidence_density"], q["experiment_density"],
                     q["configuration_density_mean_technical_fields"], q["artifact_linkage_rate_pct"], q["study_link_yield"], q["useful_paper_yield_pct"]["relevant"], q["useful_paper_yield_pct"]["with_parsed_fulltext"]])
    L += _table(rows, ["Source", "Records", "Duplicate %", "Overlap %", "Conflict %", "Evidence / record", "Experiments / record", "Mean technical fields", "Artifact link %", "Study-link yield", "Relevant %", "Full text %"])
    L += ["", "Missingness (% of the source's records) and cross-source agreement (precision proxy; true precision is NOT_MEASURABLE without Gold):", ""]
    fields = list(next(iter(sq.values()))["missingness_pct"]) if sq else []
    L += _table([[s] + [q["missingness_pct"][f] for f in fields] + [json.dumps(q["cross_source_agreement_pct"])] for s, q in sq.items()], ["Source"] + fields + ["Agreement % (title/year/doi)"])
    k = a["kpis"]
    L += ["", "## 7. KPIs (not paper counts)", ""]
    L += _table([[key, json.dumps(val) if isinstance(val, (dict, list)) else val] for key, val in k.items() if key != "technical_fields_histogram"], ["KPI", "Value"])
    L += ["", f"Technical fields per experiment (histogram): `{json.dumps(k['technical_fields_histogram'])}`", ""]
    t = a["temporal"]
    L += ["## 8. Temporal, licensing, review", "", f"- Publication years: `{json.dumps(t['publication_year'])}`", f"- Source retrieval window: {t['source_retrieval_window']}",
          f"- Paper content licenses: `{json.dumps(a['licensing']['paper_content_license'])}`", f"- Full-text license classes: `{json.dumps(a['licensing']['fulltext_license_class'])}`",
          f"- Review: `{json.dumps(a['review'])}`", ""]
    return "\n".join(L) + "\n"


def write(a: dict[str, Any], stem: str = "dataset_audit", title: str = "ResearchX dataset audit", out_dir: Path = DATA / "reports") -> tuple[Path, Path]:
    j, m = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    atomic_write_bytes(j, json.dumps(a, indent=1, default=str).encode("utf-8"))
    atomic_write_bytes(m, to_markdown(a, title).encode("utf-8"))
    return j, m
