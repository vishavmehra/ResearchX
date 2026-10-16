"""Layer 0 discovery corpus and promotion into the main corpus (docs/DATASET_DEEPENING.md §4).

Layer 0 = every harvested arXiv OAI-PMH record that (a) matches >= 1 priority area in title+abstract,
(b) carries an ML/CS arXiv category and (c) was first submitted on/after `discovery.min_v1_date`.
Rows are compact DiscoveryCandidate records; abstracts stay in the raw cache.

Promotion picks new candidates (not already in the corpus) for full text by a documented, deterministic
priority score, with a per-area minimum so every priority domain is represented. The score is a
candidate-selection signal only; it never becomes a label.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from services.datasets.manifests import match_license
from services.datasets.schemas import DiscoveryCandidate, Provenance
from services.datasets.store import DATA, atomic_write_bytes, code_version, config, now_utc, write_jsonl
from services.retrieval.corpus import match_areas
from services.retrieval.sources import arxiv_oai, openalex

CANDIDATES = DATA / "normalized" / "discovery" / "candidates.jsonl"
SELECTION = arxiv_oai.SELECTION
PERMISSIVE = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"}
CODE_URL = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/[\w.\-]+(?:/[\w.\-]+)?", re.I)
EXPERIMENT_SIGNALS = {
    "outperforms": re.compile(r"\boutperform(s|ed|ing)?\b", re.I),
    "state_of_the_art": re.compile(r"\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b"),
    "benchmark": re.compile(r"\bbenchmarks?\b", re.I),
    "ablation": re.compile(r"\bablations?\b", re.I),
    "we_train_or_finetune": re.compile(r"\bwe (?:fine[- ]?tune|train|pre-?train|evaluate)\b", re.I),
    "experiments": re.compile(r"\bexperiments? (?:on|show|demonstrate|reveal|across)\b", re.I),
    "metric_named": re.compile(r"\b(accuracy|F1|BLEU|ROUGE|exact match|pass@\d+|perplexity|nDCG|MRR|recall@\d+)\b", re.I),
    "percent_number": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
}


def _signals(abstract: str) -> list[str]:
    return sorted(name for name, pat in EXPERIMENT_SIGNALS.items() if pat.search(abstract))


def score(c: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """selection-0.1.0 priority score (documented in docs/DATASET_DEEPENING.md §4)."""
    comp = {
        "areas": float(min(len(c["areas"]), 4)),
        "experiment_signals": 0.75 * min(len(c["experiment_signals"]), 4),
        "code_url": 2.0 if c["code_urls"] else 0.0,
        "published_version": 1.0 if (c["journal_ref"] or c["published_doi"]) else 0.0,
        "multiple_versions": 0.5 if c["n_versions"] >= 2 else 0.0,
        "permissive_licence": 0.5 if c["license_id"] in PERMISSIVE else 0.0,
        "recent": 0.5 if (c["v1_date"] or "") >= "2025-01-01" else 0.0,
    }
    return round(sum(comp.values()), 3), comp


def build(min_v1_date: str | None = None) -> tuple[list[DiscoveryCandidate], dict[str, Any]]:
    cfg = config("pipeline")["discovery"]
    min_v1 = min_v1_date or cfg["min_v1_date"]
    prefixes = tuple(config("domain")["field_check"]["arxiv_category_prefixes"])
    # corpus = OpenAlex-discovered papers (pure function of the raw cache; no dependency on a previous build)
    corpus_ids = {r.arxiv_id for r in openalex.normalize_all() if r.arxiv_id}
    rows, seen = [], 0
    for e, page in arxiv_oai.iter_entries():
        seen += 1
        versions = e.get("versions") or []
        v1 = versions[0]["date"] if versions else None
        cats = e.get("categories") or []
        if not v1 or v1 < min_v1 or not any(c.startswith(prefixes) for c in cats):
            continue
        text = f"{e.get('title') or ''}\n{e.get('abstract') or ''}"
        areas = match_areas(text)
        if not areas:
            continue
        signals = _signals(e.get("abstract") or "")
        if cfg.get("require_experiment_signal") and not signals and e["arxiv_id"] not in corpus_ids:
            continue
        c = {
            "arxiv_id": e["arxiv_id"], "title": e.get("title") or "", "v1_date": v1,
            "latest_version": versions[-1]["version"] if versions else None, "n_versions": len(versions),
            "primary_category": cats[0] if cats else None, "categories": cats,
            "license_id": match_license(e.get("license")), "license_url": e.get("license"),
            "journal_ref": e.get("journal_ref"), "published_doi": e.get("doi"),
            "code_urls": sorted({u.rstrip(".") for u in CODE_URL.findall(f"{e.get('comments') or ''} {e.get('abstract') or ''}")}),
            "areas": sorted(areas), "experiment_signals": signals,  # matched terms are re-derivable: not stored
        }
        s, comp = score(c)
        rows.append(
            DiscoveryCandidate(
                candidate_id=f"cand:arxiv:{c['arxiv_id']}",
                source_record_id=f"arxivoai:{c['arxiv_id']}",
                priority_score=s,
                score_components=comp,
                in_v010_corpus=c["arxiv_id"] in corpus_ids,
                selection_version=cfg["selection_version"],
                provenance=Provenance(
                    source_name=arxiv_oai.SOURCE,
                    retrieved_at=datetime.fromisoformat(page["retrieved_at"]),
                    processing_version=code_version(),
                    extractor_version=cfg["selection_version"],
                    raw_path=page["body_path"],
                ),
                **c,
            )
        )
    eligible = len(rows)
    rows = cap_layer0(rows, cfg["layer0_max"], cfg["layer0_area_min"])
    rows.sort(key=lambda r: r.arxiv_id)
    promoted = promote(rows, cfg["promote_new_candidates"], cfg["per_area_min"])
    for r in rows:
        r.promoted = r.arxiv_id in promoted
    selection = {
        "selection_version": cfg["selection_version"],
        "created_at": now_utc().isoformat(),
        "criteria": {
            "layer0": f"v1_date >= {min_v1}, arXiv category prefix in {list(prefixes)}, >= 1 priority area (config/domain.yaml)",
            "promotion": f"top {cfg['promote_new_candidates']} new candidates by selection-0.1.0 score, >= {cfg['per_area_min']} per area",
            "score": "areas(<=4) + 0.75*experiment_signals(<=4) + 2*code_url + 1*published_version + 0.5*multiple_versions + 0.5*permissive_licence + 0.5*recent",
        },
        "counts": {"oai_records_seen": seen, "eligible_before_cap": eligible, "layer0_candidates": len(rows),
                   "already_in_corpus": sum(r.in_v010_corpus for r in rows), "promoted": len(promoted)},
        "promoted": [{"arxiv_id": r.arxiv_id, "score": r.priority_score, "areas": r.areas} for r in rows if r.promoted],
    }
    return rows, selection


def cap_layer0(rows: list[DiscoveryCandidate], n: int, area_min: int) -> list[DiscoveryCandidate]:
    """Stratified cap: corpus papers always kept; >= area_min per priority area; then global score order."""
    ranked = sorted(rows, key=lambda r: (-r.priority_score, r.arxiv_id))
    keep: dict[str, DiscoveryCandidate] = {r.arxiv_id: r for r in rows if r.in_v010_corpus}
    for area in sorted(config("domain")["areas"]):
        for r in [r for r in ranked if area in r.areas][:area_min]:
            keep.setdefault(r.arxiv_id, r)
    for r in ranked:
        if len(keep) >= n:
            break
        keep.setdefault(r.arxiv_id, r)
    return list(keep.values())


def promote(rows: list[DiscoveryCandidate], n: int, per_area_min: int) -> set[str]:
    """Stratified top-n among candidates not already in the corpus (deterministic tie-break on arxiv_id)."""
    pool = sorted((r for r in rows if not r.in_v010_corpus), key=lambda r: (-r.priority_score, r.arxiv_id))
    chosen: list[str] = []
    for area in sorted(config("domain")["areas"]):
        for r in [r for r in pool if area in r.areas][:per_area_min]:
            if r.arxiv_id not in chosen:
                chosen.append(r.arxiv_id)
    for r in pool:
        if len(chosen) >= n:
            break
        if r.arxiv_id not in chosen:
            chosen.append(r.arxiv_id)
    return set(chosen)  # may exceed n when per-area minimums require it


def write(rows: list[DiscoveryCandidate], selection: dict[str, Any]) -> dict:
    out = write_jsonl(CANDIDATES, rows)
    atomic_write_bytes(SELECTION, json.dumps(selection, indent=1).encode("utf-8"))
    return out
