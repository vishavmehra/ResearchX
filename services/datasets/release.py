"""Release gates and manifests (DATASET_SPEC §8, §28, §31, §53, §54).

- detect_leakage(): split integrity at the study level, cross-split pairs, temporal split, eval isolation.
- license_audit(): registry/licence completeness, permitted-storage rule, contamination, redistribution
  obligations that a public release must honour (restricted quotes, S2-derived fields, CC BY-SA share-alike).
- build_manifest(): release manifest with checksums of every dataset file, source versions, code hash
  and versions; also regenerates data/manifests/versions.yaml and datasets.yaml from code + data.
  A frozen version is immutable: re-freezing with different content fails (bump dataset_version).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import yaml

from services.datasets import manifests
from services.datasets.schemas import RECORD_MODELS, SCHEMA_VERSION
from services.datasets.store import DATA, ROOT, atomic_write_bytes, code_version, config, now_utc, path, read_jsonl, rel, sha256_file

MANIFEST_DIRS = ("normalized", "silver", "gold", "annotations")


def _versions() -> dict[str, str]:
    from services.canonicalization import entities, normalize
    from services.comparison import conflicts, fields, links
    from services.datasets import depth, graph, review, tasks
    from services.evidence import assertions
    from services.extraction import associate, deep, experiments, llm, parse, rules
    from services.retrieval import corpus
    from services.retrieval.sources import arxiv, arxiv_oai, github_artifacts, openalex, papers_with_code, semantic_scholar

    return {
        "schema": SCHEMA_VERSION, "normalization": normalize.NORMALIZATION_VERSION,
        "openalex_normalizer": openalex.NORMALIZER_VERSION, "semantic_scholar_normalizer": semantic_scholar.NORMALIZER_VERSION,
        "arxiv_normalizer": arxiv.NORMALIZER_VERSION, "papers_with_code_normalizer": papers_with_code.NORMALIZER_VERSION,
        "merge": corpus.MERGE_VERSION, "relevance": corpus.RELEVANCE_VERSION, "parser": parse.PARSER_VERSION,
        "rules_extractor": rules.RULES_VERSION, "pwc_import": experiments.IMPORT_VERSION, "table_crosscheck": experiments.CROSSCHECK_VERSION,
        "llm_extractor": llm.EXTRACTOR_VERSION, "llm_prompt": llm.PROMPT_VERSION, "evidence": assertions.EVIDENCE_VERSION,
        "comparison_labeler": fields.LABELER_VERSION, "study_link_rules": links.LINK_VERSION, "contradictions": conflicts.CONFLICT_VERSION,
        "abstention": conflicts.ABSTENTION_VERSION, "review_policy": review.POLICY_VERSION, "annotation_guidelines": review.GUIDELINE_VERSION,
        # schema 0.2.0 (deepening)
        "arxiv_oai_normalizer": arxiv_oai.NORMALIZER_VERSION, "discovery_selection": config("pipeline")["discovery"]["selection_version"],
        "deep_extractor": deep.EXTRACTOR_VERSION, "deep_prompt_settings": deep.PROMPT_SETTINGS, "deep_prompt_experiments": deep.PROMPT_EXPERIMENTS,
        "association": associate.ASSOC_VERSION, "artifact_linkage": github_artifacts.VERSION, "depth": depth.DEPTH_VERSION,
        "quality": depth.QUALITY_VERSION, "entities": entities.ENTITY_VERSION, "graph": graph.GRAPH_VERSION, "research_tasks": tasks.GENERATOR_VERSION,
    }


# --------------------------------------------------------------------------- leakage


def detect_leakage() -> dict[str, Any]:
    critical: list[str] = []
    warnings: list[str] = []
    papers = {p["paper_id"]: p for p in read_jsonl(path("papers"))}
    studies = {s["study_id"]: s for s in read_jsonl(path("studies"))}
    exps = {e["experiment_id"]: e for e in read_jsonl(path("experiments"))}
    cutoff = config("pipeline")["splits"]["temporal_cutoff"]
    for pid, p in papers.items():
        s = studies.get(p["study_id"])
        if s and s.get("split") != p.get("split"):
            critical.append(f"{pid}: paper split {p.get('split')} != study split {s.get('split')}")
        date = p.get("publication_date") or ""
        if p.get("split") == "TEMPORAL_TEST" and date and date < cutoff:
            critical.append(f"{pid}: TEMPORAL_TEST but published {date} < {cutoff}")
        if p.get("split") not in ("TEMPORAL_TEST", "UNASSIGNED") and date >= cutoff:
            critical.append(f"{pid}: published {date} >= {cutoff} but split {p.get('split')}")
    for eid, e in exps.items():
        if papers.get(e["paper_id"], {}).get("split") != e.get("split"):
            critical.append(f"{eid}: experiment split differs from its paper")
    for key, a, b, pool in (("comparisons", "experiment_a", "experiment_b", exps), ("hard_negatives", "experiment_a", "experiment_b", exps),
                            ("study_links", "record_a", "record_b", papers)):
        for r in read_jsonl(path(key)):
            sa, sb = pool.get(r[a], {}).get("split"), pool.get(r[b], {}).get("split")
            if sa and sb and sa != sb and r.get("split") != "CROSS_SPLIT":
                critical.append(f"{key} {r[a]}~{r[b]} spans {sa}/{sb} but is labelled {r.get('split')}")
    # the same title in two splits may be one study in two records -> leakage risk until linked or merged
    by_title: dict[str, list[dict]] = defaultdict(list)
    for p in papers.values():
        if p.get("normalized_title"):
            by_title[p["normalized_title"]].append(p)
    for title, group in by_title.items():
        if len({g.get("split") for g in group}) > 1:
            warnings.append(f"same normalized title across splits: {[g['paper_id'] + '/' + str(g.get('split')) for g in group]}")
    eval_ids = {r.get("paper_id") for f in (DATA / "eval").rglob("*.jsonl") for r in read_jsonl(f)}
    train_ids = {p for p, v in papers.items() if v.get("split") == "TRAIN"}
    for pid in sorted(eval_ids & train_ids):
        critical.append(f"{pid}: appears in data/eval and in the TRAIN split")
    return {
        "generated_at": now_utc().isoformat(),
        "passed": not critical,
        "critical": critical[:500],
        "warnings": warnings[:500],
        "split_distribution": dict(Counter(p.get("split") for p in papers.values())),
        "cross_split_pairs": sum(1 for k in ("comparisons", "hard_negatives", "study_links") for r in read_jsonl(path(k)) if r.get("split") == "CROSS_SPLIT"),
    }


# --------------------------------------------------------------------------- licences / contamination


def license_audit() -> dict[str, Any]:
    critical: list[str] = []
    warnings: list[str] = []
    registry, lics = manifests.sources(), manifests.licenses()
    contaminated = {e["dataset_name"] for e in manifests.contamination()}
    src = read_jsonl(path("source_records"))
    used_sources = Counter(s["source_name"] for s in src)
    for name, n in used_sources.items():
        entry = registry.get(name)
        if not entry:
            critical.append(f"source {name} ({n} records) missing from sources.yaml")
            continue
        if entry.get("status") != "ACTIVE" or entry.get("verification_status") != "VERIFIED":
            critical.append(f"source {name} used but not ACTIVE+VERIFIED")
        if entry.get("usage_policy") in ("EVALUATION_ONLY", "RESTRICTED"):
            critical.append(f"{entry['usage_policy']} source {name} present in the corpus")
        if name not in contaminated:
            critical.append(f"source {name} missing from contamination.yaml")
    referenced = Counter()
    for s in src:
        for key in ("content_license", "metadata_license"):
            if lic := (s.get("license") or {}).get(key):
                referenced[lic] += 1
    for lic in referenced:
        if lic not in lics:
            critical.append(f"licence id {lic!r} used by records but not defined in licenses.yaml")
    storage = Counter()
    for row in read_jsonl(DATA / "normalized" / "sources" / "fulltext_index.jsonl"):
        lic = row["license"].get("content_license")
        klass = manifests.fulltext_class(lic)
        storage[(klass, lic)] += 1
        expected = "PERMITTED" if klass == "permitted" else "RESTRICTED"
        if row["fulltext"]["license_status"] != expected or f"/fulltext/{klass}/" not in "/" + row["fulltext"]["path"]:
            critical.append(f"{row['arxiv_id']}: stored as {row['fulltext']['license_status']} at {row['fulltext']['path']} but licence {lic} -> {klass}")
    ev_files = sorted((DATA / "silver" / "evidence").glob("*.jsonl")) + sorted((DATA / "gold" / "evidence").glob("*.jsonl"))
    quotes = Counter()
    for f in ev_files:
        for e in read_jsonl(f):
            quotes["redistributable" if e.get("redistribution_allowed") is True else "not_redistributable_or_unknown"] += 1
    if quotes["not_redistributable_or_unknown"]:
        warnings.append(f"{quotes['not_redistributable_or_unknown']} evidence quotes come from sources that do not allow redistribution: "
                        "a public release must drop or truncate their evidence_text (keep ids and locations).")
    papers = read_jsonl(path("papers"))
    s2_fields = Counter(f for p in papers for f, s in (p.get("field_sources") or {}).items() if s.startswith("s2:"))
    if s2_fields:
        warnings.append(f"paper fields sourced from Semantic Scholar (API licence: no redistribution): {dict(s2_fields)}; "
                        "a public release must drop or re-source them.")
    pwc = used_sources.get("PAPERS_WITH_CODE", 0)
    if pwc:
        warnings.append(f"{pwc} PwC records (CC BY-SA 4.0): redistributed derivatives must carry attribution and share-alike.")
    return {
        "generated_at": now_utc().isoformat(),
        "passed": not critical,
        "critical": critical,
        "warnings": warnings,
        "sources_used": dict(used_sources),
        "usage_policies": {n: registry.get(n, {}).get("usage_policy") for n in used_sources},
        "licences_referenced": dict(referenced),
        "fulltext_storage": {f"{k}:{l}": n for (k, l), n in sorted(storage.items(), key=str)},
        "evidence_quote_redistribution": dict(quotes),
    }


# --------------------------------------------------------------------------- manifest


def _dataset_files() -> list[dict[str, Any]]:
    out = []
    for d in MANIFEST_DIRS:
        for f in sorted((DATA / d).rglob("*.jsonl")):
            out.append({"path": rel(f), "sha256": sha256_file(f), "bytes": f.stat().st_size, "records": len(read_jsonl(f))})
    for f in sorted((DATA / "schemas").glob("*.json")) + sorted((DATA / "manifests").glob("*.yaml")):
        out.append({"path": rel(f), "sha256": sha256_file(f), "bytes": f.stat().st_size})
    return out


def _source_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in ("openalex", "semantic_scholar", "arxiv", "papers_with_code"):
        p = DATA / "raw" / d / "acquisition.json"
        if not p.exists():
            continue
        acq = json.loads(p.read_text(encoding="utf-8"))
        if d == "papers_with_code":
            out[d] = {k: {"repo": v["repo"], "revision": v["revision"], "files": [{"url": f["url"], "sha256": f["sha256"]} for f in v["files"]]}
                      for k, v in acq["datasets"].items()}
            continue
        pages = [pg for q in acq.get("queries", []) for pg in q["pages"]] if d == "openalex" else acq.get("requests", [])
        dates = sorted(pg["retrieved_at"] for pg in pages)
        out[d] = {"acquisition_run": acq.get("run_id"), "requests": len(pages), "retrieved_from": dates[0] if dates else None,
                  "retrieved_to": dates[-1] if dates else None, "payload_sha256": sorted(pg["sha256"] for pg in pages)}
        if d == "openalex":
            out[d]["queries"] = [{"area": q["area"], "mode": q.get("mode"), "filter": q["filter"], "sort": q["sort"], "limit": q["limit"]} for q in acq["queries"]]
        if d == "semantic_scholar" and acq.get("reference_requests"):
            out[d]["reference_requests"] = len(acq["reference_requests"])
            out[d]["reference_payload_sha256"] = sorted(r["sha256"] for r in acq["reference_requests"])
    oai = DATA / "raw" / "arxiv_oai" / "acquisition.json"
    if oai.exists():
        acq = json.loads(oai.read_text(encoding="utf-8"))
        dates = sorted(pg["retrieved_at"] for pg in acq["pages"])
        out["arxiv_oai_pmh"] = {"acquisition_run": acq.get("run_id"), "sets": acq.get("sets"), "windows": len(acq.get("windows", [])),
                                "pages": len(acq["pages"]), "records": sum(pg.get("records", 0) for pg in acq["pages"]),
                                "retrieved_from": dates[0] if dates else None, "retrieved_to": dates[-1] if dates else None,
                                "payload_sha256": sorted(pg["sha256"] for pg in acq["pages"])}
    pins = [json.loads(p.read_text(encoding="utf-8")) for p in (DATA / "raw" / "github").glob("*/*/snapshot.json")]
    if pins:
        out["github"] = {"pinned_snapshots": len(pins), "pins": sorted(f"{p['repository']}@{p['commit']}" for p in pins)}
    return out


def _read_report(name: str) -> dict[str, Any]:
    p = DATA / "reports" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_generated_catalogs(counts: dict[str, int], files: list[dict[str, Any]]) -> None:
    versions = {"dataset_version": config("pipeline")["dataset_version"], "processing_version": code_version(), **_versions()}
    atomic_write_bytes(DATA / "manifests" / "versions.yaml",
                       ("# GENERATED by scripts/data/build_manifest.py from code constants. Do not edit.\n" + yaml.safe_dump(versions, sort_keys=True)).encode())
    tiers = {"normalized": "BRONZE", "silver": "SILVER", "gold": "GOLD", "annotations": "ANNOTATION"}
    by_dir = {"experiments": "experiment", "evidence": "evidence", "comparisons": "comparison", "study_links": "study_link",
              "contradictions": "contradiction", "abstention": "abstention", "promotions": "promotion", "decisions": "review_decision"}
    catalog = []
    for f in files:
        if not f["path"].endswith(".jsonl"):
            continue
        first = read_jsonl(ROOT / f["path"])[:1]
        rtype = first[0].get("record_type") if first else by_dir.get(f["path"].split("/")[-2])  # empty files: type from directory
        stem = RECORD_MODELS.get(rtype, (None, None))[1]
        catalog.append({"path": f["path"], "tier": tiers.get(f["path"].split("/")[1], "?"), "record_type": rtype, "records": f["records"],
                        "schema": f"data/schemas/{stem}.json" if stem else None})
    atomic_write_bytes(DATA / "manifests" / "datasets.yaml",
                       ("# GENERATED by scripts/data/build_manifest.py. Silver = model/rule-assisted (not ground truth);\n"
                        "# Gold = human-verified only. Training use requires the licence review in data/reports/license_report.json.\n"
                        + yaml.safe_dump({"dataset_version": versions["dataset_version"], "datasets": catalog}, sort_keys=False)).encode())


def build_manifest(freeze: bool = False) -> tuple[str, dict[str, Any]]:
    from services.datasets import stats as stats_mod

    version = config("pipeline")["dataset_version"]
    stats = stats_mod.generate()
    files = _dataset_files()
    write_generated_catalogs(stats["counts"], files)
    files = _dataset_files()  # include the regenerated catalogs
    license = _read_report("license_report.json")
    leakage = _read_report("leakage_report.json")
    validation = _read_report("validation_report.json")
    gold = sum(v.get("gold", 0) for k, v in stats["tier_distribution"].items() if isinstance(v, dict))
    manifest = {
        "dataset_version": version,
        "created_at": now_utc().isoformat(),
        "status": "SILVER_ONLY: no Gold records (Gold requires human review)" if not gold else "CONTAINS_GOLD",
        "schema_version": SCHEMA_VERSION,
        "processing_commit": code_version(),
        "processing_commit_note": "repository is not under git; processing_commit is a sha256 over services/, scripts/data/ and config/",
        "extractor_versions": _versions(),
        "annotation_versions": {"guidelines": _versions()["annotation_guidelines"], "review_policy": _versions()["review_policy"]},
        "source_versions": _source_versions(),
        "source_urls": {k: v.get("url") for k, v in manifests.sources().items() if v.get("status") == "ACTIVE"},
        "record_counts": stats["counts"],
        "checksums": files,
        "licensing_status": {"passed": license.get("passed"), "warnings": license.get("warnings", []), "sources_used": license.get("sources_used")},
        "contamination_status": {"passed": license.get("passed"), "evaluation_only_sources_in_corpus": 0 if license.get("passed") else None},
        "leakage_status": {"passed": leakage.get("passed"), "split_distribution": leakage.get("split_distribution")},
        "validation_status": {"passed": validation.get("passed"), "critical": validation.get("critical_count"), "warnings": validation.get("warning_count")},
    }
    releases = DATA / "manifests" / "releases"
    out = releases / (f"v{version}.json" if freeze else f"v{version}-draft.json")
    if freeze and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        same = all(old.get(k) == manifest[k] for k in ("checksums", "record_counts", "processing_commit"))
        if not same:
            raise SystemExit(f"{rel(out)} is frozen and differs: dataset versions are immutable; bump dataset_version in config/pipeline.yaml")
        return rel(out), old
    atomic_write_bytes(out, json.dumps(manifest, indent=1, sort_keys=False, default=str).encode())
    return rel(out), manifest
