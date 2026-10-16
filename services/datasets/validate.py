"""Dataset validation (DATASET_SPEC §46). Critical violations make the build fail loudly.

Checks: schema validity, required/duplicate identifiers, orphan references, invalid enums,
invalid numbers, invalid units, missing provenance, missing source locations, evidence
integrity (re-verified against parsed full text when available), conflicting labels,
duplicate examples, train/test leakage, benchmark contamination, license completeness,
Gold integrity (human review + promotion records), and secret leakage.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from services.canonicalization.normalize import UNITS
from services.datasets import manifests
from services.datasets.schemas import RECORD_MODELS, EvidenceAssertion, ReviewerType, SourceRecord
from services.datasets.store import DATA, atomic_write_bytes, now_utc, read_jsonl, rel
from services.evidence.assertions import load_parsed_docs, verify

SCANNED_DIRS = ("normalized", "silver", "gold", "annotations", "eval/internal")
ID_FIELD = {
    "source_record": "source_record_id", "paper": "paper_id", "study": "study_id", "experiment": "experiment_id",
    "evidence": "evidence_id", "study_link": "link_id", "comparison": "comparison_id",
    "contradiction": "contradiction_id", "abstention": "case_id", "laya_decision": "decision_id",
    "review_decision": "decision_id", "promotion": "promotion_id",
    # 0.2.0
    "result": "result_id", "association": "association_id", "artifact": "artifact_id", "artifact_evidence": "artifact_evidence_id",
    "canonical_entity": "canonical_id", "graph_node": "node_id", "graph_edge": "edge_id", "research_task": "task_id",
    "hard_case": "case_id", "discovery_candidate": "candidate_id",
}
SECRET_PATTERNS = [re.compile(p) for p in (r"\bsk-[A-Za-z0-9]{20,}", r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", r"(?i)api[_-]?key\s*[=:]\s*[A-Za-z0-9._\-]{16,}")]
# files we write (records, reports, run logs, manifests, LLM cache) - searched for the literal value of any configured API key
SECRET_SCAN_DIRS = (*SCANNED_DIRS, "reports", "runs", "manifests", "raw/llm_cache")
SECRET_SCAN_SUFFIXES = {".json", ".jsonl", ".md", ".yaml", ".yml", ".log", ".txt", ".csv"}
KEY_ENV_VARS = ("NEBIUS_API_KEY", "TAVILY_API_KEY", "S2_API_KEY", "CORE_API_KEY", "HF_TOKEN", "UNPAYWALL_EMAIL", "OPENREVIEW_USERNAME",
                "OPENREVIEW_PASSWORD", "OPENALEX_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN")
LOCATION_KEYS = ("page", "section", "paragraph", "table", "figure", "cell", "sentence", "footnote", "artifact_id")
POSITIVE_FIELDS = {"training.learning_rate", "training.batch_size", "training.epochs", "training.steps", "compute.gpu_count", "training.gradient_accumulation"}
UNIT_INTERVAL_FIELDS = {"training.dropout", "training.warmup_ratio", "inference.top_p", "training.weight_decay"}


@dataclass
class Report:
    critical: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    checked: Counter = field(default_factory=Counter)

    def add(self, level: str, check: str, msg: str, rid: str | None = None) -> None:
        (self.critical if level == "critical" else self.warnings).append({"check": check, "message": msg, "record": rid})

    def summary(self) -> dict[str, Any]:
        by_check = lambda items: dict(Counter(i["check"] for i in items))  # noqa: E731
        return {
            "generated_at": now_utc().isoformat(),
            "passed": not self.critical,
            "records_checked": dict(self.checked),
            "critical_count": len(self.critical),
            "warning_count": len(self.warnings),
            "critical_by_check": by_check(self.critical),
            "warnings_by_check": by_check(self.warnings),
            "critical": self.critical[:500],
            "warnings": self.warnings[:500],
        }


def _load(data_root: Path) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for d in SCANNED_DIRS:
        for f in sorted((data_root / d).rglob("*.jsonl")):
            tier_dir = f.relative_to(data_root).parts[0]
            for rec in read_jsonl(f):
                out[rec.get("record_type", "?")].append((tier_dir, rec))
    return out


def _numbers(report: Report, rid: str, path: str, fv: dict) -> None:
    v = fv.get("value")
    unit = fv.get("unit")
    if unit is not None and unit not in UNITS:
        report.add("critical", "invalid_unit", f"{path}: unit {unit!r} not in controlled vocabulary", rid)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(v):
            report.add("critical", "invalid_number", f"{path}: non-finite value {v}", rid)
        elif path in POSITIVE_FIELDS and v <= 0:
            report.add("critical", "invalid_number", f"{path}: must be > 0, got {v}", rid)
        elif path in UNIT_INTERVAL_FIELDS and not 0 <= v <= 1:
            report.add("critical", "invalid_number", f"{path}: must be in [0,1], got {v}", rid)


def _experiment_fields(exp: dict):
    for group, fields in exp.items():
        if isinstance(fields, dict) and group not in ("provenance", "cross_check"):
            for name, fv in fields.items():
                if isinstance(fv, dict) and "status" in fv:
                    yield f"{group}.{name}", fv
    for i, outcome in enumerate(exp.get("outcomes", [])):
        for name, fv in outcome.items():
            yield f"evaluation.{name}", fv


def validate(data_root: Path = DATA, reverify_evidence: bool = True) -> Report:
    report = Report()
    recs = _load(data_root)

    # 1-2. schema validity + identifiers --------------------------------------------------
    ids: dict[str, dict[str, dict]] = defaultdict(dict)  # record_type -> id -> record
    tiers: dict[str, str] = {}
    for rtype, items in recs.items():
        if rtype not in RECORD_MODELS:
            report.add("critical", "schema", f"unknown record_type {rtype!r} ({len(items)} records)")
            continue
        model = RECORD_MODELS[rtype][0]
        idf = ID_FIELD.get(rtype)
        for tier_dir, rec in items:
            report.checked[rtype] += 1
            rid = rec.get(idf) if idf else None
            try:
                model.model_validate(rec)
            except ValidationError as e:
                report.add("critical", "schema", f"{rtype}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}", rid)
            if idf:
                if not rid:
                    report.add("critical", "required_identifier", f"{rtype} missing {idf}")
                    continue
                key = f"{tier_dir}:{rid}"
                if key in tiers:
                    report.add("critical", "duplicate_identifier", f"{rtype} {rid} appears twice in {tier_dir}", rid)
                tiers[key] = rtype
                ids[rtype].setdefault(rid, rec)

    src, papers, studies = ids["source_record"], ids["paper"], ids["study"]
    exps, evs = ids["experiment"], ids["evidence"]

    def ref(check_rtype: str, rid: str, owner: str, what: str, pool: dict) -> None:
        if rid not in pool:
            report.add("critical", "orphan_reference", f"{check_rtype} {owner}: {what} {rid!r} not found", owner)

    # 3. orphan references -----------------------------------------------------------------
    for pid, p in papers.items():
        for s in p.get("source_record_ids", []):
            ref("paper", s, pid, "source_record", src)
        ref("paper", p["study_id"], pid, "study", studies)
    for sid, s in studies.items():
        for p in s.get("paper_ids", []):
            ref("study", p, sid, "paper", papers)
    for _, c in recs.get("citation", []):
        ref("citation", c["citing_paper_id"], c["source_record_id"], "paper", papers)
        ref("citation", c["cited_paper_id"], c["source_record_id"], "paper", papers)
    for eid, e in exps.items():
        ref("experiment", e["study_id"], eid, "study", studies)
        ref("experiment", e["paper_id"], eid, "paper", papers)
        for s in e.get("source_record_ids", []):
            ref("experiment", s, eid, "source_record", src)
        for path_, fv in _experiment_fields(e):
            _numbers(report, eid, path_, fv)
            for ev_id in fv.get("evidence_ids", []):
                ev = evs.get(ev_id)
                if ev is None:
                    report.add("critical", "orphan_reference", f"{path_} cites missing evidence {ev_id}", eid)
                    continue
                if ev["field_name"] != path_:
                    report.add("critical", "evidence_field_mismatch", f"{path_} cites evidence for {ev['field_name']}", eid)
                if ev.get("experiment_id") not in (None, eid):
                    report.add("critical", "evidence_experiment_mismatch", f"{path_} cites evidence of {ev.get('experiment_id')}", eid)
                if ev.get("status") == "REJECTED" and fv.get("status") == "KNOWN":
                    report.add("critical", "rejected_evidence_used", f"{path_} KNOWN on REJECTED evidence {ev_id}", eid)
    arts = ids["artifact"]
    for vid, v in evs.items():
        if v.get("location_kind") == "ARTIFACT_FILE":  # artifact evidence is sourced from a pinned artifact file, not a source record
            ref("evidence", v["source_record_id"], vid, "artifact", arts)
        else:
            ref("evidence", v["source_record_id"], vid, "source_record", src)
        if v.get("experiment_id"):
            ref("evidence", v["experiment_id"], vid, "experiment", exps)
        if v.get("paper_id"):
            ref("evidence", v["paper_id"], vid, "paper", papers)
        if v.get("unit") is not None and v["unit"] not in UNITS:
            report.add("critical", "invalid_unit", f"evidence unit {v['unit']!r}", vid)
        loc = any(v.get(k) is not None for k in LOCATION_KEYS)
        if not loc and not v.get("source_location"):
            report.add("critical", "missing_source_location", "evidence has no location at all", vid)
        elif not loc and not v["field_name"].startswith("paper.") and src.get(v["source_record_id"], {}).get("source_name") == "ARXIV":
            report.add("warning", "missing_source_location", "full-text evidence without section/paragraph/table/page", vid)
    for cid, c in ids["comparison"].items():
        ref("comparison", c["experiment_a"], cid, "experiment", exps)
        ref("comparison", c["experiment_b"], cid, "experiment", exps)
        for fc in c.get("field_comparisons", []):
            for ev_id in fc.get("evidence_a", []) + fc.get("evidence_b", []):
                ref("comparison", ev_id, cid, "evidence", evs)
    for lid, link in ids["study_link"].items():
        ref("study_link", link["record_a"], lid, "paper", papers)
        ref("study_link", link["record_b"], lid, "paper", papers)
    for kid, k in ids["contradiction"].items():
        for ev_id in k.get("evidence_a", []) + k.get("evidence_b", []) + k.get("resolution_evidence", []):
            ref("contradiction", ev_id, kid, "evidence", evs)
    context_pool = {**exps, **ids["comparison"], **papers, **ids["study_link"], **ids["contradiction"]}
    for aid, a in ids["abstention"].items():
        for c in a.get("context_ids", []):
            ref("abstention", c, aid, "context", context_pool)
        for ev_id in a.get("evidence_ids", []):
            ref("abstention", ev_id, aid, "evidence", evs)
    # 0.2.0 record types
    results = ids["result"]
    for rid, r in results.items():
        ref("result", r["paper_id"], rid, "paper", papers)
        ref("result", r["study_id"], rid, "study", studies)
        if r["source_record_id"] not in src and r["source_record_id"] not in arts:
            report.add("warning", "orphan_reference", f"result source {r['source_record_id']!r} is neither a source record nor an artifact", rid)
        for ev_id in r.get("evidence_ids", []):
            ref("result", ev_id, rid, "evidence", evs)
    for aid, a in ids["association"].items():
        ref("association", a["result_id"], aid, "result", results)
        for e in [a.get("experiment_id"), *a.get("candidate_experiment_ids", [])]:
            if e:
                ref("association", e, aid, "experiment", exps)
        for ev_id in a.get("evidence_ids", []):
            ref("association", ev_id, aid, "evidence", evs)
    for aid, a in arts.items():
        for p in a.get("linked_paper_ids", []):
            ref("artifact", p, aid, "paper", papers)
        if a.get("parent_artifact_id"):
            ref("artifact", a["parent_artifact_id"], aid, "artifact", arts)
    for aid, a in ids["artifact_evidence"].items():
        ref("artifact_evidence", a["artifact_id"], aid, "artifact", arts)
        ref("artifact_evidence", a["paper_id"], aid, "paper", papers)
        if a.get("experiment_id"):
            ref("artifact_evidence", a["experiment_id"], aid, "experiment", exps)
        for ev_id in a.get("artifact_evidence_ids", []) + a.get("paper_evidence_ids", []):
            ref("artifact_evidence", ev_id, aid, "evidence", evs)
    for tid, t in ids["research_task"].items():
        for c in t.get("context_ids", []):
            ref("research_task", c, tid, "context", {**context_pool, **ids["abstention"], **results})
        for ev_id in t.get("evidence_ids", []):
            ref("research_task", ev_id, tid, "evidence", evs)
    for hid, h in ids["hard_case"].items():
        pool = exps if h["record_kind"] == "experiment" else papers
        ref("hard_case", h["record_a"], hid, h["record_kind"], pool)
        ref("hard_case", h["record_b"], hid, h["record_kind"], pool)
        for ev_id in h.get("evidence_ids", []):
            ref("hard_case", ev_id, hid, "evidence", evs)
    nodes = ids["graph_node"]
    for gid, g in ids["graph_edge"].items():
        ref("graph_edge", g["source"], gid, "graph_node", nodes)
        ref("graph_edge", g["target"], gid, "graph_node", nodes)

    # 4. provenance ------------------------------------------------------------------------
    for sid, s in src.items():
        prov = s.get("provenance", {})
        for k in ("retrieved_at", "content_hash", "source_url"):
            if not prov.get(k):
                report.add("critical", "missing_provenance", f"source record provenance.{k} missing", sid)

    # 5. evidence integrity: every checked assertion is re-found in its stored source -------------
    if reverify_evidence and evs:
        docs = load_parsed_docs()
        records = {sid: SourceRecord.model_validate(s) for sid, s in src.items()}
        unverifiable = 0
        for vid, v in evs.items():
            if v.get("status") not in ("SOURCE_CHECKED", "HUMAN_REVIEWED"):
                continue
            ok, reason = verify(EvidenceAssertion.model_validate(v), docs, records, arts)  # artifact evidence: cached pinned file
            if ok:
                continue
            if reason == "no_parsed_document":  # e.g. fresh clone without git-ignored full text
                unverifiable += 1
            else:
                report.add("critical", "evidence_not_in_source", f"re-verification failed: {reason}", vid)
        if unverifiable:
            report.add("warning", "evidence_unverifiable", f"{unverifiable} checked full-text evidence records have no local parsed full text to re-verify")

    # 6. conflicting labels / duplicate examples --------------------------------------------
    for rtype, a_key, b_key, label_key in (("comparison", "experiment_a", "experiment_b", "relationship_label"), ("study_link", "record_a", "record_b", "relationship")):
        seen: dict[tuple, set] = defaultdict(set)
        for tier_dir, r in recs.get(rtype, []):
            seen[(tier_dir, *sorted((r[a_key], r[b_key])))].add(r[label_key])
        for key, labels in seen.items():
            if len(labels) > 1:
                report.add("critical", "conflicting_labels", f"{rtype} pair {key[1:]} has labels {sorted(labels)} in {key[0]}")
    for rtype, items in recs.items():
        idf = ID_FIELD.get(rtype)
        fingerprints = Counter(
            json.dumps({k: v for k, v in r.items() if k not in (idf, "provenance")}, sort_keys=True) for _, r in items
        )
        dups = sum(n - 1 for n in fingerprints.values() if n > 1)
        if dups:
            report.add("warning", "duplicate_examples", f"{rtype}: {dups} records identical apart from id/provenance")

    # 7. leakage (spec §28): splits assigned at study level --------------------------------
    study_split = {sid: s.get("split", "UNASSIGNED") for sid, s in studies.items()}
    for pid, p in papers.items():
        if study_split.get(p["study_id"]) not in (None, p.get("split", "UNASSIGNED")):
            report.add("critical", "leakage", f"paper split {p.get('split')} != study split", pid)
    for eid, e in exps.items():
        if study_split.get(e["study_id"]) not in (None, e.get("split", "UNASSIGNED")):
            report.add("critical", "leakage", f"experiment split {e.get('split')} != study split", eid)
    for rtype, a_key, b_key, pool in (("comparison", "experiment_a", "experiment_b", exps), ("study_link", "record_a", "record_b", papers)):
        for rid, r in ids[rtype].items():
            sa, sb = pool.get(r[a_key], {}).get("split"), pool.get(r[b_key], {}).get("split")
            expected = sa if sa == sb else "CROSS_SPLIT"
            if sa and sb and r.get("split", "UNASSIGNED") != expected:
                report.add("critical", "leakage", f"{rtype} split {r.get('split')} but members are {sa}/{sb}", rid)

    # 8. contamination + licences -------------------------------------------------------------
    registry = manifests.sources()
    for sid, s in src.items():
        policy = registry.get(s["source_name"], {}).get("usage_policy")
        if s["source_name"] not in registry:
            report.add("critical", "license_metadata", f"source {s['source_name']} not in sources.yaml", sid)
        elif policy in ("EVALUATION_ONLY", "RESTRICTED"):
            report.add("critical", "benchmark_contamination", f"{policy} source {s['source_name']} in corpus", sid)
    required = ("name", "url", "metadata_license", "content_license", "usage_policy", "training_allowed", "evaluation_allowed",
                "redistribution_allowed", "internal_use_allowed", "attribution_required", "restricted", "verification_status")
    for sid, s in registry.items():
        missing = [k for k in required if k not in s]
        if missing:
            report.add("critical", "license_metadata", f"sources.yaml {sid} missing {missing}")
        if s.get("status") == "ACTIVE" and s.get("verification_status") != "VERIFIED":
            report.add("critical", "license_metadata", f"ACTIVE source {sid} is not VERIFIED")
    for entry in manifests.contamination():
        missing = [k for k in ("dataset_name", "dataset_version", "source_url", "license", "training_allowed", "evaluation_allowed",
                               "redistribution_allowed", "contamination_status", "retrieval_date") if k not in entry]
        if missing:
            report.add("critical", "license_metadata", f"contamination.yaml {entry.get('dataset_name')} missing {missing}")
    for f in (data_root / "fulltext" / "permitted").rglob("provenance.json"):
        meta = json.loads(f.read_text(encoding="utf-8"))
        if manifests.fulltext_class(meta.get("content_license")) != "permitted":
            report.add("critical", "license_error", f"{rel(f)} stored as permitted with licence {meta.get('content_license')}")

    # 9. Gold integrity (spec §25): human decisions + promotion record for every Gold record ----
    promoted = {p["target_id"]: p for p in ids["promotion"].values() if p.get("to_tier") == "GOLD"}
    decisions = ids["review_decision"]
    for rtype, items in recs.items():
        for tier_dir, r in items:
            if tier_dir != "gold":
                continue
            rid = r.get(ID_FIELD.get(rtype, ""), "?")
            if r.get("quality_tier") != "GOLD":
                report.add("critical", "gold_integrity", f"record in gold/ has tier {r.get('quality_tier')}", rid)
            prom = promoted.get(rid)
            human = prom and any(decisions.get(d, {}).get("reviewer_type") == ReviewerType.HUMAN for d in prom["decision_ids"])
            if not human:
                report.add("critical", "gold_integrity", "Gold record without human-reviewed promotion", rid)
    for rid, c in ids["comparison"].items():
        if c.get("relationship_label") == "SAME_EXPERIMENT" and c.get("reviewer_status") not in ("REVIEWED", "ADJUDICATED"):
            report.add("critical", "unverified_same_label", "SAME_EXPERIMENT without human review", rid)
    for rid, link in ids["study_link"].items():
        if link.get("relationship") == "SAME_UNDERLYING_STUDY" and link.get("reviewer_status") not in ("REVIEWED", "ADJUDICATED"):
            report.add("critical", "unverified_same_label", "SAME_UNDERLYING_STUDY without human review", rid)

    # 10. secrets: generic key patterns in records + the literal value of every configured key in everything we write ----
    for d in SCANNED_DIRS:
        for f in (data_root / d).rglob("*.jsonl"):
            text = f.read_text(encoding="utf-8")
            if any(p.search(text) for p in SECRET_PATTERNS):
                report.add("critical", "secret_leak", f"possible secret in {rel(f)}")
    keys = [(n, v.encode()) for n in KEY_ENV_VARS if len(v := os.environ.get(n, "")) >= 8]
    scanned = 0
    for d in SECRET_SCAN_DIRS if keys else ():
        for f in (data_root / d).rglob("*"):
            if f.suffix.lower() in SECRET_SCAN_SUFFIXES and f.is_file():
                scanned += 1
                blob = f.read_bytes()
                for name, value in keys:
                    if value in blob:  # never echo the value, only which variable leaked where
                        report.add("critical", "secret_leak", f"value of {name} found in {rel(f)}")
    report.checked["secret_scan_files"] = scanned
    report.checked["secret_scan_keys"] = len(keys)
    return report


def write_report(report: Report, out: Path = DATA / "reports" / "validation_report.json") -> Path:
    atomic_write_bytes(out, json.dumps(report.summary(), indent=1, sort_keys=True, default=str).encode())
    return out
