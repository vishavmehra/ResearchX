"""Human verification, adjudication and Gold promotion (DATASET_SPEC §25, §26, §27, §45).

Silver -> review queue -> ReviewDecision records (append-only; disagreements never erased)
-> deterministic promotion (review-policy-0.1.0):
- Only HUMAN decisions count. AGENT decisions are kept for audit and never promote anything.
- Standard records (experiment, evidence): >= 1 human ACCEPT/CORRECT and no human REJECT.
- High-impact records (comparison, study_link, contradiction, abstention): >= 2 distinct human
  annotators agree, or an ADJUDICATOR decides.
- ACCEPT vs REJECT (or conflicting corrections) without adjudication -> DISPUTED (not promoted).
- A promoted record's cited evidence is promoted with it: the reviewer judged it in context.
- Gold files and promotion records are regenerated from the decision log on every run, so Gold
  is always a pure function of recorded human decisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable

from services.datasets.schemas import (
    AbstentionCase,
    ArtifactEvidence,
    Comparison,
    Contradiction,
    EvidenceAssertion,
    Experiment,
    ExperimentResultAssociation,
    PromotionRecord,
    QualityTier,
    ReviewDecision,
    ReviewDecisionValue as D,
    ReviewerRole,
    ReviewerType,
    StudyLink,
)
from services.datasets.store import DATA, append_jsonl, atomic_write_bytes, now_utc, path, read_jsonl, read_models, stable_id, write_jsonl

POLICY_VERSION = "review-policy-0.1.0"
GUIDELINE_VERSION = "guidelines-0.1.0"
HIGH_IMPACT = {"comparison", "study_link", "contradiction", "abstention"}
QUEUE_DIR = DATA / "annotations" / "queue"
TARGETS: dict[str, tuple[list[str], str, str, type]] = {
    "experiment": (["experiments"], "experiment_id", "gold/experiments/experiments.jsonl", Experiment),
    "evidence": (["experiment_evidence", "rules_evidence", "metadata_evidence", "deep_evidence", "artifact_field_evidence"], "evidence_id",
                 "gold/evidence/evidence.jsonl", EvidenceAssertion),
    "association": (["associations"], "association_id", "gold/results/associations.jsonl", ExperimentResultAssociation),
    "artifact_evidence": (["artifact_evidence"], "artifact_evidence_id", "gold/artifacts/artifact_evidence.jsonl", ArtifactEvidence),
    "comparison": (["comparisons", "hard_negatives"], "comparison_id", "gold/comparisons/comparisons.jsonl", Comparison),
    "study_link": (["study_links"], "link_id", "gold/study_links/study_links.jsonl", StudyLink),
    "contradiction": (["contradictions"], "contradiction_id", "gold/contradictions/contradictions.jsonl", Contradiction),
    "abstention": (["abstention"], "case_id", "gold/abstention/abstention.jsonl", AbstentionCase),
}
QUESTIONS = {  # DATASET_SPEC §45, per target type
    "experiment": ["What exactly does the source say?", "Is every KNOWN field supported by the cited evidence?",
                   "Is any value derived or inferred rather than stated?", "Does it describe one experimental condition?"],
    "evidence": ["Does the quoted text state this value for this field?", "Is the value stated directly (not derived)?"],
    "comparison": ["Is the relationship label correct given the field comparisons?", "Which fields differ materially?",
                   "Is any UNKNOWN field actually stated in the sources?", "Is there enough evidence to decide?"],
    "study_link": ["Are these records the same underlying study, a follow-up, or distinct studies?", "What evidence supports it?"],
    "contradiction": ["Do the two sources really make incompatible claims about the same thing?", "Is a resolution supported by evidence?"],
    "abstention": ["Is the expected behavior (answer / abstain / search more / escalate) correct for this evidence?"],
    "association": ["Does this result (table cell) belong to this experiment?",
                    "Do the cell's row and column headers match the experiment's model, data and metric?"],
    "artifact_evidence": ["Does the repository value refer to the same setting and experiment as the paper statement?",
                          "Is the relation (confirmed / conflicting / related / present only) correct?"],
}
# v0.2.0 active-learning strata, highest priority first: machine signals that disagree, then high-value records.
EXPERIMENT_STRATA = ("ARTIFACT_CONFLICT", "ASSOCIATION_UNCERTAIN", "CONFLICT_CANDIDATE", "DEEP_CONFIGURATION", "FOUND_ALIGNED", "OTHER")


def load_silver(target_type: str) -> dict[str, dict]:
    keys, id_field, _, _ = TARGETS[target_type]
    return {r[id_field]: r for k in keys for r in read_jsonl(path(k))}


def cited_evidence(obj: Any) -> set[str]:
    """Every evidence id referenced anywhere inside a record."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("evidence_ids", "evidence", "evidence_a", "evidence_b", "resolution_evidence", "artifact_evidence_ids",
                     "paper_evidence_ids") and isinstance(v, list):
                out.update(x for x in v if isinstance(x, str) and x.startswith("ev:"))
            else:
                out |= cited_evidence(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= cited_evidence(v)
    return out


def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _round_robin(groups: dict[str, list[dict]], n: int, key: str, order: tuple[str, ...] = ()) -> list[dict]:
    """One pick per stratum per round; strata listed in `order` pick first, the rest alphabetically."""
    rank = {g: i for i, g in enumerate(order)}
    queues = {g: sorted(items, key=lambda r: _h(key, g, json.dumps(r, sort_keys=True)[:200]))
              for g, items in sorted(groups.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0]))}
    out: list[dict] = []
    while len(out) < n and any(queues.values()):
        for g in list(queues):
            if queues[g] and len(out) < n:
                out.append(queues[g].pop(0))
    return out


def build_queue(batch_id: str, quotas: dict[str, int]) -> list[dict]:
    """Active-learning priority (spec §27): conflicts, hard negatives, corroborated/contested experiments."""
    evidence = load_silver("evidence")
    picks: dict[str, list[tuple[dict, str]]] = {}
    con = load_silver("contradiction").values()
    picks["contradiction"] = [(r, f"{r['conflict_type']} contradiction") for r in _round_robin(
        {t: [r for r in con if r["conflict_type"] == t] for t in {r["conflict_type"] for r in con}}, quotas.get("contradiction", 0), batch_id)]
    exps = list(load_silver("experiment").values())

    def cc_status(e: dict) -> str:
        sts = {o.get("status") for o in (e.get("cross_check") or {}).get("outcomes", [])}
        return "CONFLICT_CANDIDATE" if "CONFLICT_CANDIDATE" in sts else "FOUND_ALIGNED" if "FOUND_ALIGNED" in sts else "OTHER"

    art_rel = read_jsonl(path("artifact_evidence"))
    assocs = read_jsonl(path("associations"))
    art_conflict = {r.get("experiment_id") for r in art_rel if r["relation"] == "CONFLICTING"}
    assoc_uncertain = {i for a in assocs if a["status"] in ("CONFLICTING", "AMBIGUOUS") for i in [a.get("experiment_id"), *a.get("candidate_experiment_ids", [])] if i}

    def stratum(e: dict) -> str:
        if e["experiment_id"] in art_conflict:
            return "ARTIFACT_CONFLICT"
        if e["experiment_id"] in assoc_uncertain:
            return "ASSOCIATION_UNCERTAIN"
        cc = cc_status(e)
        if cc == "CONFLICT_CANDIDATE":
            return cc
        return "DEEP_CONFIGURATION" if (e.get("depth") or {}).get("tier") == "deep_configuration" else cc

    picks["experiment"] = [(r, f"stratum {stratum(r)}") for r in _round_robin(
        {s: [e for e in exps if stratum(e) == s] for s in EXPERIMENT_STRATA}, quotas.get("experiment", 0), batch_id, EXPERIMENT_STRATA)]
    picks["association"] = [(r, f"{r['status']} association") for r in _round_robin(
        {s: [a for a in assocs if a["status"] == s] for s in ("CONFLICTING", "AMBIGUOUS", "LIKELY", "VERIFIED")}, quotas.get("association", 0), batch_id,
        ("CONFLICTING", "AMBIGUOUS", "LIKELY", "VERIFIED"))]
    picks["artifact_evidence"] = [(r, f"{r['relation']} artifact relation") for r in _round_robin(
        {s: [a for a in art_rel if a["relation"] == s] for s in {a["relation"] for a in art_rel}}, quotas.get("artifact_evidence", 0), batch_id,
        ("CONFLICTING", "CONFIRMED_BY_BOTH"))]
    comps = list(load_silver("comparison").values())
    picks["comparison"] = [(r, f"hard negative {r.get('hard_negative_type')} / {r['relationship_label']}") for r in _round_robin(
        {t or "none": [c for c in comps if c.get("hard_negative_type") == t] for t in {c.get("hard_negative_type") for c in comps}}, quotas.get("comparison", 0), batch_id)]
    links = list(load_silver("study_link").values())
    picks["study_link"] = [(r, f"{r['relationship']} link candidate") for r in _round_robin(
        {l: [x for x in links if x["relationship"] == l] for l in {x["relationship"] for x in links}}, quotas.get("study_link", 0), batch_id)]
    absts = list(load_silver("abstention").values())
    picks["abstention"] = [(r, f"{r['evidence_class']} -> {r['expected_behavior']}") for r in _round_robin(
        {c: [a for a in absts if a["evidence_class"] == c] for c in {a["evidence_class"] for a in absts}}, quotas.get("abstention", 0), batch_id)]
    machine = [e for e in evidence.values() if e["extraction_model"].startswith("rules-") or "llm-extract" in e["extraction_model"]]
    picks["evidence"] = [(r, f"{'LLM' if 'llm-extract' in r['extraction_model'] else 'rule'}-extracted {r['field_name']}") for r in _round_robin(
        {f: [e for e in machine if e["field_name"] == f] for f in {e["field_name"] for e in machine}}, quotas.get("evidence", 0), batch_id)]

    items = []
    for ttype, chosen in picks.items():
        id_field = TARGETS[ttype][1]
        for rec, reason in chosen:
            evs = [evidence[i] for i in sorted(cited_evidence(rec) | ({rec[id_field]} if ttype == "evidence" else set())) if i in evidence]
            items.append({
                "record_type": "review_item",
                "item_id": stable_id("item", batch_id, ttype, rec[id_field]),
                "batch_id": batch_id,
                "target_type": ttype,
                "target_id": rec[id_field],
                "priority_reason": reason,
                "guideline_version": GUIDELINE_VERSION,
                "questions": QUESTIONS[ttype],
                "record": rec,
                "evidence": evs,
            })
    return items


def write_queue(batch_id: str, items: list[dict]) -> tuple[dict, str]:
    info = write_jsonl(QUEUE_DIR / f"{batch_id}.jsonl", items)
    lines = [f"# Review batch {batch_id}", "", f"Guidelines: data/annotations/GUIDELINES.md ({GUIDELINE_VERSION}). "
             "Record decisions with scripts/data/record_review.py; never edit data files by hand.", ""]
    for it in items:
        rec = it["record"]
        title = rec.get("description") or rec.get("claim") or rec.get("question") or rec.get("rationale") or rec.get("field_name")
        lines += [f"## [{it['target_type']}] {it['target_id']}", f"- Why selected: {it['priority_reason']}", f"- Summary: {str(title)[:300]}"]
        for key in ("relationship_label", "relationship", "conflict_type", "value_a", "value_b", "evidence_class", "expected_behavior", "extracted_value"):
            if key in rec:
                lines.append(f"- {key}: {rec[key]}")
        for ev in it["evidence"][:8]:
            lines.append(f"  - evidence `{ev['evidence_id']}` [{ev['status']}] {ev['field_name']} = {ev.get('extracted_value')!r}: "
                         f"\"{ev['evidence_text'][:240]}\" ({ev.get('source_location') or ''})")
        lines += [f"- Record: `uv run python scripts/data/record_review.py --reviewer <you> --target-type {it['target_type']} --target-id {it['target_id']} --decision ACCEPT|REJECT|CORRECT|UNSURE`", ""]
    md = QUEUE_DIR / f"{batch_id}.md"
    atomic_write_bytes(md, "\n".join(lines).encode("utf-8"))
    return info, str(md)


def record_decision(
    target_type: str, target_id: str, reviewer_id: str, decision: D, *, reviewer_type: ReviewerType = ReviewerType.HUMAN,
    role: ReviewerRole = ReviewerRole.ANNOTATOR, corrected_field: str | None = None, corrected_value: Any = None, comment: str | None = None,
) -> ReviewDecision:
    if target_type not in TARGETS:
        raise ValueError(f"unknown target_type {target_type!r}")
    if target_id not in load_silver(target_type):
        raise ValueError(f"{target_type} {target_id!r} not found in Silver")
    if not reviewer_id.strip():
        raise ValueError("reviewer_id required")
    ts = now_utc()
    d = ReviewDecision(
        decision_id=stable_id("dec", target_type, target_id, reviewer_id, ts.isoformat(), decision.value),
        target_type=target_type, target_id=target_id, reviewer_id=reviewer_id.strip(), reviewer_type=reviewer_type,
        reviewer_role=role, decision=decision, corrected_field=corrected_field, corrected_value=corrected_value,
        comment=comment, guideline_version=GUIDELINE_VERSION, timestamp=ts,
    )
    append_jsonl(path("decisions"), d)
    return d


def _set_path(rec: dict, dotted: str, value: Any) -> None:
    cur = rec
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _outcome(target_type: str, ds: list[ReviewDecision]) -> tuple[str, ReviewDecision | None, list[ReviewDecision]]:
    """-> (status, deciding decision, supporting decisions). status: PROMOTE | REJECTED | DISPUTED | PENDING."""
    human = [d for d in ds if d.reviewer_type is ReviewerType.HUMAN and d.decision is not D.UNSURE]
    if not human:
        return "PENDING", None, []
    adjudications = sorted((d for d in human if d.reviewer_role is ReviewerRole.ADJUDICATOR), key=lambda d: d.timestamp)
    if adjudications:
        final = adjudications[-1]
        return ("REJECTED" if final.decision is D.REJECT else "PROMOTE"), final, [final]
    latest: dict[str, ReviewDecision] = {}
    for d in sorted(human, key=lambda d: d.timestamp):
        latest[d.reviewer_id] = d
    votes = list(latest.values())
    kinds = {d.decision for d in votes}
    corrections = {json.dumps([d.corrected_field, d.corrected_value], sort_keys=True, default=str) for d in votes if d.decision is D.CORRECT}
    if (D.REJECT in kinds and kinds & {D.ACCEPT, D.CORRECT}) or len(corrections) > 1 or (corrections and D.ACCEPT in kinds):
        return "DISPUTED", None, votes
    if kinds == {D.REJECT}:
        return ("REJECTED" if len(votes) >= (2 if target_type in HIGH_IMPACT else 1) else "PENDING"), votes[-1], votes
    if len(votes) < (2 if target_type in HIGH_IMPACT else 1):
        return "PENDING", None, votes
    return "PROMOTE", votes[-1], votes


def promote() -> dict[str, Any]:
    decisions = read_models(path("decisions"), ReviewDecision)
    by_target: dict[tuple[str, str], list[ReviewDecision]] = defaultdict(list)
    for d in decisions:
        by_target[(d.target_type, d.target_id)].append(d)
    silver = {t: load_silver(t) for t in TARGETS}
    gold: dict[str, dict[str, dict]] = {t: {} for t in TARGETS}
    promotions: list[PromotionRecord] = []
    counts: dict[str, int] = defaultdict(int)
    counts["agent_decisions_ignored_for_gold"] = sum(d.reviewer_type is ReviewerType.AGENT for d in decisions)

    def add_gold(ttype: str, rid: str, final: ReviewDecision, support: list[ReviewDecision], reason: str) -> None:
        src = silver[ttype].get(rid)
        if src is None or rid in gold[ttype]:
            return
        rec = copy.deepcopy(src)
        if final.decision is D.CORRECT and final.corrected_field:
            _set_path(rec, final.corrected_field, final.corrected_value)
        rec["quality_tier"] = QualityTier.GOLD.value
        status = "ADJUDICATED" if final.reviewer_role is ReviewerRole.ADJUDICATOR else "REVIEWED"
        if "reviewer_status" in TARGETS[ttype][3].model_fields:
            rec["reviewer_status"] = status
        rec.setdefault("provenance", {})["annotator_status"] = status
        if ttype == "evidence":
            rec["status"] = "HUMAN_REVIEWED"
        TARGETS[ttype][3].model_validate(rec)  # Gold must satisfy the same contract
        gold[ttype][rid] = rec
        promotions.append(PromotionRecord(
            promotion_id=stable_id("prom", ttype, rid), target_type=ttype, target_id=rid, from_tier=QualityTier(src["quality_tier"]),
            to_tier=QualityTier.GOLD, decision_ids=[d.decision_id for d in support], promoted_at=max(d.timestamp for d in support),
            promoted_by=POLICY_VERSION, reason=reason,
        ))
        counts[f"promoted_{ttype}"] += 1

    for (ttype, rid), ds in sorted(by_target.items()):
        status, final, support = _outcome(ttype, ds)
        counts[f"{status.lower()}_{ttype}"] += 1
        if status != "PROMOTE" or final is None:
            continue
        add_gold(ttype, rid, final, support, f"{POLICY_VERSION}: {final.decision.value} by {len(support)} human reviewer(s)")
        for ev_id in sorted(cited_evidence(silver[ttype].get(rid, {}))):
            add_gold("evidence", ev_id, final, support, f"{POLICY_VERSION}: evidence reviewed in context of {ttype} {rid}")

    for ttype, (_, _, gold_rel, _) in TARGETS.items():
        write_jsonl(DATA / gold_rel, [gold[ttype][k] for k in sorted(gold[ttype])])
    write_jsonl(path("promotions"), sorted(promotions, key=lambda p: p.promotion_id))
    counts["gold_records"] = sum(len(v) for v in gold.values())
    return dict(counts)


def iter_queue(batch_id: str) -> Iterable[dict]:
    yield from read_jsonl(QUEUE_DIR / f"{batch_id}.jsonl")


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two annotators' labels on the same items; None when undefined (no items, or chance = 1)."""
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    labels = {x for p in pairs for x in p}
    pe = sum((sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n) for lab in labels)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def agreement_report() -> dict[str, Any]:
    """Inter-annotator agreement and annotation time from HUMAN annotator decisions only (never agents, never adjudicators)."""
    latest: dict[tuple[str, str, str], ReviewDecision] = {}
    for d in sorted(read_models(path("decisions"), ReviewDecision), key=lambda d: d.timestamp):
        if d.reviewer_type is ReviewerType.HUMAN and d.reviewer_role is ReviewerRole.ANNOTATOR:
            latest[(d.target_type, d.target_id, d.reviewer_id)] = d
    by_target: dict[tuple[str, str], dict[str, ReviewDecision]] = defaultdict(dict)
    for (ttype, tid, rid), d in latest.items():
        by_target[(ttype, tid)][rid] = d
    out: dict[str, Any] = {"policy_version": POLICY_VERSION, "human_annotator_decisions": len(latest), "by_target_type": {}}
    for ttype in TARGETS:
        items = {tid: votes for (t, tid), votes in by_target.items() if t == ttype}
        pairs = [(votes[a].decision.value, votes[b].decision.value) for votes in items.values() if len(votes) >= 2 for a, b in [sorted(votes)[:2]]]
        secs = sorted(d.annotation_seconds for votes in items.values() for d in votes.values() if d.annotation_seconds is not None)
        out["by_target_type"][ttype] = {
            "items_reviewed": len(items), "items_with_two_annotators": len(pairs),
            "observed_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
            "cohen_kappa": cohen_kappa(pairs), "median_annotation_seconds": secs[len(secs) // 2] if secs else None}
    return out
