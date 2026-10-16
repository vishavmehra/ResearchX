"""Result Discrepancy Investigator: why do two similar experiments report different outcomes?

The observed gap is computed from the reported scores. Verified differences are fields both records state with
different values (the comparison engine, services/comparison/fields.py); training settings are compared stage by stage
(services/analysis/configs.py), so a pre-training learning rate is never a "difference" from a fine-tuning one: such a
pair is reported as not compared. Fields a side does not state are listed as missing information, never assumed equal.
Explanations are HYPOTHESES tied to a verified difference: a difference can contribute to a gap, the records never show
that it caused it. Suggested checks change one factor at a time. Either side may be a runtime experiment (through the
runtime Overlay).
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from services.analysis import configs as C
from services.analysis import facts as F
from services.analysis.papers import paper_level
from services.canonicalization.normalize import canonical_label
from services.comparison.fields import LABELER_VERSION, compare
from services.evidence.grounding import number_of

VERSION = "discrepancy-0.2.0"
DIFFERENT = {"DIFFERENT", "MATERIALLY_DIFFERENT"}
SAME = {"IDENTICAL", "EQUIVALENT"}
# field -> why a difference in it can move a score (stated as a possibility, never as the cause)
WHY = {
    "evaluation.benchmark": "the scores come from different benchmarks, so they measure different things",
    "research.task": "the experiments address different tasks",
    "evaluation.dataset_split": "scores on different splits (e.g. validation vs test) are not interchangeable",
    "evaluation.metric_definition": "the metric is computed differently",
    "evaluation.evaluation_protocol": "the evaluation protocol differs (e.g. zero-shot vs fine-tuned, number of shots)",
    "model.model_name": "different models were evaluated",
    "model.checkpoint": "different checkpoints were evaluated",
    "model.parameter_count": "the model sizes differ",
    "method.method_name": "different methods were applied",
    "training.learning_rate": "the learning rate differs", "training.batch_size": "the batch size differs",
    "training.epochs": "the training length (epochs) differs", "training.steps": "the training length (steps) differs",
}
VERDICTS = {
    "RELATED_CONTEXT": ("NOT_A_DISCREPANCY", "The experiments use different benchmarks or tasks: their scores are different evaluations, not a discrepancy."),
    "MATERIALLY_DIFFERENT": ("DIFFERENCES_FOUND", "The evaluation setups differ in reported fields; each difference is a candidate explanation."),
    "INSUFFICIENT_EVIDENCE": ("UNDETERMINED", "Fields needed to tell whether the setups match are not stated in the sources read, so the gap cannot be attributed."),
    "ALIGNED_EXPERIMENT": ("UNEXPLAINED", "The evaluation setups match on every recorded equivalence field, yet the scores differ; the cause is not in the records."),
    "CONFLICTING_EVIDENCE": ("CONFLICTING_RECORDS", "At least one compared field has conflicting evidence; resolve it before attributing the gap."),
}


PART = re.compile(r"[+,/()\[\]]|\s-\s|\bw/|\bwith\b", re.I)
SPLIT_WORD = re.compile(r"\b(validation|val|test|dev)\b", re.I)


def parts(name) -> dict[str, str]:
    """Components of a model or method name: 'HRNetV2 + OCR + RMI (PaddleClas pretrained)' -> HRNetV2, OCR, RMI, PaddleClas pretrained."""
    out: dict[str, str] = {}
    for p in PART.split(str(name or "")):
        if p := p.strip(" .-"):
            out.setdefault(canonical_label(p), p)
    return out


def name_delta(a, b) -> dict:
    """Which name components only one side has; a component that extends the other side's (HRNetV2 vs HRNetV2-W48) is a variant."""
    pa, pb = parts(a), parts(b)
    only_a, only_b = {k: v for k, v in pa.items() if k not in pb}, {k: v for k, v in pb.items() if k not in pa}
    variants = []
    for ka in list(only_a):
        kb = next((kb for kb in only_b if kb.startswith(ka) or ka.startswith(kb)), None)
        if kb:
            variants.append([only_a.pop(ka), only_b.pop(kb)])
    return {"only_a": list(only_a.values()), "only_b": list(only_b.values()), "variants": variants}


def _list(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _scores(e) -> dict[str, dict]:
    out = {}
    for r in F.results(e):
        m = r["metric"]
        if m["status"] in F.STATED and r["score"]["status"] in F.STATED:
            out.setdefault(canonical_label(str(m["value"])), r)
    return out


def observed(ea, eb) -> list[dict]:
    """Reported scores of the metrics both experiments report, with the gap where both are numbers."""
    sa, sb = _scores(ea), _scores(eb)
    rows = []
    for m in sorted(sa.keys() & sb.keys()):
        ra, rb = sa[m], sb[m]
        va, vb = number_of(ra["score"]["raw"] or ra["score"]["value"]), number_of(rb["score"]["raw"] or rb["score"]["value"])
        gap = None if va is None or vb is None else round(va - vb, 6)
        rows.append({"metric": ra["metric"]["value"], "a": {"score": ra["score"]["raw"] or ra["score"]["value"], "split": ra["split"]["value"],
                                                            "evidence_ids": ra["score"]["evidence_ids"][:2]},
                     "b": {"score": rb["score"]["raw"] or rb["score"]["value"], "split": rb["split"]["value"], "evidence_ids": rb["score"]["evidence_ids"][:2]},
                     "difference": gap, "relative": round(gap / abs(vb), 4) if gap is not None and vb else None})
    return rows


def investigate(index, a: str, b: str, tr, fill=None) -> dict:
    """`fill(models, tr)`: read the full text for settings still unknown before comparing (services/analysis/enrich.py)."""
    if a == b:
        raise ValueError("Pick two different experiments.")
    with tr.stage("experiment_retrieval", experiments=[a, b]) as st:
        models = index.experiments([a, b])
        if missing := [x for x in (a, b) if x not in models]:
            raise KeyError(missing[0])
        ea, eb = models[a], models[b]
        st.detail.update(papers=[ea.paper_id, eb.paper_id], quality=[ea.quality_tier.value, eb.quality_tier.value])
    if fill:
        fill({x: m for x, m in ((a, ea), (b, eb)) if not x.startswith("rt:")}, tr)
    with tr.stage("comparison", method=f"comparison engine {LABELER_VERSION}; training settings stage by stage ({C.VERSION})") as st:
        res = compare(ea, eb)
        rows = [{"field": fc.field_name, "label": F.label(fc.field_name.split("[")[0]), "status": fc.comparison_status.value, "value_a": fc.value_a,
                 "value_b": fc.value_b, "evidence_a": fc.evidence_a[:2], "evidence_b": fc.evidence_b[:2], "rationale": fc.rationale}
                for fc in res["field_comparisons"] if not fc.field_name.startswith("evaluation.score") and C.role(fc.field_name) != "training"]
        views = C.views(index, {a: ea, b: eb})  # a pre-training value is never compared with a fine-tuning one
        rows += [{"field": r["field"], "stage": r["stage"], "label": f"{r['label']} ({r['stage_label']})", "status": r["status"], "value_a": r["value_a"],
                  "value_b": r["value_b"], "evidence_a": r["evidence_a"], "evidence_b": r["evidence_b"], "rationale": r["rationale"]}
                 for r in C.compare(views[a], views[b])]
        obs = observed(ea, eb)
        st.detail.update(label=res["label"].value, shared_metrics=len(obs), differing=sum(r["status"] in DIFFERENT for r in rows),
                         unknown=sum(r["status"] == "UNKNOWN" for r in rows), staged_rows=sum(1 for r in rows if r.get("stage")))
    with tr.stage("assessment", method="deterministic: verified differences -> labelled hypotheses; unknown fields -> missing information") as st:
        differences = [r for r in rows if r["status"] in DIFFERENT]
        for r in differences:
            if r["field"] in ("model.model_name", "method.method_name", "model.checkpoint", "model.base_model"):
                r["parts"] = name_delta(r["value_a"], r["value_b"])
        hypotheses = []
        for r in differences:
            d = r.get("parts") or {}
            said = [f"only A names {_list(d['only_a'])}"] if d.get("only_a") else []
            said += [f"only B names {_list(d['only_b'])}"] if d.get("only_b") else []
            said += [f"{va} (A) vs {vb} (B)" for va, vb in d.get("variants", [])]
            why = f"the names differ by their components: {'; '.join(said)}" if said else WHY.get(r["field"], "this setting differs")
            hypotheses.append({"factor": r["label"], "field": r["field"], "status": "HYPOTHESIS",
                               "statement": f"{r['label'].capitalize()} differs ({r['value_a']} vs {r['value_b']}): {why}. "
                                            "This may contribute to the gap; the records do not establish that it causes it.",
                               "evidence": r["evidence_a"] + r["evidence_b"]})
        unknown = [{"field": r["field"], "label": r["label"], "missing_on": [s for s, v in (("A", r["value_a"]), ("B", r["value_b"])) if v is None],
                    "stage": r.get("stage"), "stage_mismatch": "different stages" in (r.get("rationale") or ""), "rationale": r.get("rationale")}
                   for r in rows if r["status"] == "UNKNOWN"]
        for u in unknown:
            u["both"] = len(u["missing_on"]) == 2
            if u["field"] == "evaluation.dataset_split":  # a hint from the metric's own name, never recorded as the split
                named = {m.group(1).lower() for o in obs if (m := SPLIT_WORD.search(str(o["metric"])))}
                if len(named) == 1:
                    u["hint"] = f"the shared metric is named “{obs[0]['metric']}”, which suggests the {named.pop()} split; the records do not state the split"
        # context, never a verified difference: what each experiment's paper states for a field its record leaves unknown
        level = paper_level(index, list(dict.fromkeys([ea.paper_id, eb.paper_id])), [u["field"] for u in unknown])
        for u in unknown:
            u["paper_level"] = {side: level.get((pid, u["field"]), [])[:2] for side, pid in (("A", ea.paper_id), ("B", eb.paper_id))
                                if side in u["missing_on"] and level.get((pid, u["field"]))}
        repo = {x: (e.artifacts.repository_url.value if e.artifacts.repository_url.status.value == "KNOWN" else None) for x, e in ((a, ea), (b, eb))}
        checks = []
        for r in differences:
            d = r.get("parts") or {}
            if d.get("only_a") or d.get("only_b"):  # one component at a time, not "swap the whole model"
                checks += [{"test": f"Re-run experiment A without {c} (named only in A), every other setting unchanged.",
                            "establishes": f"whether {c} accounts for part of the gap", "controls": "all other fields of A",
                            "cannot_establish": "whether the components interact"} for c in d.get("only_a", [])]
                checks += [{"test": f"Re-run experiment A with {c} added (named only in B), every other setting unchanged.",
                            "establishes": f"whether {c} accounts for part of the gap", "controls": "all other fields of A",
                            "cannot_establish": "whether the components interact"} for c in d.get("only_b", [])]
            elif r["field"].startswith(("training.", "evaluation.", "model.")):
                checks.append({"test": f"Re-run experiment A with {r['label']} set to B's value ({r['value_b']}), every other setting unchanged.",
                               "establishes": f"whether {r['label']} accounts for part of the gap under A's setup", "controls": "all other fields of A",
                               "cannot_establish": "whether the remaining factors interact"})
        checks += [{"test": f"Find {u['label']} for experiment {'/'.join(u['missing_on'])} in the paper or its code"
                            + (f" ({repo[a if u['missing_on'] == ['A'] else b]})" if repo.get(a if u["missing_on"] == ["A"] else b) else "") + ".",
                    "establishes": "whether the setups really match on this field", "controls": "-", "cannot_establish": "-"}
                   for u in unknown if u["field"] in ("evaluation.dataset_split", "evaluation.metric_definition", "evaluation.evaluation_protocol")][:4]
        verdict, verdict_why = VERDICTS[res["label"].value]
        reasons = []
        if unknown:
            reasons.append(f"{len(unknown)} compared field(s) are not stated for at least one side")
        if "SILVER" in (ea.quality_tier.value, eb.quality_tier.value):
            reasons.append("silver-tier machine-extracted records (not human-reviewed)")
        if not obs:
            reasons.append("the experiments report no metric in common")
        contradictions = [c for x in (a, b) for c in index.contra.get(x, [])]
        # the findings in plain sentences, each built from the rows above (no model call, nothing beyond the records)
        findings = []
        for o in obs:
            if o["difference"] is not None:
                rel = f", {abs(o['relative']) * 100:.1f}% relative" if o["relative"] is not None else ""
                findings.append(f"{o['metric']}: A reports {o['a']['score']}, B {o['b']['score']}; A is {abs(o['difference']):g} "
                                f"{'higher' if o['difference'] > 0 else 'lower' if o['difference'] < 0 else 'apart'}{rel}.")
        same = [r for r in rows if r["status"] in SAME]
        if same:
            findings.append(f"Both state the same {_list([r['label'] for r in same])}: these settings are held equal.")
        for r in differences:
            d = r.get("parts") or {}
            bits = [f"A adds {_list(d['only_a'])}"] if d.get("only_a") else []
            bits += [f"B adds {_list(d['only_b'])}"] if d.get("only_b") else []
            bits += [f"{va} vs {vb}" for va, vb in d.get("variants", [])]
            findings.append(f"They differ in {r['label']}" + (f" ({'; '.join(bits)})" if bits else f": {r['value_a']} vs {r['value_b']}") + ".")
        if both := [u["label"] for u in unknown if u["both"]]:
            findings.append(f"Neither record states {_list(both)}, so whether these match is unknown.")
        if staged := [f"{F.label(u['field'])} ({u['rationale'].split(':')[0]})" for u in unknown if u["stage_mismatch"]]:
            findings.append(f"Not compared, stated for different training stages: {_list(staged)}.")
        if one := [f"{u['label']} ({'/'.join(s for s in ('A', 'B') if s not in u['missing_on'])} only)" for u in unknown
                   if not u["both"] and not u["stage_mismatch"]]:
            findings.append(f"Stated on one side only: {_list(one)}.")
        st.detail.update(verdict=verdict, hypotheses=len(hypotheses), missing=len(unknown), checks=len(checks), recorded_contradictions=len(contradictions))
    ids = [i for o in obs for s in ("a", "b") for i in o[s]["evidence_ids"]] + [i for r in rows for i in r["evidence_a"] + r["evidence_b"]]
    ids += [i for u in unknown for vs in u["paper_level"].values() for v in vs for i in v["evidence_ids"]]
    found = {e["id"]: e for e in index.evidence_detail(ids, limit=len(ids))} if ids else {}
    for u in unknown:
        for vs in u["paper_level"].values():
            for v in vs:
                v["evidence"] = [found[i] for i in v.pop("evidence_ids", []) if i in found]
                v.pop("key", None)
    for o in obs:
        for s in ("a", "b"):
            o[s]["evidence"] = [found[i] for i in o[s].pop("evidence_ids") if i in found]
    for r in rows:
        r["evidence_a"], r["evidence_b"] = [found[i] for i in r["evidence_a"] if i in found], [found[i] for i in r["evidence_b"] if i in found]
    return {"kind": "discrepancy", "version": VERSION, "experiments": [index.node(a), index.node(b)],
            "papers": [index.node(ea.paper_id), index.node(eb.paper_id)], "observed": obs, "label": res["label"].value,
            "label_rationale": res["rationale"], "verdict": verdict, "verdict_why": verdict_why, "findings": findings, "differences": differences,
            "same": [r for r in rows if r["status"] in SAME], "unknown": unknown, "hypotheses": hypotheses, "checks": checks,
            "contradictions": contradictions, "confidence": {"level": "LOW" if reasons else "MODERATE", "reasons": reasons},
            "summary": {"verdict": verdict, "differences": len(differences), "unknown": len(unknown), "shared_metrics": len(obs)},
            "inputs": {"experiments": [a, b]},
            "notes": ["Hypotheses are derived from reported differences; none is a verified cause.",
                      "Unknown fields are never treated as matching. A paper-level value is stated in the paper's text but not assigned to the "
                      "experiment: it is context for finding the missing setting, not evidence of a difference."]}


def candidates(index, benchmark: str | None = None, limit: int = 30, experiment: str | None = None) -> dict:
    """Pairs worth investigating: the same model on the same benchmark and metric with different reported scores (across
    experiment records), plus score contradictions recorded inside one record. Exact canonical ids only, no similarity.
    With `experiment`: only its pairs; when it has none, `related` lists experiments reporting the same benchmark and
    metric with another score (any model), each a valid B for the investigation."""
    if experiment is not None and experiment not in index.exp:
        raise KeyError(experiment)
    groups: dict[tuple, list] = defaultdict(list)
    related: list[dict] = []
    if experiment:
        mine = index.exp[experiment]
        mine_d = {link["entity_id"] for link in mine["links"] if link["edge"] == "USES_DATASET"}
        mine_r = {(r["metric_id"], number_of(r["score"])): r["score"] for r in mine["results"] if r["score"] and r["metric_id"]}
        for xid, e in index.exp.items():
            if xid == experiment or not mine_d & {link["entity_id"] for link in e["links"] if link["edge"] == "USES_DATASET"}:
                continue
            for r in e["results"]:
                v = number_of(r["score"]) if r["score"] and r["metric_id"] else None
                mine_v = next(((mv, s) for (m, mv), s in mine_r.items() if m == r["metric_id"] and mv is not None), None)
                if v is not None and mine_v and mine_v[0] != v:
                    related.append({"a": experiment, "b": xid, "score_a": mine_v[1], "score_b": r["score"], "relative": round(abs(mine_v[0] - v) / abs(v), 4) if v else None,
                                    "label": e.get("label"), "metric": index.ent.get(r["metric_id"], {}).get("name"),
                                    "same_paper": e["paper_id"] == mine["paper_id"]})
                    break
        related.sort(key=lambda p: (not p["same_paper"], p["relative"] if p["relative"] is not None else 9e9))
    for xid, e in index.exp.items():
        models = {link["entity_id"] for link in e["links"] if link["field"] == "model.model_name"}  # not base_model: a fine-tune is another model
        dsets = {link["entity_id"] for link in e["links"] if link["edge"] == "USES_DATASET"}
        if benchmark:
            dsets &= {benchmark}
        for r in e["results"]:
            v = number_of(r["score"]) if r["score"] and r["metric_id"] else None
            if v is None:
                continue
            for m in models:
                for d in dsets:
                    groups[(m, d, r["metric_id"])].append((xid, v, r["score"]))
    pairs = []
    for (m, d, metric), items in groups.items():
        for (xa, va, ra), (xb, vb, rb) in combinations(sorted(set(items)), 2):
            if xa != xb and va != vb and vb:
                pairs.append({"a": xa, "b": xb, "score_a": ra, "score_b": rb, "relative": round(abs(va - vb) / abs(vb), 4),
                              "model": index.ent.get(m, {}).get("name"), "benchmark": index.ent.get(d, {}).get("name"),
                              "metric": index.ent.get(metric, {}).get("name"), "same_paper": index.exp[xa]["paper_id"] == index.exp[xb]["paper_id"]})
    if experiment:  # oriented so the given experiment is A
        pairs = [p if p["a"] == experiment else {**p, "a": p["b"], "b": p["a"], "score_a": p["score_b"], "score_b": p["score_a"]}
                 for p in pairs if experiment in (p["a"], p["b"])]
    pairs.sort(key=lambda p: (p["same_paper"], -p["relative"]))
    recorded = [{"experiment": s, "label": index.exp.get(s, {}).get("label"), "value_a": c["value_a"], "value_b": c["value_b"],
                 "explanation": c["explanation"]} for s, cs in index.contra.items() for c in cs if c["field"] == "evaluation.score" and s in index.exp
                and experiment in (None, s)][:limit]
    return {"pairs": pairs[:limit], "recorded": recorded, "total_pairs": len(pairs), "related": related[:limit],
            "experiment_label": index.exp[experiment].get("label") if experiment else None}
