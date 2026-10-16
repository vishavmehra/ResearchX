"""Next Experiment Planner: which experiment would most reduce the uncertainty around one experiment?

Every recommendation is derived from the records around the chosen experiment, never generated:
- reproduce: a contradiction recorded for this experiment, or other records of the same model and benchmark with
  different scores (services.analysis.discrepancy.candidates).
- ablation: a component of the model's name (beyond the base model) that no record of the same paper and benchmark
  removes on its own; a sibling record that removes it (alone or with others) is shown as a reported ablation.
- missing baseline: models other papers report on the same benchmark and metric that this paper has no record of.
- robustness: benchmarks that other papers report together with this benchmark and this paper has no record on.
- reporting: unstated critical settings (seed and variance, evaluation split, protocol, metric definition) that keep the
  result from being compared with the other results on its benchmark.
Each says what question it tests, what to hold fixed, which records motivate it, and what its result could and could not
establish. It is a proposal: no outcome is predicted, nothing is claimed to be novel, and compute is shown only when the
original record states it.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from services.analysis import facts as F
from services.analysis.claims import GENERIC_NAMES, _bench, _names
from services.analysis.discrepancy import candidates, parts
from services.canonicalization.normalize import canonical_label, model_family
from services.evidence import grounding as G

VERSION = "planner-0.1.0"
REPORTING = ("training.seed", "evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition")
COMPUTE = ("compute.gpu_model", "compute.gpu_count", "compute.training_time")
NOT_A_BASELINE = GENERIC_NAMES | {"few-shot", "zero-shot", "one-shot", "cot", "human", "humans", "random", "majority", "oracle"}
PRIORITY = {"reproduce": 0, "ablation": 1, "missing_baseline": 2, "robustness": 3, "reporting": 4}


def components(name: str) -> list[str]:
    """Named components of a model name, base first; sizes, versions and patch sizes are not components."""
    return [s for s in parts(name).values() if re.search(r"[A-Za-z]{2}", s) and not all(G.VARIANT.fullmatch(t) for t in G.tokens(s))]


def _score_of(index, xid: str, metric_ids: set[str]) -> dict | None:
    r = next((r for r in index.exp[xid]["results"] if r["metric_id"] in metric_ids and G.number_of(r["score"]) is not None), None)
    return {"experiment": xid, "label": index.exp[xid]["label"], "metric": r["metric"], "score": r["score"], "evidence_ids": r["evidence_ids"][:1]} if r else None


def _stated(e, path: str) -> str | None:
    f = F.fact(path, F.field_value(e, path))
    return str(f["raw"] if f["raw"] not in (None, "") else f["value"]) if f["status"] in F.STATED else None


def resolve(index, text: str) -> str:
    """The experiment a plain request names ("HRNetV2 + OCR on ADE20K"): records of the model (claims.matched) on the named
    dataset, direct matches first, then the one with the most numeric results. KeyError when nothing matches."""
    from services.analysis.claims import _on_dataset, datasets_in, matched
    text = " ".join((text or "").split())
    dataset = next(iter(datasets_in(index, text)), None)
    model = re.split(r"\s+(?:on|for)\s+(?:the\s+)?" + re.escape(dataset), text, flags=re.I)[0] if dataset else text
    found = matched(index, [model])[0][0]
    pool = [(x, k) for x, k in found.items() if _on_dataset(index, x, dataset)[0]]
    if not pool:
        raise KeyError(text)
    nres = lambda x: sum(G.number_of(r["score"]) is not None for r in index.exp[x]["results"])  # noqa: E731
    return min(pool, key=lambda xk: (xk[1] != "direct", -nres(xk[0]), xk[0]))[0]


def plan(index, xid: str, tr, goal: str | None = None) -> dict:
    with tr.stage("experiment_retrieval", experiment=xid) as st:
        e = index.experiments([xid]).get(xid) if xid in index.exp else None
        if e is None:
            raise KeyError(xid)
        x, pid = index.exp[xid], index.exp[xid]["paper_id"]
        bench = _bench(index, xid)
        name = (_names(index, xid) or [x["label"]])[0]
        metric_ids = {r["metric_id"] for r in x["results"] if r["metric_id"] and G.number_of(r["score"]) is not None}
        siblings = [s for s in index.paper_exps.get(pid, []) if s != xid and bench and (_bench(index, s) or {}).get("entity_id") == bench["entity_id"]]
        st.detail.update(paper=pid, benchmark=bench and bench["raw"], metrics=len(metric_ids), siblings=len(siblings))
    bname = (index.ent.get(bench["entity_id"], {}).get("name") or bench["raw"]) if bench else None
    own = _score_of(index, xid, metric_ids)
    recs, reported = [], []
    with tr.stage("recommendations", method="deterministic: contradictions, name components, co-reported baselines and benchmarks, unstated settings") as st:
        # reproduce: conflicting records of this experiment, or the same model reported with other scores
        for c in index.contra.get(xid, [])[:2]:
            recs.append({"kind": "reproduce", "title": f"Resolve the conflicting {c['field'].split('.')[-1].replace('_', ' ')}",
                         "question": f"Which value of {c['field']} does a re-run of this experiment produce?",
                         "why": f"The corpus records two values for this experiment ({c['value_a']} vs {c['value_b']}): {c.get('explanation') or 'sources disagree'}.",
                         "controls": ["the full configuration of this record"], "evidence": [own] if own else [],
                         "establishes": "which recorded value a faithful re-run reproduces", "cannot_establish": "why the sources differ"})
        pairs = candidates(index, experiment=xid, limit=5)["pairs"] if bench else []
        if pairs:
            p = pairs[0]
            recs.append({"kind": "reproduce", "title": f"Reproduce {name} under one fixed setup",
                         "question": f"Does {name} reach {p['score_a']} or {p['score_b']} on {p['benchmark']} when every setting is fixed?",
                         "why": f"{len(pairs)} other record(s) of the same model, benchmark and metric report different scores ({p['score_a']} vs {p['score_b']}).",
                         "controls": ["dataset version and split", "evaluation protocol", "checkpoint"], "link": {"tab": "discrepancy", "a": xid, "b": p["b"]},
                         "evidence": [s for s in (own, _score_of(index, p["b"], metric_ids)) if s],
                         "establishes": "whether the score gap survives a controlled re-run", "cannot_establish": "which original setup was used, if unreported"})
        # ablation: components beyond the base that no sibling record removes on its own
        comps = components(name)
        if len(comps) >= 2:
            base, others = comps[0], comps[1:]
            for s in siblings:
                have = set().union(*(G.tokens(n) for n in _names(index, s)))
                lacks = [c for c in others if not G.tokens(c) <= have]
                if G.tokens(base) <= have and lacks and (sc := _score_of(index, s, metric_ids)):
                    reported.append({"sibling": sc, "removes": lacks, "this": own})
            for c in others:
                if any(r["removes"] == [c] for r in reported):
                    continue
                joint = [r for r in reported if c in r["removes"]]
                recs.append({"kind": "ablation", "title": f"Ablate {c}",
                             "question": f"How much of {name}'s result on {bname} depends on {c}?",
                             "why": (f"{c} is a component of the model and no record of the paper removes it on its own"
                                     + (f"; the paper only reports removing it together with {', '.join(y for y in joint[0]['removes'] if y != c)}" if joint else "") + "."),
                             "controls": ["every other component", "the training settings and data of this record", "evaluation split and protocol"],
                             "evidence": [s for s in [own] + [j["sibling"] for j in joint[:1]] if s],
                             "establishes": f"the contribution of {c} under this setup", "cannot_establish": "whether it holds on other datasets or at other scales"})
        # missing baselines: model families other papers report on this benchmark and metric that the paper never runs,
        # most widely reported first (a family groups sizes: "BEiT-L" and "BEiT-B" are one baseline family)
        if bench and metric_ids:
            paper_names = [n for s in index.paper_exps.get(pid, []) for n in _names(index, s)]
            seen: dict[str, dict] = {}
            for o in index.ent_exps.get(bench["entity_id"], []):
                if index.exp[o]["paper_id"] == pid or not (sc := _score_of(index, o, metric_ids)):
                    continue
                nm = (_names(index, o) or [None])[0]
                if not nm or canonical_label(nm) in NOT_A_BASELINE or not re.search(r"[A-Za-z]{2}", nm):
                    continue
                # ponytail: family = leading word of the name ("ResNet-50" -> resnet); merges unrelated names sharing a head
                # word (rare); upgrade to canonical model families if the entity table gains them
                fam = (m.group(0) if (m := re.match(r"[a-z]+", canonical_label(nm))) and len(m.group(0)) > 2 else None) or model_family(nm)[0] or canonical_label(nm)
                slot = seen.setdefault(fam, {"names": Counter(), "papers": set(), "results": []})
                slot["names"][nm] += 1
                slot["papers"].add(index.exp[o]["paper_id"])
                slot["results"].append(sc)
            for s in seen.values():
                s["name"] = s["names"].most_common(1)[0][0]
            ranked = sorted(seen.values(), key=lambda s: (-len(s["papers"]), -len(s["results"]), s["name"].casefold()))
            missing = [s for s in ranked if (G.coverage(s["name"], paper_names, {}) or 0) < 0.999 and (G.coverage(name, [s["name"]], {}) or 0) < 0.999][:3]
            for s in missing:
                recs.append({"kind": "missing_baseline", "title": f"Compare against {s['name']}",
                             "question": f"Does {name} outperform {s['name']} on {bname} under the same evaluation setup?",
                             "why": (f"{len(s['papers'])} other paper(s) report {s['name']}"
                                     + (f" or variants of it ({', '.join(n for n, _ in s['names'].most_common(3)[1:])})" if len(s["names"]) > 1 else "")
                                     + f" on {bname} with this metric; this paper has no record of that model."),
                             "controls": ["the evaluation split, protocol and metric definition", "comparable training data and budget"],
                             "evidence": s["results"][:2] + ([own] if own else []),
                             "establishes": "a same-setup comparison the cross-paper numbers cannot give", "cannot_establish": "anything about other benchmarks"})
        # robustness: benchmarks other papers co-report with this one, for the same task when the record names one
        if bench:
            tasks = {link["entity_id"] for link in x["links"] if link["node_type"] == "Topic"}

            def bench_of(o):
                return (_bench(index, o) or {}).get("entity_id")

            def same_task(o):
                return not tasks or bool(tasks & {link["entity_id"] for link in index.exp[o]["links"] if link["node_type"] == "Topic"})
            papers_here = {index.exp[o]["paper_id"] for o in index.ent_exps.get(bench["entity_id"], [])} - {pid}
            mine_b = {bench_of(o) for o in index.paper_exps.get(pid, [])}
            co: dict[str, set] = defaultdict(set)
            for p in papers_here:
                for b in {bench_of(o) for o in index.paper_exps.get(p, []) if same_task(o)} - {bench["entity_id"], None} - mine_b:
                    co[b].add(p)
            for b, ps in sorted(co.items(), key=lambda kv: (-len(kv[1]), str(index.ent.get(kv[0], {}).get("name"))))[:2]:
                if len(ps) < 2:
                    break
                other = index.ent.get(b, {}).get("name") or b
                recs.append({"kind": "robustness", "title": f"Evaluate on {other}",
                             "question": f"Does {name}'s result on {bname} carry over to {other}?",
                             "why": f"{len(ps)} paper(s) that report {bname} also report {other}; this paper has no record on {other}.",
                             "controls": ["the trained model of this record (no re-tuning)", "the evaluation protocol the other papers state"],
                             "evidence": [own] if own else [],
                             "establishes": f"whether the {bname} result transfers to {other}", "cannot_establish": "performance on benchmarks not tested"})
        # reporting: unstated settings that block comparison
        unstated = [p for p in REPORTING if _stated(e, p) is None]
        if "training.seed" in unstated and not any("uncertainty" in r or "confidence_interval" in r for r in F.results(e)):
            recs.append({"kind": "reporting", "title": "Run several seeds and report the variance",
                         "question": f"How much does {name}'s score vary across random seeds?",
                         "why": "The record states no seed and no uncertainty for any result, so a difference to another result cannot be told from noise.",
                         "controls": ["every setting except the seed"], "evidence": [own] if own else [],
                         "establishes": "the run-to-run spread of the score", "cannot_establish": "the cause of any remaining gap"})
        blocking = [p for p in unstated if p != "training.seed"]
        if blocking and bench and metric_ids:
            n = sum(1 for o in index.ent_exps.get(bench["entity_id"], []) if o != xid and _score_of(index, o, metric_ids))
            recs.append({"kind": "reporting", "title": "State the " + ", ".join(F.label(p) for p in blocking),
                         "question": "Under which conditions was the reported score measured?",
                         "why": f"Without them the result cannot be classified as comparable to the {n} other result(s) on {bname} with this metric.",
                         "controls": [], "evidence": [own] if own else [],
                         "establishes": "comparability with the other reported results", "cannot_establish": "anything about the score itself"})
        recs.sort(key=lambda r: PRIORITY[r["kind"]])
        st.detail.update(recommendations=len(recs), kinds=dict(Counter(r["kind"] for r in recs)), reported_ablations=len(reported))
    compute = {F.label(p): v for p in COMPUTE if (v := _stated(e, p))}
    shown = [s for r in recs for s in r["evidence"]] + [s for r in reported for s in (r["sibling"], r["this"]) if s]
    ids = [i for s in shown for i in s.get("evidence_ids", [])]
    ev = {d["id"]: d for d in index.evidence_detail(ids, limit=400)} if ids else {}
    for s in shown:
        if "evidence_ids" in s:  # one score dict can motivate several recommendations: attach once
            s["evidence"] = [ev[i] for i in s.pop("evidence_ids") if i in ev]
    return {"kind": "plan", "version": VERSION, "experiment": index.node(xid), "paper": index.node(pid), "model": name, "benchmark": bname,
            "goal": " ".join(goal.split())[:600] if goal else None, "recommendations": recs, "reported_ablations": reported, "compute": compute or None,
            "summary": {"recommendations": len(recs), **dict(Counter(r["kind"] for r in recs)), "reported_ablations": len(reported)},
            "inputs": {"experiment": xid, "goal": goal},
            "notes": ["Recommendations are proposals derived from the records; no outcome is predicted and none is claimed to be novel.",
                      "Compute is shown only when the original record states it." if compute else "The record states no compute, so none is estimated.",
                      "The goal you enter is kept with the plan; the recommendations come from the records of the chosen experiment."]}
