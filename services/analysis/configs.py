"""Configuration identity (docs/ARCHITECTURE.md §70): pre-training, fine-tuning / adaptation and evaluation kept apart.

One experiment record holds one value per field, so a paper that pre-trains a model, fine-tunes it and evaluates it
would collapse three learning rates into one. The stages are separated here at read time, from the evidence itself
(never from a title, never by similarity), and the configurations of one study are linked:

  stage     pretraining | finetuning | evaluation | unspecified. Evaluation conditions are evaluation by definition; a
            training setting takes the stage its evidence names: the stage cue nearest the value in the quoted text,
            else the section heading. Otherwise its stage is not stated, and it is never compared with a staged value.
  identity  cfg:<experiment>:evaluation for the evaluated condition; cfg:<paper>:<stage>:<scope key> for the paper's
            pre-training and fine-tuning configurations, the scope key being the canonical key of the dataset the
            statement names (services.canonicalization.entities.key) or "general". One paper can hold several of each.
  links     evaluation -> fine-tuning -> pre-training, each DERIVED with its rule (the statement names the evaluated
            dataset, or the paper states one configuration of that stage), and initialisation REPORTED when the record
            states it (model.pretrained_source / base_model / initialization / checkpoint) with evidence.
Nothing is written: configurations are derived from rx.evidence and request-time quotes, so experiment and evidence ids
stay what they were. Several values for one field in one configuration are VARIES (which experiment each belongs to is
not stated); a CONFLICTING record field stays CONFLICTING.
"""

from __future__ import annotations

import re
from collections import defaultdict

from services.analysis import facts as F
from services.canonicalization.entities import key as entity_key
from services.canonicalization.normalize import canonical_label
from services.comparison.fields import compare_field
from services.datasets.schemas import FIELD_GROUPS, FieldStatus, FieldValue
from services.evidence import grounding as G

VERSION = "configs-0.1.0"
STAGES = ("pretraining", "finetuning", "evaluation", "unspecified")
TRAINING_STAGES = ("pretraining", "finetuning", "unspecified")
STAGE_LABEL = {"pretraining": "pre-training", "finetuning": "fine-tuning / adaptation", "evaluation": "evaluation",
               "unspecified": "stage not stated"}
PRE = re.compile(r"pre-?train(?:ing|s)?(?![a-z])|pre-?trained (?:for|on|with|using|over|from scratch)(?![a-z])|self-supervised (?:pre-?)?training|"
                 r"(?:masked|contrastive|generative) pre-?training")
FINE = re.compile(r"fine-?tun(?:e|es|ing)(?![a-z])|fine-?tuned (?:for|on|with|using|over)(?![a-z])|transfer(?:red)? (?:learning|to)(?![a-z])|"
                  r"linear[ -]?prob(?:e|es|ing)(?![a-z])|(?<![a-z])adapt(?:ation|ing)(?![a-z])|downstream (?:training|fine-?tuning)|instruction[ -]?tun\w*")
STAGE_CUES = (("pretraining", PRE), ("finetuning", FINE))
NEAR = 160  # a stage cue farther than this from the value (in the quoted text) gives way to the section heading
FIXED_STAGE = {"model.pretrained_source": "pretraining"}  # the source a model was pre-trained on or with
TRAINING = frozenset(("method.objective", "method.loss_function", "method.training_strategy", "dataset.train_data", "dataset.split",
                      "dataset.sample_count", "dataset.augmentation", "dataset.filtering", "dataset.deduplication", "dataset.sampling_method",
                      "compute.accelerator", "compute.gpu_model", "compute.gpu_count", "compute.training_time", "compute.flops",
                      "model.pretrained_source"))
STAGED_EVALUATION = frozenset(("dataset.dataset_version", "dataset.preprocessing"))  # evaluation unless the evidence names a training stage
LINEAGE = ("model.base_model", "model.initialization", "model.checkpoint")
PRETRAINING_DATA = ("model.pretrained_source", "dataset.train_data", "dataset.dataset_name", "dataset.dataset_version")
ADAPTATION = ("zero-shot", "few-shot", "linear probe", "kNN", "frozen backbone", "fine-tuned")
PAPER_SQL = ("select id, paper_id, field_name, record from rx.evidence where paper_id = any(%s) and experiment_id is null "
             "and location_kind like 'PAPER_%%' and field_name = any(%s) order by id")


def role(path: str) -> str:
    """evaluation | training | staged_evaluation | identity (model / method names, artifacts, research question)."""
    if path in STAGED_EVALUATION:
        return "staged_evaluation"
    if path.startswith("evaluation.") or path in ("dataset.test_data", "dataset.validation_data"):
        return "evaluation"
    if path.startswith("training.") or path in TRAINING:
        return "training"
    return "identity"


TRAINING_PATHS = [f"{g}.{n}" for g, names in FIELD_GROUPS.items() for n in names if role(f"{g}.{n}") in ("training", "staged_evaluation")]


def stage_of(text: str | None, value=None, section: str | None = None) -> tuple[str | None, str]:
    """(pretraining | finetuning | None, why) for a stated training setting: the stage cue nearest the value in the quoted
    text (a cue after the value counts as farther), else a stage named by the section heading alone."""
    t = G.canon(text)
    want = G.canon(value) if value not in (None, "") else ""
    pos = t.find(want) if want and want in t else len(t) // 2
    best = None
    for stage, rx in STAGE_CUES:
        for m in rx.finditer(t):
            gap = pos - m.end() if m.end() <= pos else (m.start() - pos) + 40
            if best is None or gap < best[0]:
                best = (gap, stage, m.group(0))
    if best and best[0] <= NEAR:
        return best[1], f"the quoted text says “{best[2]}”"
    s = G.canon(section)
    hits = [stage for stage, rx in STAGE_CUES if rx.search(s)]
    if len(hits) == 1:
        return hits[0], f"stated under the section “{section}”"
    return None, "neither the quoted text nor its section names a training stage"


def statement(path: str, raw, value, unit, evidence_ids: list[str], text: str | None, section: str | None, applies_to: str | None, origin: str,
              status: str = F.REPORTED, derivation: str | None = None, stage: str | None = None, basis: str | None = None) -> dict:
    """One stated value with its stage. `origin`: record (the experiment record) | paper (the paper's text, not assigned
    to an experiment) | runtime (a quote read at request time)."""
    if stage is None:
        r = role(path)
        if r == "evaluation":
            stage, basis = "evaluation", "an evaluation condition"
        elif path in FIXED_STAGE:
            stage, basis = FIXED_STAGE[path], "the field names the pre-training source"
        else:
            found, basis = stage_of(text, raw, section)
            stage = found or ("evaluation" if r == "staged_evaluation" else "unspecified")
    return {"field": path, "raw": G.display(raw), "value": value if value not in (None, "") else raw, "unit": unit, "evidence_ids": list(evidence_ids),
            "text": text, "section": section, "applies_to": applies_to, "stage": stage, "basis": basis, "origin": origin, "status": status,
            "derivation": derivation}


def evidence_rows(index, ids) -> dict[str, dict]:
    """Quoted text, section and scope of evidence records (stored and request-time), for stage classification. The text
    is read for classification only; what is shown to people still comes from GraphIndex.evidence_detail (licence)."""
    ids = [i for i in dict.fromkeys(ids) if i]
    out = {}
    for i in ids:
        r = index.runtime_evidence.get(i)
        if r is not None:
            out[i] = {"text": r.get("text"), "section": r.get("section"), "applies_to": r.get("applies_to"), "raw": r.get("extracted_value"),
                      "stage": r.get("stage"), "basis": r.get("stage_basis")}
    stored = [i for i in ids if i not in out and not i.startswith("rt:")]
    for k in range(0, len(stored), 500):
        for i, _fld, rec in index.q("select id, field_name, record from rx.evidence where id = any(%s)", (stored[k:k + 500],)):
            out[i] = {"text": rec.get("evidence_text"), "section": rec.get("section"), "applies_to": rec.get("applies_to"), "raw": rec.get("extracted_value")}
    return out


def paper_statements(index, pids: list[str], paths: list[str]) -> dict[str, list[dict]]:
    """What the papers' own text states for these fields without assigning it to an experiment (rx.evidence PAPER_
    locations, experiment_id null), plus request-time quotes of the same papers, each with its stage."""
    out: dict[str, list[dict]] = defaultdict(list)
    for eid, pid, fld, rec in (index.q(PAPER_SQL, (pids, paths)) if pids and paths else []):
        raw = rec.get("extracted_value") if rec.get("extracted_value") not in (None, "") else rec.get("normalized_value")
        if raw in (None, ""):
            continue
        out[pid].append(statement(fld, raw, rec.get("normalized_value"), rec.get("unit"), [eid], rec.get("evidence_text"), rec.get("section"),
                                  rec.get("applies_to"), "paper", F.DERIVED if rec.get("derived") else F.REPORTED, rec.get("derivation")))
    want = set(paths)
    for rid, r in list(index.runtime_evidence.items()):
        if r.get("paper_id") in pids and r.get("experiment_id") is None and r.get("field") in want:
            out[r["paper_id"]].append(statement(r["field"], r.get("extracted_value"), r.get("normalized_value"), None, [rid], r.get("text"),
                                                r.get("section"), r.get("applies_to"), "runtime", stage=r.get("stage"), basis=r.get("stage_basis")))
    return out


def _named(text: str | None, names: list[str]) -> str | None:
    """The longest of `names` written in `text` as a whole name (canonical text)."""
    t = G.canon(text)
    hits = [n for n in names if n and (c := G.canon(n)) and re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", t)]
    return max(hits, key=len) if hits else None


def scope_of(st: dict, names: list[str]) -> str | None:
    """The dataset a statement is stated for: its recorded scope when that names a dataset, else a dataset its text names."""
    a = st.get("applies_to")
    if a and not re.search(r"\b(?:default|all|every)\b", a, re.I):
        return _named(a, names) or a
    return _named(st.get("text"), names)


def _key(path: str, st: dict) -> str:
    v = G.condition_value(path, st["raw"]) if path in G.TEXT_FIELDS else None
    if v is None and isinstance(st["value"], (int, float)) and not isinstance(st["value"], bool):
        v = repr(float(st["value"]))
    return canonical_label(str(v if v is not None else st["raw"]))


def field_entry(path: str, stmts: list[dict]) -> dict:
    """One field of one configuration: its values (deduplicated) with their evidence, status and stage basis."""
    groups: dict[str, dict] = {}
    for s in stmts:
        g = groups.setdefault(_key(path, s), {"value": s["value"], "raw": s["raw"], "unit": s["unit"], "status": s["status"], "derivation": s["derivation"],
                                              "evidence_ids": [], "origins": [], "experiments": [], "basis": s["basis"], "scope": s.get("scope")})
        g["evidence_ids"] += [i for i in s["evidence_ids"] if i not in g["evidence_ids"]]
        if s["origin"] not in g["origins"]:
            g["origins"].append(s["origin"])
        if s.get("experiment") and s["experiment"] not in g["experiments"]:
            g["experiments"].append(s["experiment"])
    values = list(groups.values())
    status = (F.CONFLICTING if any(v["status"] == F.CONFLICTING for v in values) else "VARIES" if len(values) > 1 else values[0]["status"])
    return {"field": path, "label": F.label(path), "status": status, "values": values}


def _fv(entry: dict | None) -> FieldValue:
    """A configuration field as a FieldValue for the comparison engine: one value, or UNKNOWN (VARIES is not one value)."""
    if not entry or entry["status"] == "VARIES" or not entry["values"]:
        return FieldValue()
    v = entry["values"][0]
    if entry["status"] == F.CONFLICTING:
        ids = list(dict.fromkeys(v["evidence_ids"] + ["conflict:a", "conflict:b"]))
        return FieldValue(status=FieldStatus.CONFLICTING, evidence_ids=ids)
    value = v["value"]
    if isinstance(value, str) and entry["field"] in G.TEXT_FIELDS:
        value = G.condition_value(entry["field"], value) or value
    return FieldValue(status=FieldStatus.KNOWN, raw_value=str(v["raw"]), value=value, unit=v["unit"], evidence_ids=v["evidence_ids"] or ["unlocated"])


def _own(exp, ev: dict) -> list[dict]:
    """The experiment record's own stated values (KNOWN, DERIVED at read time, CONFLICTING), each with its stage; the
    initialisation fields (base model, initialization, checkpoint) as lineage."""
    out = []
    for path, fv in exp.iter_fields():
        lineage = path in LINEAGE
        if fv.status in (FieldStatus.UNKNOWN, FieldStatus.NOT_APPLICABLE) or (role(path) == "identity" and not lineage):
            continue
        status = F.status_of(fv)
        raw = fv.raw_value if fv.raw_value not in (None, "") else fv.value
        if raw in (None, ""):
            raw = "sources disagree" if status == F.CONFLICTING else raw
        if lineage:
            out.append(statement(path, raw, fv.value, fv.unit, fv.evidence_ids, None, None, None, "record", status, fv.derivation, "lineage",
                                 "initialisation of the model"))
            continue
        e = next((ev[i] for i in fv.evidence_ids if i in ev), None) or {}
        out.append(statement(path, raw, fv.value, fv.unit, fv.evidence_ids, e.get("text"), e.get("section"), e.get("applies_to"), "record", status,
                             fv.derivation, stage=e.get("stage"), basis=e.get("basis")))
    return out


def _datasets(exp) -> list[str]:
    out = []
    for path in ("evaluation.benchmark", "dataset.dataset_name"):
        fv = F.field_value(exp, path)
        if fv.status is FieldStatus.KNOWN and fv.raw_value:
            out.append(str(fv.raw_value))
    return out


def _cfg(cid: str, stage: str, paper: str, scope: str | None, stmts: list[dict], label: str) -> dict:
    by_field: dict[str, list] = defaultdict(list)
    for s in stmts:
        by_field[s["field"]].append(s)
    return {"id": cid, "stage": stage, "stage_label": STAGE_LABEL[stage], "paper": paper, "scope": scope, "label": label,
            "fields": [field_entry(p, ss) for p, ss in sorted(by_field.items())]}


def views(index, models: dict, names: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """Configuration view of each experiment (id -> Experiment): its evaluation configuration, the paper's pre-training
    and fine-tuning configurations it is linked to (and why), training values whose stage is not stated, and the
    initialisation the record states. `names`: extra dataset names per paper (e.g. a runtime paper's) for scopes."""
    if not models:
        return {}
    ids = [i for m in models.values() for p, fv in m.iter_fields() if role(p) != "identity" for i in fv.evidence_ids]
    ev = evidence_rows(index, ids)
    pids = sorted({m.paper_id for m in models.values() if m.paper_id})
    paper = paper_statements(index, pids, TRAINING_PATHS)
    own = {x: _own(m, ev) for x, m in models.items()}
    known: dict[str, list[str]] = defaultdict(list)  # dataset names per paper: what its experiments evaluate on or start from
    for x, m in models.items():
        known[m.paper_id] += _datasets(m) + [s["raw"] for s in own[x] if s["field"] in PRETRAINING_DATA and s["raw"]]
        for s in own[x]:
            s["experiment"] = x
    for pid, extra in (names or {}).items():
        known[pid] += list(extra)
    known = {p: sorted({str(n) for n in v if n}, key=len, reverse=True) for p, v in known.items()}
    staged: dict[str, dict[str, dict[str, list]]] = {p: {"pretraining": defaultdict(list), "finetuning": defaultdict(list)} for p in pids}
    used: dict[str, set] = defaultdict(set)
    mine: dict[str, dict[str, str]] = defaultdict(dict)  # experiment -> stage -> key of the configuration its own record states
    for x, m in models.items():  # a record's staged values belong to the paper's configuration of that stage
        for s in own[x]:
            used[m.paper_id].update(s["evidence_ids"])
            if s["stage"] in ("pretraining", "finetuning"):
                s["scope"] = s["raw"] if s["field"] == "model.pretrained_source" else scope_of(s, known.get(m.paper_id, []))
                k = entity_key(s["scope"]) if s["scope"] else "general"
                staged[m.paper_id][s["stage"]][k].append(s)
                if s["status"] == F.REPORTED:  # the record's own value (a value attached at read time is linked by its rule instead)
                    mine[x].setdefault(s["stage"], k)
    for pid in pids:
        for s in paper.get(pid, []):
            if s["stage"] in ("pretraining", "finetuning") and not set(s["evidence_ids"]) & used[pid]:
                s["scope"] = s["raw"] if s["field"] == "model.pretrained_source" else scope_of(s, known.get(pid, []))
                staged[pid][s["stage"]][entity_key(s["scope"]) if s["scope"] else "general"].append(s)
    for st in staged.values():  # a statement that names no dataset holds for every configuration of its stage that does not state its own
        for stage, by in st.items():
            general = by.get("general")
            if general and len(by) > 1:
                del by["general"]
                for ss in by.values():
                    have = {s["field"] for s in ss}
                    ss += [{**s, "basis": f"{s['basis']}; stated without naming a dataset, so it holds for every {STAGE_LABEL[stage]} configuration "
                                          "that states no value of its own"} for s in general if s["field"] not in have]

    def scope(ss):
        return next((s["scope"] for s in ss if s.get("scope")), None)
    configs = {pid: {stage: {k: _cfg(f"cfg:{pid}:{stage}:{k}", stage, pid, scope(ss), ss,
                                     f"{STAGE_LABEL[stage].capitalize()} " + (f"for {scope(ss)}" if scope(ss) else "(no dataset named)"))
                             for k, ss in by.items()} for stage, by in st.items()} for pid, st in staged.items()}
    for x, stages in mine.items():  # an own statement folded into scoped configurations: the dataset rule picks the link instead
        for stage, k in list(stages.items()):
            if k not in configs.get(models[x].paper_id, {}).get(stage, {}):
                del stages[stage]
    return {x: _view(x, m, own[x], configs.get(m.paper_id, {"pretraining": {}, "finetuning": {}}), mine.get(x, {})) for x, m in models.items()}


def _pick(cfgs: dict[str, dict], wanted: list[str], what: str, mine: str | None = None, scoped_to_other: bool = False) -> tuple[dict | None, str]:
    """The configuration of one stage an experiment is linked to, and the rule: the one its own record states values of,
    else the one whose scope names a wanted dataset, else the paper's only one; several candidates and no name: none.
    `scoped_to_other` (fine-tuning): a lone configuration stated for another dataset is not this experiment's."""
    if not cfgs:
        return None, f"the paper states no {what} settings"
    if mine and mine in cfgs:
        return cfgs[mine], f"the experiment record states {what} settings"
    for cfg in cfgs.values():
        if cfg["scope"] and any(G.dataset_match(w, [cfg["scope"]]) == 2 or G.dataset_match(cfg["scope"], [w]) == 2 for w in wanted if w):
            return cfg, f"the paper's {what} statement names {cfg['scope']}"
    if len(cfgs) == 1:
        cfg = next(iter(cfgs.values()))
        if cfg["scope"] and scoped_to_other:
            return None, f"the paper's {what} settings are stated for {cfg['scope']}, not for this experiment's data"
        return cfg, f"the paper's only {what} configuration" + ("" if not cfg["scope"] else f" (stated for {cfg['scope']})")
    return None, f"the paper states {len(cfgs)} {what} configurations and none names this experiment's data"


def _view(xid: str, m, own: list[dict], paper_cfgs: dict, mine: dict) -> dict:
    datasets = _datasets(m)
    lineage = [s for s in own if s["stage"] == "lineage"]
    evaluation = _cfg(f"cfg:{xid}:evaluation", "evaluation", m.paper_id, datasets[0] if datasets else None,
                      [s for s in own if s["stage"] == "evaluation"], "Evaluation" + (f" on {datasets[0]}" if datasets else ""))
    unstaged = [s for s in own if s["stage"] == "unspecified"]
    ft, ft_rule = _pick(paper_cfgs.get("finetuning", {}), datasets, "fine-tuning", mine.get("finetuning"), scoped_to_other=True)
    sources = [s["raw"] for s in own if s["field"] == "model.pretrained_source"] + datasets
    pt, pt_rule = _pick(paper_cfgs.get("pretraining", {}), sources, "pre-training", mine.get("pretraining"))
    configs, relations = [evaluation], []
    if ft:
        configs.append(ft)
        relations.append({"from": evaluation["id"], "to": ft["id"], "relation": "evaluated after", "status": F.DERIVED, "basis": ft_rule})
    if pt:
        configs.append(pt)
        relations.append({"from": (ft or evaluation)["id"], "to": pt["id"], "relation": "starts from", "status": F.DERIVED, "basis": pt_rule})
    if unstaged:
        configs.append(_cfg(f"cfg:{xid}:unspecified", "unspecified", m.paper_id, None, unstaged, "Training settings (stage not stated)"))
    for s in lineage:
        relations.append({"from": (ft or evaluation)["id"], "to": f"ckpt:{entity_key(str(s['raw']))}", "relation": "initialised from",
                          "label": s["raw"], "field": s["field"], "status": s["status"], "evidence_ids": s["evidence_ids"][:3],
                          "basis": f"the record states the {F.label(s['field'])}"})
    unlinked = [{"id": c["id"], "stage": stage, "label": c["label"], "why": rule}
                for stage, chosen, rule in (("finetuning", ft, ft_rule), ("pretraining", pt, pt_rule))
                for c in paper_cfgs.get(stage, {}).values() if chosen is None or c["id"] != chosen["id"]]
    return {"experiment": xid, "paper": m.paper_id, "experiment_type": m.experiment_type.value, "configurations": configs, "relations": relations,
            "unlinked": unlinked, "datasets": datasets, "version": VERSION}


def stage_fields(view: dict, stage: str) -> dict[str, dict]:
    """field -> entry of the configuration of `stage` in this experiment's lineage."""
    return {f["field"]: f for c in view["configurations"] if c["stage"] == stage for f in c["fields"]}


def compare(va: dict, vb: dict) -> list[dict]:
    """Stage-wise comparison of two experiments' configurations: a training setting is compared only with the same
    setting of the same stage. A value stated for another stage, or whose stage is not stated, is never a difference."""
    rows = []
    for stage in TRAINING_STAGES:
        fa, fb = stage_fields(va, stage), stage_fields(vb, stage)
        for path in sorted(set(fa) | set(fb)):
            a, b = fa.get(path), fb.get(path)
            row = {"stage": stage, "stage_label": STAGE_LABEL[stage], "field": path, "label": F.label(path),
                   "value_a": _shown(a), "value_b": _shown(b), "evidence_a": _ids(a), "evidence_b": _ids(b)}
            if a and b and "VARIES" not in (a["status"], b["status"]):
                fc = compare_field(path, _fv(a), _fv(b))
                row.update(status=fc.comparison_status.value, rationale=fc.rationale)
            elif a and b:
                row.update(status="UNKNOWN", rationale="several values are stated for this stage; which one this experiment used is not stated")
            else:
                side, other = ("B", vb) if a else ("A", va)
                elsewhere = [s for s in TRAINING_STAGES if s != stage and path in stage_fields(other, s)]
                row.update(status="UNKNOWN", rationale=(f"side {side} states it for {STAGE_LABEL[elsewhere[0]]}, not for {STAGE_LABEL[stage]}: "
                                                        "different stages are not compared" if elsewhere else f"not stated on side {side}"))
            rows.append(row)
    return rows


def _shown(entry: dict | None):
    if not entry or not entry["values"]:
        return None
    vals = [v["raw"] if v["raw"] not in (None, "") else v["value"] for v in entry["values"]]
    return vals[0] if len(vals) == 1 else vals


def _ids(entry: dict | None) -> list[str]:
    return [i for v in (entry or {}).get("values", []) for i in v["evidence_ids"][:2]][:4]


def condition(view: dict, which: str) -> dict:
    """Comparability conditions beyond the evaluation record (Benchmark Reality Explorer): `pretraining_data` (what the
    linked pre-training configuration or the record names as pre-training source) and `adaptation` (zero-shot, few-shot,
    linear probe, kNN, frozen backbone or fine-tuned, from the protocol, else from a linked fine-tuning configuration)."""
    unknown = {"status": F.UNKNOWN, "value": None, "evidence_ids": [], "basis": None}
    if which == "pretraining_data":
        rel = next((r for r in view["relations"] if r["relation"] == "starts from"), None)
        pt = stage_fields(view, "pretraining")
        for path in PRETRAINING_DATA:
            if path in pt and pt[path]["status"] not in ("VARIES", F.CONFLICTING) and pt[path]["values"]:
                v = pt[path]["values"][0]
                own = view["experiment"] in v["experiments"]  # stated by this experiment's own record: as reported; else linked by rule
                return {"status": v["status"] if own else F.DERIVED, "value": v["raw"], "evidence_ids": v["evidence_ids"][:2],
                        "basis": f"the record states the {F.label(path)}" if own else (rel["basis"] if rel else F.label(path))}
        return unknown
    if which == "adaptation":
        protocol = stage_fields(view, "evaluation").get("evaluation.evaluation_protocol")
        if protocol and protocol["status"] not in ("VARIES", F.CONFLICTING) and protocol["values"]:
            v = protocol["values"][0]
            tags = G.protocol_tags(str(v["raw"]), need_eval_cue=False)
            kind = next((t for t in ADAPTATION if t in tags), None) or next((t for t in tags if t.endswith("-shot")), None)
            if kind:
                return {"status": v["status"], "value": kind, "evidence_ids": v["evidence_ids"][:2], "basis": "evaluation protocol"}
        rel = next((r for r in view["relations"] if r["relation"] == "evaluated after"), None)
        if rel:
            return {"status": F.DERIVED, "value": "fine-tuned", "evidence_ids": [], "basis": rel["basis"]}
        return unknown
    raise KeyError(which)


def inventory(paper_views: list[dict]) -> dict:
    """One paper's configurations across its experiments: distinct pre-training and fine-tuning configurations, the
    evaluation configurations, and how they are linked (answers "does this paper report several configurations?")."""
    by_id: dict[str, dict] = {}
    links: dict[str, set] = defaultdict(set)
    unlinked: dict[str, dict] = {}
    for v in paper_views:
        for c in v["configurations"]:
            by_id.setdefault(c["id"], c)
        for r in v["relations"]:
            if r["relation"] in ("evaluated after", "starts from"):
                links[r["to"]].add(v["experiment"])
        for u in v["unlinked"]:
            unlinked.setdefault(u["id"], u)
    stages = {s: [c for c in by_id.values() if c["stage"] == s] for s in ("pretraining", "finetuning")}
    evaluations = [c for c in by_id.values() if c["stage"] == "evaluation"]
    out = {s: [{**c, "linked_experiments": len(links.get(c["id"], ()))} for c in cs] for s, cs in stages.items()}
    out["not_linked"] = [u for i, u in unlinked.items() if i not in by_id]
    out["evaluation"] = len(evaluations)
    out["evaluation_settings"] = sorted({" · ".join(str(x) for x in [c["scope"] or "dataset not stated"] + [
        _shown(f) for f in c["fields"] if f["field"] in ("evaluation.dataset_split", "evaluation.evaluation_protocol")] if x) for c in evaluations})[:20]
    n = {s: len(stages[s]) + sum(u["stage"] == s for u in out["not_linked"]) for s in ("pretraining", "finetuning")}
    out["summary"] = (f"{n['pretraining']} pre-training and {n['finetuning']} fine-tuning configuration(s) are stated, "
                      f"with {len(evaluations)} evaluated experiment record(s).")
    out["multiple"] = {s: n[s] > 1 for s in n}
    return out
