"""Rule-based evidence extraction for explicitly stated configuration values (DATASET_SPEC §11, §42, §43).

Output is STUDY-LEVEL evidence (experiment_id=None): the paper states e.g. a learning rate, but
which experiment it belongs to is not decided here (spec §43: never assume a value from one
experiment applies to another). Values are verbatim spans; units come only from the text.
Precision over recall: ambiguous contexts (warm-up periods, contrastive-loss temperature,
top-k accuracy, related-work sections describing other papers) are skipped, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from services.canonicalization.normalize import NORMALIZATION_VERSION, canonical_unit, parse_number
from services.datasets.schemas import EvidenceAssertion, EvidenceStatus, Paper, ParsedDocument
from services.evidence.assertions import make, section_label

RULES_VERSION = "rules-hyperparams-0.1.0"
SKIP_SECTIONS = {"REFERENCES", "ACKNOWLEDGEMENTS", "RELATED_WORK"}

NUM = (
    r"(?:\d+(?:\.\d+)?\s*(?:\\times|×|\\cdot|x)\s*10\s*\^\s*\{?\s*[-−]?\s*\d+\s*\}?"
    r"|\d+(?:\.\d+)?\s*[eE]\s*\^\s*\{?\s*[-−+]?\s*\d+\s*\}?"  # TeX-style 1e^{-4}
    r"|\d+(?:\.\d+)?\s*[eE]\s*[-−+]?\s*\d+|10\s*\^\s*\{?\s*[-−]\s*\d+\s*\}?|\d*\.\d+|\d+)"
)
INT = r"(?:\d{1,3}(?:(?:,|\{,\})\d{3})+|\d+)"
POW_PREFIX = r"(?:\d+\s*\^\s*\{?\d+\}?\s*=\s*)?"  # "2^{19}=524{,}288": keep the explicit value
LINK = r"(?:\s+(?:of|is|was|to|at|set to|equal to|fixed to|fixed at))?\s*[=:]?\s*"


def _p(s: str, flags: int = re.I) -> re.Pattern:
    return re.compile(s, flags)


@dataclass(frozen=True)
class Rule:
    name: str
    field: str
    pattern: re.Pattern
    kind: str = "number"  # number | count | text | lora_rank | gpu
    unit: str | None = None  # unit stated by the pattern itself
    requires: re.Pattern | None = None  # sentence must also match
    forbids: re.Pattern | None = None  # sentence must not match


WARMUP = _p(r"warm-?\s?up")
RULES: list[Rule] = [
    Rule("lr", "training.learning_rate", _p(rf"\b(?:learning[ -]rates?|lr){LINK}(?P<value>{NUM})(?![\d.])"),
         forbids=_p(r"\bdecay(?:s|ed|ing)? (?:the )?learning rate (?:by|to)\b")),
    Rule("batch", "training.batch_size",
         _p(rf"\bbatch[ -]sizes?{LINK}(?P<value>{INT})(?![\d.])(?:\s+(?P<unit>sequences|examples|samples|tokens|images|sentences|pairs))?"), kind="count"),
    Rule("epochs_n", "training.epochs", _p(rf"\b(?P<value>{INT})\s+(?:training\s+|fine-?tuning\s+|full\s+)?epochs\b"), kind="count", unit="epochs", forbids=WARMUP),
    # explicit link required: "epochs: 3", "number of epochs to 3" (not "decayed at epoch 8")
    Rule("epochs_kw", "training.epochs", _p(rf"\b(?:number of\s+)?(?:training\s+)?epochs?(?:\s+(?:of|is|was|to|set to)\s*|\s*[=:]\s*)(?P<value>{INT})(?![\d.])"),
         kind="count", unit="epochs", forbids=WARMUP),
    Rule("steps_for", "training.steps",
         _p(rf"\bfor\s+(?:a total of\s+)?{POW_PREFIX}(?P<value>{INT}(?:\.\d+)?\s*[kKM]?)\s+(?:training\s+|optimization\s+|update\s+|gradient\s+|fine-?tuning\s+|pre-?training\s+)?(?P<unit>steps|iterations|updates)\b"),
         kind="count", forbids=WARMUP),
    Rule("steps_qualified", "training.steps",
         _p(rf"\b{POW_PREFIX}(?P<value>{INT}(?:\.\d+)?\s*[kKM]?)\s+(?:training|optimization|gradient|fine-?tuning|pre-?training)\s+(?P<unit>steps|iterations|updates)\b"),
         kind="count", forbids=WARMUP),
    Rule("warmup_n", "training.warmup_steps", _p(rf"\b(?P<value>{INT}\s*[kK]?)\s+(?:linear\s+)?warm-?\s?up\s+(?P<unit>steps|iterations|updates|epochs)\b"), kind="count"),
    Rule("warmup_kw", "training.warmup_steps",
         _p(rf"\bwarm-?\s?up(?:\s+(?P<unit1>steps|iterations|updates|epochs|period|phase))?(?:\s+(?:of|for|to|is|was|set to|=|:)|\s*\(set to)\s*(?:the first\s+)?"
            rf"(?P<value>{INT}\s*[kK]?|10\s*\^\s*\{{?\d+\}}?)(?![\d.])(?:\s+(?P<unit>steps|iterations|updates|epochs))?"), kind="count"),
    Rule("warmup_ratio", "training.warmup_ratio", _p(rf"\bwarm-?\s?up\s+(?:ratio|proportion|fraction){LINK}(?P<value>0?\.\d+)")),
    Rule("weight_decay", "training.weight_decay", _p(rf"\bweight[ -]decay(?:\s+(?:coefficient|rate|factor))?{LINK}(?P<value>{NUM})(?![\d.])")),
    Rule("dropout", "training.dropout", _p(rf"\bdropout(?:\s+(?:rate|probability|ratio|value))?{LINK}(?P<value>0?\.\d+|0|1\.0)(?![\d.])")),
    Rule("optimizer", "training.optimizer", re.compile(r"\b(?P<value>AdamW|Adam|Adafactor|SGD|LAMB|RMSProp|RMSprop|Adagrad|Lion|LARS)\b"), kind="text", requires=_p(r"optimi[sz]")),
    Rule("gpu", "compute.gpu_model",
         # boundary before the model name ("INT4"/"GPT4" must not match T4); "4\timesRTX 3090" allowed
         re.compile(r"(?:\b(?P<count>\d+)\s*(?:×|x|\\times)?\s*)?(?:NVIDIA\s+)?(?:Tesla\s+|GeForce\s+|Quadro\s+)?(?:(?<=\\times)|(?<![A-Za-z0-9]))(?P<value>A100|H100|H200|V100|A6000|A40|A10G?|L40S?|T4|P100|RTX\s?\d{4}(?:\s?Ti)?|TITAN\s?(?:RTX|Xp?|V))\b"),
         kind="gpu"),
    Rule("tpu", "compute.accelerator", re.compile(r"\b(?P<value>TPU\s?v\d\w*)\b"), kind="text"),
    Rule("seed", "training.seed", _p(rf"\b(?:random\s+)?seed{LINK}(?P<value>\d+)(?![\d.])"), kind="count"),
    Rule("precision", "training.precision", _p(r"\b(?P<value>fp16|bf16|bfloat16|fp32|float16|float32|fp8|mixed[ -]precision)\b"), kind="text",
         requires=_p(r"train|precision|mixed|comput|\buse[sd]?\b")),
    Rule("temperature", "inference.temperature", _p(rf"\btemperature(?:\s+(?:parameter|value|setting))?{LINK}(?P<value>\d*\.?\d+)(?![\d.])"),
         requires=_p(r"sampl|decod|generat|inference"), forbids=_p(r"contrastive|softmax|distill|infonce|\bloss\b|\\tau")),
    Rule("top_p", "inference.top_p", _p(rf"\b(?:top[- ]?p|nucleus sampling(?: with p)?){LINK}(?P<value>0?\.\d+|1\.0)(?![\d.])")),
    Rule("top_k", "inference.top_k", _p(rf"\btop[- ]?k(?:\s+sampling)?{LINK}(?P<value>\d+)(?![\d.])"), kind="count",
         requires=_p(r"sampl|decod|generat"), forbids=_p(r"accura|retriev|recall|precision@")),
    Rule("lora_rank", "training.lora", _p(rf"\b(?:rank|r){LINK}(?P<value>\d+)(?![\d.])"), kind="lora_rank", requires=_p(r"\bLoRA\b|low[- ]rank adapt")),
    Rule("grad_accum", "training.gradient_accumulation", _p(rf"\bgradient[ -]accumulation(?:\s+steps?)?{LINK}(?P<value>\d+)(?![\d.])"), kind="count"),
    Rule("grad_clip", "training.gradient_clipping",
         _p(r"\b(?:gradient|grad(?:ient)?[- ]norm)\s+clipping(?:\s+(?:threshold|value|norm))?(?:\s+(?:of|at|to|with (?:a )?(?:max(?:imum)? )?norm of))?\s*[=:]?\s*(?P<value>\d*\.?\d+)(?![\d.])")),
    Rule("grad_clip_b", "training.gradient_clipping",
         _p(r"\bclip(?:ped|ping|s)?\s+(?:the\s+)?(?:gradients?|grad)(?:\s+norms?)?\s+(?:at|to)\s+(?:a\s+)?(?:max(?:imum)?\s+)?(?:(?:l2\s+)?norm\s+of\s+)?(?P<value>\d*\.?\d+)(?![\d.])")),
]

TEXT_ALIASES = {  # explicit, versioned alias table (spec §13: record the normalization rule)
    "adamw": "AdamW", "adam": "Adam", "adafactor": "Adafactor", "sgd": "SGD", "lamb": "LAMB", "rmsprop": "RMSProp",
    "adagrad": "Adagrad", "lion": "Lion", "lars": "LARS", "bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16",
    "float16": "float16", "fp32": "float32", "float32": "float32", "fp8": "fp8", "mixedprecision": "mixed_precision",
    "mixed-precision": "mixed_precision",
}

_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|vs|approx|cf|resp|No|Ref|Refs|App|Appx)\.|\b[A-Z]\.)$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z(\[“\"'])", text):
        end = m.end()
        if _ABBREV.search(text[start:end]):
            continue
        spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def interpret(rule: Rule, m: re.Match) -> Iterator[tuple[str, str, object, str | None, str]]:
    """-> (field, raw_value, normalized_value, unit, normalization_rule)."""
    raw = m.group("value").strip()
    if rule.kind in ("number", "count"):
        parsed = parse_number(raw.replace("{,}", ","))
        if parsed is None:
            return
        value, nrule = parsed
        if rule.kind == "count":
            if value != int(value) or value < 0:
                return
            value = int(value)
        groups = m.groupdict()
        unit = rule.unit or canonical_unit(groups.get("unit")) or canonical_unit(groups.get("unit1"))
        yield rule.field, raw, value, unit, nrule
    elif rule.kind == "text":
        key = re.sub(r"\s+", "", raw.lower())
        yield rule.field, raw, TEXT_ALIASES.get(key, raw), None, "alias_table" if key in TEXT_ALIASES else "verbatim"
    elif rule.kind == "lora_rank":
        yield rule.field, raw, {"rank": int(raw)}, None, "lora_rank"
    elif rule.kind == "gpu":
        yield "compute.gpu_model", raw, re.sub(r"\s+", " ", raw.upper()), None, "gpu_model_upper"
        if m.group("count"):
            yield "compute.gpu_count", m.group("count"), int(m.group("count")), "count", "integer"


def extract_document(doc: ParsedDocument, paper: Paper, run_id: str, retrieved_at: datetime | None) -> list[EvidenceAssertion]:
    sections = {s.section_id: s for s in doc.sections}
    version = doc.source_record_id.rsplit(paper.arxiv_id or "", 1)[-1] or None
    out: dict[str, EvidenceAssertion] = {}
    for para in doc.paragraphs:
        sec = sections.get(para.section_id or "")
        if sec and sec.section_type in SKIP_SECTIONS:
            continue
        spans = sentence_spans(para.text)
        for rule in RULES:
            for m in rule.pattern.finditer(para.text):
                s0, s1 = next(((a, b) for a, b in spans if a <= m.start() < b), (0, len(para.text)))
                sentence = para.text[s0:s1]
                if (rule.requires and not rule.requires.search(sentence)) or (rule.forbids and rule.forbids.search(sentence)):
                    continue
                for field, raw, value, unit, nrule in interpret(rule, m):
                    ev = make(
                        source_name="ARXIV",
                        source_url=doc.source_url,
                        content_hash=doc.source_sha256,
                        source_record_id=doc.source_record_id,
                        paper_id=paper.paper_id,
                        study_id=paper.study_id,
                        experiment_id=None,
                        field_name=field,
                        extracted_value=raw,
                        normalized_value=value,
                        unit=unit,
                        normalization_rule=f"{rule.name}:{nrule}",
                        normalization_version=NORMALIZATION_VERSION,
                        evidence_text=sentence,
                        source_location=f"{doc.source_url}#{para.paragraph_id} (Section {section_label(doc, para.section_id)})",
                        section=para.section_id,
                        paragraph=para.paragraph_id,
                        char_start=s0,
                        char_end=s1,
                        source_version=version,
                        retrieval_timestamp=retrieved_at,
                        extraction_model=RULES_VERSION,
                        extraction_run_id=run_id,
                        status=EvidenceStatus.MACHINE_EXTRACTED,
                        redistribution_allowed=doc.license.redistribution_allowed,
                    )
                    out.setdefault(ev.evidence_id, ev)
    return list(out.values())
