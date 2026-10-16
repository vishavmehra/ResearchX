"""Field-level evidence grounding (docs/ARCHITECTURE.md §61). Fixes the 2026-10-04 false abstention, where correct
claims were dropped because a model quoted "520\\times 520" as "520×520" and the run abstained.

A requested parameter reaches an answer only through this chain:
    source -> evidence span -> structured fact (field, value) -> deterministic validation -> per-field sufficiency
Validation is deterministic: the span must occur in the cited source (formatting-insensitive), the claimed value must
equal a number in the span (0.02 = 2e-2 = 2x10^-2), and the field cue nearest that number must be the claimed field's
cue, so "weight decay as 0.02" can never support "learning rate = 0.02". Every fact also carries its configuration
match (model / dataset), so a related paper's numbers never answer for the wrong experiment. No model prose is trusted:
answers are rendered from validated facts only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from services.canonicalization.entities import key as ekey
from services.canonicalization.normalize import canonical_label, parse_number
from services.extraction.rules import NUM

VERSION = "grounding-0.4.0"  # 0.3.0: runtime checks for external text; 0.4.0: seeds, splits, dataset versions, protocols, metric definitions, checkpoints

# ---------------------------------------------------------------------------------------------- text canonical form
# Parsed papers keep LaTeX symbol macros that models and users write as Unicode. Longer macros first.
LATEX = (("\\times", "×"), ("\\approx", "≈"), ("\\cdot", "·"), ("\\leq", "≤"), ("\\geq", "≥"), ("\\le", "≤"), ("\\ge", "≥"),
         ("\\pm", "±"), ("\\sim", "~"), ("\\%", "%"), ("\\infty", "∞"), ("\\rightarrow", "→"), ("\\to", "→"), ("\\alpha", "α"),
         ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"), ("\\epsilon", "ε"), ("\\eta", "η"), ("\\lambda", "λ"), ("\\mu", "μ"),
         ("\\sigma", "σ"), ("\\tau", "τ"), ("\\theta", "θ"))


def canon(text) -> str:
    """Formatting-insensitive form for verbatim checks: LaTeX symbols as Unicode, content-free LaTeX commands, braces and
    math delimiters dropped, case and spacing around punctuation ignored. Words and numbers still have to match."""
    t = unicodedata.normalize("NFKC", "" if text is None else str(text))  # record fields can be numbers (model_version 2.0)
    for macro, symbol in LATEX:
        t = t.replace(macro, symbol)
    t = re.sub(r"·\s*(?=10\s*\^)", "×", t)  # 3\cdot 10^{-5} is 3×10^-5 (the number pattern knows ×, not ·)
    t = re.sub(r"\\[A-Za-z]+\s*", " ", t)
    t = re.sub(r"[{}$]", "", t).casefold()
    t = re.sub(r"\s*([^\w\s])\s*", r"\1", t)
    return " ".join(t.split())


# ---------------------------------------------------------------------------------------------- field lexicon
# canonical field -> (label, cue regex written next to the value in papers). Matching runs on canon() text.
FIELDS: dict[str, tuple[str, str]] = {
    "training.learning_rate": ("learning rate", r"(?:initial |base |peak |maximum |max |starting )?(?:learning[ -]rates?|\blr\b)"),
    "training.batch_size": ("batch size", r"(?:mini-?|global |total |effective )?batch(?:[ -]?sizes?\b|es of\b)|mini-?batch(?:es)?"),
    "training.weight_decay": ("weight decay", r"weight[ -]decay|l2[ -]regulari[sz]ation"),
    "training.epochs": ("number of epochs", r"\bepochs?\b"),
    "training.steps": ("training iterations", r"\b(?:training )?(?:iterations?|iters?|steps|updates)\b"),
    "training.warmup_steps": ("warm-up", r"warm-?\s?up(?:\s?(?:steps?|iterations?|iters?|updates|epochs?|ratio|period))?"),
    "training.dropout": ("dropout", r"\bdropout\b"),
    "training.momentum": ("momentum", r"\bmomentum\b"),
    "training.optimizer": ("optimizer", r"optimi[sz](?:er|ation|ed)"),
    "training.learning_rate_schedule": ("learning-rate schedule", r"schedul\w*|anneal\w*|(?<!weight )decay"),
    "compute.gpu_model": ("hardware", r"\bgpus?\b|\btpus?\b|accelerators?"),
    "model.parameter_count": ("parameter count", r"\bparam(?:eter)?s\b"),
    "training.seed": ("random seed", r"\b(?:random[ -])?seeds?\b|\bseeded\b"),
    # text-valued conditions (0.4.0): validated by their own rules (validate_text), outside the numeric nearest-cue contest
    "evaluation.dataset_split": ("evaluation split", r"\b(?:splits?|subsets?|partitions?|test-?dev|minival|held-?out)\b|\b(?:test|val(?:idation)?|dev)(?:\s?set|\s?split)\b"),
    "dataset.split": ("training split", r"\b(?:train(?:ing)?|trainval|train\+val)(?:\s?set|\s?split|\s?\d{4})\b"),
    "dataset.dataset_version": ("dataset version", r"\bversions?\b|\bv\d+(?:\.\d+)+\b|\breleases?\b|[a-z]-\d+k\b"),
    "evaluation.evaluation_protocol": ("evaluation protocol", r"protocols?|zero-?shot|few-?shot|\d-?shot|linear[ -]?prob\w*|single-?scale|multi-?scale|"
                                                              r"test-?time augmentation|\btta\b|\d+-?crops?|(?:single|multi)[ -]?crop|cent(?:er|re)[ -]?crop|"
                                                              r"sliding[ -]?window|checkpoints?|early stopping|\bema\b|k-?nn"),
    "evaluation.metric_definition": ("metric definition", r"\bmetrics?\b|we report|is reported|computed|calculated|measured|defined as|in terms of"),
    "model.checkpoint": ("checkpoint", r"\bcheckpoints?\b|\bckpt\b|\bweights\b|initiali[sz]ed (?:from|with)"),
    "evaluation.score": ("score", ""),  # table cells: located by the dataset pipeline, validated against the table
}
CATEGORICAL = {
    "training.optimizer": r"\b(?:adamw|adam|adafactor|sgd|lamb|rmsprop|adagrad|lion|lars)\b",
    "training.learning_rate_schedule": r"\b(?:cosine|linear|polynomial|poly|step|exponential|constant|one-?cycle|inverse square root)\b",
    "compute.gpu_model": r"\b(?:a100|h100|v100|p100|t4|a6000|rtx ?\d{4}\w*|tpu ?v\d\w*|mi\d{3}\w*)\b",
}
TEXT_FIELDS = ("evaluation.dataset_split", "dataset.split", "dataset.dataset_version", "evaluation.evaluation_protocol",
               "evaluation.metric_definition", "model.checkpoint")
VERBATIM = frozenset(CATEGORICAL) | frozenset(TEXT_FIELDS) | {"training.seed"}  # values the extractor keeps as written (no unit stripping)
CUES = {f: re.compile(c) for f, (_, c) in FIELDS.items() if c and f not in TEXT_FIELDS}
TEXT_CUES = {f: re.compile(FIELDS[f][1]) for f in TEXT_FIELDS}
SELECT_CUES = {**CUES, **TEXT_CUES}  # what a passage must name to be read for a field (services/runtime/reader.relevant)
HINTS = {  # extraction-prompt guidance per field (the label stays short for people)
    "training.seed": "random seed(s) or seed policy, e.g. 'seed 0', 'seeds 0, 1, 2', '5 random seeds'",
    "evaluation.dataset_split": "split the results are reported on, e.g. 'validation set', 'test-dev', 'val2017'",
    "dataset.split": "split the model is trained or fine-tuned on, e.g. 'train', 'train+val'",
    "dataset.dataset_version": "dataset version or subset, e.g. 'COCO 2017', 'SQuAD v1.1', 'ImageNet-21K'",
    "evaluation.evaluation_protocol": "how results are obtained: zero/few-shot, linear probe, fine-tuned, single/multi-scale testing, "
                                      "test-time augmentation, crops, sliding window, checkpoint selection",
    "evaluation.metric_definition": "how the metric is defined or computed, e.g. 'top-1 accuracy', 'mIoU (single-scale)', 'case-sensitive SacreBLEU'",
    "model.checkpoint": "the checkpoint or weights evaluated or initialised from",
}
QUERY_FIELDS = (  # question wording -> field; earlier patterns consume their words first ("learning rate schedule")
    (r"(?:learning[ -]rate\s+)?schedul\w*|annealing", "training.learning_rate_schedule"),
    (r"learning[ -]rates?|\blr\b", "training.learning_rate"),
    (r"batch[ -]?sizes?|mini-?batch", "training.batch_size"),
    (r"weight[ -]decay", "training.weight_decay"),
    (r"optimi[sz]ers?", "training.optimizer"),
    (r"warm-?\s?up(?:\s+(?:steps|iterations|epochs|ratio|period))?", "training.warmup_steps"),  # before epochs/steps
    (r"\bepochs?\b", "training.epochs"),
    (r"\b(?:training\s+)?(?:iterations|steps)\b", "training.steps"),
    (r"\bdropout\b", "training.dropout"),
    (r"\bmomentum\b", "training.momentum"),
    (r"\bgpus?\b|\bhardware\b|\baccelerators?\b", "compute.gpu_model"),
    (r"\bparameter count\b|\bnumber of parameters\b|\bmodel size\b", "model.parameter_count"),
    (r"\b(?:random[ -])?seeds?\b", "training.seed"),
    (r"\bevaluation protocols?\b|\bprotocols?\b|test-?time augmentation|multi-?scale (?:testing|inference|evaluation)|checkpoint selection",
     "evaluation.evaluation_protocol"),
    (r"\bmetric definitions?\b|\bdefinition of the metric\b|how (?:is|was|were) the metrics? (?:computed|calculated|defined)",
     "evaluation.metric_definition"),
    (r"\bdataset versions?\b|\bversions? of the (?:dataset|benchmark)\b", "dataset.dataset_version"),
    (r"\btraining (?:splits?|sets?)\b", "dataset.split"),
    (r"\b(?:evaluation |test |validation |dataset |data )?splits?\b|\b(?:test|evaluation) sets?\b", "evaluation.dataset_split"),
    (r"\bcheckpoints?\b", "model.checkpoint"),
)
METRIC = re.compile(r"\b(top[- ]?[15] accuracy|accuracy|top[- ]?[15]|miou|m?ap(?:@[\d.:]+)?|f1(?: score)?|bleu(?:-\d)?|rouge(?:-\w+)?|perplexity|exact match|"
                    r"auc|wer|psnr|ssim|fid|pass@\d+|success rate|win rate|score)\b", re.I)
VERB = r"(?:train(?:ed|ing)?|fine-?tun(?:e|ed|ing)|pre-?train(?:ed|ing)?|evaluat(?:e|ed|ing)|test(?:ed|ing)?|run|ran|does|did|do|is|are|was|were|for|of|by|use[ds]?|using)"
STOP = {"the", "a", "an", "model", "models", "method", "approach", "network", "with", "and", "of", "on", "for", "in", "to", "ours", "our"}
# Generic nouns the entity table also lists as benchmark names ("Benchmark", "Dataset"): never a dataset constraint.
GENERIC_DATASETS = {"benchmark", "benchmarks", "dataset", "datasets", "data", "corpus", "task", "tasks", "test", "testset", "train", "validation",
                    "average", "overall", "mean", "all"}
NOT_A_DATASET = re.compile(r"\b(?:gpus?|tpus?|nodes?|machines?|cores?|devices?|cards?|chips?|hours?|days?|epochs?|steps?)\b", re.I)


def _loose(name: str) -> str:
    """Pattern for a name that tolerates punctuation/spacing variants ("CodeSearchNet - JavaScript" ~ "CodeSearchNet JavaScript")."""
    return r"[\W_]*".join(re.escape(p) for p in re.split(r"[\W_]+", name) if p)


@dataclass
class Request:
    fields: list[str]
    model: str | None = None
    dataset: str | None = None
    metric: str | None = None

    @property
    def labels(self) -> list[str]:
        return [(self.metric or FIELDS[f][0]) if f == "evaluation.score" else FIELDS[f][0] for f in self.fields]


def parse_request(query: str, datasets: list[str]) -> Request:
    """Structured reading of a question: requested fields, model mention, dataset/benchmark, metric. `datasets` are the
    dataset/benchmark entity names the corpus recognised in the question (canonical_entities)."""
    q = " ".join((query or "").split()).rstrip("?. ")
    # what is asked comes before the subject's verb: "What Accuracy does Dropout 0.1 achieve" asks a score, not dropout
    asked = re.split(r"\b(?:was|were|is|are|does|did|do|has|have|had)\b", q, maxsplit=1, flags=re.I)[0]
    probe = asked if METRIC.search(asked) or any(re.search(p, asked, re.I) for p, _ in QUERY_FIELDS) else q
    fields, rest = [], probe
    for pat, name in QUERY_FIELDS:
        if re.search(pat, rest, re.I):
            fields.append(name)
            rest = re.sub(pat, " ", rest, flags=re.I)
    metric = None
    if not fields:
        m = METRIC.search(probe)
        if m and re.search(r"\b(achiev|reach|score|get|obtain|perform|report|attain)\w*", q, re.I):
            fields, metric = ["evaluation.score"], m.group(1)
    # entity-table names that are generic words or metric names ("Benchmark", "Accuracy") never name the dataset
    names = [d for d in datasets if d and _loose(d) and ekey(d) not in GENERIC_DATASETS and not METRIC.fullmatch(d.strip())]

    def found_in(text: str) -> list[str]:
        return sorted((d for d in names if re.search(rf"(?<!\w){_loose(d)}(?!\w)", text, re.I)), key=len, reverse=True)

    ons = list(re.finditer(r"\s+on\s+(?:the\s+)?", q, re.I))
    tail = q[ons[-1].end():] if ons else ""
    free = re.sub(r"\s+(?:benchmark|dataset|data ?set|task|leaderboard|test set|split)s?$", "", tail, flags=re.I).strip(" ,;")
    dataset, head = None, q
    if tail and found_in(tail):  # "... on <dataset>": the dataset is named after the last "on"
        dataset, head = found_in(tail)[0], q[: ons[-1].start()]
    elif tail and re.search(r"[A-Z0-9]", free) and not NOT_A_DATASET.search(free) and ekey(free) not in GENERIC_DATASETS:
        # ponytail: a name the corpus does not know ("on the Zentrix-77 benchmark") still constrains the configuration;
        # "on 8 GPUs" does not. Upgrade path: a dataset classifier if free-text names prove noisy.
        dataset, head = free, q[: ons[-1].start()]
    elif found_in(q):
        dataset = found_in(q)[0]
        m = re.search(rf"\s+(?:on|for|with|in)\s+(?:the\s+)?{_loose(dataset)}", q, re.I)
        head = q[: m.start()] if m else q[: re.search(_loose(dataset), q, re.I).start()]
    elif ons:
        head = q[: ons[-1].start()]
    head = re.sub(r"\s+(?:reach|reaches|achieve|achieves|get|gets|obtain|obtains|score|scores|attain|have|has)\s*$", "", head, flags=re.I)
    # a verb ending the phrase counts only in lower case: "for BERT pre-training" yes, "train RL Training" no
    verbs = [v for v in re.finditer(rf"\b{VERB}(?:\s+(?:the\s+|a\s+|an\s+)?|$)", head, re.I)
             if not (v.end() == len(head) and v.group(0)[:1].isupper())]
    model = None
    for i in range(len(verbs) - 1, -1, -1):  # the model is the last non-empty phrase after a verb ("for BERT pre-training")
        seg = head[verbs[i].end(): verbs[i + 1].start() if i + 1 < len(verbs) else len(head)].strip(" ,")
        if tokens(seg) and not re.fullmatch(r"(?i)(what|which|how)\b.*", seg):
            model = seg
            break
    return Request(fields=fields, model=model, dataset=dataset, metric=metric)


def tokens(text: str | None) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", canon(text)) if t and t not in STOP}


ACRONYM = re.compile(r"((?:[A-Za-z][\w'-]*\s+){0,7}[A-Za-z][\w'-]*)\s*\(([A-Z][A-Za-z0-9-]{1,11})\)")


def acronyms(text: str | None) -> dict[str, str]:
    """Aliases a source establishes itself, e.g. "Object-Contextual Representations (OCR)" -> {"ocr": "object contextual
    representations"}. Only acronyms whose letters are the initials of the preceding words count; nothing is invented."""
    out = {}
    for long_form, abbr in ACRONYM.findall(text or ""):
        target = re.sub(r"[^a-z]", "", abbr.lower())
        parts = [p for p in re.split(r"[\s-]+", long_form.strip()) if p]
        for i in range(len(parts)):
            if len(parts) - i >= 2 and "".join(p[0].lower() for p in parts[i:]) == target:
                out[target] = " ".join(parts[i:]).lower()
                break
    return out


def coverage(model: str | None, names: list[str | None], aliases: dict[str, str]) -> float | None:
    """Share of the question's model tokens found in the candidate's names, honouring source-established acronyms in
    both directions. None when the question names no model."""
    want = tokens(model)
    if not want:
        return None
    have = set().union(*(tokens(n) for n in names if n)) if names else set()
    hit = set()
    for abbr, long_form in aliases.items():
        lt = tokens(long_form)
        if abbr in want and lt and lt <= have:
            hit.add(abbr)  # question says OCR, source writes Object-Contextual Representations
        if lt and lt <= want and abbr in have:
            hit |= lt  # question spells it out, source writes OCR
    hit |= want & have
    if not any(len(t) >= 2 and re.search(r"[a-z]", t) for t in hit & want):
        return 0.0  # a shared version number or letter alone ("GPT-3" vs "Llama 3") is not the same model
    return len(hit & want) / len(want)


def _ds_level(asked: str, name: str) -> int:
    """2: `name` is the asked dataset or a variant of it ("ADE20K" ~ "ADE20K val"); 1: `name` is broader ("CodeSearchNet"
    for "CodeSearchNet - JavaScript"); 0: different. Word tokens, so "Java" never matches "JavaScript", nor CIFAR-10 CIFAR-100."""
    q, k = tokens(asked), tokens(name)
    if not q or not k:
        return 0
    if q <= k:
        return 2
    return 1 if k < q and any(len(t) >= 3 and re.search(r"[a-z]", t) for t in k) else 0


VARIANT = re.compile(r"(?:[a-z]|tiny|small|base|medium|large|huge|giant|x+l|v\d+|w\d+|b\d+|\d+(?:\.\d+)?[kmbt]?)")


def variant_conflict(model: str | None, names: list[str | None]) -> bool:
    """The question names a size/variant marker the candidate lacks while the candidate names a different one:
    ViT-L/16 vs ViT-B/16, ViT-B/16 vs ViT-B/32, ResNet-50 vs ResNet-101, Llama 2 7B vs 13B, BERT-large vs BERT-base."""
    want = tokens(model)
    have = set().union(*(tokens(n) for n in names if n)) if names else set()
    asked = {t for t in want - have if VARIANT.fullmatch(t)}
    return bool(asked) and any(VARIANT.fullmatch(t) for t in have - want)


def dataset_match(dataset: str | None, names: list[str | None]) -> int | None:
    """Best _ds_level over the candidate's dataset names; None when the question names no dataset."""
    if not dataset:
        return None
    return max((_ds_level(dataset, n) for n in names if n), default=0)


def dataset_exact(dataset: str | None, names: list[str | None]) -> bool | None:
    """The candidate names the asked dataset itself ("ImageNet"), not a narrower variant ("ImageNet - 1% labeled data")."""
    if not dataset:
        return None
    return any(tokens(n) == tokens(dataset) for n in names if n)


def display(raw) -> str:
    """A value as written in the source, without LaTeX markup: "73.9\\pm3.1" -> "73.9±3.1"."""
    t = unicodedata.normalize("NFKC", "" if raw is None else str(raw))
    for macro, symbol in LATEX:
        t = t.replace(macro, symbol)
    return " ".join(re.sub(r"[{}$]", "", t).split())


def config_match(req: Request, names: list[str | None], datasets: list[str | None], aliases: dict[str, str]) -> tuple[str, float]:
    """EXACT: every model token and the dataset (or a variant of it) match; STRONG: most of the model and the dataset, or
    the whole model on a broader dataset (or all of the model when no dataset is asked); RELATED: dataset or half the
    model; WEAK: neither. Only the best class present is used to answer, so STRONG never outvotes EXACT."""
    cov, dm = coverage(req.model, names, aliases), dataset_match(req.dataset, datasets)
    full = cov is None or cov >= 0.999
    score = (cov if cov is not None else 0.5) + {None: 0.0, 0: -1.0, 1: 0.5, 2: 1.0}[dm]
    if full and dm in (2, None) and (cov is not None or dm):
        return "EXACT", score
    partial_ok = (cov or 0) >= 0.5 and not variant_conflict(req.model, names)  # another variant is never "close"
    if (partial_ok and dm in (2, None)) or (full and cov is not None and dm == 1):
        return "STRONG", score
    if dm or (cov or 0) >= 0.5:
        return "RELATED", score
    return "WEAK", score


RANK = {"EXACT": 3, "STRONG": 2, "RELATED": 1, "WEAK": 0}

# ---------------------------------------------------------------------------------------------- validation


def _numbers(text: str) -> list[tuple[float, int, int]]:
    out = []
    for m in re.finditer(NUM + r"(?:\s?[kmb]\b)?", text):
        parsed = parse_number(m.group(0).replace("×", "x").replace(" ", ""))
        if parsed:
            out.append((parsed[0], m.start(), m.end()))
    return out


def number_of(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    parsed = parse_number(canon(str(value)).replace("×", "x").replace(" ", "").removesuffix("%"))  # "83.2%" claims the 83.2 of its span
    return parsed[0] if parsed else None


def _table_cell(field_name: str, match, source_text: str) -> bool:
    """A bare table cell ("4096"): the claimed field's cue must label the row or the column that holds the value."""
    rows = [[canon(c) for c in line.split("|")] for line in (source_text or "").split("\n") if "|" in line]
    if len(rows) < 2:
        return False
    header = rows[0]
    return any(match(cell) and any(CUES[field_name].search(label) for label in [r[0]] + header[j:j + 1])
               for r in rows for j, cell in enumerate(r) if j > 0)


UNIT_CUES = {"training.warmup_steps": {"training.steps", "training.epochs"}}  # warm-up is counted in steps or epochs

# ---------------------------------------------------------------------------------------------- conditions (0.4.0)
# Seeds, splits, dataset versions, evaluation protocols, metric definitions and checkpoints are words, not numbers next
# to a cue: each has its own vocabulary, and a value is accepted only when it is in the quoted span, the span is in the
# source, and the span carries the field's own context (a split noun beside the split word, an evaluation cue beside a
# crop or flip, a reporting cue beside a metric name). Values are normalised, so "val" and "validation set" compare equal.
SPLIT_WORD = (r"test-?dev|test-?std|test-?challenge|(?:test|dev)-(?:clean|other)|minival|trainval|train\+val|"
              r"(?:train|val|test)\s?(?:19|20)\d{2}|held-?out|validation|val|development|dev|testing|test|training|train")
SPLIT = re.compile(rf"(?<![a-z0-9])({SPLIT_WORD})(?![a-z0-9])")
SPLIT_NOUN = re.compile(r"(?<![a-z])(?:sets?|splits?|subsets?|partitions?|data|images|examples|samples|server|portion)(?![a-z])")
UNAMBIGUOUS_SPLIT = re.compile(r"test-?dev|test-?std|test-?challenge|-(?:clean|other)|minival|trainval|train\+val|\d{4}$|held-?out")
TRAIN_SPLITS = {"train", "train+val", "trainval"}
EVAL_CUE = re.compile(r"test(?:ing)?|inference|evaluat\w*|validation|\bval\b|at test time|during inference|reported|we report|results?")
TRAINING_AUG = re.compile(r"random(?:ly)?\s?(?:resized\s?|horizontal(?:ly)?\s?)?(?:crop|flip)|multi-?scale training|training augmentation")
PROTOCOL_TAGS: tuple[tuple[str, str, bool], ...] = (  # (canonical tag, pattern on canon text, needs an evaluation cue in the span)
    ("zero-shot", r"zero-?shot|(?<![\d.])0-?shot", False),
    ("{}-shot", r"(?<![\d.])([1-9]\d{0,2})-?shot", False),
    ("few-shot", r"few-?shot", False),
    ("linear probe", r"linear[ -]?(?:probe|probing|evaluation|classification on frozen)", False),
    ("kNN", r"(?<![a-z])k-?nn(?![a-z])|k-?nearest[ -]neighbou?rs?", False),
    ("fine-tuned", r"(?:end-to-end |full )?fine-?tun(?:ed|ing)", False),
    ("frozen backbone", r"frozen (?:backbone|features|encoder|representations?)", False),
    ("single-scale", r"single-?scale", True),
    ("multi-scale", r"multi-?scale", True),
    ("test-time augmentation", r"test-?time augmentation|(?<![a-z])tta(?![a-z])", False),
    ("flip", r"(?:horizontal(?:ly)? )?flip(?:ped|ping|s)?", True),
    ("center crop", r"cent(?:er|re)[ -]?crop(?:ped|ping|s)?", True),
    ("single-crop", r"single[ -]?crop|(?<![\d.])1-?crop", True),
    ("{}-crop", r"(?<![\d.])([2-9]|1\d|[2-9]\d)-?crops?", True),
    ("multi-crop", r"multi-?crop", True),
    ("sliding window", r"sliding[ -]?window", True),
    ("ensemble", r"ensembl(?:e|es|ing)", True),
    ("cross-validation", r"cross-?validat\w*|(?<![\d.])\d{1,2}-?fold", False),
    ("greedy decoding", r"greedy (?:decoding|search)", False),
    ("beam search", r"beam search|beam (?:size|width)", False),
    ("best checkpoint on validation", r"best (?:validation |dev )?checkpoint|checkpoint (?:with|of) the best|model selection on (?:the )?(?:validation|dev)", False),
    ("last checkpoint", r"(?:last|final) checkpoint", False),
    ("EMA weights", r"(?<![a-z])ema(?![a-z])|exponential moving average", False),
    ("early stopping", r"early[ -]stopping", False),
    ("checkpoint averaging", r"checkpoint averaging|averag\w* (?:of )?(?:the )?(?:last )?\d* ?checkpoints", False),
    ("official evaluation script", r"official (?:evaluation )?(?:script|scorer|evaluation)", False),
    ("evaluation server", r"(?:evaluation|test) server|leaderboard submission", False),
)
METRIC_NAME = re.compile(r"(?<![a-z0-9])(top-?[15](?: accuracy| error)?|accuracy|error rate|m?iou|mean iou|(?:box|mask) ap\w*|ap(?:50|75|s|m|l)?|"
                         r"m?ap(?:@[\d.:\[\]]+)?|(?:macro|micro|weighted)?[- ]?f1(?: score)?|exact match|em|sacrebleu|bleu(?:-\d)?|chrf\+*|"
                         r"rouge-?(?:1|2|l|lsum)?|meteor|cider|spice|perplexity|ppl|bits per (?:dim|byte|character)|bpd|bpc|wer|cer|psnr|ssim|lpips|"
                         r"fid|inception score|pass@\d+|auroc|auc|(?:precision|recall|ndcg|hits)@\d+|mrr|spearman|pearson|success rate|win rate)(?![a-z0-9])")
DEF_CUE = re.compile(r"we report|report(?:ed|s|ing)?|computed|calculated|measured|defined|in terms of|evaluat\w+ (?:with|using|by|in)|metrics?|"
                     r"averaged over|macro|micro|weighted|case-?(?:sensitive|insensitive)|cased|uncased|detokeni[sz]ed|tokeni[sz]ed|normali[sz]ed|"
                     r"iou threshold|per-class|class-?wise|mean over|single-?scale|multi-?scale")
VERSION_TOKEN = re.compile(r"(?<![a-z0-9])v\d+(?:\.\d+)*(?![a-z0-9])|\bversion\s?[\w.]+|(?<![\d.])(?:19|20)\d{2}(?![\d.])|"
                           r"[a-z]-?\d+(?:\.\d+)?k(?![a-z0-9])|[a-z]-\d{2,4}(?![\d.])")
CITATION_YEAR = re.compile(r"et al\.?,?\s?(?:19|20)\d{2}|\([^()]*,\s?(?:19|20)\d{2}[a-z]?\)")
CKPT_CUE = re.compile(r"checkpoints?|ckpt|weights|initiali[sz]ed (?:from|with)|released (?:model|weights)|pre-?trained (?:model|weights)")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20}
SEED_IDS = re.compile(r"seeds?\s?(?:(?:is |are |was |were )?(?:fixed |set )?(?:to|at|of|as)|=|:|is|are|was|were|values?)?\s?[(\[{]?(\d+(?:\s?(?:,|and|/|or)\s?\d+)*)")
SEED_COUNT = re.compile(rf"(?<![\d.])(\d+|{'|'.join(NUMBER_WORDS)})\s(?:different |random |independent |distinct )*(seeds|runs|trials|restarts|initiali[sz]ations)\b")


SPLIT_ALIASES = {"val": "validation", "dev": "validation", "development": "validation", "testing": "test", "training": "train",
                 "trainval": "train+val", "heldout": "held-out", "testdev": "test-dev", "teststd": "test-std", "testchallenge": "test-challenge"}


def canonical_split(text: str) -> str | None:
    """'ImageNet val', 'validation set', 'dev' -> 'validation'; 'test-dev' -> 'test-dev'; 'val 2017' -> 'val2017'."""
    m = SPLIT.search(canon(text))
    if not m:
        return None
    t = re.sub(r"\s", "", m.group(1))
    return SPLIT_ALIASES.get(t.replace("-", ""), t)


def protocol_tags(text: str, need_eval_cue: bool = True) -> list[str]:
    """Canonical evaluation-protocol tags named in `text`, in a fixed order. Crop, flip and scale words count only beside an
    evaluation cue and never as training augmentation ("random crop", "multi-scale training")."""
    t = canon(text)
    cleaned = TRAINING_AUG.sub(" ", t)
    out = []
    for tag, pat, needs_eval in PROTOCOL_TAGS:
        m = re.search(pat, cleaned)
        if not m or (needs_eval and need_eval_cue and not EVAL_CUE.search(cleaned)):
            continue
        name = tag.format(m.group(1)) if "{}" in tag else tag
        if name not in out:
            out.append(name)
    if any(x[0].isdigit() and x.endswith("-shot") for x in out):  # "few-shot (5-shot)": the number says more
        out = [x for x in out if x != "few-shot"]
    return out


METRIC_QUALIFIERS = {"macro": r"macro", "micro": r"micro", "weighted": r"weighted", "cased": r"case-?sensitive|(?<![a-z])cased",
                     "uncased": r"uncased|case-?insensitive", "sacrebleu": r"sacrebleu", "detokenized": r"detokeni[sz]ed"}
CONDITION_FIELDS = ("evaluation.dataset_split", "dataset.split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version")


def _metric_tokens(name: str) -> set[str]:
    """'top-1 accuracy' and 'top-1' -> {top-1, accuracy} (a top-k score is an accuracy); 'top-1 error' -> {top-1, error};
    'macro F1' -> {f1} (the qualifier is added separately)."""
    if m := re.fullmatch(r"top-?([15])(?: (accuracy|error))?", name):
        return {f"top-{m.group(1)}", "error" if m.group(2) == "error" else "accuracy"}
    name = re.sub(r"^(?:macro|micro|weighted)?[- ]?f1(?: score)?$", "f1", name)
    return {{"mean iou": "miou", "exact match": "em", "sacrebleu": "bleu", "ppl": "perplexity", "auc": "auroc", "error rate": "error"}.get(name, name)}


def condition_key(field_name: str, value) -> frozenset | None:
    """The comparable content of a condition: split -> {canonical split}; protocol -> its tags; metric definition -> metric
    names + qualifiers ("macro F1" -> {f1, macro}); dataset version -> the version string. None when unrecognisable."""
    if value in (None, ""):
        return None
    t = canon(value)
    if field_name in ("evaluation.dataset_split", "split", "dataset.split"):
        s = canonical_split(t)
        return frozenset([s]) if s else None
    if field_name == "evaluation.evaluation_protocol":
        tags = protocol_tags(t, need_eval_cue=False)
        return frozenset(tags) if tags else None
    if field_name == "evaluation.metric_definition":
        names = {x for m in METRIC_NAME.finditer(t) for x in _metric_tokens(m.group(1).strip(" -"))}
        return frozenset(names | {q for q, rx in METRIC_QUALIFIERS.items() if re.search(rx, t)}) if names else None
    if field_name == "dataset.dataset_version":
        return frozenset([re.sub(r"[^a-z0-9.]+", "", t)]) if VERSION_TOKEN.search(t) else None
    return frozenset([t])


def condition_value(field_name: str, value) -> str | None:
    """Canonical form of a recorded condition, or None when the value is not recognisable as one (a split field holding
    "#Params(GB)" or a protocol field holding "accuracy" is unverified, never a condition)."""
    if value in (None, ""):
        return None
    if field_name in ("evaluation.dataset_split", "split", "dataset.split"):
        return canonical_split(str(value))
    if field_name == "evaluation.evaluation_protocol":
        tags = protocol_tags(str(value), need_eval_cue=False)
        return " + ".join(tags) if tags else None
    if field_name in ("evaluation.metric_definition", "dataset.dataset_version"):
        return display(value) if condition_key(field_name, value) else None
    return display(value)


def condition_compare(field_name: str, a, b) -> tuple[str, str]:
    """SAME | DIFFERENT | UNKNOWN for two stated conditions. A side whose value is not a recognisable condition is
    unverified (UNKNOWN); one side stating a narrower form of the other ("accuracy" vs "top-1 accuracy", "multi-scale" vs
    "multi-scale + flip") cannot establish equivalence either (UNKNOWN), and is never called a difference."""
    ka, kb = condition_key(field_name, a), condition_key(field_name, b)
    what = FIELDS.get(field_name, (field_name,))[0]
    if (ka is None or kb is None) and field_name in ("evaluation.evaluation_protocol", "evaluation.metric_definition") \
            and canonical_label(str(a)) == canonical_label(str(b)):  # open vocabularies: the same wording on both sides
        return "SAME", f"identical {what} wording"
    if ka is None or kb is None:
        return "UNKNOWN", f"side {'A' if ka is None else 'B'} records “{a if ka is None else b}”, not a recognisable {what}: unverified"
    if ka == kb:
        return "SAME", f"equal after {what} normalisation"
    if ka < kb or kb < ka:
        return "UNKNOWN", f"one side states a less specific {what} ({', '.join(sorted(ka ^ kb))} stated on one side only): equivalence cannot be established"
    return "DIFFERENT", f"{what} differs ({', '.join(sorted(ka))} vs {', '.join(sorted(kb))})"


def _validate_seed(value, cs: str) -> tuple[bool, str, float | str | None]:
    """'random seed 42' -> 42; 'seeds 0, 1 and 2' -> 'seeds 0, 1, 2'; 'averaged over 5 random seeds' -> '5 seeds'."""
    want = canon(str(value))
    nums = [int(n) for n in re.findall(r"\d+", want)] or [NUMBER_WORDS[w] for w in re.findall(rf"\b(?:{'|'.join(NUMBER_WORDS)})\b", want)]
    if not nums:
        return False, "value_not_numeric", None
    for m in SEED_COUNT.finditer(cs):
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        if nums == [n] and (m.group(2) == "seeds" or re.search(r"seeds?", cs)):
            return True, "ok", f"{n} {'seeds' if m.group(2) == 'seeds' else m.group(2)}"
    for m in SEED_IDS.finditer(cs):
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if sorted(nums) == sorted(ids):
            return True, "ok", ids[0] if len(ids) == 1 else "seeds " + ", ".join(str(i) for i in ids)
    return False, ("no_field_cue" if not re.search(r"seed", cs) else "value_not_in_span"), None


def _cell_labelled(rx: re.Pattern, want: str, source_text: str) -> bool:
    """A bare table cell holding `want` whose row label or column header matches `rx`."""
    return any(want in canon(c) and (rx.search(canon(h)) or rx.search(canon(r))) for r, h, c in table_cells(source_text))


def validate_text(field_name: str, value, cs: str, source_text: str) -> tuple[bool, str, str | None]:
    """The text-valued conditions. `cs` is the canonical span, already found in the source."""
    want = canon(str(value))
    if len(want) < 2 or want not in cs:
        return False, "value_not_in_span", None
    if field_name in ("evaluation.dataset_split", "dataset.split"):
        split = canonical_split(want)
        if split is None:
            return False, "not_a_split", None
        if (split in TRAIN_SPLITS or split.startswith("train")) != (field_name == "dataset.split"):
            return False, "wrong_split_role", None
        at = cs.find(want)
        near = cs[max(0, at - 30): at + len(want) + 30]
        context = (SPLIT_NOUN.search(near) or UNAMBIGUOUS_SPLIT.search(split) or re.search(rf"(?:on|of|from)\s(?:the\s)?(?:\S+\s){{0,3}}{re.escape(want)}", cs)
                   or _cell_labelled(re.compile(r"split|set|subset|eval"), want, source_text))
        return (True, "ok", split) if context else (False, "no_field_cue", None)
    if field_name == "evaluation.evaluation_protocol":
        allowed = protocol_tags(cs)  # crop / flip / scale words need an evaluation cue in the span itself
        tags = [t for t in protocol_tags(want, need_eval_cue=False) if t in allowed]
        return (True, "ok", " + ".join(tags)) if tags else (False, "not_a_protocol", None)
    if field_name == "evaluation.metric_definition":
        if not METRIC_NAME.search(want):
            return False, "not_a_metric", None
        return (True, "ok", display(value)) if DEF_CUE.search(cs) else (False, "no_field_cue", None)
    if field_name == "dataset.dataset_version":
        if not VERSION_TOKEN.search(want) or not re.search(r"[a-z]", want):
            return False, "not_a_version", None
        if CITATION_YEAR.search(cs) and not re.search(r"v\d|version|-\d|\d+k", want):
            return False, "citation_year", None
        return True, "ok", display(value)
    if field_name == "model.checkpoint":
        if not re.search(r"[a-z]", want):
            return False, "not_a_checkpoint", None
        return (True, "ok", display(value)) if CKPT_CUE.search(cs) else (False, "no_field_cue", None)
    return False, "unknown_field", None


def validate(field_name: str, value, span: str, source_text: str) -> tuple[bool, str, float | str | None]:
    """Is (field, value) entailed by `span`, and is `span` really in `source_text`? Deterministic."""
    cs, src = canon(span), canon(source_text)
    if len(cs) < 2 or cs not in src:
        return False, "span_not_in_source", None
    if field_name in TEXT_FIELDS:
        return validate_text(field_name, value, cs, source_text)
    if field_name == "training.seed":
        return _validate_seed(value, cs)
    bare = not any(rx.search(cs) for rx in CUES.values())  # no field name in the span at all (e.g. a table cell)
    if field_name in CATEGORICAL:
        found = {canon(v) for v in re.findall(CATEGORICAL[field_name], cs)}
        want = canon(str(value))
        if want not in found:
            return False, "value_not_in_span", None
        # optimizer and hardware names identify their field ("trained with SGD"); "cosine" or "linear" need a cue
        if field_name not in ("compute.gpu_model", "training.optimizer") and not CUES[field_name].search(cs) \
                and not (bare and _table_cell(field_name, lambda c: want in re.findall(CATEGORICAL[field_name], c), source_text)):
            return False, "no_field_cue", None
        return True, "ok", want
    target = number_of(value)
    if target is None:
        return False, "value_not_numeric", None
    hits = [(s, e) for v, s, e in _numbers(cs) if abs(v - target) <= 1e-9 * max(1.0, abs(target))]
    if not hits:
        return False, "value_not_in_span", None
    if field_name == "evaluation.score":
        return True, "ok", target  # a located table cell: the dataset pipeline verified row/column association
    if bare and _table_cell(field_name, lambda c: (v := number_of(c)) is not None and abs(v - target) <= 1e-9 * max(1.0, abs(target)),
                            source_text):
        return True, "ok", target
    cues = [(f, m.start(), m.end()) for f, rx in CUES.items() for m in rx.finditer(cs)]
    # a cue inside a longer one belongs to it: "steps" in "warmup steps" is not the training-steps cue
    cues = [c for c in cues if not any(o[1] <= c[1] and c[2] <= o[2] and o[2] - o[1] > c[2] - c[1] for o in cues)]

    def gap(cue, s, e):
        _, c_start, c_end = cue
        return s - c_end if c_end <= s else (c_start - e + 8 if c_start >= e else 999)

    nearest = None
    for s, e in hits:
        best = min(cues, key=lambda c: gap(c, s, e), default=None)
        if best and gap(best, s, e) <= 48:
            if best[0] == field_name or (best[0] in UNIT_CUES.get(field_name, ()) and CUES[field_name].search(cs)):
                return True, "ok", target  # "warm up ... for the first 5K steps": steps is the unit of the warm-up
            nearest = best[0]
    return False, (f"wrong_field:{nearest}" if nearest else "no_field_cue"), None


def dataset_context(chunk_text: str, span: str, dataset: str | None, known: list[str]) -> bool | None:
    """Does the value in `span` belong to `dataset`? The nearest dataset name mentioned before the span's end decides
    (e.g. "Cityscapes: ... lr 0.01. ADE20K: ... lr 0.02"). None when the text names no known dataset at all."""
    if not dataset:
        return None
    text, cs = canon(chunk_text), canon(span)
    at = text.find(cs)
    limit = (at if at >= 0 else 0) + len(cs)
    names = {canon(n): n for n in {*known, dataset} if n}
    names.update({k.split(" ")[0]: v for k, v in list(names.items()) if " " in k and len(k.split(" ")[0]) >= 4})
    mentions = [(m.end(), len(n), v) for n, v in names.items() if n for m in re.finditer(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", text[:limit])]
    if mentions:
        name = max(mentions)[2]  # nearest mention; at the same position the longest name ("CIFAR-100" over "CIFAR")
        return bool(_ds_level(dataset, name) or _ds_level(name, dataset))
    return True if re.search(rf"(?<![a-z0-9]){re.escape(canon(dataset))}(?![a-z0-9])", text) else None


# ---------------------------------------------------------------------------------------------- runtime (free-text) checks
# Facts extracted at answer time from external text have no dataset-pipeline location behind them, so they get stricter
# rules than validate(): a score needs its metric named next to it (or a table cell whose row names the model and whose
# column names the metric), a gain, a margin or a change between two values is not a score, and a value read from a table
# with several value columns must sit in the column of the asked model or dataset.
RELATIVE_BEFORE = re.compile(r"(?:\bby|\bimprov\w*|\bgains?|\bincreas\w*|\bboost\w*|\bdrops?|\bdecreas\w*|\breduc\w*|\bmargin|\+|±)\s*(?:of\s*)?"
                             r"(?:about\s*|around\s*|roughly\s*|nearly\s*|up to\s*|over\s*|more than\s*)?$")
RELATIVE_AFTER = re.compile(r"\s*%?\s*(?:points?|pp|percentage points?|higher|lower|better|worse|more|less|improvements?|gains?|absolute|relative)\b")
CHANGE = re.compile(r"\bfrom\s*[-+]?\d[\d.]*\s*%?\s*(?:to|→)\s*[-+]?\d[\d.]*")


def metric_ok(found: str, metric: str | None) -> bool:
    """A metric name next to a value fits the asked metric: one names a subset of the other's words ("accuracy" for
    "top-1 accuracy"), so top-5 never answers for top-1."""
    a, b = tokens(found), tokens(metric)
    return not b or (bool(a) and (a <= b or b <= a))


def _same_number(text: str, target: float) -> bool:
    v = number_of(text)
    return v is not None and abs(v - target) <= 1e-9 * max(1.0, abs(target))


SIZE_WORDS = "tiny|small|base|medium|large|huge|giant"
SIZE_ALIAS = {"t": "tiny", "s": "small", "b": "base", "m": "medium", "l": "large", "h": "huge", "g": "giant"}
VALUE_SET = re.compile(r"\{[^{}]*\d[^{}]*,[^{}]*\}")  # "{0.5, 0.8, 1.0}e-4": a sweep, not one setting


def label_tokens(text: str | None) -> set[str]:
    """Tokens of a short model label with letter-digit runs split and one-letter sizes spelled out, so the table row
    "BEiT384-L (ours)" names "BEiT-large" (beit, 384, large) and "BEiT-B" does not. Magnitudes such as "7b" stay whole."""
    t = re.sub(r"(?<=[a-z])(?=\d)", " ", canon(text))
    return {SIZE_ALIAS.get(x, x) for x in re.split(r"[^a-z0-9]+", t) if x and x not in STOP}


def _family(model: str | None) -> list[str]:
    return [t for t in tokens(model) if t not in SIZE_ALIAS and not re.fullmatch(rf"{SIZE_WORDS}|\d+(?:\.\d+)?[kmbt]?", t)]


def _size_mentions(text: str | None, family: list[str]) -> list[tuple[int, str]]:
    """(position in canon text, size) of model-size mentions tied to a model: "large-size", "base model", "BEiT large",
    "large BEiT", "BEiT-L". A loose "large margin" is not one."""
    fam = "|".join(re.escape(f) for f in family) or "(?!x)x"
    rx = re.compile(rf"(?<![a-z0-9])(?:({SIZE_WORDS})[\s-]?(?:sized?|scale|models?|variants?|versions?)|(?:{fam})[\s-]?({SIZE_WORDS})|"
                    rf"({SIZE_WORDS})[\s-]?(?:{fam})|(?:{fam})-([tsbmlhg]))(?![a-z0-9])")
    return [(m.start(), m.group(1) or m.group(2) or m.group(3) or SIZE_ALIAS[m.group(4)]) for m in rx.finditer(canon(text))]


def other_size(model: str | None, span: str, value, source_text: str | None = None) -> bool:
    """The model size named nearest before the value is not the asked one: "base-size BEiT achieves 83.2%" never answers
    for BEiT-large, even when the quoted span starts after "base-size". Reads the source from the start of the span's
    sentence (at most 240 characters back) to the value. False when the question names no size or none is named there."""
    fam = _family(model)
    asked = {s for _, s in _size_mentions(model, fam)}
    if not asked:
        return False
    cs, target, src = canon(span), number_of(value), canon(source_text) if source_text else ""
    pos = next((s for v, s, _ in _numbers(cs) if target is not None and abs(v - target) <= 1e-9 * max(1.0, abs(target))), len(cs))
    at = src.find(cs) if src else -1
    if at >= 0:  # the span inside its source: look back to the sentence start (canon text keeps no space after a period)
        ends = [m.end() for m in re.finditer(r"[.!?;](?=[a-z(])", src[:at])]
        start = max(ends[-1] if ends else 0, at - 240)
        text, pos = src[start:at + pos], at + pos - start
    else:
        text = cs
    before = [s for p, s in _size_mentions(text, fam) if p < pos]
    return bool(before) and before[-1] not in asked


def other_version(model: str | None, text: str | None) -> bool:
    """`text` names the asked model family with a version the question does not ("BEiT v2" for BEiT, "Llama 2" for
    Llama): another model, not a variant of the asked one."""
    want = tokens(model)
    for fam in _family(model):
        for m in re.finditer(rf"(?<![a-z0-9]){re.escape(fam)}[\s-]?v?(\d+(?:\.\d+)?)(?![\d.]*[a-z0-9])", canon(text)):
            if not ({m.group(1), f"v{m.group(1)}"} & want or fam + m.group(1) in want or f"{fam}v{m.group(1)}" in want):
                return True
    return False


def value_set(span: str, value) -> bool:
    """The value is one of several listed together, a sweep rather than one setting: "{0.5, 0.8, 1.0}e-4",
    "a learning rate of 0.1, 0.01 or 0.001"."""
    if VALUE_SET.search(span or ""):
        return True
    cs, target = canon(span), number_of(value)
    for v, s, e in _numbers(cs):
        if (target is not None and abs(v - target) <= 1e-9 * max(1.0, abs(target))
                and (re.match(r"(?:,|/|\s?or\s)\s?[-+]?\d", cs[e:e + 6]) or re.search(r"\d\s?(?:,|/|\sor)\s?$", cs[max(0, s - 6):s]))):
            return True
    return False


def table_cells(source_text: str) -> list[tuple[str, str, str]]:
    """(row label, column header, cell) for every body cell of a ' | ' table; the first row is the header."""
    rows = [[c.strip() for c in line.split("|")] for line in (source_text or "").split("\n") if "|" in line]
    if len(rows) < 2:
        return []
    head = rows[0]
    return [(r[0], head[j] if j < len(head) else "", c) for r in rows[1:] for j, c in enumerate(r) if j > 0]


def validate_score(value, span: str, source_text: str, metric: str | None, model: str | None,
                   dataset: str | None = None) -> tuple[bool, str, float | None]:
    """Is `value` the asked metric's score in `span`, and is `span` in `source_text`? For runtime extraction from free
    text. Reasons: span_not_in_source, value_not_numeric, value_not_in_span, value_change, relative_value, wrong_metric,
    no_metric_cue."""
    cs, src = canon(span), canon(source_text)
    if len(cs) < 2 or cs not in src:
        return False, "span_not_in_source", None
    target = number_of(value)
    if target is None:
        return False, "value_not_numeric", None
    hits = [(s, e) for v, s, e in _numbers(cs) if abs(v - target) <= 1e-9 * max(1.0, abs(target))]
    if not hits:
        return False, "value_not_in_span", None
    want = label_tokens(model)
    caption = canon(" ".join(line for line in (source_text or "").split("\n") if "|" not in line))
    in_caption = any(metric_ok(m.group(1), metric) for m in METRIC.finditer(caption))
    for row, head, cell in table_cells(source_text):  # a results table: the row names the model, the column the metric
        m = METRIC.search(canon(head))  # ... or the caption names the metric and the column the asked dataset ("ImageNet")
        column = (m and metric_ok(m.group(1), metric)) or (in_caption and dataset and (_ds_level(dataset, head) or _ds_level(head, dataset)))
        if want and want <= label_tokens(row) and column and _same_number(cell, target):
            return True, "ok", target
    cues = [(m.start(), m.end(), m.group(1)) for m in METRIC.finditer(cs)]
    reason = "no_metric_cue"
    for s, e in hits:
        if any(c.start() <= s < c.end() for c in CHANGE.finditer(cs)):
            reason = "value_change"
            continue
        if RELATIVE_BEFORE.search(cs[max(0, s - 24):s]) or RELATIVE_AFTER.match(cs[e:e + 24]):
            reason = "relative_value"
            continue
        def gap(c, s=s, e=e):  # distance from the value to a metric name, as in validate()
            return s - c[1] if c[1] <= s else (c[0] - e + 8 if c[0] >= e else 999)
        near = min(cues, key=gap, default=None)
        if near is None or gap(near) > 48:
            reason = "no_metric_cue"
            continue
        if not metric_ok(near[2], metric):
            reason = "wrong_metric"
            continue
        return True, "ok", target
    return False, reason, None


def row_of(source_text: str, value, model: str | None) -> tuple[str, str] | None:
    """(label, verbatim line) of the first table row that names `model` and holds `value`; None outside tables. The line is
    the citable span: a bare "85.2" would point at whichever row holds that number first."""
    target, want = number_of(value), label_tokens(model)
    if target is None or not want:
        return None
    for line in (source_text or "").split("\n"):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 1 and want <= label_tokens(cells[0]) and any(_same_number(c, target) for c in cells[1:]):
            return display(cells[0]), line
    return None


def table_column_ok(source_text: str, value, req: Request) -> bool | None:
    """A runtime value read from a table: does a cell holding it sit in the column of the asked model or dataset? None
    when the table has a single value column (nothing to choose between) or the value is not a cell of it."""
    rows = [line for line in (source_text or "").split("\n") if "|" in line]
    if not rows or max(len(r.split("|")) for r in rows) <= 2:
        return None
    target, want = number_of(value), canon(str(value))
    grid = table_cells(source_text)
    cells = [head for _, head, c in grid if (target is not None and _same_number(c, target)) or canon(c) == want]
    if not cells:
        return None
    exact = {h for _, h, _ in grid if req.model and tokens(h) == tokens(req.model)}  # "Zeta-9" next to "Zeta-9-Small": only the first

    def fits(head: str) -> bool:  # the column is the asked model (another variant never is) or the asked dataset ("ImageNet")
        if req.dataset and (_ds_level(req.dataset, head) or _ds_level(head, req.dataset)):
            return True
        if exact:
            return head in exact
        cov = coverage(req.model, [head], {})
        return cov is not None and cov >= 0.5 and not variant_conflict(req.model, [head])
    return any(fits(head) for head in cells)


# ---------------------------------------------------------------------------------------------- facts and sufficiency


@dataclass
class Fact:
    field: str
    raw: str
    span: str
    chunk_id: str
    sid: str | None = None
    value: float | str | None = None
    start: int | None = None
    end: int | None = None
    origin: str = "corpus evidence"     # corpus evidence (paper-located) | dataset record (import) | runtime extraction
    evidence_id: str | None = None
    locator: str | None = None          # provenance location, e.g. 'https://...2002.08155v4#S4.T4 row 3 ("RoBERTa") column 2'
    experiment_ids: tuple = ()
    config: str = "WEAK"                # configuration match of the experiment/source the fact belongs to
    config_detail: str = ""
    ds_exact: bool | None = None        # linked experiment names the asked dataset itself (None: unknown or unlinked)
    dataset_ok: bool | None = None
    valid: bool = False
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.valid and self.dataset_ok is not False and self.config in ("EXACT", "STRONG")


@dataclass
class FieldResult:
    field: str
    label: str
    status: str                         # SUPPORTED | UNKNOWN | CONFLICTING | AMBIGUOUS
    facts: list[Fact] = field(default_factory=list)
    close: bool = False                 # supported only by a close (STRONG), not an exact, configuration match


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def assess(req: Request, facts: list[Fact], relevant_sources: int) -> tuple[str, list[FieldResult]]:
    """Per-field sufficiency. Different sources reporting different values for the same experiment (or unlinked sources
    disagreeing) are a conflict; values that differ between matching experiments, or several values from one source
    (a table listing several rows under one experiment record), are ambiguous."""
    results = []
    for f, label in zip(req.fields, req.labels):
        usable = [x for x in facts if x.field == f and x.usable]
        top = max((RANK[x.config] for x in usable), default=0)
        best = [x for x in usable if RANK[x.config] == top]
        if any(x.ds_exact for x in best):  # the asked dataset itself outranks its narrower variants
            best = [x for x in best if x.ds_exact is not False]
        values: list = []
        for x in best:
            if not any(_same(x.value, v) for v in values):
                values.append(x.value)
        close = top < RANK["EXACT"]
        if not best:
            results.append(FieldResult(f, label, "UNKNOWN"))
        elif len(values) == 1:
            results.append(FieldResult(f, label, "SUPPORTED", best, close))
        else:
            groups: dict[tuple, list[Fact]] = {}
            for x in best:
                groups.setdefault(x.experiment_ids or (f"source:{x.chunk_id.split('#')[0]}",), []).append(x)

            def disagree(g: list[Fact]) -> bool:  # different sources, different values; one table listing several rows is not that
                return len({repr(y.value) for y in g}) > 1 and len({y.chunk_id for y in g}) > 1

            conflict = any(disagree(g) for g in groups.values()) or (all(not x.experiment_ids for x in best) and disagree(best))
            results.append(FieldResult(f, label, "CONFLICTING" if conflict else "AMBIGUOUS", best, close))
    states = {r.status for r in results}
    if "CONFLICTING" in states:
        status = "CONFLICTING_EVIDENCE"
    elif states == {"SUPPORTED"}:
        status = "PARTIAL" if any(r.close for r in results) else "SUFFICIENT"  # closest configuration only: said so
    elif "SUPPORTED" in states:
        status = "PARTIAL"
    elif "AMBIGUOUS" in states:
        status = "AMBIGUOUS_EXPERIMENT_MATCH"
    else:
        status = "INSUFFICIENT_EVIDENCE" if relevant_sources else "NO_RELEVANT_SOURCE_FOUND"
    return status, results


def render(req: Request, status: str, results: list[FieldResult]) -> tuple[str, list[dict]]:
    """Answer text built only from validated facts (no model call), plus one claim per cited span."""
    subject = " on ".join(x for x in (req.model, req.dataset) if x) or "the requested configuration"
    lines, claims, seen = [], [], set()
    by_sid = lambda x: int(x.sid[1:]) if x.sid else 0  # noqa: E731 - values and claims in source order
    for r in results:
        label = r.label[0].upper() + r.label[1:]
        if r.status == "SUPPORTED":
            sids = sorted({x.sid for x in r.facts if x.sid}, key=lambda s: int(s[1:]))
            near = next((x.config_detail for x in r.facts if x.config_detail), "") if r.close else ""
            note = f" (closest matching configuration{': ' + near if near else ''}; no exact match)" if r.close else ""
            lines.append(f"- {label}: {r.facts[0].raw} " + "".join(f"[{s}]" for s in sids) + note)
        elif r.status in ("CONFLICTING", "AMBIGUOUS"):
            parts, shown = [], set()
            for x in sorted(r.facts, key=by_sid):
                tag = (repr(x.value), x.sid, x.config_detail if r.status == "AMBIGUOUS" else "")
                if tag not in shown:
                    shown.add(tag)
                    detail = f" ({x.config_detail})" if r.status == "AMBIGUOUS" and x.config_detail else ""
                    parts.append(f"{x.raw}{detail} [{x.sid}]")
            lead = "sources disagree:" if r.status == "CONFLICTING" else "depends on the experiment:"
            lines.append(f"- {label}: {lead} " + "; ".join(parts))
        else:
            lines.append(f"- {label}: not stated in the retrieved evidence")
        for x in sorted(r.facts, key=by_sid):
            key = (r.field, x.sid, canon(x.span), repr(x.value))
            if key in seen:
                continue
            seen.add(key)
            claims.append({"text": f"{label} = {x.raw}", "field": r.field, "value": x.value, "sources": [x.sid], "quote": x.span,
                           "evidence_id": x.evidence_id, "origin": x.origin, "config": x.config, "start": x.start, "end": x.end})
    partial = (f"No experiment in the evidence matches {subject} exactly; values for the closest matching configuration:"
               if all(r.status == "SUPPORTED" and r.close for r in results) else
               f"For {subject}, the retrieved evidence covers only part of the question:")
    head = {"SUFFICIENT": f"For {subject}:", "PARTIAL": partial,
            "CONFLICTING_EVIDENCE": f"For {subject}, the sources report different values:",
            "AMBIGUOUS_EXPERIMENT_MATCH": f"Several experiments match {subject}; the values differ between them:"}.get(status, "")
    return (head + "\n" + "\n".join(lines)).strip(), claims
