"""Deterministic normalization (DATASET_SPEC §13, §29, §42).

Rules never discard the original: callers store raw_value beside the normalized value and
record the rule name + NORMALIZATION_VERSION. Units are never inferred when unstated, and
incompatible units (epochs vs steps vs iterations) never normalize to the same unit.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

NORMALIZATION_VERSION = "norm-0.1.0"

# --------------------------------------------------------------------------- identifiers


def normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    s = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", raw.strip(), flags=re.I)
    s = s.strip().lower()
    return s or None


_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d])")
_ARXIV_OLD = re.compile(r"(?<![\w.])([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?(?!\d)")


def parse_arxiv_id(raw: str | None) -> tuple[str, str | None] | None:
    """Parse an arXiv reference (id, abs/pdf URL, arXiv DOI). Returns (id, version|None).

    Only call on strings already known to reference arXiv: a bare regex match on arbitrary
    text (e.g. a publisher DOI) is not evidence of an arXiv id.
    """
    if not raw:
        return None
    s = re.sub(r"^(https?://(dx\.)?doi\.org/)?10\.48550/arxiv\.", "", raw.strip(), flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    for pat in (_ARXIV_NEW, _ARXIV_OLD):
        m = pat.search(s)
        if m:
            return m.group(1), m.group(2)
    return None


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{arxiv_id.lower()}"


# --------------------------------------------------------------------------- titles / labels

STOPWORDS = frozenset(
    "a an and are as at by for from in into is of on or the to via with without towards toward "
    "using based we our its their this that".split()
)


def normalize_title(t: str | None) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_tokens(t: str | None) -> frozenset[str]:
    return frozenset(w for w in normalize_title(t).split() if w not in STOPWORDS and len(w) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def canonical_label(s: str | None) -> str:
    """Comparison key for dataset/metric/task/model names (rule: casefold+strip punctuation)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\s_\-/:,()\[\]]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_SIZE = re.compile(r"[\s\-_]*(\d+(?:\.\d+)?)\s*([bm])\b", re.I)


def model_family(name: str | None) -> tuple[str, str | None]:
    """Split 'LLaMA-2 13B' -> ('llama 2', '13b'). Used only for hard-negative mining."""
    label = canonical_label(name)
    m = _SIZE.search(label)
    if not m:
        return label, None
    return (label[: m.start()] + label[m.end():]).strip(), f"{m.group(1)}{m.group(2).lower()}"


# --------------------------------------------------------------------------- numbers

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})
_NUMBER_RULES: list[tuple[str, re.Pattern]] = [
    ("sci_times_pow10", re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×|·|\*|x)\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?$")),
    ("pow10", re.compile(r"^10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?$")),
    ("e_caret", re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*[eE]\s*\^\s*\{?\s*([+-]?\d+)\s*\}?$")),  # TeX 1e^{-4}
    ("int_pow", re.compile(r"^(\d+)\s*\^\s*\{?\s*(\d+)\s*\}?$")),  # 2^{19}
    ("e_notation", re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$")),
    ("decimal", re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")),
    ("thousands_separator", re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")),
    ("k_suffix", re.compile(r"^(\d+(?:\.\d+)?)\s*[kK]$")),
    ("million_suffix", re.compile(r"^(\d+(?:\.\d+)?)\s*[mM]$")),
    ("billion_suffix", re.compile(r"^(\d+(?:\.\d+)?)\s*[bB]$")),
]


def parse_number(raw: str | None) -> tuple[float, str] | None:
    """Exact (Decimal) parse of a numeric string -> (float, rule). None if not a plain number.

    '2e-5', '0.00002', '2\\times 10^{-5}' all yield the same float. Percent signs are handled
    by the caller (unit 'percent'); a percentage is never silently divided by 100.
    """
    if raw is None:
        return None
    s = raw.strip().translate(_MINUS).strip("$ ")
    s = re.sub(r"^\{(.*)\}$", r"\1", s)
    for rule, pat in _NUMBER_RULES:
        m = pat.match(s)
        if not m:
            continue
        try:
            if rule == "sci_times_pow10":
                d = Decimal(m.group(1)) * (Decimal(10) ** int(m.group(2)))
            elif rule == "pow10":
                d = Decimal(10) ** int(m.group(1))
            elif rule == "e_caret":
                d = Decimal(m.group(1)) * (Decimal(10) ** int(m.group(2)))
            elif rule == "int_pow":
                d = Decimal(int(m.group(1)) ** int(m.group(2)))
            elif rule in ("e_notation", "decimal"):
                d = Decimal(s)
            elif rule == "thousands_separator":
                d = Decimal(s.replace(",", ""))
            else:
                mult = {"k_suffix": 10**3, "million_suffix": 10**6, "billion_suffix": 10**9}[rule]
                d = Decimal(m.group(1)) * mult
        except InvalidOperation:
            return None
        return float(d), rule
    return None


# --------------------------------------------------------------------------- units

# canonical unit -> surface forms. Distinct units are never merged (spec §13, §42).
_UNIT_FORMS: dict[str, tuple[str, ...]] = {
    "epochs": ("epoch", "epochs"),
    "steps": ("step", "steps", "training steps", "update steps", "gradient steps"),
    "iterations": ("iteration", "iterations", "iters"),
    "updates": ("update", "updates"),
    "tokens": ("token", "tokens"),
    "samples": ("sample", "samples"),
    "examples": ("example", "examples"),
    "sequences": ("sequence", "sequences"),
    "sentences": ("sentence", "sentences"),
    "images": ("image", "images"),
    "pairs": ("pair", "pairs"),
    "parameters": ("parameter", "parameters", "params"),
    "seconds": ("s", "sec", "secs", "second", "seconds"),
    "minutes": ("min", "mins", "minute", "minutes"),
    "hours": ("h", "hr", "hrs", "hour", "hours"),
    "days": ("day", "days"),
    "gpu_hours": ("gpu hours", "gpu-hours", "gpu hour", "gpu-hour"),
    "percent": ("%", "percent"),
}
UNITS = frozenset(_UNIT_FORMS) | {"ratio", "bits", "gb", "flops", "count"}
_SURFACE_TO_UNIT = {form: unit for unit, forms in _UNIT_FORMS.items() for form in forms}
_DURATION_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


def canonical_unit(surface: str | None) -> str | None:
    """Map a stated unit to the controlled vocabulary. None when unrecognized (never guessed)."""
    if not surface:
        return None
    return _SURFACE_TO_UNIT.get(re.sub(r"\s+", " ", surface.strip().casefold()))


def convert_duration(value: float, from_unit: str, to_unit: str) -> tuple[float, str]:
    """Explicit, deterministic conversion between duration units. Raises on incompatible units."""
    if from_unit not in _DURATION_SECONDS or to_unit not in _DURATION_SECONDS:
        raise ValueError(f"cannot convert {from_unit} -> {to_unit}: not both durations")
    return value * _DURATION_SECONDS[from_unit] / _DURATION_SECONDS[to_unit], f"duration:{from_unit}->{to_unit}"


def comparable_units(a: str | None, b: str | None) -> bool:
    if a == b:
        return True
    return a in _DURATION_SECONDS and b in _DURATION_SECONDS


# --------------------------------------------------------------------------- misc


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as word -> positions; rebuild the original word order."""
    if not inverted_index:
        return None
    slots: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots)) or None


_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")


def clean_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", s or ""))).strip()
