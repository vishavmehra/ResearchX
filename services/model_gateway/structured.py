"""Validation of model structured output (core-engine brief §3).

Every structured model output passes, in order:
    parse       the gateway: JSON, truncation repair, a top-level object
    schema      Pydantic models here: types, enums, lengths, no unexpected fields
    semantic    the call site: known source ids, requested fields, a consistent status
    provenance  spans and quotes re-found in their cited sources (grounding.validate, orchestrator.validate_claims)
A malformed output, or a malformed item of an output, never reaches grounding: it is dropped and reported as
{status, errors, dropped}; the caller records the report in its trace stage and falls back safely (field stays unknown,
extractive answer, hybrid order). Items are validated one by one, so one bad item never discards the valid rest.
Statuses: OK (all valid), PARTIAL (some items dropped), EMPTY (a valid output with nothing in it), INVALID (unusable).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SYNTH_STATUSES = ("ANSWERED", "PARTIAL", "INSUFFICIENT")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExtractedFact(_Strict):
    """One field value from runtime field extraction (orchestrator EXTRACT_SCHEMA)."""

    source: str = Field(pattern=r"^R[1-9][0-9]*$")
    field: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=80)
    span: str = Field(min_length=2, max_length=400)

    @field_validator("value", mode="before")
    @classmethod
    def _number_as_text(cls, v):  # a bare number is the value as written; any other non-text value is invalid
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class ExtractedResult(_Strict):
    """One reported result row from runtime result extraction (orchestrator RESULTS_SCHEMA); split is "" when not stated."""

    source: str = Field(pattern=r"^R[1-9][0-9]*$")
    model: str = Field(min_length=1, max_length=120)
    dataset: str = Field(min_length=1, max_length=120)
    metric: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=40)
    split: str = Field(max_length=40)
    span: str = Field(min_length=2, max_length=600)

    @field_validator("value", mode="before")
    @classmethod
    def _number_as_text(cls, v):
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class SynthClaim(_Strict):
    """One claim of passage synthesis (orchestrator SYNTH_SCHEMA)."""

    text: str = Field(min_length=1, max_length=400)
    sources: list[Annotated[str, Field(pattern=r"^S[1-9][0-9]*$")]] = Field(min_length=1, max_length=4)
    quote: str = Field(min_length=1, max_length=500)


class FallbackAnswer(_Strict):
    answer: str = Field(min_length=1, max_length=2000)


def _messages(e: ValidationError, prefix: str = "") -> list[str]:
    return [f"{prefix}{'.'.join(str(x) for x in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors()[:3]]


def report(status: str, errors: list[str] | None = None, dropped: int = 0) -> dict:
    return {"status": status, "errors": (errors or [])[:5], "dropped": dropped}


def items(data, key: str, model: type[BaseModel], limit: int | None = None, allowed: frozenset = frozenset()) -> tuple[list, dict]:
    """The valid items of data[key], validated one by one, and a report. INVALID when `data` is not an object, has keys
    other than `key` (and `allowed`), or data[key] is not a list; items past `limit` count as dropped."""
    if not isinstance(data, dict):
        return [], report("INVALID", [f"(root): expected a JSON object, got {type(data).__name__}"])
    extra = sorted(set(data) - {key} - allowed)
    raw = data.get(key)
    if extra:
        return [], report("INVALID", [f"(root): unexpected keys {extra}"])
    if not isinstance(raw, list):
        return [], report("INVALID", [f"{key}: expected a list, got {type(raw).__name__}"])
    ok, errors = [], []
    for i, x in enumerate(raw[:limit] if limit else raw):
        try:
            ok.append(model.model_validate(x))
        except ValidationError as e:
            errors += _messages(e, f"{key}[{i}].")
    dropped = len(raw) - len(ok)
    if limit and len(raw) > limit:
        errors.append(f"{key}: {len(raw) - limit} item(s) past the limit of {limit}")
    status = "EMPTY" if not raw else "OK" if not dropped else "PARTIAL" if ok else "INVALID"
    return ok, report(status, errors, dropped)


def synthesis(data) -> tuple[str | None, list[SynthClaim], dict]:
    """(model status, schema-valid claims, report) of a synthesis output. The status is None when the output is unusable:
    not an object, an unknown status, unexpected keys, or claims that are not a list. Claims that are all malformed leave
    a usable status with no claims (the run then abstains). Valid claims still face semantic and provenance checks."""
    if not isinstance(data, dict):
        return None, [], report("INVALID", [f"(root): expected a JSON object, got {type(data).__name__}"])
    status = data.get("status")
    if status not in SYNTH_STATUSES:
        return None, [], report("INVALID", [f"status: expected one of {list(SYNTH_STATUSES)}, got {status!r}"])
    claims, rep = items(data, "claims", SynthClaim, allowed=frozenset({"status"}))
    if rep["status"] == "INVALID" and not rep["dropped"]:  # a shape error, not malformed claims
        return None, [], rep
    return status, claims, rep


def one(data, model: type[BaseModel]) -> tuple[BaseModel | None, dict]:
    """A whole output validated as one model."""
    try:
        return model.model_validate(data), report("OK")
    except ValidationError as e:
        return None, report("INVALID", _messages(e))
