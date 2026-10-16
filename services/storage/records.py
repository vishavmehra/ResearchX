"""Lossless compact storage of frozen dataset records in PostgreSQL (planning/DECISIONS.md §35).

The Free-plan database (500 MB) cannot hold every record verbatim, so a row stores typed columns plus a compact
jsonb from which the exact record is rebuilt:
  * string values already held in a typed text column are not repeated in the jsonb,
  * keys with one identical value across a whole release file are kept once (rx.load_checks.constants),
  * the provenance block is deduplicated into rx.provenance (rows keep provenance_id).
The loader proves expand(compact(r)) == r for every record before writing, and again after reading rows back.
"""

from __future__ import annotations

import hashlib
import json


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def provenance_id(prov) -> str:
    return "prov:" + hashlib.sha256(canonical(prov).encode()).hexdigest()[:24]


def file_constants(records: list[dict]) -> dict:
    """Keys present in every record of a file with one identical value (provenance is deduplicated separately)."""
    if len(records) < 2:
        return {}
    out = {}
    for k, v in records[0].items():
        c = canonical(v)
        if k != "provenance" and all(k in r and canonical(r[k]) == c for r in records[1:]):
            out[k] = v
    return out


def compact(record: dict, typed_keys: set[str], constants: dict) -> tuple[dict, str | None]:
    """typed_keys: record keys whose value is copied verbatim into a typed text column."""
    out = {k: v for k, v in record.items()
           if k != "provenance" and k not in constants and not (k in typed_keys and isinstance(v, str))}
    return out, (provenance_id(record["provenance"]) if "provenance" in record else None)


def expand(compact_record: dict, typed: dict, constants: dict, prov_id: str | None, provenance: dict) -> dict:
    """typed: record key -> value read from its typed text column."""
    r = {**compact_record, **constants}
    for k, v in typed.items():
        if v is not None and k not in r:
            r[k] = v
    if prov_id is not None:
        r["provenance"] = provenance[prov_id]
    return r


if __name__ == "__main__":  # self-check of the edge cases the loader relies on
    recs = [{"id": "a", "t": "x", "n": None, "k": 1, "provenance": {"s": 1}},
            {"id": "b", "t": "x", "n": "v", "k": 1, "provenance": {"s": 1}, "extra": [1, 2.5]}]
    const = file_constants(recs)
    assert const == {"t": "x", "k": 1}, const
    for r in recs:
        c, pid = compact(r, {"id", "n"}, const)
        typed = {"id": r["id"], "n": r["n"] if isinstance(r["n"], str) else None}
        assert expand(c, typed, const, pid, {provenance_id({"s": 1}): {"s": 1}}) == r
    assert file_constants([{"a": 1}, {"b": 1}]) == {} and file_constants([{"a": None}, {"a": None}]) == {"a": None}
    print("records self-check ok")
