"""Source registry, license and contamination manifests (DATASET_SPEC §8, §31, §63)."""

from __future__ import annotations

import re
from functools import cache
from typing import Any

from services.datasets.store import DATA, load_yaml

MANIFESTS = DATA / "manifests"


@cache
def sources() -> dict[str, dict[str, Any]]:
    return load_yaml(MANIFESTS / "sources.yaml")["sources"]


@cache
def licenses() -> dict[str, dict[str, Any]]:
    return load_yaml(MANIFESTS / "licenses.yaml")["licenses"]


@cache
def contamination() -> list[dict[str, Any]]:
    return load_yaml(MANIFESTS / "contamination.yaml")["datasets"]


def match_license(statement: str | None) -> str | None:
    """Map a license URL or label (e.g. arXiv HTML header) to a licenses.yaml id. None if unknown."""
    if not statement:
        return None
    s = statement.strip().casefold()
    for lic_id, lic in licenses().items():
        for alias in [lic.get("url", "")] + list(lic.get("aliases", [])):
            a = alias.strip().casefold()
            if a and (s == a or (len(a) > 12 and a in s)):
                return lic_id
    # common CC label forms: "CC BY 4.0", "CC BY-NC-SA 4.0", "CC0 1.0"
    m = re.search(r"\bcc[\s-]?(by(?:[\s-](?:nc|sa|nd)){0,2})[\s-]*(\d\.\d)?", s)
    if m:
        cand = "CC-" + m.group(1).upper().replace(" ", "-") + "-" + (m.group(2) or "4.0")
        if cand in licenses():
            return cand
    if re.search(r"\bcc\s?0\b|cc zero|publicdomain/zero", s):
        return "CC0-1.0"
    return None


def license_flags(lic_id: str | None) -> dict[str, Any]:
    lic = licenses().get(lic_id or "UNKNOWN") or licenses()["UNKNOWN"]
    return {
        "redistribution_allowed": lic.get("redistribution_allowed"),
        "training_allowed": lic.get("training_allowed"),
        "attribution_required": lic.get("attribution_required"),
        "share_alike": lic.get("share_alike", False),
    }


def fulltext_class(lic_id: str | None) -> str:
    """'permitted' only when redistribution AND commercial reuse are allowed; else 'restricted'."""
    lic = licenses().get(lic_id or "UNKNOWN") or {}
    if lic.get("redistribution_allowed") is True and lic.get("commercial_use") is True and not lic.get("no_derivatives"):
        return "permitted"
    return "restricted"


def usage_policy(source_id: str) -> str:
    return sources().get(source_id, {}).get("usage_policy", "UNKNOWN")
