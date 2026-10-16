"""Audit the current dataset: layers, field coverage, evidence granularity, source quality, KPIs.

Writes data/reports/<stem>.json and .md. Default stem dataset_audit (the v0.1.0 baseline audit).
"""

import argparse
import json

from services.datasets import audit
from services.datasets.store import rel, utf8_stdio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="dataset_audit")
    ap.add_argument("--title", default="ResearchX dataset audit")
    args = ap.parse_args()
    utf8_stdio()
    a = audit.audit()
    j, md = audit.write(a, args.stem, args.title)
    print(json.dumps({x["layer"]: x["count"] for x in a["layers"]}, indent=1))
    print(json.dumps(a["kpis"]["technical_fields_filled_per_experiment"]))
    print(f"wrote {rel(j)} and {rel(md)}")


if __name__ == "__main__":
    main()
