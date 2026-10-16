"""Train/test leakage checks at the study level (DATASET_SPEC §28, §52). Exits 1 on critical findings."""

import argparse
import json
import sys

from services.datasets.release import detect_leakage
from services.datasets.store import DATA, atomic_write_bytes, utf8_stdio


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    utf8_stdio()
    report = detect_leakage()
    out = DATA / "reports" / "leakage_report.json"
    atomic_write_bytes(out, json.dumps(report, indent=1).encode())
    print(f"leakage: {'PASS' if report['passed'] else 'FAIL'} critical={len(report['critical'])} warnings={len(report['warnings'])}")
    print(f"splits: {report['split_distribution']} cross_split_pairs={report['cross_split_pairs']}")
    for c in report["critical"][:10]:
        print("  CRITICAL", c)
    for w in report["warnings"][:5]:
        print("  warning", w)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
