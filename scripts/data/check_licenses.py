"""Licence, access and contamination audit (DATASET_SPEC §8, §31). Exits 1 on critical findings."""

import argparse
import json
import sys

from services.datasets.release import license_audit
from services.datasets.store import DATA, atomic_write_bytes, utf8_stdio


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    utf8_stdio()
    report = license_audit()
    out = DATA / "reports" / "license_report.json"
    atomic_write_bytes(out, json.dumps(report, indent=1).encode())
    print(f"licences: {'PASS' if report['passed'] else 'FAIL'} critical={len(report['critical'])}")
    print(f"sources: {report['sources_used']} policies: {report['usage_policies']}")
    print(f"full text storage: {report['fulltext_storage']}")
    for c in report["critical"]:
        print("  CRITICAL", c)
    for w in report["warnings"]:
        print("  obligation:", w)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
