"""Validate every dataset file (DATASET_SPEC §46). Exits 1 on any critical violation."""

import argparse
import sys

from services.datasets.store import rel, utf8_stdio
from services.datasets.validate import validate, write_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-reverify", action="store_true", help="skip re-checking evidence text against parsed full text")
    args = ap.parse_args()
    utf8_stdio()
    report = validate(reverify_evidence=not args.no_reverify)
    out = write_report(report)
    s = report.summary()
    print(f"records checked: {s['records_checked']}")
    print(f"critical: {s['critical_count']} {s['critical_by_check']}")
    print(f"warnings: {s['warning_count']} {s['warnings_by_check']}")
    for item in report.critical[:15]:
        print(f"  CRITICAL [{item['check']}] {item['record'] or ''} {item['message']}")
    print(f"report: {rel(out)}  ->  {'PASS' if s['passed'] else 'FAIL'}")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
