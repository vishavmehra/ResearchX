"""Record one reviewer decision (append-only) for a Silver record (DATASET_SPEC §26).

    uv run python scripts/data/record_review.py --reviewer alice --target-type comparison \
        --target-id cmp:... --decision ACCEPT --comment "fields verified against Table 2"
    uv run python scripts/data/record_review.py --reviewer alice --interactive batch-20260930

Human decisions (default) can promote records to Gold via promote_gold.py under the review
policy. --reviewer-type AGENT records model/agent checks for audit only: they never promote.
"""

import argparse
import json
import sys

from services.datasets.review import TARGETS, iter_queue, record_decision
from services.datasets.schemas import ReviewDecisionValue, ReviewerRole, ReviewerType
from services.datasets.store import utf8_stdio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--reviewer-type", default="HUMAN", choices=[t.value for t in ReviewerType])
    ap.add_argument("--role", default="ANNOTATOR", choices=[r.value for r in ReviewerRole])
    ap.add_argument("--target-type", choices=sorted(TARGETS))
    ap.add_argument("--target-id")
    ap.add_argument("--decision", choices=[d.value for d in ReviewDecisionValue])
    ap.add_argument("--field", help="for CORRECT: dotted field path in the record")
    ap.add_argument("--value", help="for CORRECT: corrected value (JSON if parseable)")
    ap.add_argument("--comment")
    ap.add_argument("--interactive", metavar="BATCH_ID", help="step through a review batch")
    args = ap.parse_args()
    utf8_stdio()
    kw = dict(reviewer_type=ReviewerType(args.reviewer_type), role=ReviewerRole(args.role))

    def parse_value(v: str | None):
        if v is None:
            return None
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v

    if args.interactive:
        for it in iter_queue(args.interactive):
            print(f"\n[{it['target_type']}] {it['target_id']} - {it['priority_reason']}")
            print(json.dumps(it["record"], ensure_ascii=False, indent=1)[:2500])
            for ev in it["evidence"][:10]:
                print(f"  evidence {ev['field_name']} = {ev.get('extracted_value')!r}: \"{ev['evidence_text'][:200]}\"")
            for q in it["questions"]:
                print(f"  ? {q}")
            choice = input("decision [a]ccept/[r]eject/[u]nsure/[s]kip/[q]uit: ").strip().lower()[:1]
            if choice == "q":
                break
            if choice not in ("a", "r", "u"):
                continue
            decision = {"a": ReviewDecisionValue.ACCEPT, "r": ReviewDecisionValue.REJECT, "u": ReviewDecisionValue.UNSURE}[choice]
            d = record_decision(it["target_type"], it["target_id"], args.reviewer, decision, comment=input("comment: ") or None, **kw)
            print(f"recorded {d.decision_id}")
        return 0
    if not (args.target_type and args.target_id and args.decision):
        ap.error("--target-type, --target-id and --decision are required (or use --interactive)")
    d = record_decision(args.target_type, args.target_id, args.reviewer, ReviewDecisionValue(args.decision),
                        corrected_field=args.field, corrected_value=parse_value(args.value), comment=args.comment, **kw)
    print(f"recorded {d.decision_id} ({d.reviewer_type.value} {d.reviewer_role.value} {d.decision.value})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
