"""Build a human review batch from Silver (DATASET_SPEC §26, §27). Writes .jsonl + readable .md."""

import argparse

from services.datasets.review import build_queue, write_queue
from services.datasets.store import now_utc, run_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", default=f"batch-{now_utc().strftime('%Y%m%d')}")
    defaults = (("experiment", 12), ("comparison", 12), ("contradiction", 10), ("study_link", 6), ("abstention", 6), ("evidence", 8),
                ("association", 8), ("artifact_evidence", 6))
    for t, n in defaults:
        ap.add_argument(f"--{t.replace('_', '-')}", type=int, default=n, dest=t)
    args = ap.parse_args()
    quotas = {t: getattr(args, t) for t, _ in defaults}
    with run_stage("build_review_queue", {"batch_id": args.batch_id, **quotas}) as run:
        items = build_queue(args.batch_id, quotas)
        info, md = write_queue(args.batch_id, items)
        run.output(info)
        run.note(f"readable batch: {md}")
        run.counts["items"] = len(items)


if __name__ == "__main__":
    main()
