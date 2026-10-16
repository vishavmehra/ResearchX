"""Build the dataset release manifest (DATASET_SPEC §53, §54) and regenerate versions/datasets catalogs.

Default writes data/manifests/releases/v<version>-draft.json. --freeze writes the immutable
v<version>.json; re-freezing with different content fails (bump dataset_version).
"""

import argparse

from services.datasets.release import build_manifest
from services.datasets.store import run_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    with run_stage("build_manifest", vars(args)) as run:
        out, manifest = build_manifest(freeze=args.freeze)
        run.note(f"manifest: {out} ({manifest['status']})")
        run.counts.update(files=len(manifest["checksums"]), **{k: v for k, v in manifest["record_counts"].items() if v})


if __name__ == "__main__":
    main()
