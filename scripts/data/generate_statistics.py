"""Generate dataset statistics from files on disk (DATASET_SPEC §47)."""

import argparse
import json

from services.datasets import stats
from services.datasets.store import rel, utf8_stdio


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    utf8_stdio()
    s = stats.generate()
    j, md = stats.write(s)
    print(json.dumps(s["counts"], indent=1))
    print(f"wrote {rel(j)} and {rel(md)}")


if __name__ == "__main__":
    main()
