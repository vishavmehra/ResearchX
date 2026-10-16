"""Export canonical Pydantic models to data/schemas/*.json (DATASET_SPEC §32).

--check exits 1 if the committed JSON Schemas drift from the models (used by tests/CI).
"""

import argparse
import json
import sys

from services.datasets.schemas import RECORD_MODELS, SCHEMA_VERSION
from services.datasets.store import DATA, atomic_write_bytes


def render(model, stem: str) -> bytes:
    schema = model.model_json_schema()
    schema["$id"] = f"researchx://schemas/{SCHEMA_VERSION}/{stem}.json"
    schema["x-schema-version"] = SCHEMA_VERSION
    return (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if exported schemas are stale")
    args = ap.parse_args()
    stale = []
    for model, stem in RECORD_MODELS.values():
        out = DATA / "schemas" / f"{stem}.json"
        data = render(model, stem)
        if args.check:
            if not out.exists() or out.read_bytes().replace(b"\r\n", b"\n") != data:
                stale.append(stem)
        else:
            atomic_write_bytes(out, data)
    if stale:
        print(f"stale schemas: {stale}")
        return 1
    print(f"{'checked' if args.check else 'wrote'} {len(RECORD_MODELS)} schemas (schema_version {SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
