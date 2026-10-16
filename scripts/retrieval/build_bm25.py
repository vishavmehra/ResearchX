"""Build the BM25 artifact of a release from rx.chunks (indexed text: section title + chunk text).

Writes data/indexes/bm25/v<release>.npz + .json. The artifact ships with the backend, so no index is built at
serverless cold start. Usage:
    python scripts/retrieval/build_bm25.py --release 0.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402
from services.retrieval.search.bm25 import BM25  # noqa: E402
from services.storage.db import connect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    args = ap.parse_args()
    load_env()
    with connect() as conn:
        rows = conn.execute("select id, coalesce(section_title, '') || E'\\n' || text from rx.chunks where release=%s order by id",
                            (args.release,)).fetchall()
    if not rows:
        sys.exit(f"no chunks for release {args.release}")
    corpus = hashlib.sha256("\x00".join(f"{i}\x01{t}" for i, t in rows).encode("utf-8")).hexdigest()
    t0 = time.time()
    m = BM25().build(rows)
    out = ROOT / "data" / "indexes" / "bm25" / f"v{args.release}"
    m.save(out, {"release": args.release, "chunks": len(rows), "corpus_sha256": corpus, "indexed_text": "section_title + newline + text",
                 "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    size = sum((out.parent / (out.name + s)).stat().st_size for s in (".npz", ".json")) / 2**20
    print(f"bm25: {len(rows)} chunks, {len(m.vocab)} terms, {len(m.indices)} postings, built in {time.time() - t0:.1f}s, artifact {size:.1f} MiB -> {out}")


if __name__ == "__main__":
    main()
