"""Load chunk embeddings into pgvector (rx.chunk_embeddings) at a Matryoshka prefix of the 1024-d artifact.

The prefix is truncated and re-normalised, which equals requesting `dimensions` from the API (cos 0.99994, measured
2026-10-03). Rows are keyed model = '<model>@<dims>' so several sizes can coexist. Refuses to load if the estimated
database size would exceed --max-mib (Free plan: 500 MB). Usage:
    python scripts/retrieval/load_embeddings.py --release 0.2.0 --dims 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402
from services.storage.db import connect  # noqa: E402

ROW_OVERHEAD = 224  # bytes per row besides the vector (tuple header, chunk_id, model, dims, created_at, PK entry); measured


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--dims", type=int, required=True)
    ap.add_argument("--max-mib", type=int, default=450)
    args = ap.parse_args()
    load_env()
    src = next((ROOT / "data" / "indexes" / "embeddings" / f"v{args.release}").glob("*-1024"))
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    ids = json.loads((src / "ids.json").read_text(encoding="utf-8"))
    if not np.load(src / "done.npy").all():
        sys.exit("embedding artifact incomplete")
    v = np.asarray(np.load(src / "vectors.npy", mmap_mode="r")[:, :args.dims], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    model = f"{meta['model']}@{args.dims}"
    with connect() as conn:
        total = float(conn.execute("select sum(pg_database_size(datname)) from pg_database").fetchone()[0]) / 2**20
        have = conn.execute("select count(*) from rx.chunk_embeddings where model=%s", (model,)).fetchone()[0]
        n_chunks = conn.execute("select count(*) from rx.chunks where release=%s", (args.release,)).fetchone()[0]
        estimate = total + len(ids) * (2 * args.dims + ROW_OVERHEAD) / 2**20 * (0 if have else 1)
        print(f"{model}: {len(ids)} vectors, db now {total:.0f} MiB, estimated after load {estimate:.0f} MiB (limit {args.max_mib})", flush=True)
        if n_chunks != len(ids):
            sys.exit(f"chunk count mismatch: db {n_chunks} vs artifact {len(ids)}")
        if estimate > args.max_mib:
            sys.exit("refusing: the load would leave too little headroom under the database quota")
        conn.execute("delete from rx.chunk_embeddings where model=%s", (model,))
        t0 = time.time()
        with conn.cursor() as cur, cur.copy("copy rx.chunk_embeddings (chunk_id, model, dims, embedding) from stdin") as cp:
            for cid, row in zip(ids, v):
                cp.write_row([cid, model, args.dims, "[" + ",".join(f"{x:.5f}" for x in row) + "]"])
        conn.commit()
        conn.execute("analyze rx.chunk_embeddings")
        n = conn.execute("select count(*) from rx.chunk_embeddings where model=%s", (model,)).fetchone()[0]
        size = float(conn.execute("select sum(pg_database_size(datname)) from pg_database").fetchone()[0]) / 2**20
        print(f"loaded {n} rows in {time.time() - t0:.0f}s | all databases {size:.0f} MiB")
        if n != len(ids):
            sys.exit(1)


if __name__ == "__main__":
    main()
