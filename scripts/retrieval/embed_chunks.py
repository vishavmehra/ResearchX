"""Embed every retrieval chunk of a release with the gateway's `embedding` task (resumable).

Writes data/indexes/embeddings/v<release>/<model>-<dims>/: vectors.npy (float16, row i = ids[i]), ids.json, done.npy and
meta.json. Rows already embedded are skipped on re-run, so an interrupted job resumes. Embedded text per chunk:
"<paper title>\n<section title>\n<chunk text>" capped at 6,000 characters (TEXT_VERSION). Usage:
    python scripts/retrieval/embed_chunks.py --release 0.2.0 [--workers 8] [--batch 64] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402
from services.model_gateway.gateway import ModelGateway  # noqa: E402
from services.storage.db import connect  # noqa: E402

TEXT_VERSION = "chunk-text-0.1.0"
MAX_CHARS = 6000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, help="embed at most N pending chunks (smoke test)")
    args = ap.parse_args()
    load_env()
    gw = ModelGateway()
    task = gw.cfg["tasks"]["embedding"]
    _, model = gw.resolve_model("embedding")
    with connect() as conn:
        rows = conn.execute("select c.id, concat_ws(E'\\n', p.title, c.section_title, c.text) from rx.chunks c "
                            "left join rx.papers p on p.id = c.paper_id where c.release=%s order by c.id", (args.release,)).fetchall()
    ids, texts = [r[0] for r in rows], [r[1][:MAX_CHARS] for r in rows]
    out = ROOT / "data" / "indexes" / "embeddings" / f"v{args.release}" / f"{model.split('/')[-1].lower()}-{task['dimensions']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ids.json").exists() and json.loads((out / "ids.json").read_text(encoding="utf-8")) != ids:
        sys.exit(f"{out} was built from a different chunk set: move it away to rebuild")
    (out / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    shape = (len(ids), task["dimensions"])
    vec_path = out / "vectors.npy"
    vectors = np.lib.format.open_memmap(vec_path, mode="r+" if vec_path.exists() else "w+", dtype=np.float16, shape=shape)
    done = np.load(out / "done.npy") if (out / "done.npy").exists() else np.zeros(len(ids), dtype=bool)
    todo = np.flatnonzero(~done)[: args.limit]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{model} dims={task['dimensions']} chunks={len(ids)} done={int(done.sum())} todo={len(todo)} batches={len(batches)}", flush=True)
    t0, tokens, failed = time.time(), 0, 0

    def run(idx):
        return idx, *gw.embed("embedding", [texts[i] for i in idx])

    with ThreadPoolExecutor(args.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(run, b) for b in batches]), 1):
            idx, vecs, call = fut.result()
            if vecs is None:
                failed += len(idx)
                print(f"batch failed: {call.status} {call.error}", flush=True)
                continue
            vectors[idx] = np.asarray(vecs, dtype=np.float16)
            done[idx] = True
            tokens += call.input_tokens or 0
            if n % 25 == 0 or n == len(batches):
                vectors.flush()
                np.save(out / "done.npy", done)
                print(f"{n}/{len(batches)} batches | {int(done.sum())}/{len(ids)} chunks | {tokens} tokens | {time.time() - t0:.0f}s", flush=True)
    vectors.flush()
    np.save(out / "done.npy", done)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"tokens": 0}
    meta.update({"model": model, "dimensions": task["dimensions"], "embedding_version": task["embedding_version"],
                 "query_instruction": task.get("query_instruction"), "text_version": TEXT_VERSION, "max_chars": MAX_CHARS,
                 "release": args.release, "chunks": len(ids), "embedded": int(done.sum()), "tokens": meta["tokens"] + tokens,
                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"embedded {int(done.sum())}/{len(ids)} | failed this run {failed} | {tokens} tokens | {time.time() - t0:.0f}s -> {out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
