"""Retrieval ablation on the frozen eval set (Phase 7): BM25 vs dense (128-1024-d) vs hybrid RRF [vs hybrid + LLM rerank].

Dense search here is exact cosine over the local embedding artifact (what pgvector exact search returns, up to halfvec
rounding); pgvector serving latency is measured after loading. Query embeddings are cached next to the artifact.
Hybrid parameters (RRF k, dense weight) are tuned on VALIDATION only and reported on TEST and TEMPORAL_TEST, with a
paired-bootstrap 95% CI of nDCG@10(hybrid) - nDCG@10(BM25). Writes data/reports/retrieval_eval_v<release>.json + .md.
    python scripts/eval/eval_retrieval.py --release 0.2.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402
from services.model_gateway.gateway import ModelGateway  # noqa: E402
from services.retrieval.search.bm25 import BM25  # noqa: E402
from services.retrieval.search.hybrid import rrf  # noqa: E402
from services.retrieval.search.rerank import rerank  # noqa: E402
from services.storage.db import connect  # noqa: E402

DIMS = (128, 256, 512, 1024)  # Matryoshka prefixes of one 1024-d run; pgvector budget decides among them
DEPTH = 100
GRID = [(k, w) for k in (10, 30, 60, 100) for w in (0.5, 1.0, 1.5, 2.0)]


def ndcg(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(sorted(rel.values(), reverse=True)[:k], 1))
    return dcg / idcg if idcg else 0.0


def score(q: dict, ranked: list[str]) -> dict:
    """Chunk-level metrics against the top-grade chunks, or paper-level metrics for prior_art queries."""
    if "relevant_papers" in q:
        ranked, rel, ks = list(dict.fromkeys(c.split("#")[0] for c in ranked)), {p: 1 for p in q["relevant_papers"]}, (5, 10, 20, 50)
    else:
        rel, ks = q["relevant_chunks"], (5, 10, 20, 50, 100)
    targets = {d for d, g in rel.items() if g == max(rel.values())}
    first = next((i for i, d in enumerate(ranked[:10], 1) if d in targets), None)
    return {**{f"recall@{k}": len(targets & set(ranked[:k])) / len(targets) for k in ks},
            "mrr@10": 1 / first if first else 0.0, "ndcg@10": ndcg(ranked, rel)}


def mean(rows: list[dict]) -> dict:
    return {k: round(statistics.fmean(r[k] for r in rows), 4) for k in rows[0]} if rows else {}


def bootstrap_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    rng, d = random.Random(seed), [x - y for x, y in zip(a, b)]
    means = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return round(statistics.fmean(d), 4), round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--rerank", type=int, default=0, help="LLM-rerank the top N candidates of the best hybrid (0 = off)")
    args = ap.parse_args()
    load_env()
    ev = ROOT / "data" / "eval" / "retrieval" / f"v{args.release}"
    queries = [json.loads(x) for x in (ev / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))

    bm25 = BM25.load(ROOT / "data" / "indexes" / "bm25" / f"v{args.release}")
    runs, latency = {"bm25": []}, {}
    for q in queries:
        s = time.perf_counter()
        runs["bm25"].append([d for d, _ in bm25.search(q["query"], k=DEPTH)])
        latency.setdefault("bm25", []).append((time.perf_counter() - s) * 1000)

    emb_dir = next((ROOT / "data" / "indexes" / "embeddings" / f"v{args.release}").glob("*-1024"))
    emb_meta = json.loads((emb_dir / "meta.json").read_text(encoding="utf-8"))
    if not np.load(emb_dir / "done.npy").all():
        sys.exit(f"{emb_dir} is incomplete: finish scripts/retrieval/embed_chunks.py first")
    ids = json.loads((emb_dir / "ids.json").read_text(encoding="utf-8"))
    if ids != bm25.ids:
        sys.exit("embedding and BM25 artifacts were built from different chunk sets")
    qpath = emb_dir / f"queries-{eval_meta['version']}.npy"
    if qpath.exists():
        qv = np.load(qpath)
    else:
        gw, parts = ModelGateway(), []
        for i in range(0, len(queries), 64):
            v, call = gw.embed("embedding", [q["query"] for q in queries[i:i + 64]], query=True)
            if v is None:
                sys.exit(f"query embedding failed: {call.status} {call.error}")
            parts.append(np.asarray(v, dtype=np.float32))
            latency.setdefault("query_embedding_api_per_batch", []).append(call.latency_s * 1000)
        qv = np.concatenate(parts)
        np.save(qpath, qv)
    full = np.load(emb_dir / "vectors.npy", mmap_mode="r")
    for dims in DIMS:
        m = np.asarray(full[:, :dims], dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        qd = qv[:, :dims] / np.linalg.norm(qv[:, :dims], axis=1, keepdims=True)
        name = f"dense{dims}"
        runs[name] = []
        for i in range(len(queries)):
            s = time.perf_counter()
            sims = m @ qd[i]
            top = np.argpartition(-sims, DEPTH)[:DEPTH]
            runs[name].append([ids[j] for j in top[np.argsort(-sims[top], kind="stable")]])
            latency.setdefault(name, []).append((time.perf_counter() - s) * 1000)
        del m

    splits = {s: [i for i, q in enumerate(queries) if q["split"] == s] for s in ("VALIDATION", "TEST", "TEMPORAL_TEST")}
    tuned = {}
    for dims in DIMS:  # tune RRF on VALIDATION only
        best = max(GRID, key=lambda kw: statistics.fmean(
            score(queries[i], [d for d, _ in rrf([runs["bm25"][i], runs[f"dense{dims}"][i]], kw[0], [1.0, kw[1]], DEPTH)])["ndcg@10"]
            for i in splits["VALIDATION"]))
        tuned[f"hybrid{dims}"] = {"rrf_k": best[0], "dense_weight": best[1]}
        runs[f"hybrid{dims}"] = []
        for i in range(len(queries)):
            s = time.perf_counter()
            runs[f"hybrid{dims}"].append([d for d, _ in rrf([runs["bm25"][i], runs[f"dense{dims}"][i]], best[0], [1.0, best[1]], DEPTH)])
            latency.setdefault(f"hybrid{dims}_fusion_only", []).append((time.perf_counter() - s) * 1000)

    rerank_stats = None
    if args.rerank:  # rerank the top N of the hybrid that is best on VALIDATION
        best_h = max(DIMS, key=lambda d: statistics.fmean(score(queries[i], runs[f"hybrid{d}"][i])["ndcg@10"] for i in splits["VALIDATION"]))
        base, name = runs[f"hybrid{best_h}"], f"hybrid{best_h}+rerank{args.rerank}"
        need = sorted({c for r in base for c in r[:args.rerank]})
        with connect() as conn:
            text = dict(conn.execute("select c.id, concat_ws(E'\\n', p.title, c.section_title, c.text) from rx.chunks c "
                                     "left join rx.papers p on p.id = c.paper_id where c.id = any(%s)", (need,)).fetchall())
        gw = ModelGateway()

        def job(i):
            s = time.perf_counter()
            ranked, info = rerank(gw, queries[i]["query"], [(c, text[c]) for c in base[i][:args.rerank]])
            return ranked + base[i][args.rerank:], info, (time.perf_counter() - s) * 1000

        with ThreadPoolExecutor(8) as pool:
            done = list(pool.map(job, range(len(queries))))
        runs[name] = [r for r, _, _ in done]
        latency[f"{name}_llm_call"] = [ms for _, _, ms in done]
        infos = [x for _, x, _ in done]
        rerank_stats = {"base": f"hybrid{best_h}", "top_n": args.rerank, "ok": sum(x["ok"] for x in infos), "failed": sum(not x["ok"] for x in infos),
                        "missing_grades": sum(x.get("missing", 0) for x in infos), "input_tokens": sum(x.get("input_tokens") or 0 for x in infos),
                        "output_tokens": sum(x.get("output_tokens") or 0 for x in infos), "model": next((x["model"] for x in infos if x.get("model")), None)}

    per_query = {m: [score(q, r) for q, r in zip(queries, rs)] for m, rs in runs.items()}
    results = {}
    for split, idx in splits.items():
        for qtype in sorted({queries[i]["type"] for i in idx}):
            sel = [i for i in idx if queries[i]["type"] == qtype]
            results[f"{split}/{qtype}"] = {"n": len(sel), **{m: mean([per_query[m][i] for i in sel]) for m in runs}}
    report_idx = splits["TEST"] + splits["TEMPORAL_TEST"]
    best_dense = max(DIMS, key=lambda d: statistics.fmean(per_query[f"dense{d}"][i]["ndcg@10"] for i in splits["VALIDATION"]))
    cis = {f"hybrid{d}-bm25": bootstrap_ci([per_query[f"hybrid{d}"][i]["ndcg@10"] for i in report_idx],
                                           [per_query["bm25"][i]["ndcg@10"] for i in report_idx]) for d in DIMS}
    cis[f"hybrid{best_dense}-dense{best_dense}"] = bootstrap_ci([per_query[f"hybrid{best_dense}"][i]["ndcg@10"] for i in report_idx],
                                                                [per_query[f"dense{best_dense}"][i]["ndcg@10"] for i in report_idx])
    if rerank_stats:
        rr = f"{rerank_stats['base']}+rerank{rerank_stats['top_n']}"
        cis[f"{rr}-{rerank_stats['base']}"] = bootstrap_ci([per_query[rr][i]["ndcg@10"] for i in report_idx],
                                                           [per_query[rerank_stats["base"]][i]["ndcg@10"] for i in report_idx])
    lat = {m: {"p50_ms": round(float(np.percentile(v, 50)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2)} for m, v in latency.items()}
    out = {"release": args.release, "eval_set": eval_meta, "embedding": {k: emb_meta[k] for k in ("model", "dimensions", "embedding_version", "text_version")},
           "bm25": {"version": "bm25-0.2.0", "k1": bm25.k1, "b": bm25.b}, "depth": DEPTH, "tuned_on_validation": tuned,
           "results": results, "ndcg10_diff_ci95_test_plus_temporal": cis, "latency": lat, "rerank": rerank_stats,
           "notes": ["dense = exact cosine over the local float16 artifact (equivalent to pgvector exact search)",
                     "query embedding latency is the provider API round trip per batch of 64, excluded from search latency",
                     "template queries name model/benchmark explicitly, which favours lexical retrieval"],
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rep = ROOT / "data" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"retrieval_eval_v{args.release}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    methods = list(runs)
    lines = [f"# Retrieval evaluation v{args.release}", "", f"Eval set {eval_meta['version']} ({sum(eval_meta['counts'].values())} queries); "
             f"embedding {emb_meta['model']}; hybrid tuned on VALIDATION: {json.dumps(tuned)}", ""]
    for key, r in results.items():
        cols = ["recall@10", "recall@50", "mrr@10", "ndcg@10"]
        lines += [f"## {key} (n={r['n']})", "", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {m} | " + " | ".join(f"{r[m].get(c, 0):.3f}" for c in cols) + " |" for m in methods] + [""]
    lines += ["## nDCG@10 difference, TEST + TEMPORAL_TEST (mean, 95% CI)", ""] + [f"- {k}: {v}" for k, v in cis.items()]
    lines += ["", "## Latency (ms)", ""] + [f"- {k}: p50 {v['p50_ms']} / p95 {v['p95_ms']}" for k, v in lat.items()]
    (rep / f"retrieval_eval_v{args.release}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
