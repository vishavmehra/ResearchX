"""Load a frozen dataset release into Supabase PostgreSQL (schema rx) and verify it.

1. Every JSONL file's sha256 must match the frozen manifest, otherwise nothing is loaded.
2. Profile `serving` (default) loads what the runtime queries. ARCHIVE_ONLY files stay in the frozen release and get
   a sha256 + record-count check row (status ARCHIVE_ONLY), so nothing is dropped silently. Profile `full` loads all
   files (needs a database larger than the Free plan's 500 MB).
3. Records are stored losslessly compact (services/storage/records.py): expand(compact(r)) == r is checked for every
   record before writing, and again for every row read back after loading.
4. Retrieval chunks come from the parsed full text in the release's fulltext index (sha256-checked) plus one
   title+abstract chunk per paper.

Re-running replaces the rows of that release (idempotent). Usage:
    python scripts/db/load_release.py --version 0.2.0 [--profile serving|full] [--skip-chunks]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.storage.db import connect  # noqa: E402
from services.storage.records import canonical, compact, expand, file_constants  # noqa: E402


def g(r: dict, *path, default=None):
    for p in path:
        if not isinstance(r, dict):
            return default
        r = r.get(p)
    return r if r is not None else default


FILE = object()  # typed column holding the release-relative path of the source file


def spec(table: str, *cols):
    """cols: "name" (record key of the same name), ("name", "key"), ("name", callable) or ("name", FILE).
    Columns fed by a plain key hold that exact value, so compact() can drop it from the jsonb."""
    return table, [(c, c) if isinstance(c, str) else c for c in cols]


EVIDENCE = spec("evidence", ("id", "evidence_id"), "paper_id", "experiment_id", "field_name", "status", "location_kind", ("source_file", FILE))
COMPARISON = spec("comparisons", ("id", "comparison_id"), "experiment_a", "experiment_b", ("label", "relationship_label"),
                  ("is_hard_negative", lambda r: bool(r.get("is_hard_negative"))), ("source_file", FILE))
ASSOC = spec("associations", ("id", "association_id"), "result_id", "experiment_id", "status")


def other(id_field: str):
    return spec("other_records", ("id", lambda r: str(r[id_field])), ("record_type", lambda r: r.get("record_type", "?")), ("source_file", FILE))


MAP = {
    "normalized/sources/source_records.jsonl": spec("source_records", ("id", "source_record_id"), "source_name"),
    "normalized/papers/papers.jsonl": spec("papers", ("id", "paper_id"), "study_id", "arxiv_id", "doi", "title",
                                           ("publication_year", lambda r: r.get("publication_year")),
                                           ("is_relevant", lambda r: g(r, "relevance", "is_relevant")), "split"),
    "normalized/studies/studies.jsonl": spec("studies", ("id", "study_id")),
    "normalized/citations/citations.jsonl": spec("citations", "citing_paper_id", "cited_paper_id"),
    "normalized/discovery/candidates.jsonl": other("candidate_id"),
    "normalized/sources/fulltext_index.jsonl": other("arxiv_id"),
    "silver/experiments/experiments.jsonl": spec("experiments", ("id", "experiment_id"), "paper_id", "study_id", "experiment_type",
                                                 ("depth_tier", lambda r: g(r, "depth", "tier")), "split"),
    **{f"silver/evidence/{n}.jsonl": EVIDENCE for n in ("artifact_field_evidence", "deep_evidence", "experiment_evidence", "metadata_evidence", "rules_evidence")},
    "gold/evidence/evidence.jsonl": EVIDENCE,
    "silver/results/results.jsonl": spec("results", ("id", "result_id"), "paper_id"),
    "silver/results/associations.jsonl": ASSOC,
    "gold/results/associations.jsonl": ASSOC,
    "silver/artifacts/artifacts.jsonl": spec("artifacts", ("id", "artifact_id"), "artifact_type", "repository"),
    "silver/artifacts/artifact_evidence.jsonl": spec("artifact_relations", ("id", "artifact_evidence_id"), "paper_id", "experiment_id", "field_name", "relation"),
    "silver/comparisons/comparisons.jsonl": COMPARISON,
    "silver/comparisons/hard_negatives.jsonl": COMPARISON,
    "silver/study_links/study_links.jsonl": spec("study_links", ("id", "link_id"), "record_a", "record_b", "relationship"),
    "silver/contradictions/contradictions.jsonl": spec("contradictions", ("id", "contradiction_id"), "subject_id", "field"),
    "silver/abstention/abstention.jsonl": spec("abstention_cases", ("id", "case_id"), "expected_behavior"),
    "silver/tasks/research_tasks.jsonl": spec("research_tasks", ("id", "task_id"), "task_type", "expected_behavior"),
    "silver/tasks/hard_cases.jsonl": spec("hard_cases", ("id", "case_id"), "case_type"),
    "silver/entities/canonical_entities.jsonl": spec("canonical_entities", ("id", "canonical_id"), "entity_type", "normalized_name", "preferred_name"),
    "silver/graph/nodes.jsonl": spec("graph_nodes", ("id", "node_id"), "node_type", "ref_id"),
    "silver/graph/edges.jsonl": spec("graph_edges", ("id", "edge_id"), "edge_type", "source", "target"),
    "annotations/decisions/decisions.jsonl": other("decision_id"),
    "annotations/queue/batch-20260930-01.jsonl": other("item_id"),
}

# serving profile: what the runtime never queries stays in the frozen release (prefix -> reason)
ARCHIVE_ONLY = {
    "normalized/sources/source_records.jsonl": "raw provider payloads; papers carry the normalized metadata and provenance",
    "normalized/discovery/": "acquisition candidate pool, not research evidence",
    "silver/evidence/artifact_field_evidence.jsonl": "code/config field evidence; the runtime links paper-location evidence only",
    "silver/graph/": "derived from loaded tables by graph-0.1.0 (deterministic)",
    "silver/results/": "extraction intermediates; experiments carry the outcomes and their evidence ids",
    "gold/results/": "extraction intermediates (Gold review target)",
    "silver/comparisons/": "evaluation labels for the comparison stage",
    "gold/comparisons/": "evaluation labels for the comparison stage",
    "silver/abstention/": "evaluation data, kept out of the runtime store",
    "gold/abstention/": "evaluation data, kept out of the runtime store",
    "silver/tasks/": "evaluation data, kept out of the runtime store",
    "annotations/": "annotation workflow state (offline tooling)",
}


def archive_reason(rel: str) -> str | None:
    return next((why for prefix, why in ARCHIVE_ONLY.items() if rel.startswith(prefix)), None)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def table_text(t: dict) -> str:
    rows = t.get("rows") or []
    body = "\n".join(" | ".join(str(c) for c in row) for row in rows)
    return (t.get("caption") or "") + ("\n" + body if body else "")


def doc_units(d: dict, stats: Counter) -> list[tuple[str, str, str | None, str]]:
    """(unit_type, unit_id, section_id, text) in document order. When the parser reused a unit id, a colliding unit whose
    text survives under another unit is dropped; any other collision is renamed '<id>~<n>'. No text is lost."""
    units = [(t, uid, sid, (txt or "").strip()) for t, uid, sid, txt in (
        *(("paragraph", p["paragraph_id"], p.get("section_id"), p.get("text")) for p in d.get("paragraphs", [])),
        *(("table", t["table_id"], t.get("section_id"), table_text(t)) for t in d.get("tables", [])),
        *(("caption", f["figure_id"], f.get("section_id"), f.get("caption")) for f in d.get("figures", [])),
        *(("footnote", f["footnote_id"], None, f.get("text")) for f in d.get("footnotes", [])))]
    units = [u for u in units if u[3]]
    ids, texts = Counter(u[1] for u in units), Counter((u[0], u[3]) for u in units)
    kept, seen = [], Counter()
    for u in units:
        if ids[u[1]] > 1 and texts[(u[0], u[3])] > 1:
            ids[u[1]] -= 1
            texts[(u[0], u[3])] -= 1
            stats["duplicate_units_dropped"] += 1
            continue
        seen[u[1]] += 1
        stats["renamed_unit_ids"] += seen[u[1]] > 1
        kept.append((u[0], u[1] if seen[u[1]] == 1 else f"{u[1]}~{seen[u[1]]}", u[2], u[3]))
    return kept


def build_chunks(version: str, papers: dict[str, dict]) -> tuple[list[list], dict]:
    index = read_jsonl(ROOT / "data/normalized/sources/fulltext_index.jsonl")
    chunks, stats = [], Counter(docs=0, doc_sha_mismatch=0, missing_parsed=0, duplicate_units_dropped=0, renamed_unit_ids=0)
    for p in papers.values():  # paper-level unit for every paper (title + abstract)
        text = " ".join(x for x in (p.get("title"), p.get("abstract")) if x)
        if text:
            lic = p.get("license") or {}
            chunks.append([f"{p['paper_id']}#paper", version, p["paper_id"], "paper", "paper", None, text, len(text) // 4,
                           None, None, lic.get("content_license"), lic.get("redistribution_allowed")])
    for row in index:
        ft = row.get("fulltext") or {}
        pp = ft.get("parsed_path")
        if not pp or not (ROOT / pp).is_file():
            stats["missing_parsed"] += 1
            continue
        src = ROOT / ft["path"]
        if not src.is_file() or sha256(src) != ft["sha256"]:
            stats["doc_sha_mismatch"] += 1  # never index text that differs from the recorded source
            continue
        d = json.loads((ROOT / pp).read_text(encoding="utf-8"))
        stats["docs"] += 1
        pid, lic = d["paper_id"], d.get("license") or {}
        sec = {s["section_id"]: s.get("title") for s in d.get("sections", [])}
        base = [d.get("source_record_id"), d.get("source_sha256"), lic.get("content_license"), lic.get("redistribution_allowed")]
        for unit_type, unit_id, section_id, text in doc_units(d, stats):
            chunks.append([f"{pid}#{unit_id}", version, pid, unit_type, unit_id, sec.get(section_id), text, len(text) // 4, *base])
    seen, uniq = set(), []
    for c in chunks:
        if c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    stats["duplicate_chunk_ids"] = len(chunks) - len(uniq)
    stats["chunks"] = len(uniq)
    return uniq, dict(stats)


def typed_of(cols, values) -> dict:
    """record key -> value of the typed text columns fed by a plain key."""
    return {src: v for (_, src), v in zip(cols, values) if isinstance(src, str)}


def verify_roundtrip(conn, version: str, files: dict, loaded: list[str]) -> bool:
    """Read every loaded row back, expand it and compare with the frozen file."""
    prov = dict(conn.execute("select id, provenance from rx.provenance").fetchall())
    consts = dict(conn.execute("select source_file, constants from rx.load_checks where release=%s", (version,)).fetchall())
    ok = True
    for path in loaded:
        rel = path.removeprefix("data/")
        table, cols = MAP[rel]
        by_file = any(src is FILE for _, src in cols)
        rows = conn.execute(f"select {', '.join(c for c, _ in cols)}, record, provenance_id from rx.{table} where release=%s"
                            + (" and source_file=%s" if by_file else ""), (version, rel) if by_file else (version,)).fetchall()
        got = sorted(canonical(expand(r[-2], typed_of(cols, r[:-2]), consts[path], r[-1], prov)) for r in rows)
        same = got == sorted(canonical(r) for r in read_jsonl(ROOT / path))
        ok &= same
        print(f"{'roundtrip OK ' if same else 'roundtrip BAD'} {len(rows):>7}  {path}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--profile", choices=["serving", "full"], default="serving")
    ap.add_argument("--skip-chunks", action="store_true")
    args = ap.parse_args()
    mpath = ROOT / f"data/manifests/releases/v{args.version}.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    files = {f["path"]: f for f in manifest["checksums"] if f["path"].endswith(".jsonl")}

    # 1. checksums first: refuse to load anything if a single file differs from the frozen manifest
    bad = [p for p, f in files.items() if sha256(ROOT / p) != f["sha256"]]
    unmapped = [p for p in files if p.removeprefix("data/") not in MAP and files[p]["records"] > 0]
    if bad or unmapped:
        sys.exit(f"refusing to load: checksum mismatch {bad} / unmapped non-empty files {unmapped}")
    print(f"checksums OK for {len(files)} files | profile {args.profile}", flush=True)
    archive = args.profile == "serving"

    with connect() as conn:
        conn.execute("insert into rx.dataset_releases (version, manifest_sha256, processing_commit, schema_version, manifest, load_status) "
                     "values (%s,%s,%s,%s,%s,'LOADING') on conflict (version) do update set load_status='LOADING', loaded_at=now()",
                     (args.version, sha256(mpath), manifest["processing_commit"], manifest["schema_version"],
                      json.dumps({k: v for k, v in manifest.items() if k != "checksums"})))
        tables = sorted({m[0] for m in MAP.values()})
        targets = tables + ([] if args.skip_chunks else ["chunks"])
        if conn.execute("select count(*) from rx.dataset_releases where version <> %s", (args.version,)).fetchone()[0] == 0:
            # sole release: truncate frees the space at once (delete leaves dead tuples that still count against quota)
            extra = [] if args.skip_chunks else ["chunk_embeddings"]
            conn.execute(f"truncate {', '.join('rx.' + t for t in targets + extra + ['provenance'])}")
        else:
            for t in targets:
                conn.execute(f"delete from rx.{t} where release = %s", (args.version,))
        conn.execute("delete from rx.load_checks where release = %s", (args.version,))
        conn.commit()

        seen_prov = {r[0] for r in conn.execute("select id from rx.provenance").fetchall()}
        all_ok, papers, loaded_files = True, {}, []
        for path, f in sorted(files.items()):
            rel = path.removeprefix("data/")
            recs = read_jsonl(ROOT / path)
            if rel == "normalized/papers/papers.jsonl":
                papers = {r["paper_id"]: r for r in recs}
            why = archive_reason(rel) if archive else None
            constants, loaded = {}, 0
            if why is None and recs:
                table, cols = MAP[rel]
                constants = file_constants(recs)
                keys = {src for _, src in cols if isinstance(src, str)}
                rows, provs = [], {}
                for r in recs:
                    vals = [rel if src is FILE else src(r) if callable(src) else r.get(src) for _, src in cols]
                    c, pid = compact(r, keys, constants)
                    if pid:
                        provs.setdefault(pid, r["provenance"])
                    if expand(c, {k: v for k, v in typed_of(cols, vals).items() if isinstance(v, str)}, constants, pid, provs) != r:
                        sys.exit(f"compaction is not lossless for {rel}: {vals[0]}")
                    rows.append([*vals, args.version, json.dumps(c, ensure_ascii=False), pid])
                with conn.cursor() as cur:
                    with cur.copy("copy rx.provenance (id, provenance) from stdin") as cp:
                        for pid, p in provs.items():
                            if pid not in seen_prov:
                                cp.write_row([pid, json.dumps(p, ensure_ascii=False)])
                                seen_prov.add(pid)
                    with cur.copy(f"copy rx.{table} ({', '.join(c for c, _ in cols)}, release, record, provenance_id) from stdin") as cp:
                        for row in rows:
                            cp.write_row(row)
                loaded = len(rows)
                loaded_files.append(path)
            ok = (len(recs) if why else loaded) == f["records"]
            all_ok &= ok
            conn.execute("insert into rx.load_checks (release, source_file, file_sha256, manifest_records, loaded_records, ok, status, constants) "
                         "values (%s,%s,%s,%s,%s,%s,%s,%s)", (args.version, path, f["sha256"], f["records"], loaded, ok,
                                                             "ARCHIVE_ONLY" if why else "LOADED", json.dumps(constants, ensure_ascii=False)))
            conn.commit()
            print(f"{'OK ' if ok else 'BAD'} {('archive ' + str(len(recs))) if why else loaded:>14} / {f['records']:>7}  {path}", flush=True)

        if not args.skip_chunks:
            chunks, stats = build_chunks(args.version, papers)
            cols = ["id", "release", "paper_id", "unit_type", "unit_id", "section_title", "text", "token_estimate", "source_record_id",
                    "source_sha256", "content_license", "redistribution_allowed"]
            with conn.cursor() as cur, cur.copy(f"copy rx.chunks ({', '.join(cols)}) from stdin") as cp:
                for c in chunks:
                    cp.write_row(c)
            conn.commit()
            n = conn.execute("select count(*) from rx.chunks where release=%s", (args.version,)).fetchone()[0]
            print(f"chunks: {stats} | in db {n}", flush=True)
            all_ok &= n == stats["chunks"] and stats["doc_sha_mismatch"] == 0

        all_ok &= verify_roundtrip(conn, args.version, files, loaded_files)
        conn.execute("update rx.dataset_releases set load_status=%s where version=%s", ("VERIFIED" if all_ok else "FAILED", args.version))
        conn.commit()
        db, total = conn.execute("select pg_database_size(current_database()), sum(pg_database_size(datname)) from pg_database").fetchone()
        print(f"release {args.version}: {'VERIFIED' if all_ok else 'FAILED'} | database {db / 2**20:.0f} MiB | all databases {total / 2**20:.0f} MiB "
              f"(Free-plan quota 500 MB)")
        if not all_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
