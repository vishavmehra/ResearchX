"""Artifact linkage: official GitHub repositories of full-text papers -> pinned snapshot -> configuration relations.

Repository values stay supporting evidence (CONFIRMED_BY_BOTH / CONFLICTING / PRESENT_IN_REPOSITORY /
RELATED_BUT_UNCONFIRMED); only artifacts.repository_url is set on experiments, and only with paper or PwC evidence.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.datasets.schemas import EvidenceStatus, Experiment, FailureCategory, Paper, SourceRecord
from services.datasets.store import DATA, path, read_jsonl, read_models, run_stage, write_jsonl
from services.evidence.assertions import load_parsed_docs, verify
from services.retrieval.sources import github_artifacts as gh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="at most N repositories (0 = all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--refresh", action="store_true", help="re-clone and re-pin repositories instead of replaying pinned snapshots")
    args = ap.parse_args()
    with run_stage("link_artifacts", vars(args)) as run:
        papers = {p.paper_id: p for p in read_models(path("papers"), Paper)}
        docs = load_parsed_docs()
        records = {r.source_record_id: r for r in read_models(path("source_records"), SourceRecord)}
        exps = read_models(path("experiments"), Experiment)
        by_paper: dict[str, list[Experiment]] = {}
        for e in exps:
            by_paper.setdefault(e.paper_id, []).append(e)
        evidence = {e["evidence_id"]: e for f in sorted((DATA / "silver" / "evidence").glob("*.jsonl"))
                    if f.name != "artifact_field_evidence.jsonl" for e in read_jsonl(f)}
        pwc_link_ev = {str(e.get("extracted_value") or "").lower().rstrip("/"): e["evidence_id"] for e in evidence.values()
                       if e.get("field_name") == "artifacts.repository_url"}
        jobs = []
        for pid in sorted(docs):
            p = papers.get(pid)
            if p is None or not p.arxiv_id:
                continue
            pwc = records.get(f"pwc:{p.arxiv_id}")
            oai = records.get(f"arxivoai:{p.arxiv_id}")
            arx = next((records[s] for s in p.source_record_ids if s.startswith("arxiv:") and s in records), None)
            comments = " ".join(x for x in [(oai.extra.get("comments") if oai else None), (arx.extra.get("comment") if arx else None)] if x)
            for c in gh.candidates(p, docs[pid], (pwc.extra.get("code_links", []) if pwc else []), comments):
                jobs.append((pid, c))
        seen, uniq = set(), []
        for pid, c in jobs:
            key = (pid, c.owner.lower(), c.repo.lower())
            if key not in seen:
                seen.add(key)
                uniq.append((pid, c))
        if args.limit:
            uniq = uniq[: args.limit]
        counts: Counter = Counter(candidates=len(uniq))
        snaps: dict[tuple[str, str], object] = {}

        def fetch(c):  # a pinned snapshot is replayed from the cache, so re-runs reproduce the same records
            return (None if args.refresh else gh.load_snapshot(c.owner, c.repo)) or gh.snapshot(c.owner, c.repo)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(fetch, c): (pid, c) for pid, c in uniq if (c.owner.lower(), c.repo.lower()) not in snaps}
            for i, f in enumerate(as_completed(futs), 1):
                pid, c = futs[f]
                try:
                    snaps[(c.owner.lower(), c.repo.lower())] = f.result()
                    counts["snapshots_ok"] += 1
                except Exception as err:  # network / missing repo: recorded, never fatal
                    gh.fail(run, c.owner, c.repo, err)
                    counts["snapshots_failed"] += 1
                if i % 25 == 0:
                    print(f"  repositories {i}/{len(futs)}", flush=True)

        artifacts, field_evs, relations = {}, {}, {}
        for pid, c in uniq:
            snap = snaps.get((c.owner.lower(), c.repo.lower()))
            if snap is None:
                continue
            cached = gh.store_files(snap)
            link_ev = gh.link_evidence(papers[pid], docs.get(pid), c, run)
            if link_ev and link_ev.evidence_id not in evidence:  # already stored by another stage: cite it, do not duplicate
                field_evs[link_ev.evidence_id] = link_ev
            link_id = link_ev.evidence_id if link_ev else pwc_link_ev.get(f"https://github.com/{c.owner}/{c.repo}".lower())
            arts, evs, rels = gh.link(papers[pid], by_paper.get(pid, []), c, snap, cached, evidence, run, link_id)
            for a in arts:
                if a.artifact_id in artifacts:
                    artifacts[a.artifact_id].linked_paper_ids = sorted(set(artifacts[a.artifact_id].linked_paper_ids) | set(a.linked_paper_ids))
                else:
                    artifacts[a.artifact_id] = a
            field_evs.update({e.evidence_id: e for e in evs})
            relations.update({r.artifact_evidence_id: r for r in rels})
            counts[f"basis_{c.basis}"] += 1
        counts.update(f"relation_{r.relation.value}" for r in relations.values())
        # values were read from the cached pinned files: re-find each line there now (same rule as extract_evidence)
        art_dicts = {k: a.model_dump(mode="json") for k, a in artifacts.items()}
        for ev in field_evs.values():
            if ev.artifact_id:
                ok, reason = verify(ev, {}, {}, art_dicts)
                ev.status = EvidenceStatus.SOURCE_CHECKED if ok else EvidenceStatus.REJECTED
                counts[f"artifact_evidence_{ev.status.value}"] += 1
                if not ok:
                    run.fail(FailureCategory.extraction_failure, f"{ev.evidence_id} {ev.field_name}", "artifact line re-found in cached file", reason)
        run.output(write_jsonl(path("artifacts"), [artifacts[k] for k in sorted(artifacts)]))
        run.output(write_jsonl(path("artifact_field_evidence"), [field_evs[k] for k in sorted(field_evs)]))
        run.output(write_jsonl(path("artifact_evidence"), [relations[k] for k in sorted(relations)]))
        run.output(write_jsonl(path("experiments"), sorted(exps, key=lambda e: e.experiment_id)))
        counts.update(artifacts=len(artifacts), artifact_field_evidence=len(field_evs), artifact_relations=len(relations),
                      experiments_with_repository_url=sum(1 for e in exps if e.artifacts.repository_url.status.value == "KNOWN"))
        run.counts.update({k: int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
