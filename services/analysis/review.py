"""Literature Review Completeness Auditor: what does a research question or draft review cover, and what
does the corpus suggest it leaves out?

The text is matched against the corpus's canonical entities (claims.names_in: datasets / benchmarks, models, methods,
tasks) and against the arXiv identifiers it cites. The scope is the tasks the text names, else the tasks of the
experiments on the datasets it names. Within the scope the corpus counts, per benchmark and per model family, how many
papers report results; frequently reported ones the text never names are coverage gaps. Entities the text names that
rest on a single corpus paper are flagged; contradictions recorded for scope experiments that involve a named entity are
listed; limitation passages of the most-cited scope papers are quoted, never summarised.

Nothing in the text leaves ResearchX unless the user asks for a runtime search; then only the recognised public entity
names are sent (never the draft), through the shared provider path (orchestrator._external, routed to the 'literature'
need, source gate and trace included). The report is not exhaustive: it states the searched scope (corpus release,
papers, experiments, years, providers), and absence from the corpus is not absence from the literature.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from services.analysis.claims import GENERIC_NAMES, _bench, _names, names_in
from services.canonicalization.normalize import canonical_label, model_family
from services.evidence import grounding as G

VERSION = "review-0.1.0"
ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
SECTIONS = r"limitation|discussion|future work"
MAX_TEXT = 20000


def family(name: str) -> str:
    """Leading word of a model name ("ResNet-50" -> resnet), as in the planner; the whole name when it has none."""
    m = re.match(r"[a-z]+", canonical_label(name))
    return (m.group(0) if m and len(m.group(0)) > 2 else None) or model_family(name)[0] or canonical_label(name)


def audit(index, text: str, tr, release: str | None = None, search=None) -> dict:
    """`search(tr, query) -> accepted runtime sources` runs the shared provider path; None: no runtime search."""
    text = (text or "").strip()[:MAX_TEXT]
    if len(text) < 20:
        raise ValueError("Paste a research question or a draft literature review (at least a sentence).")
    with tr.stage("analysis_request", method="canonical entity names and arXiv identifiers found in the text") as st:
        datasets = names_in(index, text, ("Dataset",))
        models = names_in(index, text, ("Model", "Method"), min_experiments=2)
        tasks = names_in(index, text, ("Topic",))
        cited = sorted({f"paper:arxiv:{m}" for m in ARXIV.findall(text)})
        cited_in = [p for p in cited if p in index.paper]
        st.detail.update(datasets=len(datasets), models=len(models), tasks=len(tasks), cited=len(cited), cited_in_corpus=len(cited_in))
    with tr.stage("experiment_retrieval", method="scope: the named tasks, else the tasks of the experiments on the named datasets or models") as st:
        task_ids = {i for i, _ in tasks}
        if not task_ids:
            seeds = [x for i, _ in datasets + models for x in index.ent_exps.get(i, [])]
            c = Counter(link["entity_id"] for x in seeds for link in index.exp[x]["links"] if link["node_type"] == "Topic")
            task_ids = {t for t, _ in c.most_common(2)}
        if not task_ids and not datasets:
            raise ValueError("The text names no task, dataset or model the corpus knows, so no scope can be formed. Name the task or the benchmarks.")
        scope = {x for t in task_ids for x in index.ent_exps.get(t, [])} | {x for i, _ in datasets for x in index.ent_exps.get(i, [])}
        papers = {index.exp[x]["paper_id"] for x in scope}
        years = sorted(y for p in papers if (y := index.paper.get(p, {}).get("year")))
        st.detail.update(tasks=[index.ent[t]["name"] for t in task_ids if t in index.ent], experiments=len(scope), papers=len(papers))
    with tr.stage("coverage", method="papers reporting each benchmark and model family in scope; the frequent ones the text never names") as st:
        bench_papers: dict[str, set] = defaultdict(set)
        fams: dict[str, dict] = defaultdict(lambda: {"names": Counter(), "papers": set()})
        for x in scope:
            pid = index.exp[x]["paper_id"]
            if b := _bench(index, x):
                bench_papers[b["entity_id"]].add(pid)
            nm = (_names(index, x) or [None])[0]
            if nm and canonical_label(nm) not in GENERIC_NAMES and re.search(r"[A-Za-z]{2}", nm):
                fams[family(nm)]["names"][nm] += 1
                fams[family(nm)]["papers"].add(pid)
        named_d, low = {i for i, _ in datasets}, text.casefold()
        named_fams = {family(n) for _, n in models}
        variant_of_named = lambda b: any(G.dataset_match(n, [index.ent[b]["name"]]) == 2 for _, n in datasets)  # noqa: E731 ("ADE20K val" for ADE20K)
        missing_benchmarks = [{"name": index.ent[b]["name"], "papers": len(ps)} for b, ps in sorted(bench_papers.items(), key=lambda kv: -len(kv[1]))
                              if b in index.ent and b not in named_d and len(ps) >= 2 and index.ent[b]["name"].casefold() not in low
                              and not variant_of_named(b)][:8]
        # whole-scope views (named and not), most reported first: the coverage chart
        scope_benchmarks = [{"name": index.ent[b]["name"], "papers": len(ps), "named": b in named_d or variant_of_named(b) or index.ent[b]["name"].casefold() in low}
                            for b, ps in sorted(bench_papers.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))) if b in index.ent][:12]
        scope_families = [{"family": f, "example": s["names"].most_common(1)[0][0], "papers": len(s["papers"]), "named": f in named_fams or f in low}
                          for f, s in sorted(fams.items(), key=lambda kv: (-len(kv[1]["papers"]), kv[0]))][:12]
        missing_models = [{"family": f, "example": s["names"].most_common(1)[0][0], "papers": len(s["papers"])}
                          for f, s in sorted(fams.items(), key=lambda kv: -len(kv[1]["papers"])) if f not in named_fams and f not in low and len(s["papers"]) >= 2][:10]
        thin = [{"name": n, "papers": k} for i, n in dict(datasets + models + tasks).items()
                if (k := len({index.exp[x]["paper_id"] for x in index.ent_exps.get(i, [])})) <= 1]
        named_ids = {i for i, _ in datasets + models}
        conflicts = [{"experiment": x, "label": index.exp[x]["label"], "field": c["field"], "value_a": c["value_a"], "value_b": c["value_b"],
                      "explanation": c.get("explanation")}
                     for x in sorted(scope) if named_ids & {link["entity_id"] for link in index.exp[x]["links"]} for c in index.contra.get(x, [])][:10]
        newest_cited = max((index.paper[p]["year"] for p in cited_in if index.paper[p].get("year")), default=None)
        newer = sum(1 for p in papers if (index.paper.get(p, {}).get("year") or 0) > (newest_cited or 9999))
        st.detail.update(benchmarks=len(bench_papers), families=len(fams), missing_benchmarks=len(missing_benchmarks),
                         missing_families=len(missing_models), thin=len(thin), conflicts=len(conflicts))
    with tr.stage("assessment", method="limitation / discussion passages of the most-cited scope papers (quoted)") as st:
        top = sorted(papers, key=lambda p: (-len(index.cites_in.get(p, ())), p))[:6]
        rows = index.q("select c.id, c.paper_id, c.section_title, c.text, c.unit_id, c.redistribution_allowed, p.arxiv_id from rx.chunks c "
                       "join rx.papers p on p.id = c.paper_id where c.paper_id = any(%s) and c.section_title ~* %s order by c.paper_id, c.id",
                       (top, SECTIONS)) if top else []
        limitations, seen = [], set()
        for cid, pid, sec, t, u, allowed, arxiv in rows:
            if pid in seen:  # one passage per paper
                continue
            seen.add(pid)
            limitations.append({"chunk_id": cid, "paper": pid, "title": index.paper.get(pid, {}).get("title"), "section": sec,
                                "text": t[:700] if allowed is not False else None, "text_withheld": allowed is False,
                                "url": f"https://arxiv.org/html/{arxiv}#{u}" if arxiv and u else None})
        st.detail.update(papers=len(top), passages=len(limitations))
    discovered, providers = [], []
    query = " ".join(dict.fromkeys([index.ent[t]["name"] for t in sorted(task_ids) if t in index.ent][:2] + [n for _, n in datasets[:2]]))
    if search is not None and query:  # only recognised entity names leave the system, never the pasted text
        accepted = search(tr, query)
        providers = sorted({a.get("provider") for a in accepted if a.get("provider")})
        for a in accepted:
            pid = f"paper:arxiv:{a['arxiv_id']}" if a.get("arxiv_id") else None
            discovered.append({"title": a.get("title"), "url": a["url"], "provider": a.get("provider"), "year": a.get("year"),
                               "in_corpus": bool(pid and pid in index.paper), "cited_in_text": bool(pid and pid in cited),
                               "reliability": a.get("reliability"), "status": "runtime metadata, not validated evidence"})
    gaps = len(missing_benchmarks) + len(missing_models) + len(thin)
    return {"kind": "review", "version": VERSION, "verdict": "GAPS_FOUND" if gaps else "NO_GAPS_IN_CORPUS_SCOPE",
            "recognised": {"datasets": [n for _, n in datasets], "models": [n for _, n in models], "tasks": [n for _, n in tasks],
                           "cited": cited, "cited_in_corpus": cited_in},
            "scope": {"tasks": [index.ent[t]["name"] for t in sorted(task_ids) if t in index.ent], "papers": len(papers), "experiments": len(scope),
                      "years": [years[0], years[-1]] if years else None, "corpus_release": release, "corpus_papers": len(index.paper),
                      "providers": providers or None, "runtime_query": query if search is not None else None},
            "missing_benchmarks": missing_benchmarks, "missing_models": missing_models, "thin_evidence": thin, "conflicts": conflicts,
            "scope_benchmarks": scope_benchmarks, "scope_families": scope_families,
            "recency": {"newest_cited_year": newest_cited, "newer_scope_papers": newer if newest_cited else None},
            "limitations": limitations, "discovered": discovered,
            "summary": {"missing_benchmarks": len(missing_benchmarks), "missing_models": len(missing_models), "thin": len(thin),
                        "conflicts": len(conflicts), "discovered": len(discovered), "not_in_corpus": sum(not d["in_corpus"] for d in discovered)},
            "inputs": {"chars": len(text), "runtime_search": search is not None},
            "notes": ["Coverage is measured against the ResearchX corpus, a sample of the literature: the report is not exhaustive.",
                      "A gap is a benchmark or model family other scope papers report and the text never names; whether it belongs in the review is the author's judgment.",
                      "Runtime results are search metadata that passed the source gate, not validated evidence.",
                      "The pasted text is never sent to external providers; a runtime search sends only the recognised task and dataset names."]}
