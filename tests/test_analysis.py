"""Analysis features A, B, C, H over a synthetic corpus (no network): the real GraphIndex, comparison engine and field
facts run on invented records, so nothing can pass by matching a string hardcoded in the modules."""

import pytest

from services.analysis import benchmark as bench
from services.analysis import claims, discrepancy, enrich, papers, planner, recipe, review, transfer
from services.evidence import grounding as G
from services.canonicalization.entities import entity_id
from services.canonicalization.normalize import model_family
from services.datasets.schemas import CanonicalEntityType as E
from services.graph import knowledge_map as km
from tests.test_knowledge_graph import FakeCorpus, FakeTracer, corpus, exp_record

BANNED_CAUSAL = ("caused", "because of", "explains the gap", "is the reason")


class AnalysisCorpus(FakeCorpus):
    """FakeCorpus plus the statements the analysis features issue (evidence detail, paper detail, passages)."""

    def __init__(self, *a, chunks=None, paper_evidence=(), sections=(), fulltext=None, evidence_text=None, **k):
        super().__init__(*a, **k)
        self.chunks, self.paper_evidence, self.sections, self.fulltext = chunks or {}, list(paper_evidence), list(sections), fulltext or {}
        self.evidence_text = evidence_text or {}  # evidence id -> record (evidence_text, section) a record's own value cites

    def __call__(self, sql, params=()):
        s = " ".join(sql.split())
        if "from rx.chunks c join rx.papers p" in s and "c.unit_type in" in s:  # a paper's stored full text (enrich)
            self.sql.append(s)
            return self.fulltext.get(params[0], [])
        if "from rx.chunks c join rx.papers p" in s:  # limitation / discussion passages (transfer, review)
            self.sql.append(s)
            return [r for r in self.sections if r[1] in params[0]]
        if s.startswith("select id, paper_id, experiment_id, field_name, status, location_kind, record from rx.evidence"):
            self.sql.append(s)
            return [(i, None, None, "field", "MACHINE_EXTRACTED", "PAPER_TABLE_CELL",
                     {"evidence_text": f"text of {i}", "source_location": f"https://arxiv.org/html/9901.00001#{i}", "section": "S4", "table": "T1"})
                    for i in params[0]]
        if s.startswith("select id, field_name, record from rx.evidence where id = any"):  # stage classification (configs.evidence_rows)
            self.sql.append(s)
            stated = {r[0]: (r[2], r[3]) for r in self.paper_evidence}  # paper-level statements by id (attached at read time)
            return [(i, stated.get(i, ("field",))[0], self.evidence_text.get(i) or (stated[i][1] if i in stated else
                                                                                 {"evidence_text": f"text of {i}", "section": "S4"})) for i in params[0]]
        if s.startswith("select title, publication_year, arxiv_id, doi"):
            self.sql.append(s)
            p = next(p for p in self.papers if p["id"] == params[0])
            return [(p["title"], p["year"], p["id"].rsplit(":", 1)[-1], None, [{"name": "A. Author"}], "An abstract.", None)]
        if s.startswith("select id, paper_id, field_name, record from rx.evidence where paper_id = any"):
            self.sql.append(s)
            return [r for r in self.paper_evidence if r[1] in params[0] and r[2] in params[1]]
        if "from rx.chunks where paper_id" in s:
            self.sql.append(s)
            return self.chunks.get(params[0], [])
        if s.startswith("select id from rx.papers where arxiv_id = %s"):  # runtime paper resolution by identifier
            self.sql.append(s)
            return [(p["id"],) for p in self.papers if p["id"].rsplit(":", 1)[-1] == params[0]][:1]
        if s.startswith("select id from rx.papers where lower(doi) = %s"):
            self.sql.append(s)
            return []
        if s.startswith("select arxiv_id from rx.papers where id"):
            self.sql.append(s)
            return [(params[0].rsplit(":", 1)[-1],)]
        return super().__call__(sql, params)


def build(extra_records=None, chunks=None, names=None, paper_evidence=(), sections=(), fulltext=None, evidence_text=None):
    base, n = corpus(names)
    recs = {**base.records, **(extra_records or {})}
    papers_ = list(base.papers)
    if any(r["paper_id"] == "paper:arxiv:9901.00003" for r in recs.values()):
        papers_.append({"id": "paper:arxiv:9901.00003", "title": "A third paper", "year": 2024})
    q = AnalysisCorpus(papers_, recs, citations=base.citations, contradictions=base.contradictions, evidence=base.evidence, chunks=chunks,
                       paper_evidence=paper_evidence, sections=sections, fulltext=fulltext, evidence_text=evidence_text)
    return km.GraphIndex(q), q, n


def test_paper_level_evidence_fills_a_cell_no_experiment_states_and_says_so():
    pe = [("ev:p1", "paper:arxiv:9901.00001", "training.optimizer", {"extracted_value": "AdamW", "evidence_text": "We use AdamW.", "applies_to": None})]
    idx, _, n = build(paper_evidence=pe)
    d = papers.diff(idx, ["paper:arxiv:9901.00001", "paper:arxiv:9901.00002"], FakeTracer())
    rows = {r["field"]: r for s in d["sections"] for r in s["rows"]}
    v = rows["training.optimizer"]["cells"][0]["values"][0]
    assert v["value"] == "AdamW" and v["status"] == "DERIVED" and rows["training.optimizer"]["status"] == "PARTIAL", \
        "the paper's only optimizer statement is attached to its experiments as DERIVED (read time)"
    assert rows["training.learning_rate"]["cells"][1]["values"][0]["scope"] == "experiment", "a record's own value comes first"
    assert d["shared"]["benchmarks"][0]["name"] == n["bench"] and d["citations"] == [{"citing": "paper:arxiv:9901.00002", "cited": "paper:arxiv:9901.00001"}]
    assert d["shared_results"][0]["benchmark"] == n["bench"] and all(d["shared_results"][0]["cells"])
    r = discrepancy.investigate(idx, "exp:a0", "exp:b0", FakeTracer())
    assert not any(u["paper_level"] for u in r["unknown"]), "paper B states nothing for the fields its record leaves unknown"


def test_paper_level_settings_attach_only_when_unambiguous_and_never_to_evaluation_conditions():
    def ev(i, field, value, applies_to=None):
        return (f"ev:pl{i}", P1, field, {"extracted_value": value, "evidence_text": f"... {value} ...", "applies_to": applies_to})
    pe = [ev(1, "training.batch_size", "32", "Kappa-Bench"), ev(2, "training.batch_size", "8", "Omega-Set"),  # aimed at the dataset
          ev(3, "training.epochs", "90"),                                                                      # the only value
          ev(4, "training.weight_decay", "0.05", "default"), ev(5, "training.weight_decay", "0.1", "Omega-Set"),  # the default
          ev(6, "training.dropout", "0.1"), ev(7, "training.dropout", "0.3")]                                   # ambiguous
    pe.append(("ev:pl8", P2, "evaluation.evaluation_protocol", {"extracted_value": "linear probe", "evidence_text": "linear probe"}))  # never attached
    idx, _, _ = build(paper_evidence=pe)
    assert idx.experiments(["exp:b0"])["exp:b0"].evaluation.evaluation_protocol.status.value == "UNKNOWN", "evaluation conditions are never carried over"
    e = idx.experiments(["exp:a0"])["exp:a0"]
    assert (e.training.batch_size.raw_value, e.training.batch_size.derived) == ("32", True) and "for Kappa-Bench" in e.training.batch_size.derivation
    assert e.training.epochs.raw_value == "90" and "only value" in e.training.epochs.derivation
    assert e.training.weight_decay.raw_value == "0.05" and "default" in e.training.weight_decay.derivation
    assert e.training.dropout.status.value == "UNKNOWN", "two different values: ambiguous, left unknown"
    assert e.evaluation.evaluation_protocol.raw_value == "single crop", "the record's own value is never replaced"


def test_full_text_is_read_for_unknown_settings_and_only_validated_quotes_are_attached():
    enrich._tried.clear()
    text = "We train Orion-3 on Kappa-Bench with a batch size of 64 for 30 epochs."
    idx, _, _ = build(fulltext={P2: [("ch:1", "paragraph", "S4.p2", "Implementation", text, True, "9901.00002")]})
    calls = []

    def extract(req, fields, sources):
        calls.append([s["chunk_id"] for s in sources])
        return [G.Fact(field="training.batch_size", raw="64", value="64", span="batch size of 64", chunk_id="ch:1", origin="runtime extraction"),
                G.Fact(field="training.optimizer", raw="SGD", value="SGD", span="optimised with SGD", chunk_id="ch:1", origin="runtime extraction")]
    models = idx.experiments(["exp:b0"])
    enrich.fill(idx, models, FakeTracer(), extract)
    e = models["exp:b0"]
    assert (e.training.batch_size.raw_value, e.training.batch_size.derived) == ("64", True) and "full text" in e.training.batch_size.derivation
    assert e.training.optimizer.status.value == "UNKNOWN", "a quote that is not in the passage is never attached"
    ev = idx.evidence_detail(e.training.batch_size.evidence_ids)[0]
    assert ev["status"] == "RUNTIME_EXTRACTED" and ev["text"] == "batch size of 64" and ev["source_location"].endswith("#S4.p2")
    assert enrich.fill(idx, idx.experiments(["exp:b0"]), FakeTracer(), extract)["status"].startswith("nothing to read") and len(calls) == 1, \
        "searched settings are not asked for again"
    assert enrich.fill(idx, models, FakeTracer(), None) == {"status": "not run"}


# ------------------------------------------------------------------------------------------------ C: recipe
def test_recipe_keeps_statuses_and_never_fills_a_default():
    idx, _, n = build()
    tr = FakeTracer()
    r = recipe.build(idx, "exp:b0", "0.2.0", tr)
    facts = {f["field"]: f for s in r["sections"] for f in s["facts"]}
    assert facts["training.learning_rate"]["status"] == "REPORTED" and facts["training.learning_rate"]["value"] == 0.01
    assert facts["training.optimizer"]["status"] == "UNKNOWN" and facts["training.optimizer"]["value"] is None
    assert facts["training.learning_rate"]["evidence"][0]["source_location"].startswith("https://arxiv.org/html/")
    assert {"random seed", "evaluation split", "metric definition", "optimizer"} <= set(r["missing"])
    assert "- [ ] optimizer: not stated in the sources read" in r["markdown"] and "- [x] learning rate: 0.01" in r["markdown"]
    assert r["export"]["format"] == "researchx-recipe" and len(r["export"]["fields"]) == sum(len(s["facts"]) for s in r["sections"])
    assert [s.name for s in tr.rows] == ["experiment_retrieval", "field_facts"]
    with pytest.raises(KeyError):
        recipe.build(idx, "exp:nope", "0.2.0", FakeTracer())


def test_recipe_marks_derived_and_conflicting_values():
    rec = exp_record("exp:c0", "paper:arxiv:9901.00003", "Lyra-5", "Kappa-Bench", "Glyph Sorting", "Q-Score", 50.0)
    rec["training"] = {"epochs": {"status": "KNOWN", "raw_value": "3 epochs", "value": 3, "evidence_ids": ["ev:c0e"], "derived": True,
                                  "derivation": "steps / steps per epoch"},
                       "batch_size": {"status": "CONFLICTING", "evidence_ids": ["ev:c0b1", "ev:c0b2"]}}
    idx, _, _ = build({"exp:c0": rec})
    r = recipe.build(idx, "exp:c0", "0.2.0", FakeTracer())
    facts = {f["field"]: f for s in r["sections"] for f in s["facts"]}
    assert facts["training.epochs"]["status"] == "DERIVED" and facts["training.epochs"]["derivation"] == "steps / steps per epoch"
    assert facts["training.batch_size"]["status"] == "CONFLICTING" and r["conflicting"] == ["batch size"]
    assert "sources disagree" in r["markdown"]


# ------------------------------------------------------------------------------------------------ A: paper diff
def test_paper_diff_rows_show_same_different_partial_and_unknown():
    chunks = {"paper:arxiv:9901.00001": [("c1", "Limitations", "Our study is limited to one benchmark.", "S7.p1", True)],
              "paper:arxiv:9901.00002": [("c2", "Conclusion", "withheld text", "S8.p1", False)]}
    idx, _, n = build(chunks=chunks)
    tr = FakeTracer()
    d = papers.diff(idx, ["paper:arxiv:9901.00001", "paper:arxiv:9901.00002"], tr)
    rows = {r["field"]: r for s in d["sections"] for r in s["rows"]}
    assert rows["evaluation.benchmark"]["status"] == "SAME" and rows["model.model_name"]["status"] == "DIFFERENT"
    assert rows["training.learning_rate"]["status"] == "PARTIAL" and rows["training.seed"]["status"] == "UNKNOWN"
    assert rows["model.model_name"]["cells"][0]["values"][0]["value"] == n["model_a"]
    assert rows["model.model_name"]["cells"][0]["values"][0]["evidence"][0]["id"] == "ev:exp:a0m"
    p1, p2 = d["papers"]
    assert p1["passages"][0]["text"] == "Our study is limited to one benchmark." and p2["passages"][0]["text"] is None
    assert d["fairness"][0]["verdict"] == "CONDITIONALLY_COMPARABLE", "paper B states no split/protocol: equivalence cannot be established"
    assert {s.name for s in tr.rows} == {"experiment_retrieval", "field_facts", "comparison"}


def test_papers_without_a_shared_benchmark_are_not_comparable():
    other = {"exp:c0": exp_record("exp:c0", "paper:arxiv:9901.00003", "Lyra-5", "Sigma-Set", "Glyph Sorting", "Q-Score", 50.0)}
    idx, _, _ = build(other)
    d = papers.diff(idx, ["paper:arxiv:9901.00001", "paper:arxiv:9901.00003"], FakeTracer())
    assert d["verdict"] == "NOT_COMPARABLE" and d["fairness"][0]["shared_benchmarks"] == []
    with pytest.raises(ValueError):
        papers.diff(idx, ["paper:arxiv:9901.00001"], FakeTracer())


# ------------------------------------------------------------------------------------------------ B: discrepancy
def test_discrepancy_lists_differences_as_hypotheses_and_unknowns_as_missing():
    idx, _, n = build()
    r = discrepancy.investigate(idx, "exp:a0", "exp:b0", FakeTracer())
    assert r["observed"][0]["difference"] == 10.0 and r["verdict"] == "UNDETERMINED"
    assert any(u["field"] == "evaluation.dataset_split" and u["missing_on"] == ["B"] for u in r["unknown"])
    assert r["hypotheses"] and all(h["status"] == "HYPOTHESIS" and "may contribute" in h["statement"] for h in r["hypotheses"])
    text = " ".join(h["statement"] for h in r["hypotheses"]).lower()
    assert not any(b in text for b in BANNED_CAUSAL), "a hypothesis never asserts a cause"
    assert r["confidence"]["level"] == "LOW" and r["contradictions"], "exp:b0 carries a recorded score contradiction"


def test_aligned_setups_with_different_scores_are_unexplained_not_explained():
    idx, _, _ = build()
    r = discrepancy.investigate(idx, "exp:a0", "exp:a1", FakeTracer())
    assert r["label"] == "ALIGNED_EXPERIMENT" and r["verdict"] == "UNEXPLAINED" and not r["hypotheses"]
    c = discrepancy.candidates(idx)
    assert any({p["a"], p["b"]} == {"exp:a0", "exp:a1"} for p in c["pairs"]) and c["recorded"][0]["experiment"] == "exp:b0"


def test_model_name_components_one_side_only_and_variants():
    d = discrepancy.name_delta("HRNetV2 + OCR + RMI (PaddleClas pretrained)", "OCR(HRNetV2-W48)")
    assert d == {"only_a": ["RMI", "PaddleClas pretrained"], "only_b": [], "variants": [["HRNetV2", "HRNetV2-W48"]]}
    assert discrepancy.name_delta("Lyra-3", "Lyra-3") == {"only_a": [], "only_b": [], "variants": []}


def test_discrepancy_findings_state_gap_held_equal_and_one_sided_fields():
    idx, _, _ = build()
    r = discrepancy.investigate(idx, "exp:a0", "exp:b0", FakeTracer())
    text = " ".join(r["findings"])
    assert "A is 10 higher" in text or "A is 10 lower" in text
    assert "Stated on one side only" in text and "evaluation split (A only)" in text
    assert not any(b in text.lower() for b in BANNED_CAUSAL)
    assert all(u["both"] == (len(u["missing_on"]) == 2) for u in r["unknown"])


def test_candidates_for_one_experiment_are_oriented_with_it_as_a():
    idx, _, _ = build()
    c = discrepancy.candidates(idx, experiment="exp:a1")
    assert c["pairs"] and all(p["a"] == "exp:a1" for p in c["pairs"])
    with pytest.raises(KeyError):
        discrepancy.candidates(idx, experiment="exp:missing")


# ------------------------------------------------------------------------------------------------ E: claims
def _claim_corpus():
    p3 = "paper:arxiv:9901.00003"
    extra = {"exp:c0": exp_record("exp:c0", p3, "Lyra-5", "Kappa-Bench", "Glyph Sorting", "Accuracy", 80.0),
             "exp:c1": exp_record("exp:c1", p3, "Lyra-6", "Kappa-Bench", "Glyph Sorting", "Accuracy", 75.0),
             "exp:c2": exp_record("exp:c2", "paper:arxiv:9901.00002", "Lyra-6", "Kappa-Bench", "Glyph Sorting", "Accuracy", 90.0)}
    return build(extra)[0]


def test_claim_comparison_counts_same_paper_pairs_and_leaves_cross_paper_inconclusive():
    idx = _claim_corpus()
    r = claims.audit(idx, "Lyra-5 outperforms Lyra-6 on Kappa-Bench", FakeTracer())
    assert r["verdict"] == "SUPPORTED" and r["basis"] == "direct"
    assert [i["same_paper"] for i in r["supporting"]] == [True]
    assert r["inconclusive"] and all(not i["same_paper"] for i in r["inconclusive"]), "the 90.0 of another paper is not comparable"
    assert "single paper" in " ".join(r["flags"])
    flipped = claims.audit(idx, "Lyra-6 is worse than Lyra-5 on Kappa-Bench", FakeTracer())
    assert flipped["verdict"] == "SUPPORTED", "'is worse than' swaps the sides"
    assert claims.audit(idx, "Lyra-6 outperforms Lyra-5 on Kappa-Bench", FakeTracer())["verdict"] == "CONTRADICTED"


def test_claim_metric_direction_unknown_is_never_assessed():
    idx, _, n = build()
    r = claims.audit(idx, f"{n['model_a']} outperforms {n['model_b']} on {n['bench']}", FakeTracer())
    assert r["verdict"] == "INSUFFICIENT_EVIDENCE" and r["basis"] == "none"
    assert claims.direction("Q-Score") is None and claims.direction("top-1 error") == "lower" and claims.direction("mIoU") == "higher"


def test_claim_value_and_setting_forms():
    idx, _, n = build()
    v = claims.audit(idx, f"{n['model_a']} achieves 70.0 {n['metric']} on {n['bench']}", FakeTracer())
    assert v["verdict"] == "MIXED", "records report 70, 71 and 72 for the same model and metric"
    assert len(v["supporting"]) == 1 and len(v["contradicting"]) == 2
    s = claims.audit(idx, f"{n['model_b']} was trained with learning rate 0.01 on {n['bench']}", FakeTracer())
    assert s["verdict"] == "SUPPORTED" and all(i["stated"] == "0.01" for i in s["supporting"])
    assert claims.audit(idx, f"{n['model_b']} was trained with learning rate 0.1", FakeTracer())["verdict"] == "CONTRADICTED"


def test_claim_without_records_or_testable_form():
    idx, _, n = build()
    r = claims.audit(idx, f"{n['model_a']} outperforms Nobody-7 on {n['bench']}", FakeTracer())
    assert r["verdict"] == "NO_EVIDENCE" and "Nobody-7" in r["verdict_why"] and "not evidence against" in r["verdict_why"]
    for bad in ("transformers are great", f"{n['model_a']} outperforms all baselines on {n['bench']}"):
        with pytest.raises(ValueError):
            claims.audit(idx, bad, FakeTracer())


# ------------------------------------------------------------------------------------------------ F: transfer
P1, P2, P3 = "paper:arxiv:9901.00001", "paper:arxiv:9901.00002", "paper:arxiv:9901.00003"


def test_transfer_measures_a_change_only_toward_a_shifted_variant_and_lists_untested_benchmarks():
    extra = {"exp:t0": exp_record("exp:t0", P1, "Zeta-9", "Kappa-Bench V2", "Glyph Sorting", "Q-Score", 65.0),
             "exp:t1": exp_record("exp:t1", P1, "Zeta-9", "Lambda-Set", "Glyph Sorting", "Q-Score", 50.0),
             "exp:u0": exp_record("exp:u0", P2, "Orion-3", "Mu-Bench", "Glyph Sorting", "Q-Score", 40.0),
             "exp:u1": exp_record("exp:u1", P3, "Lyra-5", "Mu-Bench", "Glyph Sorting", "Q-Score", 41.0)}
    sections = [("c:1", P1, "Limitations", "Our method may not transfer under domain shift.", "S9.p1", True, "9901.00001")]
    idx, _, _ = build(extra, sections=sections)
    r = transfer.analyze(idx, "Zeta-9", FakeTracer(), source="Kappa-Bench")
    by_target = {t["target"]["benchmark"]: t for t in r["transfers"]}
    assert by_target["Kappa-Bench V2"]["change"] is not None and by_target["Kappa-Bench V2"]["shifted_variant"]
    assert by_target["Lambda-Set"]["change"] is None and "measures the datasets" in by_target["Lambda-Set"]["note"]
    assert [u["name"] for u in r["untested"]] == ["Mu-Bench"] and r["limitations"][0]["text"].startswith("Our method")
    assert transfer.analyze(idx, "Nobody-7", FakeTracer())["verdict"] == "NO_EVIDENCE"
    with pytest.raises(ValueError):
        transfer.analyze(idx, "Zeta-9", FakeTracer(), source="Omega-Set")


# ------------------------------------------------------------------------------------------------ D: planner
def test_planner_recommends_single_factor_ablations_baselines_and_reporting_from_records():
    extra = {"exp:m0": exp_record("exp:m0", P1, "Zeta-9 + Flux (Prism pretrained)", "Kappa-Bench", "Glyph Sorting", "Q-Score", 75.0)}
    idx, _, _ = build(extra)
    r = planner.plan(idx, "exp:m0", FakeTracer(), goal="beat Orion")
    kinds = [x["kind"] for x in r["recommendations"]]
    assert kinds == sorted(kinds, key=planner.PRIORITY.get), "ordered by kind"
    titles = {x["title"] for x in r["recommendations"]}
    assert {"Ablate Flux", "Ablate Prism pretrained"} <= titles, "the paper only removes both together (the Zeta-9 siblings)"
    assert r["reported_ablations"] and set(r["reported_ablations"][0]["removes"]) == {"Flux", "Prism pretrained"}
    assert "Compare against Orion-3" in titles and "Run several seeds and report the variance" in titles
    assert r["compute"] is None and r["goal"] == "beat Orion"
    b = planner.plan(idx, "exp:b0", FakeTracer())
    assert [x["kind"] for x in b["recommendations"]][:2] == ["reproduce", "reproduce"], "a recorded contradiction and a same-model score gap"
    assert any(x["title"].startswith("State the evaluation split") for x in b["recommendations"])
    with pytest.raises(KeyError):
        planner.plan(idx, "exp:missing", FakeTracer())


# ------------------------------------------------------------------------------------------------ G: review coverage
def test_review_finds_unnamed_benchmarks_and_families_and_sends_only_entity_names():
    extra = {"exp:r0": exp_record("exp:r0", P3, "Orion-3", "Kappa-Bench", "Glyph Sorting", "Q-Score", 62.0),
             "exp:r1": exp_record("exp:r1", P2, "Orion-3", "Mu-Bench", "Glyph Sorting", "Q-Score", 40.0),
             "exp:r2": exp_record("exp:r2", P3, "Lyra-5", "Mu-Bench", "Glyph Sorting", "Q-Score", 41.0)}
    idx, _, _ = build(extra)
    sent = []

    def search(tr, query):
        sent.append(query)
        return [{"title": "Zeta-9 for Glyph Sorting", "url": "https://arxiv.org/abs/9901.00001", "arxiv_id": "9901.00001", "provider": "fake"},
                {"title": "Elsewhere", "url": "https://doi.org/10.1/x", "provider": "fake"}]
    draft = "In this confidential draft we review Glyph Sorting with Zeta-9 on Kappa-Bench (arXiv 9901.00002)."
    r = review.audit(idx, draft, FakeTracer(), release="test", search=search)
    assert r["recognised"]["datasets"] == ["Kappa-Bench"] and r["recognised"]["cited_in_corpus"] == ["paper:arxiv:9901.00002"]
    assert [b["name"] for b in r["missing_benchmarks"]] == ["Mu-Bench"] and r["missing_models"][0]["family"] == "orion"
    assert sent == ["Glyph Sorting Kappa-Bench"] and "confidential" not in sent[0], "only recognised entity names leave the system"
    assert [d["in_corpus"] for d in r["discovered"]] == [True, False] and r["scope"]["providers"] == ["fake"]
    assert review.audit(idx, draft, FakeTracer())["discovered"] == [], "no runtime search unless asked"
    with pytest.raises(ValueError):
        review.audit(idx, "A sentence that names nothing the corpus knows about.", FakeTracer())


# ------------------------------------------------------------------------------------------------ H: benchmark
def test_benchmark_classes_and_ranking_only_among_comparable():
    extra = {"exp:c0": exp_record("exp:c0", "paper:arxiv:9901.00003", "Lyra-5", "Kappa-Bench", "Glyph Sorting", "Q-Score", 99.0,
                                  evaluation__dataset_split="validation", evaluation__evaluation_protocol="single crop"),
             "exp:c1": exp_record("exp:c1", "paper:arxiv:9901.00003", "Lyra-6", "Kappa-Bench", "Glyph Sorting", "Q-Score", 98.0,
                                  evaluation__dataset_split="test")}
    idx, _, n = build(extra)
    b, m = entity_id(E.BENCHMARK, n["bench"]), entity_id(E.METRIC, n["metric"])
    assert bench.catalog(idx)[0]["id"] == b and bench.catalog(idx)[0]["metrics"][0]["id"] == m
    tr = FakeTracer()
    r = bench.explore(idx, b, m, tr)
    cls = {row["experiment"]: row["class"] for row in r["rows"]}
    assert {cls[f"exp:a{i}"] for i in range(3)} == {"COMPARABLE"}, "test split + single crop is the most common stated reference"
    assert cls["exp:c0"] == "INCOMPATIBLE" and cls["exp:c1"] == "CONDITIONALLY_COMPARABLE" and cls["exp:b0"] == "UNKNOWN"
    ranked = [row for row in r["rows"] if "rank" in row]
    assert [row["experiment"] for row in ranked] == ["exp:a2", "exp:a1", "exp:a0"], "the higher incompatible 99.0 is never ranked"
    assert all("confidence_interval" not in row for row in r["rows"])
    assert [s.name for s in tr.rows] == ["experiment_retrieval", "field_facts", "comparability"]
    ref = next(row["result"] for row in r["rows"] if row["experiment"] == "exp:c0")
    r2 = bench.explore(idx, b, m, FakeTracer(), reference=ref)
    assert {row["experiment"]: row["class"] for row in r2["rows"]}["exp:c0"] == "COMPARABLE"
    assert {row["experiment"]: row["class"] for row in r2["rows"]}["exp:a0"] == "INCOMPATIBLE"


def test_benchmark_without_stated_conditions_ranks_nothing():
    idx, _, n = build()
    fam = model_family(n["model_b"])[0]  # only paper B's results, which state no split or protocol
    r = bench.explore(idx, entity_id(E.BENCHMARK, n["bench"]), entity_id(E.METRIC, n["metric"]), FakeTracer(), family=fam)
    assert r["summary"]["results"] == 2 and r["summary"]["ranked"] == 0 and {row["class"] for row in r["rows"]} == {"UNKNOWN"}
    assert r["reference"]["kind"] == "none"
