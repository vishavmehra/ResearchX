"""Artifact config parsing and paper/repository relation rules (no network)."""

from services.datasets.schemas import (
    ArtifactRelation,
    Experiment,
    ExperimentType,
    ExtractionMethod,
    FieldStatus,
    FieldValue,
    MergeEvidence,
    Paper,
    Provenance,
)
from services.datasets.store import Run
from services.retrieval.sources import github_artifacts as gh

PAPER = Paper(paper_id="paper:arxiv:9999.00002", study_id="study:arxiv:9999.00002", title="T", normalized_title="t", arxiv_id="9999.00002",
              source_record_ids=["arxiv:9999.00002v1"], merge_evidence=[MergeEvidence(source_record_id="arxiv:9999.00002v1", matched_on="self", value="x")],
              provenance=Provenance(source_name="RESEARCHX", processing_version="test"))
EVIDENCE = {"ev:paper-lr": {"evidence_id": "ev:paper-lr", "location_kind": "PAPER_SENTENCE"}}


def experiment(lr: float) -> Experiment:
    e = Experiment(experiment_id="exp:1", study_id=PAPER.study_id, paper_id=PAPER.paper_id, source_record_ids=PAPER.source_record_ids,
                   experiment_type=ExperimentType.MAIN, extraction_method=ExtractionMethod.LLM, extraction_run_id="r",
                   provenance=Provenance(source_name="RESEARCHX", processing_version="test"))
    e.training.learning_rate = FieldValue(status=FieldStatus.KNOWN, raw_value=str(lr), value=lr, evidence_ids=["ev:paper-lr"])
    return e


def relations(repo: str, files: dict[str, bytes], paper_lr: float, complete: bool = True) -> list:
    snap = gh.Snapshot(repo, "a" * 40, files=files, license_hint="MIT", complete=complete)
    cand = gh.Candidate(*repo.split("/"), "PAPER_TEXT_URL", True, "S1.p1")
    cached = {p: (f"data/raw/github/{p}.gz", "0" * 64) for p in files}
    _, _, rels = gh.link(PAPER, [experiment(paper_lr)], cand, snap, cached, EVIDENCE, Run("test"), None)
    return [r for r in rels if r.field_name == "training.learning_rate"]


def test_parse_values_yaml_stack_and_cli():
    yaml = "optimizer:\n  lr: 0.001\n  weight_decay: 1e-4\nlora:\n  r: 8\n"
    got = {(f, v, n) for f, v, n, _ in gh.parse_values("configs/train.yaml", yaml)}
    assert got == {("training.learning_rate", "0.001", 2), ("training.weight_decay", "1e-4", 3), ("training.peft_rank", "8", 5)}
    sh = "python train.py --lr 3e-4 --batch_size 32 --output_dir $OUT\n"
    assert {(f, v) for f, v, _, _ in gh.parse_values("run.sh", sh)} == {("training.learning_rate", "3e-4"), ("training.batch_size", "32")}


def test_safe_cache_path_handles_windows_invalid_names():
    assert gh.safe_cache_path("annotated_sample/.C.mmp_|_Proc.json") == "annotated_sample/.C.mmp_%7C_Proc.json"
    assert gh.safe_cache_path("a/CON.txt") == "a/_CON.txt" and gh.safe_cache_path("dir./x?.yaml") == "dir._/x%3F.yaml"
    assert gh.safe_cache_path("50%/cfg.yaml") == "50%25/cfg.yaml"  # '%' itself encoded, so the mapping stays unambiguous


def test_pinned_snapshot_replays_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(gh, "CACHE", tmp_path)
    monkeypatch.setattr(gh, "rel", lambda p: str(p))
    snap = gh.Snapshot("lab/method", "b" * 40, files={"configs/a|b.yaml": b"lr: 0.01\n", "README.md": b"x"}, license_hint="MIT", complete=True)
    gh.store_files(snap)
    again = gh.load_snapshot("lab", "method")
    assert (again.commit, again.files, again.complete, again.license_hint, again.retrieved_at) == \
        (snap.commit, snap.files, True, "MIT", snap.retrieved_at)
    assert gh.load_snapshot("lab", "other") is None


def test_select_reports_incomplete_when_configs_are_cut():
    listing = [(f"configs/exp{i}.yaml", 0) for i in range(gh.MAX_FILES + 5)] + [("README.md", 0)]
    picked, complete = gh._select(listing)
    assert not complete and "README.md" in picked and len(picked) == gh.MAX_FILES
    assert gh._select([("configs/a.yaml", 0), ("README.md", 0)])[1]


def test_relation_rules():
    cfg = {"configs/train.yaml": b"lr: 0.01\n"}
    (r,) = relations("lab/method", cfg, 0.01)
    assert r.relation is ArtifactRelation.CONFIRMED_BY_BOTH and "1 of 1 file" in r.rationale
    (r,) = relations("lab/method", cfg, 0.001)
    assert r.relation is ArtifactRelation.CONFLICTING  # complete read, one configured value, differs
    (r,) = relations("lab/method", cfg, 0.001, complete=False)
    assert r.relation is ArtifactRelation.RELATED_BUT_UNCONFIRMED  # sampled configs never prove a conflict
    (r,) = relations("lab/method", {"README.md": b"python train.py --lr 0.01\n"}, 0.001)
    assert r.relation is ArtifactRelation.RELATED_BUT_UNCONFIRMED  # README usage examples never conflict
    for paper_lr in (0.01, 0.001):  # multi-model library repositories: never confirmed, never conflicting
        (r,) = relations("tensorflow/tpu", cfg, paper_lr)
        assert r.relation is ArtifactRelation.RELATED_BUT_UNCONFIRMED
