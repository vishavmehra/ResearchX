"""Source discovery registry invariants (deepening program §12-13, §32)."""

import json

from services.datasets.store import DATA, load_yaml

REG = load_yaml(DATA / "manifests" / "source_discovery.yaml")
PROBES = json.loads((DATA / "reports" / "source_probes.json").read_text(encoding="utf-8"))
DIMS = {"discovery", "fulltext", "experiment_depth", "configuration_depth", "evidence", "artifact", "study_link", "comparison",
        "retrieval", "temporal", "coverage", "license_quality", "api_quality", "maintenance", "duplication_risk"}
PERMISSION_TOKENS = {"YES", "NO", "CONDITIONAL", "UNVERIFIED", "NOT_APPLICABLE"}
REQUIRED_SOURCES = {
    "OPENALEX", "SEMANTIC_SCHOLAR", "ARXIV", "CROSSREF", "DATACITE", "OPENCITATIONS", "OPENREVIEW", "ACL_ANTHOLOGY", "CORE",
    "UNPAYWALL", "ZENODO", "FIGSHARE", "GITHUB", "HUGGINGFACE_HUB", "OPENML", "MLCOMMONS_MLPERF", "HELM", "OPEN_LLM_LEADERBOARD",
    "SCIREX", "SCIENCEIE", "SCIRIFF", "SCIREPEVAL", "QASPER", "SCIFACT", "RESEARCHQA", "SCICITE", "BRIGHT", "BEIR", "MTEB",
    "BROWSECOMP", "BROWSECOMP_PLUS", "DEEPRESEARCHGYM", "DEEPRESEARCH_BENCH", "ASSISTANTBENCH", "GAIA", "WEBARENA_VERIFIED",
    "DOCLAYNET", "PUBTABLES_1M", "TABLEBANK", "PUBLAYNET", "DOCBANK", "MLCOMMONS_SCIENCE_BENCHMARKS",
}


def test_every_required_source_is_assessed_once():
    ids = [s["source_id"] for s in REG["sources"]]
    assert len(ids) == len(set(ids))
    assert REQUIRED_SOURCES <= set(ids)


def test_assessment_is_structured_not_a_single_score():
    for s in REG["sources"]:
        assert set(s["assessment"]) == DIMS, s["source_id"]
        assert set(s["assessment"].values()) <= {"HIGH", "MEDIUM", "LOW", "NONE", "UNKNOWN"}, s["source_id"]


def test_permissions_are_explicit_and_unverified_never_means_yes():
    for s in REG["sources"]:
        for k in ("training_allowed", "evaluation_allowed", "redistribution_allowed", "attribution_required"):
            assert isinstance(s[k], str) and s[k].split(" ")[0] in PERMISSION_TOKENS, (s["source_id"], k)
        if "not verified" in s["license"].lower() or s["license"].lower().startswith("none"):
            assert not s["training_allowed"].startswith("YES"), s["source_id"]


def test_api_claims_cite_existing_probes():
    for s in REG["sources"]:
        for ref in s["verification"]:
            if ref.startswith("probe:"):
                assert ref.split(":", 1)[1] in PROBES, (s["source_id"], ref)
        if "API_TESTED" in s["status"]:
            assert any(r.startswith("probe:") for r in s["verification"]), s["source_id"]


def test_use_now_sources_have_verified_terms():
    for s in REG["sources"]:
        if s["recommendation"] == "USE_NOW":
            assert not s["training_allowed"].startswith("UNVERIFIED") or s["source_id"] == "SEMANTIC_SCHOLAR", s["source_id"]
            assert s["map_sections"] and all(m in REG["map_sections"] for m in s["map_sections"])
