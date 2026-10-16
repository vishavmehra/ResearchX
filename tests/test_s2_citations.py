"""Semantic Scholar reference ids and intra-corpus citation edges (no network)."""

from services.datasets.schemas import MergeEvidence, Paper, Provenance, SourceRecord
from services.retrieval.corpus import build_citations
from services.retrieval.sources.semantic_scholar import reference_ids


def paper(arxiv: str, srids: list[str], doi: str | None = None) -> Paper:
    return Paper(paper_id=f"paper:arxiv:{arxiv}", study_id=f"study:arxiv:{arxiv}", title="T", normalized_title="t", arxiv_id=arxiv, doi=doi,
                 source_record_ids=srids, merge_evidence=[MergeEvidence(source_record_id=srids[0], matched_on="self", value="x")],
                 provenance=Provenance(source_name="RESEARCHX", processing_version="t"))


def record(srid: str, source: str, refs: list[str]) -> SourceRecord:
    return SourceRecord(source_record_id=srid, source_name=source, source_identifier=srid, references=refs,
                        provenance=Provenance(source_name=source, processing_version="t"))


def test_reference_ids_keep_arxiv_and_doi_only():
    s2 = {"references": [{"externalIds": {"ArXiv": "1810.04805", "DOI": "10.18653/V1/N19-1423"}}, {"externalIds": {"MAG": "1"}}, None]}
    assert reference_ids(s2) == ["arxiv:1810.04805", "doi:10.18653/v1/n19-1423"]


def test_build_citations_uses_s2_refs_and_keeps_openalex_provenance():
    a, b, c = paper("2106.09685", ["s2:a", "openalex:W1"]), paper("1810.04805", ["s2:b"], doi="10.18653/v1/n19-1423"), paper("1706.03762", ["s2:c"])
    recs = [record("s2:a", "SEMANTIC_SCHOLAR", ["doi:10.18653/v1/n19-1423", "arxiv:1706.03762"]),
            record("openalex:W1", "OPENALEX", ["arxiv:1706.03762"])]
    edges = {(e.citing_paper_id, e.cited_paper_id): e.source_record_id for e in build_citations([a, b, c], recs)}
    assert edges == {(a.paper_id, b.paper_id): "s2:a", (a.paper_id, c.paper_id): "openalex:W1"}
