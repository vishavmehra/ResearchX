"""Live probes of candidate data sources ("Do not claim an API works unless it was actually tested").

Each probe is one unauthenticated HTTP GET. The result (status, latency, rate-limit headers and a
compact observation: result counts, declared licence, gating, freshness) is merged into
data/reports/source_probes.json. A probe that already succeeded is skipped unless --refresh.
GitHub's unauthenticated API allows 60 requests/hour: GitHub probes stop when fewer than 3 remain
and are recorded as DEFERRED_RATE_LIMIT for a later run. No credentials, no personal e-mail.

Usage: uv run python scripts/research/probe_sources.py [--refresh] [--only PREFIX]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.datasets.store import DATA, atomic_write_bytes, now_utc, utf8_stdio  # noqa: E402

OUT = DATA / "reports" / "source_probes.json"
UA = "ResearchX-dataset-research/0.2 (local research prototype; contact via project maintainers)"
GH = "https://api.github.com/repos/"
HF = "https://huggingface.co/api/datasets/"

# (probe_id, source_id, kind, url). Order = priority (GitHub budget is spent top-down).
PROBES: list[tuple[str, str, str, str]] = [
    # scholarly metadata / discovery
    ("openalex_work_doi", "openalex", "json", "https://api.openalex.org/works/doi:10.48550/arXiv.2106.09685"),
    ("openalex_filter_list", "openalex", "json", "https://api.openalex.org/works?filter=primary_topic.subfield.id:1702,publication_year:2025&per_page=1"),
    ("openalex_search", "openalex", "json", "https://api.openalex.org/works?search=low-rank%20adaptation&per_page=1"),
    ("openalex_oa_lookup", "openalex", "json", "https://api.openalex.org/works/doi:10.18653/v1/N19-1423?select=id,open_access,best_oa_location,primary_location"),
    ("s2_paper", "semantic_scholar", "json", "https://api.semanticscholar.org/graph/v1/paper/arXiv:2106.09685?fields=title,externalIds,year,venue,publicationVenue,openAccessPdf,citationCount,referenceCount"),
    ("s2_datasets_release", "semantic_scholar", "json", "https://api.semanticscholar.org/datasets/v1/release/latest"),
    ("arxiv_api", "arxiv", "xml", "https://export.arxiv.org/api/query?id_list=2106.09685"),
    ("arxiv_oai_identify_new", "arxiv_oai_pmh", "xml", "https://oaipmh.arxiv.org/oai?verb=Identify"),
    ("arxiv_oai_listrecords_new", "arxiv_oai_pmh", "xml", "https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv&set=cs&from=2026-09-28"),
    ("arxiv_oai_identify_old", "arxiv_oai_pmh", "xml", "https://export.arxiv.org/oai2?verb=Identify"),
    ("arxiv_html", "arxiv_html", "head", "https://arxiv.org/html/2106.09685v2"),
    ("crossref_doi", "crossref", "json", "https://api.crossref.org/works/10.18653/v1/N19-1423"),
    ("crossref_query", "crossref", "json", "https://api.crossref.org/works?query.bibliographic=LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models&rows=3&select=DOI,title,type,relation,published"),
    ("crossref_has_preprint", "crossref", "json", "https://api.crossref.org/works?filter=relation.type:has-preprint&rows=1&select=DOI,title,relation"),
    ("datacite_arxiv_doi", "datacite", "json", "https://api.datacite.org/dois/10.48550/arxiv.2106.09685"),
    ("datacite_software", "datacite", "json", "https://api.datacite.org/dois?resource-type-id=software&query=machine%20learning&page[size]=1"),
    ("opencitations_count", "opencitations", "json", "https://api.opencitations.net/index/v2/citation-count/doi:10.18653/v1/N19-1423"),
    ("opencitations_meta", "opencitations", "json", "https://api.opencitations.net/meta/v1/metadata/doi:10.18653/v1/N19-1423"),
    ("openreview_venue", "openreview", "json", "https://api2.openreview.net/notes?content.venueid=ICLR.cc/2024/Conference&limit=2"),
    ("openreview_search", "openreview", "json", "https://api2.openreview.net/notes/search?term=low-rank%20adaptation&limit=2"),
    ("acl_bib", "acl_anthology", "text", "https://aclanthology.org/N19-1423.bib"),
    ("acl_bulk_bib", "acl_anthology", "head", "https://aclanthology.org/anthology+abstracts.bib.gz"),
    ("dblp_search", "dblp", "json", "https://dblp.org/search/publ/api?q=LoRA%20Low-Rank%20Adaptation&format=json&h=3"),
    ("core_search", "core", "json", "https://api.core.ac.uk/v3/search/works?q=low-rank%20adaptation&limit=1"),
    ("unpaywall_doi", "unpaywall", "json", "https://api.unpaywall.org/v2/10.18653/v1/N19-1423?email=test@example.com"),
    ("openaire_graph", "openaire_graph", "json", "https://api.openaire.eu/graph/v1/researchProducts?search=low-rank%20adaptation&pageSize=1"),
    ("orkg_papers", "orkg", "json", "https://orkg.org/api/papers?size=1"),
    # artifacts
    ("github_api_repo", "github", "github_repo", GH + "microsoft/LoRA"),
    ("github_raw_readme", "github", "text", "https://raw.githubusercontent.com/microsoft/LoRA/main/README.md"),
    ("hf_papers", "huggingface_hub", "json", "https://huggingface.co/api/papers/2106.09685"),
    ("hf_arxiv_repos", "huggingface_hub", "json", "https://huggingface.co/api/arxiv/2106.09685/repos"),
    ("hf_models_by_arxiv", "huggingface_hub", "json", "https://huggingface.co/api/models?filter=arxiv:2106.09685&limit=5"),
    ("hf_model_index", "huggingface_hub", "hf_model", "https://huggingface.co/api/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
    ("hf_datasets_by_arxiv", "huggingface_hub", "json", "https://huggingface.co/api/datasets?filter=arxiv:1910.10683&limit=5"),
    ("zenodo_arxiv", "zenodo", "json", "https://zenodo.org/api/records?q=%22arXiv%3A2106.09685%22&size=2"),
    ("zenodo_software", "zenodo", "json", "https://zenodo.org/api/records?q=machine%20learning&type=software&size=1"),
    ("figshare_list", "figshare", "json", "https://api.figshare.com/v2/articles?page_size=1"),
    ("swh_origin", "software_heritage", "json", "https://archive.softwareheritage.org/api/1/origin/https://github.com/microsoft/LoRA/get/"),
    # ML experiment databases
    ("openml_task", "openml", "json", "https://www.openml.org/api/v1/json/task/31"),
    ("openml_data", "openml", "json", "https://www.openml.org/api/v1/json/data/31"),
    ("openml_run_list", "openml", "json", "https://www.openml.org/api/v1/json/run/list/task/31/limit/3"),
    ("openml_eval_list", "openml", "json", "https://www.openml.org/api/v1/json/evaluation/list/function/area_under_roc_curve/task/31/limit/3"),
    # benchmark / result repositories (GitHub)
    ("mlperf_training", "mlcommons_mlperf", "github_repo", GH + "mlcommons/training"),
    ("mlperf_training_results_v6_1", "mlcommons_mlperf", "github_repo", GH + "mlcommons/training_results_v6.1"),
    ("mlperf_training_results_v6_0", "mlcommons_mlperf", "github_repo", GH + "mlcommons/training_results_v6.0"),
    ("mlperf_training_results_v5_1", "mlcommons_mlperf", "github_repo", GH + "mlcommons/training_results_v5.1"),
    ("mlperf_inference_results_v6_1", "mlcommons_mlperf", "github_repo", GH + "mlcommons/inference_results_v6.1"),
    ("mlperf_inference_results_v6_0", "mlcommons_mlperf", "github_repo", GH + "mlcommons/inference_results_v6.0"),
    ("mlcommons_science", "mlcommons_science_benchmarks", "github_repo", GH + "mlcommons-science/benchmark"),
    ("helm_repo", "helm", "github_repo", GH + "stanford-crfm/helm"),
    ("helm_bucket", "helm", "xml", "https://storage.googleapis.com/crfm-helm-public/?prefix=lite/benchmark_output/releases/&delimiter=/"),
    ("ollb_contents", "open_llm_leaderboard", "hf_dataset", HF + "open-llm-leaderboard/contents"),
    ("ollb_results", "open_llm_leaderboard", "hf_dataset", HF + "open-llm-leaderboard/results"),
    ("epoch_all_models_csv", "epoch_ai", "head", "https://epoch.ai/data/all_ai_models.csv"),
    ("epoch_notable_models_csv", "epoch_ai", "head", "https://epoch.ai/data/notable_ai_models.csv"),
    # document layout / tables
    ("doclaynet_repo", "doclaynet", "github_repo", GH + "DS4SD/DocLayNet"),
    ("doclaynet_hf", "doclaynet", "hf_dataset", HF + "ds4sd/DocLayNet"),
    ("pubtables_repo", "pubtables_1m", "github_repo", GH + "microsoft/table-transformer"),
    ("pubtables_hf", "pubtables_1m", "hf_dataset", HF + "bsmock/pubtables-1m"),
    ("tablebank_repo", "tablebank", "github_repo", GH + "doc-analysis/TableBank"),
    ("publaynet_repo", "publaynet", "github_repo", GH + "ibm-aur-nlp/PubLayNet"),
    ("docbank_repo", "docbank", "github_repo", GH + "doc-analysis/DocBank"),
    ("docling_repo", "docling_tool", "github_repo", GH + "docling-project/docling"),
    ("grobid_repo", "grobid_tool", "github_repo", GH + "kermitt2/grobid"),
    # scientific NLP / evidence
    ("scirex_repo", "scirex", "github_repo", GH + "allenai/SciREX"),
    ("sciriff_hf", "sciriff", "hf_dataset", HF + "allenai/SciRIFF"),
    ("scirepeval_hf", "scirepeval", "hf_dataset", HF + "allenai/scirepeval"),
    ("qasper_hf", "qasper", "hf_dataset", HF + "allenai/qasper"),
    ("scifact_repo", "scifact", "github_repo", GH + "allenai/scifact"),
    ("scicite_hf", "scicite", "hf_dataset", HF + "allenai/scicite"),
    ("scienceie_site", "scienceie", "text", "https://scienceie.github.io/"),
    ("axcell_repo", "axcell", "github_repo", GH + "paperswithcode/axcell"),
    ("scitab_repo", "scitab", "github_repo", GH + "XinyuanLu00/SciTab"),
    ("unarxive_repo", "unarxive", "github_repo", GH + "IllDepence/unarXive"),
    ("pes2o_hf", "s2orc_pes2o", "hf_dataset", HF + "allenai/peS2o"),
    ("acl_ocl_hf", "acl_ocl", "hf_dataset", HF + "WINGNUS/ACL-OCL"),
    ("peerread_repo", "peerread", "github_repo", GH + "allenai/PeerRead"),
    ("sciqa_hf", "orkg", "hf_dataset", HF + "orkg/SciQA"),
    # retrieval / deep research / agents
    ("bright_hf", "bright", "hf_dataset", HF + "xlangai/BRIGHT"),
    ("beir_repo", "beir", "github_repo", GH + "beir-cellar/beir"),
    ("beir_scifact_hf", "beir", "hf_dataset", HF + "BeIR/scifact"),
    ("mteb_repo", "mteb", "github_repo", GH + "embeddings-benchmark/mteb"),
    ("mteb_results_repo", "mteb", "github_repo", GH + "embeddings-benchmark/results"),
    ("browsecomp_repo", "browsecomp", "github_repo", GH + "openai/simple-evals"),
    ("browsecomp_csv", "browsecomp", "head", "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"),
    ("browsecomp_plus_repo", "browsecomp_plus", "github_repo", GH + "texttron/BrowseComp-Plus"),
    ("browsecomp_plus_hf", "browsecomp_plus", "hf_dataset", HF + "Tevatron/browsecomp-plus"),
    ("deepresearch_bench_repo", "deepresearch_bench", "github_repo", GH + "Ayanami0730/deep_research_bench"),
    ("assistantbench_hf", "assistantbench", "hf_dataset", HF + "AssistantBench/AssistantBench"),
    ("gaia_hf", "gaia", "hf_dataset", HF + "gaia-benchmark/GAIA"),
    ("litsearch_hf", "litsearch", "hf_dataset", HF + "princeton-nlp/LitSearch"),
    ("paperbench_repo", "paperbench", "github_repo", GH + "openai/preparedness"),
    ("corebench_repo", "core_bench", "github_repo", GH + "siegelz/core-bench"),
    ("mlebench_repo", "mle_bench", "github_repo", GH + "openai/mle-bench"),
    ("mlagentbench_repo", "mlagentbench", "github_repo", GH + "snap-stanford/MLAgentBench"),
    ("scienceagentbench_repo", "scienceagentbench", "github_repo", GH + "OSU-NLP-Group/ScienceAgentBench"),
    ("discoverybench_repo", "discoverybench", "github_repo", GH + "allenai/discoverybench"),
    # catalogs
    ("pwc_site", "papers_with_code_archive", "head", "https://paperswithcode.com/"),
    ("pwc_eval_tables_hf", "papers_with_code_archive", "hf_dataset", HF + "pwc-archive/evaluation-tables"),
    ("pwc_links_hf", "papers_with_code_archive", "hf_dataset", HF + "pwc-archive/links-between-paper-and-code"),
    ("hf_datasets_search", "huggingface_datasets", "json", "https://huggingface.co/api/datasets?search=scientific&limit=3"),
]

# Data licences / usage terms live in READMEs and dataset cards, not in the repo's LICENSE file.
RAW = "https://raw.githubusercontent.com/"
HFR = "https://huggingface.co/datasets/"
DOCS: list[tuple[str, str, str, str]] = [
    ("doc_doclaynet", "doclaynet", "doc", RAW + "DS4SD/DocLayNet/main/README.md"),
    ("doc_doclaynet_card", "doclaynet", "doc", HFR + "ds4sd/DocLayNet/raw/main/README.md"),
    ("doc_publaynet", "publaynet", "doc", RAW + "ibm-aur-nlp/PubLayNet/master/README.md"),
    ("doc_tablebank", "tablebank", "doc", RAW + "doc-analysis/TableBank/master/README.md"),
    ("doc_docbank", "docbank", "doc", RAW + "doc-analysis/DocBank/master/README.md"),
    ("doc_pubtables", "pubtables_1m", "doc", RAW + "microsoft/table-transformer/main/README.md"),
    ("doc_scifact", "scifact", "doc", RAW + "allenai/scifact/master/README.md"),
    ("doc_scicite", "scicite", "doc", RAW + "allenai/scicite/master/README.md"),
    ("doc_scirepeval", "scirepeval", "doc", RAW + "allenai/scirepeval/main/README.md"),
    ("doc_scirepeval_card", "scirepeval", "doc", HFR + "allenai/scirepeval/raw/main/README.md"),
    ("doc_scirex", "scirex", "doc", RAW + "allenai/SciREX/master/README.md"),
    ("doc_sciriff", "sciriff", "doc", RAW + "allenai/SciRIFF/main/README.md"),
    ("doc_qasper_card", "qasper", "doc", HFR + "allenai/qasper/raw/main/README.md"),
    ("doc_unarxive", "unarxive", "doc", RAW + "IllDepence/unarXive/master/README.md"),
    ("doc_axcell", "axcell", "doc", RAW + "paperswithcode/axcell/master/README.md"),
    ("doc_scitab", "scitab", "doc", RAW + "XinyuanLu00/SciTab/main/README.md"),
    ("doc_peerread", "peerread", "doc", RAW + "allenai/PeerRead/master/README.md"),
    ("doc_beir", "beir", "doc", RAW + "beir-cellar/beir/main/README.md"),
    ("doc_bright", "bright", "doc", RAW + "xlang-ai/BRIGHT/main/README.md"),
    ("doc_browsecomp", "browsecomp", "doc", RAW + "openai/simple-evals/main/README.md"),
    ("doc_browsecomp_plus", "browsecomp_plus", "doc", RAW + "texttron/BrowseComp-Plus/main/README.md"),
    ("doc_deepresearch_bench", "deepresearch_bench", "doc", RAW + "Ayanami0730/deep_research_bench/main/README.md"),
    ("doc_assistantbench_card", "assistantbench", "doc", HFR + "AssistantBench/AssistantBench/raw/main/README.md"),
    ("doc_gaia_card", "gaia", "doc", HFR + "gaia-benchmark/GAIA/raw/main/README.md"),
    ("doc_litsearch_card", "litsearch", "doc", HFR + "princeton-nlp/LitSearch/raw/main/README.md"),
    ("doc_ollb_contents_card", "open_llm_leaderboard", "doc", HFR + "open-llm-leaderboard/contents/raw/main/README.md"),
    ("doc_mlcommons_science", "mlcommons_science_benchmarks", "doc", RAW + "mlcommons-science/benchmark/main/README.md"),
    ("doc_mlperf_training_results_v6_0", "mlcommons_mlperf", "doc", RAW + "mlcommons/training_results_v6.0/main/README.md"),
    ("doc_helm", "helm", "doc", RAW + "stanford-crfm/helm/main/README.md"),
    ("doc_mlebench", "mle_bench", "doc", RAW + "openai/mle-bench/main/README.md"),
    ("doc_discoverybench", "discoverybench", "doc", RAW + "allenai/discoverybench/main/README.md"),
    ("doc_corebench", "core_bench", "doc", RAW + "siegelz/core-bench/main/README.md"),
    ("doc_mteb_results", "mteb", "doc", RAW + "embeddings-benchmark/results/main/README.md"),
    ("doc_scienceagentbench", "scienceagentbench", "doc", RAW + "OSU-NLP-Group/ScienceAgentBench/main/README.md"),
    ("doc_pes2o_card", "s2orc_pes2o", "doc", HFR + "allenai/peS2o/raw/main/README.md"),
    ("doc_acl_ocl_card", "acl_ocl", "doc", HFR + "WINGNUS/ACL-OCL/raw/main/README.md"),
]
PROBES += DOCS
GHS = "https://api.github.com/search/repositories?per_page=3&q="
PROBES += [  # second-round discoveries (web research 2026-09-30)
    ("x_metalead_repo", "metalead", "github_repo", GH + "RoelTim/metalead"),
    ("x_eee_repo", "every_eval_ever", "github_repo", GH + "evaleval/every_eval_ever"),
    ("x_webarena_verified_repo", "webarena_verified", "github_repo", GH + "ServiceNow/webarena-verified"),
    ("x_researchqa_hf", "researchqa", "hf_dataset", HF + "realliyifei/ResearchQA"),
    ("x_reclaim_hf", "reclaim", "hf_dataset", HF + "Mithilss/reclaim"),
    ("x_common_pile_arxiv_hf", "common_pile_arxiv", "hf_dataset", HF + "common-pile/arxiv_papers"),
    ("x_search_hyperpie", "hyperpie", "json", GHS + "hyperpie+hyperparameter"),
    ("x_search_scier", "scier", "json", GHS + "SciER+entity+relation+extraction"),
    ("x_search_arxivdigestables", "arxivdigestables", "json", GHS + "arxivdigestables"),
    ("x_openreview_invitation", "openreview", "json", "https://api2.openreview.net/notes?invitation=ICLR.cc/2024/Conference/-/Submission&limit=1"),
    ("x_openreview_venueid", "openreview", "json", "https://api2.openreview.net/notes?venueid=ICLR.cc/2024/Conference&limit=1"),
    ("x_dblp_dump", "dblp", "head", "https://dblp.org/xml/dblp.xml.gz"),
    ("x_arxiv_oai_listsets", "arxiv_oai_pmh", "xml", "https://oaipmh.arxiv.org/oai?verb=ListSets"),
    ("x_openml_data_retry", "openml", "json", "https://www.openml.org/api/v1/json/data/31"),
]
LICENSE_LINE = re.compile(
    r"licen[cs]|terms of|cc[- ]?by|cc0|cdla|odc|apache|mit license|non-?commercial|research (purposes|use)|"
    r"do not (use|train|redistribute)|not be used|must not|canary|evaluation[- ]only|contaminat|gated|commercial|copyright|"
    r"maintenance|deprecated|no longer|archived",
    re.I,
)


def _first_list(obj: Any) -> tuple[str | None, list | None]:
    if isinstance(obj, list):
        return "", obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                return k, v
        for k, v in obj.items():
            if isinstance(v, dict):
                kk, vv = _first_list(v)
                if vv is not None:
                    return f"{k}.{kk}", vv
    return None, None


def observe_json(data: Any) -> dict:
    obs: dict[str, Any] = {"type": type(data).__name__}
    if isinstance(data, dict):
        obs["keys"] = list(data)[:20]
        for k in ("meta", "message", "hits", "info"):
            if isinstance(data.get(k), dict):
                sub = {kk: vv for kk, vv in data[k].items() if isinstance(vv, (int, str)) and len(str(vv)) < 80}
                obs[k] = dict(list(sub.items())[:10])
    key, lst = _first_list(data)
    if lst is not None:
        obs["first_list"] = {"key": key, "len": len(lst), "item_keys": list(lst[0])[:25] if lst and isinstance(lst[0], dict) else None}
    return obs


def observe_github(d: dict) -> dict:
    lic = d.get("license") or {}
    return {
        "full_name": d.get("full_name"), "license_spdx": lic.get("spdx_id"), "license_name": lic.get("name"),
        "archived": d.get("archived"), "pushed_at": d.get("pushed_at"), "stars": d.get("stargazers_count"),
        "size_kb": d.get("size"), "default_branch": d.get("default_branch"), "description": (d.get("description") or "")[:200],
    }


def observe_hf(d: dict) -> dict:
    card = d.get("cardData") or {}
    tags = d.get("tags") or []
    mi = card.get("model-index") or d.get("model-index")
    return {
        "id": d.get("id"), "card_license": card.get("license"), "license_tags": [t for t in tags if t.startswith("license:")],
        "gated": d.get("gated"), "private": d.get("private"), "disabled": d.get("disabled"), "last_modified": d.get("lastModified"),
        "downloads": d.get("downloads"), "task_categories": card.get("task_categories"), "size_categories": card.get("size_categories"),
        "arxiv_tags": [t for t in tags if t.startswith("arxiv:")][:5], "model_index_results": sum(len(m.get("results", [])) for m in mi) if isinstance(mi, list) else None,
        "configs": [c.get("config_name") for c in card.get("configs", [])][:10] if isinstance(card.get("configs"), list) else None,
    }


def observe_xml(text: str) -> dict:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        return {"parse_error": str(e)[:120], "head": text[:200]}
    tags = [re.sub(r"\{.*\}", "", el.tag) for el in root.iter()]
    counts = {t: tags.count(t) for t in ("record", "entry", "set", "Contents", "CommonPrefixes", "resumptionToken", "license", "error")}
    prefixes = [el.text or "" for el in root.iter() if re.sub(r"\{.*\}", "", el.tag) == "Prefix"][:15]
    err = [el.text for el in root.iter() if re.sub(r"\{.*\}", "", el.tag) == "error"][:2]
    return {"root": re.sub(r"\{.*\}", "", root.tag), "counts": {k: v for k, v in counts.items() if v}, "prefixes": prefixes or None, "errors": err or None}


def probe(client: httpx.Client, kind: str, url: str) -> dict:
    t0 = time.monotonic()
    res: dict[str, Any] = {"kind": kind, "url": url}
    try:
        if kind == "head":
            with client.stream("GET", url) as r:
                res.update(status=r.status_code, final_url=str(r.url), content_type=r.headers.get("content-type"), content_length=r.headers.get("content-length"),
                           last_modified=r.headers.get("last-modified"))
                headers = r.headers
        else:
            r = client.get(url)
            headers = r.headers
            res.update(status=r.status_code, final_url=str(r.url), content_type=r.headers.get("content-type"), bytes=len(r.content))
            body = r.text
            if kind in ("json", "github_repo", "hf_dataset", "hf_model"):
                try:
                    data = r.json()
                    res["observation"] = observe_github(data) if kind == "github_repo" and r.status_code == 200 else observe_hf(data) if kind in ("hf_dataset", "hf_model") and r.status_code == 200 else observe_json(data)
                except ValueError:
                    res["observation"] = {"non_json_head": body[:200]}
            elif kind == "xml":
                res["observation"] = observe_xml(body)
            elif kind == "doc":
                if r.status_code == 404 and ("/main/" in url or "/master/" in url):
                    alt = url.replace("/main/", "/master/") if "/main/" in url else url.replace("/master/", "/main/")
                    r = client.get(alt)
                    body = r.text
                    res.update(status=r.status_code, final_url=str(r.url), bytes=len(r.content))
                lines = [re.sub(r"\s+", " ", ln).strip()[:300] for ln in body.splitlines() if LICENSE_LINE.search(ln)]
                res["observation"] = {"length": len(body), "license_lines": lines[:30]}
            else:
                res["observation"] = {"length": len(body), "head": re.sub(r"\s+", " ", body[:300])}
        res["rate_limit_headers"] = {k: v for k, v in headers.items() if "ratelimit" in k.lower() or "rate-limit" in k.lower() or k.lower() == "retry-after"}
    except httpx.HTTPError as e:
        res.update(status=None, error=f"{type(e).__name__}: {str(e)[:200]}")
    res["latency_s"] = round(time.monotonic() - t0, 3)
    res["ok"] = bool(res.get("status") and 200 <= res["status"] < 300)
    res["probed_at"] = now_utc().isoformat()
    return res


DERIVED: list[tuple[str, str, str, Callable[[dict], str | None]]] = [
    ("openml_run_detail", "openml", "json", lambda R: _openml_run_url(R.get("openml_run_list"))),
]


def _openml_run_url(res: dict | None) -> str | None:
    if not res or not res.get("ok"):
        return None
    try:
        rid = httpx.get(res["url"], timeout=30).json()["runs"]["run"][0]["run_id"]
    except Exception:  # noqa: BLE001 - probe helper; failure = no derived probe
        return None
    return f"https://www.openml.org/api/v1/json/run/{rid}"


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    results: dict[str, dict] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    gh_remaining = 60
    with httpx.Client(headers={"User-Agent": UA, "Accept": "application/json, text/xml;q=0.9, */*;q=0.8"}, timeout=30, follow_redirects=True) as client:
        todo = [p for p in PROBES if p[0].startswith(args.only)]
        for pid, source, kind, url in todo:
            if not args.refresh and results.get(pid, {}).get("ok"):
                continue
            if url.startswith(GH) and gh_remaining < 3:
                results[pid] = {"source_id": source, "kind": kind, "url": url, "status": None, "ok": False, "error": "DEFERRED_RATE_LIMIT", "probed_at": now_utc().isoformat()}
                continue
            res = probe(client, kind, url)
            if url.startswith(GH):
                gh_remaining = int(res.get("rate_limit_headers", {}).get("x-ratelimit-remaining", gh_remaining - 1))
            results[pid] = {"source_id": source, **res}
            print(f"{pid:32s} {str(res.get('status')):5s} {res['latency_s']:6.2f}s {res.get('error') or ''}", flush=True)
            time.sleep(1.0)
        for pid, source, kind, fn in DERIVED:
            if args.only and not pid.startswith(args.only):
                continue
            url = fn(results)
            if url and (args.refresh or not results.get(pid, {}).get("ok")):
                res = probe(client, kind, url)
                results[pid] = {"source_id": source, **res}
                print(f"{pid:32s} {str(res.get('status')):5s} {res['latency_s']:6.2f}s", flush=True)
    atomic_write_bytes(OUT, json.dumps(dict(sorted(results.items())), indent=1, ensure_ascii=False).encode("utf-8"))
    ok = sum(1 for r in results.values() if r.get("ok"))
    print(f"{ok}/{len(results)} probes OK; GitHub remaining ~{gh_remaining}; wrote {OUT}")


if __name__ == "__main__":
    main()
