"""The licence table the request-time full-text reader needs ships with the code.

2026-10-21 production defect: data/manifests/licenses.yaml was git-ignored with the rest of data/, so the deployed API had
no licence table. Every arXiv full text read at request time failed with FileNotFoundError and the run fell back to
abstracts. These tests use the real table and the real reader (only the HTTP fetch is stubbed), so a checkout without
the file fails here instead of in production.
"""

import shutil
import subprocess

import httpx
import pytest

from services.datasets import manifests
from services.datasets.store import ROOT
from services.runtime import reader


def page(licence_anchor: str) -> str:
    return (f"<html><body>{licence_anchor}<article class=\"ltx_document\">"
            "<h1 class=\"ltx_title ltx_title_document\">Tiny Paper</h1>"
            "<section id=\"S4\" class=\"ltx_section\"><h2 class=\"ltx_title ltx_title_section\">4 Experimental Setup</h2>"
            "<div id=\"S4.p1\" class=\"ltx_para\"><p class=\"ltx_p\">We train with a learning rate of 0.02 and a batch size of 16.</p></div>"
            "</section></article></body></html>")


PAGES = {
    "2401.00001": page('<a id="license-tr" href="https://info.arxiv.org/help/license/index.html#licenses-available">'
                       "License: arXiv.org perpetual non-exclusive license</a>"),
    "2401.00002": page('<a id="license-tr" href="https://creativecommons.org/licenses/by/4.0/">License: CC BY 4.0</a>'),
    "2401.00003": page(""),  # no licence statement
}


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_licence_table_is_committed():
    out = subprocess.run(["git", "ls-files", "--error-unmatch", "data/manifests/licenses.yaml"], cwd=ROOT, capture_output=True)
    assert out.returncode == 0, "data/manifests/licenses.yaml must be committed: the deployed full-text reader needs it"


def test_reader_applies_licence_restrictions_from_the_shipped_table(monkeypatch):
    assert (manifests.MANIFESTS / "licenses.yaml").is_file()
    manifests.licenses.cache_clear()
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, text=PAGES[url.rsplit("/", 1)[1]], request=httpx.Request("GET", url)))
    for arxiv_id in PAGES:
        reader._cache.pop(arxiv_id, None)
    arxiv_default, cc_by, unstated = (reader.read(a) for a in PAGES)
    assert (arxiv_default["license"], arxiv_default["redistribution_allowed"]) == ("ARXIV-NONEXCLUSIVE-1.0", False)
    assert (cc_by["license"], cc_by["redistribution_allowed"]) == ("CC-BY-4.0", True)
    assert (unstated["license"], unstated["redistribution_allowed"]) == (None, False)  # unknown licence: never redistributed
    assert any("learning rate of 0.02" in u["text"] for u in cc_by["units"])
    assert manifests.fulltext_class("ARXIV-NONEXCLUSIVE-1.0") == manifests.fulltext_class(None) == "restricted"
