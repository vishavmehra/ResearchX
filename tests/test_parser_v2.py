"""Parser 0.2.0: addressable cells, header paths, spans, table types, sentences, footnotes, appendix flags."""

from services.datasets.schemas import LicenseInfo
from services.extraction.parse import cell_value, parse_html, split_sentences

HTML = """<html><body>
<h1 class="ltx_title ltx_title_document">A Paper</h1>
<section id="S4" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">4 </span>Experiments</h2>
 <div class="ltx_para" id="S4.p1"><p class="ltx_p">We fine-tune with a learning rate of 2e-5 (see Fig. 2). Results improve.<span class="ltx_note ltx_role_footnote" id="footnote1"><sup class="ltx_note_mark">1</sup><span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>We use 3 seeds.</span></span></span></p></div>
 <figure class="ltx_table" id="S4.T1">
  <figcaption>Table 1: Main results on GLUE.</figcaption>
  <table class="ltx_tabular">
   <thead><tr><th class="ltx_td ltx_th ltx_th_column" rowspan="2">Model</th><th class="ltx_td ltx_th ltx_th_column" colspan="2">GLUE</th></tr>
          <tr><th class="ltx_td ltx_th ltx_th_column">MNLI</th><th class="ltx_td ltx_th ltx_th_column">SST-2</th></tr></thead>
   <tbody><tr><td class="ltx_td">Zero-shot</td><td></td><td></td></tr>
          <tr><th class="ltx_td ltx_th ltx_th_row">BERT</th><td>84.6 ± 0.2</td><td>93.5%</td></tr>
          <tr><th class="ltx_td ltx_th ltx_th_row">Ours</th><td>86.1†</td><td>94.0</td></tr></tbody>
  </table>
 </figure>
 <figure class="ltx_table" id="S4.T2"><figcaption>Table 2: Hyperparameters.</figcaption>
  <table class="ltx_tabular"><tr><td>Learning rate</td><td>2e-5</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Epochs</td><td>3</td></tr></table>
 </figure>
 <figure class="ltx_figure" id="S4.F1"><img src="x.png"/><figcaption>Figure 1: Accuracy vs steps.</figcaption></figure>
</section>
<section id="A1" class="ltx_appendix"><h2 class="ltx_title ltx_title_appendix">Appendix A Details</h2>
 <div class="ltx_para" id="A1.p1"><p class="ltx_p">Training uses AdamW, e.g. with beta 0.9. Done.</p></div>
</section>
</body></html>"""

META = {"paper_id": "paper:test", "source_record_id": "arxiv:test", "source_url": "https://arxiv.org/html/test", "source_path": "x", "source_sha256": "0" * 64}


def doc():
    return parse_html(HTML, document_meta=META, license=LicenseInfo())


def test_cells_spans_and_header_paths():
    t = {t.table_id: t for t in doc().tables}["S4.T1"]
    assert (t.header_rows, t.header_cols, t.table_type) == (2, 1, "RESULTS")
    cells = {c.cell_id: c for c in t.cells}
    bert_mnli = cells["S4.T1/r3c1"]
    assert bert_mnli.row_header_path == ["Zero-shot", "BERT"]  # row-group label + row header
    assert bert_mnli.col_header_path == ["GLUE", "MNLI"]  # colspan header + sub-header
    assert (bert_mnli.value, bert_mnli.uncertainty) == (84.6, 0.2)
    assert (cells["S4.T1/r3c2"].value, cells["S4.T1/r3c2"].unit) == (93.5, "percent")
    assert cells["S4.T1/r4c1"].value == 86.1  # dagger decoration ignored, text kept verbatim
    assert cells["S4.T1/r4c1"].text == "86.1†"
    assert cells["S4.T1/r1c0"].spanned_from == "S4.T1/r0c0"  # rowspan coverage is explicit
    assert cells["S4.T1/r0c2"].spanned_from == "S4.T1/r0c1"  # colspan coverage is explicit


def test_table_types_and_legacy_rows_kept():
    t = {t.table_id: t for t in doc().tables}
    assert t["S4.T2"].table_type == "HYPERPARAMETERS"
    assert t["S4.T2"].rows == [["Learning rate", "2e-5"], ["Batch size", "32"], ["Epochs", "3"]]


def test_footnotes_sentences_figures_appendix():
    d = doc()
    p = {p.paragraph_id: p for p in d.paragraphs}["S4.p1"]
    assert "3 seeds" not in p.text  # footnote content is its own unit
    assert [f.text for f in d.footnotes] == ["We use 3 seeds."] and d.footnotes[0].paragraph_id == "S4.p1"
    sents = [s for s in d.sentences if s.paragraph_id == "S4.p1"]
    assert [s.text for s in sents] == ["We fine-tune with a learning rate of 2e-5 (see Fig. 2).", "Results improve."]
    assert all(p.text[s.char_start:s.char_end] == s.text for s in sents)
    assert [f.figure_id for f in d.figures] == ["S4.F1"]
    secs = {s.section_id: s for s in d.sections}
    assert secs["A1"].is_appendix and not secs["S4"].is_appendix


def test_sentence_splitter_and_cell_values():
    assert split_sentences("Use e.g. AdamW. Then stop.") == [(0, 15), (16, 26)]
    assert cell_value("71.2 (0.3)") == (71.2, None, None)
    assert cell_value("—") == (None, None, None)
    assert cell_value("1.5k") == (1500.0, None, None)
