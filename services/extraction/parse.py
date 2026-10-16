"""Scientific document parsing (DATASET_SPEC §37: structure detection + section segmentation).

LaTeXML HTML (arXiv) is preferred: it carries stable ids for sections (S4), paragraphs (S4.p1)
and tables (S5.T2), which become precise evidence locations and URL fragments. Inline math is
replaced by its TeX alttext so values like `2\\times 10^{-4}` survive as text. PDFs fall back to
page-level paragraphs with regex-detected headings (no tables).
"""

from __future__ import annotations

import io
import logging
import re

import lxml.html
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding warnings are noise, not failures

from services.canonicalization.normalize import clean_text, parse_number
from services.datasets.schemas import (
    Figure,
    Footnote,
    FulltextFormat,
    LicenseInfo,
    Paragraph,
    ParsedDocument,
    Section,
    Sentence,
    Table,
    TableCell,
)
from services.datasets.store import now_utc, stable_id

PARSER_VERSION = "parser-0.2.0"  # 0.2.0: cells + header paths, sentences, footnotes, figures, appendix flags
SECTION_TYPES = [
    ("ABSTRACT", r"\babstract\b"),
    ("INTRODUCTION", r"\bintroduction\b"),
    ("RELATED_WORK", r"related work|background|prior work|preliminar"),
    ("ABLATION", r"ablation"),
    ("EXPERIMENTS", r"experiment|evaluation|setup|implementation|training|fine-?tuning|hyper-?parameter|benchmark|dataset|baseline|downstream"),
    ("RESULTS", r"\bresults?\b|performance|comparison|main findings"),
    ("ANALYSIS", r"analysis|discussion|case stud"),
    ("METHOD", r"method|approach|model|architecture|framework|algorithm|our "),
    ("LIMITATIONS", r"limitation|broader impact|ethic"),
    ("CONCLUSION", r"conclusion|future work|summary"),
    ("ACKNOWLEDGEMENTS", r"acknowledg"),
    ("REFERENCES", r"reference|bibliograph"),
    ("APPENDIX", r"appendix|supplementary"),
]
LEVEL = {"ltx_section": 1, "ltx_appendix": 1, "ltx_subsection": 2, "ltx_subsubsection": 3, "ltx_paragraph": 4}


def classify_section(title: str, section_id: str = "") -> str:
    if re.match(r"^A\d", section_id):
        return "APPENDIX"
    t = title.casefold()
    for kind, pat in SECTION_TYPES:
        if re.search(pat, t):
            return kind
    return "OTHER"


def _text(el, drop_notes: bool = False) -> str:
    if el is None:
        return ""
    el = lxml.html.fromstring(lxml.html.tostring(el))  # work on a copy
    for m in el.xpath(".//math"):
        alt = m.get("alttext") or ""
        m.tail = f"{alt}{m.tail or ''}"
        m.drop_tree()
    for junk in el.xpath(".//*[contains(@class,'ltx_tag_section') or contains(@class,'ltx_tag_subsection') or self::script or self::style]"):
        junk.drop_tree()
    if drop_notes:  # footnote content becomes its own addressable unit (Footnote)
        for note in el.xpath(".//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_note ')]"):
            note.drop_tree()
    return clean_text(el.text_content())


# Sentence splitting: deterministic, abbreviation-aware; offsets index into Paragraph.text.
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|et al|etc|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Tabs|Ref|vs|cf|approx|resp|w\.r\.t|No|Nos|Vol|Appx|App)\.|\b[A-Z]\.)$")
_SENT_END = re.compile(r"[.!?](?=\s+[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        if _ABBREV.search(text[start:end]):
            continue
        if text[start:end].strip():
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


_UNCERTAINTY = re.compile(r"^\s*([-+−]?\d+(?:\.\d+)?)\s*(?:±|\\pm|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*(%?)\s*$")


def cell_value(text: str) -> tuple[float | None, str | None, float | None]:
    """Numeric reading of a cell: (value, unit, uncertainty). Decorations like †, *, bold marks are ignored."""
    t = re.sub(r"[†‡*§¶♣♠♢♡]+|\^\{?[†‡*\w]\}?$", "", text).strip()
    m = _UNCERTAINTY.match(t)
    if m:
        v = parse_number(m.group(1))
        u = parse_number(m.group(2))
        return (v[0] if v else None, "percent" if m.group(3) else None, u[0] if u else None)
    unit = None
    if t.endswith("%"):
        t, unit = t[:-1].strip(), "percent"
    t = re.sub(r"\s*\(\s*[-+]?\d+(?:\.\d+)?\s*\)$", "", t)  # "71.2 (0.3)": keep the main value only
    v = parse_number(t)
    return (v[0] if v else None, unit, None)


_HYPER_ROW = re.compile(r"learning[- ]rate|\blr\b|batch[- ]size|epochs?|warm-?up|weight[- ]decay|dropout|optimi[sz]er|\bsteps?\b|seed|lora|rank|alpha|gradient|scheduler|max(imum)? (sequence|length)|beam", re.I)
_TYPE_RULES = [
    ("HYPERPARAMETERS", re.compile(r"hyper-?parameters?|training (details|settings|configuration|setup)|implementation details|fine-?tuning (details|settings)", re.I)),
    ("DATASET_STATISTICS", re.compile(r"statistic|dataset (details|summary|overview|sizes?)|number of (examples|samples|instances|documents|queries)|#\s?(train|test|samples)", re.I)),
    ("RESULTS", re.compile(r"results?|performance|accuracy|comparison|compar(e|ing)|ablation|evaluat|scores?|\bF1\b|BLEU|ROUGE|perplexity|win rate|success rate", re.I)),
]


def _classify_table(caption: str, grid: dict[tuple[int, int], str], n_rows: int, n_cols: int, header_rows: int) -> str:
    first_col = [grid.get((r, 0), "") for r in range(header_rows, n_rows)]
    hyper_rows = sum(1 for t in first_col if _HYPER_ROW.search(t))
    if _TYPE_RULES[0][1].search(caption) or (hyper_rows >= 3 and hyper_rows >= 0.4 * max(len(first_col), 1)):
        return "HYPERPARAMETERS"
    if _TYPE_RULES[1][1].search(caption):
        return "DATASET_STATISTICS"
    data = [grid.get((r, c), "") for r in range(header_rows, n_rows) for c in range(1, n_cols)]
    numeric = sum(1 for t in data if cell_value(t)[0] is not None)
    if _TYPE_RULES[2][1].search(caption) or (data and numeric >= 0.5 * len(data)):
        return "RESULTS"
    return "OTHER"


def _build_table(fig, tid: str, section_id: str | None) -> Table:
    cap = fig.xpath(".//figcaption")
    caption = _text(cap[0]) if cap else ""
    trs = fig.xpath(".//tr")
    rows = [[_text(c) for c in tr.xpath("./td|./th")] for tr in trs]
    rows = [r for r in rows if any(r)]  # legacy view (v0.1.0 semantics, used by the PwC cross-check)

    # grid with rowspan/colspan expansion
    origin: dict[tuple[int, int], tuple[int, int]] = {}
    texts: dict[tuple[int, int], str] = {}
    meta: dict[tuple[int, int], dict] = {}
    r_out = 0
    for tr in trs:
        tds = tr.xpath("./td|./th")
        if not tds:
            continue
        c = 0
        for td in tds:
            while (r_out, c) in origin:
                c += 1
            try:
                rs, cs = max(int(td.get("rowspan", 1)), 1), max(int(td.get("colspan", 1)), 1)
            except ValueError:
                rs, cs = 1, 1
            cls = td.get("class") or ""
            in_thead = bool(td.xpath("ancestor::thead"))
            txt = _text(td)
            for dr in range(rs):
                for dc in range(cs):
                    origin[(r_out + dr, c + dc)] = (r_out, c)
                    texts[(r_out + dr, c + dc)] = txt
            meta[(r_out, c)] = {
                "rs": rs, "cs": cs,
                "col_header": in_thead or "ltx_th_column" in cls or (td.tag == "th" and "ltx_th_row" not in cls),
                "row_header": "ltx_th_row" in cls,
            }
            c += cs
        r_out += 1
    n_rows = max((r for r, _ in origin), default=-1) + 1
    n_cols = max((c for _, c in origin), default=-1) + 1

    # header rows: leading rows whose non-empty cells are all column headers (or, without markup, non-numeric
    # rows above numeric data); header cols: leading columns marked as row headers (or textual first column)
    def row_is_header(r: int) -> bool:
        cells = [meta[origin[(r, c)]] for c in range(n_cols) if (r, c) in origin and texts[(r, c)]]
        if not cells or not all(m["col_header"] for m in cells):
            return False
        if r == 0:
            return True  # first marked row: trust markup even when headers are numbers (e.g. resolutions)
        body = [texts.get((r, c), "") for c in range(1, n_cols) if texts.get((r, c))]
        return sum(1 for t in body if cell_value(t)[0] is not None) < 0.5 * max(len(body), 1)  # numeric rows are data

    header_rows = 0
    while header_rows < n_rows and row_is_header(header_rows):
        header_rows += 1
    if header_rows == 0 and n_rows > 1:
        first_numeric = any(cell_value(texts.get((0, c), ""))[0] is not None for c in range(n_cols))
        later_numeric = any(cell_value(texts.get((r, c), ""))[0] is not None for r in range(1, n_rows) for c in range(1, n_cols))
        if not first_numeric and later_numeric:
            header_rows = 1
    # row-group labels: a body row whose only non-empty content is its first cell (or one full-width span)
    group_label: dict[int, str] = {}
    group_rows: set[int] = set()
    current = None
    for r in range(header_rows, n_rows):
        vals = [texts.get((r, c), "") for c in range(n_cols)]
        distinct = {v for v in vals if v}
        if len(distinct) == 1 and (vals[0] and (all(not v for v in vals[1:]) or all(v == vals[0] for v in vals))) and n_cols > 1:
            current = vals[0]
            group_rows.add(r)
            continue
        if current:
            group_label[r] = current

    marked_row_headers = any(m["row_header"] for m in meta.values())
    body_rows = [r for r in range(header_rows, n_rows) if r not in group_rows]
    header_cols = 0
    if marked_row_headers:
        while header_cols < n_cols and all(
            meta[origin[(r, header_cols)]]["row_header"] or not texts.get((r, header_cols))
            for r in body_rows if (r, header_cols) in origin
        ):
            header_cols += 1
    elif n_cols > 1:
        col0 = [texts.get((r, 0), "") for r in body_rows]
        if col0 and sum(1 for t in col0 if t and cell_value(t)[0] is None) >= 0.6 * len(col0):
            header_cols = 1

    cells: list[TableCell] = []
    for (r, c) in sorted(origin):
        o = origin[(r, c)]
        txt = texts[(r, c)]
        is_header = r < header_rows or c < header_cols
        col_path: list[str] = []
        for hr in range(header_rows):
            t = texts.get((hr, c), "")
            if t and (not col_path or col_path[-1] != t):
                col_path.append(t)
        row_path: list[str] = [group_label[r]] if r in group_label else []
        for hc in range(header_cols):
            t = texts.get((r, hc), "")
            if t and (not row_path or row_path[-1] != t):
                row_path.append(t)
        value, unit, unc = (None, None, None) if is_header else cell_value(txt)
        m = meta[o]
        cells.append(TableCell(
            cell_id=f"{tid}/r{r}c{c}", row=r, col=c, text=txt, is_header=is_header,
            row_span=m["rs"] if o == (r, c) else 1, col_span=m["cs"] if o == (r, c) else 1,
            spanned_from=None if o == (r, c) else f"{tid}/r{o[0]}c{o[1]}",
            row_header_path=[] if c < header_cols else row_path, col_header_path=[] if r < header_rows else col_path,
            value=value, unit=unit, uncertainty=unc,
        ))
    notes = [_text(p) for p in fig.xpath(".//p[contains(@class,'ltx_p')][not(ancestor::figcaption)][not(ancestor::table)]") if _text(p)]
    return Table(
        table_id=tid, section_id=section_id, caption=caption, rows=rows, header_rows=header_rows, header_cols=header_cols,
        cells=cells, table_type=_classify_table(caption, texts, n_rows, n_cols, header_rows), notes=notes,
    )


def _section_of(el) -> str | None:
    for anc in el.iterancestors("section"):
        if anc.get("id"):
            return anc.get("id")
    return None


def parse_html(html: str, *, document_meta: dict, license: LicenseInfo) -> ParsedDocument:
    doc = lxml.html.fromstring(html)
    warnings: list[str] = []
    title_el = doc.xpath("//h1[contains(@class,'ltx_title_document')]")
    abstract_el = doc.xpath("//div[contains(@class,'ltx_abstract')]")
    abstract = _text(abstract_el[0]) if abstract_el else None
    if abstract and abstract.casefold().startswith("abstract"):
        abstract = abstract[8:].strip()

    sections = []
    for sec in doc.xpath("//section[@id]"):
        cls = next((c for c in (sec.get("class") or "").split() if c in LEVEL), None)
        if not cls or sec.get("id") == "bib":
            continue
        head = sec.xpath("./*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][contains(@class,'ltx_title')]")
        tag = head[0].xpath(".//span[contains(@class,'ltx_tag')]") if head else []
        number = clean_text(tag[0].text_content()).rstrip(".") if tag else None
        title = _text(head[0]) if head else ""
        if number and title.startswith(number):
            title = title[len(number):].strip()
        sid = sec.get("id")
        is_appendix = "ltx_appendix" in (sec.get("class") or "") or bool(re.match(r"^A\d", sid)) or bool(
            sec.xpath("ancestor::section[contains(@class,'ltx_appendix')]"))
        sections.append(
            Section(section_id=sid, number=number or None, title=title, level=LEVEL[cls],
                    section_type=classify_section(title, sid), parent_id=_section_of(sec), is_appendix=is_appendix)
        )

    paragraphs, sentences = [], []
    for para in doc.xpath("//div[contains(concat(' ', normalize-space(@class), ' '), ' ltx_para ')][@id]"):
        if para.xpath("ancestor::figure") or para.xpath("ancestor::*[@id='bib']"):
            continue
        text = _text(para, drop_notes=True)
        if text:
            pid = para.get("id")
            paragraphs.append(Paragraph(paragraph_id=pid, section_id=_section_of(para), text=text))
            for k, (a, b) in enumerate(split_sentences(text), 1):
                sentences.append(Sentence(sentence_id=f"{pid}.s{k}", paragraph_id=pid, char_start=a, char_end=b, text=text[a:b]))

    footnotes = []
    for i, note in enumerate(doc.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_note ')][contains(@class,'ltx_role_footnote')]"), 1):
        content = note.xpath(".//*[contains(@class,'ltx_note_content')]")
        body = content[0] if content else note
        for mark in body.xpath(".//*[contains(@class,'ltx_note_mark')]"):
            mark.drop_tree()
        text = _text(body)
        if text:
            para = note.xpath("ancestor::div[contains(concat(' ', normalize-space(@class), ' '), ' ltx_para ')][@id]")
            footnotes.append(Footnote(footnote_id=note.get("id") or f"footnote{i}", paragraph_id=para[0].get("id") if para else None, text=text))

    figures = []
    for fig in doc.xpath("//figure[contains(@class,'ltx_figure')][@id][not(contains(@class,'ltx_table'))]"):
        if fig.xpath("ancestor::figure[contains(@class,'ltx_table')]"):
            continue
        cap = fig.xpath("./figcaption")
        if cap and _text(cap[0]):
            figures.append(Figure(figure_id=fig.get("id"), section_id=_section_of(fig), caption=_text(cap[0])))

    # outermost table figures only (sub-tables are part of their parent; v0.1.0 duplicated them)
    tables = [_build_table(fig, fig.get("id"), _section_of(fig))
              for fig in doc.xpath("//figure[contains(@class,'ltx_table')][@id][not(ancestor::figure[contains(@class,'ltx_table')])]")]
    if not sections:
        warnings.append("no sections detected")
    if not paragraphs:
        warnings.append("no paragraphs detected")
    return ParsedDocument(
        document_id=stable_id("doc", document_meta["source_record_id"], document_meta["source_sha256"]),
        format=FulltextFormat.HTML,
        parser_version=PARSER_VERSION,
        parsed_at=now_utc(),
        title=_text(title_el[0]) if title_el else None,
        abstract=abstract,
        sections=sections,
        paragraphs=paragraphs,
        tables=tables,
        sentences=sentences,
        footnotes=footnotes,
        figures=figures,
        license=license,
        warnings=warnings,
        **document_meta,
    )


_HEADING = re.compile(r"^(?P<num>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*))\.?\s+(?P<title>[A-Z][^\n]{2,80})$")


def parse_pdf(pdf: bytes, *, document_meta: dict, license: LicenseInfo) -> ParsedDocument:
    reader = PdfReader(io.BytesIO(pdf))
    sections: list[Section] = []
    paragraphs: list[Paragraph] = []
    current = None
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i, block in enumerate(re.split(r"\n\s*\n", text)):
            block = block.strip()
            first = block.split("\n", 1)[0].strip()
            m = _HEADING.match(first)
            if m and len(block) < 120:
                sid = f"pdf.s{len(sections) + 1}"
                sections.append(Section(section_id=sid, number=m.group("num"), title=m.group("title"), level=m.group("num").count(".") + 1,
                                        section_type=classify_section(m.group("title")), parent_id=None))
                current = sid
                continue
            if block:
                paragraphs.append(Paragraph(paragraph_id=f"p{page_no}.{i + 1}", section_id=current, text=clean_text(block), page=page_no))
    return ParsedDocument(
        document_id=stable_id("doc", document_meta["source_record_id"], document_meta["source_sha256"]),
        format=FulltextFormat.PDF,
        parser_version=PARSER_VERSION,
        parsed_at=now_utc(),
        sections=sections,
        paragraphs=paragraphs,
        license=license,
        warnings=["pdf: tables not extracted; headings regex-detected"],
        **document_meta,
    )
