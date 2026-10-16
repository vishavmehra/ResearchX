"""Evidence assertions: construction, span location and verification (DATASET_SPEC §11, §12).

Status semantics used by the pipeline:
- MACHINE_EXTRACTED: produced by an extractor/importer, not yet re-checked.
- SOURCE_CHECKED: the evidence text (or structured value) was re-found verbatim at the stated
  location of the stored source (parsed full text or stored source record). This verifies that
  the source says it; it does NOT confirm the field assignment. Only humans set HUMAN_REVIEWED.
- REJECTED: re-check failed; the evidence may not support any KNOWN field.
"""

from __future__ import annotations

from typing import Any

from services.datasets.schemas import EvidenceAssertion, ParsedDocument, Provenance, SourceRecord, Table
from services.datasets.store import DATA, code_version, stable_id

EVIDENCE_VERSION = "evidence-0.1.0"


def load_parsed_docs() -> dict[str, ParsedDocument]:
    """paper_id -> ParsedDocument (from data/fulltext/<class>/<id>/parsed.json)."""
    docs = {}
    for f in sorted((DATA / "fulltext").glob("*/*/parsed.json")):
        d = ParsedDocument.model_validate_json(f.read_text(encoding="utf-8"))
        docs[d.paper_id] = d
    return docs


def table_text(tb: Table) -> str:
    # Must stay identical to services.datasets.validate (evidence re-verification).
    return " | ".join(" | ".join(r) for r in tb.rows) + " " + tb.caption


def location_texts(doc: ParsedDocument) -> dict[str, str]:
    """Every addressable unit: paragraphs, tables (legacy flat text), footnotes, cells (0.2.0)."""
    texts = {p.paragraph_id: p.text for p in doc.paragraphs}
    texts.update({t.table_id: table_text(t) for t in doc.tables})
    texts.update({f.footnote_id: f.text for f in doc.footnotes})
    texts.update({c.cell_id: c.text for t in doc.tables for c in t.cells if c.spanned_from is None})
    texts.update({f"{t.table_id}#caption": t.caption for t in doc.tables if t.caption})
    return texts


def find_span(hay: str, needle: str) -> tuple[int, int] | None:
    """Exact match first; else whitespace- and case-insensitive match mapped to original offsets."""
    if not needle.strip():
        return None
    i = hay.find(needle)
    if i >= 0:
        return i, i + len(needle)
    chars, index = [], []
    prev_space = False
    for j, ch in enumerate(hay):
        if ch.isspace():
            if not prev_space:
                chars.append(" ")
                index.append(j)
            prev_space = True
            continue
        low = ch.lower()
        chars.append(low if len(low) == 1 else ch)
        index.append(j)
        prev_space = False
    norm_hay = "".join(chars)
    norm_needle = " ".join(needle.split()).lower()
    k = norm_hay.find(norm_needle)
    if k < 0:
        return None
    return index[k], index[k + len(norm_needle) - 1] + 1


def section_label(doc: ParsedDocument, section_id: str | None) -> str | None:
    for s in doc.sections:
        if s.section_id == section_id:
            return f"{s.number} {s.title}".strip() if s.number else s.title
    return section_id


def leaderboard_locator(r: dict[str, Any]) -> str:
    return f"leaderboard:{r['task']}::{r['dataset']}::row{r['row_index']}"


def make(
    *,
    source_name: str,
    source_url: str | None = None,
    content_hash: str | None = None,
    **fields: Any,
) -> EvidenceAssertion:
    eid = stable_id(
        "ev",
        fields["source_record_id"],
        fields.get("experiment_id"),
        fields["field_name"],
        # 0.2.0 cell / footnote locations first (two cells of one table may hold the same text); v0.1.0 ids unchanged
        fields.get("cell") or fields.get("footnote") or fields.get("artifact_id") or fields.get("paragraph") or fields.get("table") or fields.get("source_location"),
        fields.get("char_start"),
        fields.get("extracted_value"),
        fields["evidence_text"][:300],
    )
    prov = Provenance(
        source_name=source_name,
        source_url=source_url,
        source_identifier=fields["source_record_id"],
        source_version=fields.get("source_version"),
        retrieved_at=fields.get("retrieval_timestamp"),
        content_hash=content_hash,
        processing_version=code_version(),
        extractor_version=fields["extraction_model"],
        parent_record_ids=[fields["source_record_id"]],
    )
    return EvidenceAssertion(evidence_id=eid, provenance=prov, **fields)


def verify(ev: EvidenceAssertion, docs: dict[str, ParsedDocument], records: dict[str, SourceRecord],
           artifacts: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Re-find the evidence at its stated location. Returns (ok, reason)."""
    if ev.artifact_id:  # repository file / model card / run record: re-read the cached copy
        art = (artifacts or {}).get(ev.artifact_id)
        if art is None or not art.get("raw_path"):
            return False, "artifact_not_found"
        from services.retrieval.http import read_body

        p = DATA.parent / art["raw_path"]
        if not p.exists():
            return False, "artifact_cache_missing"
        return ev.evidence_text in read_body(p).decode("utf-8", errors="replace"), "text_not_in_artifact"
    if ev.cell or ev.footnote:
        doc = docs.get(ev.paper_id or "")
        if doc is None:
            return False, "no_parsed_document"
        if doc.source_record_id != ev.source_record_id:
            return False, "source_version_mismatch"
        hay = location_texts(doc).get(ev.cell or ev.footnote)
        if hay is None:
            return False, "location_not_found"
        return ev.evidence_text in hay, "text_not_found"
    if ev.paragraph or (ev.table and not ev.table.startswith("leaderboard:")):
        doc = docs.get(ev.paper_id or "")
        if doc is None:
            return False, "no_parsed_document"
        if doc.source_record_id != ev.source_record_id:
            return False, "source_version_mismatch"
        hay = location_texts(doc).get(ev.paragraph or ev.table)
        if hay is None:
            return False, "location_not_found"
        if ev.paragraph and ev.char_start is not None:
            return hay[ev.char_start : ev.char_end] == ev.evidence_text, "offset_text_mismatch"
        return ev.evidence_text in hay, "text_not_found"
    rec = records.get(ev.source_record_id)
    if rec is None:
        return False, "no_source_record"
    if ev.field_name.startswith("paper."):  # metadata claim: compare with the stored source record
        attr = ev.field_name.split(".", 1)[1]
        actual = getattr(rec, attr, None)
        actual = str(len(actual)) if attr == "authors" else str(actual)
        return actual == ev.extracted_value, "metadata_value_mismatch"
    for r in rec.extra.get("evaluation_results", []):
        if ev.table == leaderboard_locator(r):
            values = {r["task"], r["dataset"], r["model_name"], *r["metrics"].keys(), *map(str, r["metrics"].values())}
            return ev.extracted_value in values, "value_not_in_source_record"
    for link in rec.extra.get("code_links", []):
        if ev.extracted_value and ev.extracted_value == link.get("repo_url"):
            return True, "ok"
    return False, "structured_location_not_found"
