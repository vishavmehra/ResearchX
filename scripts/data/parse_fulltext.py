"""Parse acquired full text into ParsedDocuments (sections, paragraphs, tables) (DATASET_SPEC §37).

Output sits beside the original (same licence class): data/fulltext/<class>/<arxiv_id>/parsed.json.
"""

import argparse
import json

from services.datasets.schemas import FailureCategory, LicenseInfo
from services.datasets.store import ROOT, atomic_write_bytes, rel, run_stage, sha256_file
from services.extraction.parse import parse_html, parse_pdf
from services.retrieval.fulltext import load_index, write_index


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_stage("parse_fulltext") as run:
        index = load_index()
        for arxiv_id, row in sorted(index.items()):
            ft = row["fulltext"]
            src = ROOT / ft["path"]
            if not src.exists() or sha256_file(src) != ft["sha256"]:
                run.fail(FailureCategory.dataset_error, ft["path"], "full text matching recorded sha256", "missing or checksum mismatch", severity="HIGH")
                continue
            meta = {
                "paper_id": f"paper:arxiv:{arxiv_id}",
                "source_record_id": f"arxiv:{arxiv_id}{ft['retrieval_url'].rsplit(arxiv_id, 1)[-1]}",
                "source_url": ft["retrieval_url"],
                "source_path": ft["path"],
                "source_sha256": ft["sha256"],
            }
            lic = LicenseInfo.model_validate(row["license"])
            try:
                if ft["format"] == "HTML":
                    doc = parse_html(src.read_text(encoding="utf-8", errors="replace"), document_meta=meta, license=lic)
                else:
                    doc = parse_pdf(src.read_bytes(), document_meta=meta, license=lic)
            except Exception as e:  # parsing must never crash the batch; failure is recorded
                run.fail(FailureCategory.parsing_failure, ft["path"], "ParsedDocument", f"{type(e).__name__}: {e}", severity="HIGH")
                continue
            if doc.warnings and not doc.paragraphs:
                run.fail(FailureCategory.parsing_failure, ft["path"], "paragraphs", "; ".join(doc.warnings))
            out = src.parent / "parsed.json"
            atomic_write_bytes(out, json.dumps(doc.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"))
            ft["parsed_path"] = rel(out)
            run.counts["parsed"] = run.counts.get("parsed", 0) + 1
            run.counts["paragraphs"] = run.counts.get("paragraphs", 0) + len(doc.paragraphs)
            run.counts["tables"] = run.counts.get("tables", 0) + len(doc.tables)
            if run.counts["parsed"] % 100 == 0:  # checkpoint; one index rewrite per document was O(n^2) I/O
                write_index(index)
        run.output(write_index(index))


if __name__ == "__main__":
    main()
