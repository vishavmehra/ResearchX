"""Paths, atomic JSONL storage, checksums, run log and failure log.

Every stage writes complete files atomically (temp file + os.replace) so a crash never
leaves a half-written canonical dataset (spec §33: partial downloads must not corrupt).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from pydantic import BaseModel

from services.datasets.schemas import FailureCase, FailureCategory

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CONFIG = ROOT / "config"
RUNS = DATA / "runs"

# Canonical dataset files (repo-relative to DATA).
PATHS = {
    "source_records": "normalized/sources/source_records.jsonl",
    "papers": "normalized/papers/papers.jsonl",
    "studies": "normalized/studies/studies.jsonl",
    "citations": "normalized/citations/citations.jsonl",
    "experiments": "silver/experiments/experiments.jsonl",
    "experiment_evidence": "silver/evidence/experiment_evidence.jsonl",  # PwC import + table cross-check + LLM
    "rules_evidence": "silver/evidence/rules_evidence.jsonl",  # study-level rule extraction
    "metadata_evidence": "silver/evidence/metadata_evidence.jsonl",  # source-record claims behind contradictions
    "study_links": "silver/study_links/study_links.jsonl",
    "comparisons": "silver/comparisons/comparisons.jsonl",
    "hard_negatives": "silver/comparisons/hard_negatives.jsonl",
    "contradictions": "silver/contradictions/contradictions.jsonl",
    "abstention": "silver/abstention/abstention.jsonl",
    "decisions": "annotations/decisions/decisions.jsonl",
    "promotions": "annotations/promotions/promotions.jsonl",
    # schema 0.2.0 layers (docs/DATASET_DEEPENING.md §1)
    "discovery": "normalized/discovery/candidates.jsonl",
    "deep_evidence": "silver/evidence/deep_evidence.jsonl",  # LLM settings/identity + table-cell evidence
    "artifact_field_evidence": "silver/evidence/artifact_field_evidence.jsonl",  # values read from repository / card files
    "results": "silver/results/results.jsonl",
    "associations": "silver/results/associations.jsonl",
    "artifacts": "silver/artifacts/artifacts.jsonl",
    "artifact_evidence": "silver/artifacts/artifact_evidence.jsonl",  # paper-vs-artifact relations
    "research_tasks": "silver/tasks/research_tasks.jsonl",
    "hard_cases": "silver/tasks/hard_cases.jsonl",
    "canonical_entities": "silver/entities/canonical_entities.jsonl",
    "graph_nodes": "silver/graph/nodes.jsonl",
    "graph_edges": "silver/graph/edges.jsonl",
}


def utf8_stdio() -> None:
    """Windows consoles default to cp1252; scientific text needs UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def path(key: str) -> Path:
    return DATA / PATHS[key]


def rel(p: Path | str) -> str:
    """Repo-relative posix path (absolute posix path for locations outside the repo, e.g. temp dirs)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def load_yaml(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG / f"{name}.yaml")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(prefix: str, *parts: Any, n: int = 16) -> str:
    """Deterministic content-derived id: same inputs -> same id -> reproducible builds."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:n]}"


def atomic_write_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(8):  # Windows: a scanner/indexer briefly holding the target makes replace fail
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.25 * (attempt + 1))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def dump_record(rec: BaseModel | dict) -> dict:
    """Compact JSON form: defaults (e.g. UNKNOWN fields) omitted, identity keys always kept."""
    if isinstance(rec, dict):
        return rec
    d = rec.model_dump(mode="json", exclude_defaults=True)
    for key in ("record_type", "quality_tier"):
        if hasattr(rec, key):
            d[key] = getattr(rec, key)
    return d


def write_jsonl(p: Path, records: Iterable[BaseModel | dict]) -> dict[str, Any]:
    lines = [json.dumps(dump_record(r), ensure_ascii=False, sort_keys=True) for r in records]
    data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    atomic_write_bytes(p, data)
    return {"path": rel(p), "sha256": sha256_bytes(data), "count": len(lines)}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_models(p: Path, model: type[BaseModel]) -> list:
    return [model.model_validate(d) for d in read_jsonl(p)]


def append_jsonl(p: Path, record: BaseModel | dict) -> None:
    """Append-only logs (runs, failures, review decisions). One line per event."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dump_record(record), ensure_ascii=False, sort_keys=True) + "\n")


@cache
def code_version() -> str:
    """Hash of pipeline code + config. Stands in for a git commit (repo is not under git)."""
    h = hashlib.sha256()
    files = sorted((ROOT / "services").rglob("*.py")) + sorted((ROOT / "scripts" / "data").glob("*.py"))
    files += sorted(CONFIG.glob("*.yaml"))
    for f in files:
        h.update(rel(f).encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return f"code-{h.hexdigest()[:12]}"


class Run:
    """Collects outputs, counts and failures for one stage execution (audit trail, spec §33)."""

    def __init__(self, stage: str, params: dict[str, Any] | None = None):
        self.stage = stage
        self.run_id = f"{stage}-{now_utc().strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
        self.params = params or {}
        self.started_at = now_utc()
        self.outputs: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.failures = 0
        self.notes: list[str] = []

    def output(self, info: dict[str, Any]) -> dict[str, Any]:
        self.outputs.append(info)
        return info

    def fail(
        self,
        category: FailureCategory,
        input_summary: str,
        expected: str,
        observed: str,
        *,
        severity: str = "MEDIUM",
        root_cause: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.failures += 1
        append_jsonl(
            RUNS / "failures.jsonl",
            FailureCase(
                failure_id=stable_id("fail", self.run_id, self.failures),
                category=category,
                stage=self.stage,
                run_id=self.run_id,
                timestamp=now_utc(),
                input=input_summary[:500],
                expected_behavior=expected,
                observed_behavior=observed[:1000],
                root_cause=root_cause,
                severity=severity,
                details=details or {},
            ),
        )

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        print(f"[{self.stage}] {msg}")


@contextmanager
def run_stage(stage: str, params: dict[str, Any] | None = None) -> Iterator[Run]:
    utf8_stdio()
    run = Run(stage, params)
    status = "FAILED"
    try:
        yield run
        status = "SUCCEEDED"
    finally:
        append_jsonl(
            RUNS / "runs.jsonl",
            {
                "record_type": "pipeline_run",
                "run_id": run.run_id,
                "stage": stage,
                "status": status,
                "started_at": run.started_at.isoformat(),
                "finished_at": now_utc().isoformat(),
                "code_version": code_version(),
                "params": run.params,
                "outputs": run.outputs,
                "counts": run.counts,
                "failures": run.failures,
                "notes": run.notes,
            },
        )
        print(f"[{stage}] {status} run_id={run.run_id} counts={run.counts} failures={run.failures}")
