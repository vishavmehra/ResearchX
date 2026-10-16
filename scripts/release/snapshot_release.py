"""Snapshot a frozen dataset release into data/releases/<version>/ (DATASET_SPEC §53).

Deterministic archives (sorted members, zeroed mtime/owner, no gzip timestamp): same inputs, same bytes.

  code.tar.gz      pipeline code + config + tests + lockfile + docs; restores the exact processing version
  data.tar.gz      the release's versioned dataset files (normalized, silver, gold, annotations, schemas,
                   manifests, reports, runs, eval without eval/external)
  internal/raw_cache.tar.gz, internal/fulltext.tar.gz
                   acquisition cache and full text needed for offline replay. Third-party content:
                   internal use only, never redistributed (gitignored).

Refuses to snapshot unless (a) the code hash recomputed from the code archive equals the manifest's
processing_commit and (b) every checksum in the manifest matches the member inside data.tar.gz.
This lives outside scripts/data/ on purpose: packaging does not change outputs, so it is not part of
the processing-code hash.

Usage: uv run python scripts/release/snapshot_release.py --version 0.1.0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.datasets.store import DATA, ROOT, now_utc, sha256_file, utf8_stdio  # noqa: E402

CODE = ["services", "scripts", "config", "tests", "pyproject.toml", "uv.lock", "CLAUDE.md", "docs", "planning"]
DATASET = ["normalized", "silver", "gold", "eval", "annotations", "schemas", "manifests", "reports", "runs", "README.md"]
INTERNAL = {"raw_cache": ["raw"], "fulltext": ["fulltext", "restricted"]}
SKIP_PARTS = {"__pycache__", ".pytest_cache", "external"}  # eval/external = evaluation-only benchmark content


def code_hash(root: Path) -> str:
    """Mirror of store.code_version() over an arbitrary root (services/**, scripts/data/*.py, config/*.yaml)."""
    h = hashlib.sha256()
    files = sorted((root / "services").rglob("*.py")) + sorted((root / "scripts" / "data").glob("*.py"))
    files += sorted((root / "config").glob("*.yaml"))
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return f"code-{h.hexdigest()[:12]}"


def members(base: Path, names: list[str], exclude_release_dir: bool = True) -> list[tuple[Path, str]]:
    out = []
    for name in names:
        p = base / name
        if p.is_file():
            out.append((p, p.relative_to(ROOT).as_posix()))
        elif p.is_dir():
            for f in p.rglob("*"):
                if not f.is_file() or SKIP_PARTS & set(f.relative_to(base).parts):
                    continue
                if exclude_release_dir and "releases" in f.relative_to(ROOT).parts[:2]:
                    continue  # never nest data/releases inside a snapshot
                out.append((f, f.relative_to(ROOT).as_posix()))
    return sorted(out, key=lambda t: t[1])


def write_archive(path: Path, files: list[tuple[Path, str]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for src, arc in files:
            info = tarfile.TarInfo(arc)
            info.size = src.stat().st_size
            info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = 0, 0o644, 0, 0, "", ""
            with open(src, "rb") as fh:
                tar.addfile(info, fh)
    path.write_bytes(buf.getvalue())
    return {"file": path.name if path.parent.name != "internal" else f"internal/{path.name}", "members": len(files), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(version_dir: Path, manifest: dict) -> dict:
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(version_dir / "code.tar.gz") as tar:
            tar.extractall(tmp, filter="data")
        result["code_hash_from_archive"] = code_hash(Path(tmp))
    result["code_hash_expected"] = manifest["processing_commit"]
    result["code_hash_ok"] = result["code_hash_from_archive"] == result["code_hash_expected"]
    mismatched = []
    with tarfile.open(version_dir / "data.tar.gz") as tar:
        for entry in manifest["checksums"]:
            try:
                data = tar.extractfile(entry["path"]).read()
            except KeyError:
                mismatched.append(("missing", entry["path"]))
                continue
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                mismatched.append(("changed", entry["path"]))
    result["manifest_files_checked"] = len(manifest["checksums"])
    result["manifest_files_mismatched"] = mismatched
    result["data_ok"] = not mismatched
    return result


def main() -> None:
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--no-internal", action="store_true", help="skip raw cache / full-text archives")
    args = ap.parse_args()
    manifest_path = DATA / "manifests" / "releases" / f"v{args.version}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = DATA / "releases" / f"v{args.version}"
    if (out / "snapshot.json").exists():
        sys.exit(f"{out} already holds a snapshot; releases are immutable (delete it by hand only with a reason)")
    live = code_hash(ROOT)
    if live != manifest["processing_commit"]:
        sys.exit(f"live code {live} != release {manifest['processing_commit']}: restore that code before snapshotting")
    archives = [
        write_archive(out / "code.tar.gz", members(ROOT, CODE)),
        write_archive(out / "data.tar.gz", members(DATA, DATASET)),
    ]
    if not args.no_internal:
        for name, dirs in INTERNAL.items():
            files = members(DATA, dirs)
            if files:
                archives.append(write_archive(out / "internal" / f"{name}.tar.gz", files))
    checks = verify(out, manifest)
    if not (checks["code_hash_ok"] and checks["data_ok"]):
        sys.exit(f"snapshot verification FAILED: {checks}")
    snap = {
        "record_type": "release_snapshot",
        "dataset_version": manifest["dataset_version"],
        "schema_version": manifest["schema_version"],
        "processing_commit": manifest["processing_commit"],
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "snapshotted_at": now_utc().isoformat(),
        "archives": archives,
        "verification": checks,
    }
    (out / "snapshot.json").write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join(f"{a['sha256']}  {a['file']}\n" for a in archives), encoding="utf-8")
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
