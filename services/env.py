"""Environment loading: `.env` then `.env.local` for local development; production sets real environment variables
(deployment secret store). Existing environment variables always win, so the same code runs everywhere.
Values are never logged. Stdlib only (no python-dotenv dependency).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")
_loaded = False


def _value(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]  # single quotes: literal
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return bytes(raw[1:-1], "utf-8").decode("unicode_escape")
    return raw.split(" #", 1)[0].strip()  # unquoted: strip trailing comment


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = _value(m.group(2))
    return out


def load_env(root: Path = ROOT) -> None:
    """Fill missing os.environ entries from .env and .env.local (local overrides .env); never overrides the real env."""
    global _loaded
    if _loaded:
        return
    merged: dict[str, str] = {}
    for name in (".env", ".env.local"):
        p = root / name
        if p.is_file():
            merged.update(parse(p.read_text(encoding="utf-8")))
    for k, v in merged.items():
        if v and k not in os.environ:
            os.environ[k] = v
    _loaded = True


def require(name: str) -> str:
    load_env()
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing environment variable {name} (see .env.example)")
    return v
