"""Verify Nebius Token Factory model capabilities with minimal live requests (architecture audit 2026-10-02).

A capability is VERIFIED only when a live request demonstrates it; NOT_SUPPORTED when the provider rejects the
request shape (4xx) or the model ignores it; FAILED when it answers wrongly; UNTESTED otherwise. Names and
marketing are never evidence of a capability.

Writes data/manifests/model_capabilities.json (input to the model capability registry, ARCHITECTURE §19).
The key comes from NEBIUS_API_KEY and is never printed or stored. Run through the key wrapper:
    python <scratchpad>/with_nebius.py scripts/research/probe_models.py [--list] [--models id1,id2]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "manifests" / "model_capabilities.json"
PROBE_VERSION = "model-probe-0.1.0"
EMBED_HINTS = ("embed", "bge", "e5-")
JSON_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}, "required": ["answer"],
               "additionalProperties": False}
TOOL = {"type": "function", "function": {"name": "get_paper", "description": "Fetch metadata of an arXiv paper by id.",
                                         "parameters": {"type": "object", "properties": {"arxiv_id": {"type": "string"}},
                                                        "required": ["arxiv_id"]}}}
# ~2.4K-token stable prefix: if the provider caches prefixes, the 2nd identical request reports cached prompt tokens
CACHE_PREFIX = "ResearchX evidence policy. " + " ".join(f"Rule {i}: every claim needs a verbatim quote at its cited location." for i in range(220))


def red_png(n: int = 32) -> str:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * n for _ in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


class Probe:
    def __init__(self, client: httpx.Client):
        self.c = client

    def chat(self, model: str, messages: list, max_tokens: int = 400, **extra) -> tuple[int, dict, float]:
        t = time.time()
        r = self.c.post("chat/completions", json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra})
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:300]}
        return r.status_code, body, round(time.time() - t, 2)

    @staticmethod
    def content(body: dict) -> str:
        try:
            return (body["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def err(status: int, body: dict) -> str:
        e = body.get("error") if isinstance(body, dict) else None
        msg = (e.get("message") if isinstance(e, dict) else e) or body.get("detail") or body.get("raw") or ""
        return f"HTTP {status}: {str(msg)[:200]}"

    def run(self, model: str) -> dict:
        res: dict = {"model_id": model, "probe_version": PROBE_VERSION, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if any(h in model.lower() for h in EMBED_HINTS):
            t = time.time()
            r = self.c.post("embeddings", json={"model": model, "input": ["evidence-first research"]})
            ok = r.status_code == 200 and r.json().get("data")
            res["embedding"] = {"status": "VERIFIED" if ok else "FAILED", "dimensions": len(r.json()["data"][0]["embedding"]) if ok else None,
                                "latency_s": round(time.time() - t, 2), "detail": None if ok else self.err(r.status_code, r.json() if r.content else {})}
            return res
        thinking_off = {"chat_template_kwargs": {"enable_thinking": False}}
        # 1. plain chat (thinking switched off where the template supports it; retried without if rejected)
        s, b, lat = self.chat(model, [{"role": "user", "content": "Reply with exactly one word: ok"}], **thinking_off)
        extra = thinking_off
        if s != 200:
            s, b, lat = self.chat(model, [{"role": "user", "content": "Reply with exactly one word: ok"}])
            extra = {}
        txt = self.content(b)
        res["chat"] = {"status": "VERIFIED" if s == 200 and txt else "FAILED", "latency_s": lat, "usage": b.get("usage"),
                       "finish_reason": (b.get("choices") or [{}])[0].get("finish_reason") if s == 200 else None,
                       "detail": None if s == 200 else self.err(s, b)}
        res["thinking_toggle_accepted"] = bool(extra)
        if s != 200:
            return res
        # 2. structured output (json_schema)
        s, b, lat = self.chat(model, [{"role": "user", "content": "Is 7 a prime number? Answer as JSON."}], 300,
                              response_format={"type": "json_schema", "json_schema": {"name": "yes_no", "schema": JSON_SCHEMA, "strict": True}}, **extra)
        try:
            parsed = json.loads(self.content(b))
            valid = isinstance(parsed, dict) and set(parsed) == {"answer"} and parsed["answer"] in ("yes", "no")
        except ValueError:
            parsed, valid = None, False
        res["structured_output_json_schema"] = {"status": "VERIFIED" if s == 200 and valid else ("NOT_SUPPORTED" if 400 <= s < 500 else "FAILED"),
                                                "correct": bool(valid and parsed["answer"] == "yes"), "latency_s": lat,
                                                "detail": None if s == 200 else self.err(s, b)}
        # 3. tool calling
        s, b, lat = self.chat(model, [{"role": "user", "content": "Use the tool to fetch arXiv paper 2106.09685."}], 300,
                              tools=[TOOL], tool_choice="auto", **extra)
        calls = ((b.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [] if s == 200 else []
        good = bool(calls) and calls[0].get("function", {}).get("name") == "get_paper" and "2106.09685" in calls[0]["function"].get("arguments", "")
        res["tool_calling"] = {"status": "VERIFIED" if good else ("NOT_SUPPORTED" if (400 <= s < 500 or (s == 200 and not calls)) else "FAILED"),
                               "latency_s": lat, "detail": None if s == 200 else self.err(s, b)}
        # 4. vision (solid red 32x32 PNG)
        s, b, lat = self.chat(model, [{"role": "user", "content": [{"type": "text", "text": "What colour is this image? One word."},
                                                                   {"type": "image_url", "image_url": {"url": red_png()}}]}], 300, **extra)
        ans = self.content(b).lower()
        res["vision_input"] = {"status": "VERIFIED" if s == 200 and "red" in ans else ("NOT_SUPPORTED" if 400 <= s < 500 else "FAILED"),
                               "answer": ans[:40] if s == 200 else None, "latency_s": lat, "detail": None if s == 200 else self.err(s, b)}
        return res

    def prefix_cache(self, model: str, extra: dict) -> dict:
        usages = []
        for _ in range(2):
            s, b, _ = self.chat(model, [{"role": "system", "content": CACHE_PREFIX}, {"role": "user", "content": "Reply: ok"}], 20, **extra)
            usages.append(b.get("usage") if s == 200 else None)
        cached = (((usages[1] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")) if usages[1] else None
        return {"status": "VERIFIED" if cached else ("NOT_OBSERVED" if usages[1] else "FAILED"), "cached_tokens_second_call": cached,
                "prompt_tokens": (usages[1] or {}).get("prompt_tokens")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the live catalog (ids + metadata keys) and exit")
    ap.add_argument("--models", default="", help="comma-separated ids (default: whole catalog)")
    ap.add_argument("--cache-models", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", help="models for the prefix-cache probe")
    args = ap.parse_args()
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY not set")
    base = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["providers"]["nebius"]["base_url"]
    with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=180) as client:
        r = client.get("models")
        r.raise_for_status()
        catalog = {m["id"]: m for m in r.json()["data"]}
        if args.list:
            for mid, m in sorted(catalog.items()):
                print(mid, sorted(k for k in m if k != "id"))
            return
        ids = [m for m in args.models.split(",") if m] or sorted(catalog)
        probe = Probe(client)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(probe.run, ids))
        by_id = {r["model_id"]: r for r in results}
        for mid in [m for m in args.cache_models.split(",") if m in by_id]:
            by_id[mid]["prefix_cache"] = probe.prefix_cache(mid, {"chat_template_kwargs": {"enable_thinking": False}} if by_id[mid].get("thinking_toggle_accepted") else {})
        for mid, rec in by_id.items():
            rec["catalog_metadata"] = {k: v for k, v in catalog.get(mid, {}).items() if k != "id"}
    OUT.write_text(json.dumps({"provider": "nebius", "base_url": base, "probe_version": PROBE_VERSION, "catalog_size": len(catalog),
                               "models": [by_id[k] for k in sorted(by_id)]}, indent=1, sort_keys=True), encoding="utf-8")
    for rec in sorted(by_id.values(), key=lambda r: r["model_id"]):
        caps = {k: rec[k]["status"] for k in ("chat", "structured_output_json_schema", "tool_calling", "vision_input", "embedding", "prefix_cache") if k in rec}
        print(rec["model_id"], caps)


if __name__ == "__main__":
    main()
