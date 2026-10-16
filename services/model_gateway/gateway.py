"""Model gateway (ARCHITECTURE §2.2, §16, §19; API_SPEC §13-14).

Application code never calls a provider directly: it asks the gateway for a *task*
(e.g. experiment_extraction). Routing, model ids, decoding parameters and structured-output
mode live in config/models.yaml. Candidate model names are resolved against the provider's
live catalog (authenticated GET /models) because exact ids are only visible to an
authenticated account. Secrets come from environment variables and never reach logs,
call records, cache keys or prompts. Call records store hashes and sizes, not prompt text.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import httpx

from services.datasets.store import RUNS, append_jsonl, config, now_utc, sha256_bytes


HEALTH_FAILURES, HEALTH_WINDOW_S = 3, 300


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class ModelCall:
    call_id: str
    task: str
    provider: str
    model_requested: list[str]
    model_resolved: str | None
    prompt_version: str
    request_sha256: str
    started_at: str
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: str = "PENDING"  # OK | HTTP_ERROR | INVALID_JSON | REFUSED | UNAVAILABLE
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1) if m else text


def _repair_truncated(text: str, max_attempts: int = 400) -> dict | None:
    """Salvage `{"key": [ {...}, {...}, <cut` by cutting after the last complete item and closing brackets.

    Only complete items survive; nothing is invented. Returns None when no prefix parses."""
    ends = [i for i, ch in enumerate(text) if ch == "}"]
    for i in reversed(ends[-max_attempts:]):
        for closing in ("]}", "}]}", "]}]}", "}"):
            try:
                parsed = json.loads(text[: i + 1] + closing)
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else None
    return None


class ModelGateway:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path | None = RUNS / "model_calls.jsonl",
    ):
        self.cfg = cfg or config("models")
        self.transport = transport
        self.env = os.environ if env is None else env
        self.log_path = log_path
        self._catalog: dict[str, list[str]] = {}
        self._clients: dict[tuple[str, float], httpx.Client] = {}
        self._clients_lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}  # task -> times of recent transport/HTTP failures (healthy)

    def available(self, task: str) -> bool:
        provider = self.cfg["tasks"][task]["provider"]
        return bool(self.env.get(self.cfg["providers"][provider]["api_key_env"]))

    def healthy(self, task: str) -> bool:
        """False after HEALTH_FAILURES transport/HTTP failures of a task within HEALTH_WINDOW_S and no success since: the
        router then skips the task's route (circuit open) until the window passes. Invalid outputs do not count."""
        recent = [t for t in self._failures.get(task, []) if time.monotonic() - t < HEALTH_WINDOW_S]
        self._failures[task] = recent
        return len(recent) < HEALTH_FAILURES

    def _client(self, provider: str, timeout_s: float | None = None) -> httpx.Client:
        """One pooled client per provider and timeout, kept for the gateway's life: a fresh client paid a TLS handshake
        on every call (Nebius embedding 570 ms per call with a fresh client, 270 ms reused; 2026-10-09). httpx clients are
        thread-safe. timeout_s: a task's own budget (interactive runtime tasks) instead of the provider default (batch jobs)."""
        p = self.cfg["providers"][provider]
        key = self.env.get(p["api_key_env"])
        if not key:
            raise ModelUnavailable(f"{p['api_key_env']} is not set")
        timeout = timeout_s or p.get("timeout_s", 180)
        with self._clients_lock:
            client = self._clients.get((provider, timeout))
            if client is None or client.is_closed:
                client = self._clients[(provider, timeout)] = httpx.Client(
                    base_url=p["base_url"], headers={"Authorization": f"Bearer {key}"}, timeout=timeout, transport=self.transport,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=8, keepalive_expiry=60))
        return client

    def resolve_model(self, task: str) -> tuple[str, str]:
        t = self.cfg["tasks"][task]
        provider = t["provider"]
        if provider not in self._catalog:
            ids: list[str] = []
            if self.cfg["providers"][provider].get("list_models"):
                try:
                    r = self._client(provider).get("models")
                    r.raise_for_status()
                    ids = [m["id"] for m in r.json().get("data", [])]
                except (httpx.HTTPError, ValueError, KeyError):
                    ids = []
            self._catalog[provider] = ids
        ids = self._catalog[provider]
        for cand in t["candidates"]:
            if not ids:
                return provider, cand  # catalog unavailable: use configured id as-is
            if cand in ids:
                return provider, cand
            fuzzy = sorted(i for i in ids if i.lower() == cand.lower() or i.lower().endswith("/" + cand.lower()))
            if fuzzy:
                return provider, fuzzy[0]
        raise ModelUnavailable(f"none of {t['candidates']} found in the {provider} catalog")

    def complete_json(self, task: str, system: str, user: str, schema: dict[str, Any], prompt_version: str) -> tuple[dict | None, ModelCall]:
        t = self.cfg["tasks"][task]
        call = ModelCall(
            call_id=str(uuid.uuid4()),
            task=task,
            provider=t["provider"],
            model_requested=list(t["candidates"]),
            model_resolved=None,
            prompt_version=prompt_version,
            request_sha256=sha256_bytes((system + "\x00" + user).encode("utf-8")),
            started_at=now_utc().isoformat(),
            extra={"user_chars": len(user)},
        )
        try:
            provider, model = self.resolve_model(task)
        except ModelUnavailable as e:
            call.status, call.error = "UNAVAILABLE", str(e)
            self._log(call)
            return None, call
        call.model_resolved = model
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": t.get("temperature", 0.0),
            "max_tokens": t.get("max_tokens", 4096),
        }
        mode = t.get("structured_output", "json_schema")
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": task, "schema": schema}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        body.update(t.get("extra_body") or {})  # e.g. chat_template_kwargs.enable_thinking=false (config)
        t0 = time.monotonic()
        data = None
        try:
            client = self._client(provider, t.get("timeout_s"))
        except ModelUnavailable as e:
            call.status, call.error = "UNAVAILABLE", str(e)
            self._log(call)
            return None, call
        data = self._post(client, "chat/completions", body, call, t.get("max_retries", 3))
        call.latency_s = round(time.monotonic() - t0, 3)
        if data is None:
            self._log(call)
            return None, call
        usage = data.get("usage") or {}
        call.input_tokens, call.output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        message = (data.get("choices") or [{}])[0].get("message") or {}
        if message.get("refusal"):
            call.status, call.error = "REFUSED", "model refusal"
            self._log(call)
            return None, call
        content = _strip_fences(message.get("content") or "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            parsed = _repair_truncated(content)  # output cut at max_tokens: keep the complete leading items
            if parsed is None:
                call.status, call.error = "INVALID_JSON", str(e)[:200]
                self._log(call)
                return None, call
            call.extra["repaired_truncated_json"] = True
        if not isinstance(parsed, dict):  # every task's schema is an object: a list, string or number is unusable
            call.status, call.error = "INVALID_SHAPE", f"top-level JSON is a {type(parsed).__name__}, not an object"
            self._log(call)
            return None, call
        call.status, call.error = "OK", None
        self._log(call)
        return parsed, call

    def embed(self, task: str, texts: list[str], query: bool = False) -> tuple[list[list[float]] | None, ModelCall]:
        """Embed documents, or queries (the task's query_instruction is prefixed). Documents and queries must use the
        same task so model and dimensions match; vectors follow the order of `texts`."""
        t = self.cfg["tasks"][task]
        inputs = [t["query_instruction"] + x for x in texts] if query and t.get("query_instruction") else list(texts)
        call = ModelCall(call_id=str(uuid.uuid4()), task=task, provider=t["provider"], model_requested=list(t["candidates"]),
                         model_resolved=None, prompt_version=t.get("embedding_version", task),
                         request_sha256=sha256_bytes("\x00".join(inputs).encode("utf-8")), started_at=now_utc().isoformat(),
                         extra={"texts": len(texts), "dimensions": t.get("dimensions"), "query": query})
        try:
            provider, model = self.resolve_model(task)
            client = self._client(provider, t.get("timeout_s"))
        except ModelUnavailable as e:
            call.status, call.error = "UNAVAILABLE", str(e)
            self._log(call)
            return None, call
        call.model_resolved = model
        body = {"model": model, "input": inputs, **({"dimensions": t["dimensions"]} if t.get("dimensions") else {})}
        t0 = time.monotonic()
        data = self._post(client, "embeddings", body, call, t.get("max_retries", 3))
        call.latency_s = round(time.monotonic() - t0, 3)
        vectors = [d["embedding"] for d in sorted((data or {}).get("data") or [], key=lambda d: d["index"])]
        if data is not None and len(vectors) != len(texts):
            call.status, call.error, data = "INVALID_RESPONSE", f"{len(vectors)} vectors for {len(texts)} texts", None
        if data is not None:
            call.input_tokens = (data.get("usage") or {}).get("prompt_tokens")
            call.status, call.error = "OK", None
        self._log(call)
        return (vectors if data is not None else None), call

    @staticmethod
    def _post(client: httpx.Client, path: str, body: dict, call: ModelCall, retries: int) -> dict | None:
        """POST with backoff on transport errors, 429 and 5xx; other 4xx fail at once. Sets call.status/error on failure."""
        for attempt in range(retries):
            try:
                r = client.post(path, json=body)
            except httpx.TransportError as e:
                call.status, call.error = "HTTP_ERROR", type(e).__name__
                time.sleep(2 * 2**attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                call.status, call.error = "HTTP_ERROR", f"HTTP {r.status_code}"
                time.sleep(2 * 2**attempt)
                continue
            if r.status_code >= 400:
                call.status, call.error = "HTTP_ERROR", f"HTTP {r.status_code}"
                return None
            return r.json()
        return None

    def _log(self, call: ModelCall) -> None:
        if call.status in ("HTTP_ERROR", "UNAVAILABLE"):  # ponytail: per-process health, shared state if workers multiply
            self._failures.setdefault(call.task, []).append(time.monotonic())
        elif call.status == "OK":
            self._failures.pop(call.task, None)
        if self.log_path is not None:
            with _LOG_LOCK:  # extraction runs calls from several threads
                append_jsonl(self.log_path, {"record_type": "model_call", **asdict(call)})


_LOG_LOCK = threading.Lock()
