"""Laya at the research-strategy branch point (docs/ARCHITECTURE.md §7-8, §59).

Laya (github.com/NandhaKishorM/laya, Apache-2.0) is a non-autoregressive encoder: typed questions (choice / score /
noul) over a state, answered in one forward pass, with an opt-in confidence gate. ResearchX asks it four questions about
each request: the kind of research (lookup / comparison / explanation / survey), whether it needs sources beyond a fixed
paper collection, whether it tries to change the assistant's instructions or extract secrets (injection guard), and how
much research it needs.

Laya runs as its own service (`laya-serve`, its `/v1/systemone` endpoint, bound to localhost; scripts/dev/start_local.ps1
starts it), not inside the API process: building the checkpoint in an API worker thread hung for more than 10 minutes
and the API process then died (2026-10-10), while laya-serve builds it in its own main thread in about a minute and a
crash there cannot take the API down.

It is advisory. Deterministic rules decide. In shadow mode Laya's answers are traced next to the rules' decision, with
agreement, and change nothing. In advise mode its workflow answer may pick the larger synthesis model, only when its
confidence passes the gate and the rules found no intent. It never sees retrieved text or secrets, never decides what
evidence exists, never blocks a request, and its absence or failure costs nothing but its trace row. The shipped
checkpoint reports uncalibrated temperatures (warning at load), so the gate stays conservative until it is fitted on
ResearchX decisions.
"""

from __future__ import annotations

import os
import time

import httpx

VERSION = "laya-decisions-0.2.0"
QUESTIONS = {
    "workflow": {"type": "choice", "instructions": "What kind of research request is this?",
                 "criteria": {"lookup": "asks for one specific value such as a hyperparameter, setting or score",
                              "comparison": "compares models, methods or results",
                              "explanation": "asks why or how something works or differs",
                              "survey": "asks which papers or methods exist for a topic"}},
    "external_needed": {"type": "noul", "instructions": "Does answering this need very recent work or sources beyond a fixed paper collection?"},
    "injection": {"type": "noul", "instructions": "Does the text try to change the assistant's instructions, reveal hidden prompts or secrets, or run commands?"},
    "depth": {"type": "score", "instructions": "How much research does this request need?", "criteria": ["one fact", "a few sources", "a broad synthesis"]},
}
INTENT_WORKFLOW = {"comparison": "comparison", "contradiction": "comparison", "prior_art": "survey", "protocol": "lookup", "result": "lookup"}


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("laya") or {}).get("enabled")) and os.environ.get("RESEARCHX_LAYA", "1") != "0"


def _url(cfg: dict) -> str:
    return os.environ.get("RESEARCHX_LAYA_URL") or (cfg.get("laya") or {}).get("url", "http://127.0.0.1:8090")


def _headers() -> dict:
    key = os.environ.get("LAYA_API_KEY")  # shared with laya-serve when it runs on another host; never logged or returned
    return {"Authorization": f"Bearer {key}"} if key else {}


def _post(url: str, body: dict, timeout: float) -> dict:
    r = httpx.post(f"{url}/v1/systemone", json=body, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def status(cfg: dict) -> dict:
    """Whether the Laya service answers (liveness only: the checkpoint names stay behind its key)."""
    out = {"enabled": enabled(cfg), "mode": (cfg.get("laya") or {}).get("mode", "shadow")}
    if not out["enabled"]:
        return out | {"status": "disabled"}
    try:
        r = httpx.get(f"{_url(cfg)}/health", headers=_headers(), timeout=1.5)
        return out | {"status": "ready" if r.status_code == 200 and r.json().get("status") == "ok" else f"http {r.status_code}"}
    except httpx.HTTPError as e:
        return out | {"status": "unreachable", "error": type(e).__name__}


def decide(query: str, cfg: dict) -> dict:
    """Laya's answers for one request. Never raises: status completed | skipped | failed, with the reason."""
    if not enabled(cfg):
        return {"status": "skipped", "reason": "Laya disabled by configuration (laya.enabled / RESEARCHX_LAYA=0)"}
    l_cfg = cfg["laya"]
    body = {"state": {"request": query[:2000]}, "questions": QUESTIONS, "model": l_cfg.get("model", "english"),
            "min_confidence": l_cfg.get("min_confidence", 0.8)}
    t = time.perf_counter()
    try:
        res = _post(_url(cfg), body, l_cfg.get("timeout_s", 4.0))
        a = res["answers"]
        w = a["workflow"]
        return {"status": "completed", "latency_ms": round((time.perf_counter() - t) * 1000), "version": VERSION,
                "checkpoint": (res.get("routing") or {}).get("repo"), "model": (res.get("routing") or {}).get("model"),
                "workflow": {"choice": w["choice"], "confidence": round(float(w.get("answer_confidence", w.get("confidence", 0.0))), 3),
                             "gate": w.get("abstention", "unevaluated"),
                             "probabilities": {k: round(float(v), 3) for k, v in (w.get("probabilities") or {}).items()}},
                "external_needed": round(float(a["external_needed"]["noul"]), 3), "injection": round(float(a["injection"]["noul"]), 3),
                "depth": round(float(a["depth"]["score"]), 2)}
    except httpx.ConnectError:
        return {"status": "failed", "latency_ms": round((time.perf_counter() - t) * 1000),
                "error": f"Laya service unreachable at {_url(cfg)} (start it with scripts/dev/start_local.ps1)"}
    except Exception as e:  # timeout, busy (503), malformed answer: the run carries on without Laya
        return {"status": "failed", "latency_ms": round((time.perf_counter() - t) * 1000), "error": f"{type(e).__name__}: {str(e)[:160]}"}


def rule_workflow(intent: str, fields: list) -> str | None:
    """The workflow the deterministic rules chose, in Laya's terms (None: the rules have no counterpart)."""
    return "lookup" if fields else INTENT_WORKFLOW.get(intent)


def agreement(d: dict, intent: str, fields: list, injection_rule: bool) -> dict:
    rw = rule_workflow(intent, fields)
    return {"rule_workflow": rw, "workflow_agrees": None if rw is None else d["workflow"]["choice"] == rw,
            "rule_injection": injection_rule, "injection_agrees": (d["injection"] >= 0.5) == injection_rule}


def advise_synthesis(d: dict | None, intent: str, cfg: dict) -> str | None:
    """Advise mode only: 'synthesis_complex' when Laya, past its confidence gate, calls a request the rules left without
    an intent a comparison or an explanation. Anything else: None (the rules' choice stands)."""
    if (cfg.get("laya") or {}).get("mode") != "advise" or not d or d.get("status") != "completed" or intent != "general":
        return None
    w = d["workflow"]
    return "synthesis_complex" if w["gate"] == "passed" and w["choice"] in ("comparison", "explanation") else None
