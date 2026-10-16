"""Model routing policy (docs/ARCHITECTURE.md §67, docs/DECISIONS.md D-28; routing decision contract).

Every generative model call of a research run goes through a RoutingDecision: a validated RoutingContext (task type,
complexity, sizes, verification and comparison needs, latency budget, privacy scope, retry count) goes in; the route
(a task of config/models.yaml), the model it resolves to, the reason and the fallback come out. The policy is
deterministic code. Nothing a user sends can name a model or a route (the context has no such field and rejects
unknown ones). Laya reaches it only in the optional advise mode (off by default), and then only as the synthesis
complexity hint, which the policy below treats like a comparison question. Embeddings (Qwen3-Embedding-8B) are a
separate route and are never selectable for generation.

Policy routing-0.1.0: the cheapest suitable model first; escalation only for complexity, verification or a failed output.
  reranking            Nano. No model fallback: the hybrid (RRF) order is kept.
  synthesis            Nano; Super for comparison or contradiction questions. Recorded contradictions in the context do not
                       escalate: they are checked deterministically (rx.contradictions, CONFLICTING status), and in the
                       routing benchmark (2026-10-14) Super abstained on 2 such questions whose Nano claims all validated.
                       The other tier once when the output fails schema or citation validation and the latency budget
                       has room for it. Last resort: the strongest passages quoted verbatim.
  extraction           Nano over stored corpus passages; Super (reasoning) once when Nano's output is malformed.
  fulltext_extraction  Super with reasoning (2026-10-14: Nano returned values without verbatim spans on full texts, all
                       rejected); Nano once when Super finds nothing. Last resort: the fields stay unknown.
Ultra is configured (difficult_verification) but never selected: its benefit has not been measured. A long prompt alone
never escalates. A route that is not configured, has no key, failed repeatedly in the last minutes, or whose provider is
not approved for private documents (when the context holds them) is skipped, and its fallback takes over.
Modes static / fixed_nano / fixed_super reproduce the rules before routing-0.1.0 and the fixed baselines for
scripts/eval/bench_routing.py; the API runs dynamic (config/runtime.yaml routing.mode).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.model_gateway.gateway import ModelUnavailable

POLICY_VERSION = "routing-0.1.0"
MODES = ("dynamic", "static", "fixed_nano", "fixed_super")
# task type -> (Nano-tier route, Super-tier route or None, deterministic last resort)
ROUTES: dict[str, tuple[str, str | None, str]] = {
    "reranking": ("reranking", None, "hybrid (RRF) order kept"),
    "synthesis": ("synthesis", "synthesis_complex", "extractive answer: the strongest passages quoted verbatim"),
    "extraction": ("field_extraction", "fulltext_extraction", "the fields stay unknown (partial answer or abstention)"),
    "fulltext_extraction": ("field_extraction", "fulltext_extraction", "the fields stay unknown (partial answer or abstention)"),
    "fallback_answer": ("synthesis", None, "a failure message without an answer"),
}


class RoutingContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_type: Literal["reranking", "synthesis", "extraction", "fulltext_extraction", "fallback_answer"]
    complexity: Literal["low", "medium", "high"] = "low"
    input_tokens: int = Field(ge=0)  # estimated: prompt characters / 4
    context_required: int = Field(default=0, ge=0)  # sources or passages in the prompt
    structured_output: bool = True
    verification_required: bool = False
    comparison_count: int = Field(default=0, ge=0)
    evidence_sufficiency: Literal["sufficient", "insufficient", "unknown"] = "unknown"
    latency_budget_ms: int = Field(gt=0)
    privacy_scope: Literal["public", "project-private", "user-private"] = "public"
    retry_count: int = Field(default=0, ge=0)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str
    task_type: str
    selected_route: str | None
    selected_model: str | None
    provider: str | None
    selection_reason: str
    complexity: str
    input_tokens: int
    latency_budget_ms: int
    fallback_route: str | None
    fallback_model: str | None
    fallback_strategy: str
    structured_output_required: bool
    verification_required: bool
    privacy_scope: str
    mode: str
    routing_policy_version: str = POLICY_VERSION


def typical_ms(gw, route: str) -> int:
    """Measured typical latency of a route (config/models.yaml typical_ms); 0 when unmeasured."""
    return int(((getattr(gw, "cfg", None) or {}).get("tasks", {}).get(route) or {}).get("typical_ms", 0))


def usable(gw, route: str, privacy_scope: str = "public") -> tuple[bool, str]:
    """Whether `route` can serve this call: a generative task of the configuration, its provider approved for private
    documents when the context holds them, its key set, healthy, and resolvable in the provider catalog."""
    cfg = getattr(gw, "cfg", None)
    if cfg is not None:
        t = cfg["tasks"].get(route)
        if t is None or t.get("embedding_version") or t.get("dimensions"):
            return False, "not a generative route"
        if privacy_scope != "public" and not cfg["providers"][t["provider"]].get("private_data"):
            return False, f"provider {t['provider']} is not approved for private documents"
        if hasattr(gw, "available") and not gw.available(route):
            return False, "provider key not configured"
    if not getattr(gw, "healthy", lambda _r: True)(route):
        return False, "repeated failures in the last minutes (circuit open)"
    try:
        gw.resolve_model(route)
    except ModelUnavailable as e:
        return False, str(e)[:160]
    return True, ""


def decide(ctx: RoutingContext, gw, mode: str = "dynamic") -> RoutingDecision:
    if mode not in MODES:
        raise ValueError(f"unknown routing mode {mode!r}")
    small, large, last = ROUTES[ctx.task_type]
    why: list[str] = []
    if large is None:
        order = [small]
        why.append(f"{small}: the only model route for {ctx.task_type}")
    elif mode in ("fixed_nano", "fixed_super"):
        order = [small if mode == "fixed_nano" else large]
        why.append(f"{mode.replace('_', ' ')} (benchmark baseline)")
    elif ctx.task_type == "synthesis":
        hard = ctx.complexity == "high"
        order = [large, small] if hard else [small, large]
        why.append("Super: comparison or contradiction question" if hard else f"Nano: routine synthesis over {ctx.context_required} source(s)")
        if not hard and ctx.verification_required:  # measured: Super abstained where Nano's claims all validated (routing bench 2026-10-14)
            why.append("recorded contradictions are checked deterministically: no escalation")
        if not hard and ctx.input_tokens > 2000:
            why.append(f"{ctx.input_tokens} input tokens: length alone does not escalate")
        order = order if mode == "dynamic" else order[:1]
    elif ctx.task_type == "fulltext_extraction":
        order = [large, small]  # the Nano fallback predates routing-0.1.0, so static keeps it too
        why.append("Super with reasoning: full-text passages (Nano returned values without verbatim spans on full texts)")
    else:
        order = [small, large] if mode == "dynamic" else [small]
        why.append("Nano: short stored passages, routine structured extraction")
    ok = []
    for r in order:
        good, reason = usable(gw, r, ctx.privacy_scope)
        if good:
            ok.append(r)
        else:
            why.append(f"{r} skipped: {reason}")
    if mode == "dynamic" and ctx.task_type == "synthesis" and len(ok) == 2 and typical_ms(gw, ok[0]) > ctx.latency_budget_ms >= typical_ms(gw, ok[1]):
        why.append(f"{ok[0]} typically takes {typical_ms(gw, ok[0])} ms, over the {ctx.latency_budget_ms} ms budget")
        ok.reverse()
    sel, fb = (ok + [None, None])[:2]
    model = lambda r: gw.resolve_model(r)[1] if r else None  # noqa: E731
    cfg = getattr(gw, "cfg", None)
    if sel is None:
        why.append(f"no usable model: {last}")
    return RoutingDecision(
        decision_id=str(uuid.uuid4()), task_type=ctx.task_type, selected_route=sel, selected_model=model(sel),
        provider=cfg["tasks"][sel]["provider"] if cfg is not None and sel else None, selection_reason="; ".join(why),
        complexity=ctx.complexity, input_tokens=ctx.input_tokens, latency_budget_ms=ctx.latency_budget_ms,
        fallback_route=fb, fallback_model=model(fb),
        fallback_strategy=(f"{fb} once if the output of {sel} fails validation and the latency budget has room; then {last}" if fb else last),
        structured_output_required=ctx.structured_output, verification_required=ctx.verification_required,
        privacy_scope=ctx.privacy_scope, mode=mode)


def execute(gw, d: RoutingDecision, attempt: Callable[[str], tuple[Any, Any, int]]) -> tuple[Any, list[dict]]:
    """Run decision `d`: its route, then its fallback once when the first output is not accepted and the latency budget
    still has room for the fallback's typical latency. attempt(route) -> (result, call, quality): 2 accepted, 1 output
    parsed but rejected by validation, 0 no usable output. Returns the best result (highest quality, earliest first) and
    one trace record per attempt or skip. Model calls are reads, so a second attempt duplicates no side effect."""
    t0, best, out = time.monotonic(), (None, -1), []
    for n, route in enumerate([r for r in (d.selected_route, d.fallback_route) if r], 1):
        spent = round((time.monotonic() - t0) * 1000)
        if n > 1 and spent + typical_ms(gw, route) > d.latency_budget_ms:
            out.append({"attempt": n, "route": route, "event": "fallback skipped", "reason":
                        f"latency budget: {spent} ms spent + ~{typical_ms(gw, route)} ms typical > {d.latency_budget_ms} ms"})
            break
        t = time.monotonic()
        result, call, quality = attempt(route)
        out.append({"attempt": n, "route": route, "model": call.model_resolved, "event": "selected" if n == 1 else "fallback",
                    "status": call.status, "error": call.error, "latency_ms": round((time.monotonic() - t) * 1000),
                    "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
                    "outcome": ("no usable output", "rejected by validation", "accepted")[quality]})
        if quality > best[1]:
            best = (result, quality)
        if quality == 2:
            break
    return best[0], out


def trace(d: RoutingDecision, attempts: list[dict]) -> dict:
    """The decision and its execution as one trace detail (model used, attempts, fallbacks, tokens; no cost: the
    configuration holds no verified provider pricing)."""
    used = next((a for a in reversed(attempts) if a.get("outcome") == "accepted"), next((a for a in reversed(attempts) if "model" in a), None))
    return {**d.model_dump(), "attempts": attempts, "model_used": used["model"] if used else None,
            "fallback_used": any(a.get("event") == "fallback" for a in attempts)}
