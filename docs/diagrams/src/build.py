"""The ResearchX diagram set (docs/diagrams/*.drawio). Run: python docs/diagrams/src/build.py [slug ...]
then node docs/diagrams/src/render.mjs to export PNG and SVG with the draw.io viewer and check the rendered layout.

Content follows the implementation (services/, apps/web, config/, supabase/migrations); anything not built or not verified
is labelled as such. Geometry is hand-placed on a 4 px grid and every connector has explicit waypoints, so the files can
be adjusted by hand in draw.io without fighting an auto-layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawkit import PAPER, SURFACE2, Diagram  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
DIAGRAMS = {}
W = 1360  # every diagram shares the width, so they scale alike in the README


def diagram(fn):
    DIAGRAMS[fn.__name__.replace("_", "-")] = fn
    return fn


# ------------------------------------------------------------------------------------------------ 01
@diagram
def architecture():
    d = Diagram("01-system-architecture", "ResearchX system architecture",
                "What runs where, as implemented. Diagram 11 shows the public deployment.", W, 1032)
    for name, y, h in [("CLIENT", 112, 136), ("API", 280, 136), ("SERVICES", 448, 160), ("DATA ·\nSUPABASE", 640, 144),
                       ("EXTERNAL\nSERVICES", 816, 152)]:
        d.zone(32, y, 1296, h, name)
    d.node("web", 448, 128, 536, 104, "Web app", "Research chat · Knowledge Map · Analyze (eight analyses) · private library · settings",
           tag="NEXT.JS 16 · REACT 19 · apps/web")
    d.node("auth", 1080, 128, 192, 104, "Authentication", "E-mail · Google · GitHub · TOTP MFA", tag="SUPABASE AUTH")
    d.node("api", 448, 296, 536, 104, "API gateway",
           "Verifies the session token with Supabase Auth · runs user queries under RLS · rate limits · streams trace events (SSE)",
           tag="FASTAPI · services/api")
    d.node("laya", 152, 464, 208, 128, "Laya decision layer", "Traced beside each run", tag="LAYA-SERVE :8090", kind="advisory")
    d.badge(164, 546, "SHADOW MODE", "warn", w=184, sub="advisory only, never applied")
    d.node("orch", 448, 464, 240, 128, "Research orchestrator",
           "Deterministic pipeline: retrieve, rerank, ground, synthesise, validate citations, abstain", tag="services/runtime", kind="focal")
    d.node("router", 768, 464, 216, 128, "Model router", "Task-aware routes · health circuit · one validated fallback within the latency budget",
           tag="routing-0.1.0")
    d.node("analyses", 1080, 464, 192, 128, "Eight analyses", "Shared corpus index · configuration identity · comparison engine",
           tag="services/analysis")
    d.node("pg", 448, 656, 240, 112, "Postgres · schema rx", "Corpus · traces · caches · pgvector · RLS", tag="SUPABASE", kind="store")
    d.node("storage", 1080, 656, 192, 112, "Private library", "Bucket user-documents, one folder per user", tag="SUPABASE STORAGE", kind="store")
    d.node("providers", 152, 832, 208, 120, "Evidence providers", "Semantic Scholar · OpenAlex · arXiv · CORE · OpenReview · Tavily",
           tag="SCHOLARLY + WEB", kind="external")
    d.node("nebius", 768, 832, 216, 120, "Nemotron + Qwen3", "Nemotron Nano and Super for generation · Qwen3-Embedding-8B for embeddings",
           tag="NEBIUS TOKEN FACTORY", kind="external")
    d.edge("web", "api", (0.5, 1), (0.5, 0), label="HTTPS /api/*", pos=(716, 264))
    d.edge("web", "auth", (1, 0.5), (0, 0.5), label="SIGN IN", label_bg=SURFACE2)
    d.edge("api", "auth", (1, 0.25), (0.5, 1), points=[(1176, 322)], label="VERIFY JWT", pos=(1176, 264))
    d.edge("api", "orch", ((568 - 448) / 536, 1), (0.5, 0), label="RESEARCH RUN", kind="accent", pos=(568, 432))
    d.edge("api", "analyses", ((920 - 448) / 536, 1), (0.25, 0), points=[(920, 432), (1128, 432)], label="ANALYSIS", pos=(1024, 432))
    d.edge("api", "storage", (1, 0.75), (1, 0.5), points=[(1300, 374), (1300, 712)], label="UPLOADS", pos=(1300, 432))
    d.edge("orch", "laya", (0, 0.5), (1, 0.5), label="SHADOW", kind="advisory", label_bg=SURFACE2)
    d.edge("orch", "router", (1, 0.5), (0, 0.5), label="ROUTE", kind="accent", label_bg=SURFACE2)
    d.edge("analyses", "router", (0, 0.5), (1, 0.5), label="EXTRACT", label_bg=SURFACE2)
    d.edge("router", "nebius", (0.5, 1), (0.5, 0), label="INFERENCE", kind="accent", label_bg=SURFACE2, pos=(876, 712))
    d.edge("orch", "pg", (0.5, 1), (0.5, 0), label="SQL · PGVECTOR", kind="accent", pos=(568, 624))
    d.edge("orch", "providers", ((488 - 448) / 240, 1), (0.5, 0), points=[(488, 624), (256, 624)], label="EVIDENCE GAP", kind="async",
           label_bg=SURFACE2, pos=(256, 712))
    d.legend(992, [("focal", "Research path"), ("service", "Service"), ("store", "Data store"), ("external", "External"),
                   ("advisory", "Advisory (shadow)"), ("arrow:accent", "Primary call"), ("arrow:default", "Call"),
                   ("arrow:async", "Conditional"), ("arrow:advisory", "Shadow, never applied")])
    return d


# ------------------------------------------------------------------------------------------------ 02
@diagram
def lifecycle():
    d = Diagram("02-research-lifecycle", "Lifecycle of a research question",
                "From the browser to a cited answer or an abstention. Each step is recorded in the Research Trace.", W, 1392)
    for name, x, w in [("WEB APP · API · ORCHESTRATOR", 32, 456), ("LAYA · SHADOW", 496, 202), ("SUPABASE POSTGRES", 706, 202),
                       ("NEBIUS · NVIDIA MODELS", 916, 202), ("EVIDENCE PROVIDERS", 1126, 202)]:
        d.zone(x, 112, w, 1184, name)
    y = [152 + i * 96 for i in range(12)]
    steps = [("s1", "1 · Ask a question", "Web app · POST /api/research · stage events stream back as they happen (SSE)"),
             ("s2", "2 · Verify and open a run", "API · Supabase Auth checks the token · rate limit · run row written as the user (RLS)"),
             ("s3", "3 · Understand the question", "Intent, entities, requested fields · strategy · a route per model call (routing-0.1.0)"),
             ("s4", "4 · Look up the answer cache", "Key: pipeline version, release, user and normalised question"),
             ("s5", "5 · Embed the question", "Started in parallel with step 3; a second request is sent after 0.45 s"),
             ("s6", "6 · Hybrid retrieval", "BM25 top 50 (in-process index) + pgvector top 50, fused by reciprocal rank"),
             ("s7", "7 · Rerank the top 20", "Graded 0–3; if the reranker fails, the fused order is kept"),
             None,
             ("s9", "9 · Contradictions and evidence selection", "Recorded conflicts checked · up to 8 corpus and 4 external sources"),
             ("s10", "10 · Synthesis and citation validation", "Every claim needs a verbatim quote from the source it cites; others are removed"),
             ("s11", "11 · Outcome and persistence", "ANSWERED, PARTIAL, CONFLICTING or ABSTAINED · trace rows and runtime graph saved"),
             ("s12", "12 · Show the answer", "Citations · Research Trace · Knowledge Map · a cache hit skips steps 5–11")]
    for i, s in enumerate(steps):
        if s:
            d.node(s[0], 88, y[i], 384, 76, s[1], s[2])
    d.decision("s8", 160, y[7], 240, 76, "8 · Evidence sufficient?")
    order = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"]
    for i, (a, b) in enumerate(zip(order, order[1:])):
        lab = {"s4": "MISS", "s8": "YES"}.get(a)
        d.edge(a, b, (0.5, 1), (0.5, 0), kind="accent", label=lab, label_bg=SURFACE2, pos=(280, y[i] + 86) if lab else None)
    side = [(2, "laya", 508, "Shadow decision", "Traced beside the rules' choice; never applied", "advisory"),
            (3, "cache", 718, "Answer cache", "Per user · 168 h · RLS", "store"),
            (4, "emb", 928, "Query embedding", "Qwen3-Embedding-8B, 256-d", "external"),
            (5, "pgv", 718, "pgvector search", "Top 50 · halfvec(256)", "store"),
            (6, "rr", 928, "Nemotron Nano", "Grades each candidate 0–3", "external"),
            (7, "ext", 1138, "External search", "Providers chosen by need · source gate", "external"),
            (8, "contra", 718, "rx.contradictions", "Recorded conflicts", "store"),
            (9, "syn", 928, "Nano or Super", "Super for comparisons; one validated fallback", "external"),
            (10, "persist", 718, "Persist", "Trace · graph · cache", "store")]
    for i, cid, x, name, detail, kind in side:
        d.node(cid, x, y[i], 178, 76, name, detail, kind=kind)
        if cid != "ext":
            d.edge(order[i], cid, (1, 0.5), (0, 0.5), kind="advisory" if kind == "advisory" else "default")
    d.edge("s8", "ext", (1, 0.5), (0, 0.5), kind="async", label="NO", label_bg=SURFACE2, pos=(436, y[7] + 38))
    d.edge("ext", "s9", (0.5, 1), ((440 - 88) / 384, 0), points=[(1227, y[7] + 86), (440, y[7] + 86)], kind="async")
    d.edge("s4", "s12", (0, 0.5), (0, 0.5), points=[(60, y[3] + 38), (60, y[11] + 38)], kind="ok", label="HIT", label_bg=SURFACE2,
           pos=(60, 760))
    d.legend(1320, [("service", "Step (our code)"), ("advisory", "Laya, shadow only"), ("store", "Supabase Postgres"),
                    ("external", "External service"), ("arrow:accent", "Main flow"), ("arrow:ok", "Cache hit"),
                    ("arrow:async", "Only on an evidence gap"), ("arrow:advisory", "Shadow, never applied"), ("arrow:default", "Call")])
    return d


# ------------------------------------------------------------------------------------------------ 03
@diagram
def evidence():
    d = Diagram("03-retrieval-evidence-abstention", "Retrieval, evidence validation and abstention",
                "How a question becomes validated evidence, and when the answer is partial, conflicting or an abstention.", W, 1032)
    d.zone(32, 112, 1296, 200, "1 · RETRIEVE")
    d.zone(32, 344, 1296, 288, "2 · ASSESS\nEVIDENCE")
    d.zone(32, 664, 1296, 304, "3 · ANSWER\nAND VERIFY")
    d.node("q", 176, 160, 192, 104, "Question", "Intent · entities · requested fields · metric", tag="PARSED")
    d.node("bm25", 432, 128, 232, 80, "BM25", "Top 50 lexical candidates", tag="IN-PROCESS INDEX")
    d.node("pgv", 432, 216, 232, 80, "Dense search", "Top 50, 256-d embeddings", tag="PGVECTOR · QWEN3-EMBEDDING-8B")
    d.node("rrf", 744, 160, 232, 104, "Reciprocal rank fusion", "k = 10 · dense weight 2.0 · tuned on the validation split", tag="HYBRID FUSION")
    d.node("rerank", 1056, 160, 256, 104, "LLM rerank", "Grades 0–3 per candidate · on failure the fused order is kept",
           tag="NEMOTRON NANO · TOP 20")
    d.edge("q", "bm25", (1, 0.5), (0, 0.5), points=[(400, 212), (400, 168)], kind="accent")
    d.edge("q", "pgv", (1, 0.5), (0, 0.5), points=[(400, 212), (400, 256)], kind="accent")
    d.edge("bm25", "rrf", (1, 0.5), (0, 0.25), points=[(704, 168), (704, 186)], kind="accent")
    d.edge("pgv", "rrf", (1, 0.5), (0, 0.75), points=[(704, 256), (704, 238)], kind="accent")
    d.edge("rrf", "rerank", (1, 0.5), (0, 0.5), kind="accent")

    d.node("sel", 176, 376, 272, 112, "Evidence selection", "Passages graded 2+ first · recorded contradictions attached (rx.contradictions)",
           tag="UP TO 8 CORPUS + 4 EXTERNAL")
    d.decision("suff", 1072, 376, 224, 112, "Evidence sufficient?")
    d.note(520, 372, 480, 44, "Sufficient: at least one passage graded 3 or two graded 2+, and the question's rare terms appear in them.")
    d.node("ext", 1040, 512, 272, 104, "External search", "Providers chosen by the missing evidence · per-user cache",
           tag="ONLY ON A GAP · DEADLINE 9 S", kind="external")
    d.node("gate", 704, 512, 272, 104, "Validate sources", "HTTPS · no instruction-like text · not a duplicate · enough content",
           tag="SOURCE GATE")
    d.node("xtr", 368, 512, 272, 104, "External extraction", "Requested fields from abstracts; one arXiv full text if a value is still missing",
           tag="NEMOTRON")
    d.edge("rerank", "suff", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("suff", "sel", (0, 0.5), (1, 0.5), kind="accent", label="YES", label_bg=SURFACE2, pos=(1040, 432))
    d.edge("suff", "ext", (0.5, 1), ((1184 - 1040) / 272, 0), kind="async", label="NO", label_bg=SURFACE2, pos=(1184, 500))
    d.edge("ext", "gate", (0, 0.5), (1, 0.5), kind="async")
    d.edge("gate", "xtr", (0, 0.5), (1, 0.5), kind="async")
    d.edge("xtr", "sel", ((400 - 368) / 272, 0), ((400 - 176) / 272, 1), kind="async")

    d.decision("route", 152, 692, 224, 96, "Parameter or score question?")
    d.node("fg", 448, 688, 272, 104, "Field-level grounding", "A value per requested field, with its verbatim span and conditions",
           tag="GROUNDING-0.4.0")
    d.node("fv", 784, 688, 256, 104, "Fact validation", "Span in the source and holds the value; mismatched conditions stay UNKNOWN",
           tag="DETERMINISTIC")
    d.node("syn", 448, 824, 272, 104, "Passage synthesis", "Claims, each with a verbatim quote and the number of its source",
           tag="NANO · SUPER FOR COMPARISONS")
    d.node("cv", 784, 824, 256, 104, "Citation validation", "The quote must appear in the cited source, with the claim's numbers",
           tag="DETERMINISTIC")
    d.node("status", 1088, 688, 224, 240, "Run status", kind="plain")
    for k, (s, kind, what) in enumerate([("ANSWERED", "ok", "Every claim or field validated"), ("PARTIAL", "warn", "Claims removed, fields unknown"),
                                         ("CONFLICTING", "warn", "Cited sources disagree"), ("ABSTAINED", "info", "“Insufficient evidence found.”")]):
        d.badge(1100, 728 + k * 48, s, kind)
        d.text(1100, 750 + k * 48, 200, 20, what, size=11.5, color="#5b616d")
    d.edge("sel", "route", ((264 - 176) / 272, 1), (0.5, 0), kind="accent")
    d.edge("route", "fg", (1, 0.5), (0, 0.5), kind="accent", label="YES", label_bg=SURFACE2, pos=(412, 740))
    d.edge("route", "syn", (0.5, 1), (0, 0.5), points=[(264, 876)], kind="accent", label="NO", label_bg=SURFACE2, pos=(264, 832))
    d.edge("fg", "fv", (1, 0.5), (0, 0.5), kind="accent")
    d.edge("syn", "cv", (1, 0.5), (0, 0.5), kind="accent")
    d.edge("fv", "status", (1, 0.5), (0, (740 - 688) / 240), kind="accent")
    d.edge("cv", "status", (1, 0.5), (0, (876 - 688) / 240), kind="accent")
    d.note(152, 896, 272, 56, "Second look: if passages graded sufficient yield no valid claim, external search runs once before the run abstains.")
    d.legend(992, [("service", "Stage"), ("external", "External providers"), ("arrow:accent", "Main path"),
                   ("arrow:async", "Only on an evidence gap")])
    return d


# ------------------------------------------------------------------------------------------------ 04
@diagram
def routing():
    d = Diagram("04-model-routing", "Dynamic model routing and fallback",
                "Every generative call goes through one deterministic decision (routing-0.1.0). Nebius Token Factory serves the NVIDIA models.",
                W, 960)
    d.zone(32, 112, 696, 784, "PER MODEL CALL")
    d.zone(752, 112, 576, 784, "ROUTES BY TASK · config/models.yaml")
    d.node("ctx", 56, 152, 360, 104, "Routing context",
           "Task type · complexity · input size · verification need · latency budget · privacy scope · retry count",
           tag="VALIDATED · UNKNOWN FIELDS REJECTED")
    d.node("pol", 56, 288, 360, 104, "Policy orders the routes",
           "Cheapest suitable model first; Super for comparison questions and full-text reading", tag="ROUTING-0.1.0")
    d.decision("usable", 126, 424, 220, 96, "First usable route in the order?")
    d.node("call", 56, 552, 360, 96, "Call the model", "JSON-schema output · per-route timeout and retries", tag="NEBIUS TOKEN FACTORY")
    d.decision("valid", 126, 680, 220, 96, "Output valid?")
    d.node("res", 56, 808, 360, 72, "Use the result", "Decision traced: route, model, reason, fallback", kind="ok")
    d.node("other", 448, 552, 256, 96, "Other tier", "Nano and Super swap; the output is validated again", tag="ONCE")
    d.decision("fits", 456, 680, 240, 96, "Fallback fits the latency budget?")
    d.node("last", 448, 808, 256, 72, "Deterministic last resort", "Per task, see the table", kind="warn")
    d.edge("ctx", "pol", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("pol", "usable", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("usable", "call", (0.5, 1), (0.5, 0), kind="accent", label="YES", label_bg=SURFACE2, pos=(236, 536))
    d.edge("call", "valid", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("valid", "res", (0.5, 1), (0.5, 0), kind="accent", label="YES", label_bg=SURFACE2, pos=(236, 792))
    d.edge("valid", "fits", (1, 0.5), (0, 0.5), label="NO", label_bg=SURFACE2, pos=(400, 728))
    d.edge("fits", "other", (0.5, 0), (0.5, 1), label="YES", label_bg=SURFACE2, pos=(576, 664))
    d.edge("other", "call", (0, 0.5), (1, 0.5))
    d.edge("fits", "last", (0.5, 1), (0.5, 0), label="NO", label_bg=SURFACE2, pos=(576, 792))
    d.edge("usable", "last", (1, 0.5), (1, 0.5), points=[(716, 472), (716, 844)], label="NO", label_bg=SURFACE2, pos=(380, 472))
    rows = [["TASK", "FIRST CHOICE", "FALLBACK, ONCE", "LAST RESORT"],
            ["reranking", "Nemotron Nano", "none", "fused (RRF) order kept"],
            ["synthesis", "Nano; Super for comparison or contradiction questions", "the other tier", "strongest passages quoted verbatim"],
            ["extraction", "Nano over stored passages", "Super with reasoning, if malformed", "fields stay UNKNOWN"],
            ["full-text extraction", "Super with reasoning", "Nano, if Super finds nothing", "fields stay UNKNOWN"],
            ["fallback answer", "Nano", "none", "failure message, no answer"],
            ["embedding", "Qwen3-Embedding-8B", "separate route, never used for generation", "BM25 alone (traced)"]]
    d.table(768, 152, [132, 140, 140, 132], rows, [32] + [64] * 6)
    bullets = ["Nothing a user sends can name a model or a route: the context rejects unknown fields.",
               "A route is skipped when it is not configured, has no key, failed repeatedly in the last minutes (circuit open), "
               "or its provider is not approved for private documents the context holds.",
               "Ultra is configured but never selected: its benefit has not been measured. A long prompt alone never escalates.",
               "Fallback budgets: reranking 8 s · synthesis 20 s · extraction 12 s · full-text extraction 30 s.",
               "Laya reaches the policy only in advise mode (off by default), as the synthesis complexity hint."]
    d.text(768, 592, 544, 280, "<br><br>".join(f"•&nbsp; {b}" for b in bullets), size=12.5, color="#5b616d", html_value=True)
    d.legend(920, [("service", "Step"), ("ok", "Validated result"), ("warn", "Deterministic fallback"), ("arrow:accent", "Main path"),
                   ("arrow:default", "Fallback path")])
    return d


# ------------------------------------------------------------------------------------------------ 05
@diagram
def laya():
    d = Diagram("05-laya-shadow-mode", "Laya: advisory decisions, orchestrator authority",
                "Laya answers beside every run and is traced; the deterministic rules decide. Shadow mode is the default.", W, 720)
    d.zone(32, 112, 632, 544, "RESEARCH ORCHESTRATOR · AUTHORITATIVE")
    d.zone(776, 112, 552, 544, "LAYA · laya-serve :8090 · OWN PROCESS", fill="#f8f1e4")
    d.node("q", 72, 176, 256, 80, "Question arrives", "The run starts; Laya is asked in parallel")
    d.node("rules", 72, 288, 256, 104, "Rules decide", "Intent, requested fields and the injection check set strategy and routing",
           tag="DETERMINISTIC", kind="focal")
    d.node("strat", 72, 424, 256, 88, "Strategy and routing", "The rules' choice is applied")
    d.node("run", 72, 544, 256, 88, "Retrieval, evidence, answer", "Unchanged by Laya in shadow mode")
    d.node("trace", 384, 288, 256, 136, "laya_decision row",
           "Laya's decision beside the rules' choice, with agreement, latency, mode and an authority note", tag="RESEARCH TRACE")
    d.node("laya", 816, 164, 264, 104, "Typed decision", "workflow · external_needed · injection · depth · confidence",
           tag="~421M-PARAMETER CHECKPOINT", kind="advisory")
    d.node("shadow", 816, 336, 232, 128, "Shadow mode", "Traced beside the rules' choice; never applied", tag="DEFAULT · ENABLED", kind="warn")
    d.node("advise", 1072, 336, 232, 128, "Advise mode",
           "May pick the larger synthesis model only at confidence 0.8+ when the rules found no intent; waits at most 1.5 s",
           tag="OFF BY DEFAULT", kind="planned")
    d.node("fail", 816, 496, 488, 104, "A failure costs only its trace row",
           "4 s timeout per decision · slow or unreachable: traced as failed and the run carries on · below 0.8 confidence: "
           "traced as abstained · RESEARCHX_LAYA=0 turns it off", kind="info")
    d.edge("q", "rules", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("rules", "strat", (0.5, 1), (0.5, 0), kind="accent", label="APPLIED", label_bg=SURFACE2, pos=(200, 408))
    d.edge("strat", "run", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("rules", "trace", (1, 0.5), (0, (340 - 288) / 136))
    d.edge("q", "laya", (1, 0.5), (0, 0.5), kind="advisory", label="QUESTION TEXT ONLY", label_bg=SURFACE2, pos=(496, 216))
    d.edge("laya", "trace", (0, (247 - 164) / 104), (1, (356 - 288) / 136), points=[(720, 247), (720, 356)], kind="advisory",
           label="TYPED DECISION", pos=(720, 300))
    d.edge("advise", "strat", (0.5, 1), (1, (480 - 424) / 88), points=[(1188, 480)], kind="faint",
           label="ADVISE MODE ONLY: SYNTHESIS HINT", label_bg=SURFACE2, pos=(520, 480))
    d.note(816, 290, 488, 40, "Laya never sees retrieved text or secrets, never decides what evidence exists and never blocks a request.", size=12.5)
    d.legend(680, [("focal", "Authoritative"), ("advisory", "Advisory"), ("warn", "Active mode"), ("planned", "Off by default"),
                   ("arrow:accent", "Applied"), ("arrow:advisory", "Advisory, traced"), ("arrow:faint", "Advise mode only")])
    return d


# ------------------------------------------------------------------------------------------------ 06
@diagram
def graphs():
    d = Diagram("06-knowledge-graphs", "Canonical Knowledge Graph and per-run Runtime Graph",
                "Shared, reviewed knowledge stays apart from what one run found; identity merges only on identifiers, promotion is explicit.",
                W, 864)
    d.zone(32, 112, 584, 464, "CANONICAL KNOWLEDGE GRAPH · SHARED")
    d.zone(744, 112, 584, 464, "RUNTIME GRAPH · ONE PER RESEARCH RUN · OWNER ONLY")
    d.zone(32, 608, 1296, 192, "PROMOTION · EXPLICIT AND REVIEWED")
    d.node("src", 72, 152, 240, 120, "Reviewed dataset", "Papers, experiments, entities, evidence, study links", tag="SCHEMA rx · FROZEN RELEASE",
           kind="store")
    d.node("km", 344, 152, 240, 120, "Knowledge Map", "Landscape, neighbourhoods, paper diffs and questions over the shared corpus",
           tag="/api/graph/*")
    d.node("ids", 72, 320, 512, 88, "Canonical identity",
           "The same normalised entity name gives the same id; papers by DOI, arXiv id or corpus id", tag="services/canonicalization", kind="info")
    d.note(72, 432, 512, 40, "A matching title alone never merges nodes: they stay separate, linked by a POSSIBLE edge.", size=12.5)
    d.note(72, 488, 512, 40, "Read by every signed-in user; changed only by a new reviewed release.", size=12.5)
    d.node("ev", 784, 152, 240, 120, "Sources and facts",
           "Corpus passages · gated external sources · private documents · validated facts · verified claims", tag="THIS RUN'S EVIDENCE")
    d.node("rg", 1056, 152, 248, 120, "Runtime graph", "Built after the answer, never changes it · stored with the run (owner-only RLS) · shown in the Knowledge Map",
           tag="runtime-graph-0.1.0", kind="focal")
    d.node("prov", 784, 320, 520, 112, "Provenance on every node and edge",
           "origin: canonical, runtime_external, private_document, user_input, derived · status: validated, unverified, related, promoted, "
           "rejected · provider, URL, identifiers, retrieval time, content hash, run id", tag="ON EVERY NODE AND EDGE", kind="info")
    d.node("priv", 784, 456, 520, 96, "Private documents are never proposed",
           "promotion_checks refuses origin private_document; private nodes stay in their owner's run", kind="bad")
    d.edge("src", "km", (1, 0.5), (0, 0.5))
    d.edge("ev", "rg", (1, 0.5), (0, 0.5), kind="accent")
    d.edge("prov", "ids", (0, (364 - 320) / 112), (1, 0.5), label="MERGE ON IDS", pos=(684, 364))
    d.node("p1", 1112, 648, 192, 112, "External source", "A paper or source found in a run", tag="RUNTIME_EXTERNAL")
    d.node("p2", 872, 648, 208, 112, "Owner proposes it", "From the run's graph", tag="POST /api/graph/promotions")
    d.node("p3", 608, 648, 232, 112, "Checks", "Passed the source gate · complete provenance · backs a validated quote", tag="promotion_checks")
    d.node("p4", 360, 648, 216, 112, "CANDIDATE", "No user linkage", tag="rx.runtime_sources", kind="store")
    d.node("p5", 72, 648, 256, 112, "Reviewed release", "Review, licence check, re-extraction, next corpus release", tag="NOT AUTOMATED YET",
           kind="planned")
    for a, b in [("p1", "p2"), ("p2", "p3"), ("p3", "p4")]:
        d.edge(a, b, (0, 0.5), (1, 0.5))
    d.edge("p4", "p5", (0, 0.5), (1, 0.5), kind="faint")
    d.edge("p5", "src", ((320 - 72) / 256, 0), (0, 0.5), points=[(320, 592), (52, 592), (52, 212)], kind="faint")
    d.legend(824, [("store", "Data"), ("service", "Component"), ("focal", "Built per run"), ("info", "Rule"), ("bad", "Never"),
                   ("planned", "Not automated yet"), ("arrow:faint", "Planned step")])
    return d


# ------------------------------------------------------------------------------------------------ 07
@diagram
def caches():
    d = Diagram("07-cache-architecture", "Cache architecture",
                "Four separate caches, all scoped to one user. A hit skips real work; a miss computes and stores.", W, 1064)
    d.node("q", 96, 128, 360, 80, "Question", "Every cache key includes the user id", tag="SIGNED-IN USER")
    d.decision("d1", 156, 232, 240, 96, "Answer cached for this user?")
    d.node("emb", 96, 384, 360, 88, "Query embedding", "Cache first; on a miss Nebius embeds the question")
    d.node("rr", 96, 520, 360, 88, "Rerank", "Cache first for the exact candidate set; on a miss Nemotron grades it")
    d.node("ext", 96, 656, 360, 88, "External search (on an evidence gap)", "Each provider's cached results first; on a miss it is called")
    d.node("syn", 96, 792, 360, 88, "Synthesis, validation, outcome", "Never cached: computed from the selected evidence")
    d.node("out", 96, 928, 360, 72, "Answer shown", "cached: true on a hit (no model or provider calls)", kind="ok")
    d.node("c_ans", 616, 208, 680, 144, "Answer cache",
           "Per user (user_id + RLS) · 168 h · key: pipeline version, release, user, normalised question. Written only for ANSWERED, PARTIAL "
           "or CONFLICTING while the database is under 450 MB; emptied for the user when they upload or delete a document. "
           "A hit returns the stored answer without model calls.", tag="POSTGRES · rx.cache_entries KIND 'model' · DURABLE", kind="store")
    d.node("c_emb", 616, 372, 680, 112, "Query-embedding cache",
           "Per user · 6 h · 1,024 entries, least recently used evicted · key: model, embedding version, user, question · emptied by a restart",
           tag="IN PROCESS · services/runtime/memo.py")
    d.node("c_rr", 616, 508, 680, 112, "Reranker-grade cache",
           "Per user · 6 h · 512 entries · key: model, prompt version, user, question, each candidate's id and text hash · emptied by a restart",
           tag="IN PROCESS · services/runtime/memo.py")
    d.node("c_ext", 616, 640, 680, 120, "External-provider cache",
           "Per user, never shared · 24 h · key: user, provider, query, limit · written while the database is under 450 MB · a hit skips "
           "that provider", tag="POSTGRES · rx.cache_entries KIND 'api' · DURABLE", kind="store")
    d.edge("q", "d1", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("d1", "emb", (0.5, 1), (0.5, 0), kind="accent", label="MISS", pos=(276, 356))
    for a, b in [("emb", "rr"), ("rr", "ext"), ("ext", "syn"), ("syn", "out")]:
        d.edge(a, b, (0.5, 1), (0.5, 0), kind="accent")
    d.edge("d1", "c_ans", (1, 0.5), (0, 0.5), label="LOOKUP", pos=(506, 280))
    for a, b in [("emb", "c_emb"), ("rr", "c_rr"), ("ext", "c_ext")]:
        d.edge(a, b, (1, 0.5), (0, 0.5), start_arrow=True, label="LOOKUP · STORE")
    d.edge("syn", "c_ans", (1, 0.5), (1, 0.5), points=[(1320, 836), (1320, 280)], label="WRITE IF ALLOWED", pos=(900, 836))
    d.edge("d1", "out", (0, 0.5), (0, 0.5), points=[(64, 280), (64, 964)], kind="ok", label="HIT", pos=(64, 620))
    d.legend(1024, [("service", "Step"), ("store", "Durable cache (Postgres)"), ("ok", "Answer"), ("arrow:accent", "Main flow"),
                   ("arrow:ok", "Answer-cache hit"), ("arrow:default", "Cache read or write")])
    return d


# ------------------------------------------------------------------------------------------------ 08
@diagram
def trace():
    d = Diagram("08-research-trace", "Research Trace and observability",
                "Every stage of every run is recorded as it happens, streamed live and stored per user.", W, 776)
    d.node("stages", 56, 160, 240, 112, "Stages", "Each stage runs inside tr.stage(…) or is recorded with tr.record(…)",
           tag="ORCHESTRATOR · ANALYSES")
    d.node("tracer", 376, 160, 256, 112, "Research Trace", "Status, latency, budget and detail per stage; written as the user",
           tag="RunTracer · services/runtime", kind="focal")
    d.node("sse", 712, 128, 288, 96, "Stage events", "POST /api/research streams each stage as it ends", tag="LIVE · SERVER-SENT EVENTS")
    d.node("db", 712, 248, 288, 112, "research_runs · run_events", "One row per stage; a CHECK lists the 34 known step names",
           tag="SUPABASE rx · OWNER-ONLY RLS", kind="store")
    d.node("ui", 1080, 128, 232, 96, "Trace panel", "Live while the run streams; reopened from stored rows", tag="WEB APP")
    d.node("apis", 1080, 248, 232, 144, "Owner-only reads",
           "GET /api/research/{id}/trace · GET /api/metrics/architecture: own stage metrics and global aggregates, no content",
           tag="READ APIS")
    d.edge("stages", "tracer", (1, 0.5), (0, 0.5), kind="accent")
    d.edge("tracer", "sse", (1, 0.25), (0, 0.5), points=[(672, 188), (672, 176)], kind="accent")
    d.edge("tracer", "db", (1, 0.75), (0, 0.5), points=[(672, 244), (672, 304)], kind="accent")
    d.edge("sse", "ui", (1, 0.5), (0, 0.5), label="LIVE", pos=(1040, 176))
    d.edge("db", "apis", (1, 0.5), (0, (304 - 248) / 144), label="STORED", pos=(1040, 304))
    d.edge("apis", "ui", (0.5, 0), (0.5, 1))
    d.node("fields", 56, 320, 576, 120, "What a trace row holds",
           "step · status (completed, failed, skipped, not_required) · latency_ms · budget_ms and over_budget · detail: model, routing "
           "decision, counts, reasons, errors", tag="EVERY EVENT", kind="info")
    d.note(712, 400, 600, 44, "A step name missing from the CHECK list is rejected by the database and the run reports trace_persisted: false.",
           size=12.5)
    d.zone(32, 472, 1296, 248, "THE 34 STEP NAMES")
    groups = [("Run setup · 5", "query_understanding, laya_decision, research_strategy, model_routing, cache_lookup"),
              ("Retrieval · 4", "bm25, pgvector, hybrid_fusion, reranking"),
              ("Evidence · 10", "evidence_assessment, evidence_gap, external_sources, tavily, source_validation, external_extraction, "
                                "evidence_extraction, experiment_extraction, contradiction_detection, evidence_selection"),
              ("Answer · 6", "synthesis, citation_validation, abstention, fallback, answer, graph_enrichment"),
              ("Analyses · 9", "analysis_request, experiment_retrieval, field_facts, comparison, comparability, runtime_enrichment, "
                               "assessment, recommendations, coverage")]
    for k, (name, steps) in enumerate(groups):
        d.node(f"g{k}", 56 + k * 252, 512, 236, 192, name, steps)
    d.legend(736, [("service", "Component"), ("focal", "Tracer"), ("store", "Stored rows"), ("info", "Row format"),
                   ("arrow:accent", "Recorded"), ("arrow:default", "Read")])
    return d


# ------------------------------------------------------------------------------------------------ 09
@diagram
def access():
    d = Diagram("09-auth-private-library", "Authentication, private library and isolation",
                "Identity comes only from a token Supabase Auth has verified; PostgreSQL row-level security enforces ownership.", W, 1120)
    d.zone(32, 112, 312, 720, "BROWSER")
    d.zone(376, 112, 400, 720, "API · services/api")
    d.zone(808, 112, 520, 720, "SUPABASE")
    d.zone(32, 864, 1296, 136, "MODEL AND EVIDENCE PROVIDERS")
    d.node("b_web", 56, 152, 264, 104, "Web app", "Signs in with the publishable key only; no secret reaches the browser", tag="SUPABASE CLIENT")
    d.node("b_req", 56, 296, 264, 104, "Research and library calls", "Authorization: Bearer <access token> on every request")
    d.node("b_up", 56, 576, 264, 96, "Upload a document", "PDF or text, sent to the API")
    d.node("a_ver", 400, 280, 352, 136, "Verify the access token",
           "Asks Supabase Auth (GET /auth/v1/user), accepted for at most 30 s · MFA users need an aal2 session · request bodies reject "
           "unknown fields, so a forged user_id is refused", tag="services/auth/supabase.py", kind="focal")
    d.node("a_utx", 400, 448, 352, 96, "Queries as the user", "Role authenticated with exactly the verified claims; no service-role key in the API",
           tag="user_tx()")
    d.node("a_doc", 400, 576, 352, 104, "Document pipeline", "File to the user's folder · chunks and embeddings · the user's answer cache is emptied",
           tag="POST /api/documents")
    d.node("a_run", 400, 712, 352, 96, "Private retrieval in a run", "The user's chunks, searched under RLS", tag="PRIVACY SCOPE USER-PRIVATE")
    d.node("s_auth", 832, 152, 472, 104, "Accounts and sessions", "E-mail and password · Google · GitHub · TOTP MFA", tag="SUPABASE AUTH")
    d.node("s_pg", 832, 440, 472, 160, "Owner-only rows",
           "Documents, chunks, research runs, trace events, caches, conversations · the shared corpus is read through the backend's "
           "service connection", tag="POSTGRES · ROW-LEVEL SECURITY", kind="store")
    d.node("s_sto", 832, 632, 472, 112, "One folder per user", "<user id>/<document> · storage policies keep each user in their own folder",
           tag="STORAGE · BUCKET user-documents", kind="store")
    d.node("p_neb", 400, 888, 352, 88, "Approved for private data", "May receive private passages for synthesis and extraction",
           tag="NEBIUS TOKEN FACTORY · NVIDIA", kind="external")
    d.node("p_ext", 832, 888, 472, 88, "Evidence providers", "Receive search terms from the question only; never private text",
           tag="SCHOLARLY + WEB", kind="external")
    d.edge("b_web", "s_auth", (1, 0.5), (0, 0.5), label="SIGN IN · SESSION", label_bg=SURFACE2, pos=(576, 204))
    d.edge("b_req", "a_ver", (1, 0.5), (0, (348 - 280) / 136), kind="accent")
    d.edge("a_ver", "s_auth", (1, (312 - 280) / 136), (0.5, 1), points=[(1068, 312)], label="VERIFY", label_bg=SURFACE2, pos=(1068, 284))
    d.edge("a_ver", "a_utx", (0.5, 1), (0.5, 0), kind="accent")
    d.edge("a_utx", "s_pg", (1, 0.5), (0, (496 - 440) / 160), kind="accent", label="SQL", pos=(792, 496))
    d.edge("b_up", "a_doc", (1, 0.5), (0, (624 - 576) / 104))
    d.edge("a_doc", "s_pg", (1, (592 - 576) / 104), (0, (592 - 440) / 160))
    d.edge("a_doc", "s_sto", (1, (656 - 576) / 104), (0, (656 - 632) / 112))
    d.edge("a_run", "p_neb", (0.5, 1), (0.5, 0), label="PRIVATE PASSAGES", pos=(576, 848))
    d.edge("a_run", "p_ext", (1, 0.5), (0.5, 0), points=[(1068, 760)], label="SEARCH TERMS ONLY", pos=(1068, 848))
    d.note(40, 1024, 1280, 40, "Private documents are never promoted to the shared graph · deleting the account removes everything it owns "
                               "(recent sign-in required) · Laya and the evidence providers never receive private text.", size=12.5)
    d.legend(1080, [("service", "Component"), ("focal", "Trust boundary check"), ("store", "Supabase data"), ("external", "External provider"),
                    ("arrow:accent", "Authenticated path"), ("arrow:default", "Data flow")])
    return d


# ------------------------------------------------------------------------------------------------ 10
@diagram
def analyses():
    d = Diagram("10-eight-analyses", "The eight research analyses",
                "One API, one shared foundation of validated records; papers outside the corpus are read at request time, never written.",
                W, 1472)
    d.node("in1", 56, 128, 392, 96, "Analyze tab", "Pick experiments, papers or a benchmark; type a claim or a draft review", tag="WEB APP")
    d.node("in2", 480, 128, 392, 96, "Papers and Benchmark modes", "Paper-to-Paper Diff and Benchmark Reality Explorer from plain text",
           tag="RESEARCH CHAT")
    d.node("in3", 904, 128, 400, 96, "Links on sources and nodes", "Open an analysis on a cited paper, experiment or benchmark",
           tag="KNOWLEDGE MAP · ANSWERS")
    d.node("api", 56, 264, 1248, 80, "Analysis API",
           "Each call is a traced research run of the caller (owner-only RLS); the run keeps a compact summary, not the evidence payload",
           tag="FASTAPI · /api/analysis/*", kind="focal")
    for cid, x in [("in1", 252), ("in2", 676), ("in3", 1104)]:
        d.edge(cid, "api", (0.5, 1), ((x - 56) / 1248, 0), kind="accent")
    d.zone(32, 384, 1296, 416, "THE EIGHT ANALYSES", cid="grid")
    eight = [("Paper-to-Paper Diff", "GET /paper-diff · POST /papers/ask",
              "Two to four papers field by field, training settings stage by stage, with a fairness verdict; also from chat text", True),
             ("Result Discrepancy Investigator", "GET /discrepancy",
              "Why two similar experiments report different results: the score gap, verified differences and what is not stated", False),
             ("Reproducibility Recipe Builder", "GET /recipe",
              "One experiment's configuration as a validated reproduction spec; unstated settings stay UNKNOWN, never defaulted", False),
             ("Next Experiment Planner", "GET /plan",
              "Reproduce, ablate or transfer: the experiment that would most reduce uncertainty, derived from the records", False),
             ("Claim-to-Evidence Auditor", "POST /claim",
              "A stated comparison, reported value or training setting tested against the corpus records", False),
             ("Generalization and Transferability Analyzer", "GET /transfer",
              "Where else a model was evaluated, with the pre-training data and adaptation behind each result", False),
             ("Literature Review Completeness Auditor", "POST /review",
              "What a question or draft review covers, and the benchmarks and model families the corpus suggests it leaves out", False),
             ("Benchmark Reality Explorer", "GET /benchmarks · GET /benchmark",
              "Every reported result for a benchmark and metric: COMPARABLE, CONDITIONALLY_COMPARABLE, INCOMPATIBLE or UNKNOWN; "
              "only comparable results are ranked", True)]
    for k, (name, route, what, runtime) in enumerate(eight):
        x, y = 56 + (k % 4) * 320, 416 + (k // 4) * 192
        d.node(f"a{k}", x, y, 296, 168, name, what, tag=route)
        if runtime:
            d.badge(x + 12, y + 138, "+ RUNTIME PAPERS ON REQUEST", "accent")
    d.edge("api", "grid", ((680 - 56) / 1248, 1), ((680 - 32) / 1296, 0), kind="accent")
    d.zone(32, 840, 1296, 256, "SHARED FOUNDATION · services/analysis · services/comparison", cid="found")
    found = [("Validated experiment records", "SCHEMA rx", "Papers, experiments and evidence of the reviewed corpus release"),
             ("Field facts", "services/analysis/facts.py", "REPORTED · DERIVED · CONFLICTING · UNKNOWN, each with its evidence"),
             ("Configuration identity", "services/analysis/configs.py",
              "Pre-training, fine-tuning and evaluation kept apart; links DERIVED with their rule"),
             ("Comparison engine", "SILVER-LABELER-0.2.0",
              "Field by field after normalisation (val = validation set); UNKNOWN is never a match"),
             ("Request-time full-text reading", "services/analysis/enrich.py",
              "Before a setting shows as not stated, passages that name it are read; checked values become DERIVED"),
             ("Research Trace", "RunTracer", "Every call is a traced run: analysis_request, field_facts, comparison, coverage …")]
    for k, (name, tag, what) in enumerate(found):
        d.node(f"f{k}", 56 + (k % 3) * 424, 872 + (k // 3) * 112, 400, 96, name, what, tag=tag)
    d.edge("grid", "found", ((680 - 32) / 1296, 1), ((680 - 32) / 1296, 0), kind="accent", label="ALL EIGHT READ", pos=(680, 820))
    d.zone(32, 1136, 1296, 272, "RUNTIME PAPERS · READ AT REQUEST TIME FOR PAPER-TO-PAPER DIFF AND BENCHMARK REALITY EXPLORER · "
                                "NOTHING IS WRITTEN TO THE CORPUS")
    pipe = [("r1", "Reference or topic", "Title, DOI, arXiv id or URL, Semantic Scholar or ACL Anthology URL", "WHAT THE USER TYPED", "service"),
            ("r2", "Resolve", "Corpus first by DOI, arXiv id or exact title; if opted in, Semantic Scholar, OpenAlex, arXiv",
             "IDENTIFIERS FIRST", "service"),
            ("r3", "Read", "arXiv full text with its licence recorded, else the abstract alone", "runtime/reader.py", "service"),
            ("r4", "Extract", "Schema-bound settings, evaluation conditions and result rows", "fulltext_extraction", "service"),
            ("r5", "Check", "Span holds the value · metric and dataset named · no instruction-like text", "DETERMINISTIC", "service"),
            ("r6", "Runtime records", "Configuration-level ids, stage kept; served through an Overlay beside corpus records",
             "PER USER · IN PROCESS", "focal")]
    for k, (cid, name, what, tag, kind) in enumerate(pipe):
        d.node(cid, 56 + k * 208, 1168, 188, 152, name, what, tag=tag, kind=kind)
    for a, b in zip(["r1", "r2", "r3", "r4", "r5"], ["r2", "r3", "r4", "r5", "r6"]):
        d.edge(a, b, (1, 0.5), (0, 0.5), kind="accent")
    d.edge("r6", "found", (0.5, 0), ((1190 - 32) / 1296, 1), label="OVERLAY", pos=(1190, 1116))
    d.note(56, 1336, 1248, 56, "A typed title resolves only on a close title match with no equal rival, otherwise the candidates are returned to "
                               "pick from; a work already in the corpus is that corpus paper. Corpus papers without experiment records are read the "
                               "same way when opted in. Only what the user typed reaches a provider; private documents are never read here.",
           size=12.5)
    d.legend(1432, [("service", "Component"), ("focal", "Entry or result"), ("arrow:accent", "Request flow"),
                    ("arrow:default", "Read beside corpus records")])
    return d


# ------------------------------------------------------------------------------------------------ 11
@diagram
def deployment():
    d = Diagram("11-deployment", "Deployment",
                "One public web address; the API, Supabase and the model and evidence providers sit behind it. Status as of 2026-10-16.",
                W, 720)
    d.zone(32, 112, 264, 384, "BROWSER")
    d.zone(328, 112, 336, 384, "VERCEL")
    d.zone(696, 112, 336, 384, "BACKEND HOST")
    d.zone(1064, 112, 264, 384, "MANAGED SERVICES")
    d.node("user", 56, 200, 216, 104, "Researcher or judge", "Opens one public URL")
    d.node("web", 352, 176, 288, 176, "Public web app",
           "Serves the UI and rewrites /api/* on the server to RESEARCHX_API_ORIGIN, so the browser talks to one origin",
           tag="NEXT.JS 16 · apps/web", kind="focal")
    d.badge(364, 318, "NOT YET DEPLOYED", "muted")
    d.node("api", 720, 176, 288, 176, "ResearchX API",
           "Same code as local · RESEARCHX_LAYA=0, Laya is not deployed · secrets only in the host's environment", tag="FASTAPI · services/api")
    d.badge(732, 318, "NOT YET DEPLOYED", "muted")
    d.node("nebius", 1088, 144, 216, 80, "Nebius Token Factory", "NVIDIA Nemotron · Qwen3 embeddings", kind="external")
    d.node("prov", 1088, 256, 216, 80, "Evidence providers", "Semantic Scholar, OpenAlex, arXiv, …", kind="external")
    d.node("supa", 1088, 368, 216, 104, "Supabase", "Postgres (rx, pgvector, RLS) · Auth · Storage", kind="store")
    d.edge("user", "web", (1, 0.5), (0, (252 - 176) / 176), kind="accent", label="HTTPS", pos=(312, 252))
    d.edge("web", "api", (1, 0.5), (0, 0.5), kind="accent", label="/api/*", pos=(680, 264))
    d.edge("api", "nebius", (1, (184 - 176) / 176), (0, 0.5))
    d.edge("api", "prov", (1, (296 - 176) / 176), (0, 0.5))
    d.edge("api", "supa", ((912 - 720) / 288, 1), (0, (424 - 368) / 104), points=[(912, 424)])
    d.edge("user", "supa", (0.5, 1), (0.5, 1), points=[(164, 528), (1196, 528)], label="SIGN IN WITH SUPABASE AUTH", pos=(680, 528))
    d.note(40, 568, 1280, 64, "Secrets live only in the hosts' environment settings: NEBIUS_API_KEY, the database connection, provider keys "
                              "(TAVILY_API_KEY, CORE_API_KEY, OPENALEX_API_KEY). The browser receives only SUPABASE_URL and the publishable key; "
                              "the Supabase redirect list must include the public URL. Nothing secret is in the repository.", size=12.5)
    d.legend(672, [("focal", "Public entry"), ("service", "Our service"), ("store", "Managed data"), ("external", "External provider"),
                   ("arrow:accent", "Request path"), ("arrow:default", "Server-side call")])
    return d


# ------------------------------------------------------------------------------------------------ 12
@diagram
def demo():
    d = Diagram("12-judge-demo-journey", "Judge demo journey",
                "Eight steps, one story: a researcher asks, checks the evidence, compares, explores and sees the system refuse to guess.",
                W, 640)
    steps = [("1 · Sign in", "SUPABASE AUTH", "E-mail account, MFA if enrolled. Every later request carries the verified token; all data shown is the user's own."),
             ("2 · Ask the flagship question", "RESEARCH CHAT",
              "“What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?” Values with citations to verbatim evidence."),
             ("3 · Open the Research Trace", "RESEARCH TRACE",
              "Each stage with timing and budget, the model routing decision and the Laya shadow row (never applied)."),
             ("4 · Open the Knowledge Map", "KNOWLEDGE MAP",
              "The run's runtime graph beside the canonical corpus graph, with provenance on every node and edge."),
             ("5 · Compare the cited papers", "PAPER-TO-PAPER DIFF",
              "Field by field and stage by stage, with a fairness verdict; unknown fields stay unknown."),
             ("6 · Ask the same question again", "ANSWER CACHE",
              "cached: true and no model calls, only if the first answer was stored (database under the 450 MB guard)."),
             ("7 · The other seven analyses", "ANALYZE TAB",
              "Result Discrepancy Investigator · Reproducibility Recipe Builder · Next Experiment Planner · Claim-to-Evidence Auditor · "
              "Generalization and Transferability Analyzer · Literature Review Completeness Auditor · Benchmark Reality Explorer"),
             ("8 · Ask what the evidence cannot answer", "ABSTENTION",
              "“Insufficient evidence found.” with what was searched, instead of a guess.")]
    xs = [56, 384, 712, 1040]
    for k, (name, tag, what) in enumerate(steps):
        x, y, h = (xs[k], 152, 160) if k < 4 else (xs[7 - k], 376, 200)
        d.node(f"j{k}", x, y, 264, h, name, what, tag=tag, kind="focal" if k in (1, 4) else "service")
    for k in range(3):
        d.edge(f"j{k}", f"j{k + 1}", (1, 0.5), (0, 0.5), kind="accent")
    d.edge("j3", "j4", (0.5, 1), (0.5, 0), kind="accent")
    for k in range(4, 7):
        d.edge(f"j{k}", f"j{k + 1}", (0, 0.5), (1, 0.5), kind="accent")
    d.legend(600, [("focal", "Core moment"), ("service", "Step"), ("arrow:accent", "Demo order")])
    return d


def main(argv):
    failed = 0
    for name in argv or list(DIAGRAMS):
        path, problems = DIAGRAMS[name]().save(OUT)
        print(f"{path.name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems:
            print("   ", p)
        failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
