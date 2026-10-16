"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { api, ApiError, authRedirect, research, type Message, type Result, type Source, type Stage, type StreamEvent } from "@/lib/api";
import { AnalysisCard, CHAT_MODES, runAnalysis, type ChatMode } from "@/components/analyze";
import { toggleTray, TRAY_MAX, useTray } from "@/lib/diff-tray";
import { FEATURE } from "@/lib/features";
import { useApp } from "@/components/app-shell";
import { Alert, cx, Spinner } from "@/components/ui";
import * as I from "@/components/icons";

const EXAMPLES = [
  "What learning rate and batch size were used to train HRNetV2 + OCR on ADE20K?",
  "What top-1 accuracy does ViT-B/16 reach on ImageNet?",
  "Which optimizer and weight decay were used to fine-tune BEiT on ADE20K?",
  "Which papers report results on BoolQ, and with which metric?",
];

// Stage names come from the backend trace (rx.run_events); labels are presentation only.
const LABEL: Record<string, string> = {
  query_understanding: "Query understanding",
  laya_decision: "Laya decision",
  research_strategy: "Research strategy",
  model_routing: "Model routing",
  cache_lookup: "Answer cache",
  bm25: "BM25 lexical retrieval",
  pgvector: "pgvector semantic retrieval",
  hybrid_fusion: "Hybrid fusion (RRF)",
  reranking: "Reranking",
  experiment_extraction: "Configuration matching",
  evidence_extraction: "Evidence extraction & validation",
  evidence_assessment: "Evidence sufficiency",
  evidence_gap: "Evidence gap",
  external_sources: "Scholarly search APIs",
  tavily: "Tavily web research",
  source_validation: "Source validation",
  external_extraction: "Runtime evidence extraction",
  contradiction_detection: "Contradiction detection",
  evidence_selection: "Evidence selection",
  synthesis: "Grounded synthesis",
  citation_validation: "Citation validation",
  abstention: "Abstention",
  answer: "Answer",
  graph_enrichment: "Knowledge graph update",
  fallback: "Model-only fallback",
};
// runtime providers (services/runtime/tools.py PROVIDERS)
const PROVIDER: Record<string, string> = {
  semantic_scholar: "Semantic Scholar", openalex: "OpenAlex", arxiv: "arXiv", core: "CORE", openreview: "OpenReview", tavily: "Web · Tavily",
};
// why a candidate value found in a source was not used (services/evidence/grounding.validate, configuration matching)
const WHY: Record<string, string> = {
  different_dataset: "reported for a different dataset",
  configuration_related: "reported for another model or configuration",
  configuration_weak: "from a source that does not match the model and dataset asked",
  span_not_in_source: "its quoted span is not in the source text",
  value_not_in_span: "the value is not in its quoted span",
  no_field_cue: "the span does not name the requested field",
  value_not_numeric: "the value is not a number",
  source_not_found: "the source could not be re-read",
  // runtime rules for text read from outside the corpus (services/evidence/grounding.validate_score, table_column_ok)
  other_column: "it is in another model's or dataset's table column",
  other_variant: "it is reported for another size of the model",
  value_set: "it is one of several values tried, not the setting used",
  dataset_not_stated: "the source does not say which dataset the value is for",
  relative_value: "it is a gain or margin, not a score",
  value_change: "it is a change between two values, not one result",
  wrong_metric: "it is reported for another metric",
  no_metric_cue: "the span does not name the metric",
};
const CONFIG_REASONS = new Set(["different_dataset", "other_column", "other_variant", "dataset_not_stated"]);
// why an external result the run found was not used (services/runtime/orchestrator._unused_sources, tools.validate_sources)
const UNUSED_WHY: Record<string, string> = {
  not_read: "not read: beyond the run's limit of external sources",
  not_cited: "read, but no verified statement cites it",
  configuration_related: "about another model or dataset",
  configuration_weak: "does not match the model and dataset asked",
  no_requested_value: "read, but it states no value for what was asked",
  values_not_usable: "its candidate values failed validation (listed above)",
  closer_match_used: "a closer matching source was used instead",
  instruction_like_text: "rejected by the source gate: instruction-like text",
  empty_content: "rejected by the source gate: no usable text",
  not_https: "rejected by the source gate: not an https link",
};
const EXTERNAL_UNIT: Record<string, string> = { paragraph: "full-text passage", table: "full-text table" };
// why a statement was removed from a passage answer (services/runtime/orchestrator.validate_claims)
const REMOVED_WHY: Record<string, string> = {
  no_valid_source: "it cites no retrieved source",
  quote_not_found: "its quote is not in the cited source",
  number_not_in_quote: "a number in it is not in its quote",
};

const VERDICT: Record<string, { label: string; tone: string }> = {
  ANSWERED: { label: "Supported by evidence", tone: "bg-ok-soft text-ok" },
  PARTIAL: { label: "Partially supported", tone: "bg-warn-soft text-warn" },
  CONFLICTING: { label: "Conflicting evidence", tone: "bg-bad-soft text-bad" },
  ABSTAINED: { label: "Insufficient evidence", tone: "bg-surface-2 text-muted" },
  FAILED: { label: "Pipeline unavailable", tone: "bg-bad-soft text-bad" },
};
// Field-level grounding outcome (parameter and score questions); more precise than the run status when present.
const EVIDENCE: Record<string, { label: string; tone: string }> = {
  SUFFICIENT: VERDICT.ANSWERED,
  PARTIAL: VERDICT.PARTIAL,
  CONFLICTING_EVIDENCE: VERDICT.CONFLICTING,
  AMBIGUOUS_EXPERIMENT_MATCH: { label: "Depends on the experiment", tone: "bg-warn-soft text-warn" },
  INSUFFICIENT_EVIDENCE: VERDICT.ABSTAINED,
  NO_RELEVANT_SOURCE_FOUND: { label: "No relevant source found", tone: "bg-surface-2 text-muted" },
};
// configuration match of the source's paper (or document) against the question's model + dataset
const MATCH: Record<string, string> = {
  EXACT: "paper matches model + dataset",
  STRONG: "paper closely matches",
  RELATED: "paper on another model or dataset",
  DATASET: "other paper · same dataset",
  WEAK: "weak match",
};
// Parsed papers keep LaTeX symbol macros; shown as symbols (display only: spans are located on the raw text first).
const LATEX: [RegExp, string][] = [
  [/\\times\s?/g, "×"], [/\\Box\s?/g, "□ "], [/\\cdot\s?/g, "·"], [/\\pm\s?/g, "±"], [/\\approx\s?/g, "≈"], [/\\leq?\s?/g, "≤"],
  [/\\geq?\s?/g, "≥"], [/\\sim\s?/g, "~"], [/\\%/g, "%"], [/\\infty/g, "∞"], [/\\(?:rightarrow|to)\s?/g, "→"],
];
const pretty = (t: string) => LATEX.reduce((s, [re, sym]) => s.replace(re, sym), t);
const UNIT: Record<string, string> = { paragraph: "Paper passage", table: "Paper table", caption: "Figure/table caption", footnote: "Paper footnote", paper: "Paper abstract" };

// a turn is a research run, or (mode set) one of the analyses run on the typed text, shown as a visual card
type Turn = { id: string; query: string; modality: "text" | "voice"; result?: Result; live?: Stage[]; error?: string; pending?: boolean;
  mode?: ChatMode; analysis?: unknown };

function toTurns(messages: Message[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of messages) {
    if (m.role === "user") turns.push({ id: m.id, query: m.content, modality: m.input_modality });
    else if (turns.length) {
      const last = turns[turns.length - 1];
      last.result = m.result ? { ...m.result, research_run_id: m.research_run_id ?? m.result.research_run_id } : undefined;
      if (!m.result) last.error = m.content;
    }
  }
  return turns;
}

function upsert(stages: Stage[], e: StreamEvent): Stage[] {
  const next = { stage: e.stage!, status: (e.status ?? "running") as Stage["status"], latency_ms: e.latency_ms, detail: e.detail, error: e.error };
  const i = stages.findIndex((s) => s.stage === e.stage);
  return i < 0 ? [...stages, next] : stages.map((s, j) => (j === i ? { ...s, ...next } : s));
}

export function Chat({ conversationId }: { conversationId: string | null }) {
  const router = useRouter();
  const { refresh } = useApp();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(!!conversationId);
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!conversationId) return;
    let alive = true;
    api<{ title: string | null; messages: Message[] }>(`/api/conversations/${conversationId}/messages?limit=100`)
      .then((r) => alive && setTurns(toTurns(r.messages)))
      .catch((e) => {
        const to = authRedirect(e);
        if (to) router.replace(to);
        else if (e instanceof ApiError && e.status === 404) router.replace("/");
        else setNotice(e.message);
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [conversationId, router]);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns]);

  const ask = useCallback(
    async (query: string, modality: "text" | "voice", mode?: ChatMode) => {
      const id = crypto.randomUUID();
      setTurns((t) => [...t, { id, query, modality, mode, pending: true, live: [] }]);
      setRunning(true);
      setNotice(null);
      if (mode) {  // an analysis on the typed text: deterministic, traced as its own run, not saved into the conversation
        try {
          const analysis = await runAnalysis(mode, query);
          setTurns((t) => t.map((x) => (x.id === id ? { ...x, pending: false, analysis } : x)));
        } catch (e) {
          const to = authRedirect(e);
          if (to) return router.replace(to);
          setTurns((t) => t.map((x) => (x.id === id ? { ...x, pending: false, error: e instanceof Error ? e.message : "The analysis failed." } : x)));
        } finally {
          setRunning(false);
        }
        return;
      }
      const onEvent = (e: StreamEvent) => {
        if (e.type === "stage" && e.stage) setTurns((t) => t.map((x) => (x.id === id ? { ...x, live: upsert(x.live ?? [], e) } : x)));
      };
      try {
        const result = await research({ query, conversation_id: conversationId ?? undefined, input_modality: modality }, onEvent);
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, pending: false, result } : x)));
        void refresh();
        if (!conversationId) router.replace(`/c/${result.conversation_id}`);
      } catch (e) {
        const to = authRedirect(e);
        if (to) return router.replace(to);
        const message = e instanceof ApiError && e.status === 429 ? e.message : e instanceof Error ? e.message : "The research run failed.";
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, pending: false, error: message } : x)));
      } finally {
        setRunning(false);
      }
    },
    [conversationId, refresh, router],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[780px] px-4 pb-8 pt-6">
          {notice && <Alert tone="bad">{notice}</Alert>}
          {loading ? (
            <div className="flex justify-center py-16 text-faint">
              <Spinner className="h-5 w-5" />
            </div>
          ) : turns.length === 0 ? (
            <Welcome onPick={(q) => ask(q, "text")} disabled={running} />
          ) : (
            <div className="space-y-10">
              {turns.map((t) => (
                <TurnView key={t.id} turn={t} onRetry={() => ask(t.query, t.modality, t.mode)} disabled={running} />
              ))}
            </div>
          )}
          <div ref={end} />
        </div>
      </div>
      <div className="border-t border-line bg-bg/95 px-4 pb-4 pt-3">
        <div className="mx-auto w-full max-w-[780px]">
          <Composer onSubmit={ask} disabled={running || loading} />
        </div>
      </div>
    </div>
  );
}

const ANALYSES = (["benchmark", "papers", "discrepancy", "recipe", "claim", "transfer", "plan", "review"] as const).map((tab) => ({ tab, label: FEATURE[tab] }));

/** Shortcuts from an answer into the analysis workspace, prefilled with the answer's own papers, experiments and dataset. */
function AnalyzeActions({ result }: { result: Result }) {
  const enc = encodeURIComponent;
  const papers = [...new Set((result.sources ?? []).map((s) => s.paper_id).filter((p): p is string => !!p && p.startsWith("paper:")))];
  const exps = [...new Set((result.sources ?? []).flatMap((s) => s.experiment_ids ?? []).filter((x) => x.startsWith("exp:")))];
  const dataset = result.request?.dataset;
  const model = result.request?.model;
  const href: Record<(typeof ANALYSES)[number]["tab"], string> = {
    benchmark: `/analyze?tab=benchmark${dataset ? `&bq=${enc(dataset)}` : ""}`,
    papers: `/analyze?tab=papers${papers.length ? `&papers=${enc(papers.slice(0, 4).join(","))}` : ""}`,
    discrepancy: `/analyze?tab=discrepancy${exps[0] ? `&a=${enc(exps[0])}` : ""}${exps[1] ? `&b=${enc(exps[1])}` : ""}`,
    recipe: exps[0] ? `/analyze?tab=recipe&experiment=${enc(exps[0])}` : papers[0] ? `/analyze?tab=recipe&paper=${enc(papers[0])}` : "/analyze?tab=recipe",
    claim: "/analyze?tab=claim",
    transfer: model ? `/analyze?tab=transfer&model=${enc(model)}${dataset ? `&source=${enc(dataset)}` : ""}` : "/analyze?tab=transfer",
    plan: exps[0] ? `/analyze?tab=plan&experiment=${enc(exps[0])}` : "/analyze?tab=plan",
    review: "/analyze?tab=review",
  };
  const label: Record<string, string> = {
    benchmark: dataset ? `${FEATURE.benchmark}: ${dataset}` : FEATURE.benchmark,
    papers: papers.length >= 2 ? `${FEATURE.papers}: the ${Math.min(papers.length, 4)} source papers` : FEATURE.papers,
    discrepancy: FEATURE.discrepancy,
    recipe: FEATURE.recipe,
    claim: FEATURE.claim,
    transfer: model ? `${FEATURE.transfer}: ${model}` : FEATURE.transfer,
    plan: FEATURE.plan,
    review: FEATURE.review,
  };
  return (
    <div>
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">Analyze further</h3>
      <div className="flex flex-wrap gap-2">
        {ANALYSES.map((a) => (
          <Link key={a.tab} href={href[a.tab]} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink transition hover:border-accent hover:text-accent">
            <I.Bars width={14} height={14} /> {label[a.tab]}
          </Link>
        ))}
      </div>
    </div>
  );
}

function Welcome({ onPick, disabled }: { onPick: (q: string) => void; disabled: boolean }) {
  return (
    <div className="pt-[10vh]">
      <h1 className="font-serif text-[2rem] font-semibold leading-tight text-ink">What do you want to find out?</h1>
      <p className="mt-3 max-w-[60ch] text-[0.95rem] leading-relaxed text-muted">
        ResearchX searches a curated corpus of ML papers and experiments with BM25 and vector retrieval, reranks the evidence, and only states claims it can
        quote verbatim. Every answer comes with its full research trace.
      </p>
      <div className="mt-8 grid gap-2.5 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            disabled={disabled}
            onClick={() => onPick(q)}
            className="rounded-xl border border-line bg-surface p-3.5 text-left text-sm leading-snug text-ink transition hover:border-accent disabled:opacity-60"
          >
            {q}
          </button>
        ))}
      </div>
      <h2 className="mt-8 text-xs font-medium uppercase tracking-wider text-faint">Or go straight to an analysis</h2>
      <div className="mt-2 flex flex-wrap gap-2">
        {ANALYSES.map((a) => (
          <Link key={a.tab} href={`/analyze?tab=${a.tab}`} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink transition hover:border-accent hover:text-accent">
            <I.Bars width={14} height={14} /> {a.label}
          </Link>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn, onRetry, disabled }: { turn: Turn; onRetry: () => void; disabled: boolean }) {
  return (
    <article className="rx-enter space-y-4">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-[0.95rem] leading-relaxed text-ink">
          {turn.mode && <span className="mb-1 block text-xs font-medium text-accent">{CHAT_MODES[turn.mode].label}</span>}
          {turn.query}
          {turn.modality === "voice" && <span className="ml-2 text-xs text-muted">· dictated</span>}
        </div>
      </div>
      {turn.pending && turn.mode ? (
        <section className="rounded-2xl border border-line bg-surface p-4" aria-live="polite" aria-busy="true">
          <p className="flex items-center gap-2 text-sm font-medium text-ink"><Spinner className="text-accent" /> {CHAT_MODES[turn.mode].label}…</p>
        </section>
      ) : turn.pending ? (
        <LiveRun stages={turn.live ?? []} />
      ) : turn.mode && turn.analysis ? (
        <AnalysisCard mode={turn.mode} data={turn.analysis} query={turn.query} />
      ) : turn.result ? (
        <AnswerView result={turn.result} />
      ) : turn.error ? (
        <Alert tone="bad">
          {turn.error}{" "}
          <button onClick={onRetry} disabled={disabled} className="ml-1 font-medium underline underline-offset-4">
            Try again
          </button>
        </Alert>
      ) : null}
    </article>
  );
}

function LiveRun({ stages }: { stages: Stage[] }) {
  const current = [...stages].reverse().find((s) => s.status === "running");
  return (
    <section className="rounded-2xl border border-line bg-surface p-4" aria-live="polite" aria-busy="true">
      <p className="flex items-center gap-2 text-sm font-medium text-ink">
        <Spinner className="text-accent" /> {current ? `${LABEL[current.stage] ?? current.stage}…` : stages.length ? "Working…" : "Starting research run…"}
      </p>
      {stages.length > 0 && (
        <div className="mt-3 border-t border-line pt-3">
          <TraceList stages={stages} compact />
        </div>
      )}
    </section>
  );
}

export function AnswerView({ result }: { result: Result }) {
  const [active, setActive] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [trace, setTrace] = useState<Stage[] | null>(result.stages ?? null);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const verdict = (result.evidence_status && EVIDENCE[result.evidence_status]) || VERDICT[result.status] || VERDICT.ABSTAINED;
  const sources = result.sources ?? [];
  const cited = new Set(result.cited_sources ?? []);
  const text = result.mode === "fallback" ? result.answer.replace(/^\*\*Fallback response[^*]*\*\*\s*/, "") : result.answer;
  const removed = (result.removed ?? []).filter((r) => r.reason && r.reason !== "answer_sentences_removed").length;
  const config = [result.request?.model, result.request?.dataset].filter(Boolean).join(" on ");
  const ran = (trace ?? []).filter((s) => s.status === "completed" || s.status === "failed");  // stages that actually executed
  const steps = ran.length;
  const searched = ran.some((s) => s.stage === "external_sources" || s.stage === "tavily");

  const cite = (sid: string) => {
    setActive(sid);
    document.getElementById(`${result.research_run_id}-${sid}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  async function openTrace() {
    setTraceOpen((o) => !o);
    if (!trace && result.research_run_id) {
      try {
        setTrace((await api<{ stages: Stage[] }>(`/api/research/${result.research_run_id}/trace`)).stages);
      } catch (e) {
        setTraceError(e instanceof Error ? e.message : "Trace unavailable.");
      }
    }
  }

  async function download() {
    const data = await api(`/api/research/${result.research_run_id}/trace`);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `${result.research_run_id}.json` });
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="space-y-4">
      {result.mode === "fallback" && (
        <Alert tone="warn">
          <strong>Fallback response — ResearchX retrieval pipeline unavailable.</strong> This answer was generated without consulting any sources and is
          unverified.{result.fallback_reason ? <span className="block text-xs text-muted">Reason: {result.fallback_reason}</span> : null}
        </Alert>
      )}
      <div>
        <button onClick={openTrace} aria-expanded={traceOpen} className="inline-flex items-center gap-1.5 text-sm text-muted transition hover:text-ink">
          <I.Chevron width={15} height={15} className={cx("transition", traceOpen && "rotate-90")} />
          {result.cached ? "Answered from your cache" : typeof result.total_ms === "number" ? `Researched for ${(result.total_ms / 1000).toFixed(1)} s` : "Research trace"}
          {steps ? ` · ${steps} steps` : ""}
          {searched ? " · searched outside the corpus" : ""}
        </button>
        {traceOpen && (
          <div className="rx-enter mt-2 rounded-xl border border-line bg-surface p-4">
            {traceError ? <p className="text-sm text-bad">{traceError}</p> : trace ? <TraceList stages={trace} /> : <Spinner className="text-faint" />}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className={cx("rounded-full px-2.5 py-0.5 text-xs font-medium", verdict.tone)}>{verdict.label}</span>{" "}
        {result.cached && <span className="rounded-full bg-info-soft px-2.5 py-0.5 text-xs text-muted">From your answer cache</span>}{" "}
        {config && <span className="text-xs text-muted">Configuration: {config}</span>}
      </div>

      <div className="font-serif text-[1.08rem] leading-[1.7] text-ink">{renderAnswer(text, cite)}</div>

      {result.claims?.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">Evidence spans</h3>
          <ul className="space-y-2">
            {result.claims.map((c, i) => (
              <li key={i} className="border-l border-line py-0.5 pl-3">
                {c.field && (
                  <p className="mb-0.5 flex flex-wrap items-center gap-x-2 text-xs">
                    <span className="font-medium text-ink">{c.text}</span>{" "}
                    {c.verified && (
                      <span className="inline-flex items-center gap-1 text-ok">
                        <I.Check width={12} height={12} /> {c.origin === "dataset record" ? "value present in the cited record" : "span verified in the cited source"}
                      </span>
                    )}
                  </p>
                )}
                <q className="text-sm italic leading-relaxed text-ink">{pretty(c.quote)}</q>
                <span className="ml-1.5 inline-flex gap-1 align-middle">
                  {c.sources.map((s) => (
                    <CitePill key={s} sid={s} onClick={cite} />
                  ))}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {removed > 0 && (
        <Dropdown summary={`${removed} statement${removed > 1 ? "s were" : " was"} removed: the quoted evidence could not be verified in the cited source`}>
          <ul className="space-y-1.5">
            {(result.removed ?? []).filter((r) => r.reason && r.reason !== "answer_sentences_removed").map((r, i) => (
              <li key={i} className="text-xs leading-relaxed">
                {r.text && <span className="text-ink">{r.text}</span>} <span className="text-muted">· {REMOVED_WHY[r.reason] ?? r.reason.replaceAll("_", " ")}</span>
              </li>
            ))}
          </ul>
        </Dropdown>
      )}
      {!!result.rejected_facts && <NotUsed count={result.rejected_facts} items={result.rejected_candidates} />}
      {!!result.unused_sources?.length && <UnusedSources items={result.unused_sources} />}

      {result.status === "CONFLICTING" && result.contradictions?.length > 0 && (
        <Alert tone="warn">
          <strong>Sources disagree.</strong>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {result.contradictions.slice(0, 3).map((c) => (
              <li key={c.id}>{c.explanation ?? `${c.field} differs between sources.`}</li>
            ))}
          </ul>
        </Alert>
      )}

      {sources.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">Sources</h3>
          <ol className="grid gap-2">
            {sources.map((s) => (
              <SourceCard key={s.sid} s={s} id={`${result.research_run_id}-${s.sid}`} active={active === s.sid} cited={cited.has(s.sid)} />
            ))}
          </ol>
        </div>
      )}

      {result.mode !== "fallback" && <AnalyzeActions result={result} />}

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line pt-3 text-xs text-muted">
        <button
          onClick={async () => {
            await navigator.clipboard.writeText(result.research_run_id);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="inline-flex items-center gap-1.5 font-mono hover:text-ink"
          title="Copy research run id"
        >
          {copied ? <I.Check width={14} height={14} /> : <I.Copy width={14} height={14} />} {result.research_run_id}
        </button>
        {typeof result.total_ms === "number" && <span>{(result.total_ms / 1000).toFixed(1)} s</span>}
        {result.research_run_id && (
          <button onClick={download} className="inline-flex items-center gap-1 hover:text-ink">
            <I.Download width={14} height={14} /> Trace JSON
          </button>
        )}
        {result.graph && (
          <Link href={`/map?run=${encodeURIComponent(result.research_run_id)}`} className="inline-flex items-center gap-1 font-medium text-accent hover:underline">
            <I.Graph width={14} height={14} /> Knowledge map
            {(() => {
              const found = Number((result.graph.stats?.by_origin as Record<string, number> | undefined)?.runtime_external ?? 0);
              return found ? ` · ${found} found outside the corpus` : "";
            })()}
          </Link>
        )}
      </footer>
    </section>
  );
}

/** A disclosure in the answer (like a chat assistant's "details" dropdown): one line, the details on demand. */
function Dropdown({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-line bg-surface">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full items-start gap-1.5 px-3 py-2 text-left text-xs text-muted transition hover:text-ink">
        <I.Chevron width={14} height={14} className={cx("mt-px shrink-0 transition", open && "rotate-90")} />
        {summary}
      </button>
      {open && <div className="rx-enter border-t border-line px-3 py-2.5">{children}</div>}
    </div>
  );
}

type Candidate = NonNullable<Result["rejected_candidates"]>[number];

function NotUsed({ count, items }: { count: number; items?: Candidate[] }) {
  const summary = `${count} other candidate value${count > 1 ? "s were" : " was"} found in the retrieved sources but not used`;
  if (!items?.length) return <p className="text-xs text-muted">{summary}: another model, dataset or configuration, or failed validation.</p>;
  const config = (r: Candidate) => CONFIG_REASONS.has(r.reason) || r.reason.startsWith("configuration_");
  const groups: [string, (r: Candidate) => boolean][] = [
    ["About another model, dataset or configuration", config],
    ["Failed validation against the source", (r) => !config(r)],
  ];
  return (
    <Dropdown summary={`${summary}. Why?`}>
      <div className="space-y-3">
        {groups.map(([title, test]) => {
          const rows = items.filter(test);
          if (!rows.length) return null;
          return (
            <div key={title}>
              <p className="mb-1.5 text-[0.7rem] font-semibold uppercase tracking-wider text-faint">
                {title} ({rows.length})
              </p>
              <ul className="space-y-2">
                {rows.map((r, i) => (
                  <li key={i} className="text-xs leading-relaxed">
                    <span className="font-medium text-ink">
                      {r.label} = {pretty(String(r.value))}
                    </span>{" "}
                    <span className="text-muted">· {WHY[r.reason] ?? r.reason.replaceAll("_", " ")}</span>
                    {r.context && <span className="text-muted"> · {pretty(r.context)}</span>}
                    <span className="block text-faint">
                      {r.source.url ? (
                        <a href={r.source.url} target="_blank" rel="noopener noreferrer" className="hover:text-ink hover:underline">
                          {r.source.title ?? r.source.url}
                        </a>
                      ) : (
                        (r.source.title ?? "Source")
                      )}
                      {r.source.section ? ` · ${r.source.section}` : ""}
                      {r.source.provider ? ` · ${PROVIDER[r.source.provider] ?? r.source.provider}` : ""}
                    </span>
                    {r.span && <q className="block italic text-muted">{pretty(r.span)}</q>}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
        {items.length < count && <p className="text-xs text-faint">Showing {items.length} of {count}; the research trace has the run&apos;s full record.</p>}
      </div>
    </Dropdown>
  );
}

type Unused = NonNullable<Result["unused_sources"]>[number];

/** External results found but not used, each with why: the research outside the corpus stays visible, not only what it yielded. */
function UnusedSources({ items }: { items: Unused[] }) {
  const host = (url: string | null) => {
    try {
      return url ? new URL(url).hostname : null;
    } catch {
      return null;
    }
  };
  return (
    <Dropdown summary={`${items.length} source${items.length > 1 ? "s" : ""} found outside the corpus ${items.length > 1 ? "were" : "was"} not used. Why?`}>
      <ul className="space-y-2">
        {items.map((r, i) => (
          <li key={i} className="text-xs leading-relaxed">
            <span className="font-medium text-ink">
              {r.url && r.title ? (
                <a href={r.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
                  {r.title}
                </a>
              ) : (
                (r.title ?? host(r.url) ?? "Untitled result")
              )}
            </span>{" "}
            <span className="text-muted">· {UNUSED_WHY[r.reason] ?? r.reason.replaceAll("_", " ")}</span>
            <span className="block text-faint">
              {[r.provider ? (PROVIDER[r.provider] ?? r.provider) : null, ...(r.also_found_by ?? []).map((p) => `also ${PROVIDER[p] ?? p}`), r.arxiv_id ? `arXiv ${r.arxiv_id}` : r.doi ? `DOI ${r.doi}` : null]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </li>
        ))}
      </ul>
    </Dropdown>
  );
}

function renderAnswer(text: string, cite: (sid: string) => void): ReactNode {
  const withCites = (line: string, key: string) =>
    line.split(/(\[S\d+\])/g).map((part, i) => {
      const m = /^\[(S\d+)\]$/.exec(part);
      return m ? <CitePill key={`${key}-${i}`} sid={m[1]} onClick={cite} /> : <span key={`${key}-${i}`}>{part}</span>;
    });
  // paragraphs and "- " bullet runs, in order ("For X on Y:" followed by one bullet per requested field)
  const out: ReactNode[] = [];
  let run: string[] = [];
  const flush = () => {
    if (!run.length) return;
    const k = out.length;
    out.push(
      <ul key={`u${k}`} className={cx("list-disc space-y-1.5 pl-5", k > 0 && "mt-2")}>
        {run.map((l, i) => (
          <li key={i}>{withCites(l.slice(2), `b${k}-${i}`)}</li>
        ))}
      </ul>,
    );
    run = [];
  };
  text
    .split("\n")
    .filter((l) => l.trim())
    .forEach((l, i) => {
      if (l.startsWith("- ")) return void run.push(l);
      flush();
      out.push(
        <p key={`p${i}`} className={cx(out.length > 0 && "mt-3", l.startsWith("Only part") && "font-sans text-sm text-muted")}>
          {withCites(l, `p${i}`)}
        </p>,
      );
    });
  flush();
  return out;
}

function CitePill({ sid, onClick }: { sid: string; onClick: (sid: string) => void }) {
  return (
    <button
      onClick={() => onClick(sid)}
      className="mx-0.5 inline-flex -translate-y-px items-center rounded-md border border-accent/30 bg-accent-soft px-1.5 font-sans text-[0.7rem] font-semibold leading-5 text-accent hover:border-accent"
      aria-label={`Show source ${sid}`}
    >
      {sid}
    </button>
  );
}

/** Character ranges of the evidence spans inside the displayed passage (stored offsets first, then exact search). */
function spanRanges(text: string, spans: Source["spans"]): [number, number][] {
  const out: [number, number][] = [];
  for (const sp of spans ?? []) {
    if (!sp.text) continue;
    const at = sp.start != null && sp.end != null && text.slice(sp.start, sp.end) === sp.text ? sp.start : text.indexOf(sp.text);
    if (at >= 0) out.push([at, at + sp.text.length]);
  }
  out.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const r of out) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push(r);
  }
  return merged;
}

function Excerpt({ text, spans, open }: { text: string; spans: Source["spans"]; open: boolean }) {
  const ranges = spanRanges(text, spans);
  // collapsed: a window around the evidence instead of the passage head
  const from = open || !ranges.length ? 0 : Math.max(0, ranges[0][0] - 140);
  const to = open || !ranges.length ? text.length : Math.min(text.length, ranges[ranges.length - 1][1] + 160);
  const parts: ReactNode[] = [];
  let at = from;
  ranges.forEach(([a, b], i) => {
    if (b <= from || a >= to) return;
    parts.push(pretty(text.slice(at, Math.max(at, a))));
    parts.push(
      <mark key={i} className="rounded-sm bg-warn-soft px-0.5 font-medium text-ink">
        {pretty(text.slice(Math.max(a, from), Math.min(b, to)))}
      </mark>,
    );
    at = Math.min(b, to);
  });
  parts.push(pretty(text.slice(at, to)));
  return (
    <p className={cx("mt-1.5 whitespace-pre-line text-[0.8rem] leading-relaxed text-muted", !open && !ranges.length && "line-clamp-3")}>
      {from > 0 && "… "}
      {parts}
      {to < text.length && " …"}
    </p>
  );
}

/** Per-source shortcuts into the analyses: add the source's paper to the paper diff, open its reproduction recipe, or investigate its result. */
function SourceActions({ s, paper }: { s: Source; paper: string }) {
  const tray = useTray();
  const inTray = tray.some((p) => p.id === paper);  // the passage is a shortcut to its paper: every passage of that paper shows it
  const full = !inTray && tray.length >= TRAY_MAX;
  const exp = s.experiment_ids?.find((x) => x.startsWith("exp:"));
  const enc = encodeURIComponent;
  const link = "inline-flex items-center gap-1 font-medium text-accent hover:underline";
  return (
    <>
      <button onClick={() => toggleTray({ id: paper, title: s.title ?? paper })} disabled={full}
        title={full ? `${FEATURE.papers} already holds ${TRAY_MAX} papers` : inTray ? `This paper is in ${FEATURE.papers} (click to remove)` : `Add this passage's paper to ${FEATURE.papers}`}
        className={cx(link, "disabled:cursor-not-allowed disabled:text-faint disabled:no-underline")}>
        {inTray ? <I.Check width={12} height={12} /> : <I.Plus width={12} height={12} />} {inTray ? `In ${FEATURE.papers}` : `Add to ${FEATURE.papers}`}
      </button>
      <Link href={exp ? `/analyze?tab=recipe&experiment=${enc(exp)}` : `/analyze?tab=recipe&paper=${enc(paper)}`} className={link}>
        <I.Bars width={12} height={12} /> {FEATURE.recipe}
      </Link>
      {exp && (
        <Link href={`/analyze?tab=discrepancy&a=${enc(exp)}`} className={link}>
          <I.Bars width={12} height={12} /> {FEATURE.discrepancy}
        </Link>
      )}
    </>
  );
}

function SourceCard({ s, id, active, cited }: { s: Source; id: string; active: boolean; cited: boolean }) {
  const [open, setOpen] = useState(false);
  const type =
    s.kind === "private"
      ? "Your document"
      : s.kind === "record"
        ? `Dataset record · ${s.provider ?? "import"}`
        : s.kind === "external" && s.corpus_paper_id
          ? `Corpus paper · full text read at question time${EXTERNAL_UNIT[s.unit_type ?? ""] ? ` · arXiv ${EXTERNAL_UNIT[s.unit_type ?? ""]}` : ""}`
          : s.kind === "external"
            ? `${PROVIDER[s.provider ?? ""] ?? s.provider ?? "External"}${EXTERNAL_UNIT[s.unit_type ?? ""] ? ` · arXiv ${EXTERNAL_UNIT[s.unit_type ?? ""]}` : ""}${s.also_found_by?.length ? ` · also ${s.also_found_by.map((a) => PROVIDER[a.provider] ?? a.provider).join(", ")}` : ""} · outside the corpus`
            : (UNIT[s.unit_type ?? ""] ?? "Corpus paper");
  const corpusPaper = s.kind === "internal" || s.kind === "record" ? s.paper_id : s.corpus_paper_id;  // knowledge map + diff links
  const meta = [s.year, s.section, s.kind === "internal" && s.locator ? `§ ${s.locator}` : null].filter(Boolean).join(" · ");
  const grounded = !!s.role;
  const supports = s.role === "SUPPORTS";
  const unmatched = (s.spans ?? []).filter((sp) => !spanRanges(s.text, [sp]).length);
  const cells = (s.spans ?? []).filter((sp) => sp.cell);
  const why = supports
    ? null
    : grounded
      ? s.match === "EXACT" || s.match === "STRONG"
        ? "From the matching paper, but this passage does not state the requested values for this configuration. Shown for context, not used as support."
        : s.match === "DATASET"
          ? "Another paper describing its own setup on this dataset (a different model). Shown for comparison, not used as support: add both papers to the diff to compare them."
          : "About another model or dataset. Shown for context, not used as support."
      : null;
  return (
    <li id={id} className={cx("rounded-xl border bg-surface p-3 transition", active ? "border-accent ring-2 ring-accent/20" : "border-line", grounded && !supports && "opacity-90")}>
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 rounded-md bg-surface-2 px-1.5 font-mono text-[0.7rem] font-semibold leading-5 text-ink">{s.sid}</span>
        <div className="min-w-0 flex-1">
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.7rem]">
            <span className={cx("font-medium uppercase tracking-wide", s.kind === "private" ? "text-accent" : "text-faint")}>{type}</span>{" "}
            {grounded ? (
              <span className={cx("rounded-full px-2 py-px font-medium", supports ? "bg-ok-soft text-ok" : "bg-surface-2 text-muted")}>
                {supports ? "Supports the answer" : "Related · not used as support"}
              </span>
            ) : (
              cited && <span className="rounded-full bg-ok-soft px-2 py-px font-medium text-ok">Cited in the answer</span>
            )}{" "}
            {s.match && <span className="text-faint">{MATCH[s.match]}</span>}
          </p>
          <p className="mt-0.5 text-sm font-medium leading-snug text-ink">
            {s.url ? (
              <a href={s.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-start gap-1 hover:underline">
                {s.title || s.url}
                <I.External width={13} height={13} className="mt-1 shrink-0 text-faint" />
              </a>
            ) : (
              s.title
            )}
          </p>
          {meta && <p className="text-xs text-muted">{meta}</p>}
          {s.kind === "record" && s.locator && <p className="mt-0.5 wrap-break-word font-mono text-[0.68rem] text-faint">{s.locator}</p>}
          {supports && !!s.supports?.length && (
            <ul className="mt-1.5 flex flex-wrap gap-1.5 text-xs">
              {s.supports.map((f) => (
                <li key={`${f.field}-${f.value}`} className="inline-flex items-center gap-1 rounded-md bg-ok-soft px-1.5 py-0.5 text-ok">
                  <I.Check width={12} height={12} /> {f.label} = <strong className="font-semibold">{f.value}</strong>
                </li>
              ))}
            </ul>
          )}
          {cells.map((sp) => (
            <p key={sp.cell} className="mt-1 text-xs text-muted">
              Table cell {sp.cell} = <strong className="font-semibold text-ink">{sp.value ?? sp.text}</strong>
            </p>
          ))}
          <Excerpt text={s.text} spans={s.spans} open={open} />
          {unmatched.map((sp, i) => (
            <q key={i} className="mt-1 block text-xs italic text-ink">
              {sp.text}
            </q>
          ))}
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            {s.text.length > 300 && (
              <button onClick={() => setOpen((o) => !o)} className="font-medium text-accent hover:underline">
                {open ? "Show less" : "Show full passage"}
              </button>
            )}{" "}
            {s.verification && (
              <span className="inline-flex items-center gap-1 text-ok">
                <I.Check width={12} height={12} /> {s.verification}
              </span>
            )}
            {corpusPaper && (
              <Link href={`/map?node=${encodeURIComponent(corpusPaper)}`} className="inline-flex items-center gap-1 font-medium text-accent hover:underline">
                <I.Graph width={12} height={12} /> In the knowledge map
              </Link>
            )}
            {corpusPaper && <SourceActions s={s} paper={corpusPaper} />}
          </div>
          {why && <p className="mt-1 text-xs text-muted">{why}</p>}
        </div>
      </div>
    </li>
  );
}

function summary(s: Stage): string {
  const d = (s.detail ?? {}) as Record<string, unknown>;
  if (s.status === "running") return "running…";
  if (s.status === "failed") return s.error ?? "failed";
  if (s.status === "skipped" || s.status === "not_required") return String(d.reason ?? s.status.replace("_", " "));
  const short = (m: unknown) => String(m ?? "").split("/").pop();
  const n = (v: unknown) => (Array.isArray(v) ? v.length : 0);
  const lower = (v: unknown) => String(v ?? "").toLowerCase().replaceAll("_", " ");
  switch (s.stage) {
    case "query_understanding": {
      const r = d.request as { model?: string; dataset?: string; fields?: string[] } | undefined;
      const asked = r?.fields?.length ? ` · asks ${r.fields.join(", ")}${r.model ? ` for ${r.model}` : ""}${r.dataset ? ` on ${r.dataset}` : ""}` : "";
      return `intent: ${d.intent ?? "general"} · ${n(d.entities)} known entities${asked}`;
    }
    case "experiment_extraction": {
      const m = (d.matched ?? []) as { match: string }[];
      const exact = m.filter((x) => x.match === "EXACT").length;
      return `${d.experiments_checked ?? 0} experiments checked · ${m.length} match the configuration (${exact} exact)`;
    }
    case "evidence_extraction": {
      const rt = d.runtime_extraction as { facts?: number; valid?: number } | string | undefined;
      const runtime = rt && typeof rt === "object" ? ` · runtime extraction ${rt.valid ?? 0}/${rt.facts ?? 0} valid` : "";
      const records = d.record_facts ? ` + ${d.record_facts} records` : "";
      return `${d.corpus_facts ?? 0} corpus facts${records} · ${d.valid ?? 0} pass validation · ${d.usable ?? 0} match the configuration${runtime}`;
    }
    case "model_routing":
      return Object.entries(d).map(([k, v]) => `${k}: ${short(v)}`).join(" · ");
    case "cache_lookup":
      return d.hit ? "hit: answered from your cache" : "miss";
    case "bm25":
      return `${d.candidates} candidates`;
    case "pgvector":
      return `${d.candidates} candidates · ${d.search ?? ""}`;
    case "hybrid_fusion": {
      const p = d.private_documents as Record<string, number> | string | undefined;
      const priv = p && typeof p === "object" && p.chunks_in_scope ? ` · ${p.lexical + p.dense} private hits` : "";
      return `${d.candidates} fused · overlap ${d.overlap}${priv}`;
    }
    case "reranking":
      return `${d.graded ?? 0} graded by ${short(d.model)}${d.cache === "hit" ? " · cached grades" : ""}`;
    case "evidence_assessment": {
      if (!d.status) return `${d.sufficient ? "sufficient" : "insufficient"} · ${d.grade3} decisive, ${d.grade2_plus} relevant passages`;
      const fields = Object.entries((d.fields ?? {}) as Record<string, { status: string }>).map(([k, v]) => `${k}: ${lower(v.status)}`);
      return [lower(d.status), ...fields].join(" · ");
    }
    case "external_sources": {
      const p = d.providers as Record<string, { status: string; results?: number; query?: string }> | undefined;
      if (!p) return `${d.results ?? 0} results`;
      const first = (d.queries as string[] | undefined)?.[0];
      const each = Object.entries(p).map(([k, v]) => {
        const got = v.status === "completed" || v.status === "cache_hit" ? String(v.results ?? 0) : lower(v.status);
        return `${PROVIDER[k] ?? k} ${got}${v.query && first && v.query !== first ? ` via "${v.query}"` : ""}`;
      });
      return `${first ? `searched "${first}" · ` : ""}${d.results ?? 0} results · ${each.join(" · ")}`;
    }
    case "evidence_gap":
      return `${d.second_look ? "second look: " : ""}${String(d.reason ?? "gap found")}`;
    case "laya_decision": {
      const w = d.workflow as { choice?: string; confidence?: number; gate?: string } | undefined;
      const ag = d.agreement as { rule_workflow?: string | null; workflow_agrees?: boolean | null } | undefined;
      if (!w) return String(d.reason ?? "");
      const agree = ag?.workflow_agrees == null ? "no rule counterpart" : ag.workflow_agrees ? `agrees with the rules (${ag.rule_workflow})` : `rules chose ${ag.rule_workflow}`;
      const guard = Number(d.injection ?? 0) >= 0.5 ? ` · flags possible prompt injection (${d.injection})` : "";
      return `${d.mode === "advise" ? "advise" : "shadow"} · ${w.choice} (${w.confidence}, gate ${w.gate}) · ${agree}${guard}${d.applied ? ` · applied: ${d.applied}` : " · not applied"}`;
    }
    case "tavily":
      return `${d.results ?? 0} results${d.cached ? " · cached" : ""}`;
    case "graph_enrichment":
      return `${d.runtime_nodes ?? 0} runtime + ${d.canonical_nodes ?? 0} corpus nodes · ${d.edges ?? 0} links${d.merged_by_identifier ? ` · ${d.merged_by_identifier} matched to the corpus by identifier` : ""}`;
    case "source_validation":
      return `${d.accepted}/${d.received} accepted`;
    case "external_extraction": {
      const steps = (d.extraction ?? []) as { read: string; usable?: number; facts?: number }[];
      const ft = d.fulltext as { status?: string; arxiv_id?: string; units_read?: string[]; error?: string } | string | undefined;
      const full =
        ft && typeof ft === "object"
          ? ft.status === "completed"
            ? ` · arXiv ${ft.arxiv_id} full text: ${n(ft.units_read)} passage(s) read`
            : ` · arXiv ${ft.arxiv_id} full text unavailable`
          : "";
      const read = steps.map((x) => `${x.read}: ${x.usable ?? 0}/${x.facts ?? 0} usable`).join(" · ");
      return `${read || "nothing to extract"}${full} · ${d.usable ?? 0} runtime fact(s) usable`;
    }
    case "contradiction_detection":
      if (d.found !== undefined) return `${d.found} found`;
      return `${n(d.field_conflicts) ? `values disagree for ${(d.field_conflicts as string[]).join(", ")}` : "no disagreeing values"} · ${d.recorded_contradictions ?? 0} recorded`;
    case "evidence_selection":
      if (d.supporting) return `${n(d.supporting)} supporting · ${n(d.related)} related (context only) · ${d.excluded_weak ?? 0} excluded`;
      return `${d.internal} corpus + ${d.external} external sources`;
    case "synthesis": {
      if (!d.model) return `deterministic rendering of ${d.claims ?? 0} validated fact(s), no model call`;
      const r = d.routing as { selection_reason?: string; fallback_used?: boolean } | undefined;
      const why = r?.selection_reason ? ` · ${r.selection_reason.split(";")[0]}` : "";
      return `${short(d.model)} · ${String(d.model_status ?? "").toLowerCase()}${r?.fallback_used ? " · fallback model used" : ""}${why}`;
    }
    case "citation_validation":
      if (d.total !== undefined) return `${d.verified}/${d.total} citations verified · ${d.offsets_verified ?? 0} character offsets re-checked`;
      return `${d.valid}/${d.claims} claims verified`;
    case "answer":
      return String(d.status ?? "").toLowerCase() + (d.cached ? " · cached" : "");
    default:
      return "";
  }
}

function StatusIcon({ status }: { status: Stage["status"] }) {
  const map = {
    running: <Spinner className="h-3.5 w-3.5 text-accent" />,
    completed: <I.Check width={14} height={14} className="text-ok" />,
    failed: <I.Cross width={14} height={14} className="text-bad" />,
    skipped: <I.Ring width={14} height={14} className="text-faint" />,
    not_required: <I.Dash width={14} height={14} className="text-faint" />,
  };
  return (
    <span className="mt-0.5 grid h-4 w-4 place-items-center" title={status.replace("_", " ")}>
      {map[status]}
      <span className="sr-only">{status.replace("_", " ")}</span>
    </span>
  );
}

function TraceList({ stages, compact }: { stages: Stage[]; compact?: boolean }) {
  return (
    <ol className={cx("space-y-1.5", compact && "max-h-56 overflow-y-auto pr-1")}>
      {stages.map((s, i) => (
        <li key={`${s.stage}-${i}`} className="flex gap-2.5">
          <StatusIcon status={s.status} />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-3">
              <span className={cx("text-sm", s.status === "not_required" || s.status === "skipped" ? "text-muted" : "font-medium text-ink")}>
                {LABEL[s.stage] ?? s.stage}
              </span>
              {!!s.latency_ms && <span className="shrink-0 font-mono text-[0.7rem] text-faint">{s.latency_ms} ms</span>}
            </div>
            <p className={cx("text-xs", s.status === "failed" ? "text-bad" : "text-muted")}>{summary(s)}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

type Recognition = { lang: string; interimResults: boolean; continuous: boolean; start(): void; stop(): void; onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null; onend: (() => void) | null; onerror: (() => void) | null };

function useVoice(onText: (text: string) => void) {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const rec = useRef<Recognition | null>(null);
  useEffect(() => {
    setSupported("SpeechRecognition" in window || "webkitSpeechRecognition" in window);
  }, []);
  const start = (prefix: string) => {
    const w = window as unknown as { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };
    const SR = w.SpeechRecognition ?? w.webkitSpeechRecognition;
    if (!SR) return;
    const r = new SR();
    r.lang = navigator.language || "en-US";
    r.interimResults = true;
    r.continuous = false;
    const base = prefix ? `${prefix.trimEnd()} ` : "";
    r.onresult = (e) => onText(base + Array.from(e.results, (res) => res[0].transcript).join(""));
    r.onend = () => setListening(false);
    r.onerror = () => setListening(false);
    rec.current = r;
    r.start();
    setListening(true);
  };
  return { supported, listening, start, stop: () => rec.current?.stop() };
}

function Composer({ onSubmit, disabled }: { onSubmit: (q: string, modality: "text" | "voice", mode?: ChatMode) => void; disabled: boolean }) {
  const [text, setText] = useState("");
  const [dictated, setDictated] = useState(false);
  const [mode, setMode] = useState<ChatMode | undefined>(undefined);  // undefined: a research question
  const area = useRef<HTMLTextAreaElement>(null);
  const voice = useVoice((t) => {
    setText(t);
    setDictated(true);
  });

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  useEffect(() => {  // a question handed over from the Knowledge Map (?q=): prefilled, never sent without the user
    const q = new URLSearchParams(window.location.search).get("q");
    if (q) {
      setText(q.slice(0, 2000));
      area.current?.focus();
    }
  }, []);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const q = text.trim();
    if (!q || disabled) return;
    voice.stop();
    onSubmit(q, dictated ? "voice" : "text", mode);
    setText("");
    setDictated(false);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <form onSubmit={submit} className="rounded-2xl border border-line bg-surface p-2 shadow-[0_1px_2px_rgba(0,0,0,0.04)] focus-within:border-accent">
      <div role="radiogroup" aria-label="What to do with your text" className="flex flex-wrap gap-1 px-1 pb-1">
        {([undefined, "papers", "benchmark", "claim", "transfer", "plan", "review"] as (ChatMode | undefined)[]).map((m) => (
          <button key={m ?? "ask"} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
            className={cx("rounded-full border px-2.5 py-0.5 text-xs transition", mode === m ? "border-accent bg-accent-soft font-medium text-accent" : "border-line text-muted hover:text-ink")}>
            {m ? CHAT_MODES[m].label : "Ask"}
          </button>
        ))}
      </div>
      <label htmlFor="rx-query" className="sr-only">
        {mode ? CHAT_MODES[mode].label : "Research question"}
      </label>
      <textarea
        id="rx-query"
        ref={area}
        rows={1}
        maxLength={2000}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={voice.listening ? "Listening…" : mode ? CHAT_MODES[mode].placeholder : "Ask a research question about ML experiments"}
        className="block max-h-[200px] w-full resize-none bg-transparent px-2.5 py-2 text-[0.95rem] leading-relaxed text-ink placeholder:text-faint focus:outline-none"
      />
      <div className="flex items-center justify-between gap-2 px-1 pt-1">
        <div className="flex items-center gap-2">
          {voice.supported && (
            <button
              type="button"
              onClick={() => (voice.listening ? voice.stop() : voice.start(text))}
              aria-pressed={voice.listening}
              aria-label={voice.listening ? "Stop dictation" : "Dictate your question"}
              title="Dictation uses your browser's speech recognition service"
              className={cx("rounded-lg p-2 transition", voice.listening ? "bg-bad-soft text-bad" : "text-muted hover:bg-surface-2 hover:text-ink")}
            >
              <I.Mic />
            </button>
          )}
          <span className="hidden text-xs text-faint sm:inline">Answers cite verifiable sources · Enter to send, Shift+Enter for a new line</span>
        </div>
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          aria-label="Send"
          className="grid h-9 w-9 place-items-center rounded-xl bg-accent text-accent-ink transition hover:opacity-90 disabled:opacity-40"
        >
          {disabled ? <Spinner /> : <I.Send />}
        </button>
      </div>
    </form>
  );
}
