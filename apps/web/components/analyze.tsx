"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, authRedirect, type Stage } from "@/lib/api";
import { Alert, Button, cx, Spinner } from "@/components/ui";
import * as I from "@/components/icons";
import { FEATURE } from "@/lib/features";

// Analysis workspace (features over the shared evidence layer, docs/ARCHITECTURE.md §66). Every value shown comes from a
// corpus record with its status and evidence; derived verdicts say which rule produced them; unknown stays unknown.

type Evidence = { id: string; field?: string; text?: string | null; text_withheld?: boolean; source_location?: string | null; section?: string | null; table?: string | null; cell?: string | null };
type Fact = { field: string; label: string; status: string; value: unknown; raw?: string | null; unit?: string | null; derivation?: string | null; evidence?: Evidence[];
  stage?: string | null };
type Base = { research_run_id: string; status: string; stages: Stage[]; total_ms: number; notes?: string[] };
type NodeRef = { id: string; label: string; type?: string; meta?: Record<string, unknown> };
type SearchHit = { id: string; label: string; type: string; meta?: Record<string, unknown> };

const TABS = [
  { id: "benchmark", label: FEATURE.benchmark, blurb: "Reported results for one benchmark and metric, classified by comparability (split, protocol, metric definition, pre-training data, adaptation) before anything is ranked; papers outside the corpus on request." },
  { id: "papers", label: FEATURE.papers, blurb: "Two to four papers side by side, field by field and stage by stage (pre-training, fine-tuning, evaluation), with a fairness verdict; name them by title, DOI, arXiv id or URL, including papers outside the corpus." },
  { id: "discrepancy", label: FEATURE.discrepancy, blurb: "Verified differences, missing information and clearly labelled hypotheses for two experiments; training settings compared within one stage only." },
  { id: "recipe", label: FEATURE.recipe, blurb: "An experiment's setup with the status, stage and source of every field, its pre-training and fine-tuning configurations, as a checklist and a validated export." },
  { id: "claim", label: FEATURE.claim, blurb: "Supporting, contradicting and inconclusive records for a stated claim, with comparability problems and overgeneralization flags." },
  { id: "transfer", label: FEATURE.transfer, blurb: "Where else a model has been evaluated: direct transfers within one paper, other papers' results, untested benchmarks, and the setup differences between them." },
  { id: "plan", label: FEATURE.plan, blurb: "Ablations, missing baselines, robustness tests and reporting fixes derived from the records around one experiment." },
  { id: "review", label: FEATURE.review, blurb: "What a research question or draft review covers, and the benchmarks, model families and conflicts the corpus suggests it misses." },
] as const;
type TabId = (typeof TABS)[number]["id"];

const TONE: Record<string, string> = {
  REPORTED: "bg-ok-soft text-ok", DERIVED: "bg-info-soft text-accent", CONFLICTING: "bg-warn-soft text-warn", UNKNOWN: "bg-surface-2 text-faint",
  NOT_APPLICABLE: "bg-surface-2 text-faint", VARIES: "bg-warn-soft text-warn", SAME: "bg-ok-soft text-ok", DIFFERENT: "bg-info-soft text-accent",
  PARTIAL: "bg-surface-2 text-muted", COMPARABLE: "bg-ok-soft text-ok", CONDITIONALLY_COMPARABLE: "bg-warn-soft text-warn",
  INCOMPATIBLE: "bg-bad-soft text-bad", NOT_COMPARABLE: "bg-bad-soft text-bad", HYPOTHESIS: "bg-warn-soft text-warn", LOW: "bg-warn-soft text-warn",
  MODERATE: "bg-info-soft text-accent", UNDETERMINED: "bg-surface-2 text-muted", UNEXPLAINED: "bg-warn-soft text-warn",
  DIFFERENCES_FOUND: "bg-info-soft text-accent", NOT_A_DISCREPANCY: "bg-surface-2 text-muted", CONFLICTING_RECORDS: "bg-warn-soft text-warn",
  SUPPORTED: "bg-ok-soft text-ok", SUPPORTS: "bg-ok-soft text-ok", CONTRADICTED: "bg-bad-soft text-bad", CONTRADICTS: "bg-bad-soft text-bad",
  MIXED: "bg-warn-soft text-warn", INCONCLUSIVE: "bg-surface-2 text-muted", INSUFFICIENT_EVIDENCE: "bg-surface-2 text-muted", NO_EVIDENCE: "bg-surface-2 text-faint",
  OBSERVED: "bg-ok-soft text-ok", NO_DIRECT_TRANSFER: "bg-surface-2 text-muted", GAPS_FOUND: "bg-warn-soft text-warn", NO_GAPS_IN_CORPUS_SCOPE: "bg-ok-soft text-ok",
  reproduce: "bg-warn-soft text-warn", ablation: "bg-info-soft text-accent", missing_baseline: "bg-info-soft text-accent", robustness: "bg-ok-soft text-ok",
  reporting: "bg-surface-2 text-muted", UNVERIFIED: "bg-warn-soft text-warn", runtime: "bg-info-soft text-accent", corpus: "bg-surface-2 text-muted",
  ambiguous: "bg-warn-soft text-warn", not_found: "bg-surface-2 text-faint", unsupported: "bg-bad-soft text-bad", completed: "bg-ok-soft text-ok",
  failed: "bg-bad-soft text-bad",
};
const BAR: Record<string, string> = { COMPARABLE: "var(--color-ok)", CONDITIONALLY_COMPARABLE: "var(--color-warn)", INCOMPATIBLE: "var(--color-bad)", UNKNOWN: "var(--color-faint)" };
const words = (s: string) => s.toLowerCase().replaceAll("_", " ");
// what was searched before a setting is shown as not stated (services/analysis/enrich.py)
const NOT_STATED_TITLE = "Searched: the experiment record, the paper's extracted text statements and, for training settings, seeds, splits, protocols, metric definitions and dataset versions, the paper's full text read at request time.";
const STAGE_TONE: Record<string, string> = { pretraining: "border-accent/40 text-accent", finetuning: "border-ok/40 text-ok", evaluation: "border-line text-muted", unspecified: "border-dashed border-line text-faint" };

export function Badge({ status, children }: { status: string; children?: ReactNode }) {
  return <span className={cx("inline-flex shrink-0 items-center rounded px-1.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide", TONE[status] ?? "bg-surface-2 text-muted")}>{children ?? words(status)}</span>;
}

function shown(v: unknown, unit?: string | null): string {
  if (v === null || v === undefined || v === "") return "—";
  const s = typeof v === "object" ? JSON.stringify(v) : String(v);
  return unit && !s.includes(unit) ? `${s} ${unit}` : s;
}

function EvidenceList({ evidence }: { evidence?: Evidence[] }) {
  if (!evidence?.length) return null;
  return (
    <details className="mt-1 text-xs">
      <summary className="cursor-pointer text-accent">Evidence ({evidence.length})</summary>
      <ul className="mt-1 space-y-1.5">
        {evidence.map((e) => (
          <li key={e.id} className="rounded border border-line bg-bg px-2 py-1.5">
            <div className="flex flex-wrap items-center gap-x-2 text-faint">
              {[e.section, e.table, e.cell].filter(Boolean).join(" · ") || "source"}
              {e.source_location?.startsWith("http") && (
                <a href={e.source_location} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 text-accent hover:underline">
                  open <I.External width={11} height={11} />
                </a>
              )}
            </div>
            <p className="mt-0.5 text-ink">{e.text_withheld || !e.text ? <em className="text-muted">Text withheld by the source&apos;s licence; open the location.</em> : `“${e.text}”`}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Stages({ r }: { r: Base }) {
  return (
    <details className="mt-6 rounded-lg border border-line bg-surface px-3 py-2 text-sm">
      <summary className="cursor-pointer text-muted">
        Research trace · {r.stages.length} stages · {r.total_ms} ms · <span className="font-mono text-xs">{r.research_run_id}</span>
      </summary>
      <ol className="mt-2 space-y-1">
        {r.stages.map((s, i) => (
          <li key={`${s.stage}-${i}`} className="flex flex-wrap items-baseline gap-x-2 text-xs">
            <span className={cx("font-medium", s.status === "failed" ? "text-bad" : "text-ink")}>{words(s.stage)}</span>
            <span className="text-faint">{words(s.status)}{s.latency_ms ? ` · ${s.latency_ms} ms` : ""}</span>
            <span className="text-muted">
              {Object.entries(s.detail ?? {})
                .filter(([, v]) => typeof v === "number" || (typeof v === "string" && v.length < 90))
                .map(([k, v]) => `${words(k)}: ${v}`)
                .join(" · ")}
            </span>
            {s.error && <span className="text-bad">{s.error}</span>}
          </li>
        ))}
      </ol>
      <p className="mt-2 text-xs text-faint">Stages are the ones that actually ran, with their real timings; the trace is visible only to you.</p>
    </details>
  );
}

function Notes({ notes }: { notes?: string[] }) {
  return notes?.length ? (
    <ul className="mt-4 list-disc space-y-0.5 pl-5 text-xs text-muted">
      {notes.map((n) => <li key={n}>{n}</li>)}
    </ul>
  ) : null;
}

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

function useRun<T>() {
  const router = useRouter();
  const [data, setData] = useState<T | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = useCallback(
    async (path: string, init?: RequestInit) => {
      setBusy(true);
      setError(null);
      try {
        const r = await api<T>(path, init);
        setData(r);
        return r;
      } catch (e) {
        const to = authRedirect(e);
        if (to) router.replace(to);
        else setError(e instanceof Error ? e.message : "The analysis failed.");
        return null;
      } finally {
        setBusy(false);
      }
    },
    [router],
  );
  return { data, setData, busy, error, run };
}

function Picker({ types, placeholder, onPick, exclude = [] }: { types: string; placeholder: string; onPick: (h: SearchHit) => void; exclude?: string[] }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  useEffect(() => {
    if (q.trim().length < 2) return setHits([]);
    const t = setTimeout(async () => {
      try {
        const r = await api<{ results: SearchHit[] }>(`/api/graph/search?q=${encodeURIComponent(q.trim())}&types=${types}&limit=8`);
        setHits(r.results.filter((h) => !exclude.includes(h.id)));
      } catch {
        setHits([]);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [q, types, exclude]);
  return (
    <div className="relative">
      <label className="relative block">
        <span className="sr-only">{placeholder}</span>
        <I.Search className="pointer-events-none absolute left-2.5 top-2.5 text-faint" width={16} height={16} />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={placeholder}
          className="w-full rounded-lg border border-line bg-surface py-2 pl-8 pr-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
      </label>
      {hits.length > 0 && (
        <ul className="absolute z-20 mt-1 max-h-72 w-full overflow-y-auto rounded-lg border border-line bg-surface p-1 shadow-lg">
          {hits.map((h) => (
            <li key={h.id}>
              <button onClick={() => (onPick(h), setQ(""), setHits([]))} className="w-full rounded-md px-2 py-1.5 text-left text-sm hover:bg-surface-2">
                <span className="text-ink">{h.label}</span>
                <span className="ml-2 text-xs text-faint">{[h.meta?.year, h.meta?.paper].filter(Boolean).join(" · ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Analyze() {
  const params = useSearchParams();
  const tab = (TABS.some((t) => t.id === params.get("tab")) ? params.get("tab") : "benchmark") as TabId;
  // native history: Next.js keeps useSearchParams in sync, with no server round trip per selection (router.push could
  // leave the navigation uncommitted while a previous transition was pending: "Investigate" did nothing, 2026-10-10)
  const go = useCallback((next: Record<string, string | null>) => {
    const p = new URLSearchParams(window.location.search);
    Object.entries(next).forEach(([k, v]) => (v === null ? p.delete(k) : p.set(k, v)));
    window.history.pushState(null, "", `/analyze?${p.toString()}`);
  }, []);
  const current = TABS.find((t) => t.id === tab)!;
  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl px-4 py-6">
        <h1 className="font-serif text-2xl font-semibold text-ink">Analyze</h1>
        <p className="mt-1 text-sm text-muted">Evidence-grounded analyses over the ResearchX corpus. Every value carries its status and source; nothing unknown is filled in.</p>
        <div role="tablist" aria-label="Analyses" className="mt-4 flex flex-wrap gap-1 border-b border-line">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => go({ tab: t.id })}
              className={cx("-mb-px border-b-2 px-3 py-2 text-sm transition", tab === t.id ? "border-accent font-medium text-ink" : "border-transparent text-muted hover:text-ink")}>
              {t.label}
            </button>
          ))}
        </div>
        <p className="mt-3 text-sm text-muted">{current.blurb}</p>
        <div className="mt-4">
          {tab === "benchmark" && <BenchmarkTab params={params} go={go} />}
          {tab === "papers" && <PapersTab params={params} go={go} />}
          {tab === "discrepancy" && <DiscrepancyTab params={params} go={go} />}
          {tab === "recipe" && <RecipeTab params={params} go={go} />}
          {tab === "claim" && <ClaimTab params={params} go={go} />}
          {tab === "transfer" && <TransferTab params={params} go={go} />}
          {tab === "plan" && <PlanTab params={params} go={go} />}
          {tab === "review" && <ReviewTab />}
        </div>
      </div>
    </main>
  );
}

type TabProps = { params: URLSearchParams; go: (next: Record<string, string | null>) => void };

// ---------------------------------------------------------------------------------------------- Benchmark Reality Explorer
type Catalog = { benchmarks: { id: string; name: string; papers: number; metrics: { id: string; name: string; results: number; papers: number }[] }[] };
type Cond = { status: string; value: unknown; basis?: string | null; note?: string };
type BRow = { result: string; experiment: string; paper: string; paper_title?: string | null; year?: number | null; model: string; family: string; score: string;
  value: number | null; conditions: Record<string, Cond>; class: string; why: string; rank?: number; evidence: Evidence[]; quality_tier: string; origin: string;
  runtime?: { provider?: string | null; access?: string | null; url?: string | null; metric_as_written?: string | null; dataset_as_written?: string | null } };
type BResult = Base & { benchmark: NodeRef & { name: string }; metric: { id: string; name: string }; rows: BRow[]; summary: Record<string, number>;
  reference: { kind: string; conditions: Record<string, string>; support?: number; result?: string }; options: { families: string[]; years: number[] };
  runtime?: RuntimeSummary };
const COND_LABEL: Record<string, string> = { split: "split", "evaluation.evaluation_protocol": "protocol", "evaluation.metric_definition": "metric definition",
  "dataset.dataset_version": "dataset version", "dataset.preprocessing": "preprocessing", pretraining_data: "pre-training data", adaptation: "adaptation" };

function BenchmarkTab({ params, go }: TabProps) {
  const [catalog, setCatalog] = useState<Catalog["benchmarks"] | null>(null);
  const { data, busy, error, run } = useRun<BResult>();
  const [sel, setSel] = useState<BRow | null>(null);
  const [question, setQuestion] = useState("");
  const fromQuestion = useRef<string | null>(null);  // ids set from a question's answer: shown already, not fetched again
  const benchmark = params.get("benchmark");
  const metric = params.get("metric");
  const family = params.get("family");
  const reference = params.get("reference");
  const external = params.get("external") === "1";
  useEffect(() => {
    api<Catalog>("/api/analysis/benchmarks").then((c) => setCatalog(c.benchmarks)).catch(() => setCatalog([]));
  }, []);
  const bench = catalog?.find((b) => b.id === benchmark) ?? null;
  useEffect(() => {  // a benchmark chosen elsewhere (Knowledge Map dataset node) without a metric: its most reported metric
    const b = benchmark && !metric ? catalog?.find((x) => x.id === benchmark) : null;
    if (b) go({ metric: b.metrics[0].id });
  }, [catalog, benchmark, metric, go]);
  useEffect(() => {
    if (!catalog?.length || benchmark || params.get("q")) return;
    const name = (params.get("bq") ?? "").toLowerCase();  // a dataset named in an answer: the catalog entry with that name, if any
    const hit = (name && (catalog.find((b) => b.name.toLowerCase() === name) ?? catalog.find((b) => b.name.toLowerCase().includes(name)))) || catalog[0];
    go({ benchmark: hit.id, metric: hit.metrics[0].id, bq: null });
  }, [catalog, benchmark, go, params]);
  const ask = useCallback(async (text: string) => {  // "Which ImageNet top-1 results are comparable ...?": the server names the benchmark and metric
    const q = new URLSearchParams({ q: text });
    if (external) q.set("external", "true");
    const r = await run(`/api/analysis/benchmark?${q.toString()}`);
    if (r) {
      fromQuestion.current = `${r.benchmark.id}|${r.metric.id}|${external}`;
      go({ benchmark: r.benchmark.id, metric: r.metric.id, reference: null, family: null, q: null });
    }
  }, [external, go, run]);
  useEffect(() => {  // a question handed over in the URL (from the chat)
    const q = params.get("q");
    if (q && !benchmark) void ask(q);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!benchmark || !metric) return;
    setSel(null);
    if (fromQuestion.current === `${benchmark}|${metric}|${external}` && !family && !reference) {
      fromQuestion.current = null;
      return;
    }
    const q = new URLSearchParams({ benchmark, metric });
    if (family) q.set("family", family);
    if (reference) q.set("reference", reference);
    if (external) q.set("external", "true");
    void run(`/api/analysis/benchmark?${q.toString()}`);
  }, [benchmark, metric, family, reference, external, run]);

  const exportRows = (fmt: "csv" | "json") => {
    if (!data) return;
    const rows = data.rows.map((r) => ({ result_id: r.result, experiment_id: r.experiment, paper_id: r.paper, paper: r.paper_title, year: r.year, model: r.model,
      score: r.score, comparability: r.class, rank: r.rank ?? null, reason: r.why, origin: r.origin, quality_tier: r.quality_tier,
      ...Object.fromEntries(Object.entries(r.conditions).map(([k, c]) => [COND_LABEL[k], c.value ?? null])),
      evidence: r.evidence.map((e) => e.source_location).filter(Boolean).join(" | ") }));
    const name = `researchx-${data.benchmark.name}-${data.metric.name}`.replace(/[^\w.-]+/g, "_");
    if (fmt === "json") return download(`${name}.json`, JSON.stringify({ benchmark: data.benchmark, metric: data.metric, reference: data.reference, research_run_id: data.research_run_id, rows }, null, 2), "application/json");
    const head = Object.keys(rows[0] ?? {});
    const esc = (v: unknown) => `"${String(v ?? "").replaceAll('"', '""')}"`;
    download(`${name}.csv`, [head.join(","), ...rows.map((r) => head.map((h) => esc((r as Record<string, unknown>)[h])).join(","))].join("\n"), "text/csv");
  };

  return (
    <section>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <span className="block text-xs text-muted">Benchmark</span>
          <select value={benchmark ?? ""} onChange={(e) => { const b = catalog?.find((x) => x.id === e.target.value); go({ benchmark: e.target.value, metric: b?.metrics[0]?.id ?? null, reference: null, family: null }); }}
            className="mt-1 max-w-[18rem] rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink">
            {catalog === null ? <option>Loading…</option> : catalog.map((b) => <option key={b.id} value={b.id}>{b.name} ({b.papers} papers)</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="block text-xs text-muted">Metric</span>
          <select value={metric ?? ""} onChange={(e) => go({ metric: e.target.value, reference: null })} className="mt-1 max-w-[16rem] rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink">
            {(bench?.metrics ?? []).map((m) => <option key={m.id} value={m.id}>{m.name} ({m.results})</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="block text-xs text-muted">Model family</span>
          <select value={family ?? ""} onChange={(e) => go({ family: e.target.value || null })} className="mt-1 max-w-[12rem] rounded-lg border border-line bg-surface px-2 py-1.5 text-sm text-ink">
            <option value="">All</option>
            {(data?.options.families ?? []).map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        {reference && <Button variant="secondary" onClick={() => go({ reference: null })}>Reset reference</Button>}
        {data && (
          <div className="ml-auto flex gap-2">
            <Button variant="secondary" onClick={() => exportRows("csv")}><I.Download width={15} height={15} /> CSV</Button>
            <Button variant="secondary" onClick={() => exportRows("json")}><I.Download width={15} height={15} /> JSON</Button>
          </div>
        )}
      </div>
      <form className="mt-3 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); if (question.trim().length >= 3) void ask(question.trim()); }}>
        <label className="min-w-[16rem] flex-1">
          <span className="sr-only">Ask in words</span>
          <input value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={400}
            placeholder="Or ask: which ImageNet top-1 results are actually comparable after pre-training data, fine-tuning, split and protocol?"
            className="w-full rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink placeholder:text-faint focus:border-accent" />
        </label>
        <Button type="submit" variant="secondary" disabled={busy || question.trim().length < 3}>Ask</Button>
        <ExternalToggle checked={external} onChange={(v) => go({ external: v ? "1" : null })}
          note="Finds papers outside the corpus that report this benchmark (from its name) and reads them now: slower, only the benchmark and metric names are searched." />
      </form>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && external && <p className="mt-2 text-xs text-muted">Reading papers outside the corpus can take up to a minute.</p>}
      {busy && !data && <div className="flex justify-center py-10 text-faint"><Spinner /></div>}
      {data && (
        <div className={cx("mt-4 transition-opacity", busy && "opacity-60")}>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {(["comparable", "conditionally_comparable", "incompatible", "unknown"] as const).map((k) => (
              <span key={k} className="inline-flex items-center gap-1.5"><Badge status={k.toUpperCase()} /> {data.summary[k] ?? 0}</span>
            ))}
            <span className="text-muted">· {data.summary.ranked} ranked of {data.summary.results} reported results</span>
          </div>
          <p className="mt-2 text-sm text-muted">
            Reference conditions:{" "}
            {Object.keys(data.reference.conditions).length
              ? Object.entries(data.reference.conditions).map(([k, v]) => `${k} = ${v}`).join(", ")
              : "none (no result states its split)"}
            {data.reference.kind === "chosen result" ? " · from the result you chose" : data.reference.support ? ` · stated by ${data.reference.support} result(s)` : ""}.
            {data.summary.ranked === 0 && " Nothing is ranked: no result states the same split and protocol as the reference."}
          </p>
          <BarChart rows={data.rows} selected={sel?.result ?? null} onSelect={setSel} />
          {sel && <ResultDetail row={sel} onReference={() => go({ reference: sel.result })} onClose={() => setSel(null)} />}
          <ResultsTable rows={data.rows} onSelect={setSel} selected={sel?.result ?? null} />
          {data.runtime && <RuntimePanel rt={data.runtime} />}
          <Notes notes={data.notes} />
          <Stages r={data} />
        </div>
      )}
    </section>
  );
}

function BarChart({ rows, selected, onSelect }: { rows: BRow[]; selected: string | null; onSelect: (r: BRow) => void }) {
  const bars = rows.filter((r) => r.value !== null).slice(0, 60);
  const groups = useMemo(() => {
    const m = new Map<string, BRow[]>();
    bars.forEach((r) => m.set(r.model, [...(m.get(r.model) ?? []), r]));
    return [...m.entries()];
  }, [bars]);
  if (!bars.length) return <p className="mt-4 text-sm text-muted">No numeric result to chart.</p>;
  const max = Math.max(...bars.map((r) => r.value as number), 0);
  const min = Math.min(...bars.map((r) => r.value as number), 0);
  const H = 220, bw = 14, gap = 4, gpad = 14, top = 16, left = 44;
  const W = left + groups.reduce((w, [, rs]) => w + rs.length * (bw + gap) + gpad, 0) + 10;
  const y = (v: number) => top + H - ((v - min) / (max - min || 1)) * H;
  let x = left;
  return (
    <figure className="mt-4 overflow-x-auto rounded-lg border border-line bg-surface p-2">
      <svg width={W} height={top + H + 96} role="img" aria-label="Reported results grouped by model; colour shows comparability with the reference">
        {[min, (min + max) / 2, max].map((t) => (
          <g key={t}>
            <line x1={left - 4} x2={W} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
            <text x={left - 6} y={y(t) + 4} textAnchor="end" fontSize="10" fill="var(--color-muted)">{Number(t.toFixed(2))}</text>
          </g>
        ))}
        {groups.map(([model, rs]) => {
          const gx = x;
          x += rs.length * (bw + gap) + gpad;
          return (
            <g key={model}>
              {rs.map((r, i) => {
                const bx = gx + i * (bw + gap);
                const v = r.value as number;
                return (
                  <g key={r.result} role="button" tabIndex={0} aria-label={`${r.model}: ${r.score}, ${words(r.class)}`} className="cursor-pointer outline-none"
                    onClick={() => onSelect(r)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(r)}>
                    <rect x={bx} y={Math.min(y(v), y(0))} width={bw} height={Math.max(2, Math.abs(y(0) - y(v)))} rx="2" fill={BAR[r.class]}
                      opacity={selected && selected !== r.result ? 0.45 : 1} stroke={selected === r.result ? "var(--color-ink)" : "none"} strokeWidth="1.5" />
                    <title>{`${r.model} · ${r.score} · ${r.paper_title ?? r.paper} (${r.year ?? "year unknown"}) · ${words(r.class)}`}</title>
                  </g>
                );
              })}
              <text transform={`translate(${gx + 2},${top + H + 10}) rotate(40)`} fontSize="10" fill="var(--color-muted)">{model.length > 22 ? model.slice(0, 21) + "…" : model}</text>
            </g>
          );
        })}
      </svg>
      <figcaption className="flex flex-wrap gap-3 px-1 pt-1 text-xs text-muted">
        {Object.entries(BAR).map(([k, c]) => <span key={k} className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />{words(k)}</span>)}
        <span>· one bar per reported result, grouped by model; heights are the reported values on one metric (never rescaled).</span>
      </figcaption>
    </figure>
  );
}

function ResultDetail({ row, onReference, onClose }: { row: BRow; onReference: () => void; onClose: () => void }) {
  return (
    <div className="mt-3 rounded-lg border border-accent/40 bg-surface p-3 text-sm">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="font-medium text-ink">{row.model} · <span className="font-mono">{row.score}</span> <Badge status={row.class} /> {row.runtime && <Badge status="runtime">runtime</Badge>}</p>
          <p className="text-xs text-muted">{row.paper_title ?? row.paper}{row.year ? ` (${row.year})` : ""} · {row.origin} · {row.quality_tier.toLowerCase()} record</p>
          {row.runtime && (
            <p className="text-xs text-muted">
              Found by {row.runtime.provider ?? "a scholarly provider"} · {row.runtime.access}
              {row.runtime.metric_as_written ? ` · metric as written: “${row.runtime.metric_as_written}”` : ""}
              {row.runtime.dataset_as_written ? ` · dataset as written: “${row.runtime.dataset_as_written}”` : ""}
              {row.runtime.url && <a href={row.runtime.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">paper</a>}
            </p>
          )}
        </div>
        <button onClick={onClose} aria-label="Close details" className="rounded p-1 text-faint hover:bg-surface-2"><I.Cross width={15} height={15} /></button>
      </div>
      <p className="mt-2 text-muted">{row.why}</p>
      <table className="mt-2 w-full text-xs">
        <tbody>
          {Object.entries(row.conditions).map(([k, c]) => (
            <tr key={k} className="border-t border-line align-top">
              <td className="py-1 pr-2 text-muted">{COND_LABEL[k]}</td>
              <td className="py-1 pr-2 text-ink">{shown(c.value)}{(c.basis || c.note) && <span className="block text-[0.7rem] text-faint">{c.note ?? c.basis}</span>}</td>
              <td className="py-1"><Badge status={c.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <EvidenceList evidence={row.evidence} />
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" onClick={onReference}>Use as reference</Button>
        <Link href={`/analyze?tab=recipe&experiment=${encodeURIComponent(row.experiment)}`} className="rounded-lg border border-line px-3 py-2 text-sm text-ink hover:bg-surface-2">{FEATURE.recipe}</Link>
        {!row.runtime && <Link href={`/analyze?tab=discrepancy&a=${encodeURIComponent(row.experiment)}`} className="rounded-lg border border-line px-3 py-2 text-sm text-ink hover:bg-surface-2">{FEATURE.discrepancy}</Link>}
        {!row.runtime && <Link href={`/map?node=${encodeURIComponent(row.paper)}`} className="rounded-lg border border-line px-3 py-2 text-sm text-ink hover:bg-surface-2">Paper in the Knowledge Map</Link>}
      </div>
    </div>
  );
}

function ResultsTable({ rows, onSelect, selected }: { rows: BRow[]; onSelect: (r: BRow) => void; selected: string | null }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-lg border border-line">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-surface-2 text-xs text-muted">
          <tr>{["#", "Model", "Paper", "Split", "Protocol", "Reported", "Comparability"].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.result} onClick={() => onSelect(r)} className={cx("cursor-pointer border-t border-line hover:bg-surface-2", selected === r.result && "bg-accent-soft")}>
              <td className="px-2 py-1.5 text-faint">{r.rank ?? "—"}</td>
              <td className="px-2 py-1.5 text-ink">{r.model} {r.runtime && <Badge status="runtime">runtime</Badge>}</td>
              <td className="max-w-[16rem] truncate px-2 py-1.5 text-muted" title={r.paper_title ?? ""}>{r.paper_title ?? r.paper} {r.year ? `(${r.year})` : ""}</td>
              <td className="px-2 py-1.5">{shown(r.conditions.split?.value)}</td>
              <td className="px-2 py-1.5">{shown(r.conditions["evaluation.evaluation_protocol"]?.value)}</td>
              <td className="px-2 py-1.5 font-mono">{r.score}</td>
              <td className="px-2 py-1.5"><Badge status={r.class} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- Paper-to-Paper Diff
type CellValue = { value: unknown; unit?: string | null; status: string; scope?: string; stage?: string; basis?: string | null; applies_to?: string | null;
  experiment_count: number; evidence: Evidence[] };
type Cell = { status: string; values: CellValue[]; more_values: number };
type Named = { id: string; name: string; count?: number };
type CfgValue = { value?: unknown; raw?: unknown; status: string; basis?: string | null; evidence?: Evidence[] };
type CfgField = { field: string; label: string; status: string; values: CfgValue[] };
type Cfg = { id: string; stage: string; stage_label?: string; label: string; scope?: string | null; fields: CfgField[]; linked_experiments?: number; link?: string | null };
type Inventory = { pretraining: Cfg[]; finetuning: Cfg[]; not_linked: { id: string; stage: string; label: string; why: string }[]; evaluation: number;
  evaluation_settings?: string[]; summary: string; multiple: { pretraining: boolean; finetuning: boolean } };
type Candidate = { title?: string | null; year?: number | null; url?: string | null; doi?: string | null; arxiv_id?: string | null; provider?: string | null;
  title_match?: number; paper_id?: string; in_corpus?: boolean };
type Resolution = { input: string; kind: string; why?: string; matched_by?: string; paper_id?: string; candidates?: Candidate[] };
type RuntimePaper = { key: string; title?: string | null; provider?: string | null; access?: string; license?: string | null; url?: string | null; retrieved_at?: string | null;
  error?: string | null; cached?: boolean; status?: string; experiments: number; extraction?: Record<string, unknown> };
type RuntimeSummary = { papers: RuntimePaper[]; resolutions: Resolution[]; notes: string[];
  discovery?: { topic: string; candidates: Candidate[]; why: string; provider_error?: string | null } | null };
type DiffPaper = { id: string; title?: string | null; year?: number | null; url?: string | null; experiments: number; abstract?: string | null;
  results: { label: string; metric?: string | null; score?: string | null; split?: string | null }[]; passages: { section?: string | null; text?: string | null; text_withheld: boolean; url?: string | null }[];
  covers?: Record<"benchmarks" | "models" | "tasks" | "metrics", Named[]>; venue?: string | null; authors?: string[]; origin?: string;
  provenance?: RuntimePaper | null; configurations?: Inventory };
type DiffResult = Base & {
  kind: "paper_diff"; verdict: string; papers: DiffPaper[];
  shared?: Record<"benchmarks" | "models" | "tasks" | "metrics", Named[]>; citations?: { citing: string; cited: string }[];
  shared_results?: { benchmark: string; metric: string; cells: { model: string; score: string; split?: string | null }[][]; more: number[] }[];
  sections: { name: string; rows: { field: string; stage?: string; label: string; status: string; cells: Cell[] }[] }[];
  fairness: { papers: string[]; verdict: string; why: string; shared_benchmarks: string[]; missing?: string[];
    stage_differences?: { stage_label: string; label: string; value_a: unknown; value_b: unknown }[]; stage_not_compared?: { stage_label: string; label: string; rationale: string }[] }[];
  gaps: { paper: string; fields: string[] }[]; summary: Record<string, number>; runtime?: RuntimeSummary;
};
type InventoryResult = Base & { kind: "paper_configurations"; paper: { id: string; title?: string | null; year?: number | null; url?: string | null; origin: string; provenance?: RuntimePaper | null };
  configurations: Inventory; experiments: { experiment: string; label: string; conditions: Record<string, CfgField>; seed: CfgField;
    lineage: { relation: string; label?: string; basis?: string; status?: string }[] }[]; more_experiments: number;
  unknown: { label: string; experiments: number; of: number }[]; summary: Record<string, unknown>; runtime?: RuntimeSummary };
type ResolutionResult = Base & { kind: "paper_resolution"; runtime: RuntimeSummary };
type PapersResult = DiffResult | InventoryResult | ResolutionResult;

/** Opt-in for papers outside the corpus: only what the user typed reaches the scholarly providers. */
function ExternalToggle({ checked, onChange, note }: { checked: boolean; onChange: (v: boolean) => void; note: string }) {
  return (
    <label className="inline-flex items-center gap-2 text-sm text-muted" title={note}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> Include papers outside the corpus
    </label>
  );
}

/** How each typed reference was resolved, which runtime papers were read, and what could not be used. */
function RuntimePanel({ rt, onUse }: { rt: RuntimeSummary; onUse?: (ref: string) => void }) {
  const use = (c: Candidate) => c.arxiv_id ? `arXiv:${c.arxiv_id}` : c.doi ? `doi:${c.doi}` : c.paper_id ?? c.title ?? "";
  return (
    <div className="mt-4 rounded-lg border border-line bg-surface p-3 text-sm">
      <h3 className="font-semibold text-ink">Papers named and papers read at request time</h3>
      {rt.resolutions.length > 0 && (
        <ul className="mt-2 space-y-2">
          {rt.resolutions.map((r, i) => (
            <li key={i}>
              <span className="text-ink">“{r.input}”</span> <Badge status={r.kind}>{words(r.kind)}</Badge>
              <span className="ml-1 text-xs text-muted">{r.matched_by ? `matched by ${r.matched_by}` : r.why}</span>
              {!!r.candidates?.length && r.kind !== "corpus" && (
                <ul className="ml-4 mt-1 space-y-0.5 text-xs">
                  {r.candidates.map((c, j) => (
                    <li key={j} className="flex flex-wrap items-center gap-2 text-muted">
                      <span className="text-ink">{c.title}</span>{c.year ? ` (${c.year})` : ""} · {c.provider}{c.title_match !== undefined ? ` · title match ${c.title_match}` : ""}
                      {onUse && <button onClick={() => onUse(use(c))} className="text-accent hover:underline">use this paper</button>}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
      {rt.discovery && (
        <div className="mt-2 text-xs text-muted">
          Topic “{rt.discovery.topic}”: {rt.discovery.why}.{rt.discovery.provider_error ? ` ${rt.discovery.provider_error}.` : ""}
          {rt.discovery.candidates.length > 0 && (
            <ul className="ml-4 mt-1 space-y-0.5">
              {rt.discovery.candidates.slice(0, 10).map((c, j) => (
                <li key={j} className="flex flex-wrap items-center gap-2">
                  <span className="text-ink">{c.title}</span>{c.year ? ` (${c.year})` : ""} {c.in_corpus ? <Badge status="corpus">corpus</Badge> : <Badge status="runtime">outside corpus</Badge>}
                  {onUse && <button onClick={() => onUse(use(c))} className="text-accent hover:underline">compare this one</button>}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {rt.papers.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs">
          {rt.papers.map((p) => (
            <li key={p.key} className="text-muted">
              <Badge status={p.status ?? "completed"} /> <span className="text-ink">{p.title ?? p.key}</span> · {p.provider ?? "provider"} · {p.access}
              {p.license ? ` · licence ${p.license}` : " · licence unknown (quotes withheld)"} · {p.experiments} result record(s) kept
              {p.cached ? " · from your cache" : ""}{p.error ? ` · ${p.error}` : ""}
              {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">source</a>}
            </li>
          ))}
        </ul>
      )}
      <Notes notes={rt.notes} />
    </div>
  );
}

function CfgValues({ f }: { f: CfgField }) {
  if (!f.values.length) return <span className="text-faint" title={NOT_STATED_TITLE}>not stated</span>;
  return (
    <>
      {f.values.map((v, i) => (
        <span key={i} className="mr-2">
          <span className="text-ink">{shown(v.raw ?? v.value)}</span> <Badge status={f.status === "VARIES" ? "VARIES" : v.status} />
          {v.basis && <span className="block text-[0.7rem] text-faint">{v.basis}</span>}
          <EvidenceList evidence={v.evidence} />
        </span>
      ))}
    </>
  );
}

/** A paper's pre-training and fine-tuning configurations, each with its fields, evidence and how many experiments link to it. */
function ConfigList({ inv }: { inv: Inventory }) {
  const cfgs = [...inv.pretraining, ...inv.finetuning];
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted">{inv.summary}</p>
      {cfgs.map((c) => (
        <details key={c.id} className="rounded border border-line bg-bg px-2 py-1.5 text-xs">
          <summary className="cursor-pointer">
            <span className={cx("mr-1 rounded border px-1 text-[0.65rem] uppercase tracking-wide", STAGE_TONE[c.stage])}>{c.stage_label ?? c.stage}</span>
            <span className="text-ink">{c.label}</span>{c.linked_experiments !== undefined && <span className="text-faint"> · {c.linked_experiments} experiment(s)</span>}
          </summary>
          <ul className="mt-1 space-y-1">
            {c.fields.map((f) => <li key={f.field}><span className="text-muted">{f.label}:</span> <CfgValues f={f} /></li>)}
          </ul>
        </details>
      ))}
      {inv.not_linked.length > 0 && (
        <p className="text-xs text-faint">Not linked to the compared experiments: {inv.not_linked.map((u) => `${u.label} (${u.why})`).join("; ")}.</p>
      )}
    </div>
  );
}

/** One paper: its configurations and, per evaluated experiment, split, protocol, metric definition, dataset version and seed. */
function InventoryView({ data }: { data: InventoryResult }) {
  const cols = ["evaluation.dataset_split", "evaluation.evaluation_protocol", "evaluation.metric_definition", "dataset.dataset_version"];
  return (
    <div className="mt-4 space-y-3 text-sm">
      <div>
        <h2 className="text-lg font-semibold text-ink">{data.paper.title ?? data.paper.id} {data.paper.origin === "runtime" && <Badge status="runtime">runtime</Badge>}</h2>
        <p className="text-xs text-muted">{data.paper.year ?? ""} {data.paper.url && <a href={data.paper.url} target="_blank" rel="noreferrer" className="text-accent hover:underline">paper</a>}</p>
      </div>
      <Alert tone="info">
        {data.configurations.summary}{" "}
        {data.configurations.multiple.pretraining || data.configurations.multiple.finetuning
          ? `This paper reports several ${[data.configurations.multiple.pretraining && "pre-training", data.configurations.multiple.finetuning && "fine-tuning"].filter(Boolean).join(" and ")} configurations; they are kept apart below.`
          : "No paper-level evidence of several pre-training or fine-tuning configurations was found."}
      </Alert>
      <ConfigList inv={data.configurations} />
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[760px] text-left text-xs">
          <thead className="bg-surface-2 text-muted">
            <tr>{["Experiment", "Split", "Protocol", "Metric definition", "Dataset version", "Seed", "Lineage"].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}</tr>
          </thead>
          <tbody>
            {data.experiments.map((x) => (
              <tr key={x.experiment} className="border-t border-line align-top">
                <td className="px-2 py-1.5 text-ink">{x.label}</td>
                {cols.map((c) => <td key={c} className="px-2 py-1.5"><CfgValues f={x.conditions[c]} /></td>)}
                <td className="px-2 py-1.5"><CfgValues f={x.seed} /></td>
                <td className="px-2 py-1.5 text-muted">{x.lineage.map((l, i) => <span key={i} className="block">{l.relation}{l.label ? ` ${l.label}` : ""}{l.basis ? ` (${l.basis})` : ""}</span>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.more_experiments > 0 && <p className="text-xs text-faint">+{data.more_experiments} more experiment record(s).</p>}
      {data.unknown.length > 0 && <p className="text-xs text-muted">Not stated in the sources read: {data.unknown.map((u) => `${u.label} (${u.experiments} of ${u.of})`).join(", ")}.</p>}
    </div>
  );
}

function PapersTab({ params, go }: TabProps) {
  const ids = useMemo(() => (params.get("papers") ?? "").split(",").filter(Boolean), [params]);
  const refs = useMemo(() => params.getAll("ref"), [params]);
  const topic = params.get("topic");
  const external = params.get("external") === "1";
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [refText, setRefText] = useState("");
  const [topicText, setTopicText] = useState("");
  const { data, setData, busy, error, run } = useRun<PapersResult>();
  const [onlyDiff, setOnlyDiff] = useState(false);
  const [showEmpty, setShowEmpty] = useState(false);
  const total = ids.length + refs.length;
  const compare = useCallback(() => {
    const q = new URLSearchParams();
    if (ids.length) q.set("papers", ids.join(","));
    refs.forEach((r) => q.append("ref", r));
    if (topic) q.set("topic", topic);
    if (external) q.set("external", "true");
    void run(`/api/analysis/paper-diff?${q.toString()}`);
  }, [ids, refs, topic, external, run]);
  useEffect(() => {  // corpus papers only: deterministic and quick, so the comparison runs as soon as two are picked
    if (ids.length >= 2 && !refs.length && !topic) compare();
    else if (!total && !topic) setData(null);
  }, [ids.join(","), refs.join("\n"), topic]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (next: string[]) => go({ papers: next.length ? next.join(",") : null });
  const setRefs = (next: string[]) => {  // repeated ?ref= parameters (titles, DOIs, arXiv ids, URLs)
    const p = new URLSearchParams(window.location.search);
    p.delete("ref");
    next.slice(0, 4).forEach((r) => p.append("ref", r));
    window.history.pushState(null, "", `/analyze?${p.toString()}`);
  };
  const addRef = (r: string) => { const t = r.trim(); if (t && !refs.includes(t) && total < 4) setRefs([...refs, t]); };
  const diff = data?.kind === "paper_diff" ? data : null;
  return (
    <section>
      <div className="flex flex-wrap items-center gap-2">
        {ids.map((p) => (
          <span key={p} className="inline-flex items-center gap-1 rounded-full border border-line bg-surface px-2.5 py-1 text-sm">
            {labels[p] ?? diff?.papers.find((x) => x.id === p)?.title ?? p}
            <button aria-label="Remove paper" onClick={() => set(ids.filter((x) => x !== p))} className="text-faint hover:text-bad"><I.Cross width={13} height={13} /></button>
          </span>
        ))}
        {refs.map((r) => (
          <span key={r} className="inline-flex items-center gap-1 rounded-full border border-dashed border-line bg-surface px-2.5 py-1 text-sm" title="Typed reference: resolved when you compare">
            {r.length > 60 ? r.slice(0, 59) + "…" : r}
            <button aria-label="Remove reference" onClick={() => setRefs(refs.filter((x) => x !== r))} className="text-faint hover:text-bad"><I.Cross width={13} height={13} /></button>
          </span>
        ))}
        {topic && (
          <span className="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-surface px-2.5 py-1 text-sm">
            topic: {topic}<button aria-label="Remove topic" onClick={() => go({ topic: null })} className="text-faint hover:text-bad"><I.Cross width={13} height={13} /></button>
          </span>
        )}
      </div>
      <div className="mt-2 grid gap-2 md:grid-cols-2">
        {total < 4 && (
          <Picker types="Paper" placeholder={total < 2 ? "Search a corpus paper (pick two to four)" : "Add another corpus paper"} exclude={ids}
            onPick={(h) => (setLabels((l) => ({ ...l, [h.id]: h.label })), set([...ids, h.id]))} />
        )}
        {total < 4 && (
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); addRef(refText); setRefText(""); }}>
            <input value={refText} onChange={(e) => setRefText(e.target.value)} maxLength={300} placeholder="…or a title, DOI, arXiv id or URL"
              className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
            <Button type="submit" variant="secondary" disabled={!refText.trim()}>Add</Button>
          </form>
        )}
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (topicText.trim().length >= 3) go({ topic: topicText.trim() }); setTopicText(""); }}>
          <input value={topicText} onChange={(e) => setTopicText(e.target.value)} maxLength={200} placeholder="…or find recent papers on a method or topic"
            className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
          <Button type="submit" variant="secondary" disabled={topicText.trim().length < 3}>Find</Button>
        </form>
        <div className="flex flex-wrap items-center gap-3">
          <ExternalToggle checked={external} onChange={(v) => go({ external: v ? "1" : null })}
            note="Looks up references outside the corpus with scholarly search (Semantic Scholar, OpenAlex, arXiv) and reads their arXiv full text now. Only what you type is sent; private documents never are." />
          <Button onClick={compare} disabled={busy || (!total && !topic)}>{total === 1 && !topic ? "Show configurations" : topic ? "Find and compare" : "Compare"}</Button>
        </div>
      </div>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && (
        <div className="flex flex-col items-center gap-2 py-8 text-faint"><Spinner />
          {(external || refs.length > 0 || topic) && <p className="text-xs text-muted">Resolving references and reading papers outside the corpus can take up to a minute.</p>}
        </div>
      )}
      {data?.kind === "paper_resolution" && !busy && (
        <div className="mt-4"><Alert tone="warn">No paper could be resolved. Pick a candidate, give a DOI or arXiv id, or turn on papers outside the corpus.</Alert>
          <RuntimePanel rt={data.runtime} onUse={addRef} /></div>
      )}
      {data?.kind === "paper_configurations" && !busy && (
        <>
          <InventoryView data={data} />
          {data.runtime && <RuntimePanel rt={data.runtime} onUse={addRef} />}
          <Notes notes={data.notes} />
          <Stages r={data} />
        </>
      )}
      {diff && !busy && (
        <div className="mt-4">
          <DiffOverview data={diff} />
          <div className="mt-4 space-y-2">
            {diff.fairness.map((f) => (
              <div key={f.papers.join()} className="rounded-lg border border-line bg-surface p-3 text-sm">
                <p className="flex flex-wrap items-center gap-2"><Badge status={f.verdict} /> <span className="text-ink">{f.why}</span></p>
                <p className="mt-1 text-xs text-muted">
                  Shared benchmarks: {f.shared_benchmarks.length ? f.shared_benchmarks.join(", ") : "none"}
                  {f.missing?.length ? ` · not stated in either paper: ${f.missing.join(", ")}` : ""} · derived by the comparison engine from the evaluation setups, not the scores
                </p>
                {!!f.stage_differences?.length && (
                  <p className="mt-1 text-xs text-ink">Same-stage training differences of that pair: {f.stage_differences.map((d) => `${d.label} (${d.stage_label}): ${shown(d.value_a)} vs ${shown(d.value_b)}`).join("; ")}.</p>
                )}
                {!!f.stage_not_compared?.length && (
                  <p className="mt-1 text-xs text-faint">Not compared, stated for different stages: {f.stage_not_compared.map((d) => d.label).join(", ")}.</p>
                )}
              </div>
            ))}
          </div>
          <label className="mt-3 inline-flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} /> Only rows that differ or are partly unknown
          </label>
          <label className="ml-4 mt-3 inline-flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={showEmpty} onChange={(e) => setShowEmpty(e.target.checked)} /> Show the {diff.summary.unknown ?? 0} fields neither paper states
          </label>
          <div className="mt-2 overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="bg-surface-2 text-xs text-muted">
                <tr>
                  <th className="w-44 px-2 py-2 font-medium">Field</th>
                  {diff.papers.map((p) => (
                    <th key={p.id} className="px-2 py-2 font-medium">
                      <span className="text-ink">{p.title}</span> {p.year ? `(${p.year})` : ""} {p.origin === "runtime" && <Badge status="runtime">runtime</Badge>}
                      <span className="block font-normal text-faint">{p.experiments} experiment record(s)</span>
                    </th>
                  ))}
                </tr>
              </thead>
              {diff.sections.map((s) => {
                const rows = s.rows.filter((r) => (showEmpty || r.status !== "UNKNOWN") && (!onlyDiff || r.status !== "SAME"));
                return rows.length ? (
                  <tbody key={s.name}>
                    <tr><td colSpan={diff.papers.length + 1} className="bg-bg px-2 pt-3 text-xs font-semibold uppercase tracking-wide text-faint">{s.name}</td></tr>
                    {rows.map((r) => (
                      <tr key={`${r.field}:${r.stage ?? ""}`} className="border-t border-line align-top">
                        <td className="px-2 py-1.5"><span className="text-ink">{r.label}</span> <Badge status={r.status} /></td>
                        {r.cells.map((c, i) => (
                          <td key={i} className="px-2 py-1.5">
                            {c.values.length === 0 ? (
                              <span className="text-faint" title={NOT_STATED_TITLE}>{c.status === "CONFLICTING" ? "sources disagree" : "not stated in the paper"}</span>
                            ) : (
                              c.values.map((v, j) => (
                                <div key={j} className="mb-1">
                                  <span className="text-ink">{shown(v.value, v.unit)}</span> <Badge status={v.status} />
                                  {v.scope === "paper" && <span className="ml-1 rounded border border-dashed border-line px-1 text-[0.65rem] text-muted" title="Stated in the paper's text but not assigned to an experiment record">paper-level{v.applies_to ? ` · ${v.applies_to}` : ""}</span>}
                                  {c.values.length > 1 && v.scope !== "paper" && <span className="ml-1 text-xs text-faint">{v.experiment_count} exp.</span>}
                                  {v.basis && r.stage && <span className="block text-[0.7rem] text-faint" title="Why this value belongs to this stage">{v.basis}</span>}
                                  <EvidenceList evidence={v.evidence} />
                                </div>
                              ))
                            )}
                            {c.more_values > 0 && <span className="text-xs text-faint">+{c.more_values} more value(s)</span>}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                ) : null;
              })}
            </table>
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {diff.papers.map((p) => (
              <div key={p.id} className="rounded-lg border border-line bg-surface p-3 text-sm">
                <p className="font-medium text-ink">{p.title}</p>
                {p.configurations && (p.configurations.pretraining.length + p.configurations.finetuning.length > 0) && (
                  <div className="mt-2"><p className="text-[0.68rem] uppercase tracking-wide text-faint">Training configurations</p><ConfigList inv={p.configurations} /></div>
                )}
                {p.passages.length ? p.passages.map((x, i) => (
                  <blockquote key={i} className="mt-2 border-l-2 border-line pl-2 text-xs text-muted">
                    <span className="font-semibold text-faint">{x.section}</span>{" "}
                    {x.text_withheld ? <em>Text withheld by the source&apos;s licence.</em> : `“${x.text}”`}
                    {x.url && <a href={x.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">source</a>}
                  </blockquote>
                )) : <p className="mt-2 text-xs text-faint">No limitations or conclusion section in the text read.</p>}
                {p.results.length > 0 && (
                  <ul className="mt-2 space-y-0.5 text-xs text-muted">
                    {p.results.slice(0, 6).map((r, i) => <li key={i}>{r.label}: {r.metric ?? "metric unknown"} = <span className="font-mono">{r.score ?? "?"}</span>{r.split ? ` (${r.split})` : ""}</li>)}
                  </ul>
                )}
              </div>
            ))}
          </div>
          {diff.runtime && <RuntimePanel rt={diff.runtime} onUse={addRef} />}
          <Notes notes={diff.notes} />
          <Stages r={diff} />
        </div>
      )}
    </section>
  );
}

function Chips({ items, tone = "bg-surface-2 text-ink" }: { items?: Named[]; tone?: string }) {
  if (!items?.length) return <span className="text-xs text-faint">none recorded</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {items.map((i) => <span key={i.id} className={cx("rounded px-1.5 py-0.5 text-xs", tone)}>{i.name}{i.count ? <span className="ml-1 text-faint">{i.count}</span> : null}</span>)}
    </span>
  );
}

/** What each paper covers, what they share, citations between them and their results on shared benchmarks. */
function DiffOverview({ data }: { data: DiffResult }) {
  const title = (id: string) => data.papers.find((p) => p.id === id)?.title ?? id;
  const kinds = [["benchmarks", "Benchmarks"], ["models", "Models"], ["tasks", "Tasks"], ["metrics", "Metrics"]] as const;
  return (
    <div className="space-y-3">
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${data.papers.length}, minmax(0, 1fr))` }}>
        {data.papers.map((p) => (
          <div key={p.id} className="rounded-lg border border-line bg-surface p-3 text-sm">
            <p className="font-medium text-ink">{p.title} {p.origin === "runtime" ? <Badge status="runtime">runtime</Badge> : <Badge status="corpus">corpus</Badge>}</p>
            <p className="text-xs text-muted">{[p.year, p.venue, p.authors?.slice(0, 3).join(", ")].filter(Boolean).join(" · ")} · {p.experiments} experiment record(s)
              {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">paper</a>}</p>
            {p.provenance && (
              <p className="mt-1 text-xs text-muted" title="Read at request time; not in the ResearchX corpus; unreviewed">
                {p.provenance.provider ?? "provider"} · {p.provenance.access}{p.provenance.license ? ` · licence ${p.provenance.license}` : " · licence unknown"}
                {p.provenance.retrieved_at ? ` · read ${p.provenance.retrieved_at.slice(0, 16).replace("T", " ")}` : ""}
              </p>
            )}
            {kinds.map(([k, label]) => (
              <div key={k} className="mt-1.5"><p className="text-[0.68rem] uppercase tracking-wide text-faint">{label}</p><Chips items={p.covers?.[k]} /></div>
            ))}
            {p.abstract && <details className="mt-2 text-xs text-muted"><summary className="cursor-pointer text-accent">Abstract (as published)</summary><p className="mt-1">{p.abstract}</p></details>}
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-line bg-surface p-3 text-sm">
        <h3 className="font-semibold text-ink">In common</h3>
        <div className="mt-1 grid gap-2 md:grid-cols-4">
          {kinds.map(([k, label]) => <div key={k}><p className="text-[0.68rem] uppercase tracking-wide text-faint">{label}</p><Chips items={data.shared?.[k]} tone="bg-ok-soft text-ok" /></div>)}
        </div>
        <p className="mt-2 text-xs text-muted">
          {data.citations?.length ? data.citations.map((c) => `“${title(c.citing)}” cites “${title(c.cited)}”`).join(" · ") : "Neither paper cites the other in the corpus citation graph."}
        </p>
      </div>
      {!!data.shared_results?.length && (
        <div className="overflow-x-auto rounded-lg border border-line">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="bg-surface-2 text-xs text-muted">
              <tr><th className="px-2 py-1.5 font-medium">Shared benchmark · metric</th>{data.papers.map((p) => <th key={p.id} className="px-2 py-1.5 font-medium">{p.title}</th>)}</tr>
            </thead>
            <tbody>
              {data.shared_results.map((r) => (
                <tr key={r.benchmark + r.metric} className="border-t border-line align-top">
                  <td className="px-2 py-1.5 text-ink">{r.benchmark} · {r.metric}</td>
                  {r.cells.map((c, i) => (
                    <td key={i} className="px-2 py-1.5 text-xs">
                      {c.map((x, j) => <div key={j}><span className="text-muted">{x.model}</span> <span className="font-mono text-ink">{x.score}</span>{x.split ? <span className="text-faint"> ({x.split})</span> : null}</div>)}
                      {r.more[i] > 0 && <span className="text-faint">+{r.more[i]} more</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-2 py-1.5 text-xs text-faint">Values as each paper reports them; whether they are comparable is the fairness verdict below, not the numbers.</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- Result Discrepancy Investigator
type Cands = { pairs: { a: string; b: string; score_a: string; score_b: string; relative: number; model?: string; benchmark?: string; metric?: string; same_paper: boolean }[];
  recorded: { experiment: string; label?: string; value_a: unknown; value_b: unknown; explanation?: string }[]; total_pairs: number;
  experiment?: string | null; experiment_label?: string | null;
  related?: { a: string; b: string; score_a: string; score_b: string; relative: number | null; label?: string; metric?: string; same_paper: boolean }[] };
type Row = { field: string; stage?: string; label: string; status: string; value_a: unknown; value_b: unknown; evidence_a: Evidence[]; evidence_b: Evidence[];
  parts?: { only_a: string[]; only_b: string[]; variants: [string, string][] } };
type Disc = Base & { experiments: NodeRef[]; papers: NodeRef[]; verdict: string; verdict_why: string; label: string; label_rationale: string; findings?: string[];
  observed: { metric: string; a: { score: string; split?: string | null; evidence: Evidence[] }; b: { score: string; split?: string | null; evidence: Evidence[] }; difference: number | null; relative: number | null }[];
  differences: Row[]; same: Row[];
  unknown: { field: string; label: string; missing_on: string[]; both?: boolean; hint?: string; stage?: string | null; stage_mismatch?: boolean; rationale?: string;
    paper_level?: Record<string, { value: unknown; unit?: string | null; applies_to?: string | null; evidence?: Evidence[] }[]> }[]; hypotheses: { factor: string; statement: string; status: string }[];
  checks: { test: string; establishes: string }[]; contradictions: { field: string; explanation?: string; value_a: unknown; value_b: unknown }[]; confidence: { level: string; reasons: string[] } };

function DiscrepancyTab({ params, go }: TabProps) {
  const a = params.get("a");
  const b = params.get("b");
  const cands = useRun<Cands & { kind: string }>();
  const inv = useRun<Disc>();
  useEffect(() => {
    if (a && b) void inv.run(`/api/analysis/discrepancy?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
    else {
      const one = a ?? b;  // one experiment chosen (e.g. from a source card): the pairs it is part of
      const path = one ? `/api/analysis/discrepancy?a=${encodeURIComponent(one)}` : "/api/analysis/discrepancy";
      if ((cands.data?.experiment ?? null) !== (one ?? null) || !cands.data) void cands.run(path);
    }
  }, [a, b]); // eslint-disable-line react-hooks/exhaustive-deps
  const pick = (other: string) => go(a ? { b: other } : { a: other });
  const d = inv.data;
  return (
    <section>
      <div className="grid gap-2 md:grid-cols-2">
        <div><p className="mb-1 text-xs text-muted">Experiment A {a && <span className="font-mono">{a}</span>}</p><Picker types="Experiment" placeholder="Search an experiment" onPick={(h) => go({ a: h.id })} /></div>
        <div><p className="mb-1 text-xs text-muted">Experiment B {b && <span className="font-mono">{b}</span>}</p><Picker types="Experiment" placeholder="Search an experiment" onPick={(h) => go({ b: h.id })} /></div>
      </div>
      {(a || b) && <button onClick={() => go({ a: null, b: null })} className="mt-2 text-sm text-accent hover:underline">Clear and show candidates</button>}
      {(inv.error || cands.error) && <div className="mt-3"><Alert tone="bad">{inv.error ?? cands.error}</Alert></div>}
      {(inv.busy || cands.busy) && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {!(a && b) && cands.data?.experiment && !cands.busy && (
        <div className="mt-4">
          <Alert tone="info">Experiment A is <span className="font-medium">{cands.data.experiment_label ?? cands.data.experiment}</span>. Pick experiment B: one of the candidates below, or search above.</Alert>
          {cands.data.pairs.length > 0 && (
            <>
              <h2 className="mt-3 text-sm font-semibold text-ink">Same model, benchmark and metric as A, different reported score</h2>
              <ul className="mt-2 divide-y divide-line rounded-lg border border-line bg-surface text-sm">
                {cands.data.pairs.slice(0, 15).map((p) => (
                  <li key={p.b} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                    <span className="text-ink">{p.model} · {p.benchmark} · {p.metric}: A <span className="font-mono">{p.score_a}</span> vs <span className="font-mono">{p.score_b}</span>
                      <span className="ml-1 text-xs text-faint">({Math.round(p.relative * 100)}% apart{p.same_paper ? ", same paper" : ""})</span></span>
                    <Button variant="secondary" onClick={() => pick(p.b)}>Investigate</Button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {!!cands.data.related?.length && (
            <>
              <h2 className="mt-3 text-sm font-semibold text-ink">Other experiments on A&apos;s benchmark and metric, different reported score</h2>
              <p className="text-xs text-muted">Any model, so the setups may differ by design; the investigation shows which fields differ.</p>
              <ul className="mt-2 divide-y divide-line rounded-lg border border-line bg-surface text-sm">
                {cands.data.related.slice(0, 15).map((p) => (
                  <li key={p.b} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                    <span className="text-ink">{p.label ?? p.b} · {p.metric}: A <span className="font-mono">{p.score_a}</span> vs <span className="font-mono">{p.score_b}</span>
                      {p.same_paper && <span className="ml-1 text-xs text-faint">(same paper)</span>}</span>
                    <Button variant="secondary" onClick={() => pick(p.b)}>Investigate</Button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {!cands.data.pairs.length && !cands.data.related?.length && (
            <p className="mt-3 text-sm text-muted">No corpus experiment reports A&apos;s benchmark and metric with a different score. Search for experiment B above.</p>
          )}
        </div>
      )}
      {!(a || b) && cands.data && !cands.data.experiment && !cands.busy && (
        <div className="mt-4">
          <h2 className="text-sm font-semibold text-ink">Same model, same benchmark and metric, different reported scores</h2>
          <p className="text-xs text-muted">{cands.data.total_pairs} pairs found by exact canonical ids (no similarity); the largest relative gaps across papers first.</p>
          <ul className="mt-2 divide-y divide-line rounded-lg border border-line bg-surface text-sm">
            {cands.data.pairs.slice(0, 15).map((p) => (
              <li key={p.a + p.b} className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                <span className="text-ink">{p.model} · {p.benchmark} · {p.metric}: <span className="font-mono">{p.score_a}</span> vs <span className="font-mono">{p.score_b}</span>
                  <span className="ml-1 text-xs text-faint">({Math.round(p.relative * 100)}% apart{p.same_paper ? ", same paper" : ""})</span></span>
                <Button variant="secondary" onClick={() => go({ a: p.a, b: p.b })}>Investigate</Button>
              </li>
            ))}
          </ul>
          {cands.data.recorded.length > 0 && (
            <>
              <h2 className="mt-4 text-sm font-semibold text-ink">Score contradictions recorded inside one record</h2>
              <ul className="mt-2 space-y-1 text-sm">
                {cands.data.recorded.slice(0, 8).map((c, i) => <li key={i} className="text-muted">{c.label}: {String(c.value_a)} vs {String(c.value_b)} — {c.explanation}</li>)}
              </ul>
            </>
          )}
        </div>
      )}
      {a && b && d && !inv.busy && (
        <div className="mt-4 space-y-4 text-sm">
          <div className="grid gap-2 md:grid-cols-2">
            {d.experiments.map((x, i) => (
              <div key={x.id} className="rounded-lg border border-line bg-surface p-3">
                <p className="text-xs text-faint">Experiment {i ? "B" : "A"}</p>
                <p className="font-medium text-ink">{x.label}</p>
                <p className="text-xs text-muted">{d.papers[i]?.label}</p>
              </div>
            ))}
          </div>
          <div>
            <h2 className="font-semibold text-ink">Observed discrepancy</h2>
            {d.observed.length ? d.observed.map((o) => (
              <p key={o.metric} className="mt-1">{o.metric}: A <span className="font-mono">{o.a.score}</span>{o.a.split ? ` (${o.a.split})` : ""} vs B <span className="font-mono">{o.b.score}</span>{o.b.split ? ` (${o.b.split})` : ""}
                {o.difference !== null && <span className="text-muted"> · difference {o.difference > 0 ? "+" : ""}{Number(o.difference.toFixed(4))}</span>}</p>
            )) : <p className="text-muted">The two experiments report no metric in common.</p>}
          </div>
          {!!d.findings?.length && (
            <div className="rounded-lg border border-line bg-surface p-3">
              <h2 className="font-semibold text-ink">What the records show</h2>
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-ink">{d.findings.map((f) => <li key={f}>{f}</li>)}</ul>
            </div>
          )}
          <Alert tone={d.verdict === "DIFFERENCES_FOUND" ? "info" : "warn"}><Badge status={d.verdict} /> {d.verdict_why}</Alert>
          <Block title={`Verified differences (${d.differences.length})`} empty="No reported field differs.">
            {d.differences.map((r) => <FieldRow key={`${r.field}:${r.stage ?? ""}`} r={r} />)}
          </Block>
          <Block title={`Held equal (${d.same.length})`} empty="No compared field is stated equal on both sides.">
            {d.same.map((r) => <FieldRow key={`${r.field}:${r.stage ?? ""}`} r={r} />)}
          </Block>
          <Block title={`Hypotheses (${d.hypotheses.length})`} empty="No hypothesis: nothing reported differs.">
            {d.hypotheses.map((h) => <li key={h.factor} className="py-1.5"><Badge status="HYPOTHESIS" /> <span className="text-ink">{h.statement}</span></li>)}
          </Block>
          <Block title={`Missing information (${d.unknown.length})`} empty="Every compared field is reported on both sides.">
            {d.unknown.map((u) => (
              <li key={`${u.field}:${u.stage ?? ""}`} className="py-1 text-muted">
                {u.label}: {u.stage_mismatch ? `not compared: ${u.rationale}` : u.both ? "not stated for either experiment" : `not stated for ${u.missing_on.join(" and ")}`}
                <span className="ml-1 text-xs text-faint" title={NOT_STATED_TITLE}>(sources read)</span>
                {u.hint && <span className="block text-xs text-faint">Hint: {u.hint}.</span>}
                {Object.entries(u.paper_level ?? {}).map(([side, vs]) => (
                  <div key={side} className="ml-3 text-xs">
                    <span className="rounded border border-dashed border-line px-1 text-[0.65rem]">paper-level</span> paper {side} states:{" "}
                    {vs.map((v, i) => <span key={i} className="text-ink">{shown(v.value, v.unit)}{v.applies_to ? ` (for ${v.applies_to})` : ""}{i < vs.length - 1 ? "; " : ""}</span>)}
                    <span className="text-faint"> · not assigned to this experiment</span>
                    <EvidenceList evidence={vs.flatMap((v) => v.evidence ?? [])} />
                  </div>
                ))}
              </li>
            ))}
          </Block>
          <Block title="Suggested checks" empty="No check suggested.">
            {d.checks.map((c, i) => <li key={i} className="py-1"><span className="text-ink">{c.test}</span> <span className="text-xs text-muted">Tests {c.establishes}.</span></li>)}
          </Block>
          {d.contradictions.length > 0 && (
            <Block title="Recorded contradictions" empty="">
              {d.contradictions.map((c, i) => <li key={i} className="py-1 text-muted">{c.field}: {String(c.value_a)} vs {String(c.value_b)} — {c.explanation}</li>)}
            </Block>
          )}
          <p className="text-xs text-muted"><Badge status={d.confidence.level} /> confidence limited by: {d.confidence.reasons.join("; ") || "—"}. Comparison label: {words(d.label)} ({d.label_rationale}).</p>
          <Notes notes={d.notes} />
          <Stages r={d} />
        </div>
      )}
    </section>
  );
}

function Block({ title, empty, children }: { title: string; empty: string; children: ReactNode }) {
  const has = Array.isArray(children) ? children.length > 0 : !!children;
  return (
    <div>
      <h2 className="font-semibold text-ink">{title}</h2>
      {has ? <ul className="mt-1 divide-y divide-line">{children}</ul> : <p className="mt-1 text-muted">{empty}</p>}
    </div>
  );
}

function FieldRow({ r }: { r: Row }) {
  const p = r.parts;
  const delta = p ? [p.only_a.length ? `only A: ${p.only_a.join(", ")}` : "", p.only_b.length ? `only B: ${p.only_b.join(", ")}` : "",
    ...p.variants.map(([x, y]) => `${x} (A) vs ${y} (B)`)].filter(Boolean) : [];
  return (
    <li className="grid gap-2 py-1.5 md:grid-cols-[12rem_1fr_1fr]">
      <span className="text-ink">{r.label} <Badge status={r.status === "MATERIALLY_DIFFERENT" ? "DIFFERENT" : r.status} /></span>
      <span>A: {shown(r.value_a)}<EvidenceList evidence={r.evidence_a} /></span>
      <span>B: {shown(r.value_b)}<EvidenceList evidence={r.evidence_b} /></span>
      {delta.length > 0 && <span className="text-xs text-muted md:col-span-3">Name components: {delta.join(" · ")}</span>}
    </li>
  );
}

// ---------------------------------------------------------------------------------------------- Reproducibility Recipe Builder
type Recipe = Base & { experiment: NodeRef; paper: { id: string; title?: string | null; url?: string | null; year?: number | null };
  sections: { name: string; facts: Fact[] }[]; results: { metric: Fact; score: Fact; split: Fact }[]; summary: Record<string, number>; missing: string[];
  conflicting: string[]; provenance: { corpus_release: string; quality_tier: string; reviewer_status: string; extraction_method: string }; export: unknown; markdown: string;
  configurations?: Cfg[]; missing_by_stage?: Record<string, string[]> };
type Options = { kind: "recipe_options"; paper: NodeRef; experiments: (NodeRef & { stated_fields: number; results: number })[] };

function RecipeTab({ params, go }: TabProps) {
  const experiment = params.get("experiment");
  const paper = params.get("paper");
  const rec = useRun<Recipe>();
  const opts = useRun<Options>();
  const [all, setAll] = useState(false);
  useEffect(() => {
    if (experiment) void rec.run(`/api/analysis/recipe?experiment=${encodeURIComponent(experiment)}`);
    else if (paper) void opts.run(`/api/analysis/recipe?paper=${encodeURIComponent(paper)}`);
  }, [experiment, paper]); // eslint-disable-line react-hooks/exhaustive-deps
  const r = rec.data;
  return (
    <section>
      <div className="grid gap-2 md:grid-cols-2">
        <Picker types="Experiment" placeholder="Search an experiment" onPick={(h) => go({ experiment: h.id, paper: null })} />
        <Picker types="Paper" placeholder="…or a paper, then choose one of its experiments" onPick={(h) => go({ paper: h.id, experiment: null })} />
      </div>
      {(rec.error || opts.error) && <div className="mt-3"><Alert tone="bad">{rec.error ?? opts.error}</Alert></div>}
      {(rec.busy || opts.busy) && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {!experiment && paper && opts.data && !opts.busy && (
        <ul className="mt-4 divide-y divide-line rounded-lg border border-line bg-surface text-sm">
          {opts.data.experiments.map((x) => (
            <li key={x.id} className="flex items-center justify-between gap-2 px-3 py-2">
              <span className="text-ink">{x.label} <span className="text-xs text-faint">{x.stated_fields} fields stated · {x.results} result(s)</span></span>
              <Button variant="secondary" onClick={() => go({ experiment: x.id })}>Recipe</Button>
            </li>
          ))}
        </ul>
      )}
      {experiment && r && !rec.busy && (
        <div className="mt-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-ink">{r.experiment.label}</h2>
              <p className="text-sm text-muted">{r.paper.title}{r.paper.year ? ` (${r.paper.year})` : ""}
                {r.paper.url && <a href={r.paper.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">paper</a>}</p>
              <p className="text-xs text-faint">Corpus {r.provenance.corpus_release} · {r.provenance.quality_tier.toLowerCase()} record · {words(r.provenance.reviewer_status)} · {words(r.provenance.extraction_method)} extraction</p>
            </div>
            <div className="flex gap-2">
              <Button variant="secondary" onClick={() => download("recipe.md", r.markdown, "text/markdown")}><I.Download width={15} height={15} /> Checklist</Button>
              <Button variant="secondary" onClick={() => download("recipe.json", JSON.stringify(r.export, null, 2), "application/json")}><I.Download width={15} height={15} /> JSON</Button>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-sm">
            {(["REPORTED", "DERIVED", "CONFLICTING", "UNKNOWN"] as const).map((k) => <span key={k} className="inline-flex items-center gap-1.5"><Badge status={k} /> {r.summary[k] ?? 0}</span>)}
          </div>
          {r.missing.length > 0 && <div className="mt-3"><Alert tone="warn">Still missing before a reproduction can start: {r.missing.join(", ")}. {NOT_STATED_TITLE} Nothing is guessed: ask the authors or check the code repository.</Alert></div>}
          <label className="mt-3 inline-flex items-center gap-2 text-sm text-muted"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> Show unreported fields too</label>
          <div className="mt-2 grid gap-3 md:grid-cols-2">
            {r.sections.map((s) => {
              const facts = s.facts.filter((f) => all || f.status !== "UNKNOWN");
              return (
                <div key={s.name} className="rounded-lg border border-line bg-surface p-3">
                  <h3 className="text-sm font-semibold text-ink">{s.name}</h3>
                  {facts.length === 0 ? <p className="mt-1 text-xs text-faint">Nothing reported.</p> : (
                    <ul className="mt-1 space-y-1.5 text-sm">
                      {facts.map((f) => (
                        <li key={f.field}>
                          <span className={cx("mr-1", f.status === "UNKNOWN" ? "text-faint" : "text-muted")}>{f.status === "UNKNOWN" ? "☐" : "☑"} {f.label}:</span>
                          <span className={f.status === "UNKNOWN" ? "text-faint" : "text-ink"} title={f.status === "UNKNOWN" ? NOT_STATED_TITLE : undefined}>
                            {f.status === "UNKNOWN" ? "not stated in the paper" : f.status === "CONFLICTING" ? "sources disagree" : shown(f.raw ?? f.value, f.unit)}</span> <Badge status={f.status} />
                          {f.stage && f.stage !== "evaluation" && <span className={cx("ml-1 rounded border px-1 text-[0.65rem]", STAGE_TONE[f.stage])}>{f.stage === "unspecified" ? "stage not stated" : f.stage}</span>}
                          {f.derivation && <span className="ml-1 text-xs text-faint">({f.derivation})</span>}
                          <EvidenceList evidence={f.evidence} />
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
          {!!r.configurations?.length && (
            <div className="mt-3 rounded-lg border border-line bg-surface p-3 text-sm">
              <h3 className="font-semibold text-ink">Pre-training and fine-tuning configurations in this experiment&apos;s lineage</h3>
              <div className="mt-1 space-y-2">
                {r.configurations.map((c) => (
                  <div key={c.id} className="rounded border border-line bg-bg px-2 py-1.5 text-xs">
                    <p><span className={cx("mr-1 rounded border px-1 text-[0.65rem] uppercase tracking-wide", STAGE_TONE[c.stage])}>{c.stage}</span>
                      <span className="text-ink">{c.label}</span>{c.link && <span className="text-faint"> · linked: {c.link}</span>}</p>
                    <ul className="mt-1 space-y-1">{c.fields.map((f) => <li key={f.field}><span className="text-muted">{f.label}:</span> <CfgValues f={f} /></li>)}</ul>
                    {!!r.missing_by_stage?.[c.stage === "pretraining" ? "pre-training" : "fine-tuning / adaptation"]?.length && (
                      <p className="mt-1 text-faint">Not stated for this stage: {r.missing_by_stage[c.stage === "pretraining" ? "pre-training" : "fine-tuning / adaptation"].join(", ")}.</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          {r.results.length > 0 && (
            <div className="mt-3 rounded-lg border border-line bg-surface p-3 text-sm">
              <h3 className="font-semibold text-ink">Reported results</h3>
              <ul className="mt-1 space-y-1">
                {r.results.map((x, i) => <li key={i}>{shown(x.metric.value)} = <span className="font-mono">{shown(x.score.raw ?? x.score.value)}</span>{x.split.value ? ` (${shown(x.split.value)})` : ""}<EvidenceList evidence={x.score.evidence} /></li>)}
              </ul>
            </div>
          )}
          <Notes notes={["Statuses: reported = stated by a source; derived = computed from a source (rule shown: the paper's own text statement, or a quote found in its full text at request time); conflicting = sources disagree; unknown = stated in none of the sources read.",
            "Silver-tier records are machine-extracted; open the evidence before relying on a value."]} />
          <Stages r={r} />
        </div>
      )}
    </section>
  );
}

/** Questions for the full research pipeline (corpus + live scholarly sources), prefilled in the chat, never sent automatically. */
function AskLinks({ questions }: { questions?: string[] }) {
  if (!questions?.length) return null;
  return (
    <div className="text-sm">
      <p className="text-xs text-muted">Gather more evidence with the full research pipeline (corpus and live scholarly sources):</p>
      <div className="mt-1 flex flex-wrap gap-2">
        {questions.map((q) => <Link key={q} href={`/?q=${encodeURIComponent(q)}`} className="rounded-lg border border-line px-2.5 py-1 text-accent hover:border-accent">{q}</Link>)}
      </div>
    </div>
  );
}

function PaperLine({ s }: { s: { title?: string | null; year?: number | null; paper?: string } }) {
  return <span className="text-xs text-muted">{s.title ?? s.paper}{s.year ? ` (${s.year})` : ""}</span>;
}

// ---------------------------------------------------------------------------------------------- charts (plain SVG/CSS, no dependency)
// Status colours carry stance and always come with a text label and count; one scale per chart; values in text ink.
const num = (s?: string | null) => {
  const m = s?.match(/-?\d+(?:\.\d+)?/);
  return m ? parseFloat(m[0]) : null;
};

function Swatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return <span className="inline-block h-2.5 w-2.5 rounded-sm" style={dashed ? { border: `1.5px dashed ${color}` } : { background: color }} />;
}

/** Share of supporting / contradicting / inconclusive evidence for a claim, as one stacked bar with a counted legend. */
function StanceBar({ s, c, i, direct }: { s: number; c: number; i: number; direct?: [number, number, number] }) {
  const total = s + c + i;
  if (!total) return null;
  const parts = [["Supporting", s, "var(--color-ok)", direct?.[0]], ["Contradicting", c, "var(--color-bad)", direct?.[1]],
    ["Inconclusive", i, "var(--color-faint)", direct?.[2]]] as const;
  return (
    <figure className="rounded-lg border border-line bg-surface p-3">
      <figcaption className="text-xs font-medium text-muted">Evidence found ({total} record{total === 1 ? "" : "s"} or result pairs)</figcaption>
      <div className="mt-2 flex h-3 w-full gap-[2px]" role="img" aria-label={parts.map(([l, n]) => `${l} ${n}`).join(", ")}>
        {parts.filter(([, n]) => n > 0).map(([l, n, col]) => (
          <div key={l} title={`${l}: ${n}`} className="h-full first:rounded-l last:rounded-r" style={{ width: `${(n / total) * 100}%`, background: col }} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {parts.map(([l, n, col, d]) => (
          <span key={l} className="inline-flex items-center gap-1.5"><Swatch color={col} /> {l} <span className="font-medium text-ink">{n}</span>
            {d !== undefined && n > 0 && <span className="text-faint">({d} direct)</span>}</span>
        ))}
      </div>
      {direct && <p className="mt-1 text-[0.7rem] text-faint">The verdict uses direct evidence when there is any: the claim&apos;s own model and dataset, not variants.</p>}
    </figure>
  );
}

/** Each compared result pair: the two reported scores on their own zero-based scale (same benchmark and metric only). */
function PairBars({ items, a, b }: { items: ClaimItem[]; a: string; b: string }) {
  const rows = items.filter((it) => it.b && num(it.a.score) !== null && num(it.b.score) !== null).slice(0, 6);
  if (!rows.length) return null;
  const A = "var(--color-accent)", Bc = "var(--color-muted)";
  return (
    <figure className="rounded-lg border border-line bg-surface p-3">
      <figcaption className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-medium text-muted">Reported scores of each assessed pair</span>
        <span className="flex flex-wrap gap-3 text-muted"><span className="inline-flex items-center gap-1.5"><Swatch color={A} /> {a}</span><span className="inline-flex items-center gap-1.5"><Swatch color={Bc} /> {b}</span></span>
      </figcaption>
      <ul className="mt-2 space-y-3">
        {rows.map((it, k) => {
          const va = num(it.a.score)!, vb = num(it.b!.score)!, max = Math.max(Math.abs(va), Math.abs(vb), 1e-9);
          return (
            <li key={k}>
              <p className="text-xs text-muted">{it.benchmark} · {it.metric}{it.same_paper ? " · same paper" : ""} <Badge status={it.stance} /></p>
              {[[a, va, it.a.score, A], [b, vb, it.b!.score, Bc]].map(([who, v, raw, col]) => (
                <div key={String(who)} className="mt-1 grid grid-cols-[minmax(0,7rem)_1fr_auto] items-center gap-2 text-xs" title={`${who}: ${raw}`}>
                  <span className="truncate text-ink">{who}</span>
                  <span className="h-2.5 rounded-r" style={{ width: `${Math.max(2, (Math.abs(Number(v)) / max) * 100)}%`, background: String(col) }} />
                  <span className="font-mono text-ink">{String(raw)}</span>
                </div>
              ))}
            </li>
          );
        })}
      </ul>
    </figure>
  );
}

/** One measure (papers) for two groups: filled = present, dashed outline = absent (a second encoding beside the colour). */
function CoverageBars({ title, rows, onLabel, offLabel }: { title: string; rows: { label: string; value: number; on: boolean; note?: string }[]; onLabel: string; offLabel: string }) {
  if (!rows.length) return null;
  const max = Math.max(1, ...rows.map((r) => r.value));
  const col = "var(--color-accent)";
  return (
    <figure className="rounded-lg border border-line bg-surface p-3">
      <figcaption className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-medium text-muted">{title}</span>
        <span className="flex flex-wrap gap-3 text-muted"><span className="inline-flex items-center gap-1.5"><Swatch color={col} /> {onLabel}</span><span className="inline-flex items-center gap-1.5"><Swatch color={col} dashed /> {offLabel}</span></span>
      </figcaption>
      <ul className="mt-2 space-y-1.5">
        {rows.map((r) => (
          <li key={r.label} className="grid grid-cols-[minmax(0,11rem)_1fr_auto] items-center gap-2 text-xs" title={`${r.label}: ${r.value} paper(s) · ${r.on ? onLabel : offLabel}${r.note ? ` · ${r.note}` : ""}`}>
            <span className="truncate text-ink">{r.label}</span>
            <span className="h-2.5 rounded-r" style={{ width: `${Math.max(3, (r.value / max) * 100)}%`, ...(r.on ? { background: col } : { border: `1.5px dashed ${col}` }) }} />
            <span className="tabular-nums text-muted">{r.value}{r.on ? "" : ` · ${offLabel.toLowerCase()}`}</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[0.7rem] text-faint">Bar length: number of corpus papers reporting results there.</p>
    </figure>
  );
}

/** Measured changes (same paper, same metric, shifted variant of the source dataset) around a zero line. */
function ChangeBars({ rows }: { rows: { label: string; from: string; to: string; change: number }[] }) {
  if (!rows.length) return null;
  const max = Math.max(...rows.map((r) => Math.abs(r.change)), 1e-9);
  return (
    <figure className="rounded-lg border border-line bg-surface p-3">
      <figcaption className="text-xs font-medium text-muted">Measured changes: same paper and metric, toward a shifted variant of the source dataset</figcaption>
      <ul className="mt-2 space-y-1.5">
        {rows.map((r, k) => (
          <li key={k} className="grid grid-cols-[minmax(0,13rem)_1fr_auto] items-center gap-2 text-xs" title={`${r.label}: ${r.from} → ${r.to}`}>
            <span className="truncate text-ink">{r.label}</span>
            <span className="relative h-2.5">
              <span className="absolute inset-y-[-3px] left-1/2 w-px bg-line" />
              <span className="absolute inset-y-0 rounded" style={{ background: "var(--color-accent)", width: `${(Math.abs(r.change) / max) * 50}%`, ...(r.change >= 0 ? { left: "50%" } : { right: "50%" }) }} />
            </span>
            <span className="font-mono text-ink">{r.change > 0 ? "+" : ""}{Number(r.change.toFixed(3))}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------- Claim-to-Evidence Auditor
type Side = { experiment?: string; label?: string; paper?: string; title?: string | null; year?: number | null; match?: string; score?: string | null;
  metric?: string | null; split?: string | null; evidence?: Evidence[] };
type ClaimItem = { stance: string; direct: boolean; reason: string; benchmark?: string | null; metric?: string | null; same_paper?: boolean; comparability?: string;
  stated?: string; scope?: string; evidence?: Evidence[]; a: Side; b?: Side };
type ClaimResult = Base & { claim: string; verdict: string; verdict_why: string; basis: string; flags: string[]; research_questions: string[];
  parsed: { form: string; a: string; b?: string; dataset?: string | null; metric?: string | null; value?: string; settings?: { label: string; value: string }[] };
  supporting: ClaimItem[]; contradicting: ClaimItem[]; inconclusive: ClaimItem[]; confidence: { level: string; reasons: string[] };
  summary: { supporting: number; contradicting: number; inconclusive: number }; coverage: { mention: string; records: number; results: number }[] };

function readAs(p: ClaimResult["parsed"]): string {
  const on = p.dataset ? ` on ${p.dataset}` : " (no dataset named)";
  if (p.form === "comparison") return `${p.a} better than ${p.b}${on}${p.metric ? `, ${p.metric}` : ""}`;
  if (p.form === "value") return `${p.a} reports ${p.value}${p.metric ? ` ${p.metric}` : ""}${on}`;
  return `${p.a} trained with ${(p.settings ?? []).map((s) => `${s.label} ${s.value}`).join(", ")}${p.dataset ? ` on ${p.dataset}` : ""}`;
}

function ClaimEvidence({ title, items, total }: { title: string; items: ClaimItem[]; total: number }) {
  return (
    <div>
      <h2 className="font-semibold text-ink">{title} ({total})</h2>
      {items.length === 0 ? <p className="mt-1 text-sm text-muted">None.</p> : (
        <ul className="mt-1 divide-y divide-line rounded-lg border border-line bg-surface text-sm">
          {items.map((it, i) => (
            <li key={i} className="px-3 py-2">
              <p className="flex flex-wrap items-center gap-1.5">
                <Badge status={it.stance} /> <span className={cx("rounded px-1.5 py-0.5 text-[0.68rem]", it.direct ? "bg-ok-soft text-ok" : "bg-surface-2 text-muted")}>{it.direct ? "direct" : "indirect"}</span>
                {it.same_paper && <span className="rounded bg-info-soft px-1.5 py-0.5 text-[0.68rem] text-accent">same paper</span>}
                {it.comparability && <Badge status={it.comparability} />}
                {it.benchmark && <span className="text-xs text-muted">{it.benchmark}{it.metric ? ` · ${it.metric}` : ""}</span>}
              </p>
              <p className="mt-1 text-ink">{it.reason}</p>
              <div className={cx("mt-1 grid gap-2", it.b && "md:grid-cols-2")}>
                {[it.a, it.b].filter((s): s is Side => !!s).map((s, j) => (
                  <div key={j} className="rounded border border-line bg-bg px-2 py-1.5 text-xs">
                    {s.label && <p className="text-ink">{s.label}{s.score ? <> · <span className="font-mono">{s.score}</span></> : null}{s.split ? ` (${s.split})` : ""}</p>}
                    <PaperLine s={s} />
                    <EvidenceList evidence={s.evidence} />
                  </div>
                ))}
              </div>
              <EvidenceList evidence={it.evidence} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ClaimTab({ params, go }: TabProps) {
  const claim = params.get("claim") ?? "";
  const [text, setText] = useState(claim);
  const { data, busy, error, run } = useRun<ClaimResult>();
  useEffect(() => {
    if (claim.trim().length >= 8) void run("/api/analysis/claim", { method: "POST", body: JSON.stringify({ claim }) });
  }, [claim, run]);
  return (
    <section>
      <form onSubmit={(e) => (e.preventDefault(), go({ claim: text.trim() || null }))} className="space-y-2">
        <label className="block text-sm text-muted" htmlFor="claim-text">Claim</label>
        <textarea id="claim-text" value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={600}
          placeholder="e.g. ModelA outperforms ModelB on DatasetX in accuracy"
          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-faint">Testable forms: {CLAIM_FORMS.join(" · ")}</p>
          <Button type="submit" disabled={text.trim().length < 8 || busy}>Check the claim</Button>
        </div>
      </form>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {data && !busy && claim && <div className="mt-4"><ClaimView data={data} /></div>}
    </section>
  );
}

export function ClaimView({ data }: { data: ClaimResult }) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">Read as: <span className="text-ink">{readAs(data.parsed)}</span></p>
      <Alert tone={data.verdict === "SUPPORTED" ? "info" : "warn"}><Badge status={data.verdict} /> {data.verdict_why}
        {data.basis !== "none" && <span className="ml-1 text-xs">(from {data.basis} evidence)</span>}</Alert>
      <StanceBar s={data.summary.supporting} c={data.summary.contradicting} i={data.summary.inconclusive}
        direct={[data.supporting, data.contradicting, data.inconclusive].map((xs) => xs.filter((x) => x.direct).length) as [number, number, number]} />
      {data.parsed.form === "comparison" && <PairBars items={[...data.supporting, ...data.contradicting]} a={data.parsed.a} b={data.parsed.b ?? "B"} />}
      {data.flags.map((f) => <Alert key={f} tone="warn">{f}</Alert>)}
      <p className="text-xs text-muted">Records found: {data.coverage.map((c) => `${c.mention}: ${c.results} result(s) in ${c.records} record(s)`).join(" · ")}</p>
      <ClaimEvidence title="Supporting" items={data.supporting} total={data.summary.supporting} />
      <ClaimEvidence title="Contradicting" items={data.contradicting} total={data.summary.contradicting} />
      <ClaimEvidence title="Inconclusive" items={data.inconclusive} total={data.summary.inconclusive} />
      <p className="text-xs text-muted"><Badge status={data.confidence.level} /> confidence limited by: {data.confidence.reasons.join("; ")}.</p>
      <AskLinks questions={data.research_questions} />
      <Notes notes={data.notes} />
      <Stages r={data} />
    </div>
  );
}

const CLAIM_FORMS = ["<model> outperforms <model> on <dataset> [in <metric>]", "<model> achieves <value> <metric> on <dataset>",
  "<model> was trained with <setting> <value> [on <dataset>]"];

// ---------------------------------------------------------------------------------------------- Generalization and Transferability Analyzer
type TRow = Side & { benchmark: string; protocol?: string | null };
type TransferResult = Base & { model: string; verdict: string; source: { benchmarks: string[]; results: TRow[]; chosen: string } | null;
  benchmarks: { name: string; records: number; papers: number; tasks: string[] }[];
  transfers: { source: TRow; target: TRow; same_metric: boolean; change: number | null; note: string }[]; elsewhere: TRow[];
  untested: { name: string; task?: string | null; papers: number; models: string[] }[];
  limitations: { chunk_id: string; title?: string | null; section?: string | null; text?: string | null; text_withheld: boolean; url?: string | null }[];
  research_questions: string[]; summary: Record<string, number> };

function TransferTab({ params, go }: TabProps) {
  const model = params.get("model") ?? "";
  const source = params.get("source") ?? "";
  const [m, setM] = useState(model);
  const [s, setS] = useState(source);
  const { data, busy, error, run } = useRun<TransferResult>();
  useEffect(() => {
    if (model.trim().length >= 2) void run(`/api/analysis/transfer?model=${encodeURIComponent(model)}${source ? `&source=${encodeURIComponent(source)}` : ""}`);
  }, [model, source, run]);
  return (
    <section>
      <form onSubmit={(e) => (e.preventDefault(), go({ model: m.trim() || null, source: s.trim() || null }))} className="grid gap-2 md:grid-cols-[1fr_1fr_auto]">
        <input value={m} onChange={(e) => setM(e.target.value)} placeholder="Model or method (e.g. as named in a paper)" aria-label="Model"
          className="rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
        <input value={s} onChange={(e) => setS(e.target.value)} placeholder="Source dataset (optional)" aria-label="Source dataset"
          className="rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
        <Button type="submit" disabled={m.trim().length < 2 || busy}>Analyze</Button>
      </form>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {data && !busy && model && <div className="mt-4"><TransferView data={data} /></div>}
    </section>
  );
}

export function TransferView({ data }: { data: TransferResult }) {
  const cell = (r: TRow) => <><span className="text-ink">{r.benchmark}</span> · {r.metric} <span className="font-mono text-ink">{r.score}</span>{r.split ? ` (${r.split})` : ""}{r.protocol ? `, ${r.protocol}` : ""}</>;
  const coverage = [...data.benchmarks.map((b) => ({ label: b.name, value: b.papers, on: true, note: `${b.records} record(s)` })),
    ...data.untested.map((u) => ({ label: u.name, value: u.papers, on: false, note: u.task ?? undefined }))];
  const changes = data.transfers.filter((t) => t.change !== null).slice(0, 8)
    .map((t) => ({ label: `${t.source.benchmark} → ${t.target.benchmark}`, from: String(t.source.score), to: String(t.target.score), change: t.change as number }));
  return (
        <div className="space-y-4 text-sm">
          <p className="flex flex-wrap items-center gap-2"><Badge status={data.verdict} />
            <span className="text-muted">{data.summary.benchmarks} benchmark(s) in {data.summary.papers} paper(s) · {data.summary.transfers} direct transfer(s),
              {" "}{data.summary.measured} with a measured change · {data.summary.elsewhere} result(s) from other papers · {data.summary.untested} untested benchmark(s)</span></p>
          {data.source && <p className="text-muted">Source: <span className="text-ink">{data.source.benchmarks.join(", ")}</span> ({data.source.chosen})</p>}
          <CoverageBars title={`Where ${data.model} has been evaluated, and benchmarks of its tasks it has no record on`} rows={coverage}
            onLabel="Has a record" offLabel="Untested" />
          <ChangeBars rows={changes} />
          {data.transfers.length > 0 && (
            <div className="overflow-x-auto rounded-lg border border-line">
              <table className="w-full min-w-[720px] text-left">
                <thead className="bg-surface-2 text-xs text-muted"><tr><th className="px-2 py-1.5 font-medium">Source result</th><th className="px-2 py-1.5 font-medium">Target result</th><th className="px-2 py-1.5 font-medium">Change</th><th className="px-2 py-1.5 font-medium">Paper</th></tr></thead>
                <tbody>
                  {data.transfers.map((t, i) => (
                    <tr key={i} className="border-t border-line align-top">
                      <td className="px-2 py-1.5">{cell(t.source)}</td>
                      <td className="px-2 py-1.5">{cell(t.target)}<EvidenceList evidence={t.target.evidence} /></td>
                      <td className="px-2 py-1.5">{t.change !== null ? <span className="font-mono text-ink">{t.change > 0 ? "+" : ""}{Number(t.change.toFixed(4))}</span> : <span className="text-xs text-faint">{t.note}</span>}</td>
                      <td className="px-2 py-1.5"><PaperLine s={t.target} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data.elsewhere.length > 0 && (
            <Block title={`Evaluated by other papers (${data.elsewhere.length})`} empty="">
              {data.elsewhere.map((r, i) => <li key={i} className="py-1">{cell(r)} <PaperLine s={r} /><EvidenceList evidence={r.evidence} /></li>)}
            </Block>
          )}
          <Block title={`Untested in the corpus (${data.untested.length})`} empty="No benchmark of the model's tasks lacks a record of it.">
            {data.untested.map((u) => <li key={u.name} className="py-1"><span className="text-ink">{u.name}</span> <span className="text-xs text-muted">{u.task ? `${u.task} · ` : ""}{u.papers} paper(s) report results, e.g. {u.models.slice(0, 3).join(", ")}</span></li>)}
          </Block>
          {data.untested.length > 0 && <p className="text-xs text-faint">No record of this model on these benchmarks; that is not evidence it fails there.</p>}
          {data.limitations.length > 0 && (
            <div>
              <h2 className="font-semibold text-ink">What the papers say about shift and transfer</h2>
              {data.limitations.map((p) => (
                <blockquote key={p.chunk_id} className="mt-2 border-l-2 border-line pl-2 text-xs text-muted">
                  <span className="font-semibold text-faint">{p.title} · {p.section}</span>{" "}
                  {p.text_withheld || !p.text ? <em>Text withheld by the source&apos;s licence.</em> : `“${p.text}”`}
                  {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">source</a>}
                </blockquote>
              ))}
            </div>
          )}
          <AskLinks questions={data.research_questions} />
          <Notes notes={data.notes} />
          <Stages r={data} />
        </div>
  );
}

// ---------------------------------------------------------------------------------------------- Next Experiment Planner
type Ref = { experiment: string; label: string; metric?: string | null; score?: string | null; evidence?: Evidence[] };
type Rec = { kind: string; title: string; question: string; why: string; controls: string[]; evidence: Ref[]; establishes: string; cannot_establish: string;
  link?: { tab: string; a: string; b: string } };
type PlanResult = Base & { experiment: NodeRef; paper: NodeRef; model: string; benchmark?: string | null; goal?: string | null; recommendations: Rec[];
  reported_ablations: { sibling: Ref; removes: string[]; this?: Ref | null }[]; compute: Record<string, string> | null; summary: Record<string, number> };

function PlanTab({ params, go }: TabProps) {
  const experiment = params.get("experiment");
  const goal = params.get("goal") ?? "";
  const [g, setG] = useState(goal);
  const { data, busy, error, run } = useRun<PlanResult>();
  useEffect(() => {
    if (experiment) void run(`/api/analysis/plan?experiment=${encodeURIComponent(experiment)}${goal ? `&goal=${encodeURIComponent(goal)}` : ""}`);
  }, [experiment, goal, run]);
  return (
    <section>
      <div className="grid gap-2 md:grid-cols-2">
        <Picker types="Experiment" placeholder="Search the experiment to build on" onPick={(h) => go({ experiment: h.id })} />
        <form onSubmit={(e) => (e.preventDefault(), go({ goal: g.trim() || null }))} className="flex gap-2">
          <input value={g} onChange={(e) => setG(e.target.value)} placeholder="Your research goal (optional, kept with the plan)" aria-label="Research goal" maxLength={600}
            className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
          <Button type="submit" variant="secondary">Set</Button>
        </form>
      </div>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {experiment && data && !busy && <div className="mt-4"><PlanView data={data} /></div>}
    </section>
  );
}

const KIND_STRIPE: Record<string, string> = { reproduce: "var(--color-warn)", ablation: "var(--color-accent)", missing_baseline: "var(--color-accent)",
  robustness: "var(--color-ok)", reporting: "var(--color-faint)" };

export function PlanView({ data }: { data: PlanResult }) {
  const kinds = ["reproduce", "ablation", "missing_baseline", "robustness", "reporting"].filter((k) => data.summary[k]);
  return (
        <div className="space-y-4 text-sm">
          <div>
            <h2 className="text-lg font-semibold text-ink">{data.experiment.label}</h2>
            <p className="text-muted">{data.paper.label}{data.goal ? <> · goal: <span className="text-ink">{data.goal}</span></> : null}</p>
            {data.compute && <p className="text-xs text-faint">Original run: {Object.entries(data.compute).map(([k, v]) => `${k} ${v}`).join(" · ")}</p>}
          </div>
          {kinds.length > 0 && (
            <div className="rounded-lg border border-line bg-surface p-3">
              <p className="text-xs font-medium text-muted">Plan at a glance ({data.recommendations.length} proposals, highest priority first)</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {kinds.map((k) => <span key={k} className="inline-flex items-center gap-1.5 text-xs"><Badge status={k}>{words(k)}</Badge> <span className="font-medium text-ink">{data.summary[k]}</span></span>)}
                {data.reported_ablations.length > 0 && <span className="text-xs text-muted">· {data.reported_ablations.length} ablation(s) already reported</span>}
              </div>
            </div>
          )}
          {data.recommendations.length === 0 && <Alert tone="info">The records around this experiment suggest no next step.</Alert>}
          <ol className="space-y-3">
            {data.recommendations.map((r, i) => (
              <li key={i} className="rounded-lg border border-line border-l-4 bg-surface p-3" style={{ borderLeftColor: KIND_STRIPE[r.kind] }}>
                <p className="flex flex-wrap items-center gap-2"><span className="font-mono text-xs text-faint">{i + 1}</span> <Badge status={r.kind}>{words(r.kind)}</Badge>
                  <span className="font-medium text-ink">{r.title}</span></p>
                <p className="mt-1 text-ink">{r.question}</p>
                <p className="mt-1 text-muted">Why: {r.why}</p>
                {r.controls.length > 0 && <p className="mt-1 text-xs text-muted">Hold fixed: {r.controls.join("; ")}</p>}
                <p className="mt-1 text-xs text-muted">Could establish: {r.establishes}. Could not establish: {r.cannot_establish}.</p>
                {r.evidence.length > 0 && (
                  <ul className="mt-1 space-y-0.5 text-xs">
                    {r.evidence.map((s, j) => <li key={j}><span className="text-muted">{s.label}</span>{s.score ? <> · {s.metric} <span className="font-mono text-ink">{s.score}</span></> : null}<EvidenceList evidence={s.evidence} /></li>)}
                  </ul>
                )}
                {r.link && <Link href={`/analyze?tab=${r.link.tab}&a=${encodeURIComponent(r.link.a)}&b=${encodeURIComponent(r.link.b)}`} className="mt-1 inline-block text-xs text-accent hover:underline">Investigate the score gap</Link>}
              </li>
            ))}
          </ol>
          {data.reported_ablations.length > 0 && (
            <Block title="Ablations the paper already reports" empty="">
              {data.reported_ablations.map((a, i) => (
                <li key={i} className="py-1">Without {a.removes.join(" and ")}: <span className="text-ink">{a.sibling.label}</span> {a.sibling.metric} <span className="font-mono">{a.sibling.score}</span>
                  {a.this?.score ? <> (this experiment: <span className="font-mono">{a.this.score}</span>)</> : null}</li>
              ))}
            </Block>
          )}
          <Notes notes={data.notes} />
          <Stages r={data} />
        </div>
  );
}

// ---------------------------------------------------------------------------------------------- Literature Review Completeness Auditor
type ReviewResult = Base & { verdict: string; recognised: { datasets: string[]; models: string[]; tasks: string[]; cited: string[]; cited_in_corpus: string[] };
  scope: { tasks: string[]; papers: number; experiments: number; years: number[] | null; corpus_release?: string | null; corpus_papers: number; providers: string[] | null; runtime_query: string | null };
  missing_benchmarks: { name: string; papers: number }[]; missing_models: { family: string; example: string; papers: number }[];
  thin_evidence: { name: string; papers: number }[]; conflicts: { experiment: string; label: string; field: string; value_a: unknown; value_b: unknown; explanation?: string | null }[];
  recency: { newest_cited_year: number | null; newer_scope_papers: number | null };
  limitations: { chunk_id: string; title?: string | null; section?: string | null; text?: string | null; text_withheld: boolean; url?: string | null }[];
  discovered: { title?: string | null; url: string; provider?: string; year?: number | null; in_corpus: boolean; cited_in_text: boolean }[];
  scope_benchmarks?: { name: string; papers: number; named: boolean }[]; scope_families?: { family: string; example: string; papers: number; named: boolean }[] };

function ReviewTab() {
  const [text, setText] = useState("");
  const [external, setExternal] = useState(false);
  const { data, busy, error, run } = useRun<ReviewResult>();
  const submit = () => void run("/api/analysis/review", { method: "POST", body: JSON.stringify({ text: text.trim(), runtime_search: external }) });
  return (
    <section>
      <form onSubmit={(e) => (e.preventDefault(), submit())} className="space-y-2">
        <label className="block text-sm text-muted" htmlFor="review-text">Research question or draft literature review</label>
        <textarea id="review-text" value={text} onChange={(e) => setText(e.target.value)} rows={6} maxLength={20000}
          placeholder="Paste the question or the review text. It stays in ResearchX."
          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent" />
        <div className="flex flex-wrap items-center justify-between gap-2">
          <label className="inline-flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={external} onChange={(e) => setExternal(e.target.checked)} /> Also search Semantic Scholar and OpenAlex (sends only the recognised task and dataset names, never your text)
          </label>
          <Button type="submit" disabled={text.trim().length < 20 || busy}>Audit coverage</Button>
        </div>
      </form>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
      {busy && <div className="flex justify-center py-8 text-faint"><Spinner /></div>}
      {data && !busy && <div className="mt-4"><ReviewView data={data} /></div>}
    </section>
  );
}

export function ReviewView({ data }: { data: ReviewResult }) {
  const chips = (xs: string[]) => (xs.length ? xs.map((x) => <span key={x} className="mr-1 inline-block rounded bg-surface-2 px-1.5 py-0.5 text-xs text-ink">{x}</span>) : <span className="text-xs text-faint">none</span>);
  return (
        <div className="space-y-4 text-sm">
          <div className="rounded-lg border border-line bg-surface p-3">
            <p><Badge status={data.verdict} /> <span className="ml-1 text-muted">Searched scope: {data.scope.tasks.join(", ") || "the named datasets"} · {data.scope.papers} of {data.scope.corpus_papers} corpus papers
              · {data.scope.experiments} experiment records{data.scope.years ? ` · ${data.scope.years[0]}–${data.scope.years[1]}` : ""}
              {data.scope.corpus_release ? ` · corpus ${data.scope.corpus_release}` : ""} · runtime providers: {data.scope.providers?.join(", ") ?? "none"}</span></p>
            {data.scope.runtime_query && <p className="mt-1 text-xs text-faint">Sent to the providers: “{data.scope.runtime_query}”</p>}
            <div className="mt-2 grid gap-1 md:grid-cols-2">
              <p>Datasets: {chips(data.recognised.datasets)}</p><p>Tasks: {chips(data.recognised.tasks)}</p>
              <p>Models: {chips(data.recognised.models)}</p><p>Cited (arXiv): {data.recognised.cited.length} · in the corpus: {data.recognised.cited_in_corpus.length}</p>
            </div>
          </div>
          <CoverageBars title="Benchmarks in scope: named in your text or not" onLabel="Named" offLabel="Missing"
            rows={(data.scope_benchmarks ?? []).filter((b) => b.named || b.papers >= 2).map((b) => ({ label: b.name, value: b.papers, on: b.named }))} />
          <CoverageBars title="Model families in scope reported by two or more papers: named in your text or not" onLabel="Named" offLabel="Missing"
            rows={(data.scope_families ?? []).filter((f) => f.papers >= 2).map((f) => ({ label: f.example, value: f.papers, on: f.named, note: `family ${f.family}` }))} />
          <Block title={`Benchmarks other scope papers report that the text never names (${data.missing_benchmarks.length})`} empty="None.">
            {data.missing_benchmarks.map((b) => <li key={b.name} className="py-1"><span className="text-ink">{b.name}</span> <span className="text-xs text-muted">{b.papers} paper(s)</span></li>)}
          </Block>
          <Block title={`Model families other scope papers report that the text never names (${data.missing_models.length})`} empty="None.">
            {data.missing_models.map((m) => <li key={m.family} className="py-1"><span className="text-ink">{m.example}</span> <span className="text-xs text-muted">family “{m.family}”, {m.papers} paper(s)</span></li>)}
          </Block>
          <Block title={`Named entities resting on a single corpus paper (${data.thin_evidence.length})`} empty="None.">
            {data.thin_evidence.map((t) => <li key={t.name} className="py-1 text-ink">{t.name}</li>)}
          </Block>
          {data.conflicts.length > 0 && (
            <Block title={`Recorded conflicts involving named entities (${data.conflicts.length})`} empty="">
              {data.conflicts.map((c, i) => <li key={i} className="py-1 text-muted">{c.label}: {c.field} {String(c.value_a)} vs {String(c.value_b)}{c.explanation ? ` — ${c.explanation}` : ""}</li>)}
            </Block>
          )}
          {data.recency.newest_cited_year && <p className="text-muted">Newest cited corpus paper: {data.recency.newest_cited_year}; {data.recency.newer_scope_papers} scope paper(s) are newer.</p>}
          {data.limitations.length > 0 && (
            <div>
              <h2 className="font-semibold text-ink">Limitations stated by the most-cited scope papers</h2>
              {data.limitations.map((p) => (
                <blockquote key={p.chunk_id} className="mt-2 border-l-2 border-line pl-2 text-xs text-muted">
                  <span className="font-semibold text-faint">{p.title} · {p.section}</span>{" "}
                  {p.text_withheld || !p.text ? <em>Text withheld by the source&apos;s licence.</em> : `“${p.text}”`}
                  {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="ml-1 text-accent hover:underline">source</a>}
                </blockquote>
              ))}
            </div>
          )}
          {data.discovered.length > 0 && (
            <Block title={`Found at runtime (${data.discovered.length}) — search metadata, not validated evidence`} empty="">
              {data.discovered.map((d) => (
                <li key={d.url} className="py-1"><a href={d.url} target="_blank" rel="noreferrer" className="text-accent hover:underline">{d.title ?? d.url}</a>
                  <span className="ml-1 text-xs text-muted">{d.provider}{d.year ? ` · ${d.year}` : ""} · {d.in_corpus ? "in the corpus" : "not in the corpus"}{d.cited_in_text ? " · cited in your text" : ""}</span></li>
              ))}
            </Block>
          )}
          <Notes notes={data.notes} />
          <Stages r={data} />
        </div>
  );
}

// ---------------------------------------------------------------------------------------------- the analyses in the chat
// The research chat can run six analyses on what the user types; the card shows the same view as the Analyze page (or a
// summary of it) and links to it. Cards are not saved into the conversation.
export type ChatMode = "papers" | "benchmark" | "claim" | "transfer" | "plan" | "review";
export const CHAT_MODES: Record<ChatMode, { label: string; placeholder: string; analyze: (q: string) => string }> = {
  papers: { label: FEATURE.papers, placeholder: "Compare <title, DOI or arXiv id> vs <…>, or: find recent papers on <method> and compare their settings",
    analyze: () => "/analyze?tab=papers" },
  benchmark: { label: FEATURE.benchmark, placeholder: "Which <benchmark> <metric> results are actually comparable?", analyze: (q) => `/analyze?tab=benchmark&q=${encodeURIComponent(q)}` },
  claim: { label: FEATURE.claim, placeholder: "A claim, e.g. <model> outperforms <model> on <dataset>", analyze: (q) => `/analyze?tab=claim&claim=${encodeURIComponent(q)}` },
  transfer: { label: FEATURE.transfer, placeholder: "A model, optionally with its source dataset: <model> on <dataset>", analyze: (q) => `/analyze?tab=transfer&model=${encodeURIComponent(q)}` },
  plan: { label: FEATURE.plan, placeholder: "The experiment to build on: <model> on <dataset>", analyze: () => "/analyze?tab=plan" },
  review: { label: FEATURE.review, placeholder: "A research question or a draft literature review (stays in ResearchX)", analyze: () => "/analyze?tab=review" },
};

export async function runAnalysis(mode: ChatMode, q: string): Promise<unknown> {
  const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
  if (mode === "papers") return api("/api/analysis/papers/ask", post({ text: q, external: false }));  // outside the corpus only when the text asks for it
  if (mode === "benchmark") return api(`/api/analysis/benchmark?q=${encodeURIComponent(q)}`);
  if (mode === "claim") return api("/api/analysis/claim", post({ claim: q }));
  if (mode === "transfer") return api(`/api/analysis/transfer?model=${encodeURIComponent(q)}`);
  if (mode === "plan") return api(`/api/analysis/plan?query=${encodeURIComponent(q)}`);
  return api("/api/analysis/review", post({ text: q, runtime_search: false }));
}

/** The Analyze link that reproduces a chat result of Paper-to-Paper Diff (corpus ids + typed references + topic). */
function papersHref(data: PapersResult): string {
  const inputs = ((data as { inputs?: { papers?: string[]; refs?: string[]; topic?: string | null; external?: boolean } }).inputs) ?? {};
  const p = new URLSearchParams({ tab: "papers" });
  const corpus = (inputs.papers ?? []).filter((x) => x.startsWith("paper:"));
  if (corpus.length) p.set("papers", corpus.join(","));
  (inputs.refs ?? []).forEach((r) => p.append("ref", r));
  if (inputs.topic) p.set("topic", inputs.topic);
  if (inputs.external) p.set("external", "1");
  return `/analyze?${p.toString()}`;
}

function PapersCard({ data }: { data: PapersResult }) {
  if (data.kind === "paper_resolution") return <RuntimePanel rt={data.runtime} />;
  if (data.kind === "paper_configurations") return <InventoryView data={data} />;
  return (
    <div className="space-y-2 text-sm">
      <ul className="space-y-1">
        {data.papers.map((p) => <li key={p.id} className="text-ink">{p.title} {p.year ? `(${p.year})` : ""} {p.origin === "runtime" ? <Badge status="runtime">runtime</Badge> : <Badge status="corpus">corpus</Badge>}</li>)}
      </ul>
      {data.fairness.map((f) => (
        <p key={f.papers.join()}><Badge status={f.verdict} /> <span className="text-ink">{f.why}</span>
          {!!f.stage_differences?.length && <span className="block text-xs text-muted">Same-stage differences: {f.stage_differences.map((d) => `${d.label} (${d.stage_label})`).join(", ")}</span>}</p>
      ))}
      <p className="text-xs text-muted">{data.summary.rows} rows: {data.summary.same ?? 0} same, {data.summary.different ?? 0} different, {data.summary.partial ?? 0} partly stated, {data.summary.unknown ?? 0} unknown.</p>
      {data.runtime && data.runtime.papers.length > 0 && <p className="text-xs text-muted">{data.runtime.papers.length} paper(s) read outside the corpus at request time (unreviewed).</p>}
    </div>
  );
}

function BenchmarkCard({ data }: { data: BResult }) {
  const ranked = data.rows.filter((r) => r.rank).slice(0, 5);
  return (
    <div className="space-y-2 text-sm">
      <p className="text-ink">{data.benchmark.name} · {data.metric.name}</p>
      <p className="flex flex-wrap gap-2 text-xs">
        {(["comparable", "conditionally_comparable", "incompatible", "unknown"] as const).map((k) => <span key={k} className="inline-flex items-center gap-1"><Badge status={k.toUpperCase()} /> {data.summary[k] ?? 0}</span>)}
      </p>
      <p className="text-xs text-muted">Reference: {Object.entries(data.reference.conditions).map(([k, v]) => `${k} = ${v}`).join(", ") || "none"}.</p>
      {ranked.length > 0 ? (
        <ol className="list-decimal pl-5 text-xs">
          {ranked.map((r) => <li key={r.result}><span className="text-ink">{r.model}</span> <span className="font-mono">{r.score}</span> <span className="text-faint">{r.paper_title}</span> {r.runtime && <Badge status="runtime">runtime</Badge>}</li>)}
        </ol>
      ) : <p className="text-xs text-muted">Nothing is ranked: no result states the same split and protocol as the reference.</p>}
    </div>
  );
}

export function AnalysisCard({ mode, data, query }: { mode: ChatMode; data: unknown; query: string }) {
  const m = CHAT_MODES[mode];
  const href = mode === "plan" ? `/analyze?tab=plan&experiment=${encodeURIComponent((data as PlanResult).experiment.id)}`
    : mode === "papers" ? papersHref(data as PapersResult)
    : mode === "benchmark" ? `/analyze?tab=benchmark&benchmark=${encodeURIComponent((data as BResult).benchmark.id)}&metric=${encodeURIComponent((data as BResult).metric.id)}`
    : m.analyze(query);
  return (
    <section className="rounded-2xl border border-line bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-ink"><I.Bars width={15} height={15} /> {m.label}</h3>
        {mode !== "review" && <Link href={href} className="text-xs text-accent hover:underline">Open in Analyze</Link>}
      </div>
      {mode === "papers" && <PapersCard data={data as PapersResult} />}
      {mode === "benchmark" && <BenchmarkCard data={data as BResult} />}
      {mode === "claim" && <ClaimView data={data as ClaimResult} />}
      {mode === "transfer" && <TransferView data={data as TransferResult} />}
      {mode === "plan" && <PlanView data={data as PlanResult} />}
      {mode === "review" && <ReviewView data={data as ReviewResult} />}
    </section>
  );
}
