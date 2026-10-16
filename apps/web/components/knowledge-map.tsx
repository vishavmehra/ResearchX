"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { authRedirect } from "@/lib/api";
import { toggleTray, TRAY_MAX, useTray } from "@/lib/diff-tray";
import { FEATURE } from "@/lib/features";
import {
  graphApi,
  isRuntimeId,
  NODE_TYPES,
  runtimeQuestions,
  type AskResult,
  type Contradiction,
  type Diff,
  type Discovery,
  type EdgeDetail,
  type Evidence,
  type GEdge,
  type GNode,
  type Landscape,
  type NodeDetail,
  type Promotion,
  type Question,
  type RunGraph,
  type RunSummary,
} from "@/lib/graph";
import { hue, MapCanvas, MAX_NODES, type CanvasHandle } from "@/components/map-canvas";
import { Alert, Button, cx, Spinner } from "@/components/ui";
import * as I from "@/components/icons";

// The Knowledge Map: the curated corpus graph (canonical) and, when a research run is open, what that run found for its
// question (runtime). Nothing here decides what is true: every node and edge comes from the backend with its derivation,
// evidence and status, and the panel shows them as recorded.

type Layer = "all" | "canonical" | "runtime" | "private" | "evidence";
type Selection = { kind: "node"; node: GNode } | { kind: "edge"; edge: GEdge } | null;
type Expansion = { offset: number; total: number; hasMore: boolean; added: string[] };
type Rec = Record<string, unknown>;

const LAYERS: { key: Layer; label: string; title: string }[] = [
  { key: "all", label: "All", title: "Everything on the map" },
  { key: "canonical", label: "Canonical", title: "What ResearchX already knew: the curated corpus" },
  { key: "runtime", label: "Runtime", title: "What the open research run found for its question outside the corpus" },
  { key: "private", label: "Private", title: "Your own documents (visible only to you)" },
  { key: "evidence", label: "Evidence-backed", title: "Only links that carry recorded evidence" },
];
const QUICK = ["Experiment", "Paper", "Model", "Dataset", "Metric"];
const ORIGIN: Record<string, string> = {
  canonical: "Corpus", runtime_external: "Found at runtime", private_document: "Your document", user_input: "Your question", derived: "Derived at runtime",
};
const STATUS_TONE: Record<string, string> = {
  validated: "bg-ok-soft text-ok", promoted: "bg-accent-soft text-accent", related: "bg-surface-2 text-muted", unverified: "bg-warn-soft text-warn",
  rejected: "bg-bad-soft text-bad",
};
const STRENGTH: Record<string, { label: string; tone: string }> = {
  EXACT: { label: "Recorded", tone: "bg-surface-2 text-ink" },
  STRONG: { label: "Strong", tone: "bg-accent-soft text-accent" },
  RELATED: { label: "Related", tone: "bg-surface-2 text-muted" },
  POSSIBLE: { label: "Possible", tone: "bg-warn-soft text-warn" },
  CONTRADICTORY: { label: "Contradictory", tone: "bg-bad-soft text-bad" },
};
const FIELD_TONE: Record<string, string> = {
  IDENTICAL: "text-ok", EQUIVALENT: "text-ok", DIFFERENT: "text-warn", MATERIALLY_DIFFERENT: "text-warn", UNKNOWN: "text-faint",
  CONFLICTING: "text-bad", NOT_APPLICABLE: "text-faint",
};
// rx.runtime_sources.state of an external source (the corpus promotion workflow)
const REVIEW: Record<string, string> = {
  RUNTIME: "not proposed", CANDIDATE: "proposed, awaiting review", PROMOTED: "promoted to the corpus", REJECTED: "rejected by review",
};
const COMPARISON: Record<string, string> = {
  ALIGNED_EXPERIMENT: "Aligned evaluation setup", MATERIALLY_DIFFERENT: "Materially different", RELATED_CONTEXT: "Related context",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence to align", CONFLICTING_EVIDENCE: "Conflicting evidence",
};

const fmt = (v: unknown): string =>
  v === null || v === undefined || v === "" ? "—" : Array.isArray(v) ? v.map(fmt).join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v);
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** A question for the research chat about a node (the chat answers it with the full grounding pipeline). */
function chatQuestion(n: GNode): string {
  const map: Record<string, string> = {
    Experiment: `What configuration and results are reported for ${n.label}?`,
    Model: `On which datasets has ${n.label} been evaluated, and with what results?`,
    Dataset: `Which models report results on ${n.label}, and with which metrics?`,
    Metric: `Which models report ${n.label}, and on which datasets?`,
    Method: `Which experiments use ${n.label}, and what results do they report?`,
    Topic: `What experimental results are reported for ${n.label}?`,
    Paper: `What experiments and results does "${n.label}" report?`,
    Source: `What experiments and results does "${n.label}" report?`,
  };
  return map[n.type] ?? n.label;
}

// ---------------------------------------------------------------------------------------------------------------- page
export function KnowledgeMap() {
  const router = useRouter();
  const params = useSearchParams();
  const runId = params.get("run");
  const focusParam = params.get("node");
  const canvas = useRef<CanvasHandle>(null);
  const nodes = useRef(new Map<string, GNode>());
  const edges = useRef(new Map<string, GEdge>());
  const expanded = useRef(new Map<string, Expansion>());
  const [rev, setRev] = useState(0);
  const [run, setRun] = useState<RunGraph | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [layer, setLayer] = useState<Layer>("all");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [compare, setCompare] = useState<GNode[]>([]);
  const [legend, setLegend] = useState(true);
  const [landscape, setLandscape] = useState<Landscape | null>(null);

  const fail = useCallback(
    (e: unknown) => {
      const to = authRedirect(e);
      if (to) router.replace(to);
      else setError(e instanceof Error ? e.message : "The request failed.");
    },
    [router],
  );

  const setUrl = useCallback(
    (next: { run?: string | null; node?: string | null }) => {
      const p = new URLSearchParams(window.location.search);
      for (const [k, v] of Object.entries(next)) {
        if (v) p.set(k, v);
        else p.delete(k);
      }
      const q = p.toString();
      router.replace(`/map${q ? `?${q}` : ""}`, { scroll: false });
    },
    [router],
  );

  const addToMap = useCallback((ns: GNode[], es: GEdge[], around?: string) => {
    const dropped = canvas.current?.add(ns, es, around) ?? 0;
    for (const n of ns) if (canvas.current?.has(n.id)) nodes.current.set(n.id, { ...nodes.current.get(n.id), ...n });
    for (const e of es) if (nodes.current.has(e.source) && nodes.current.has(e.target)) edges.current.set(e.id, e);
    if (dropped) setNotice(`The map shows at most ${MAX_NODES} nodes; ${dropped} more were not added. Collapse a node or reset the map to explore further.`);
    setRev((r) => r + 1);
  }, []);

  const clearMap = useCallback(() => {
    canvas.current?.clear();
    nodes.current.clear();
    edges.current.clear();
    expanded.current.clear();
    setSelection(null);
    setCompare([]);
    setRev((r) => r + 1);
  }, []);

  const serverTypes = useMemo(() => (hidden.size ? NODE_TYPES.filter((t) => !hidden.has(t)) : undefined), [hidden]);

  const expand = useCallback(
    async (n: GNode, more = false) => {
      if (isRuntimeId(n.id)) {
        setNotice("Everything this research run recorded is already on the map; expand a corpus node to explore further.");
        return;
      }
      const prev = expanded.current.get(n.id);
      if (prev && !more) {
        canvas.current?.focus(n.id);
        return;
      }
      setBusy(n.id);
      setError(null);
      try {
        const before = new Set(nodes.current.keys());
        const res = await graphApi.neighborhood(n.id, { offset: more && prev ? prev.offset + 40 : 0, limit: 40, nodeTypes: serverTypes });
        addToMap(res.nodes, res.edges, n.id);
        expanded.current.set(n.id, {
          offset: res.offset, total: res.total, hasMore: res.has_more,
          added: [...(prev?.added ?? []), ...res.nodes.map((x) => x.id).filter((id) => id !== n.id && !before.has(id))],
        });
        setRev((r) => r + 1);
      } catch (e) {
        fail(e);
      } finally {
        setBusy(null);
      }
    },
    [addToMap, fail, serverTypes],
  );

  const collapse = useCallback((n: GNode) => {
    const exp = expanded.current.get(n.id);
    if (!exp) return;
    const drop = exp.added.filter((id) => !expanded.current.has(id) && !isRuntimeId(id) && (canvas.current?.neighbours(id) ?? []).every((x) => x === n.id));
    canvas.current?.remove(drop);
    const gone = new Set(drop);
    gone.forEach((id) => nodes.current.delete(id));
    for (const [id, e] of edges.current) if (gone.has(e.source) || gone.has(e.target)) edges.current.delete(id);
    expanded.current.delete(n.id);
    setRev((r) => r + 1);
  }, []);

  const focus = useCallback(
    async (id: string) => {
      setError(null);
      if (!canvas.current?.has(id)) {
        if (isRuntimeId(id)) {
          setNotice("That item belongs to a research run that is not open on the map.");
          return;
        }
        setBusy(id);
        try {
          const before = new Set(nodes.current.keys());
          const res = await graphApi.neighborhood(id, { limit: 30, nodeTypes: serverTypes });
          addToMap([res.center], []);
          addToMap(res.nodes, res.edges, id);
          expanded.current.set(id, { offset: 0, total: res.total, hasMore: res.has_more, added: res.nodes.map((x) => x.id).filter((x) => x !== id && !before.has(x)) });
        } catch (e) {
          fail(e);
          return;
        } finally {
          setBusy(null);
        }
      }
      canvas.current?.focus(id);
      const n = nodes.current.get(id);
      if (n) setSelection({ kind: "node", node: n });
      setUrl({ node: id });
    },
    [addToMap, fail, serverTypes, setUrl],
  );

  // a research run's graph: everything it recorded, laid out around the run
  useEffect(() => {
    if (!runId) {
      setRun(null);
      return;
    }
    let alive = true;
    graphApi
      .run(runId)
      .then((g) => {
        if (!alive) return;
        clearMap();
        setRun(g);
        addToMap(g.nodes, g.edges);
        const root = `run:${g.research_run_id}`;
        canvas.current?.layout(root);
        const rn = g.nodes.find((n) => n.id === root);
        if (rn) setSelection({ kind: "node", node: rn });
      })
      .catch(fail);
    return () => {
      alive = false;
    };
  }, [runId, addToMap, clearMap, fail]);

  useEffect(() => {
    if (!focusParam || (runId && !run)) return;
    if (selection?.kind === "node" && selection.node.id === focusParam) return;
    void focus(focusParam);
    // runs when the URL or the open run changes, not on every selection
  }, [focusParam, run]);

  useEffect(() => {
    graphApi.landscape().then(setLandscape).catch(fail);
  }, [fail]);

  // visibility: layer + node types (client-side over what is loaded; expansions also filter server-side)
  const backed = useMemo(() => {
    const ids = new Set<string>();
    for (const e of edges.current.values())
      if (e.evidence_ids.length || e.type === "CONTRADICTS") {
        ids.add(e.source);
        ids.add(e.target);
      }
    for (const n of nodes.current.values()) if (n.type === "Evidence") ids.add(n.id);
    return ids;
    // recomputed when the map changes (rev)
  }, [rev]);
  useEffect(() => {
    const showNode = (n: GNode) => {
      if (hidden.has(n.type)) return false;
      const o = n.origin ?? "canonical";
      if (layer === "canonical") return o === "canonical";
      if (layer === "runtime") return o !== "canonical" && o !== "private_document";
      if (layer === "private") return o === "private_document" || n.type === "Run";
      if (layer === "evidence") return n.type === "Run" || backed.has(n.id);
      return true;
    };
    canvas.current?.setVisible(showNode, (e) => layer !== "evidence" || e.evidence_ids.length > 0 || e.type === "CONTRADICTS");
  }, [layer, hidden, backed, rev]);

  const selectNode = useCallback((n: GNode) => setSelection({ kind: "node", node: nodes.current.get(n.id) ?? n }), []);
  const addCompare = useCallback((n: GNode) => setCompare((c) => (c.some((x) => x.id === n.id) ? c : [...c, n].slice(-2))), []);
  const resetAll = () => {
    clearMap();
    setUrl({ run: null, node: null });
  };
  const runNode = run?.nodes.find((n) => n.type === "Run");
  const runtimeCount = run ? run.nodes.filter((n) => n.origin === "runtime_external").length : 0;
  const privateCount = run ? run.nodes.filter((n) => n.origin === "private_document").length : 0;
  const empty = !runId && nodes.current.size === 0;
  const selected = selection?.kind === "node" ? nodes.current.get(selection.node.id) ?? selection.node : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-surface px-4 py-2.5">
        <h1 className="font-serif text-lg font-semibold text-ink">Knowledge Map</h1>
        <SearchBox onPick={(n) => void focus(n.id)} />
        <div role="radiogroup" aria-label="Layer" className="flex flex-wrap rounded-lg border border-line bg-bg p-0.5 text-xs">
          {LAYERS.map((l) => (
            <button
              key={l.key}
              role="radio"
              aria-checked={layer === l.key}
              title={l.title}
              onClick={() => setLayer(l.key)}
              className={cx("rounded-md px-2.5 py-1 font-medium transition", layer === l.key ? "bg-surface text-ink shadow-sm" : "text-muted hover:text-ink")}
            >
              {l.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1 text-xs" role="group" aria-label="Node types">
          {QUICK.map((t) => {
            const on = !hidden.has(t);
            return (
              <button
                key={t}
                aria-pressed={on}
                onClick={() =>
                  setHidden((h) => {
                    const next = new Set(h);
                    if (on) next.add(t);
                    else next.delete(t);
                    return next;
                  })
                }
                className={cx("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 transition", on ? "border-line bg-surface text-ink" : "border-dashed border-line text-faint")}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: on ? hue(t) : "transparent", border: `1px solid ${hue(t)}` }} />
                {`${t}s`}
              </button>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" className="px-2.5 py-1.5 text-xs" onClick={() => setLegend((l) => !l)} aria-pressed={legend}>
            Legend
          </Button>
          <Button variant="ghost" className="px-2.5 py-1.5 text-xs" onClick={resetAll}>
            Reset
          </Button>
        </div>
      </div>
      {run && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-surface-2 px-4 py-1.5 text-xs text-muted">
          <span className="font-medium text-ink">Research run</span>
          <span className="max-w-[46ch] truncate" title={runNode?.label}>
            “{runNode?.label}”
          </span>
          <span>
            {plural(runtimeCount, "item")} found outside the corpus{privateCount ? ` · ${plural(privateCount, "private document")}` : ""} · the rest is corpus
            knowledge it used
          </span>
          <button className="ml-auto font-medium text-accent hover:underline" onClick={resetAll}>
            Close run
          </button>
        </div>
      )}
      {(error || notice) && (
        <div className="space-y-2 px-4 pt-2">
          {error && <Alert tone="bad">{error}</Alert>}
          {notice && (
            <div className="flex items-start gap-2 rounded-lg border border-line bg-info-soft px-3 py-2 text-sm text-ink" role="status">
              <span className="flex-1">{notice}</span>
              <button onClick={() => setNotice(null)} aria-label="Dismiss" className="text-muted hover:text-ink">
                <I.Cross width={14} height={14} />
              </button>
            </div>
          )}
        </div>
      )}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto md:flex-row md:overflow-hidden">
        <div className="relative h-[56vh] min-h-80 shrink-0 border-b border-line md:h-auto md:flex-1 md:shrink md:border-b-0">
          <MapCanvas
            ref={canvas}
            onNode={selectNode}
            onEdge={(e) => setSelection({ kind: "edge", edge: e })}
            onExpand={(n) => void expand(n)}
            onBackground={() => setSelection(null)}
          />
          {legend && <Legend onClose={() => setLegend(false)} />}
          {empty && <EmptyState landscape={landscape} onPick={(id) => void focus(id)} />}
          {busy && (
            <div className="absolute right-3 top-3 inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs text-muted shadow-sm">
              <Spinner className="h-3.5 w-3.5" /> Loading
            </div>
          )}
        </div>
        <aside aria-label="Details" className="w-full bg-surface md:min-h-0 md:w-100 md:overflow-y-auto md:border-l md:border-line">
          {compare.length > 0 && <CompareTray items={compare} runId={runId} onRemove={(id) => setCompare((c) => c.filter((x) => x.id !== id))} onFocus={focus} fail={fail} />}
          {selected ? (
            <NodePanel
              key={selected.id}
              node={selected}
              run={run}
              runId={runId}
              edges={[...edges.current.values()].filter((e) => e.source === selected.id || e.target === selected.id)}
              nodeOf={(id) => nodes.current.get(id)}
              expansion={expanded.current.get(selected.id)}
              busy={busy === selected.id}
              onFocus={focus}
              onExpand={expand}
              onCollapse={collapse}
              onCompare={addCompare}
              fail={fail}
            />
          ) : selection?.kind === "edge" ? (
            <EdgePanel key={selection.edge.id} edge={selection.edge} run={run} nodeOf={(id) => nodes.current.get(id)} onFocus={focus} fail={fail} />
          ) : (
            <Overview landscape={landscape} run={run} mapNodes={[...nodes.current.values()]} onFocus={focus} onOpenRun={(id) => setUrl({ run: id, node: null })} fail={fail} />
          )}
        </aside>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------- search
function SearchBox({ onPick }: { onPick: (n: GNode) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<GNode[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    const t = q.trim();
    if (t.length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    const n = ++seq.current;
    setLoading(true);
    const h = setTimeout(() => {
      graphApi
        .search(t)
        .then((r) => {
          if (n !== seq.current) return;
          setResults(r.results);
          setActive(0);
          setOpen(true);
        })
        .catch(() => {
          if (n === seq.current) setResults([]);
        })
        .finally(() => {
          if (n === seq.current) setLoading(false);
        });
    }, 220);
    return () => clearTimeout(h);
  }, [q]);

  const pick = (n: GNode) => {
    onPick(n);
    setOpen(false);
    setQ("");
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter" && results[active]) {
      e.preventDefault();
      pick(results[active]);
    } else if (e.key === "Escape") setOpen(false);
  };
  return (
    <div className="relative w-full max-w-85">
      <label className="relative block">
        <span className="sr-only">Search the knowledge map</span>
        <I.Search className="pointer-events-none absolute left-2.5 top-2.5 text-faint" width={16} height={16} />
        <input
          type="search"
          role="combobox"
          aria-expanded={open && results.length > 0}
          aria-controls="km-search-results"
          aria-autocomplete="list"
          aria-activedescendant={open && results[active] ? `km-opt-${active}` : undefined}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onKey}
          onFocus={() => results.length > 0 && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search topics, models, datasets, papers"
          className="w-full rounded-lg border border-line bg-bg py-2 pl-8 pr-8 text-sm text-ink placeholder:text-faint focus:border-accent"
        />
        {loading && <Spinner className="absolute right-2.5 top-2.5 h-4 w-4 text-faint" />}
      </label>
      {open && (
        <ul id="km-search-results" role="listbox" className="absolute z-20 mt-1 max-h-80 w-full overflow-y-auto rounded-lg border border-line bg-surface py-1 shadow-lg">
          {results.length === 0 ? (
            <li className="px-3 py-2 text-sm text-muted">Nothing in the corpus matches that.</li>
          ) : (
            results.map((n, i) => (
              <li
                key={n.id}
                id={`km-opt-${i}`}
                role="option"
                aria-selected={i === active}
                onMouseDown={(e) => {
                  e.preventDefault();
                  pick(n);
                }}
                onMouseEnter={() => setActive(i)}
                className={cx("flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm", i === active && "bg-surface-2")}
              >
                <TypeDot type={n.type} />
                <span className="min-w-0 flex-1 truncate text-ink">{n.label}</span>
                <span className="shrink-0 text-xs text-faint">
                  {n.type === "Dataset" && typeof n.meta?.entity_type === "string" ? n.meta.entity_type : n.type}
                  {typeof n.meta?.experiments === "number" ? ` · ${n.meta.experiments} exp.` : typeof n.meta?.results === "number" ? ` · ${n.meta.results} results` : ""}
                </span>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------- small parts
function TypeDot({ type }: { type: string }) {
  return <span aria-hidden="true" className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: hue(type) }} />;
}

function NodeChip({ node, onFocus, extra }: { node: GNode; onFocus: (id: string) => void; extra?: ReactNode }) {
  return (
    <button
      onClick={() => onFocus(node.id)}
      className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-line bg-bg px-2 py-1 text-left text-xs text-ink transition hover:border-accent"
      title={`${node.type}: ${node.label}`}
    >
      <TypeDot type={node.type} />
      <span className="truncate">{node.label}</span>
      {extra}
    </button>
  );
}

function Badge({ tone, children, title }: { tone: string; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={cx("inline-flex items-center rounded-full px-2 py-px text-[0.7rem] font-medium", tone)}>
      {children}
    </span>
  );
}

function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="border-t border-line px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-faint">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

function KV({ rows }: { rows: [string, ReactNode][] }) {
  const shown = rows.filter(([, v]) => v !== null && v !== undefined && v !== "" && v !== "—");
  if (!shown.length) return null;
  return (
    <dl className="grid grid-cols-[minmax(92px,auto)_1fr] gap-x-3 gap-y-1 text-sm">
      {shown.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 wrap-break-word text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Groups({ groups, onFocus }: { groups: Record<string, GNode[]>; onFocus: (id: string) => void }) {
  const entries = Object.entries(groups).filter(([, v]) => v.length);
  if (!entries.length) return <p className="text-sm text-muted">No linked records.</p>;
  return (
    <div className="space-y-3">
      {entries.map(([type, items]) => {
        const max = Math.max(...items.map((i) => i.count ?? 0), 1);
        return (
          <div key={type}>
            <p className="mb-1 text-xs font-medium text-muted">{type === "Topic" ? "Topics (tasks)" : `${type}s`}</p>
            <ul className="space-y-0.5">
              {items.map((i) => (
                <li key={i.id}>
                  <button onClick={() => onFocus(i.id)} className="grid w-full grid-cols-[1fr_auto] items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-surface-2">
                    <span className="relative min-w-0">
                      <span className="absolute inset-y-0 left-0 rounded-sm opacity-15" style={{ width: `${((i.count ?? 0) / max) * 100}%`, background: hue(type) }} />
                      <span className="relative block truncate px-1 text-sm text-ink">
                        {i.label}
                        {type === "Dataset" && typeof i.meta?.entity_type === "string" && <span className="text-faint"> · {i.meta.entity_type}</span>}
                      </span>
                    </span>
                    <span className="font-mono text-xs text-muted">{i.count}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

function EvidenceCard({ ev }: { ev: Evidence }) {
  return (
    <li className="rounded-lg border border-line bg-bg p-2.5 text-sm">
      <p className="flex flex-wrap items-center gap-x-2 text-xs text-muted">
        <span className="font-medium text-ink">{ev.field}</span>
        {ev.location_kind && <span>{ev.location_kind.toLowerCase().replaceAll("_", " ")}</span>}
        {ev.status && <span>{ev.status.toLowerCase().replaceAll("_", " ")}</span>}
      </p>
      {ev.text ? (
        <q className="mt-1 block text-[0.85rem] italic leading-snug text-ink">{ev.text}</q>
      ) : ev.text_withheld ? (
        <p className="mt-1 text-xs text-muted">The source&apos;s licence does not allow showing its text here; its location and the extracted value are below.</p>
      ) : null}
      <div className="mt-1.5">
        <KV
          rows={[
            ["Value", fmt(ev.extracted_value)],
            ["Location", ev.source_location ? <span className="break-all font-mono text-[0.7rem]">{ev.source_location}</span> : null],
            ["Retrieved", ev.retrieval_timestamp ? ev.retrieval_timestamp.slice(0, 10) : null],
            ["Extracted by", ev.extraction_model ?? null],
          ]}
        />
      </div>
    </li>
  );
}

function ContradictionCard({ c }: { c: Contradiction }) {
  return (
    <li className="rounded-lg border border-bad/30 bg-bad-soft p-2.5 text-sm">
      <p className="font-medium text-ink">{c.field}</p>
      <p className="mt-0.5 text-ink">
        <strong>{fmt(c.value_a)}</strong> <span className="text-muted">({c.source_a ?? "source A"})</span> vs <strong>{fmt(c.value_b)}</strong>{" "}
        <span className="text-muted">({c.source_b ?? "source B"})</span>
      </p>
      {c.explanation && <p className="mt-1 text-xs text-muted">{c.explanation}</p>}
    </li>
  );
}

// ---------------------------------------------------------------------------------------------------------------- node panel
type NodePanelProps = {
  node: GNode;
  run: RunGraph | null;
  runId: string | null;
  edges: GEdge[];
  nodeOf: (id: string) => GNode | undefined;
  expansion?: Expansion;
  busy: boolean;
  onFocus: (id: string) => void;
  onExpand: (n: GNode, more?: boolean) => void;
  onCollapse: (n: GNode) => void;
  onCompare: (n: GNode) => void;
  fail: (e: unknown) => void;
};

/** The analyses that take this corpus node as input, one click away (same targets as the answer source cards). */
function AnalysisLinks({ node, title }: { node: { id: string; type: string }; title: string }) {
  const tray = useTray();
  const enc = encodeURIComponent;
  const cls = "inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium text-accent hover:bg-surface-2";
  if (node.type === "Paper") {
    const inTray = tray.some((p) => p.id === node.id);
    return (
      <>
        <button className={cx(cls, "disabled:text-faint")} disabled={!inTray && tray.length >= TRAY_MAX} onClick={() => toggleTray({ id: node.id, title })}>
          {inTray ? <I.Check width={13} height={13} /> : <I.Plus width={13} height={13} />} {inTray ? `In ${FEATURE.papers}` : `Add to ${FEATURE.papers}`}
        </button>
        <Link href={`/analyze?tab=recipe&paper=${enc(node.id)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.recipe}</Link>
      </>
    );
  }
  if (node.type === "Experiment")
    return (
      <>
        <Link href={`/analyze?tab=recipe&experiment=${enc(node.id)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.recipe}</Link>
        <Link href={`/analyze?tab=discrepancy&a=${enc(node.id)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.discrepancy}</Link>
        <Link href={`/analyze?tab=plan&experiment=${enc(node.id)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.plan}</Link>
      </>
    );
  if (node.type === "Dataset")
    return <Link href={`/analyze?tab=benchmark&benchmark=${enc(node.id)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.benchmark}</Link>;
  if (node.type === "Model" || node.type === "Method")
    return <Link href={`/analyze?tab=transfer&model=${enc(title)}`} className={cls}><I.Bars width={13} height={13} /> {FEATURE.transfer}</Link>;
  return null;
}

function NodePanel({ node, run, runId, edges, nodeOf, expansion, busy, onFocus, onExpand, onCollapse, onCompare, fail }: NodePanelProps) {
  const runtime = isRuntimeId(node.id) || (!!node.origin && node.origin !== "canonical");
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [loading, setLoading] = useState(!runtime);
  const [scope, setScope] = useState<Landscape | null>(null);

  useEffect(() => {
    if (runtime) return;
    let alive = true;
    graphApi
      .node(node.id)
      .then((d) => {
        if (alive) setDetail(d);
      })
      .catch(fail)
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [node.id, runtime, fail]);

  const questions: Question[] = runtime ? runtimeQuestions(node) : (detail?.questions ?? []);
  const canLandscape = !runtime && ["Topic", "Dataset", "Model", "Method", "Metric", "Paper"].includes(node.type);
  const flags = node.flags ?? [];
  const remaining = expansion ? Math.max(expansion.total - expansion.added.length, 0) : 0;

  return (
    <div className="rx-enter">
      <header className="space-y-1.5 px-4 pb-3 pt-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted">
            <TypeDot type={node.type} /> {node.type}
          </span>
          <Badge tone={node.origin && node.origin !== "canonical" ? "border border-dashed border-line text-ink" : "bg-surface-2 text-muted"}>
            {ORIGIN[node.origin ?? "canonical"]}
          </Badge>
          {node.origin && node.origin !== "canonical" && node.status && <Badge tone={STATUS_TONE[node.status] ?? "bg-surface-2 text-muted"}>{node.status}</Badge>}
          {flags.includes("conflict") && <Badge tone="bg-bad-soft text-bad">conflict recorded</Badge>}
        </div>
        <h2 className="font-serif text-lg font-semibold leading-snug text-ink">{typeof detail?.title === "string" ? detail.title : node.label}</h2>
        <p className="break-all font-mono text-[0.68rem] text-faint">{node.id}</p>
        <div className="flex flex-wrap gap-1.5 pt-1">
          {!runtime && (
            <Button variant="secondary" className="px-2.5 py-1 text-xs" busy={busy} onClick={() => onExpand(node)}>
              {expansion ? "Show on map" : "Expand"}
            </Button>
          )}
          {expansion?.hasMore && (
            <Button variant="secondary" className="px-2.5 py-1 text-xs" busy={busy} onClick={() => onExpand(node, true)}>
              Load more
            </Button>
          )}
          {expansion && (
            <Button variant="ghost" className="px-2.5 py-1 text-xs" onClick={() => onCollapse(node)}>
              Collapse
            </Button>
          )}
          {node.type === "Experiment" && (
            <Button variant="secondary" className="px-2.5 py-1 text-xs" onClick={() => onCompare(node)}>
              Add to compare
            </Button>
          )}
          {canLandscape && (
            <Button variant="ghost" className="px-2.5 py-1 text-xs" onClick={() => graphApi.landscape(node.id).then(setScope).catch(fail)}>
              Area at a glance
            </Button>
          )}
          {node.type !== "Run" && node.type !== "Evidence" && (
            <Link href={`/?q=${encodeURIComponent(chatQuestion(node))}`} className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium text-accent hover:bg-surface-2">
              Ask ResearchX <I.Chevron width={13} height={13} />
            </Link>
          )}
          {!runtime && <AnalysisLinks node={node} title={typeof detail?.title === "string" ? detail.title : node.label} />}
        </div>
        {expansion && (
          <p className="text-xs text-muted">
            {plural(expansion.total, "neighbour")} recorded{remaining ? ` · ${remaining} not shown yet` : ""}
          </p>
        )}
      </header>

      {scope && (
        <Section title="Area at a glance" aside={<button className="text-xs text-muted hover:text-ink" onClick={() => setScope(null)}>Hide</button>}>
          <LandscapeView l={scope} onFocus={onFocus} />
        </Section>
      )}

      {runtime ? (
        <RuntimeDetail node={node} run={run} runId={runId} edges={edges} nodeOf={nodeOf} onFocus={onFocus} fail={fail} />
      ) : loading ? (
        <div className="flex justify-center py-8 text-faint">
          <Spinner />
        </div>
      ) : detail ? (
        <CanonicalDetail d={detail} onFocus={onFocus} />
      ) : null}

      {questions.length > 0 && <AskPanel node={node} questions={questions} runId={runId} onFocus={onFocus} fail={fail} />}
    </div>
  );
}

function CanonicalDetail({ d, onFocus }: { d: NodeDetail; onFocus: (id: string) => void }) {
  const t = d.node.type;
  const contradictions = (d.contradictions as Contradiction[] | undefined) ?? [];
  const related = d.related as Record<string, GNode[]> | undefined;
  const studyLinks = (d.study_links as { other: GNode | null; label: string; why: string; strength: string }[] | undefined) ?? [];
  const authors = (d.authors as string[] | undefined) ?? [];
  const total = (d.authors_total as number | undefined) ?? 0;
  return (
    <>
      {t === "Paper" && (
        <Section title="Paper">
          <KV
            rows={[
              ["Year", fmt(d.year)],
              ["Venue", fmt(d.venue)],
              ["Authors", authors.length ? authors.join(", ") + (total > authors.length ? ` +${total - authors.length}` : "") : null],
              ["Experiments", fmt(d.experiments)],
              ["Cites", fmt(d.cites)],
              ["Cited by", fmt(d.cited_by)],
              [
                "Source",
                typeof d.url === "string" ? (
                  <a href={d.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-accent hover:underline">
                    {d.arxiv_id ? `arXiv ${String(d.arxiv_id)}` : String(d.doi)} <I.External width={12} height={12} />
                  </a>
                ) : null,
              ],
            ]}
          />
          {typeof d.abstract === "string" && d.abstract && <Abstract text={d.abstract} />}
        </Section>
      )}
      {t === "Experiment" && <ExperimentDetail d={d} onFocus={onFocus} />}
      {["Model", "Dataset", "Method", "Metric", "Topic"].includes(t) && (
        <Section title={t === "Topic" ? "Topic (task)" : t}>
          <KV
            rows={[
              ["Entity type", String(d.entity_type ?? "").toLowerCase()],
              ["Experiments", fmt(d.experiments)],
              ["Papers", fmt(d.papers)],
              ["Results", d.results ? fmt(d.results) : null],
              ["Years", Array.isArray(d.years) ? `${d.years[0]}–${d.years[1]}` : null],
              ["Also written", ((d.surface_forms as { original_name: string }[] | undefined) ?? []).map((s) => s.original_name).filter((s) => s !== d.name).slice(0, 6).join(" · ")],
            ]}
          />
        </Section>
      )}
      {t === "Result" && (
        <Section title="Result">
          <KV rows={[["Metric", fmt(d.metric)], ["Score", <strong key="s">{fmt(d.score)}</strong>], ["Split", fmt(d.split)]]} />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {d.experiment ? <NodeChip node={d.experiment as GNode} onFocus={onFocus} /> : null}
            {d.paper ? <NodeChip node={d.paper as GNode} onFocus={onFocus} /> : null}
          </div>
          <EvidenceList items={(d.evidence as Evidence[] | undefined) ?? []} />
        </Section>
      )}
      {t === "Configuration" && (
        <Section title="Configuration">
          {d.experiment ? (
            <div className="mb-2">
              <NodeChip node={d.experiment as GNode} onFocus={onFocus} />
            </div>
          ) : null}
          <ConfigTable rows={(d.fields as Rec[] | undefined) ?? []} />
        </Section>
      )}
      {t === "Evidence" && (
        <Section title="Evidence">
          {d.evidence ? (
            <ul>
              <EvidenceCard ev={d.evidence as Evidence} />
            </ul>
          ) : (
            <p className="text-sm text-muted">Evidence record not found.</p>
          )}
        </Section>
      )}
      {t === "Study" && (
        <Section title="Papers in this study">
          <div className="flex flex-wrap gap-1.5">
            {((d.papers as GNode[] | undefined) ?? []).map((p) => (
              <NodeChip key={p.id} node={p} onFocus={onFocus} />
            ))}
          </div>
        </Section>
      )}
      {studyLinks.length > 0 && (
        <Section title="Candidate study links (unreviewed)">
          <ul className="space-y-2">
            {studyLinks.map((l, i) => (
              <li key={i} className="text-sm">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone={STRENGTH[l.strength]?.tone ?? "bg-surface-2 text-muted"}>{l.label}</Badge>
                  {l.other && <NodeChip node={l.other} onFocus={onFocus} />}
                </div>
                <p className="mt-1 text-xs text-muted">{l.why}</p>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {contradictions.length > 0 && (
        <Section title={`Recorded contradictions (${contradictions.length})`}>
          <ul className="space-y-2">
            {contradictions.map((c) => (
              <ContradictionCard key={c.id} c={c} />
            ))}
          </ul>
        </Section>
      )}
      {related && Object.keys(related).length > 0 && (
        <Section title="Linked in the same experiment records">
          <Groups groups={related} onFocus={onFocus} />
        </Section>
      )}
    </>
  );
}

function ExperimentDetail({ d, onFocus }: { d: NodeDetail; onFocus: (id: string) => void }) {
  const fields = (d.fields as { field: string; value: string; node: string; evidence: number }[] | undefined) ?? [];
  const results = (d.results as { id: string; metric: string | null; score: string | null; split: string | null }[] | undefined) ?? [];
  const extraction = (d.extraction as { method?: string; model?: string } | undefined) ?? {};
  return (
    <>
      <Section title="Experiment">
        {d.paper ? (
          <div className="mb-2">
            <NodeChip node={d.paper as GNode} onFocus={onFocus} />
          </div>
        ) : null}
        <KV
          rows={[
            ["Type", String(d.experiment_type ?? "").toLowerCase().replaceAll("_", " ")],
            ["Record depth", fmt(d.depth)],
            ["Extraction", [extraction.method, extraction.model].filter(Boolean).join(" · ")],
            ["Unresolved fields", d.unresolved_fields ? fmt(d.unresolved_fields) : null],
          ]}
        />
        {typeof d.description === "string" && d.description && <p className="mt-2 text-sm text-muted">{d.description}</p>}
      </Section>
      {fields.length > 0 && (
        <Section title="Model, method, data and task">
          <ul className="space-y-1">
            {fields.map((f) => (
              <li key={f.field} className="flex items-center justify-between gap-2 text-sm">
                <span className="text-muted">{f.field}</span>
                <button onClick={() => onFocus(f.node)} className="truncate text-right font-medium text-accent hover:underline" title={`${f.evidence} evidence span(s)`}>
                  {f.value}
                </button>
              </li>
            ))}
          </ul>
        </Section>
      )}
      <Section title="Configuration">
        <ConfigTable rows={(d.configuration as Rec[] | undefined) ?? []} />
      </Section>
      {results.length > 0 && (
        <Section title={`Results (${results.length})`}>
          <ul className="space-y-1">
            {results.map((r) => (
              <li key={r.id} className="flex items-center justify-between gap-2 text-sm">
                <button onClick={() => onFocus(r.id)} className="truncate text-left text-ink hover:underline">
                  {r.metric ?? "metric unknown"}
                  {r.split ? <span className="text-muted"> · {r.split}</span> : null}
                </button>
                <span className="font-mono font-medium text-ink">{r.score ?? "—"}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

function ConfigTable({ rows }: { rows: Rec[] }) {
  if (!rows.length) return <p className="text-sm text-muted">No configuration field is recorded with evidence.</p>;
  return (
    <ul className="space-y-1">
      {rows.map((r) => (
        <li key={String(r.field)} className="flex items-baseline justify-between gap-3 text-sm">
          <span className="text-muted">{String(r.field)}</span>
          <span className="text-right font-medium text-ink">
            {fmt(r.value)}
            {r.unit ? <span className="text-muted"> {String(r.unit)}</span> : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

function EvidenceList({ items }: { items: Evidence[] }) {
  if (!items.length) return null;
  return (
    <ul className="mt-2 space-y-2">
      {items.map((ev) => (
        <EvidenceCard key={ev.id} ev={ev} />
      ))}
    </ul>
  );
}

function Abstract({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <p className={cx("text-sm leading-relaxed text-muted", !open && "line-clamp-4")}>{text}</p>
      <button onClick={() => setOpen((o) => !o)} className="mt-1 text-xs font-medium text-accent hover:underline">
        {open ? "Show less" : "Show abstract"}
      </button>
    </div>
  );
}

// a node that exists in a research run's graph: provenance, what it is linked to, and why
type RuntimeProps = {
  node: GNode;
  run: RunGraph | null;
  runId: string | null;
  edges: GEdge[];
  nodeOf: (id: string) => GNode | undefined;
  onFocus: (id: string) => void;
  fail: (e: unknown) => void;
};

function RuntimeDetail({ node, run, runId, edges, nodeOf, onFocus, fail }: RuntimeProps) {
  const p = (node.provenance ?? {}) as Rec;
  const m = (node.meta ?? {}) as Rec;
  const fields = (m.fields as { field: string; label: string; value: unknown }[] | undefined) ?? [];
  const [promotion, setPromotion] = useState<Promotion | null>(null);
  const [promoting, setPromoting] = useState(false);
  const canPromote = node.origin === "runtime_external" && (node.type === "Paper" || node.type === "Source") && !!runId;
  const stats = node.type === "Run" ? (run?.stats as Rec | undefined) : undefined;
  const promote = async () => {
    if (!runId) return;
    setPromoting(true);
    try {
      setPromotion(await graphApi.promote(runId, node.id));
    } catch (e) {
      fail(e);
    } finally {
      setPromoting(false);
    }
  };
  return (
    <>
      {node.type === "Run" && stats && (
        <Section title="This research run">
          <KV
            rows={[
              ["Status", m.run_status ? String(m.run_status).toLowerCase() : null],
              ["Evidence", m.evidence_status ? String(m.evidence_status).toLowerCase().replaceAll("_", " ") : null],
              ["Nodes", fmt(stats.nodes)],
              ["Links", fmt(stats.edges)],
              ["Found outside corpus", fmt((stats.by_origin as Rec | undefined)?.runtime_external ?? 0)],
              ["Matched to corpus by id", fmt(stats.merged_by_identifier)],
              ["Uncertain identities", fmt(stats.uncertain_identity)],
              ["Rejected facts", fmt(stats.rejected_facts)],
            ]}
          />
          <p className="mt-2 text-xs text-muted">
            Runtime knowledge stays with this run: it is never added to the shared corpus automatically. External sources can be proposed for review.
          </p>
        </Section>
      )}
      {node.type !== "Run" && (
        <Section title="Provenance">
          <KV
            rows={[
              ["Provider", fmt(p.provider)],
              [
                "Source",
                typeof p.url === "string" ? (
                  <a href={p.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 break-all text-accent hover:underline">
                    {p.url} <I.External width={12} height={12} />
                  </a>
                ) : null,
              ],
              ["DOI", fmt(p.doi)],
              ["arXiv", fmt(p.arxiv_id)],
              ["Retrieved", typeof p.retrieved_at === "string" ? `${p.retrieved_at.replace("T", " ").slice(0, 16)} UTC` : null],
              ["Content hash", typeof p.content_sha256 === "string" ? <span className="font-mono text-[0.7rem]">{p.content_sha256.slice(0, 16)}…</span> : null],
              ["Checks", Array.isArray(p.source_validation) ? (p.source_validation as string[]).map((c) => c.replaceAll("_", " ")).join(" · ") : null],
              ["Also found by", Array.isArray(p.also_found_by) && p.also_found_by.length ? (p.also_found_by as Rec[]).map((a) => String(a.provider)).join(", ") : null],
              ["Scope", fmt(p.scope)],
              ["Extraction", fmt(p.extraction)],
              ["Quoted from", fmt(p.sid)],
              ["Review", REVIEW[String(m.review_state)] ?? null],
            ]}
          />
          {node.origin === "runtime_external" && node.status === "unverified" && (
            <p className="mt-2 text-xs text-muted">Unverified: retrieved and accepted by the source gate, but nothing in it was validated as evidence for this answer.</p>
          )}
        </Section>
      )}
      {(typeof m.quote === "string" || typeof m.claim === "string" || m.score !== undefined) && (
        <Section title={node.type === "Evidence" ? "Evidence span" : "Recorded values"}>
          {typeof m.claim === "string" && <p className="mb-1 text-sm text-ink">{m.claim}</p>}
          {typeof m.quote === "string" && <q className="block text-[0.85rem] italic leading-snug text-ink">{m.quote}</q>}
          <div className="mt-1.5">
            <KV rows={[["Field", fmt(m.field)], ["Value", m.value !== undefined ? fmt(m.value) : null], ["Metric", fmt(m.metric)], ["Score", m.score !== undefined ? fmt(m.score) : null]]} />
          </div>
        </Section>
      )}
      {fields.length > 0 && (
        <Section title="Configuration (validated at runtime)">
          <ul className="space-y-1">
            {fields.map((f) => (
              <li key={`${f.field}-${fmt(f.value)}`} className="flex items-baseline justify-between gap-3 text-sm">
                <span className="text-muted">{f.label}</span>
                <span className="font-medium text-ink">{fmt(f.value)}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {canPromote && (
        <Section title="Corpus promotion">
          {promotion ? (
            <div className="space-y-2 text-sm">
              <p className="text-ink">
                <strong>
                  {promotion.state === "CANDIDATE" ? "Proposed for review." : promotion.state === "NOT_ELIGIBLE" ? "Not eligible." : `State: ${promotion.state.toLowerCase()}.`}
                </strong>{" "}
                {promotion.reason}
              </p>
              <ul className="space-y-0.5 text-xs">
                {promotion.checks.map((c) => (
                  <li key={c.check} className="flex items-center gap-1.5">
                    {c.ok === true ? (
                      <I.Check width={13} height={13} className="text-ok" />
                    ) : c.ok === false ? (
                      <I.Cross width={13} height={13} className="text-bad" />
                    ) : (
                      <I.Ring width={13} height={13} className="text-faint" />
                    )}
                    <span className="text-muted">{c.detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <>
              <p className="mb-2 text-xs text-muted">
                Proposing runs the provenance, identifier and evidence checks and queues the source for the next reviewed corpus release. Nothing is added to the corpus
                now.
              </p>
              <Button variant="secondary" className="px-2.5 py-1 text-xs" busy={promoting} onClick={promote}>
                Propose for the corpus
              </Button>
            </>
          )}
        </Section>
      )}
      <Section title={`Links (${edges.length})`}>
        <EdgeList node={node} edges={edges} nodeOf={nodeOf} onFocus={onFocus} />
      </Section>
    </>
  );
}

function EdgeList({ node, edges, nodeOf, onFocus }: { node: GNode; edges: GEdge[]; nodeOf: (id: string) => GNode | undefined; onFocus: (id: string) => void }) {
  if (!edges.length) return <p className="text-sm text-muted">No links on the map.</p>;
  return (
    <ul className="space-y-2">
      {edges.slice(0, 40).map((e) => {
        const other = nodeOf(e.source === node.id ? e.target : e.source);
        return (
          <li key={e.id} className="text-sm">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-medium text-muted">{e.source === node.id ? e.label : `← ${e.label}`}</span>
              <Badge tone={STRENGTH[e.strength]?.tone ?? "bg-surface-2 text-muted"}>{STRENGTH[e.strength]?.label ?? e.strength}</Badge>
              {e.status && e.status !== "validated" && <Badge tone={STATUS_TONE[e.status] ?? "bg-surface-2 text-muted"}>{e.status}</Badge>}
              {other && <NodeChip node={other} onFocus={onFocus} />}
            </div>
            <p className="mt-0.5 text-xs text-muted">{e.why}</p>
          </li>
        );
      })}
    </ul>
  );
}

// ---------------------------------------------------------------------------------------------------------------- Ask the Map
function AskPanel({ node, questions, runId, onFocus, fail }: { node: GNode; questions: Question[]; runId: string | null; onFocus: (id: string) => void; fail: (e: unknown) => void }) {
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const ask = async (q: Question) => {
    setLoading(q.key);
    try {
      setAnswer(await graphApi.ask(node.id, q.key, runId));
    } catch (e) {
      fail(e);
    } finally {
      setLoading(null);
    }
  };
  return (
    <Section title="Ask the map">
      <div className="flex flex-col gap-1.5">
        {questions.map((q) => (
          <button
            key={q.key}
            onClick={() => ask(q)}
            disabled={!!loading}
            aria-pressed={answer?.key === q.key}
            className={cx(
              "flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-left text-sm transition",
              answer?.key === q.key ? "border-accent bg-accent-soft text-ink" : "border-line text-ink hover:border-accent",
            )}
          >
            {q.question}
            {loading === q.key ? <Spinner className="h-3.5 w-3.5 text-faint" /> : <I.Chevron width={14} height={14} className="text-faint" />}
          </button>
        ))}
      </div>
      {answer && (
        <div className="rx-enter mt-3 space-y-2" aria-live="polite">
          <p className="text-sm font-medium text-ink">{answer.answer}</p>
          {answer.derivation && <p className="text-xs text-faint">From {answer.derivation}. Computed from the graph; no model was asked.</p>}
          {answer.items.length > 0 && (
            <ul className="space-y-1.5">
              {answer.items.map((i) => (
                <li key={i.id}>
                  <NodeChip
                    node={i}
                    onFocus={onFocus}
                    extra={
                      i.count !== undefined ? (
                        <span className="ml-1 font-mono text-faint">{i.count}</span>
                      ) : i.relation ? (
                        <span className="ml-1 text-faint">{COMPARISON[i.relation] ?? i.relation}</span>
                      ) : undefined
                    }
                  />
                  {i.why && <p className="mt-0.5 pl-1 text-xs text-muted">{i.why}</p>}
                </li>
              ))}
            </ul>
          )}
          {!!answer.evidence?.length && (
            <ul className="space-y-2">
              {answer.evidence.map((ev) =>
                "type" in ev ? (
                  <li key={ev.id} className="rounded-lg border border-line bg-bg p-2.5 text-sm">
                    <q className="block italic text-ink">{String(ev.meta?.quote ?? ev.label)}</q>
                    <p className="mt-1 text-xs text-muted">
                      {fmt(ev.provenance?.sid)} · {fmt(ev.provenance?.extraction)}
                    </p>
                  </li>
                ) : (
                  <EvidenceCard key={ev.id} ev={ev} />
                ),
              )}
            </ul>
          )}
          {!!answer.contradictions?.length && (
            <ul className="space-y-2">
              {answer.contradictions.map((c) => (
                <ContradictionCard key={c.id} c={c} />
              ))}
            </ul>
          )}
          {!!answer.rejected?.length && (
            <ul className="space-y-1 text-xs">
              {answer.rejected.map((r, i) => (
                <li key={i} className="text-muted">
                  <span className="font-medium text-ink">{r.field}</span> = {fmt(r.value)} · {r.reason}
                  {r.source ? ` · ${r.source}` : ""}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------------------------------------------- edges
function EdgePanel({ edge, run, nodeOf, onFocus, fail }: { edge: GEdge; run: RunGraph | null; nodeOf: (id: string) => GNode | undefined; onFocus: (id: string) => void; fail: (e: unknown) => void }) {
  const local = !!edge.origin && (edge.origin !== "canonical" || isRuntimeId(edge.source) || isRuntimeId(edge.target));
  const [detail, setDetail] = useState<EdgeDetail | null>(null);
  const [loading, setLoading] = useState(!local);
  useEffect(() => {
    if (local) return;
    let alive = true;
    graphApi
      .edge(edge)
      .then((d) => {
        if (alive) setDetail(d);
      })
      .catch((e) => {
        if (!edge.origin) fail(e); // a corpus link recorded by a run may not be re-derivable on its own: the run's record is shown
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [edge, local, fail]);
  const e = detail?.edge ?? edge;
  const runEvidence = local && run ? e.evidence_ids.map((id) => run.nodes.find((n) => n.id === id)).filter((n): n is GNode => !!n) : [];
  const ends = [nodeOf(e.source) ?? detail?.source, nodeOf(e.target) ?? detail?.target].filter((n): n is GNode => !!n);
  return (
    <div className="rx-enter">
      <header className="space-y-1.5 px-4 pb-3 pt-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-medium uppercase tracking-wide text-muted">{e.type.replaceAll("_", " ").toLowerCase()}</span>
          <Badge tone={STRENGTH[e.strength]?.tone ?? "bg-surface-2 text-muted"}>{STRENGTH[e.strength]?.label ?? e.strength}</Badge>
          {e.origin && <Badge tone="bg-surface-2 text-muted">{ORIGIN[e.origin] ?? e.origin}</Badge>}
          {e.status && e.status !== "validated" && <Badge tone={STATUS_TONE[e.status] ?? "bg-surface-2 text-muted"}>{e.status}</Badge>}
        </div>
        <h2 className="font-serif text-lg font-semibold leading-snug text-ink">{e.label}</h2>
        <div className="flex flex-wrap items-center gap-1.5">
          {ends.map((n, i) => (
            <span key={n.id} className="inline-flex items-center gap-1.5">
              {i === 1 && <I.Send width={13} height={13} className="text-faint" />}
              <NodeChip node={n} onFocus={onFocus} />
            </span>
          ))}
        </div>
      </header>
      <Section title="Why this link exists">
        <p className="text-sm text-ink">{e.why}</p>
        <p className="mt-1.5 text-xs text-muted">Derivation: {e.derivation}</p>
      </Section>
      {loading && (
        <div className="flex justify-center py-6 text-faint">
          <Spinner />
        </div>
      )}
      {detail?.comparison && (
        <Section title="Field-by-field comparison">
          <DiffView diff={detail.comparison} onFocus={onFocus} />
        </Section>
      )}
      {detail?.contradiction && (
        <Section title="Recorded contradiction">
          <ul>
            <ContradictionCard c={detail.contradiction} />
          </ul>
        </Section>
      )}
      {!!detail?.evidence?.length && (
        <Section title={`Evidence (${detail.evidence.length})`}>
          <EvidenceList items={detail.evidence} />
        </Section>
      )}
      {runEvidence.length > 0 && (
        <Section title={`Evidence (${runEvidence.length})`}>
          <ul className="space-y-2">
            {runEvidence.map((n) => (
              <li key={n.id} className="rounded-lg border border-line bg-bg p-2.5 text-sm">
                <q className="block italic text-ink">{String(n.meta?.quote ?? n.label)}</q>
                <p className="mt-1 text-xs text-muted">
                  {ORIGIN[n.origin ?? "canonical"]} · {fmt(n.provenance?.sid)} · {fmt(n.provenance?.extraction)}
                </p>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {!loading && !detail?.evidence?.length && !runEvidence.length && !detail?.comparison && (
        <Section title="Evidence">
          <p className="text-sm text-muted">
            {e.evidence_ids.length
              ? `${plural(e.evidence_ids.length, "evidence id")} recorded: ${e.evidence_ids.slice(0, 4).join(", ")}`
              : "This link is recorded metadata (for example a citation or the paper a record came from), not a measured claim."}
          </p>
        </Section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------- compare
function CompareTray({ items, runId, onRemove, onFocus, fail }: { items: GNode[]; runId: string | null; onRemove: (id: string) => void; onFocus: (id: string) => void; fail: (e: unknown) => void }) {
  const [diff, setDiff] = useState<Diff | null>(null);
  const [loading, setLoading] = useState(false);
  const [a, b] = [items[0]?.id, items[1]?.id];
  useEffect(() => {
    setDiff(null);
    if (!a || !b) return;
    let alive = true;
    setLoading(true);
    graphApi
      .diff(a, b, runId)
      .then((d) => {
        if (alive) setDiff(d);
      })
      .catch(fail)
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [a, b, runId, fail]);
  return (
    <section className="border-b border-line bg-surface-2 px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-faint">Compare experiments</h3>
        <span className="text-xs text-muted">{items.length < 2 ? "Add one more experiment" : "A vs B"}</span>
      </div>
      <ul className="flex flex-wrap gap-1.5">
        {items.map((n, i) => (
          <li key={n.id} className="inline-flex max-w-full items-center gap-1">
            <span className="font-mono text-xs text-faint">{i ? "B" : "A"}</span>
            <NodeChip node={n} onFocus={onFocus} />
            <button onClick={() => onRemove(n.id)} aria-label={`Remove ${n.label} from the comparison`} className="text-faint hover:text-ink">
              <I.Cross width={13} height={13} />
            </button>
          </li>
        ))}
      </ul>
      {loading && (
        <div className="flex justify-center py-4 text-faint">
          <Spinner />
        </div>
      )}
      {diff && (
        <div className="mt-3">
          <DiffView diff={diff} onFocus={onFocus} />
        </div>
      )}
    </section>
  );
}

function DiffView({ diff, onFocus }: { diff: Diff; onFocus: (id: string) => void }) {
  const groups: [string, string[]][] = [
    ["Different", ["MATERIALLY_DIFFERENT", "DIFFERENT"]],
    ["Conflicting", ["CONFLICTING"]],
    ["Same", ["IDENTICAL", "EQUIVALENT"]],
    ["Unknown on a side", ["UNKNOWN"]],
  ];
  return (
    <div className="space-y-3 text-sm">
      <div>
        <Badge tone={diff.label === "ALIGNED_EXPERIMENT" ? "bg-accent-soft text-accent" : diff.label === "CONFLICTING_EVIDENCE" ? "bg-bad-soft text-bad" : "bg-surface text-ink"}>
          {COMPARISON[diff.label] ?? diff.label}
        </Badge>
        <p className="mt-1 text-ink">{diff.rationale.charAt(0).toUpperCase() + diff.rationale.slice(1)}.</p>
        {diff.uncertainty && <p className="mt-0.5 text-xs text-muted">{diff.uncertainty}</p>}
      </div>
      {groups.map(([title, statuses]) => {
        const rows = diff.fields.filter((f) => statuses.includes(f.status));
        if (!rows.length) return null;
        return (
          <div key={title}>
            <p className="mb-1 text-xs font-medium text-muted">
              {title} ({rows.length})
            </p>
            <table className="w-full table-fixed text-xs">
              <thead className="sr-only">
                <tr>
                  <th>Field</th>
                  <th>Experiment A</th>
                  <th>Experiment B</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.field} className="border-t border-line align-top">
                    <td className="w-[38%] py-1 pr-2 text-muted" title={r.rationale}>
                      {r.field.replace(/^(evaluation|training|model|method|research)\./, "")}
                    </td>
                    <td className={cx("wrap-break-word py-1 pr-2", FIELD_TONE[r.status])} title={`${r.evidence_a.length} evidence span(s)`}>
                      {fmt(r.value_a)}
                    </td>
                    <td className={cx("wrap-break-word py-1", FIELD_TONE[r.status])} title={`${r.evidence_b.length} evidence span(s)`}>
                      {fmt(r.value_b)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
      {diff.contradictions.length > 0 && (
        <ul className="space-y-2">
          {diff.contradictions.map((c) => (
            <ContradictionCard key={c.id} c={c} />
          ))}
        </ul>
      )}
      <p className="text-xs text-faint">
        Compared by {diff.labeler_version}: unknown is never treated as a match, and a difference in setup is not evidence of why results differ.
      </p>
      <div className="flex flex-wrap gap-1.5">
        <NodeChip node={diff.a} onFocus={onFocus} />
        <NodeChip node={diff.b} onFocus={onFocus} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------- overview
function LandscapeView({ l, onFocus }: { l: Landscape; onFocus: (id: string) => void }) {
  const years = Object.entries(l.years);
  const maxYear = Math.max(...years.map(([, n]) => n), 1);
  return (
    <div className="space-y-3">
      <p className="text-sm text-ink">
        {plural(l.experiments, "experiment record")} · {plural(l.papers, "paper")} · {plural(l.results, "result")}
        {l.contradictions ? <span className="text-bad"> · {plural(l.contradictions, "recorded contradiction")}</span> : null}
      </p>
      {years.length > 1 && (
        <div>
          <div aria-hidden="true" className="flex h-10 items-end gap-px">
            {years.map(([y, n]) => (
              <div key={y} title={`${y}: ${n} paper(s)`} className="flex-1 rounded-t-sm bg-accent/40" style={{ height: `${Math.max((n / maxYear) * 100, 6)}%` }} />
            ))}
          </div>
          <p className="mt-0.5 flex justify-between text-[0.65rem] text-faint">
            <span>{years[0][0]}</span>
            <span>papers per year</span>
            <span>{years[years.length - 1][0]}</span>
          </p>
        </div>
      )}
      <Groups groups={l.groups} onFocus={onFocus} />
      {l.model_families.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-medium text-muted">Model families (name-derived: size variants grouped)</p>
          <p className="text-sm text-ink">{l.model_families.map((f) => `${f.family} (${f.count})`).join(" · ")}</p>
        </div>
      )}
    </div>
  );
}

type OverviewProps = {
  landscape: Landscape | null;
  run: RunGraph | null;
  mapNodes: GNode[];
  onFocus: (id: string) => void;
  onOpenRun: (id: string) => void;
  fail: (e: unknown) => void;
};

function Overview({ landscape, run, mapNodes, onFocus, onOpenRun, fail }: OverviewProps) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [found, setFound] = useState<{ items: Discovery[]; counts: Record<string, number>; runs: number } | null>(null);
  useEffect(() => {
    graphApi.runs().then((r) => setRuns(r.runs)).catch(fail);
    graphApi.discoveries(1).then(setFound).catch(fail);
  }, [fail]);
  const byType = useMemo(() => {
    const g: Record<string, GNode[]> = {};
    for (const n of mapNodes) (g[n.type] ??= []).push(n);
    return Object.entries(g).sort(([x], [y]) => x.localeCompare(y));
  }, [mapNodes]);
  return (
    <div className="rx-enter">
      <header className="px-4 pb-2 pt-4">
        <h2 className="font-serif text-lg font-semibold text-ink">{run ? "Research run" : "Corpus at a glance"}</h2>
        <p className="mt-0.5 text-sm text-muted">
          {run
            ? "Select a node or a link to see what it is, where it comes from and why it is connected."
            : "Every count is a count of evidence-linked experiment records. Search or pick an item to start a map."}
        </p>
      </header>
      {!run && <Section title="Corpus">{landscape ? <LandscapeView l={landscape} onFocus={onFocus} /> : <Spinner className="text-faint" />}</Section>}
      <Section title="Found outside the corpus today">
        {found === null ? (
          <Spinner className="text-faint" />
        ) : found.items.length === 0 ? (
          <p className="text-sm text-muted">Your research runs today found nothing outside the corpus.</p>
        ) : (
          <>
            <p className="mb-2 text-sm text-ink">
              {plural(found.items.length, "item")} across {plural(found.runs, "run")}:{" "}
              {Object.entries(found.counts)
                .map(([k, v]) => `${v} ${k}`)
                .join(", ")}
            </p>
            <ul className="space-y-1.5">
              {found.items.slice(0, 8).map((d) => (
                <li key={d.id} className="flex items-center gap-2 text-sm">
                  <Badge tone={STATUS_TONE[d.status ?? "unverified"]}>{d.status}</Badge>
                  <button onClick={() => d.runs[0] && onOpenRun(d.runs[0].research_run_id)} className="min-w-0 flex-1 truncate text-left text-ink hover:underline" title={d.label}>
                    {d.label}
                  </button>
                  <span className="shrink-0 text-xs text-faint">{d.provider ?? (d.origin === "private_document" ? "your document" : "")}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </Section>
      <Section title="Your research runs">
        {runs === null ? (
          <Spinner className="text-faint" />
        ) : runs.length === 0 ? (
          <p className="text-sm text-muted">
            No research run has a graph yet.{" "}
            <Link href="/" className="font-medium text-accent hover:underline">
              Ask a question
            </Link>{" "}
            and open its map from the answer.
          </p>
        ) : (
          <ul className="space-y-1">
            {runs.slice(0, 10).map((r) => {
              const ext = Number((r.stats?.by_origin as Rec | undefined)?.runtime_external ?? 0);
              return (
                <li key={r.research_run_id}>
                  <button
                    onClick={() => onOpenRun(r.research_run_id)}
                    className={cx("w-full rounded-md px-1.5 py-1 text-left hover:bg-surface-2", run?.research_run_id === r.research_run_id && "bg-surface-2")}
                  >
                    <span className="block truncate text-sm text-ink">{r.query}</span>
                    <span className="text-xs text-faint">
                      {r.status.toLowerCase()} · {fmt(r.stats?.nodes)} nodes{ext ? ` · ${ext} found outside the corpus` : ""}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </Section>
      {mapNodes.length > 0 && (
        <Section title={`On the map (${mapNodes.length})`}>
          <details>
            <summary className="cursor-pointer text-sm text-accent">List every node (keyboard navigation)</summary>
            <div className="mt-2 space-y-2">
              {byType.map(([type, ns]) => (
                <div key={type}>
                  <p className="mb-1 text-xs font-medium text-muted">{type}</p>
                  <div className="flex flex-wrap gap-1">
                    {ns.slice(0, 60).map((n) => (
                      <NodeChip key={n.id} node={n} onFocus={onFocus} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </details>
        </Section>
      )}
    </div>
  );
}

function EmptyState({ landscape, onPick }: { landscape: Landscape | null; onPick: (id: string) => void }) {
  const picks = landscape ? ["Topic", "Dataset", "Model"].flatMap((t) => (landscape.groups[t] ?? []).slice(0, 2)) : [];
  return (
    <div className="pointer-events-none absolute inset-0 grid place-items-center p-6">
      <div className="pointer-events-auto max-w-sm rounded-xl border border-line bg-surface p-5 text-center shadow-sm">
        <p className="font-serif text-base font-semibold text-ink">Start from what you are researching</p>
        <p className="mt-1 text-sm text-muted">Search for a topic, model, dataset or paper, or open one of your research runs from the panel.</p>
        {picks.length > 0 && (
          <div className="mt-3 flex flex-wrap justify-center gap-1.5">
            {picks.map((p) => (
              <button key={p.id} onClick={() => onPick(p.id)} className="inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-ink hover:border-accent">
                <TypeDot type={p.type} /> {p.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Legend({ onClose }: { onClose: () => void }) {
  const types = ["Run", "Topic", "Paper", "Experiment", "Model", "Method", "Dataset", "Metric", "Result", "Configuration", "Evidence"];
  return (
    <div className="absolute bottom-3 left-3 max-w-[calc(100%-1.5rem)] rounded-lg border border-line bg-surface/95 p-2.5 text-[0.7rem] text-muted shadow-sm">
      <div className="mb-1.5 flex items-center justify-between gap-4">
        <span className="font-semibold uppercase tracking-wider text-faint">Legend</span>
        <button onClick={onClose} aria-label="Hide legend" className="hover:text-ink">
          <I.Cross width={12} height={12} />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
        {types.map((t) => (
          <span key={t} className="inline-flex items-center gap-1.5">
            <TypeDot type={t} /> {t}
          </span>
        ))}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-muted" /> corpus (filled)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border-2 border-dashed border-muted" /> found at runtime
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border-2 border-dotted border-muted" /> your document
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border-2 border-bad" /> conflict recorded
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-faint" /> recorded
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-accent" /> strong
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-dashed border-faint" /> related
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-dotted border-faint" /> possible
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-bad" /> contradictory
        </span>
        <span className="inline-flex items-center gap-1.5 opacity-60">
          <span className="w-5 border-t-2 border-faint" /> unverified
        </span>
      </div>
    </div>
  );
}
