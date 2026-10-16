import { api } from "@/lib/api";

// Knowledge Map client (docs/API_SPEC.md §50). Canonical nodes come from the shared corpus graph; runtime nodes come from
// one of the user's research runs and carry origin, status and provenance. Every relationship says why it exists.

export type Origin = "canonical" | "runtime_external" | "private_document" | "user_input" | "derived";
export type NodeStatus = "validated" | "unverified" | "related" | "promoted" | "rejected";
export type Strength = "EXACT" | "STRONG" | "RELATED" | "POSSIBLE" | "CONTRADICTORY";

export type GNode = {
  id: string;
  type: string;
  label: string;
  meta?: Record<string, unknown>;
  flags?: string[];
  origin?: Origin;
  status?: NodeStatus;
  provenance?: Record<string, unknown>;
  canonical_meta?: Record<string, unknown>;
  // Ask / landscape items
  count?: number;
  relation?: string;
  why?: string;
};
export type GEdge = {
  id: string;
  type: string;
  source: string;
  target: string;
  strength: Strength;
  label: string;
  why: string;
  derivation: string;
  evidence_ids: string[];
  provenance?: Record<string, unknown> | string | null;
  origin?: Origin;
  status?: NodeStatus;
};
export type Evidence = {
  id: string;
  field: string;
  status?: string;
  location_kind?: string | null;
  paper_id?: string | null;
  experiment_id?: string | null;
  text?: string | null;
  text_withheld?: boolean;
  extracted_value?: unknown;
  normalized_value?: unknown;
  source_location?: string | null;
  section?: string | null;
  table?: string | null;
  cell?: string | null;
  source_record_id?: string | null;
  source_version?: string | null;
  retrieval_timestamp?: string | null;
  extraction_model?: string | null;
};
export type Contradiction = {
  id: string;
  subject: string;
  field: string;
  claim?: string;
  explanation?: string;
  value_a?: unknown;
  value_b?: unknown;
  source_a?: string;
  source_b?: string;
  evidence_a?: string[];
  evidence_b?: string[];
  conflict_type?: string;
};
export type Neighborhood = {
  center: GNode;
  nodes: GNode[];
  edges: GEdge[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
  counts: Record<string, number>;
  edge_counts: Record<string, number>;
};
export type DiffRow = { field: string; status: string; value_a: unknown; value_b: unknown; evidence_a: string[]; evidence_b: string[]; rationale: string };
export type Diff = {
  a: GNode;
  b: GNode;
  label: string;
  rationale: string;
  uncertainty?: string | null;
  missing: string[];
  fields: DiffRow[];
  contradictions: Contradiction[];
  labeler_version: string;
};
export type Landscape = {
  root: GNode | null;
  experiments: number;
  papers: number;
  results: number;
  years: Record<string, number>;
  groups: Record<string, GNode[]>;
  model_families: { family: string; count: number }[];
  contradictions: number;
};
export type Question = { key: string; question: string };
export type NodeDetail = { node: GNode; questions: Question[] } & Record<string, unknown>;
export type EdgeDetail = {
  edge: GEdge;
  source: GNode | null;
  target: GNode | null;
  evidence?: Evidence[];
  comparison?: Diff;
  contradiction?: Contradiction | null;
  study_link?: Record<string, unknown> | null;
};
export type AskResult = {
  key: string;
  question: string;
  answer: string;
  items: GNode[];
  evidence?: (Evidence | GNode)[];
  contradictions?: Contradiction[];
  rejected?: { field: string; value: unknown; reason: string; source?: string | null }[];
  derivation?: string;
};
export type RunGraph = {
  version: string;
  research_run_id: string;
  nodes: GNode[];
  edges: GEdge[];
  rejected: { field: string; value: unknown; reason: string; source?: string | null; origin?: string }[];
  stats: Record<string, unknown>;
};
export type RunSummary = { research_run_id: string; query: string; status: string; started_at: string; stats: Record<string, unknown> | null };
export type Discovery = GNode & { provider?: string | null; url?: string | null; runs: { research_run_id: string; query: string; started_at: string }[] };
export type Promotion = { state: string; eligible: boolean; reason: string; checks: { check: string; ok: boolean | null; detail: string }[]; licence?: unknown; note?: string };

export const NODE_TYPES = ["Topic", "Paper", "Experiment", "Configuration", "Model", "Method", "Dataset", "Metric", "Result", "Evidence"] as const;
export const EDGE_TYPES = ["REPORTS", "CONTAINS", "USES_MODEL", "USES_METHOD", "USES_DATASET", "HAS_CONFIGURATION", "PRODUCES_RESULT", "MEASURES",
  "SUPPORTED_BY", "CITES", "RELATED_TO", "REPLICATES", "EXTENDS", "DIFFERS_FROM", "CONTRADICTS"] as const;
/** What a run did (not research relationships): drawn thin and dotted. */
export const PROVENANCE_EDGES = new Set(["ASKED_ABOUT", "RETRIEVED", "QUOTED_FROM"]);

export const isRuntimeId = (id: string) => id.startsWith("rt:") || id.startsWith("doc:") || id.startsWith("run:");

/** Questions for a node that exists only in a research run (mirrors services/graph/runtime.py questions_for). */
export function runtimeQuestions(n: GNode): Question[] {
  if (n.type === "Run")
    return [
      { key: "discoveries", question: "What did this research run discover?" },
      { key: "evidence", question: "Which evidence supports this answer?" },
      { key: "rejected", question: "Which extracted facts were rejected, and why?" },
    ];
  const evidence = { key: "evidence", question: "What evidence supports this?" };
  return n.type === "Experiment" ? [{ key: "closest", question: "Which corpus experiments are closest to this one?" }, evidence] : [evidence];
}

const qs = (p: Record<string, string | number | null | undefined>) =>
  new URLSearchParams(Object.entries(p).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => [k, String(v)])).toString();

export const graphApi = {
  stats: () => api<Record<string, unknown>>("/api/graph/stats"),
  search: (q: string, types?: string[]) => api<{ results: GNode[] }>(`/api/graph/search?${qs({ q, types: types?.join(","), limit: 12 })}`),
  landscape: (root?: string | null) => api<Landscape>(`/api/graph/landscape?${qs({ root })}`),
  neighborhood: (node: string, o: { nodeTypes?: string[]; edgeTypes?: string[]; offset?: number; limit?: number } = {}) =>
    api<Neighborhood>(`/api/graph/neighborhood?${qs({ node, node_types: o.nodeTypes?.join(","), edge_types: o.edgeTypes?.join(","), offset: o.offset, limit: o.limit })}`),
  node: (id: string) => api<NodeDetail>(`/api/graph/node?${qs({ id })}`),
  edge: (e: Pick<GEdge, "type" | "source" | "target">) => api<EdgeDetail>(`/api/graph/edge?${qs({ type: e.type, source: e.source, target: e.target })}`),
  diff: (a: string, b: string, run?: string | null) => api<Diff>(`/api/graph/diff?${qs({ a, b, run })}`),
  ask: (node: string, q: string, run?: string | null) => api<AskResult>(`/api/graph/ask?${qs({ node, q, run })}`),
  run: (id: string) => api<RunGraph>(`/api/graph/run/${encodeURIComponent(id)}`),
  runs: () => api<{ runs: RunSummary[] }>("/api/graph/runs"),
  discoveries: (days = 1) => api<{ days: number; runs: number; counts: Record<string, number>; items: Discovery[] }>(`/api/graph/discoveries?${qs({ days })}`),
  promote: (research_run_id: string, node_id: string) =>
    api<Promotion>("/api/graph/promotions", { method: "POST", body: JSON.stringify({ research_run_id, node_id }) }),
};
