import { supabase } from "@/lib/supabase/client";

// Typed client for the ResearchX backend. Every call carries the user's Supabase access token; the backend derives the
// user from that token alone (never from request bodies) and enforces ownership with PostgreSQL RLS.

export type StageStatus = "running" | "completed" | "skipped" | "not_required" | "failed";
export type Stage = { stage: string; status: StageStatus; latency_ms?: number; detail?: Record<string, unknown>; error?: string | null };
export type Match = "EXACT" | "STRONG" | "RELATED" | "WEAK" | "DATASET"; // DATASET: another paper on the asked dataset (comparison only)
export type Span = { text: string; start?: number | null; end?: number | null; field: string; value?: string; cell?: string | null };
export type Source = {
  sid: string;
  kind: "internal" | "external" | "private" | "record";  title?: string | null;
  url?: string | null;
  text: string;
  section?: string | null;
  locator?: string | null;
  unit_type?: string | null;
  year?: number | null;
  grade?: number | null;
  provider?: string;
  reliability?: string;
  paper_id?: string | null;
  corpus_paper_id?: string | null; // external-kind source that is a corpus paper stored abstract-only, its full text read at question time
  doi?: string | null;
  arxiv_id?: string | null;
  retrieved_at?: string | null;
  also_found_by?: { provider: string; url?: string }[];
  evidence_ids?: string[];
  experiment_ids?: string[];
  // field-level grounding (parameter and score questions)
  role?: "SUPPORTS" | "RELATED";
  match?: Match;
  supports?: { field: string; label: string; value: string }[];
  spans?: Span[];
  verification?: string | null;
};
export type Claim = { text: string; sources: string[]; quote: string; field?: string; origin?: string; verified?: boolean };
export type EvidenceStatus =
  | "SUFFICIENT"
  | "PARTIAL"
  | "CONFLICTING_EVIDENCE"
  | "AMBIGUOUS_EXPERIMENT_MATCH"
  | "INSUFFICIENT_EVIDENCE"
  | "NO_RELEVANT_SOURCE_FOUND";
export type FieldFact = { field: string; label: string; status: "SUPPORTED" | "UNKNOWN" | "CONFLICTING" | "AMBIGUOUS"; values: string[]; sources: string[] };
export type Contradiction = { id: string; subject_id: string; field: string; explanation?: string; value_a?: unknown; value_b?: unknown };
export type RunStatus = "ANSWERED" | "PARTIAL" | "CONFLICTING" | "ABSTAINED" | "FAILED" | "RUNNING";
export type Result = {
  research_run_id: string;
  conversation_id: string;
  status: RunStatus;
  mode: "pipeline" | "fallback";
  fallback_reason?: string | null;
  intent?: string;
  answer: string;
  claims: Claim[];
  sources: Source[];
  cited_sources: string[];
  contradictions: Contradiction[];
  cached: boolean;
  origin_research_run_id?: string | null;
  total_ms?: number;
  stages?: Stage[];
  usage?: Record<string, number>;
  routing?: Record<string, unknown>; // route -> model, plus policy, mode and per-call decisions (router.py)
  removed?: { reason: string; text?: string }[];
  evidence_status?: EvidenceStatus;
  fields?: FieldFact[];
  request?: { model: string | null; dataset: string | null; fields: string[] };
  rejected_facts?: number;
  // candidate values the run found but did not use, each with why and where
  rejected_candidates?: {
    field: string;
    label: string;
    value: string;
    reason: string;
    context?: string | null;
    span?: string;
    origin?: string;
    source: { title?: string | null; paper_id?: string | null; section?: string | null; year?: number | null; url?: string | null; provider?: string | null };
  }[];
  // external results the run found but did not use (not read, not cited, another configuration, rejected by the gate…)
  unused_sources?: { title: string | null; url: string | null; provider: string | null; doi?: string | null; arxiv_id?: string | null; also_found_by?: string[]; reason: string }[];
  // the run's runtime research graph (summary; the full graph is loaded by the Knowledge Map)
  graph?: { research_run_id: string; stats?: Record<string, unknown> } | null;
};
export type Conversation = { id: string; title: string | null; created_at: string; updated_at: string };
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  input_modality: "text" | "voice";
  created_at: string;
  research_run_id: string | null;
  result: Result | null;
};
export type StreamEvent = { type: string; stage?: string; status?: string; research_run_id?: string; latency_ms?: number; detail?: Record<string, unknown>; error?: string | null; mode?: string; total_ms?: number };

export class ApiError extends Error {
  constructor(public status: number, public code: string | undefined, message: string) {
    super(message);
  }
}

async function accessToken(): Promise<string> {
  const { data } = await supabase().auth.getSession();
  if (!data.session) throw new ApiError(401, "missing_token", "Sign in to continue.");
  return data.session.access_token;
}

async function fail(res: Response): Promise<never> {
  const body = await res.json().catch(() => ({}));
  const d = body?.detail;
  const message = typeof d === "string" ? d : Array.isArray(d) ? "The request was not accepted." : d?.message ?? `Request failed (${res.status}).`;
  throw new ApiError(res.status, d && typeof d === "object" && !Array.isArray(d) ? d.code : undefined, message);
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${await accessToken()}`);
  if (typeof init.body === "string" && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) await fail(res);
  return (await res.json()) as T;
}

/** Runs research and streams the run's real stage events (server-sent events) until the final result arrives. */
export async function research(
  body: { query: string; conversation_id?: string; input_modality: "text" | "voice" },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<Result> {
  const res = await fetch("/api/research", {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${await accessToken()}` },
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!res.ok || !res.body) await fail(res);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = /^event: (.+)$/m.exec(raw)?.[1];
      const data = /^data: (.+)$/m.exec(raw)?.[1];
      if (!event || !data) continue;
      const payload = JSON.parse(data);
      if (event === "result") return payload as Result;
      if (event === "error") throw new ApiError(500, "run_failed", payload.error ?? "The research run failed.");
      onEvent({ ...payload, type: event });
    }
  }
  throw new ApiError(500, "stream_ended", "The research run ended without a result.");
}

/** Where to send the user when the backend refuses the session. */
export function authRedirect(e: unknown): string | null {
  if (!(e instanceof ApiError)) return null;
  if (e.status === 401) return "/login";
  if (e.status === 403 && e.code === "mfa_required") return "/mfa";
  return null;
}
