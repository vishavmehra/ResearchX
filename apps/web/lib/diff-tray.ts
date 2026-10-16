"use client";

import { useSyncExternalStore } from "react";

// Papers collected for a paper diff from anywhere in the app (answer sources, Knowledge Map), kept in this browser only.
// A source passage is only a shortcut to its paper: diffs compare different papers, never passages of one paper.
export type TrayPaper = { id: string; title: string };
export const TRAY_MAX = 4; // the paper diff compares two to four papers

const KEY = "rx-diff-tray";
const EMPTY: TrayPaper[] = [];
const listeners = new Set<() => void>();
let cache: TrayPaper[] | null = null;

function read(): TrayPaper[] {
  if (cache) return cache;
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    cache = Array.isArray(parsed) ? parsed.filter((p) => typeof p?.id === "string" && p.id.startsWith("paper:") && !p.id.includes("#")).slice(0, TRAY_MAX) : EMPTY;
  } catch {
    cache = EMPTY;
  }
  return cache!;
}

function write(next: TrayPaper[]) {
  cache = next.slice(0, TRAY_MAX);
  try {
    localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    // storage blocked (private window): the tray still works for this page
  }
  listeners.forEach((l) => l());
}

export function toggleTray(p: TrayPaper) {
  const cur = read();
  if (cur.some((x) => x.id === p.id)) write(cur.filter((x) => x.id !== p.id));
  else if (cur.length < TRAY_MAX) write([...cur, p]);
}

export function clearTray() {
  write([]);
}

export function useTray(): TrayPaper[] {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      const onStorage = (e: StorageEvent) => {
        if (e.key === KEY) {
          cache = null;
          cb();
        }
      };
      window.addEventListener("storage", onStorage);
      return () => {
        listeners.delete(cb);
        window.removeEventListener("storage", onStorage);
      };
    },
    read,
    () => EMPTY,
  );
}

export const diffHref = (papers: TrayPaper[]) => `/analyze?tab=papers&papers=${encodeURIComponent(papers.map((p) => p.id).join(","))}`;
