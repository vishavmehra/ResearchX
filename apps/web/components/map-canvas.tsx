"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type cytoscape from "cytoscape";
import { PROVENANCE_EDGES, type GEdge, type GNode } from "@/lib/graph";
import * as I from "@/components/icons";

// Canvas for the Knowledge Map (Cytoscape.js, loaded only on this page). Visual language: a node's shape and hue say what
// it is; a filled node is corpus knowledge, a hollow one was found at runtime (dashed: external, dotted: your private
// document, double: your question); a red ring marks a recorded conflict. Edge line style says how strong a link is.

export const MAX_NODES = 300;

export type CanvasHandle = {
  /** Adds nodes and edges (deduplicated), new nodes placed around `around`; returns how many nodes the cap left out. */
  add(nodes: GNode[], edges: GEdge[], around?: string): number;
  remove(ids: string[]): void;
  clear(): void;
  layout(root?: string): void;
  focus(id: string): void;
  select(id: string | null): void;
  fit(): void;
  setVisible(node: (n: GNode) => boolean, edge: (e: GEdge) => boolean): void;
  has(id: string): boolean;
  size(): number;
  neighbours(id: string): string[];
};

type Props = {
  onNode: (n: GNode) => void;
  onEdge: (e: GEdge) => void;
  onExpand: (n: GNode) => void;
  onBackground: () => void;
};

const SHAPE: Record<string, cytoscape.Css.NodeShape> = {
  Run: "round-octagon", Topic: "round-hexagon", Paper: "round-rectangle", Study: "round-rectangle", Source: "round-rectangle",
  Document: "round-tag", Experiment: "ellipse", Configuration: "round-tag", Model: "round-diamond", Method: "round-diamond",
  Dataset: "barrel", Metric: "round-triangle", Result: "ellipse", Evidence: "round-rectangle",
};
const SIZE: Record<string, number> = {
  Run: 34, Topic: 26, Paper: 26, Study: 26, Source: 22, Document: 24, Experiment: 22, Model: 24, Method: 22, Dataset: 24, Metric: 20,
  Result: 15, Configuration: 15, Evidence: 13,
};
export const TYPE_RANK: Record<string, number> = {
  Run: 0, Topic: 1, Paper: 2, Study: 2, Source: 2, Document: 2, Experiment: 3, Model: 4, Method: 4, Dataset: 4, Metric: 5,
  Configuration: 5, Result: 5, Evidence: 6,
};
const HUE_LIGHT: Record<string, string> = {
  Run: "#16181d", Topic: "#5f6b7a", Paper: "#3d4451", Study: "#4b5563", Source: "#6b7280", Document: "#7a5230", Experiment: "#0e6b66",
  Configuration: "#8b95a3", Model: "#4c5fae", Method: "#7656a0", Dataset: "#9a6a12", Metric: "#2f6f9f", Result: "#5d7f2e", Evidence: "#8f8676",
};
const HUE_DARK: Record<string, string> = {
  Run: "#e7e8eb", Topic: "#94a0b0", Paper: "#aab2c0", Study: "#9aa3b2", Source: "#9aa0aa", Document: "#c69a6b", Experiment: "#58b8ad",
  Configuration: "#7d8796", Model: "#8c9cf0", Method: "#b49ae0", Dataset: "#e0b45a", Metric: "#7fb7e0", Result: "#9cc46a", Evidence: "#b5ab98",
};

export function hue(type: string): string {
  const dark = typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return (dark ? HUE_DARK : HUE_LIGHT)[type] ?? "#6b7280";
}

function stylesheet(): cytoscape.StylesheetJson {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const H = dark ? HUE_DARK : HUE_LIGHT;
  const font = getComputedStyle(document.body).fontFamily;
  const color = (e: cytoscape.NodeSingular) => H[e.data("type")] ?? v("--muted");
  return [
    {
      selector: "node",
      style: {
        "background-color": color, shape: (e: cytoscape.NodeSingular) => SHAPE[e.data("type")] ?? "ellipse",
        width: (e: cytoscape.NodeSingular) => SIZE[e.data("type")] ?? 18, height: (e: cytoscape.NodeSingular) => SIZE[e.data("type")] ?? 18,
        label: "data(label)", "font-family": font, "font-size": 10, color: v("--muted"), "text-valign": "bottom", "text-margin-y": 5,
        "text-wrap": "ellipsis", "text-max-width": "130px", "min-zoomed-font-size": 7, "border-width": 1.5, "border-color": v("--surface"),
        "overlay-opacity": 0, "text-background-color": v("--bg"), "text-background-opacity": 0.75, "text-background-padding": "1px",
      },
    },
    { selector: 'node[origin = "runtime_external"]', style: { "background-color": v("--surface"), "border-color": color, "border-width": 2.5, "border-style": "dashed" } },
    { selector: 'node[origin = "private_document"]', style: { "background-color": v("--surface"), "border-color": color, "border-width": 2.5, "border-style": "dotted" } },
    { selector: 'node[origin = "user_input"]', style: { "background-color": v("--surface"), "border-color": color, "border-width": 4, "border-style": "double" } },
    { selector: 'node[origin = "derived"]', style: { "background-color": v("--surface-2"), "border-color": color, "border-width": 1.5 } },
    { selector: 'node[status = "unverified"]', style: { opacity: 0.6 } },
    { selector: 'node[status = "promoted"]', style: { "border-color": v("--accent"), "border-style": "solid", "border-width": 3 } },
    { selector: "node[?conflict]", style: { "border-color": v("--bad"), "border-width": 3, "border-style": "solid" } },
    {
      selector: "edge",
      style: {
        width: 1.4, "line-color": v("--faint"), "target-arrow-color": v("--faint"), "target-arrow-shape": "triangle", "arrow-scale": 0.7,
        "curve-style": "bezier", opacity: 0.8, "font-family": font, "font-size": 9, color: v("--muted"), "text-rotation": "autorotate",
        "text-background-color": v("--bg"), "text-background-opacity": 0.9, "text-background-padding": "2px", "overlay-opacity": 0,
      },
    },
    { selector: 'edge[strength = "STRONG"]', style: { width: 2.4, "line-color": v("--accent"), "target-arrow-color": v("--accent") } },
    { selector: 'edge[strength = "RELATED"]', style: { "line-style": "dashed" } },
    { selector: 'edge[strength = "POSSIBLE"]', style: { "line-style": "dotted", width: 1.2, opacity: 0.7 } },
    { selector: 'edge[strength = "CONTRADICTORY"]', style: { width: 2.4, "line-color": v("--bad"), "target-arrow-color": v("--bad") } },
    { selector: "edge[?provenance]", style: { width: 1, "line-style": "dotted", opacity: 0.55, "target-arrow-shape": "none" } },
    { selector: 'edge[status = "unverified"]', style: { opacity: 0.4 } },
    { selector: "node:selected", style: { "border-color": v("--accent"), "border-width": 3.5, "border-style": "solid", color: v("--ink"), "font-weight": 600 } },
    { selector: "edge:selected, edge.hover", style: { label: "data(label)", width: 3, opacity: 1 } },
    { selector: ".faded", style: { opacity: 0.12 } },
    { selector: ".hidden", style: { display: "none" } },
  ];
}

const nodeData = (n: GNode) => ({
  id: n.id, label: n.origin === "private_document" ? `Private · ${n.label}` : n.label, type: n.type, origin: n.origin ?? "canonical",
  status: n.status ?? "validated", conflict: (n.flags ?? []).includes("conflict"), ref: n,
});
const edgeData = (e: GEdge) => ({
  id: e.id, source: e.source, target: e.target, type: e.type, strength: e.strength, status: e.status ?? "validated",
  provenance: PROVENANCE_EDGES.has(e.type), label: e.label, ref: e,
});

export const MapCanvas = forwardRef<CanvasHandle, Props>(function MapCanvas({ onNode, onEdge, onExpand, onBackground }, ref) {
  const box = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const handlers = useRef({ onNode, onEdge, onExpand, onBackground });
  const [ready, setReady] = useState(false);
  const [hover, setHover] = useState<{ node: GNode; x: number; y: number; w: number } | null>(null);
  const queue = useRef<(() => void)[]>([]);

  useEffect(() => {
    handlers.current = { onNode, onEdge, onExpand, onBackground };
  });

  useEffect(() => {
    let cy: cytoscape.Core | null = null;
    let resize: ResizeObserver | null = null;
    let alive = true;
    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    const restyle = () => cy?.style(stylesheet());
    import("cytoscape").then(({ default: cytoscape }) => {
      if (!alive || !box.current) return;
      cy = cytoscape({ container: box.current, style: stylesheet(), minZoom: 0.15, maxZoom: 2.5, boxSelectionEnabled: false });
      const c = cy;
      resize = new ResizeObserver(() => c.resize());  // the panel and the window change the canvas size; Cytoscape measures only on demand
      resize.observe(box.current);
      c.on("tap", "node", (e) => {
        const n = e.target as cytoscape.NodeSingular;
        c.elements().addClass("faded");
        n.closedNeighborhood().removeClass("faded");
        handlers.current.onNode(n.data("ref"));
      });
      c.on("dbltap", "node", (e) => handlers.current.onExpand((e.target as cytoscape.NodeSingular).data("ref")));
      c.on("tap", "edge", (e) => {
        const ed = e.target as cytoscape.EdgeSingular;
        c.elements().addClass("faded");
        ed.union(ed.connectedNodes()).removeClass("faded");
        handlers.current.onEdge(ed.data("ref"));
      });
      c.on("tap", (e) => {
        if (e.target === c) {
          c.elements().removeClass("faded");
          handlers.current.onBackground();
        }
      });
      c.on("mouseover", "node", (e) => {
        const p = (e.target as cytoscape.NodeSingular).renderedPosition();
        setHover({ node: e.target.data("ref"), x: p.x, y: p.y, w: c.width() });
      });
      c.on("mouseout", "node", () => setHover(null));
      c.on("mouseover", "edge", (e) => e.target.addClass("hover"));
      c.on("mouseout", "edge", (e) => e.target.removeClass("hover"));
      c.on("pan zoom", () => setHover(null));
      cyRef.current = c;
      scheme.addEventListener("change", restyle);
      setReady(true);
      queue.current.splice(0).forEach((f) => f());
    });
    return () => {
      alive = false;
      scheme.removeEventListener("change", restyle);
      resize?.disconnect();
      cy?.destroy();
      cyRef.current = null;
    };
  }, []);

  useImperativeHandle(ref, () => {
    const run = (f: (cy: cytoscape.Core) => void) => (cyRef.current ? f(cyRef.current) : queue.current.push(() => f(cyRef.current!)));
    const layout = (cy: cytoscape.Core, root?: string) => {
      const r = root ? cy.getElementById(root) : cy.nodes().max((n) => n.degree(false)).ele;
      if (!r || r.empty()) return;
      // rings by link distance from the root (what it touches directly, then one step further, ...), each ring ordered by
      // node type then label: deterministic, and the root always sits in the middle
      const depth = new Map<string, number>([[r.id(), 0]]);
      const queue: cytoscape.NodeSingular[] = [r];
      while (queue.length) {
        const n = queue.shift()!;
        n.neighborhood("node").forEach((m) => {
          if (!depth.has(m.id())) {
            depth.set(m.id(), depth.get(n.id())! + 1);
            queue.push(m);
          }
        });
      }
      const far = Math.max(0, ...depth.values()) + 1;
      const rings = new Map<number, cytoscape.NodeSingular[]>();
      cy.nodes().forEach((n) => {
        const d = depth.get(n.id()) ?? far;
        rings.set(d, [...(rings.get(d) ?? []), n]);
      });
      const pos = new Map<string, cytoscape.Position>([[r.id(), { x: 0, y: 0 }]]);
      let radius = 0;
      for (const d of [...rings.keys()].sort((a, b) => a - b).filter((d) => d > 0)) {
        const ring = rings.get(d)!.sort((a, b) => (TYPE_RANK[a.data("type")] ?? 9) - (TYPE_RANK[b.data("type")] ?? 9) || a.data("label").localeCompare(b.data("label")));
        radius = Math.max(radius + 150, (ring.length * 90) / (2 * Math.PI));  // ~90 px of arc per node keeps labels apart
        ring.forEach((n, i) => {
          const a = (2 * Math.PI * i) / ring.length - Math.PI / 2 + d * 0.35;
          pos.set(n.id(), { x: radius * Math.cos(a), y: radius * Math.sin(a) });
        });
      }
      cy.batch(() => cy.nodes().forEach((n) => {
        n.position(pos.get(n.id()) ?? { x: 0, y: 0 });
      }));
      cy.fit(undefined, 40);
    };
    return {
      add(nodes, edges, around) {
        let dropped = 0;
        run((cy) => {
          const existing = cy.nodes().length;
          const fresh = nodes.filter((n, i, all) => cy.getElementById(n.id).empty() && all.findIndex((m) => m.id === n.id) === i);
          const take = fresh.slice(0, Math.max(0, MAX_NODES - existing));
          dropped = fresh.length - take.length;
          const center = around ? cy.getElementById(around) : null;
          const ext = cy.extent();
          const c = center && center.nonempty() ? center.position() : { x: (ext.x1 + ext.x2) / 2, y: (ext.y1 + ext.y2) / 2 };
          const radius = 150 + take.length * 5;
          const start = existing ? (around ?? "").length % 7 : 0;
          cy.batch(() => {
            take.forEach((n, i) => {
              const a = start + (2 * Math.PI * i) / Math.max(take.length, 1);
              cy.add({ group: "nodes", data: nodeData(n), position: { x: c.x + radius * Math.cos(a), y: c.y + radius * Math.sin(a) } });
            });
            for (const n of nodes) {
              const el = cy.getElementById(n.id);
              if (el.nonempty() && !take.includes(n)) {  // seen again: may now carry a stronger status or a conflict flag
                const old = el.data("ref") as GNode;
                el.data(nodeData({ ...old, ...n, flags: [...new Set([...(old.flags ?? []), ...(n.flags ?? [])])] }));
              }
            }
            cy.add(edges.filter((e) => cy.getElementById(e.id).empty() && cy.getElementById(e.source).nonempty() && cy.getElementById(e.target).nonempty())
              .map((e) => ({ group: "edges" as const, data: edgeData(e) })));
          });
          if (existing <= 1) layout(cy, around);  // a fresh map (or just its anchor): lay everything out and fit
          else if (take.length && center && center.nonempty())
            cy.animate({ fit: { eles: center.closedNeighborhood(), padding: 48 } }, { duration: 250 });  // keep the new neighbours in view
        });
        return dropped;
      },
      remove(ids) {
        const gone = new Set(ids);
        run((cy) => {
          cy.remove(cy.nodes().filter((n) => gone.has(n.id())));
        });
      },
      clear() {
        run((cy) => {
          cy.elements().remove();
        });
      },
      layout(root) {
        run((cy) => layout(cy, root));
      },
      focus(id) {
        run((cy) => {
          const n = cy.getElementById(id);
          if (n.empty()) return;
          cy.elements().addClass("faded");
          n.closedNeighborhood().removeClass("faded");
          cy.$(":selected").unselect();
          n.select();
          cy.animate({ center: { eles: n } }, { duration: 250 });
        });
      },
      select(id) {
        run((cy) => {
          cy.$(":selected").unselect();
          if (id) cy.getElementById(id).select();
          else cy.elements().removeClass("faded");
        });
      },
      fit() {
        run((cy) => {
          cy.fit(cy.elements(":visible"), 40);
        });
      },
      setVisible(node, edge) {
        run((cy) =>
          cy.batch(() => {
            cy.nodes().forEach((n) => {
              n.toggleClass("hidden", !node(n.data("ref")));
            });
            cy.edges().forEach((e) => {
              e.toggleClass("hidden", !edge(e.data("ref")) || e.source().hasClass("hidden") || e.target().hasClass("hidden"));
            });
          }),
        );
      },
      has: (id) => !!cyRef.current?.getElementById(id).nonempty(),
      size: () => cyRef.current?.nodes().length ?? 0,
      neighbours: (id) => cyRef.current?.getElementById(id).neighborhood("node").map((n) => n.id()) ?? [],
    };
  }, []);

  const zoom = (f: number) => {
    const cy = cyRef.current;
    if (cy) cy.animate({ zoom: { level: cy.zoom() * f, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: 150 });
  };
  return (
    <div className="absolute inset-0">
      {/* inline position: Cytoscape injects an unlayered `position: relative` rule for its container that outranks Tailwind's
          layered utilities, which collapsed the canvas to zero height */}
      <div ref={box} style={{ position: "absolute", inset: 0 }} role="img" aria-label="Knowledge map canvas. The search box and the details panel offer the same navigation by keyboard." />
      {!ready && <div className="absolute inset-0 grid place-items-center text-sm text-faint">Loading map…</div>}
      <div className="absolute left-3 top-3 flex flex-col overflow-hidden rounded-lg border border-line bg-surface shadow-sm">
        <button onClick={() => zoom(1.25)} className="p-1.5 text-muted hover:bg-surface-2 hover:text-ink" aria-label="Zoom in" title="Zoom in">
          <I.Plus width={16} height={16} />
        </button>
        <button onClick={() => zoom(0.8)} className="border-t border-line p-1.5 text-muted hover:bg-surface-2 hover:text-ink" aria-label="Zoom out" title="Zoom out">
          <I.Dash width={16} height={16} />
        </button>
        <button
          onClick={() => cyRef.current?.fit(cyRef.current.elements(":visible"), 40)}
          className="border-t border-line p-1.5 text-muted hover:bg-surface-2 hover:text-ink"
          aria-label="Fit to screen"
          title="Fit to screen"
        >
          <I.Ring width={16} height={16} />
        </button>
      </div>
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-70 -translate-x-1/2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-md"
          style={{ left: Math.min(Math.max(hover.x, 144), hover.w - 144), top: Math.max(hover.y - 52, 4) }}
        >
          <p className="font-medium text-ink">{hover.node.label}</p>
          <p className="text-muted">
            {hover.node.type}
            {hover.node.origin && hover.node.origin !== "canonical" ? ` · ${hover.node.origin.replace("_", " ")} · ${hover.node.status}` : ""}
            {(hover.node.flags ?? []).includes("conflict") ? " · conflict recorded" : ""}
          </p>
        </div>
      )}
    </div>
  );
});
