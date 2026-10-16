// Export diagrams/*.drawio to PNG (2x) and SVG with the official draw.io viewer in headless Chromium, then check the
// rendered layout at its real size: text that overflows its shape, connector labels that touch shapes, text or other
// connectors, connectors that run through text, and anything closer than 16 px to the canvas edge. The SVG embeds the
// Latin subsets of IBM Plex Sans / Mono and Source Serif 4, so it renders the same where web fonts cannot load.
// Usage: node diagrams/src/render.mjs [prefix ...]   (playwright-core: set PLAYWRIGHT_CORE to its path if it is not
// installed here; needs network access for the viewer script and the fonts). Exit code 2 when a file fails to render or
// a check fails.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || "playwright-core");
const DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FONTS = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600"
  + "&family=Source+Serif+4:ital,wght@0,600;1,400&display=swap";
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";

async function embeddedFonts() {  // the Latin @font-face rules with their woff2 files inlined as data: URIs
  const css = await (await fetch(FONTS, { headers: { "user-agent": UA } })).text();
  const faces = [];
  for (const block of css.split("/* ").filter((b) => b.startsWith("latin */"))) {
    const face = block.slice(block.indexOf("@font-face"));
    const url = face.match(/url\((https:[^)]+)\)/)[1];
    const data = Buffer.from(await (await fetch(url)).arrayBuffer()).toString("base64");
    faces.push(face.replace(url, `data:font/woff2;base64,${data}`).trim());
  }
  return faces.join("\n");
}

function checkLayout() {  // runs in the page: returns a list of problems
  const g = window.__viewer.graph, view = g.view, model = g.getModel(), out = [];
  const svgBox = g.container.querySelector("svg").getBoundingClientRect();
  const rel = (r) => ({ x: r.left - svgBox.left, y: r.top - svgBox.top, w: r.width, h: r.height });
  const content = (st) => {
    const divs = st && st.text && st.text.node ? st.text.node.querySelectorAll("foreignObject div") : [];
    return divs.length >= 3 ? rel(divs[2].getBoundingClientRect()) : null;
  };
  const hit = (a, b, pad = 0) => a.x < b.x + b.w - pad && b.x < a.x + a.w - pad && a.y < b.y + b.h - pad && b.y < a.y + a.h - pad;
  const inside = (a, b, tol = 1) => a.x >= b.x - tol && a.y >= b.y - tol && a.x + a.w <= b.x + b.w + tol && a.y + a.h <= b.y + b.h + tol;
  const f = (r) => `${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.w)}x${Math.round(r.h)}`;
  const bg = view.getState(model.getCell("bg"));
  const canvas = { x: bg.x + 16, y: bg.y + 16, w: bg.width - 32, h: bg.height - 32 };
  const shapes = [], texts = [], labels = [], segs = [], nodeText = [];
  for (const c of Object.values(model.cells)) {
    const st = view.getState(c);
    if (!st || c.id === "0" || c.id === "1" || c.id === "bg") continue;
    const style = c.style || "";
    if (c.vertex) {
      const box = { id: c.id, x: st.x, y: st.y, w: st.width, h: st.height };
      if (!inside(box, canvas, 0.5)) out.push(`${c.id} is closer than 16 px to the canvas edge`);
      const r = content(st);
      const isText = style.startsWith("text;"), isZone = /^z\d/.test(c.id) || /fillColor=[^;]+;strokeColor=none/.test(style) && !r;
      if (r && r.w > 0) {
        const limit = { ...box };
        if (style.includes("cylinder3")) limit.h -= 9;
        if (style.startsWith("rhombus")) {
          const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
          const corners = [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]];
          if (corners.some(([px, py]) => Math.abs(px - cx) / (box.w / 2) + Math.abs(py - cy) / (box.h / 2) > 1.04))
            out.push(`text leaves the diamond ${c.id}: ${f(r)}`);
        } else if (!inside(r, limit, isText ? 3 : 1)) out.push(`text overflows ${c.id}: text ${f(r)} in box ${f(limit)}`);
        if (isText) texts.push({ ...r, id: c.id });
        else if (!/^b\d/.test(c.id)) nodeText.push({ ...r, id: c.id });
      }
      if (!isText && !/^z\d/.test(c.id) && !/^l\d/.test(c.id) && !isZone) shapes.push(box);
    } else if (c.edge) {
      const pts = st.absolutePoints || [];
      for (let i = 0; i + 1 < pts.length; i++) segs.push({ id: c.id, a: pts[i], b: pts[i + 1] });
      for (const p of pts) if (p.x < canvas.x - 0.5 || p.y < canvas.y - 0.5 || p.x > canvas.x + canvas.w + 0.5 || p.y > canvas.y + canvas.h + 0.5)
        out.push(`connector ${c.id} runs closer than 16 px to the canvas edge`);
      const r = c.value ? content(st) : null;
      if (r) labels.push({ ...r, id: c.id, value: c.value });
    }
  }
  const segHit = (s, r) => {
    const lo = { x: Math.min(s.a.x, s.b.x), y: Math.min(s.a.y, s.b.y) }, hi = { x: Math.max(s.a.x, s.b.x), y: Math.max(s.a.y, s.b.y) };
    return lo.x < r.x + r.w - 1 && hi.x > r.x + 1 && lo.y < r.y + r.h - 1 && hi.y > r.y + 1;
  };
  for (const l of labels) {
    for (const s of shapes) if (hit(l, s)) out.push(`label "${l.value}" touches ${s.id}`);
    for (const t of texts.concat(nodeText)) if (hit(l, t)) out.push(`label "${l.value}" touches the text of ${t.id}`);
    for (const s of segs) if (s.id !== l.id && segHit(s, l)) out.push(`label "${l.value}" sits on connector ${s.id}`);
  }
  for (let i = 0; i < labels.length; i++) for (const m of labels.slice(i + 1)) if (hit(labels[i], m)) out.push(`labels overlap: "${labels[i].value}" / "${m.value}"`);
  for (const t of texts) {
    for (const s of segs) if (segHit(s, t)) out.push(`connector ${s.id} runs through the text of ${t.id}`);
    for (const s of shapes) if (hit(t, s, 1) && !inside(t, s)) out.push(`text ${t.id} crosses the border of ${s.id}`);
    for (const n of nodeText) if (hit(t, n, 1)) out.push(`text ${t.id} overlaps the text of ${n.id}`);
  }
  for (const s of shapes.filter((s) => /^b\d/.test(s.id))) for (const n of nodeText) if (hit(s, n, 1)) out.push(`badge ${s.id} covers the text of ${n.id}`);
  return [...new Set(out)];
}

const wanted = process.argv.slice(2);
const files = fs.readdirSync(DIR).filter((f) => f.endsWith(".drawio") && (!wanted.length || wanted.some((w) => f.startsWith(w))));
const fontCss = await embeddedFonts();
const browser = await chromium.launch();
const results = [];
for (const f of files) {
  const xml = fs.readFileSync(path.join(DIR, f), "utf8");
  const page = await browser.newPage({ viewport: { width: 1800, height: 1600 }, deviceScaleFactor: 2 });
  await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="${FONTS}">
    <style>:root{color-scheme:light}html,body{margin:0;background:#f6f5f1}</style></head><body><div id="c"></div></body></html>`);
  await page.evaluate(() => Promise.all(["400 12px 'IBM Plex Sans'", "600 12px 'IBM Plex Sans'", "400 12px 'IBM Plex Mono'",
    "600 12px 'IBM Plex Mono'", "600 26px 'Source Serif 4'", "italic 400 13px 'Source Serif 4'"].map((s) => document.fonts.load(s))));
  await page.evaluate((x) => document.getElementById("c").setAttribute("data-mxgraph",
    JSON.stringify({ xml: x, toolbar: "", nav: false, resize: false, lightbox: false, "auto-fit": false })), xml);
  await page.addScriptTag({ url: "https://viewer.diagrams.net/js/viewer-static.min.js" });
  try {
    await page.evaluate(() => new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("the viewer did not render within 30 s")), 30000);
      GraphViewer.createViewerForElement(document.getElementById("c"), (v) => { clearTimeout(t); window.__viewer = v; resolve(); });
    }));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(400);
    const problems = await page.evaluate(checkLayout);
    const el = await page.$("#c svg");
    const base = f.replace(/\.drawio$/, "");
    await el.screenshot({ path: path.join(DIR, `${base}.png`) });
    const svg = await page.evaluate((css) => {
      const s = document.querySelector("#c svg").cloneNode(true);
      s.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      s.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
      const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
      style.textContent = css;
      s.insertBefore(style, s.firstChild);
      return new XMLSerializer().serializeToString(s);
    }, fontCss);
    fs.writeFileSync(path.join(DIR, `${base}.svg`), svg);
    const box = await el.boundingBox();
    results.push({ file: f, ok: problems.length === 0, size: [Math.round(box.width), Math.round(box.height)], problems });
  } catch (e) {
    results.push({ file: f, ok: false, error: String(e.message).slice(0, 200) });
  }
  await page.close();
}
await browser.close();
for (const r of results) {
  console.log(`${r.ok ? "ok  " : "FAIL"} ${r.file}${r.size ? ` ${r.size.join("x")}` : ""}${r.error ? ` ${r.error}` : ""}`);
  for (const p of r.problems || []) console.log(`     ${p}`);
}
if (results.some((r) => !r.ok)) process.exit(2);
