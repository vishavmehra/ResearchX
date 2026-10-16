"""Tiny draw.io writer for the ResearchX diagrams: one style system, hand-placed geometry, explicit orthogonal routes.

Every connector is drawn with draw.io's `edgeStyle=none` and explicit waypoints, so the file renders exactly as laid out
(no automatic routing) and stays easy to adjust by hand in draw.io. `Diagram.check()` reports diagonal segments,
connectors that pass behind a box they do not connect, connectors that cross or run along each other, labels that touch
boxes, other labels or other connectors, overlapping boxes and anything closer than 16 px to the canvas edge.
render.mjs then measures the rendered text in the browser (text overflowing its shape, labels touching text).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# Palette: the web app's tokens (apps/web/app/globals.css): paper and ink, one teal accent, semantic colours for status.
PAPER, SURFACE, SURFACE2, INK, MUTED, FAINT = "#f6f5f1", "#ffffff", "#efede6", "#16181d", "#5b616d", "#8a8f99"
RULE, ACCENT, ACCENT_SOFT = "#cfcabd", "#0e6b66", "#e2f0ee"
OK, OK_SOFT, WARN, WARN_SOFT, BAD, BAD_SOFT, INFO = "#1e7a46", "#e4f3ea", "#8f5405", "#fbf0dc", "#b42318", "#fbe9e7", "#eef1f6"
SANS, MONO, SERIF = "IBM Plex Sans", "IBM Plex Mono", "Source Serif 4"
FONT_CSS = {SANS: "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap",
            MONO: "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap",
            SERIF: "https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,600;1,400&display=swap"}
NODE = {  # kind -> (fill, stroke, stroke width, extra style)
    "service": (SURFACE, RULE, 1.25, ""),
    "focal": (ACCENT_SOFT, ACCENT, 2, ""),
    "store": (SURFACE, RULE, 1.25, "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=9;"),
    "external": (SURFACE, FAINT, 1.25, "dashed=1;dashPattern=5 4;"),
    "advisory": (WARN_SOFT, WARN, 1.25, "dashed=1;dashPattern=5 4;"),
    "ok": (OK_SOFT, OK, 1.5, ""),
    "warn": (WARN_SOFT, WARN, 1.5, ""),
    "bad": (BAD_SOFT, BAD, 1.5, ""),
    "info": (INFO, "#9aa6b8", 1.25, ""),
    "plain": (SURFACE, INK, 1.25, ""),
    "planned": (SURFACE, FAINT, 1.25, "dashed=1;dashPattern=2 3;"),
}
ARROW = {  # kind -> (colour, width, dashed)
    "default": (MUTED, 1.4, False), "accent": (ACCENT, 2, False), "async": (MUTED, 1.4, True), "advisory": (WARN, 1.4, True),
    "ok": (OK, 1.6, False), "bad": (BAD, 1.6, True), "faint": (FAINT, 1.25, True),
}
MARGIN = 16  # nothing closer than this to the canvas edge
LABEL_CHAR, LABEL_PAD = 6.3, 12  # connector labels: IBM Plex Mono 10.5 px advances 0.6 em per character


def font(family: str) -> str:
    return f"fontFamily={family};fontSource={quote(FONT_CSS[family], safe='')};"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    kind: str = "service"
    solid: bool = True  # takes part in the overlap / transit checks

    def point(self, fx: float, fy: float) -> tuple[float, float]:
        return self.x + fx * self.w, self.y + fy * self.h


@dataclass
class Diagram:
    slug: str
    title: str
    subtitle: str
    width: int
    height: int
    cells: list[str] = field(default_factory=list)
    boxes: dict[str, Box] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    labels: list[tuple[str, str, float, float, float, float]] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    n: int = 0

    def __post_init__(self):
        self._cell("bg", "", f"rounded=0;fillColor={PAPER};strokeColor=none;", 0, 0, self.width, self.height, kind="bg", solid=False)
        self.text(40, 26, self.width - 80, 36, self.title, size=26, family=SERIF, bold=True)
        self.text(40, 66, self.width - 80, 22, self.subtitle, size=14, color=MUTED)

    # -------------------------------------------------------------------------------------------- primitives
    def _cell(self, cid, value, style, x, y, w, h, kind="service", solid=True, track=True):
        self._claim(cid)
        self.cells.append(f'<mxCell id="{cid}" value="{html.escape(value, quote=True)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        if track:
            self.boxes[cid] = Box(cid, x, y, w, h, kind, solid)

    def _claim(self, cid):
        """Cell ids must be unique: draw.io silently drops one of two cells that share an id."""
        if cid in self.ids:
            raise ValueError(f"{self.slug}: duplicate cell id {cid!r}")
        self.ids.add(cid)

    def _id(self, prefix):
        self.n += 1
        while f"{prefix}{self.n}" in self.ids:
            self.n += 1
        return f"{prefix}{self.n}"

    def text(self, x, y, w, h, s, size=12, family=SANS, color=INK, bold=False, align="left", italic=False, html_value=False, valign="top"):
        style = (f"text;html=1;whiteSpace=wrap;{font(family)}fontSize={size};fontColor={color};align={align};verticalAlign={valign};"
                 f"spacing=0;fontStyle={(1 if bold else 0) + (2 if italic else 0)};")
        self._cell(self._id("t"), s if html_value else esc(s), style, x, y, w, h, track=False)

    def zone(self, x, y, w, h, label=None, fill=SURFACE2, stroke=None, dashed=False, label_color=MUTED, cid=None):
        """A background band. With `cid` it can also be a connector end (it never blocks connectors)."""
        style = (f"rounded=1;arcSize=10;absoluteArcSize=1;fillColor={fill};strokeColor={stroke or 'none'};"
                 + ("dashed=1;dashPattern=6 4;" if dashed else ""))
        self._cell(cid or self._id("z"), "", style, x, y, w, h, kind="zone", solid=False)
        if label:
            lines = label.split("\n")
            body = "<br>".join(esc(s) for s in lines)
            self.text(x + 14, y + 10, w - 28, 15 * len(lines) + 2, f'<span style="letter-spacing:0.08em">{body}</span>', size=10.5,
                      family=MONO, color=label_color, html_value=True)

    def node(self, cid, x, y, w, h, name, detail=None, tag=None, kind="service", align="left", name_size=15, detail_size=12):
        fill, stroke, width, extra = NODE[kind]
        parts = []
        if tag:
            parts.append(f'<div style="font-family:{MONO};font-size:10.5px;letter-spacing:0.04em;color:{FAINT};line-height:1.3;">{esc(tag)}</div>')
        if name:
            parts.append(f'<div style="font-size:{name_size}px;font-weight:600;color:{INK};margin-top:{3 if tag else 0}px;line-height:1.25;">'
                         f'{esc(name)}</div>')
        if detail:
            parts.append(f'<div style="font-size:{detail_size}px;color:{MUTED};margin-top:4px;line-height:1.38;">{esc(detail)}</div>')
        top = 2 if kind == "store" else 10
        style = (f"rounded=1;arcSize=8;absoluteArcSize=1;whiteSpace=wrap;html=1;align={align};verticalAlign={'top' if align == 'left' else 'middle'};"
                 f"spacingLeft={12 if align == 'left' else 8};spacingRight={10 if align == 'left' else 8};spacingTop={top if align == 'left' else 0};"
                 f"{font(SANS)}fillColor={fill};strokeColor={stroke};strokeWidth={width};{extra}")
        self._cell(cid, "".join(parts), style, x, y, w, h, kind=kind)
        return cid

    def chip(self, cid, x, y, w, s, kind="service", h=30, size=11.5):
        """A compact one-line box for an identifier (a script stage, a table) in IBM Plex Mono."""
        fill, stroke, width, extra = NODE[kind]
        style = (f"rounded=1;arcSize=6;absoluteArcSize=1;whiteSpace=wrap;html=1;align=left;verticalAlign=middle;spacingLeft=10;spacingRight=6;"
                 f"{font(MONO)}fontSize={size};fontColor={INK};fillColor={fill};strokeColor={stroke};strokeWidth={width};"
                 + extra.replace("size=9", "size=5"))
        self._cell(cid, esc(s), style, x, y, w, h, kind=kind)
        return cid

    def decision(self, cid, x, y, w, h, s, size=12.5):
        style = (f"rhombus;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;{font(SANS)}fontSize={size};fontColor={INK};"
                 f"fontStyle=1;fillColor={SURFACE};strokeColor={INK};strokeWidth=1.25;spacing=14;")
        self._cell(cid, esc(s), style, x, y, w, h, kind="decision")
        return cid

    def badge(self, x, y, s, kind="ok", w=None, sub=None):
        """A small status chip; `sub` adds a second, regular-weight line."""
        fill, stroke, color = {"ok": (OK_SOFT, OK, OK), "warn": (WARN_SOFT, WARN, WARN), "bad": (BAD_SOFT, BAD, BAD), "info": (INFO, "#9aa6b8", INK),
                               "muted": (SURFACE, FAINT, MUTED), "accent": (ACCENT_SOFT, ACCENT, ACCENT)}[kind]
        w = w or round(max(len(s), len(sub or "")) * 6.1 + 16)
        value = esc(s) if sub is None else f'<div style="font-weight:600">{esc(s)}</div><div style="font-weight:400">{esc(sub)}</div>'
        style = (f"rounded=1;arcSize=4;absoluteArcSize=1;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;{font(MONO)}"
                 f"fontSize=10;fontStyle={1 if sub is None else 0};fontColor={color};fillColor={fill};strokeColor={stroke};strokeWidth=1;"
                 + ("dashed=1;dashPattern=3 3;" if kind == "muted" else ""))
        self._cell(self._id("b"), value if sub is not None else esc(s), style, x, y, w, 18 if sub is None else 34, track=False)
        return w

    def note(self, x, y, w, h, s, size=13):
        self.text(x, y, w, h, s, size=size, family=SERIF, color=MUTED, italic=True)

    def table(self, x, y, widths, rows, heights, header=True):
        """A grid of cells; rows[0] is the header when `header`. Registered as one solid box for the checks."""
        cy = y
        for r, (row, h) in enumerate(zip(rows, heights)):
            cx = x
            for c, (s, w) in enumerate(zip(row, widths)):
                head = header and r == 0
                style = (f"rounded=0;whiteSpace=wrap;html=1;align=left;verticalAlign=middle;spacingLeft=10;spacingRight=8;"
                         + (f"{font(MONO)}fontSize=10.5;fontColor={MUTED};fillColor={SURFACE2};" if head else
                            f"{font(SANS)}fontSize=12;fontColor={INK if c == 0 else MUTED};fillColor={SURFACE};fontStyle={1 if c == 0 else 0};")
                         + f"strokeColor={RULE};strokeWidth=1;")
                self._cell(self._id("c"), esc(s), style, cx, cy, w, h, track=False)
                cx += w
            cy += h
        tid = self._id("tbl")
        self.boxes[tid] = Box(tid, x, y, sum(widths), sum(heights), "table", True)
        return cy

    def edge(self, src, dst, exit, entry, points=(), label=None, at=0.5, seg=None, kind="default", label_bg=PAPER, start_arrow=False,
             pos=None, jump=False):
        """`exit` / `entry`: (fx, fy) on the source / target box; `points`: absolute corner points. The label sits at `pos`
        (an absolute point on the route) or on segment `seg` (default: the longest) at fraction `at`, on an opaque
        background of `label_bg`. `jump`: draw line jumps where this connector crosses another (crossing allowed)."""
        a, b = self.boxes[src], self.boxes[dst]
        path = [a.point(*exit), *points, b.point(*entry)]
        color, width, dashed = ARROW[kind]
        style = (f"edgeStyle=none;rounded=1;arcSize=12;html=1;exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;entryX={entry[0]};"
                 f"entryY={entry[1]};entryDx=0;entryDy=0;endArrow=block;endFill=1;endSize=6;"
                 + ("startArrow=block;startFill=1;startSize=6;" if start_arrow else "startArrow=none;")
                 + ("jumpStyle=arc;jumpSize=10;" if jump else "")
                 + f"strokeWidth={width};strokeColor={color};{'dashed=1;dashPattern=6 4;' if dashed else ''}{font(MONO)}fontSize=10.5;"
                 f"fontColor={color if kind in ('accent', 'advisory', 'ok', 'bad') else MUTED};labelBackgroundColor={label_bg};")
        eid = self._id("e")
        self._claim(eid)
        geo = '<mxGeometry relative="1" as="geometry">'
        if points:
            geo += '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
        if label:
            segs = list(zip(path, path[1:]))
            lens = [abs(q[0] - p[0]) + abs(q[1] - p[1]) for p, q in segs]
            if pos is not None:  # the segment that holds `pos`
                i = next(k for k, (p, q) in enumerate(segs) if min(p[0], q[0]) - .5 <= pos[0] <= max(p[0], q[0]) + .5
                         and min(p[1], q[1]) - .5 <= pos[1] <= max(p[1], q[1]) + .5)
                lx, ly = pos
            else:
                i = seg if seg is not None else max(range(len(segs)), key=lambda k: lens[k])
                (x1, y1), (x2, y2) = segs[i]
                lx, ly = x1 + (x2 - x1) * at, y1 + (y2 - y1) * at
            x1, y1 = segs[i][0]
            done = sum(lens[:i]) + abs(lx - x1) + abs(ly - y1)
            geo = geo.replace('relative="1"', f'relative="1" x="{round(2 * done / sum(lens) - 1, 4)}"')
            w = len(label) * LABEL_CHAR + LABEL_PAD
            self.labels.append((eid, label, lx - w / 2, ly - 8, w, 16))
        geo += "</mxGeometry>"
        self.cells.append(f'<mxCell id="{eid}" value="{html.escape(label or "", quote=True)}" style="{style}" edge="1" parent="1" '
                          f'source="{src}" target="{dst}">{geo}</mxCell>')
        self.edges.append({"id": eid, "src": src, "dst": dst, "path": path, "label": label, "jump": jump})
        return eid

    def legend(self, y, items, x=40):
        """A horizontal strip, wrapped onto a second row when needed. items: (kind, text); kinds are node kinds, or arrow
        kinds prefixed 'arrow:'. Returns the y below the legend."""
        cx, row = x, y
        for kind, s in items:
            w = len(s) * 6.9 + 8
            need = (36 if kind.startswith("arrow:") else 30) + w
            if cx + need > self.width - 40:
                cx, row = x, row + 28
            if kind.startswith("arrow:"):
                color, width, dashed = ARROW[kind.split(":", 1)[1]]
                style = f"line;strokeWidth={width};strokeColor={color};{'dashed=1;dashPattern=6 4;' if dashed else ''}"
                self._cell(self._id("l"), "", style, cx, row + 8, 28, 2, track=False)
                cx += 36
            else:
                fill, stroke, _, extra = NODE[kind]
                extra = extra.replace("size=9", "size=4").replace("boundedLbl=1;backgroundOutline=1;", "")
                self._cell(self._id("l"), "", f"rounded=1;arcSize=3;absoluteArcSize=1;fillColor={fill};strokeColor={stroke};{extra}", cx, row + 1, 22, 16,
                           track=False)
                cx += 30
            self.text(cx, row + 1, w, 18, s, size=11.5, color=MUTED)
            cx += w + 18
        return row + 18

    # -------------------------------------------------------------------------------------------- checks + output
    def check(self) -> list[str]:
        problems = []
        solid = [b for b in self.boxes.values() if b.solid]
        for b in self.boxes.values():
            if b.id != "bg" and (b.x < MARGIN or b.y < MARGIN or b.x + b.w > self.width - MARGIN or b.y + b.h > self.height - MARGIN):
                problems.append(f"{b.id} closer than {MARGIN} px to the canvas edge")
        for i, a in enumerate(solid):
            for b in solid[i + 1:]:
                if a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h:
                    problems.append(f"boxes overlap: {a.id} / {b.id}")
        segs = []
        for e in self.edges:
            for (x1, y1), (x2, y2) in zip(e["path"], e["path"][1:]):
                if abs(x1 - x2) > 0.5 and abs(y1 - y2) > 0.5:
                    problems.append(f"{e['id']} {e['src']}->{e['dst']}: diagonal segment ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
                    continue
                if abs(x1 - x2) + abs(y1 - y2) < 0.5:
                    continue
                if not (MARGIN <= min(x1, x2) and max(x1, x2) <= self.width - MARGIN and MARGIN <= min(y1, y2) and max(y1, y2) <= self.height - MARGIN):
                    problems.append(f"{e['id']} {e['src']}->{e['dst']} runs closer than {MARGIN} px to the canvas edge")
                lo_x, hi_x, lo_y, hi_y = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                for b in solid:
                    if b.id not in (e["src"], e["dst"]) and lo_x < b.x + b.w - 1 and hi_x > b.x + 1 and lo_y < b.y + b.h - 1 and hi_y > b.y + 1:
                        problems.append(f"{e['id']} {e['src']}->{e['dst']} passes behind {b.id}")
                segs.append((e, (x1, y1), (x2, y2)))
        for i, (e1, p1, q1) in enumerate(segs):
            h1 = abs(p1[1] - q1[1]) < .5
            for e2, p2, q2 in segs[i + 1:]:
                if e1 is e2:
                    continue
                h2 = abs(p2[1] - q2[1]) < .5
                if h1 != h2:  # perpendicular: a crossing strictly inside both segments
                    (hp, hq), (vp, vq) = ((p1, q1), (p2, q2)) if h1 else ((p2, q2), (p1, q1))
                    x, y = vp[0], hp[1]
                    if (min(hp[0], hq[0]) + 1 < x < max(hp[0], hq[0]) - 1 and min(vp[1], vq[1]) + 1 < y < max(vp[1], vq[1]) - 1
                            and not (e1["jump"] or e2["jump"])):
                        problems.append(f"{e1['id']} {e1['src']}->{e1['dst']} crosses {e2['id']} {e2['src']}->{e2['dst']} at ({x:.0f},{y:.0f})")
                else:  # parallel: a shared run, allowed only for connectors that share an end (a fan-out or fan-in trunk)
                    c, r = (1, 0) if h1 else (0, 1)
                    if abs(p1[c] - p2[c]) < 1:
                        lo = max(min(p1[r], q1[r]), min(p2[r], q2[r]))
                        hi = min(max(p1[r], q1[r]), max(p2[r], q2[r]))
                        if hi - lo > 1 and e1["src"] != e2["src"] and e1["dst"] != e2["dst"]:
                            problems.append(f"{e1['id']} and {e2['id']} run along each other")
        for k, (eid, name, lx, ly, lw, lh) in enumerate(self.labels):
            for b in solid:
                if lx < b.x + b.w and b.x < lx + lw and ly < b.y + b.h and b.y < ly + lh:
                    problems.append(f"label {name!r} overlaps {b.id}")
            for e, (x1, y1), (x2, y2) in segs:
                if e["id"] != eid and min(x1, x2) < lx + lw - 1 and max(x1, x2) > lx + 1 and min(y1, y2) < ly + lh - 1 and max(y1, y2) > ly + 1:
                    problems.append(f"label {name!r} sits on connector {e['id']} {e['src']}->{e['dst']}")
            for _, other, ox, oy, ow, oh in self.labels[k + 1:]:
                if lx < ox + ow and ox < lx + lw and ly < oy + oh and oy < ly + lh:
                    problems.append(f"labels overlap: {name!r} / {other!r}")
        return problems

    def save(self, out_dir: Path) -> tuple[Path, list[str]]:
        xml = (f'<mxfile host="ResearchX diagrams/src/build.py" type="device"><diagram id="{self.slug}" name="{esc(self.title)}">'
               f'<mxGraphModel dx="{self.width}" dy="{self.height}" grid="1" gridSize="4" guides="1" tooltips="1" connect="1" arrows="1" '
               f'fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" background="{PAPER}" math="0" shadow="0">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{"".join(self.cells)}</root></mxGraphModel></diagram></mxfile>')
        path = out_dir / f"{self.slug}.drawio"
        path.write_text(xml, encoding="utf-8")
        return path, self.check()
