// Figures for about.html, drawn as inline SVG from data/writeup.json (written by
// `subway export`). Colors come from CSS custom properties, so light/dark follows the page.
// Each SVG's viewBox width equals its typical displayed width, so text renders at true size.

const NS = "http://www.w3.org/2000/svg";
const fmt = new Intl.NumberFormat("en-US");
const tip = document.getElementById("tip");
const COL = 680;    // .wrap content width
const WIDE = 1000;  // .wide content width

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, parent = null, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(n.style, v);
    else n.setAttribute(k, v);
  }
  if (text !== undefined) n.textContent = text;
  if (parent) parent.append(n);
  return n;
}
function html(tag, attrs = {}, parent = null, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(n.style, v);
    else n.setAttribute(k, v);
  }
  if (text !== undefined) n.textContent = text;
  if (parent) parent.append(n);
  return n;
}
const linear = (d0, d1, r0, r1) => {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
  return f;
};
function niceTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
// 12-hour clock in the article's style: "8:45 a.m."; short form for axis ticks: "9a", "noon".
function clock(s, short = false) {
  const h = Math.floor(s / 3600) % 24, m = Math.floor((s % 3600) / 60);
  const h12 = ((h + 11) % 12) + 1;
  if (short) {
    if (m === 0 && h === 12) return "noon";
    if (m === 0 && h === 0) return "midnight";
    return `${h12}${m ? ":" + String(m).padStart(2, "0") : ""}${h < 12 ? "a" : "p"}`;
  }
  return `${h12}:${String(m).padStart(2, "0")} ${h < 12 ? "a.m." : "p.m."}`;
}
const kfmt = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`);

// Tooltip: value first, then secondary lines. Text only (textContent), never HTML.
function showTip(lines, x, y) {
  tip.replaceChildren();
  lines.forEach((line, i) => html("div", { class: i === 0 ? "v" : "s" }, tip, line));
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.min(y + 14, innerHeight - r.height - 8)}px`;
}
const hideTip = () => { tip.style.display = "none"; };
// Hover (and, for sparse marks, keyboard focus) shows a tooltip. Dense marks (hundreds of
// trains or segments) aren't focusable; their numbers live in tables and the live map.
function hover(node, lines, focusable = true) {
  const get = () => (typeof lines === "function" ? lines() : lines);
  node.addEventListener("pointermove", (e) => showTip(get(), e.clientX, e.clientY));
  node.addEventListener("pointerleave", hideTip);
  if (focusable) {
    node.classList.add("hit");
    node.setAttribute("tabindex", "0");
    node.addEventListener("focus", () => { const r = node.getBoundingClientRect(); showTip(get(), r.right, r.top); });
    node.addEventListener("blur", hideTip);
  }
}
function legend(parent, items) {
  const box = html("div", { class: "legend" }, parent);
  for (const it of items) {
    const s = html("span", {}, box);
    if (it.kind === "dot") html("span", { class: "key-dot", style: { background: it.color } }, s);
    else if (it.kind === "swatch") html("span", { class: "key-line", style: { borderTopColor: it.color, borderTopWidth: `${it.width || 2}px`, opacity: it.opacity ?? 1 } }, s);
    else html("span", { class: "key-line", style: { borderTopColor: it.color, borderTopStyle: it.dashed ? "dashed" : "solid" } }, s);
    s.append(document.createTextNode(it.label));
  }
  return box;
}
function caption(fig, bold, rest) {
  const c = html("figcaption", {}, fig);
  if (bold) html("b", {}, c, bold);
  c.append(document.createTextNode((bold ? " " : "") + rest));
}
function table(parent, header, rows) {
  const det = html("details", {}, parent);
  html("summary", {}, det, "Data table");
  const t = html("table", {}, det);
  const h = html("tr", {}, t);
  for (const c of header) html("th", {}, h, c);
  for (const r of rows) { const tr = html("tr", {}, t); for (const c of r) html("td", {}, tr, c); }
}
function axisX(g, x, ticks, y0, fmtTick, label) {
  for (const t of ticks) {
    el("line", { x1: x(t), x2: x(t), y1: y0, y2: y0 + 4, class: "axis" }, g);
    el("text", { x: x(t), y: y0 + 17, class: "tick", "text-anchor": "middle" }, g, fmtTick(t));
  }
  if (label) el("text", { x: x(ticks[ticks.length - 1]), y: y0 + 34, class: "label-muted", "text-anchor": "end" }, g, label);
}
function gridY(g, y, ticks, x0, x1, fmtTick) {
  for (const t of ticks) {
    el("line", { x1: x0, x2: x1, y1: y(t), y2: y(t), class: t === 0 ? "axis" : "grid" }, g);
    el("text", { x: x0 - 6, y: y(t) + 4, class: "tick", "text-anchor": "end" }, g, fmtTick(t));
  }
}
// Crosshair for line charts: snaps to the nearest x and lists every series there.
function crosshair(svg, xs, x, top, bottom, left, right, readout) {
  const line = el("line", { y1: top, y2: bottom, visibility: "hidden", style: { stroke: "var(--text-secondary)" } }, svg);
  const hit = el("rect", { x: left, y: top, width: right - left, height: bottom - top, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const v = x.invert(p.matrixTransform(svg.getScreenCTM().inverse()).x);
    let i = 0;
    xs.forEach((xv, j) => { if (Math.abs(xv - v) < Math.abs(xs[i] - v)) i = j; });
    line.setAttribute("x1", x(xs[i])); line.setAttribute("x2", x(xs[i])); line.setAttribute("visibility", "visible");
    showTip(readout(i), e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => { line.setAttribute("visibility", "hidden"); hideTip(); });
}
// Line families for the legend: the MTA colors trunk lines, and expresses/shuttles share them.
const FAMILY = { "5X": "5", "6X": "6", "7X": "7", FX: "F", GS: "S", FS: "S", H: "S" };

// ------------------------------------------------------------------ hero: the whole system at 8:45
function heroFigure(fig, D) {
  const H = D.hero, routes = D.routes;
  const lon0 = -74.035, lon1 = -73.745, lat0 = 40.565, lat1 = 40.895, k = Math.cos((40.73 * Math.PI) / 180);
  const W = 520, S = W / ((lon1 - lon0) * k), Hh = Math.round((lat1 - lat0) * S);
  const px = (lon) => (lon - lon0) * k * S, py = (lat) => (lat1 - lat) * S;
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img",
    "aria-label": `Map of all ${H.trains.length} subway trains at ${clock(H.t)}, sized by riders aboard` }, fig);
  for (const [r, flat] of H.tracks) {
    let d = "";
    for (let i = 0; i < flat.length; i += 2) d += `${i ? "L" : "M"}${px(flat[i]).toFixed(1)},${py(flat[i + 1]).toFixed(1)}`;
    el("path", { d, fill: "none", stroke: routes[r].color, "stroke-width": 1.3, "stroke-opacity": 0.4, "stroke-linecap": "round" }, svg);
  }
  for (const [name, lon, lat, anchor] of [["MANHATTAN", -74.03, 40.805, "start"], ["BROOKLYN", -73.915, 40.598, "middle"],
    ["QUEENS", -73.80, 40.775, "middle"], ["THE BRONX", -73.80, 40.877, "middle"]]) {
    el("text", { x: px(lon), y: py(lat), "text-anchor": anchor, class: "label-muted",
      style: { fontSize: "11px", letterSpacing: "0.12em", fontWeight: 600 } }, svg, name);
  }
  const rad = (v) => Math.max(1.5, Math.sqrt(v) * 0.3);
  for (const [lon, lat, r, load] of H.trains) {
    const c = el("circle", { cx: px(lon), cy: py(lat), r: rad(load), fill: routes[r].color,
      style: { stroke: "var(--dot-ring)", strokeWidth: 0.8 } }, svg);
    hover(c, [`${fmt.format(load)} riders aboard`, `${routes[r].name} train`], false);
  }
  // side panel: the moment, the number, and the keys
  const side = html("div", { class: "hero-side" }, fig);
  html("div", { class: "hero-time" }, side, clock(H.t));
  html("div", { class: "hero-stat" }, side, `${fmt.format(H.aboard)} riders aboard ${H.trains.length} trains`);
  html("div", {}, side, "Each dot is a train, colored by line. Its area is the number of riders the model puts aboard.");
  const sz = el("svg", { viewBox: "0 0 300 56", role: "img", "aria-label": "Size key", style: { maxWidth: "300px", marginTop: "14px" } }, side);
  let x = 2;
  for (const v of [250, 1000, 2000]) {
    const r = rad(v);
    el("circle", { cx: x + r, cy: 40 - r, r, fill: "none", style: { stroke: "var(--text-secondary)", strokeWidth: 1.2 } }, sz);
    el("text", { x: x + 2 * r + 5, y: 40 - r + 4, class: "tick" }, sz, fmt.format(v));
    x += 2 * r + 48;
  }
  el("text", { x: 2, y: 54, class: "tick" }, sz, "riders aboard");
  const fams = new Map();   // keyed by color, except that all shuttles are one "S"
  for (const r of routes) {
    const fam = FAMILY[r.id] || r.id, base = routes.find((q) => q.id === fam) || r;
    const key = fam === "S" ? "S" : base.color;
    if (!fams.has(key)) fams.set(key, { color: fam === "S" ? (routes.find((q) => q.id === "GS") || r).color : key, names: [] });
    if (!fams.get(key).names.includes(fam)) fams.get(key).names.push(fam);
  }
  legend(side, [...fams.values()].map((f) => ({ kind: "dot", color: f.color, label: f.names.join(" ") })));
  html("div", { style: { marginTop: "14px" } }, side).append(
    document.createTextNode("Hover a dot for its count. "),
    Object.assign(document.createElement("a"), { href: "index.html", textContent: "The live map" }),
    document.createTextNode(" animates the whole day."));
}

// ------------------------------------------------------------------ one turnstile, read like an odometer
function turnstileFigure(fig, D) {
  const T = D.turnstile, reads = T.reads, n = reads.length;
  html("div", { class: "chart-title" }, fig, "One Times Square turnstile's entries register, read every four hours");
  const W = COL, H = 330, colW = W / n;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Turnstile register readings and riders per interval" }, fig);
  // odometer strip
  const digitW = 11.5, digitH = 22;
  reads.forEach(([t, e], i) => {
    const cx = colW * (i + 0.5), s = String(e).padStart(7, "0"), x0 = cx - (digitW * s.length) / 2;
    el("text", { x: cx, y: 14, class: "tick", "text-anchor": "middle" }, svg, clock(t).replace(" ", " "));
    const prev = i ? String(reads[i - 1][1]).padStart(7, "0") : s;
    [...s].forEach((ch, j) => {
      const changed = i > 0 && ch !== prev[j];
      el("rect", { x: x0 + j * digitW, y: 22, width: digitW - 1.5, height: digitH, rx: 2,
        style: { fill: changed ? "var(--s1)" : "var(--code-bg)" } }, svg);
      el("text", { x: x0 + j * digitW + (digitW - 1.5) / 2, y: 22 + 16, "text-anchor": "middle",
        style: { fill: changed ? "#fff" : "var(--text-primary)", fontFamily: "ui-monospace, Menlo, monospace", fontSize: "13px" } }, svg, ch);
    });
    if (i > 0) el("text", { x: colW * i, y: 22 + digitH + 18, class: "label-muted", "text-anchor": "middle" }, svg, `+${fmt.format(e - reads[i - 1][1])}`);
  });
  // riders per interval: entries and exits side by side (differences of the two registers)
  const top = 92, bottom = H - 34, left = 44;
  const dE = [], dX = [];
  for (let i = 0; i < n - 1; i++) { dE.push(reads[i + 1][1] - reads[i][1]); dX.push(reads[i + 1][2] - reads[i][2]); }
  const yMax = Math.max(...dE, ...dX) * 1.12, y = linear(0, yMax, bottom, top + 18);
  gridY(svg, y, niceTicks(0, yMax, 4), left, W - 4, fmt.format);
  el("text", { x: left, y: top + 6, class: "label" }, svg, "Riders per four-hour interval");
  const lg = el("g", { transform: `translate(${left + 210}, ${top - 5})` }, svg);
  el("rect", { x: 0, y: 0, width: 10, height: 10, rx: 2, style: { fill: "var(--s1)" } }, lg);
  el("text", { x: 15, y: 10, class: "tick" }, lg, "entries");
  el("rect", { x: 72, y: 0, width: 10, height: 10, rx: 2, style: { fill: "var(--s2)" } }, lg);
  el("text", { x: 87, y: 10, class: "tick" }, lg, "exits");
  const band = (W - left - 4) / (n - 1), bw = Math.min(26, band / 2 - 4);
  for (let i = 0; i < n - 1; i++) {
    const cx = left + band * (i + 0.5);
    for (const [v, color, dx, name] of [[dE[i], "var(--s1)", -bw - 1, "entries"], [dX[i], "var(--s2)", 1, "exits"]]) {
      const r = el("rect", { x: cx + dx, y: y(v), width: bw, height: Math.max(bottom - y(v), 0.5), rx: 3, style: { fill: color } }, svg);
      hover(r, [`${fmt.format(v)} ${name}`, `${clock(reads[i][0])} – ${clock(reads[i + 1][0])}`]);
    }
    el("text", { x: cx, y: bottom + 17, class: "tick", "text-anchor": "middle" }, svg,
      `${clock(reads[i][0], true)}–${clock(reads[i + 1][0], true)}`);
  }
  caption(fig, `Turnstile ${T.id}.`, "Subtracting readings gives riders per interval. Exits peak in the morning and entries in the evening: people arriving for work, then heading home.");
}

// ------------------------------------------------------------------ exits / entries, a beeswarm
function exitsFigure(fig, D) {
  const rows = D.exit_ratios;
  html("div", { class: "chart-title" }, fig, "Recorded exits as a share of entries, by station complex");
  const W = COL, H = 300, m = { l: 14, r: 14, t: 34, b: 36 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Exits as a share of entries for each station complex" }, fig);
  const xMax = 2.0, x = linear(0, xMax, m.l, W - m.r), mid = (m.t + H - m.b) / 2;
  axisX(svg, x, [0, 0.5, 1, 1.5, 2], H - m.b, (t) => `${Math.round(t * 100)}%`);
  const total = rows.reduce((a, r) => a + r[1], 0), sys = rows.reduce((a, r) => a + r[2] * r[1], 0) / total;
  const rad = (e) => Math.max(1.8, Math.sqrt(e) / 44);
  const jitter = (str) => { let h = 2166136261; for (const ch of str) h = Math.imul(h ^ ch.charCodeAt(0), 16777619); return ((h >>> 0) % 1000) / 1000; };
  const half = (H - m.t - m.b) / 2 - 4;
  // greedy beeswarm: place each dot (largest first) at the smallest vertical offset that doesn't overlap
  const placed = [];
  const items = [...rows].sort((a, b) => b[1] - a[1]).map(([name, e, ratio]) => ({ name, e, ratio, cx: x(Math.min(ratio, xMax)), r: rad(e) }));
  for (const it of items) {
    let best = null;
    for (let k = 0; k < 400; k++) {
      const off = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 1.2;
      if (Math.abs(off) + it.r > half) break;   // band full: overlap rather than spill out of the plot
      if (placed.every((p) => (p.cx - it.cx) ** 2 + (p.cy - (mid + off)) ** 2 >= (p.r + it.r + 0.6) ** 2)) { best = off; break; }
    }
    it.cy = mid + (best ?? (jitter(it.name) - 0.5) * 2 * (half - it.r));
    placed.push(it);
  }
  for (const [v, label, dash] of [[1, "exits = entries", "3 3"], [sys, `systemwide ${Math.round(sys * 100)}%`, null]]) {
    el("line", { x1: x(v), x2: x(v), y1: m.t - 12, y2: H - m.b, style: { stroke: "var(--text-secondary)", strokeDasharray: dash || "none" } }, svg);
    el("text", { x: x(v) + (dash ? 5 : -5), y: m.t - 16, class: "label-muted", "text-anchor": dash ? "start" : "end" }, svg, label);
  }
  for (const it of placed) {
    const c = el("circle", { cx: it.cx, cy: it.cy, r: it.r,
      style: { fill: "var(--s1)", fillOpacity: "var(--swarm-op)", stroke: "var(--surface-1)", strokeWidth: 0.8 } }, svg);
    hover(c, [`${Math.round(it.ratio * 100)}% of entries recorded as exits`, it.name, `${fmt.format(it.e)} entries a day`], false);
  }
  for (const [want, short, above] of [["Dyckman St (A)", "Dyckman St", true], ["Times Sq-42 St", "Times Sq", false],
    ["Aqueduct-N Conduit Av (A)", "Aqueduct–N Conduit Av", true]]) {
    const it = placed.find((p) => p.name.startsWith(want));
    if (!it) continue;
    el("circle", { cx: it.cx, cy: it.cy, r: it.r + 1.5, fill: "none", style: { stroke: "var(--text-primary)", strokeWidth: 1.3 } }, svg);
    const ty = above ? it.cy - it.r - 8 : it.cy + it.r + 16;
    el("text", { x: it.cx, y: ty, class: "label", "text-anchor": it.cx < 90 ? "start" : it.cx > W - 140 ? "end" : "middle" }, svg,
      `${short} ${Math.round(it.ratio * 100)}%`);
  }
  table(fig, ["Station complex", "Entries", "Exits / entries"],
    [...rows].sort((a, b) => b[1] - a[1]).map(([n, e, r]) => [n, fmt.format(e), `${Math.round(r * 100)}%`]));
  caption(fig, null, "Each dot is a station complex, sized by its daily entries. Stations far to the left lose most of their exits through uncounted gates; those right of 100% see more people leave than enter.");
}

// ------------------------------------------------------------------ the tomography analogy
function tomographyFigure(fig) {
  const W = COL, H = 330;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Diagram: a CT scan measures sums along rays; subway counts are row and column sums of an unknown table of trips" }, fig);
  const muted = "var(--text-secondary)";
  // Left: one projection angle through a body with hidden density.
  const cx = 110, cy = 165, R = 74;
  el("text", { x: 130, y: 22, class: "label", "text-anchor": "middle", style: { fontWeight: 600 } }, svg, "CT scan, one angle");
  el("circle", { cx, cy, r: R, fill: "none", style: { stroke: muted, strokeWidth: 1.5 } }, svg);
  const blobs = [[-24, -16, 22], [26, 22, 16], [6, -42, 10]];
  for (const [dx, dy, r] of blobs) el("circle", { cx: cx + dx, cy: cy + dy, r, style: { fill: muted, fillOpacity: 0.28 } }, svg);
  const detX = cx + R + 22;
  for (let i = -6; i <= 6; i++) {
    const yy = cy + i * 11.5;
    el("line", { x1: cx - R - 22, x2: detX - 3, y1: yy, y2: yy, style: { stroke: "var(--s1)", strokeOpacity: 0.3 } }, svg);
    let s = 0;
    for (const [dx, dy, r] of blobs) { const h = Math.abs(yy - (cy + dy)); if (h < r) s += 2 * Math.sqrt(r * r - h * h); }
    el("rect", { x: detX, y: yy - 4, width: 3 + s * 0.85, height: 8, rx: 2, style: { fill: "var(--s1)" } }, svg);
  }
  el("text", { x: 130, y: H - 34, class: "label-muted", "text-anchor": "middle" }, svg, "Measures sums along each ray.");
  el("text", { x: 130, y: H - 16, class: "label-muted", "text-anchor": "middle" }, svg, "Many shapes fit one angle.");
  // Right: an origin x destination table of unknown trips, with row sums (entries) and column sums (exits).
  const n = 5, cell = 32, gx = 400, gy = 70;
  const vals = [[0, 9, 4, 6, 2], [3, 0, 7, 2, 5], [8, 2, 0, 4, 1], [1, 6, 3, 0, 7], [5, 1, 8, 3, 0]];
  const names = ["A", "B", "C", "D", "E"];
  el("text", { x: gx + (n * cell) / 2 + 25, y: 22, class: "label", "text-anchor": "middle", style: { fontWeight: 600 } }, svg, "Subway, two angles");
  el("text", { x: gx + (n * cell) / 2, y: gy - 24, class: "tick", "text-anchor": "middle" }, svg, "to destination");
  el("text", { x: gx - 26, y: gy + (n * cell) / 2, class: "tick", "text-anchor": "middle",
    transform: `rotate(-90 ${gx - 26} ${gy + (n * cell) / 2})` }, svg, "from origin");
  for (let i = 0; i < n; i++) {
    el("text", { x: gx + i * cell + cell / 2, y: gy - 6, class: "tick", "text-anchor": "middle" }, svg, names[i]);
    el("text", { x: gx - 8, y: gy + i * cell + cell / 2 + 4, class: "tick", "text-anchor": "end" }, svg, names[i]);
    for (let j = 0; j < n; j++) {
      const diag = i === j;
      el("rect", { x: gx + j * cell + 1, y: gy + i * cell + 1, width: cell - 2, height: cell - 2, rx: 3,
        style: { fill: diag ? "transparent" : "var(--code-bg)", stroke: "var(--hairline)" } }, svg);
      if (!diag) el("text", { x: gx + j * cell + cell / 2, y: gy + i * cell + cell / 2 + 5, class: "label-muted", "text-anchor": "middle" }, svg, "?");
    }
  }
  const rowSum = vals.map((r) => r.reduce((a, b) => a + b, 0)), colSum = names.map((_, j) => vals.reduce((a, r) => a + r[j], 0));
  const bx = gx + n * cell + 8, by = gy + n * cell + 8;
  rowSum.forEach((v, i) => el("rect", { x: bx, y: gy + i * cell + 9, width: v * 3.2, height: cell - 18, rx: 2, style: { fill: "var(--s1)" } }, svg));
  colSum.forEach((v, j) => el("rect", { x: gx + j * cell + 9, y: by, width: cell - 18, height: v * 2.4, rx: 2, style: { fill: "var(--s2)" } }, svg));
  el("text", { x: bx, y: gy - 8, class: "label-muted" }, svg, "entries = row sums");
  el("text", { x: gx + n * cell + 8, y: by + 16, class: "label-muted" }, svg, "← exits = column sums");
  el("text", { x: gx + (n * cell) / 2 + 25, y: H - 16, class: "label-muted", "text-anchor": "middle" }, svg, "The real table: 424 × 424 stations, for every hour.");
  // the bridge between the halves
  el("text", { x: 300, y: cy - 8, class: "label-muted", "text-anchor": "middle" }, svg, "same");
  el("text", { x: 300, y: cy + 8, class: "label-muted", "text-anchor": "middle" }, svg, "problem");
  caption(fig, "Network tomography.", "Both a CT scan and the subway's counts are sums over things you can't see directly. The timetable adds more angles: exit times depend on where riders came from.");
}

// ------------------------------------------------------------------ Union Square, black hole
function unionSquareFigure(fig, D) {
  const U = D.union_square, hrs = U.hours;
  const wrap = html("div", { class: "two-up" }, fig);
  const left = html("div", {}, wrap), right = html("div", {}, wrap);
  html("div", { class: "chart-title" }, left, "Trips ending at 14 St–Union Sq, by the hour they started");
  legend(left, [
    { label: "Second attempt", color: "var(--s2)" },
    { label: "Final model (peaks near 14k)", color: "var(--s1)" },
    { label: "Everyone entering anywhere", color: "var(--text-secondary)", dashed: true },
  ]);
  const W = 600, H = 290, m = { l: 48, r: 14, t: 12, b: 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Union Square arrivals by hour, second attempt versus final model" }, left);
  const x = linear(hrs[0], hrs[hrs.length - 1], m.l, W - m.r);
  const yMax = Math.max(...U.first_attempt, ...U.system_entries) * 1.05, y = linear(0, yMax, H - m.b, m.t);
  gridY(svg, y, niceTicks(0, yMax, 4), m.l, W - m.r, kfmt);
  axisX(svg, x, hrs.filter((h) => h % 3 === 0), H - m.b, (h) => clock(h * 3600, true));
  const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(hrs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  el("path", { d: path(U.system_entries), fill: "none", style: { stroke: "var(--text-secondary)", strokeWidth: 1.5, strokeDasharray: "4 4" } }, svg);
  el("path", { d: path(U.first_attempt), fill: "none", style: { stroke: "var(--s2)", strokeWidth: 2.2 } }, svg);
  el("path", { d: path(U.final), fill: "none", style: { stroke: "var(--s1)", strokeWidth: 2.2 } }, svg);
  const peak = U.first_attempt.indexOf(Math.max(...U.first_attempt));
  const peakPct = Math.round((100 * U.first_attempt[peak]) / U.system_entries[peak]);
  el("circle", { cx: x(hrs[peak]), cy: y(U.first_attempt[peak]), r: 4, style: { fill: "var(--s2)", stroke: "var(--surface-1)", strokeWidth: 1.5 } }, svg);
  crosshair(svg, hrs, x, m.t, H - m.b, m.l, W - m.r, (i) => [
    `${clock(hrs[i] * 3600)} – ${clock((hrs[i] + 1) * 3600)}`,
    `Second attempt: ${fmt.format(U.first_attempt[i])}`, `Final model: ${fmt.format(U.final[i])}`,
    `Entering anywhere: ${fmt.format(U.system_entries[i])}`]);
  // right: the fix
  html("div", { class: "chart-title" }, right, "The fix: share of exits recorded, by hour");
  html("div", { class: "legend" }, right, "1.0 = a typical hour. Night values rest on few riders.");
  const W2 = 360, m2 = { l: 36, r: 10, t: 12, b: 34 };
  const svg2 = el("svg", { viewBox: `0 0 ${W2} ${H}`, role: "img", "aria-label": "Systemwide exit recording factor by hour" }, right);
  const f = U.recording_factor, lo = Math.min(0.8, ...f), hi = Math.max(1.1, ...f) + 0.02;
  const x2 = linear(hrs[0], hrs[hrs.length - 1], m2.l, W2 - m2.r), y2 = linear(lo, hi, H - m2.b, m2.t);
  gridY(svg2, y2, niceTicks(lo, hi, 4), m2.l, W2 - m2.r, (v) => v.toFixed(1));
  axisX(svg2, x2, hrs.filter((h) => h % 6 === 0), H - m2.b, (h) => clock(h * 3600, true));
  el("line", { x1: m2.l, x2: W2 - m2.r, y1: y2(1), y2: y2(1), style: { stroke: "var(--text-secondary)", strokeDasharray: "3 3" } }, svg2);
  el("path", { d: f.map((v, i) => `${i ? "L" : "M"}${x2(hrs[i]).toFixed(1)},${y2(v).toFixed(1)}`).join(""), fill: "none",
    style: { stroke: "var(--s1)", strokeWidth: 2.2 } }, svg2);
  const am = f.indexOf(Math.min(...f.slice(2, 9)));
  el("text", { x: x2(hrs[am]) + 6, y: y2(f[am]) + 18, class: "label" }, svg2, "morning crowds skip counters");
  crosshair(svg2, hrs, x2, m2.t, H - m2.b, m2.l, W2 - m2.r, (i) => [
    `${f[i].toFixed(2)}× a typical hour`, `${clock(hrs[i] * 3600)} – ${clock((hrs[i] + 1) * 3600)}`]);
  table(fig, ["Hour", "Second attempt", "Final model", "Entering anywhere", "Exit recording"],
    hrs.map((h, i) => [clock(h * 3600), fmt.format(U.first_attempt[i]), fmt.format(U.final[i]), fmt.format(U.system_entries[i]), f[i].toFixed(2)]));
  caption(fig, "Union Square, black hole.", `At its ${clock(hrs[peak] * 3600)} peak the second attempt sent ${peakPct}% of everyone entering the subway to this one station, ${(U.first_attempt_day / 1e6).toFixed(2)} million trips over the day. The final model sends ${fmt.format(U.final_day)}, about as many as enter there.`);
}

// ------------------------------------------------------------------ the 7 line: time-distance diagram + load profile
function mareyFigure(fig, D) {
  const M = D.marey, [t0, t1] = M.window, P = M.profile;
  html("div", { class: "chart-title" }, fig, `Every 7 train from ${clock(t0)} to ${clock(t1)}: line thickness is riders aboard`);
  const wPx = (l) => 0.7 + (6.5 * l) / 1000;
  legend(fig, [
    { kind: "swatch", label: "Manhattan-bound express", color: M.color, width: 4 },
    { kind: "swatch", label: "Manhattan-bound local", color: M.color, width: 4, opacity: 0.45 },
    { kind: "swatch", label: "Flushing-bound (both panels)", color: "var(--text-secondary)", width: 4, opacity: 0.6 },
  ]);
  const W = WIDE, H = 560, m = { l: 150, r: 270, t: 16, b: 36 }, pw = 220;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Time-distance diagram of the 7 line with a load profile" }, fig);
  const yMax = Math.max(...M.stations.map((s) => s[1]));
  const x = linear(t0, t1, m.l, W - m.r), y = linear(yMax, 0, m.t, H - m.b);
  const ticks = []; for (let t = Math.ceil(t0 / 900) * 900; t <= t1; t += 900) ticks.push(t);
  for (const t of ticks) el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "grid" }, svg);
  axisX(svg, x, ticks, H - m.b, (t) => clock(t).replace(" a.m.", ""));
  // station labels, nudged apart where stations are close together
  const labels = M.stations.map(([name, d]) => ({ name, d, y: y(d) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13.5);
  for (const lb of labels) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(lb.d), y2: y(lb.d), class: "grid" }, svg);
    el("text", { x: m.l - 8, y: lb.y + 4, class: "tick", "text-anchor": "end" }, svg, lb.name.replace(" - ", "–"));
  }
  const clip = el("clipPath", { id: "marey-clip" }, el("defs", {}, svg));
  el("rect", { x: m.l, y: 0, width: W - m.r - m.l, height: H }, clip);
  const g = el("g", { "clip-path": "url(#marey-clip)", "stroke-linecap": "round" }, svg);
  const nameAt = (d) => (M.stations.find((s) => Math.abs(s[1] - d) < 1e-6) || ["?"])[0];
  // faint return trips first, then locals, then expresses on top
  const order = [...M.segs].sort((a, b) => a[6] - b[6] || a[5] - b[5]);
  for (const [a, ya, b, yb, load, express, toManhattan] of order) {
    if (b < t0 || a > t1) continue;
    const ln = el("line", { x1: x(a), y1: y(ya), x2: x(b), y2: y(yb), "stroke-width": wPx(load),
      style: toManhattan ? { stroke: M.color, strokeOpacity: express ? 0.95 : 0.45 }
        : { stroke: "var(--text-secondary)", strokeOpacity: 0.3 } }, g);
    hover(ln, () => [`${fmt.format(load)} riders aboard`, ya === yb ? `${express ? "7 express" : "7 local"}, waiting at ${nameAt(ya)}`
      : `${express ? "7 express" : "7 local"}, ${nameAt(ya)} → ${nameAt(yb)}`, `${ya === yb ? "from" : "departs"} ${clock(a)}`], false);
  }
  // annotate the fullest train
  const F = M.fullest, seg = M.segs.find((s) => s[0] === F.t && s[4] === F.load && s[1] !== s[3]);
  if (seg) {
    const fx = (x(seg[0]) + x(seg[2])) / 2, fy = (y(seg[1]) + y(seg[3])) / 2;
    el("circle", { cx: fx, cy: fy, r: 9, fill: "none", style: { stroke: "var(--text-primary)", strokeWidth: 1.5 } }, svg);
    el("line", { x1: fx + 9, y1: fy - 4, x2: fx + 40, y2: fy - 30, style: { stroke: "var(--text-primary)" } }, svg);
    el("rect", { x: fx + 40, y: fy - 50, width: 200, height: 52, rx: 4, style: { fill: "var(--surface-1)", fillOpacity: 0.92 } }, svg);
    el("text", { x: fx + 46, y: fy - 35, class: "label", style: { fontWeight: 600 } }, svg, `${fmt.format(F.load)} riders, the fullest`);
    el("text", { x: fx + 46, y: fy - 19, class: "label-muted" }, svg, `7 express, ${F.from.replace(" - ", "–")} →`);
    el("text", { x: fx + 46, y: fy - 4, class: "label-muted" }, svg, F.to.replace(" - ", "–"));
  }
  // load profile: riders carried past each stretch in the hour, by direction
  const px0 = W - pw, xp = linear(0, Math.max(...P.toward_manhattan) * 1.1, px0, W - 10);
  el("text", { x: px0, y: m.t - 2, class: "label" }, svg, "Riders past each point, per hour");
  for (const t of niceTicks(0, Math.max(...P.toward_manhattan) * 1.1, 3)) {
    el("line", { x1: xp(t), x2: xp(t), y1: m.t + 6, y2: H - m.b, class: t === 0 ? "axis" : "grid" }, svg);
    el("text", { x: xp(t), y: H - m.b + 17, class: "tick", "text-anchor": "middle" }, svg, kfmt(t));
  }
  for (let i = 0; i < P.y.length - 1; i++) {
    const yTop = y(P.y[i + 1]), yBot = y(P.y[i]), hgt = Math.max(yBot - yTop - 1, 1);
    for (const [vals, name, frac] of [[P.toward_manhattan, "Manhattan-bound", 1], [P.toward_flushing, "Flushing-bound", 0.34]]) {
      const hh = Math.max(hgt * frac, 1);
      const r = el("rect", { x: px0, y: yTop + 0.5 + (hgt - hh) / 2, width: Math.max(xp(vals[i]) - px0, 0.5), height: hh,
        style: frac < 1 ? { fill: "var(--text-secondary)" } : { fill: M.color, fillOpacity: 0.8 } }, svg);
      hover(r, [`${fmt.format(vals[i])} riders in the hour`, `${name}, ${nameAt(P.y[i + 1])} – ${nameAt(P.y[i])}`]);
    }
  }
  table(fig, ["Between", "Manhattan-bound riders/hour", "Flushing-bound riders/hour"],
    P.y.slice(0, -1).map((yy, i) => [`${nameAt(P.y[i + 1])} – ${nameAt(yy)}`, fmt.format(P.toward_manhattan[i]), fmt.format(P.toward_flushing[i])]).reverse());
  caption(fig, "The 7 line in the morning rush.", "Left: each line is one train, sloping down toward Manhattan; hover for its load. Right: the same hour summed into riders carried past each point.");
}

// ------------------------------------------------------------------ validation and calibration
function validationFigure(fig, D) {
  const F = D.facts;
  html("div", { class: "chart-title" }, fig, "How closely each station's destinations match the MTA's estimates");
  const steps = [["null", "Station size only", "var(--o1)"], ["prior", "+ travel time", "var(--o2)"], ["fitted", "+ fit to exit counts", "var(--o3)"]];
  legend(fig, steps.map(([, label, color]) => ({ kind: "dot", label, color })));
  const W = COL, H = 160, m = { l: 120, r: 20, t: 18, b: 42 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Overlap scores for three model stages" }, fig);
  const x = linear(0.45, 0.8, m.l, W - m.r);
  axisX(svg, x, [0.5, 0.6, 0.7, 0.8], H - m.b, (t) => t.toFixed(1), "overlap, 1 = identical");
  [["Whole day", F.overlap_day], ["Morning peak", F.overlap_am]].forEach(([label, v], i) => {
    const yy = m.t + 22 + i * 48;
    el("text", { x: m.l - 12, y: yy + 4, class: "label", "text-anchor": "end" }, svg, label);
    el("line", { x1: x(v.null), x2: x(v.fitted), y1: yy, y2: yy, class: "axis" }, svg);
    for (const [key, name, color] of steps) {
      const c = el("circle", { cx: x(v[key]), cy: yy, r: 7, style: { fill: color, stroke: "var(--surface-1)", strokeWidth: 2 } }, svg);
      hover(c, [`${v[key].toFixed(3)} overlap`, `${name}, ${label.toLowerCase()}`]);
      el("text", { x: x(v[key]), y: yy - 12, class: "tick", "text-anchor": "middle" }, svg, v[key].toFixed(2));
    }
  });
  table(fig, ["", ...steps.map((s) => s[1])], [["Whole day", ...steps.map(([k]) => F.overlap_day[k].toFixed(3))],
    ["Morning peak", ...steps.map(([k]) => F.overlap_am[k].toFixed(3))]]);
  caption(fig, "Checked against MTA estimates for February 2023 Wednesdays.", "Each step of the model moves its destinations closer to the MTA's.");
}

function calibrationFigure(fig, D) {
  const C = D.facts.calibration;
  html("div", { class: "chart-title" }, fig, "The borrowed number: how long trips take");
  legend(fig, [
    { kind: "line", label: "MTA 2023 (the target)", color: "var(--text-primary)" },
    { kind: "dot", label: "model before calibration", color: "var(--s2)" },
    { kind: "dot", label: "after", color: "var(--s1)" },
  ]);
  const W = COL, H = 230, m = { l: 92, r: 20, t: 10, b: 42 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Trip-length percentiles: MTA target, model before and after calibration" }, fig);
  const x = linear(0, 60, m.l, W - m.r);
  axisX(svg, x, [0, 15, 30, 45, 60], H - m.b, (t) => `${t}`, "minutes door to door");
  const names = { p10: "10th percentile", p25: "25th", p50: "median", p75: "75th", p90: "90th", mean: "mean" };
  C.labels.forEach((lab, i) => {
    const yy = m.t + 14 + i * 29;
    el("text", { x: m.l - 10, y: yy + 4, class: "tick", "text-anchor": "end" }, svg, names[lab]);
    el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: "grid" }, svg);
    for (const [v, color, nm, dy] of [[C.before[i], "var(--s2)", "before calibration", -5], [C.after[i], "var(--s1)", "after calibration", 5]]) {
      const c = el("circle", { cx: x(v), cy: yy + dy, r: 5.5, style: { fill: color, stroke: "var(--surface-1)", strokeWidth: 1.5 } }, svg);
      hover(c, [`${v.toFixed(1)} minutes`, `${names[lab]}, ${nm}`, `MTA 2023: ${C.target[i].toFixed(1)} minutes`]);
    }
    el("line", { x1: x(C.target[i]), x2: x(C.target[i]), y1: yy - 12, y2: yy + 12, style: { stroke: "var(--text-primary)", strokeWidth: 2.5 } }, svg);
  });
  table(fig, ["", "MTA 2023", "before", "after"], C.labels.map((lab, i) => [names[lab], C.target[i].toFixed(1), C.before[i].toFixed(1), C.after[i].toFixed(1)]));
  caption(fig, null, "Each row is one point of the distribution of door-to-door trip times. After calibration, the model's points sit on the MTA's ticks; this panel is a fit, not a test.");
}

// ------------------------------------------------------------------ facts in the text
function fillFacts(D) {
  const F = D.facts, H = D.hero, M = D.marey;
  const values = {
    hero_aboard_round: fmt.format(Math.round(H.aboard / 1000) * 1000),
    n_complexes: fmt.format(F.n_complexes), xw_max_dev_pct: `${(F.xw_max_dev * 100).toFixed(1)}%`,
    xw_within5: fmt.format(F.xw_within5), n_exits_gt_entries: fmt.format(F.n_exits_gt_entries),
    n_trips: fmt.format(F.n_trips), n_hops_round: fmt.format(Math.round(F.n_hops / 5000) * 5000),
    n_unknowns_round: `${(F.n_unknowns / 1e6).toFixed(1)} million`, n_exit_obs: fmt.format(F.n_exit_obs),
    n_pulls: fmt.format(F.n_pulls), us_first_day: `${(D.union_square.first_attempt_day / 1e6).toFixed(2)} million`,
    marey_fullest: fmt.format(M.fullest.load),
    ov_day_null: F.overlap_day.null.toFixed(2), ov_day_prior: F.overlap_day.prior.toFixed(2), ov_day_fitted: F.overlap_day.fitted.toFixed(2),
    ov_am_null: F.overlap_am.null.toFixed(2), ov_am_prior: F.overlap_am.prior.toFixed(2), ov_am_fitted: F.overlap_am.fitted.toFixed(2),
  };
  for (const node of document.querySelectorAll("[data-fact]")) {
    const v = values[node.dataset.fact];
    if (v !== undefined) node.textContent = v;
  }
}

async function main() {
  const D = await (await fetch("data/writeup.json")).json();
  fillFacts(D);
  const figs = {
    "fig-hero": heroFigure, "fig-turnstile": turnstileFigure, "fig-exits": exitsFigure, "fig-tomo": tomographyFigure,
    "fig-union": unionSquareFigure, "fig-marey": mareyFigure, "fig-validation": validationFigure,
    "fig-calibration": calibrationFigure,
  };
  for (const [id, draw] of Object.entries(figs)) {
    const fig = document.getElementById(id);
    try { draw(fig, D); } catch (err) { console.error(id, err); html("p", { class: "label-muted" }, fig, `(figure failed to draw: ${err.message})`); }
  }
}
main().catch((err) => console.error(err));
