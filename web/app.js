// Animated map of modeled riders aboard every NYC subway train over one day.
// Data: data/day.json (written by `subway export`). Rendering: MapLibre + deck.gl.

const { MapboxOverlay, TripsLayer, ScatterplotLayer, PathLayer } = deck;

const STYLES = {
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
};
const SURFACE = { dark: [26, 26, 25], light: [252, 252, 251] };
const STATION_INK = { dark: [195, 194, 183], light: [82, 81, 78] };
// Sequential blue ramp (steps 100..700), light -> dark. In dark mode the anchor flips
// so that more riders read as brighter against the dark basemap.
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"].map(hexToRgb);
const RAMP = { light: BLUE.slice(3), dark: BLUE.slice(0, 11).reverse() }; // light: 250..700, dark: 600..100
const RAMP_MAX = 2000;            // riders at the saturated end of the color ramp
const TRAIL_S = 150;              // trail length behind each train, seconds
const BUCKET_S = 300;             // index of which trips run when
const TRAIL_BUCKET_S = 60;        // trail fragments are cut per minute of sim time
const SIZE_LEGEND = [250, 1000, 2000];

const $ = (id) => document.getElementById(id);
// ?perf: show a frame-rate readout and expose window.__subway.bench() for profiling.
const PERF = new URLSearchParams(location.search).has("perf");
const perf = { frames: 0, since: performance.now(), jsMs: 0, hud: null };
const state = {
  t: 6.5 * 3600, playing: true, speed: 180, colorBy: "line", themePref: "auto",
  zoom: 11, trains: [], hovered: null, lastTable: 0,
};

function hexToRgb(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function luminance([r, g, b]) {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function theme() {
  if (state.themePref !== "auto") return state.themePref;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function rampColor(v) {
  const r = RAMP[theme()];
  const x = Math.min(Math.max(v / RAMP_MAX, 0), 1) * (r.length - 1);
  const i = Math.min(Math.floor(x), r.length - 2), f = x - i;
  return r[i].map((c, k) => Math.round(c + f * (r[i + 1][k] - c)));
}
// Marker radius in px is sqrt(riders) * scale, so area is proportional to riders. Markers
// grow with zoom, but more slowly than the map, so the citywide view isn't a blob.
function radiusScale(zoom) {
  return 0.27 * 2 ** ((Math.min(Math.max(zoom, 9), 16) - 11) * 0.5);
}
function clock(t) {
  const day = Math.floor(t / 86400), s = ((t % 86400) + 86400) % 86400;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return { hm: `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")}`, ampm: h < 12 ? "AM" : "PM", nextDay: day > 0 };
}
const fmt = new Intl.NumberFormat("en-US");

// ---------------------------------------------------------------- data preparation
function prepare(data) {
  const routes = data.routes.map((r) => ({ ...r, rgb: hexToRgb(r.color) }));
  const segs = data.segs.map((flat) => {
    const n = flat.length / 2, frac = new Float64Array(n), k = Math.cos((40.7 * Math.PI) / 180);
    for (let i = 1; i < n; i++) {
      const dx = (flat[2 * i] - flat[2 * i - 2]) * k, dy = flat[2 * i + 1] - flat[2 * i - 1];
      frac[i] = frac[i - 1] + Math.hypot(dx, dy);
    }
    for (let i = 1; i < n; i++) frac[i] = frac[n - 1] > 0 ? frac[i] / frac[n - 1] : i / (n - 1);
    return { flat, frac };
  });
  const segRoute = new Int32Array(segs.length).fill(-1);
  const trips = data.trips.map((tr) => {
    const path = [], ts = [];
    for (let k = 0; k < tr.g.length; k++) {
      const { flat, frac } = segs[tr.g[k]], d = tr.d[k], a = tr.a[k];
      if (segRoute[tr.g[k]] < 0) segRoute[tr.g[k]] = tr.r;
      // Consecutive segments share the station vertex; keep it twice only if the
      // train dwells there, so the trail pauses rather than jumps.
      const skipFirst = k > 0 && ts[ts.length - 1] === d;
      for (let i = skipFirst ? 1 : 0; i < frac.length; i++) {
        path.push(flat[2 * i], flat[2 * i + 1]);
        ts.push(d + frac[i] * (a - d));
      }
    }
    return {
      r: tr.r, h: tr.h, s: tr.s, d: tr.d, a: tr.a, l: tr.l,
      path: Float64Array.from(path), ts: Float64Array.from(ts),
      start: tr.d[0], end: tr.a[tr.a.length - 1],
    };
  });
  const tMin = Math.min(...trips.map((t) => t.start)), tMax = Math.max(...trips.map((t) => t.end));
  const buckets = Array.from({ length: Math.ceil((tMax - tMin) / BUCKET_S) + 1 }, () => []);
  for (const tr of trips) {
    for (let b = Math.floor((tr.start - tMin) / BUCKET_S); b <= Math.floor((tr.end - tMin) / BUCKET_S); b++) buckets[b].push(tr);
  }
  const tracks = segs.map((s, i) => ({ path: s.flat, color: routes[Math.max(segRoute[i], 0)].rgb }))
    .filter((_, i) => segRoute[i] >= 0);
  return { ...data, routes, trips, tracks, buckets, tMin };
}

function upperBound(arr, x) { // first index with arr[i] > x
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

// Trail fragments for the minute containing t: for each trip running then, the part of
// its path timestamped within [minute start - TRAIL_S, minute end], as zero-copy
// subarray views. Drawing whole-day paths for every trip meant ~4M vertices per frame;
// this is ~550 short fragments, and the layer's data only changes once per sim minute.
const trailCache = new Map();
function trailsAt(D, t) {
  const b = Math.floor((t - D.tMin) / TRAIL_BUCKET_S);
  let data = trailCache.get(b);
  if (data) return data;
  const hi = D.tMin + (b + 1) * TRAIL_BUCKET_S, lo = hi - TRAIL_BUCKET_S - TRAIL_S;
  const seen = new Set();
  data = [];
  for (let k = Math.floor((lo - D.tMin) / BUCKET_S); k <= Math.floor((hi - D.tMin) / BUCKET_S); k++) {
    for (const tr of D.buckets[k] || []) {
      if (seen.has(tr) || tr.end < lo || tr.start > hi) continue;
      seen.add(tr);
      const i0 = Math.max(upperBound(tr.ts, lo) - 1, 0);
      const i1 = Math.min(upperBound(tr.ts, hi), tr.ts.length - 1);
      if (i1 > i0) data.push({ r: tr.r, path: tr.path.subarray(2 * i0, 2 * i1 + 2), ts: tr.ts.subarray(i0, i1 + 1) });
    }
  }
  trailCache.set(b, data);
  if (trailCache.size > 32) trailCache.delete(trailCache.keys().next().value);
  return data;
}

function trainsAt(D, t) {
  const b = Math.floor((t - D.tMin) / BUCKET_S);
  const out = [];
  for (const tr of D.buckets[b] || []) {
    if (t < tr.start || t > tr.end) continue;
    const i = Math.min(Math.max(upperBound(tr.ts, t) - 1, 0), tr.ts.length - 2);
    const t0 = tr.ts[i], t1 = tr.ts[i + 1], f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    const pos = [tr.path[2 * i] + f * (tr.path[2 * i + 2] - tr.path[2 * i]),
      tr.path[2 * i + 1] + f * (tr.path[2 * i + 3] - tr.path[2 * i + 1])];
    const k = Math.min(Math.max(upperBound(tr.d, t) - 1, 0), tr.d.length - 1);
    out.push({ trip: tr, pos, k, load: tr.l[k] });
  }
  out.sort((a, b) => b.load - a.load); // draw big markers first so small ones stay visible
  return out;
}

// ---------------------------------------------------------------- map + layers
function layers(D) {
  const th = theme();
  const surface = SURFACE[th];
  const lineMode = state.colorBy === "line";
  return [
    new PathLayer({
      id: "tracks", data: D.tracks, positionFormat: "XY", _pathType: "open", getPath: (d) => d.path,
      getColor: (d) => [...d.color, th === "dark" ? 90 : 110], widthUnits: "pixels", getWidth: 1.5,
      updateTriggers: { getColor: [th] },
    }),
    new TripsLayer({
      id: "trails", data: trailsAt(D, state.t), positionFormat: "XY", _pathType: "open",
      getPath: (d) => d.path, getTimestamps: (d) => d.ts,
      getColor: (d) => D.routes[d.r].rgb, widthMinPixels: 2, capRounded: true, jointRounded: true,
      trailLength: TRAIL_S, fadeTrail: true, currentTime: state.t, opacity: lineMode ? 0.55 : 0.25,
    }),
    new ScatterplotLayer({
      id: "stations", data: D.stations, getPosition: (d) => d.p, radiusUnits: "pixels", getRadius: 2,
      getFillColor: [...STATION_INK[th], 200], pickable: true, updateTriggers: { getFillColor: [th] },
    }),
    new ScatterplotLayer({
      id: "trains", data: state.trains, getPosition: (d) => d.pos,
      radiusUnits: "pixels", getRadius: (d) => Math.sqrt(Math.max(d.load, 1)),
      radiusScale: radiusScale(state.zoom), radiusMinPixels: 2.5,
      getFillColor: (d) => (lineMode ? D.routes[d.trip.r].rgb : rampColor(d.load)),
      stroked: true, getLineColor: [...surface, 230], lineWidthUnits: "pixels", getLineWidth: 1.5,
      pickable: true, updateTriggers: { getFillColor: [state.colorBy, th] },
    }),
  ];
}

// ---------------------------------------------------------------- side panel, legend, tooltip
function bullet(route) {
  const el = document.createElement("span");
  el.className = "bullet" + (luminance(route.rgb) > 0.45 ? " dark-text" : "");
  el.style.background = route.color;
  el.textContent = route.name === "SIR" ? "SI" : route.name.replace("X", "");
  el.title = route.long;
  return el;
}

function renderLegend(D) {
  const box = $("legend-color");
  box.replaceChildren();
  if (state.colorBy === "line") {
    const wrap = document.createElement("div");
    wrap.className = "legend-routes";
    const seen = new Set();
    for (const r of D.routes) {
      const key = r.name.replace("X", "");
      if (seen.has(key)) continue;
      seen.add(key);
      wrap.append(bullet(r));
    }
    box.append(wrap);
  } else {
    const ramp = document.createElement("div");
    ramp.className = "ramp";
    const stops = Array.from({ length: 9 }, (_, i) => `rgb(${rampColor((i / 8) * RAMP_MAX).join(",")}) ${(i / 8) * 100}%`);
    ramp.style.background = `linear-gradient(to right, ${stops.join(",")})`;
    const labels = document.createElement("div");
    labels.className = "ramp-labels";
    for (const s of ["0", "1,000", "2,000+ riders"]) { const e = document.createElement("span"); e.textContent = s; labels.append(e); }
    box.append(ramp, labels);
  }
  renderSizeLegend();
}

function renderSizeLegend() {
  const box = $("legend-size");
  box.replaceChildren();
  const s = radiusScale(state.zoom);
  for (const v of SIZE_LEGEND) {
    const r = Math.max(Math.sqrt(v) * s, 2.5), size = 2 * r + 4;
    const cell = document.createElement("div");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", size); svg.setAttribute("height", size);
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", size / 2); c.setAttribute("cy", size / 2); c.setAttribute("r", r);
    svg.append(c);
    const label = document.createElement("span");
    label.textContent = `${fmt.format(v)} riders`;
    cell.append(svg, label);
    box.append(cell);
  }
}

function stationName(D, s) { return D.stations[s].n; }

function renderSide(D) {
  const total = state.trains.reduce((a, d) => a + d.load, 0);
  $("stat-aboard").textContent = fmt.format(Math.round(total));
  $("stat-trains").textContent = fmt.format(state.trains.length);
  const body = $("top-body");
  body.replaceChildren();
  for (const d of state.trains.slice(0, 8)) {
    const tr = d.trip, route = D.routes[tr.r];
    const row = document.createElement("tr");
    const c1 = document.createElement("td"); c1.append(bullet(route));
    const c2 = document.createElement("td");
    const head = document.createElement("div"); head.textContent = `to ${tr.h}`;
    const where = document.createElement("div"); where.className = "where";
    where.textContent = `${stationName(D, tr.s[d.k])} → ${stationName(D, tr.s[d.k + 1])}`;
    c2.append(head, where);
    const c3 = document.createElement("td"); c3.className = "num"; c3.textContent = fmt.format(d.load);
    row.append(c1, c2, c3);
    body.append(row);
  }
}

function showTooltip(D, x, y) {
  const tip = $("tooltip");
  const h = state.hovered;
  if (!h) { tip.style.display = "none"; return; }
  tip.replaceChildren();
  if (h.kind === "train") {
    const d = state.trains.find((t) => t.trip === h.trip);
    if (!d) { state.hovered = null; tip.style.display = "none"; return; }
    const tr = d.trip, route = D.routes[tr.r];
    const v = document.createElement("div"); v.className = "value"; v.textContent = `${fmt.format(d.load)} riders aboard`;
    const r1 = document.createElement("div"); r1.className = "row";
    const name = document.createElement("span"); name.textContent = `to ${tr.h}`;
    r1.append(bullet(route), name);
    const r2 = document.createElement("div"); r2.className = "row secondary";
    r2.textContent = `${stationName(D, tr.s[d.k])} → ${stationName(D, tr.s[d.k + 1])}`;
    const c = clock(tr.d[d.k]);
    const r3 = document.createElement("div"); r3.className = "row secondary";
    r3.textContent = `Scheduled departure ${c.hm} ${c.ampm}`;
    tip.append(v, r1, r2, r3);
  } else {
    const st = h.station, cx = D.complexes[st.c];
    const hour = Math.floor((state.t - D.t0) / 3600);
    const e = cx && cx.e[hour] !== undefined ? cx.e[hour] : null;
    const v = document.createElement("div"); v.className = "value";
    v.textContent = e === null ? st.n : `${fmt.format(e)} entries this hour`;
    const r1 = document.createElement("div"); r1.className = "row secondary";
    r1.textContent = cx ? cx.n : st.n;
    tip.append(v, r1);
  }
  if (x !== undefined) { tip.style.left = `${x + 14}px`; tip.style.top = `${y + 14}px`; }
  tip.style.display = "block";
}

// ---------------------------------------------------------------- timeline (area chart + scrubber)
function timeline(D) {
  const svg = $("timeline"), NS = "http://www.w3.org/2000/svg";
  const n = D.aboard.length, maxY = Math.max(...D.aboard) * 1.08;
  let W = 0, H = 0, PAD_B = 18, PAD_T = 16, hoverX = null;
  const x = (min) => (min / n) * W;
  const y = (v) => PAD_T + (1 - v / maxY) * (H - PAD_T - PAD_B);
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  let cursor, dot, hoverLine, hoverText;
  function draw() {
    const r = svg.getBoundingClientRect(); W = r.width; H = r.height;
    svg.replaceChildren();
    for (const v of [100000, 200000]) {
      svg.append(el("line", { class: "grid", x1: 0, x2: W, y1: y(v), y2: y(v) }));
      svg.append(el("text", { class: "tick", x: 2, y: y(v) - 3 }, `${v / 1000}k`));
    }
    let dA = `M0,${y(0)}`, dL = "";
    for (let i = 0; i < n; i++) { dA += `L${x(i)},${y(D.aboard[i])}`; dL += `${i ? "L" : "M"}${x(i)},${y(D.aboard[i])}`; }
    svg.append(el("path", { class: "area", d: `${dA}L${x(n - 1)},${y(0)}Z` }));
    svg.append(el("path", { class: "line", d: dL }));
    for (let hh = 0; hh <= 24; hh += 3) {
      const c = clock(D.t0 + hh * 3600);
      const label = c.hm.replace(":00", "") + (c.ampm === "AM" ? "a" : "p");
      svg.append(el("text", { class: "tick", x: Math.min(Math.max(x(hh * 60), 8), W - 12), y: H - 4, "text-anchor": "middle" },
        hh === 9 ? "noon" : label.replace("12p", "noon").replace("12a", "midnight")));
    }
    hoverLine = el("line", { class: "hover", y1: PAD_T, y2: H - PAD_B, visibility: "hidden" });
    hoverText = el("text", { class: "tick", y: 11, visibility: "hidden" });
    cursor = el("line", { class: "cursor", y1: PAD_T - 4, y2: H - PAD_B });
    dot = el("circle", { class: "dot", r: 4 });
    svg.append(hoverLine, hoverText, cursor, dot);
    update();
  }
  function update() {
    const min = Math.min(Math.max((state.t - D.t0) / 60, 0), n - 1);
    cursor.setAttribute("x1", x(min)); cursor.setAttribute("x2", x(min));
    dot.setAttribute("cx", x(min)); dot.setAttribute("cy", y(D.aboard[Math.round(min)]));
    svg.setAttribute("aria-valuenow", Math.round(min));
    const c = clock(state.t);
    svg.setAttribute("aria-valuetext", `${c.hm} ${c.ampm}`);
    if (hoverX !== null) {
      const hm = Math.min(Math.max(Math.round((hoverX / W) * n), 0), n - 1), hc = clock(D.t0 + hm * 60);
      hoverLine.setAttribute("x1", hoverX); hoverLine.setAttribute("x2", hoverX);
      hoverText.textContent = `${hc.hm} ${hc.ampm} · ${fmt.format(D.aboard[hm])} aboard`;
      hoverText.setAttribute("x", Math.min(Math.max(hoverX + 6, 0), W - 170));
      hoverLine.setAttribute("visibility", "visible"); hoverText.setAttribute("visibility", "visible");
    } else {
      hoverLine.setAttribute("visibility", "hidden"); hoverText.setAttribute("visibility", "hidden");
    }
  }
  const timeAt = (ev) => D.t0 + Math.min(Math.max((ev.clientX - svg.getBoundingClientRect().left) / W, 0), 1) * (D.t1 - D.t0 - 60);
  let dragging = false;
  svg.addEventListener("pointerdown", (ev) => { dragging = true; svg.setPointerCapture(ev.pointerId); seek(timeAt(ev)); });
  svg.addEventListener("pointermove", (ev) => {
    hoverX = ev.clientX - svg.getBoundingClientRect().left;
    if (dragging) seek(timeAt(ev)); else update();
  });
  svg.addEventListener("pointerup", () => { dragging = false; });
  svg.addEventListener("pointerleave", () => { hoverX = null; update(); });
  svg.addEventListener("keydown", (ev) => {
    const step = (ev.shiftKey ? 60 : 5) * 60;
    if (ev.key === "ArrowRight") { seek(state.t + step); ev.preventDefault(); }
    if (ev.key === "ArrowLeft") { seek(state.t - step); ev.preventDefault(); }
  });
  addEventListener("resize", draw);
  draw();
  return { update };
}

// ---------------------------------------------------------------- main
let D, overlay, map, tl;
function seek(t) {
  state.t = Math.min(Math.max(t, D.t0), D.t1 - 60);
  frame(true);
}
function frame(force = false) {
  const t0 = PERF ? performance.now() : 0;
  state.trains = trainsAt(D, state.t);
  overlay.setProps({ layers: layers(D) });
  const c = clock(state.t);
  const clk = $("clock");
  clk.textContent = c.hm;
  const small = document.createElement("small");
  small.textContent = ` ${c.ampm}${c.nextDay ? " (Thu)" : ""}`;
  clk.append(small);
  tl.update();
  const now = performance.now();
  if (force || now - state.lastTable > 300) { renderSide(D); state.lastTable = now; }
  if (state.hovered) showTooltip(D);
  if (PERF) perf.jsMs += performance.now() - t0;
}

function perfTick(now) {
  perf.frames++;
  if (now - perf.since < 1000) return;
  const fps = (perf.frames * 1000) / (now - perf.since);
  const m = overlay && overlay._deck && overlay._deck.metrics;
  perf.hud.textContent = `${fps.toFixed(0)} fps · js ${(perf.jsMs / perf.frames).toFixed(1)} ms/frame` +
    (m ? ` · deck cpu ${m.cpuTimePerFrame.toFixed(1)} ms · gpu ${m.gpuTimePerFrame.toFixed(1)} ms` : "");
  perf.frames = 0; perf.jsMs = 0; perf.since = now;
}

// Synchronous benchmark: n frames at successive times, each fully drawn and waited on
// (readPixels forces the GPU to finish). Only meaningful in a visible tab: Chrome
// throttles GPU work in background tabs (an empty clear + readPixels takes ~0.5 s there).
function bench(n = 20, t = 8.5 * 3600, step = 20) {
  const deckgl = overlay._deck;
  const gl = deckgl.canvas.getContext("webgl2") || deckgl.canvas.getContext("webgl");
  const px = new Uint8Array(4);
  const saved = state.t;
  let js = 0;
  const start = performance.now();
  for (let i = 0; i < n; i++) {
    state.t = t + i * step;
    const a = performance.now();
    frame(true);
    js += performance.now() - a;
    deckgl.redraw("bench");
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }
  const total = (performance.now() - start) / n;
  state.t = saved;
  frame(true);
  return { msPerFrame: +total.toFixed(1), jsMsPerFrame: +(js / n).toFixed(1), impliedFps: +(1000 / total).toFixed(1),
    canvas: [deckgl.canvas.width, deckgl.canvas.height] };
}

function setPlaying(p) {
  state.playing = p;
  $("play").textContent = p ? "❚❚ Pause" : "▶ Play";
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
}

async function main() {
  const raw = await (await fetch("data/day.json")).json();
  D = prepare(raw);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { state.t = 8.5 * 3600; state.playing = false; }

  // Frame the whole system in the part of the map the panels leave uncovered.
  const narrow = innerWidth < 900;
  map = new maplibregl.Map({
    container: "map", style: STYLES[theme()], minZoom: 9, maxZoom: 17,
    bounds: [[-74.03, 40.64], [-73.83, 40.83]],  // the dense core; zoom out for the rest
    fitBoundsOptions: { padding: narrow ? 20 : { top: 20, left: 60, right: 350, bottom: 175 } },
    attributionControl: { customAttribution: "Subway data: MTA via data.ny.gov · Model: this project" },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  overlay = new MapboxOverlay({
    interleaved: false, layers: [], pickingRadius: 12,
    onHover: (info) => {
      if (info.object && info.layer.id === "trains") state.hovered = { kind: "train", trip: info.object.trip };
      else if (info.object && info.layer.id === "stations") state.hovered = { kind: "station", station: info.object };
      else state.hovered = null;
      showTooltip(D, info.x, info.y);
    },
  });
  map.addControl(overlay);
  state.zoom = map.getZoom();
  map.on("zoom", () => { state.zoom = map.getZoom(); renderSizeLegend(); if (!state.playing) frame(); });

  tl = timeline(D);
  renderLegend(D);
  setPlaying(state.playing);
  $("loading").remove();

  $("play").onclick = () => setPlaying(!state.playing);
  $("speed").onchange = (e) => { state.speed = +e.target.value; };
  for (const b of document.querySelectorAll("[data-color]")) {
    b.onclick = () => {
      state.colorBy = b.dataset.color;
      for (const o of document.querySelectorAll("[data-color]")) o.setAttribute("aria-pressed", String(o === b));
      renderLegend(D); frame(true);
    };
  }
  const applyTheme = () => {
    if (state.themePref === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = state.themePref;
    $("theme").textContent = `Theme: ${state.themePref}`;
    map.setStyle(STYLES[theme()]);
    renderLegend(D); frame(true);
  };
  $("theme").onclick = () => {
    state.themePref = { auto: "dark", dark: "light", light: "auto" }[state.themePref];
    applyTheme();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.themePref === "auto") applyTheme(); });
  addEventListener("keydown", (ev) => {
    if (ev.key === " " && !["BUTTON", "SELECT"].includes(document.activeElement.tagName)) { setPlaying(!state.playing); ev.preventDefault(); }
  });

  if (PERF) {
    perf.hud = document.createElement("div");
    perf.hud.style.cssText = "position:absolute;left:60px;top:12px;z-index:5;font:12px ui-monospace,monospace;" +
      "background:var(--panel);border:1px solid var(--hairline);border-radius:6px;padding:4px 8px";
    document.querySelector("main").append(perf.hud);
    window.__subway = { state, D, overlay, frame, bench };
  }

  let last = performance.now();
  const loop = (now) => {
    if (PERF) perfTick(now);
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (state.playing) {
      state.t += dt * state.speed;
      if (state.t >= D.t1 - 60) state.t = D.t0;
      frame();
    }
    requestAnimationFrame(loop);
  };
  frame(true);
  requestAnimationFrame(loop);
}

main().catch((err) => {
  const el = $("loading");
  if (el) el.textContent = `Couldn't load the day: ${err.message}. Run \`subway export\` and serve web/ over HTTP.`;
  console.error(err);
});
