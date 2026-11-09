/* core.js - shared shell, API client, formatting, theming and chart helpers (no build step). */

// ---------------------------------------------------------------------------------------------
// tiny utilities
// ---------------------------------------------------------------------------------------------
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const nfUSD0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const nfUSD2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const nfNum = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
export const fmt = {
  usd: (v, d = 0) => (v == null || Number.isNaN(v) ? "-" : (d ? nfUSD2 : nfUSD0).format(v)),
  num: (v, d = 1) => (v == null || Number.isNaN(v) ? "-" : new Intl.NumberFormat("en-US", { maximumFractionDigits: d }).format(v)),
  pct: (v, d = 2) => (v == null || Number.isNaN(v) ? "-" : `${Number(v).toFixed(d)}%`),
  signed: (v, d = 1) => (v == null ? "-" : `${v >= 0 ? "+" : ""}${Number(v).toFixed(d)}%`),
  date: (s) => (s ? new Date(`${s}T00:00:00`).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" }) : "-"),
  compact: (v) => (v == null ? "-" : new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v)),
  ms: (v) => (v == null ? "-" : `${v} ms`),
};
export const debounce = (fn, ms = 250) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export function makeGuard() { let n = 0; return { next: () => ++n, ok: (id) => id === n }; }

// ---------------------------------------------------------------------------------------------
// state (persisted selection)
// ---------------------------------------------------------------------------------------------
const store = (() => { try { return window.localStorage; } catch { return null; } })();
const params = new URLSearchParams(location.search);
function readSaved() { try { return JSON.parse(store?.getItem("rdf.sel") || "{}"); } catch { return {}; } }
const saved = readSaved();
export const state = {
  store: Number(params.get("store")) || saved.store || 1,
  dept: Number(params.get("dept")) || saved.dept || 1,
  metric: params.get("metric") || saved.metric || "wape",
};
function persist() {
  try { store?.setItem("rdf.sel", JSON.stringify(state)); } catch { /* storage unavailable */ }
  const u = new URL(location.href);
  u.searchParams.set("store", state.store); u.searchParams.set("dept", state.dept); u.searchParams.set("metric", state.metric);
  history.replaceState(null, "", u);
}
const seriesListeners = new Set();
export const onSeriesChange = (cb) => { seriesListeners.add(cb); return () => seriesListeners.delete(cb); };
const themeListeners = new Set();
export const onThemeChange = (cb) => themeListeners.add(cb);

// ---------------------------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------------------------
let lastMs = null;
function setApiPill(ms, bad = false) {
  const el = $("#api-pill");
  if (!el) return;
  if (ms != null) lastMs = ms;
  el.innerHTML = `<span class="status-dot ${bad ? "bad" : ""}"></span>${bad ? "API error" : `Live Python API${lastMs != null ? ` · ${lastMs} ms` : ""}`}`;
}
export async function api(path, { params: q, method = "GET", body, signal } = {}) {
  const url = new URL(path, location.origin);
  if (q) Object.entries(q).forEach(([k, v]) => v != null && url.searchParams.set(k, v));
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(url, {
      method, signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    setApiPill(null, true);
    throw new Error("Could not reach the API. Check your connection and try again.");
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    setApiPill(null, true);
    const err = new Error(data?.error || `Request failed (${res.status})`);
    err.status = res.status; err.data = data;
    throw err;
  }
  setApiPill(Math.round(performance.now() - t0));
  return data;
}
export const seriesParams = (extra = {}) => ({ store: state.store, dept: state.dept, metric: state.metric, ...extra });

// ---------------------------------------------------------------------------------------------
// loading / error UI
// ---------------------------------------------------------------------------------------------
export const loadingHTML = (msg = "Computing live on the Python engine…") =>
  `<div class="loading-label"><span class="spinner"></span>${esc(msg)}</div><div class="skeleton" style="height:260px"></div>`;
export const errorHTML = (err) =>
  `<div class="errorbox"><b>Couldn't compute this view.</b><br>${esc(err?.message || err)}</div>`;
export function toast(msg, bad = false) {
  let host = $(".toast-host");
  if (!host) { host = document.createElement("div"); host.className = "toast-host"; document.body.append(host); }
  const t = document.createElement("div");
  t.className = `toast${bad ? " bad" : ""}`; t.textContent = msg; host.append(t);
  setTimeout(() => t.remove(), 3800);
}

// ---------------------------------------------------------------------------------------------
// theme
// ---------------------------------------------------------------------------------------------
function initialTheme() {
  try { const t = store?.getItem("rdf.theme"); if (t) return t; } catch { /* ignore */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
document.documentElement.dataset.theme = initialTheme();
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { store?.setItem("rdf.theme", next); } catch { /* ignore */ }
  themeListeners.forEach((cb) => cb(next));
}

// ---------------------------------------------------------------------------------------------
// chart helpers (Plotly, themed from CSS variables)
// ---------------------------------------------------------------------------------------------
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const colors = () => ({
  actual: cssVar("--c-actual"), forecast: cssVar("--c-forecast"), rolling: cssVar("--c-rolling"),
  baseline: cssVar("--c-baseline"), holiday: cssVar("--c-holiday"), interval: cssVar("--c-interval"),
  band: cssVar("--c-band"), grid: cssVar("--grid"), border: cssVar("--border"), text: cssVar("--text"),
  muted: cssVar("--muted"), surface: cssVar("--surface"), ok: cssVar("--ok"), sage: cssVar("--sage"), amber: cssVar("--amber"),
});
export function layout({ x, y, yfmt = "$", xfmt, legend = true, margin, ...extra } = {}) {
  const c = colors();
  const axis = (title) => ({
    title: title ? { text: title, font: { size: 12, color: c.muted }, standoff: 8 } : undefined,
    tickfont: { size: 11, color: c.muted }, gridcolor: c.grid, linecolor: c.border, zeroline: false, automargin: true, autorange: true,
  });
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: getComputedStyle(document.body).fontFamily, color: c.text, size: 12 },
    margin: margin || { l: 56, r: 18, t: legend ? 34 : 14, b: 46 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: c.surface, bordercolor: c.border, font: { color: c.text, size: 12 } },
    showlegend: legend,
    legend: { orientation: "h", x: 1, xanchor: "right", y: 1.1, yanchor: "bottom", font: { size: 11.5, color: c.muted } },
    xaxis: { ...axis(x), tickformat: xfmt },
    yaxis: { ...axis(y), tickprefix: yfmt === "$" ? "$" : "", ticksuffix: yfmt === "%" ? "%" : "", tickformat: yfmt === "$" ? "," : undefined },
  };
  const out = { ...base, ...extra };
  for (const ax of ["xaxis", "yaxis"]) {
    if (extra[ax]) {
      out[ax] = { ...base[ax], ...extra[ax] };
      if (extra[ax].range) out[ax].autorange = false;   // explicit range wins
    }
  }
  return out;
}
export function draw(el, traces, lay, config = {}) {
  const node = typeof el === "string" ? $(el) : el;
  if (!node || !window.Plotly) return Promise.resolve();
  return window.Plotly.react(node, traces, lay, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"], ...config });
}
export function resizeCharts(root = document) {
  if (!window.Plotly) return;
  $$(".js-plotly-plot", root).forEach((n) => { try { window.Plotly.Plots.resize(n); } catch { /* hidden */ } });
}
/** Plotly shapes/annotations for vertical event lines. */
export const vline = (x, color, dash = "dot") => ({ type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1.4, dash } });

// ---------------------------------------------------------------------------------------------
// tables & tabs
// ---------------------------------------------------------------------------------------------
/** columns: [{label, key|get, fmt, num, cls}] ; rows: array ; opts: {rowClass(row)} */
export function table(columns, rows, opts = {}) {
  const head = columns.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows.map((r) => {
    const cells = columns.map((c) => {
      const raw = c.get ? c.get(r) : r[c.key];
      const out = c.html ? c.html(r) : esc(c.fmt ? c.fmt(raw, r) : raw ?? "-");
      return `<td class="${c.num ? "num" : ""} ${c.cls || ""}">${out}</td>`;
    }).join("");
    return `<tr class="${opts.rowClass ? opts.rowClass(r) : ""}">${cells}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
export function tabs(root, onShow) {
  const btns = $$("[role=tab]", root);
  const panels = $$("[role=tabpanel]", root);
  const show = (id) => {
    btns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === id)));
    panels.forEach((p) => { p.hidden = p.dataset.panel !== id; });
    const panel = panels.find((p) => p.dataset.panel === id);
    requestAnimationFrame(() => { resizeCharts(panel); onShow?.(id); });
  };
  btns.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  show(btns[0].dataset.tab);
  return show;
}

// ---------------------------------------------------------------------------------------------
// shell
// ---------------------------------------------------------------------------------------------
const ICON = {
  overview: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
  eda: '<path d="M3 3v18h18"/><path d="M7 15v3M12 9v9M17 5v13"/>',
  models: '<circle cx="12" cy="8" r="6"/><path d="M15.5 13.5 17 22l-5-3-5 3 1.5-8.5"/>',
  forecast: '<path d="m3 17 6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
  inventory: '<path d="M16.5 9.4 7.5 4.2"/><path d="M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4a2 2 0 0 0 1-1.7z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
  simulation: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  insights: '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  cart: '<circle cx="9" cy="20" r="1.4"/><circle cx="18" cy="20" r="1.4"/><path d="M2 3h3l2.6 12.4a2 2 0 0 0 2 1.6h7.8a2 2 0 0 0 2-1.5L21 8H6"/>',
};
export const icon = (name, w = 2) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name] || ""}</svg>`;

export const NAV = [
  { id: "overview", href: "/", label: "Overview", icon: "overview", blurb: "KPIs, trend and champion model" },
  { id: "eda", href: "/eda", label: "Sales Analysis", icon: "eda", blurb: "Seasonality, holidays, promotions" },
  { id: "models", href: "/models", label: "Model Leaderboard", icon: "models", blurb: "8 models, 2 validation schemes" },
  { id: "forecast", href: "/forecast", label: "Demand Forecast", icon: "forecast", blurb: "Multi-step forecast with 95% bounds" },
  { id: "inventory", href: "/inventory", label: "Inventory Planning", icon: "inventory", blurb: "Safety stock, reorder point, order size" },
  { id: "simulation", href: "/simulation", label: "Live Simulation", icon: "simulation", blurb: "Weekly retrain + replenish replay", tag: "LIVE" },
  { id: "insights", href: "/insights", label: "Insights & AI", icon: "insights", blurb: "Findings, caveats, AI analyst" },
];

let META = null;
export const getMeta = () => META;

/**
 * Build the page chrome and return the content element.
 * opts: {page, eyebrow, title, subtitle(): html, body: html, series: bool}
 */
export async function mount(opts) {
  const { page, eyebrow, title, body, series = true } = opts;
  document.title = `${title} · Retail Demand Forecasting`;
  const nav = NAV.map((n) => `<a href="${n.href}" ${n.id === page ? 'aria-current="page"' : ""}>${icon(n.icon)}<span>${esc(n.label)}</span>${n.tag ? `<span class="tag">${n.tag}</span>` : ""}</a>`).join("");
  const seriesCtl = series ? `
      <div class="ctl"><label for="sel-store">Store</label><select id="sel-store" aria-label="Store"></select></div>
      <div class="ctl"><label for="sel-dept">Dept</label><select id="sel-dept" aria-label="Department"></select></div>
      <div class="ctl"><label for="sel-metric">Rank by</label><select id="sel-metric" aria-label="Ranking metric"></select></div>` : "";
  $("#app").innerHTML = `
  <div class="app">
    <aside class="sidebar" id="sidebar">
      <a class="brand" href="/"><span class="brand-mark">${icon("cart", 2.2)}</span>
        <span><div class="brand-title">Retail Demand<br>Forecasting</div><div class="brand-sub">Decision support</div></span></a>
      <div class="nav-label">Workspace</div>
      <nav class="nav" aria-label="Primary">${nav}</nav>
      <div class="side-foot"><span id="ai-state"></span><br>Sales are weekly revenue in USD, not unit counts. Every model trains only on the past.</div>
    </aside>
    <div class="main">
      <header class="topbar">
        <button class="icon-btn menu-btn" id="menu-btn" aria-label="Open navigation">${icon("menu")}</button>
        ${seriesCtl}
        <span class="spacer"></span>
        <span class="api-pill" id="api-pill"><span class="status-dot off"></span>Connecting…</span>
        <button class="icon-btn" id="theme-btn" aria-label="Toggle light / dark theme">${icon("sun")}</button>
      </header>
      <main class="content" id="content">
        <div class="page-head"><span class="eyebrow">${esc(eyebrow || "")}</span><h1 class="page-title">${esc(title)}</h1><div class="page-sub" id="page-sub"></div></div>
        ${body}
      </main>
      <footer class="footer">Retail Demand Forecasting &amp; Inventory Decision Support · Python (pandas · scikit-learn · statsmodels) on Vercel · Plotly.js</footer>
    </div>
  </div>`;

  $("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#content").addEventListener("click", () => $("#sidebar").classList.remove("open"));
  $("#theme-btn").addEventListener("click", toggleTheme);

  try {
    META = META || (await api("/api/meta"));
  } catch (e) {
    $("#content").insertAdjacentHTML("afterbegin", errorHTML(e));
    throw e;
  }
  api("/api/ai/status").then((s) => {
    $("#ai-state").innerHTML = `<span class="status-dot ${s.configured ? "" : "off"}"></span>${s.configured ? `AI analyst: ${esc(s.model)}` : "AI analyst: add GEMINI_API_KEY"}`;
  }).catch(() => {});

  if (series) wireSelectors();
  return $("#content");
}

function modelableDepts(storeId) {
  const s = META.stores.find((x) => x.id === storeId);
  return (s?.depts || []).filter(([, n]) => n >= META.min_model_obs);
}
function wireSelectors() {
  const selStore = $("#sel-store"), selDept = $("#sel-dept"), selMetric = $("#sel-metric");
  selStore.innerHTML = META.stores.map((s) => `<option value="${s.id}">Store ${s.id} · type ${s.type}</option>`).join("");
  selMetric.innerHTML = Object.entries(META.metrics).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  const fillDepts = () => {
    const list = modelableDepts(state.store);
    if (!list.some(([d]) => d === state.dept)) state.dept = list[0]?.[0] ?? 1;
    selDept.innerHTML = list.map(([d, n]) => `<option value="${d}">Dept ${d} · ${n} wks</option>`).join("");
    selDept.value = String(state.dept);
  };
  if (!META.stores.some((s) => s.id === state.store)) state.store = 1;
  if (!META.metrics[state.metric]) state.metric = "wape";
  selStore.value = String(state.store); selMetric.value = state.metric; fillDepts(); persist();
  const fire = (kind) => { persist(); seriesListeners.forEach((cb) => cb(kind)); };
  selStore.addEventListener("change", () => { state.store = Number(selStore.value); fillDepts(); fire("series"); });
  selDept.addEventListener("change", () => { state.dept = Number(selDept.value); fire("series"); });
  selMetric.addEventListener("change", () => { state.metric = selMetric.value; fire("metric"); });
}

/** Standard subtitle: which series is being analysed. */
export function setSubtitle(html) { const el = $("#page-sub"); if (el) el.innerHTML = html; }
export const seriesLabel = () => `<b>Store ${state.store}</b> · <b>Department ${state.dept}</b>`;
export function download(filename, text, mime = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}
export const kpi = (label, value, sub = "", cls = "", small = false) =>
  `<div class="kpi ${cls}"><div class="kpi-label">${esc(label)}</div><div class="kpi-value ${small ? "sm" : ""}">${value}</div><div class="kpi-sub">${sub}</div></div>`;
export const modelPill = (r) => (r.rank === 1 ? '<span class="pill win">Champion</span>' : r.status !== "Valid" ? '<span class="pill bad">Failed</span>' : "");

// ---------------------------------------------------------------------------------------------
// shared chart/metric helpers used by several pages
// ---------------------------------------------------------------------------------------------
export const metricFmt = (key, v) => (key === "wape" || key === "mape" ? fmt.pct(v) : fmt.usd(v, key === "rmse" || key === "mae" ? 0 : 0));

/** Weekly sales + 4-week moving average + holiday markers (+ optional linear trend). */
export function drawTrend(el, s, { showTrend = false } = {}) {
  const c = colors();
  const hol = s.is_holiday.map((h, i) => (h ? i : -1)).filter((i) => i >= 0);
  const traces = [
    { x: s.dates, y: s.sales, name: "Weekly sales", mode: "lines+markers", line: { color: c.actual, width: 2 }, marker: { size: 4.5 }, hovertemplate: "$%{y:,.0f}" },
    { x: s.dates, y: s.rolling4, name: "4-week average", mode: "lines", line: { color: c.rolling, width: 2.4, dash: "dash" }, hovertemplate: "$%{y:,.0f}" },
    { x: hol.map((i) => s.dates[i]), y: hol.map((i) => s.sales[i]), name: "Holiday weeks", mode: "markers", marker: { color: c.holiday, size: 9, symbol: "diamond", line: { color: c.surface, width: 1 } }, hovertemplate: "Holiday · $%{y:,.0f}" },
  ];
  if (showTrend) traces.push({ x: s.dates, y: s.trend, name: "Linear trend", mode: "lines", line: { color: c.forecast, width: 2, dash: "dot" }, hovertemplate: "$%{y:,.0f}" });
  return draw(el, traces, layout({ x: "Week", y: "Weekly sales (USD)" }));
}
