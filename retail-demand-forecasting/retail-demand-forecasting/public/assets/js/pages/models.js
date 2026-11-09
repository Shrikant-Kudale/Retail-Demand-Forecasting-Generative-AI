import { $, $$, api, mount, setSubtitle, seriesLabel, state, seriesParams, onSeriesChange, onThemeChange, makeGuard, loadingHTML, errorHTML, fmt, kpi, table, tabs, draw, layout, colors, esc, getMeta, metricFmt, modelPill } from "../core.js";

await mount({
  page: "models", eyebrow: "Model validation · 8 candidates", title: "Model leaderboard & diagnostics",
  body: `<div id="view"></div>`,
});

const guard = makeGuard();
let ev = null, activeTab = "cv", holdModel = null, resModel = null;

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Training 8 models: chronological holdout + 4-fold rolling-origin validation…");
  setSubtitle(`${seriesLabel()} · every model is trained only on data preceding each forecast origin.`);
  try {
    const r = await api("/api/evaluate", { params: seriesParams() });
    if (!guard.ok(id)) return;
    ev = r; holdModel = holdModel && r.predictions[holdModel] ? holdModel : r.winner.holdout.key;
    resModel = r.winner.holdout.key; render();
  } catch (e) { if (guard.ok(id)) $("#view").innerHTML = errorHTML(e); }
}

const metricCols = [
  { label: "WAPE", num: true, get: (r) => fmt.pct(r.wape) }, { label: "MAPE", num: true, get: (r) => fmt.pct(r.mape) },
  { label: "MAE", num: true, get: (r) => fmt.usd(r.mae) }, { label: "RMSE", num: true, get: (r) => fmt.usd(r.rmse) },
];
const boardTable = (rows) => table([
  { label: "Rank", get: (r) => r.rank ?? "–" },
  { label: "Model", html: (r) => `${esc(r.name)} ${modelPill(r)}` }, { label: "Family", key: "family" },
  ...metricCols, { label: "Folds", num: true, key: "folds" }, { label: "Eval. obs", num: true, key: "obs" }, { label: "Status", key: "status" },
], rows, { rowClass: (r) => (r.rank === 1 ? "win" : "") });

const modelSelect = (id, val) => `<select id="${id}">${ev.holdout.map((r) => `<option value="${r.key}" ${r.key === val ? "selected" : ""}>${esc(r.name)}${r.rank === 1 ? " ★" : ""}</option>`).join("")}</select>`;

function render() {
  if (!ev) return;
  const cvw = ev.winner.cv, how = ev.winner.holdout, cmp = ev.comparison;
  const res = ev.residuals[resModel];
  const hs = res?.holiday_stats || {};
  const holN = hs.Holiday?.n ?? 0;
  const catalog = getMeta().models;

  $("#view").innerHTML = `
  <div class="kpis k4">
    ${kpi("CV champion", esc(cvw.name), `${esc(cvw.family)} · primary selection`, "accent", true)}
    ${kpi(`CV ${ev.metric_label}`, metricFmt(state.metric, cvw[state.metric]), `${ev.cv_folds} folds × ${ev.cv_horizon}-week horizon`, "good")}
    ${kpi("Holdout champion", esc(how.name), `Last ${ev.test_weeks} weeks (20%)`, "", true)}
    ${kpi(`Holdout ${ev.metric_label}`, metricFmt(state.metric, how[state.metric]), `Trained on ${ev.train_weeks} weeks`)}
  </div>
  <div id="tabs">
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" data-tab="cv">Rolling-origin CV</button>
      <button class="tab" role="tab" data-tab="hold">Chronological holdout</button>
      <button class="tab" role="tab" data-tab="resid">Residuals &amp; shocks</button>
      <button class="tab" role="tab" data-tab="cmp">Baseline vs improved</button>
      <button class="tab" role="tab" data-tab="cat">Model catalog</button>
    </div>

    <div role="tabpanel" data-panel="cv">
      <div class="note"><b>Primary selection method.</b> Four expanding-window folds, each forecasting 4 weeks beyond its origin, are far more reliable than one split: the champion must hold up across several seasons of the calendar. Ranked by <b>${esc(ev.metric_label)}</b>.</div>
      <section class="panel"><div class="panel-head"><div class="panel-title">${esc(ev.metric_label)} by model · lower is better</div></div><div id="c-cv" class="chart"></div></section>
      <section class="panel"><div class="panel-head"><div class="panel-title">Rolling-origin leaderboard</div></div>${boardTable(ev.cv)}</section>
    </div>

    <div role="tabpanel" data-panel="hold" hidden>
      <div class="note warn"><b>Secondary check.</b> A single 80:20 chronological split (train ${fmt.date(ev.holdout_test.train_start)} → ${fmt.date(ev.holdout_test.train_end)}; test = last ${ev.test_weeks} weeks) is volatile for short series and a poor single judge on its own.</div>
      <section class="panel"><div class="panel-head"><div class="panel-title">Holdout leaderboard</div></div>${boardTable(ev.holdout)}</section>
      <section class="panel"><div class="panel-head"><div class="panel-title">Actual vs predicted on the test window</div><div class="ctl"><label>Model</label>${modelSelect("sel-hold", holdModel)}</div></div><div id="c-hold" class="chart tall"></div></section>
    </div>

    <div role="tabpanel" data-panel="resid" hidden>
      <div class="panel-head"><div class="panel-title">Residual analysis</div><div class="ctl"><label>Model</label>${modelSelect("sel-res", resModel)}</div></div>
      <div class="grid g2">
        <section class="panel"><div class="panel-head"><div class="panel-title">Residual error over time (actual − predicted)</div></div><div id="c-res" class="chart"></div></section>
        <section class="panel"><div class="panel-head"><div class="panel-title">Error distribution</div><span class="panel-note">mean ${fmt.usd(res.mean)} · std ${fmt.usd(res.std)}</span></div><div id="c-hist" class="chart"></div></section>
      </div>
      <div class="grid g2" style="margin-top:18px">
        <section class="panel"><div class="panel-head"><div class="panel-title">5 largest errors</div></div>
          ${table([{ label: "Week", get: (r) => fmt.date(r.date) }, { label: "Actual", num: true, get: (r) => fmt.usd(r.actual) }, { label: "Predicted", num: true, get: (r) => fmt.usd(r.predicted) },
            { label: "Error", num: true, get: (r) => fmt.usd(r.error) }, { label: "Holiday", get: (r) => (r.is_holiday ? "Yes" : "No") }], res.top_errors)}</section>
        <section class="panel"><div class="panel-head"><div class="panel-title">Holiday vs non-holiday accuracy</div></div>
          ${table([{ label: "Period", get: (r) => r[0] }, { label: "N", num: true, get: (r) => r[1].n }, { label: "MAE", num: true, get: (r) => fmt.usd(r[1].mae) },
            { label: "RMSE", num: true, get: (r) => fmt.usd(r[1].rmse) }, { label: "WAPE", num: true, get: (r) => fmt.pct(r[1].wape) }], Object.entries(hs))}</section>
      </div>
      <section class="panel" style="margin-top:18px"><div class="panel-head"><div class="panel-title">Holiday shock comparison · all models on the test window</div></div>
        ${table([{ label: "Model", key: "model" }, { label: "Overall WAPE", num: true, get: (r) => fmt.pct(r.overall_wape) }, { label: "Non-holiday WAPE", num: true, get: (r) => fmt.pct(r.non_holiday_wape) },
          { label: "Holiday WAPE", num: true, get: (r) => fmt.pct(r.holiday_wape) }, { label: "Holiday MAE", num: true, get: (r) => fmt.usd(r.holiday_mae) }, { label: "N hol / non-hol", num: true, get: (r) => `${r.holiday_n} / ${r.non_holiday_n}` }], ev.shocks)}</section>
      <div class="note danger"><b>Interpret with care:</b> the test window contains only <b>${holN}</b> holiday week${holN === 1 ? "" : "s"}. Any holiday accuracy figure from so few observations is anecdotal, not statistically reliable.</div>
    </div>

    <div role="tabpanel" data-panel="cmp" hidden>
      ${cmp ? `
      <div class="kpis k4" style="grid-template-columns:repeat(3,minmax(0,1fr))">
        ${kpi("Original 4-lag regression", fmt.pct(cmp.baseline_wape), `MAE ${fmt.usd(cmp.baseline_mae)}`, "bad")}
        ${kpi("Improved seasonal regression", fmt.pct(cmp.improved_wape), `MAE ${fmt.usd(cmp.improved_mae)}`, "good")}
        ${kpi("Error reduction", `${cmp.reduction_pct}%`, "Relative WAPE improvement", "accent")}
      </div>
      <div class="grid g-3-2">
        <section class="panel"><div class="panel-head"><div class="panel-title">WAPE &amp; MAE · before vs after</div></div><div id="c-cmp" class="chart"></div></section>
        <section class="panel"><div class="panel-head"><div class="panel-title">What changed</div></div>
          <ul class="list-clean"><li><b>52-week lag</b> and 13-week rolling mean carry last year's seasonal shape.</li><li><b>Fourier week-of-year terms</b> (sin / cos) let a linear model bend with the calendar.</li><li><b>Holiday flag</b> and markdown-availability marker handle shocks and the Nov-2011 data regime change.</li><li><b>Shifted rolling statistics</b> keep every feature strictly in the past - no leakage.</li></ul></section>
      </div>` : `<div class="empty">Comparison unavailable for this series.</div>`}
    </div>

    <div role="tabpanel" data-panel="cat" hidden>
      <div class="grid g3">${catalog.map((m) => `<div class="insight ${m.type === "ml" ? "amber" : ""}"><h3>${esc(m.name)}</h3><p><span class="pill">${esc(m.family)}</span></p><p>${esc(m.description)}</p>${m.features.length ? `<p class="hint">${m.features.length} features: ${esc(m.features.join(", "))}</p>` : ""}</div>`).join("")}</div>
    </div>
  </div>`;

  tabs($("#tabs"), (id) => { activeTab = id; });
  $$("[role=tab]").forEach((b) => { if (b.dataset.tab === activeTab) b.click(); });
  $("#sel-hold")?.addEventListener("change", (e) => { holdModel = e.target.value; drawHold(); });
  $("#sel-res")?.addEventListener("change", (e) => { resModel = e.target.value; render(); });
  drawCv(); drawHold(); drawRes(); drawCmp();
}

function drawCv() {
  const c = colors();
  const rows = ev.cv.filter((r) => r.status === "Valid").slice().reverse();
  draw("#c-cv", [{
    y: rows.map((r) => r.name), x: rows.map((r) => r[state.metric]), type: "bar", orientation: "h",
    marker: { color: rows.map((r) => (r.rank === 1 ? c.forecast : c.rolling)) },
    text: rows.map((r) => metricFmt(state.metric, r[state.metric])), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}: %{text}<extra></extra>",
  }], layout({ x: ev.metric_label, yfmt: "", legend: false, hovermode: "closest", margin: { l: 220, r: 70, t: 10, b: 46 } }));
}

function drawHold() {
  if (!$("#c-hold")) return;
  const c = colors(), t = ev.holdout_test;
  const ctxN = 26, td = t.train_dates.slice(-ctxN), ts = t.train_sales.slice(-ctxN);
  const sel = ev.holdout.find((r) => r.key === holdModel);
  const traces = [
    { x: td, y: ts, name: "Training (last 26 wks)", mode: "lines", line: { color: c.baseline, width: 2 }, hovertemplate: "$%{y:,.0f}" },
    { x: t.dates, y: t.actual, name: "Actual", mode: "lines+markers", line: { color: c.actual, width: 2.4 }, marker: { size: 6 }, hovertemplate: "$%{y:,.0f}" },
  ];
  if (holdModel !== "naive" && ev.predictions.naive) traces.push({ x: t.dates, y: ev.predictions.naive, name: "Naive baseline", mode: "lines", line: { color: c.baseline, width: 2, dash: "dot" }, hovertemplate: "$%{y:,.0f}" });
  traces.push({ x: t.dates, y: ev.predictions[holdModel], name: sel?.name || holdModel, mode: "lines+markers", line: { color: c.forecast, width: 2.6, dash: "dash" }, marker: { size: 6, symbol: "diamond" }, hovertemplate: "$%{y:,.0f}" });
  draw("#c-hold", traces, layout({ x: "Week", y: "Weekly sales (USD)", shapes: [{ type: "line", xref: "x", yref: "paper", x0: t.dates[0], x1: t.dates[0], y0: 0, y1: 1, line: { color: c.muted, width: 1.2, dash: "dot" } }] }));
}

function drawRes() {
  if (!$("#c-res")) return;
  const c = colors(), r = ev.residuals[resModel];
  draw("#c-res", [{ x: r.dates, y: r.error, type: "bar", marker: { color: r.error.map((e) => (e >= 0 ? c.rolling : c.holiday)) }, hovertemplate: "$%{y:,.0f}" }],
    layout({ x: "Week", y: "Error (USD)", legend: false, shapes: [{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: c.muted, width: 1 } }] }));
  draw("#c-hist", [
    { x: r.error, type: "histogram", nbinsx: 14, marker: { color: c.rolling, line: { color: c.surface, width: 1 } }, yaxis: "y", name: "Errors" },
    { x: r.error, type: "box", orientation: "h", yaxis: "y2", marker: { color: c.forecast }, name: "", boxpoints: false },
  ], layout({ x: "Error (USD)", y: "Weeks", yfmt: "", legend: false, hovermode: "closest", yaxis: { domain: [0, 0.78] }, yaxis2: { domain: [0.84, 1], showticklabels: false, gridcolor: "rgba(0,0,0,0)" } }));
}

function drawCmp() {
  if (!$("#c-cmp") || !ev.comparison) return;
  const c = colors(), m = ev.comparison;
  draw("#c-cmp", [
    { x: ["WAPE (%)"], y: [m.baseline_wape], name: "Original 4-lag", type: "bar", marker: { color: c.baseline }, text: [fmt.pct(m.baseline_wape)], textposition: "outside" },
    { x: ["WAPE (%)"], y: [m.improved_wape], name: "Improved seasonal", type: "bar", marker: { color: c.forecast }, text: [fmt.pct(m.improved_wape)], textposition: "outside" },
  ], layout({ y: "WAPE (%)", yfmt: "", hovermode: "closest", barmode: "group" }));
}

onSeriesChange(load);
onThemeChange(render);
load();
