import { $, api, mount, setSubtitle, seriesLabel, state, seriesParams, onSeriesChange, onThemeChange, makeGuard, loadingHTML, errorHTML, fmt, kpi, table, draw, layout, colors, esc, getMeta, download, debounce, toast } from "../core.js";

await mount({
  page: "forecast", eyebrow: "Forward-looking demand", title: "Demand forecast",
  body: `
  <section class="panel"><div class="controls-row">
    <div class="field" style="margin:0;min-width:260px"><label for="f-model">Forecasting model</label><select id="f-model"></select></div>
    <div class="field" style="margin:0;min-width:240px"><label for="f-h">Horizon: <output id="f-hout">8</output> weeks</label><input id="f-h" type="range" min="1" max="26" value="8"></div>
    <span class="hint" id="f-hint"></span></div></section>
  <div id="view"></div>`,
});

const guard = makeGuard();
let model = "auto", horizon = 8, data = null;
const sel = $("#f-model");

function fillModels(champ) {
  sel.innerHTML = `<option value="auto">Auto · validated champion</option>` + getMeta().models.map((m) => `<option value="${m.key}">${esc(m.name)}${m.key === champ ? " ★" : ""}</option>`).join("");
  sel.value = model;
}
fillModels();

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Fitting the model and rolling the forecast forward week by week…");
  setSubtitle(`${seriesLabel()} · recursive multi-step forecast with prediction intervals calibrated on the model's own validation errors.`);
  try {
    const r = await api("/api/forecast", { params: seriesParams({ model, horizon }) });
    if (!guard.ok(id)) return;
    data = r; fillModels(r.champion_key); render();
  } catch (e) { if (guard.ok(id)) $("#view").innerHTML = errorHTML(e); }
}

function render() {
  if (!data) return;
  const d = data, f = d.forecast, peak = f.reduce((a, b) => (b.forecast > a.forecast ? b : a));
  $("#f-hint").textContent = `${d.known_calendar_weeks} of ${d.horizon} future weeks use the dataset's own holiday and weather calendar.`;
  $("#view").innerHTML = `
  <div class="kpis k4">
    ${kpi("Model", esc(d.model.name), d.is_champion ? "Validated champion ★" : "Manual selection", "accent", true)}
    ${kpi(`Next ${d.horizon} weeks total`, fmt.usd(d.total_forecast), `~${fmt.usd(d.total_forecast / d.horizon)} per week`)}
    ${kpi("First forecast week", fmt.usd(f[0].forecast), fmt.date(f[0].date))}
    ${kpi("Peak forecast week", fmt.usd(peak.forecast), `${fmt.date(peak.date)}${peak.is_holiday ? " · " + esc(peak.holiday) : ""}`, "good")}
  </div>
  <section class="panel"><div class="panel-head"><div class="panel-title">Historical sales &amp; forecast with 95% prediction interval</div><span class="panel-note">σ = ${fmt.usd(d.residual_std)} (validation RMSE)</span></div><div id="c-fc" class="chart tall"></div></section>
  <div class="grid g-3-2">
    <section class="panel"><div class="panel-head"><div class="panel-title">Forecast table</div><button class="btn" id="dl">Download CSV</button></div>
      ${table([{ label: "Week ending", get: (r) => fmt.date(r.date) }, { label: "ISO wk", num: true, key: "week" }, { label: "Event", key: "holiday" },
        { label: "Forecast", num: true, get: (r) => fmt.usd(r.forecast) }, { label: "Lower 95%", num: true, get: (r) => fmt.usd(r.lower) }, { label: "Upper 95%", num: true, get: (r) => fmt.usd(r.upper) }], f)}</section>
    <section class="panel"><div class="panel-head"><div class="panel-title">What drives this forecast</div></div>
      ${d.drivers.length ? `<div id="c-drv" class="chart"></div><p class="hint">Share of the model's decision weight per feature. Lags carry momentum; Lag_52 and Week_Sin/Cos carry annual seasonality.</p>` : `<div class="empty">${esc(d.model.name)} has no per-feature drivers (it forecasts from the series' own history).</div>`}
    </section>
  </div>
  <div class="note"><b>How to read it:</b> each predicted week is fed back as next week's lag (recursive strategy), so uncertainty widens with the square root of the horizon. ${esc(d.model.description)}</div>`;

  const c = colors(), h = d.history;
  const fx = f.map((r) => r.date);
  draw("#c-fc", [
    { x: h.dates, y: h.sales, name: "Historical (26 wks)", mode: "lines+markers", line: { color: c.actual, width: 2.2 }, marker: { size: 4 }, hovertemplate: "$%{y:,.0f}" },
    { x: fx, y: f.map((r) => r.upper), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: fx, y: f.map((r) => r.lower), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.interval, name: "95% interval", hoverinfo: "skip" },
    { x: [h.dates.at(-1), fx[0]], y: [h.sales.at(-1), f[0].forecast], mode: "lines", line: { color: c.forecast, width: 1.6, dash: "dot" }, showlegend: false, hoverinfo: "skip" },
    { x: fx, y: f.map((r) => r.forecast), name: "Forecast", mode: "lines+markers", line: { color: c.forecast, width: 2.8, dash: "dash" }, marker: { size: 7, symbol: "diamond" }, hovertemplate: "$%{y:,.0f}" },
    { x: f.filter((r) => r.is_holiday).map((r) => r.date), y: f.filter((r) => r.is_holiday).map((r) => r.forecast), name: "Holiday week", mode: "markers", marker: { color: c.holiday, size: 11, symbol: "star" }, hovertemplate: "Holiday · $%{y:,.0f}" },
  ], layout({ x: "Week", y: "Weekly sales (USD)" }));

  if (d.drivers.length) {
    const dr = d.drivers.slice(0, 10).reverse();
    draw("#c-drv", [{ y: dr.map((x) => x.feature), x: dr.map((x) => x.importance), type: "bar", orientation: "h", marker: { color: c.rolling },
      text: dr.map((x) => `${x.importance.toFixed(1)}%`), textposition: "outside", cliponaxis: false, hovertemplate: "%{x:.1f}%<extra></extra>" }],
      layout({ x: "Importance (%)", yfmt: "", legend: false, hovermode: "closest", margin: { l: 120, r: 50, t: 10, b: 46 } }));
  }
  $("#dl").addEventListener("click", () => {
    const rows = ["Date,ISO_Week,Event,Forecast_USD,Lower_95_USD,Upper_95_USD", ...f.map((r) => [r.date, r.week, `"${r.holiday}"`, r.forecast, r.lower, r.upper].join(","))];
    download(`forecast_store${state.store}_dept${state.dept}.csv`, rows.join("\n")); toast("Forecast CSV downloaded");
  });
}

sel.addEventListener("change", () => { model = sel.value; load(); });
const reload = debounce(load, 300);
$("#f-h").addEventListener("input", (e) => { horizon = Number(e.target.value); $("#f-hout").textContent = horizon; reload(); });
onSeriesChange((k) => { if (k === "series") model = "auto", sel.value = "auto"; load(); });
onThemeChange(render);
load();
