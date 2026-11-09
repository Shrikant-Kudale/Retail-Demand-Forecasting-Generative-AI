import { $, $$, api, mount, setSubtitle, seriesLabel, seriesParams, onSeriesChange, onThemeChange, makeGuard, loadingHTML, errorHTML, fmt, kpi, table, tabs, drawTrend, draw, layout, colors, esc, getMeta } from "../core.js";

await mount({
  page: "eda", eyebrow: "Exploratory analysis · demand drivers", title: "Historical sales analysis",
  body: `<div id="view"></div>`,
});

const guard = makeGuard();
let data = null, activeTab = "season", showTrend = true;

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Decomposing seasonality, holidays and promotions…");
  setSubtitle(`${seriesLabel()} · dissecting 52-week cycles, holiday demand shocks and promotional markdown impact.`);
  try {
    const [series, eda] = await Promise.all([api("/api/series", { params: seriesParams() }), api("/api/eda", { params: seriesParams() })]);
    if (!guard.ok(id)) return;
    data = { series, eda };
    render();
  } catch (e) { if (guard.ok(id)) $("#view").innerHTML = errorHTML(e); }
}

const weekRows = (rows) => table([
  { label: "ISO week", key: "week" }, { label: "Avg sales", num: true, get: (r) => fmt.usd(r.mean) },
  { label: "Std dev", num: true, get: (r) => fmt.usd(r.std) }, { label: "Years seen", num: true, key: "count" },
], rows);

function render() {
  if (!data) return;
  const { series: s, eda } = data;
  const h = eda.holiday;
  const cmp = eda.promo_comparison;

  $("#view").innerHTML = `
  <div id="tabs">
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" data-tab="season">Seasonality &amp; trend</button>
      <button class="tab" role="tab" data-tab="holiday">Holiday surges</button>
      <button class="tab" role="tab" data-tab="promo">Promotional markdowns</button>
      <button class="tab" role="tab" data-tab="drivers">External drivers</button>
    </div>

    <div role="tabpanel" data-panel="season">
      <section class="panel">
        <div class="panel-head"><div class="panel-title">Weekly sales, 4-week average &amp; overall trend</div>
          <div class="seg"><button aria-pressed="${!showTrend}" data-t="0">Hide trend</button><button aria-pressed="${showTrend}" data-t="1">Show linear trend</button></div></div>
        <div id="c-trend" class="chart"></div>
        <p class="hint">Linear trend slope: <b>${fmt.usd(s.trend_slope_per_week)}</b> per week - ${Math.abs(s.trend_slope_per_week) < 0.01 * s.kpis.avg_weekly ? "essentially flat; demand is driven by seasonality, not growth." : s.trend_slope_per_week > 0 ? "demand is drifting upwards." : "demand is drifting downwards."}</p>
      </section>
      <section class="panel">
        <div class="panel-head"><div class="panel-title">Annual seasonal profile · average sales by week of year</div><span class="panel-note">Shaded band = ±1 standard deviation</span></div>
        <div id="c-season" class="chart tall"></div>
      </section>
      <div class="grid g2">
        <section class="panel"><div class="panel-head"><div class="panel-title">Top 5 peak weeks</div></div>${weekRows(eda.top_weeks)}</section>
        <section class="panel"><div class="panel-head"><div class="panel-title">Lowest 5 demand weeks</div></div>${weekRows(eda.low_weeks)}</section>
      </div>
    </div>

    <div role="tabpanel" data-panel="holiday" hidden>
      <div class="grid g-3-2">
        <section class="panel"><div class="panel-head"><div class="panel-title">Sales distribution · holiday vs normal weeks</div></div><div id="c-box" class="chart tall"></div></section>
        <section class="panel">
          <div class="panel-head"><div class="panel-title">Aggregate holiday statistics</div></div>
          <div class="kpis k4" style="grid-template-columns:1fr 1fr;margin-bottom:14px">
            ${kpi("Holiday mean uplift", fmt.signed(h.uplift_pct), "vs non-holiday weeks", h.uplift_pct >= 0 ? "good" : "bad")}
            ${kpi("Holiday sample", `${h.holiday_n} wks`, `${h.holiday_share_pct}% of history`)}
          </div>
          ${table([
            { label: "Period", key: "period" }, { label: "N", num: true, key: "n" },
            { label: "Mean", num: true, get: (r) => fmt.usd(r.mean) }, { label: "Median", num: true, get: (r) => fmt.usd(r.median) },
            { label: "Std", num: true, get: (r) => fmt.usd(r.std) }, { label: "Share", num: true, get: (r) => `${r.share_pct}%` },
          ], h.period_stats)}
        </section>
      </div>
      <section class="panel">
        <div class="panel-head"><div class="panel-title">Specific retail event breakdown · department demand sensitivity</div></div>
        ${table([
          { label: "Retail event / period", key: "event" }, { label: "N", num: true, key: "n" },
          { label: "Mean weekly sales", num: true, get: (r) => fmt.usd(r.mean) }, { label: "Median", num: true, get: (r) => fmt.usd(r.median) },
          { label: "Std dev", num: true, get: (r) => fmt.usd(r.std) }, { label: "Uplift vs regular", num: true, get: (r) => fmt.signed(r.uplift_pct) },
        ], h.events)}
      </section>
      <div class="note"><b>Causal caution:</b> differences reflect the retail calendar (pre-Thanksgiving grocery prep, Super Bowl snacking) and not proof that a holiday flag by itself causes revenue. Departments differ sharply, and some even dip right after a major event.</div>
    </div>

    <div role="tabpanel" data-panel="promo" hidden>
      <div class="note warn">Walmart began recording promotional markdowns in <b>${fmt.date(getMeta().markdown_intro_date)}</b>. Earlier weeks are <i>unrecorded</i> (not zero promotion); after that date a blank means no markdown that week. The model receives a <code>MarkDown_Available</code> flag for exactly this reason.</div>
      <section class="panel"><div class="panel-head"><div class="panel-title">Promotional markdowns over time (MarkDown1 – MarkDown5)</div></div>
        ${eda.markdown.dates.length ? `<div id="c-md" class="chart tall"></div>` : `<div class="empty">No post-November-2011 records for this selection.</div>`}</section>
      <div class="grid g-3-2">
        <section class="panel"><div class="panel-head"><div class="panel-title">Average sales · promo vs non-promo weeks</div></div><div id="c-promo" class="chart short"></div>
          <p class="hint">Same definition as the original notebook (any positive total markdown = promo week). Because markdowns only exist after Nov-2011 this comparison is confounded by time and season - treat it as descriptive.</p></section>
        <section class="panel"><div class="panel-head"><div class="panel-title">Top 10 store-wide markdown weeks</div></div>
          ${table([{ label: "Week", get: (r) => fmt.date(r.Date) }, { label: "Total markdown", num: true, get: (r) => fmt.usd(r.Total_MarkDown) }, { label: "Holiday", get: (r) => (r.IsHoliday ? "Yes" : "No") }], eda.top_promotions)}</section>
      </div>
    </div>

    <div role="tabpanel" data-panel="drivers" hidden>
      <section class="panel"><div class="panel-head"><div class="panel-title">Correlation of weekly sales with external drivers</div><span class="panel-note">Pearson r over this series</span></div>
        <div id="c-drv" class="chart"></div>
        <p class="hint">Correlation is not causation. Temperature mostly proxies the season (winter demand vs summer). Tree models (Random Forest, Gradient Boosting) use these drivers directly; the seasonal regression relies on calendar harmonics instead.</p></section>
    </div>
  </div>`;

  const c = colors();
  tabs($("#tabs"), (id) => { activeTab = id; });
  $$("[role=tab]").forEach((b) => { if (b.dataset.tab === activeTab) b.click(); });

  drawTrend("#c-trend", s, { showTrend });
  $$("[data-t]").forEach((b) => b.addEventListener("click", () => { showTrend = b.dataset.t === "1"; render(); }));

  const sp = eda.seasonal_profile;
  draw("#c-season", [
    { x: sp.map((r) => r.week), y: sp.map((r) => r.mean + r.std), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: sp.map((r) => r.week), y: sp.map((r) => Math.max(0, r.mean - r.std)), mode: "lines", fill: "tonexty", fillcolor: c.band, line: { width: 0 }, name: "±1 std dev", hoverinfo: "skip" },
    { x: sp.map((r) => r.week), y: sp.map((r) => r.mean), name: "Mean weekly sales", mode: "lines+markers", line: { color: c.actual, width: 2.4 }, marker: { size: 5 }, hovertemplate: "$%{y:,.0f}" },
  ], layout({ x: "Week of year (1 – 52)", y: "Average sales (USD)" }));

  draw("#c-box", [
    { y: h.box.non_holiday, name: "Non-holiday", type: "box", boxpoints: "all", jitter: 0.5, pointpos: 0, marker: { color: c.actual, size: 4 }, line: { color: c.actual } },
    { y: h.box.holiday, name: "Holiday", type: "box", boxpoints: "all", jitter: 0.5, pointpos: 0, marker: { color: c.holiday, size: 6 }, line: { color: c.holiday } },
  ], layout({ y: "Weekly sales (USD)", legend: false, hovermode: "closest" }));

  if (eda.markdown.dates.length) {
    const pal = [c.actual, c.rolling, c.sage, c.forecast, c.baseline];
    draw("#c-md", Object.entries(eda.markdown.series).map(([k, v], i) => ({ x: eda.markdown.dates, y: v, name: k, type: "bar", marker: { color: pal[i % pal.length] }, hovertemplate: "$%{y:,.0f}" })),
      layout({ x: "Week", y: "Total markdown (USD)", barmode: "stack" }));
  }
  draw("#c-promo", [{
    x: ["No markdown", "Active markdown"], y: [cmp.without_promo_mean, cmp.with_promo_mean], type: "bar",
    marker: { color: [c.baseline, c.forecast] }, text: [cmp.without_promo_mean, cmp.with_promo_mean].map((v) => fmt.usd(v)), textposition: "outside",
    customdata: [cmp.without_promo_n, cmp.with_promo_n], hovertemplate: "%{y:$,.0f} · %{customdata} weeks<extra></extra>",
  }], layout({ y: "Average weekly sales (USD)", legend: false, hovermode: "closest" }));

  const drv = [...eda.drivers].reverse();
  draw("#c-drv", [{
    y: drv.map((d) => d.feature), x: drv.map((d) => d.correlation), type: "bar", orientation: "h",
    marker: { color: drv.map((d) => (d.correlation >= 0 ? c.rolling : c.holiday)) },
    text: drv.map((d) => d.correlation.toFixed(2)), textposition: "outside", hovertemplate: "r = %{x:.3f}<extra></extra>",
  }], layout({ x: "Correlation with weekly sales", yfmt: "", legend: false, hovermode: "closest", xaxis: { range: [-1, 1], zeroline: true } }));
}

onSeriesChange(load);
onThemeChange(render);
load();
