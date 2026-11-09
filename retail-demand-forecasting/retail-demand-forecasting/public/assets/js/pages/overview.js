import { $, api, mount, setSubtitle, seriesLabel, state, seriesParams, onSeriesChange, onThemeChange, makeGuard, loadingHTML, errorHTML, fmt, kpi, table, drawTrend, metricFmt, icon, NAV, esc, getMeta } from "../core.js";

await mount({
  page: "overview", eyebrow: "Executive summary", title: "Retail demand, at a glance",
  body: `<div id="view"></div>`,
});

const guard = makeGuard();
let data = null;
let showTrend = false;

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Loading series, engineering features and validating 8 models…");
  setSubtitle(`${seriesLabel()} · computing live…`);
  try {
    const [series, ev] = await Promise.all([api("/api/series", { params: seriesParams() }), api("/api/evaluate", { params: seriesParams() })]);
    if (!guard.ok(id)) return;
    data = { series, ev };
    render();
  } catch (e) {
    if (guard.ok(id)) { $("#view").innerHTML = errorHTML(e); setSubtitle(seriesLabel()); }
  }
}

function render() {
  if (!data) return;
  const { series: s, ev } = data;
  const k = s.kpis, win = ev.winner.cv, meta = getMeta();
  setSubtitle(`${seriesLabel()} · store type <b>${esc(s.store_type)}</b> · ${k.observations} weekly observations from <b>${fmt.date(s.diagnostics.start_date)}</b> to <b>${fmt.date(s.diagnostics.end_date)}</b>.`);
  const tiles = NAV.filter((n) => n.id !== "overview").map((n) => `<a class="tile" href="${n.href}?store=${state.store}&dept=${state.dept}&metric=${state.metric}">${icon(n.icon)}<h3>${esc(n.label)}</h3><p>${esc(n.blurb)}</p></a>`).join("");

  $("#view").innerHTML = `
    <div class="note warn"><b>Domain notice:</b> <code>Weekly_Sales</code> is <b>monetary revenue (USD)</b>, not units sold. Inventory buffers are therefore expressed in dollars, or in units through a declared average selling price.</div>
    <div class="kpis">
      ${kpi("Historical revenue", fmt.usd(k.total_revenue), `${k.observations} weeks`)}
      ${kpi("Average weekly sales", fmt.usd(k.avg_weekly), `Std dev ${fmt.usd(k.std_weekly)}`)}
      ${kpi("Latest observed week", fmt.usd(k.latest_sales), fmt.date(k.latest_date))}
      ${kpi("Champion model", esc(win.name), `${esc(win.family)} family`, "accent", true)}
      ${kpi(`Champion ${ev.metric_label}`, metricFmt(state.metric, win[state.metric]), "Rolling-origin CV", "good")}
    </div>
    <section class="panel">
      <div class="panel-head"><div class="panel-title">Historical demand &amp; 4-week moving average</div>
        <div class="seg" role="group" aria-label="Trend line"><button aria-pressed="${!showTrend}" data-t="0">Hide trend</button><button aria-pressed="${showTrend}" data-t="1">Show linear trend</button></div></div>
      <div id="trend" class="chart tall"></div>
    </section>
    <div class="grid g-3-2">
      <section class="panel">
        <div class="panel-head"><div class="panel-title">Model selection summary · rolling-origin CV</div><a class="panel-note" href="/models">Full leaderboard →</a></div>
        ${table([
          { label: "#", get: (r) => r.rank ?? "–" },
          { label: "Model", html: (r) => `${esc(r.name)} ${r.rank === 1 ? '<span class="pill win">Champion</span>' : ""}` },
          { label: "WAPE", num: true, get: (r) => fmt.pct(r.wape) },
          { label: "MAE", num: true, get: (r) => fmt.usd(r.mae) },
          { label: "RMSE", num: true, get: (r) => fmt.usd(r.rmse) },
        ], ev.cv, { rowClass: (r) => (r.rank === 1 ? "win" : "") })}
      </section>
      <section class="panel">
        <div class="panel-head"><div class="panel-title">Data integrity diagnostics</div></div>
        <ul class="list-clean">
          <li><b>Records evaluated:</b> ${k.observations} weeks (${s.diagnostics.warmup_weeks} used as lag warm-up, ${s.diagnostics.model_rows} for modelling)</li>
          <li><b>Temporal gaps:</b> ${s.diagnostics.gap_count} non-weekly intervals</li>
          <li><b>Holiday weeks observed:</b> ${k.holiday_weeks}</li>
          <li><b>Peak demand:</b> ${fmt.usd(k.peak_sales)} <span class="hint">(${fmt.date(k.peak_date)})</span></li>
          <li><b>Lowest demand:</b> ${fmt.usd(k.low_sales)} <span class="hint">(${fmt.date(k.low_date)})</span></li>
          <li><b>Negative-sales weeks (returns):</b> ${k.negative_weeks}</li>
          <li><b>Look-ahead prevention:</b> lags and rolling statistics are shifted so week <i>t</i> never sees its own sales</li>
          <li><b>Dataset:</b> ${fmt.num(meta.dataset.rows, 0)} rows · ${meta.dataset.stores} stores · ${meta.dataset.departments} departments</li>
        </ul>
        <p class="hint" style="margin-top:12px">Computed in ${ev.elapsed_ms} ms (8 models × 2 validation schemes) + ${s.elapsed_ms} ms (series).</p>
      </section>
    </div>
    <div class="panel-title" style="margin:6px 0 10px">Explore the workspace</div>
    <div class="tiles">${tiles}</div>`;

  drawTrend("#trend", s, { showTrend });
  document.querySelectorAll("[data-t]").forEach((b) => b.addEventListener("click", () => { showTrend = b.dataset.t === "1"; render(); }));
}

onSeriesChange(load);
onThemeChange(render);
load();
