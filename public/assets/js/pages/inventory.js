import { $, $$, api, mount, setSubtitle, seriesLabel, state, seriesParams, onSeriesChange, onThemeChange, makeGuard, loadingHTML, errorHTML, fmt, kpi, table, draw, layout, colors, esc, getMeta, debounce } from "../core.js";

await mount({
  page: "inventory", eyebrow: "Replenishment policy", title: "Inventory planning",
  body: `<div id="view"></div>`,
});

const guard = makeGuard();
let defaults = null, out = null;
const P = { mode: "money", asp: 25, forecast: 0, std: 0, lead: 2, review: 1, service: 95, onhand: 0, onorder: 0, back: 0 };

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Forecasting next week to seed the planning scenario…");
  setSubtitle(`${seriesLabel()} · periodic-review (R, S) policy seeded from the champion model's first forecast week.`);
  try {
    const f = await api("/api/forecast", { params: seriesParams({ horizon: 4 }) });
    if (!guard.ok(id)) return;
    defaults = f;
    P.forecast = Math.round(f.forecast[0].forecast); P.std = Math.round(f.demand_std);
    P.onhand = Math.round(P.forecast * 1.2); P.onorder = Math.round(P.forecast * 0.8); P.back = 0;
    shell(); await recompute();
  } catch (e) { if (guard.ok(id)) $("#view").innerHTML = errorHTML(e); }
}

const money = () => P.mode === "money";
const unit = () => (money() ? "$" : "units");
const asUnits = (v) => (money() ? v : v / P.asp);
const show = (v) => (money() ? fmt.usd(v) : `${fmt.num(v, 0)} units`);

function shell() {
  const lv = getMeta().service_levels;
  $("#view").innerHTML = `
  <div class="note ${money() ? "warn" : ""}"><b>${money() ? "Monetary demand planning scenario." : "Unit conversion scenario."}</b>
    ${money() ? "All quantities are US dollars of retail revenue. Cost columns equal the dollar order value." : `Dollars are converted to units at an assumed average selling price of <b>$${P.asp}</b> per unit (adjustable). Treat unit counts as illustrative.`}</div>
  <div class="grid g-3-2">
    <section class="panel">
      <div class="panel-head"><div class="panel-title">Scenario inputs</div>
        <div class="seg"><button data-mode="money" aria-pressed="${money()}">Monetary ($)</button><button data-mode="units" aria-pressed="${!money()}">Units</button></div></div>
      <div class="grid g2" style="gap:14px 22px">
        <div>
          ${!money() ? `<div class="field"><label>Average selling price ($ / unit)</label><input type="number" id="i-asp" min="1" step="1" value="${P.asp}"></div>` : ""}
          <div class="field"><label>Expected weekly demand (${unit()})</label><input type="number" id="i-forecast" min="0" step="any" value="${asUnits(P.forecast).toFixed(money() ? 0 : 1)}"><span class="hint">Seeded from forecast week 1</span></div>
          <div class="field"><label>Weekly demand std dev (${unit()})</label><input type="number" id="i-std" min="0" step="any" value="${asUnits(P.std).toFixed(money() ? 0 : 1)}"></div>
          <div class="field"><label>Service level</label><select id="i-service">${lv.map((l) => `<option value="${l}" ${l === P.service ? "selected" : ""}>${l}%</option>`).join("")}</select></div>
          <div class="field"><label>Lead time: <output id="o-lead">${P.lead}</output> weeks</label><input type="range" id="i-lead" min="1" max="8" value="${P.lead}"></div>
        </div>
        <div>
          <div class="field"><label>Review period: <output id="o-rev">${P.review}</output> weeks</label><input type="range" id="i-review" min="1" max="4" value="${P.review}"></div>
          <div class="field"><label>Current on-hand (${unit()})</label><input type="number" id="i-onhand" min="0" step="any" value="${asUnits(P.onhand).toFixed(money() ? 0 : 1)}"></div>
          <div class="field"><label>Incoming open orders (${unit()})</label><input type="number" id="i-onorder" min="0" step="any" value="${asUnits(P.onorder).toFixed(money() ? 0 : 1)}"></div>
          <div class="field"><label>Backorders (${unit()})</label><input type="number" id="i-back" min="0" step="any" value="${asUnits(P.back).toFixed(money() ? 0 : 1)}"></div>
          <button class="btn" id="reset">Reset to forecast</button>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-head"><div class="panel-title">Policy equations</div></div>
      <div class="formula">SS = Z · σ · √(L + R)</div>
      <div class="formula">ROP = d · L + SS</div>
      <div class="formula">S = d · (L + R) + SS</div>
      <div class="formula">Q = max(0, S − (On-hand + On-order − Backorders))</div>
      <p class="hint">d: weekly demand · σ: weekly demand std · L: lead time · R: review period · Z: normal quantile of the service level · S: order-up-to level.</p>
    </section>
  </div>
  <div id="results"></div>`;

  $$("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    const next = b.dataset.mode; if (next === P.mode) return; P.mode = next; shell(); recompute();
  }));
  const bind = (id, key, conv = true) => $(id)?.addEventListener("input", (e) => {
    const v = Number(e.target.value); if (Number.isNaN(v)) return;
    P[key] = conv && !money() ? v * P.asp : v;
    if (id === "#i-lead") $("#o-lead").textContent = v; if (id === "#i-review") $("#o-rev").textContent = v;
    debRecompute();
  });
  bind("#i-forecast", "forecast"); bind("#i-std", "std"); bind("#i-onhand", "onhand"); bind("#i-onorder", "onorder"); bind("#i-back", "back");
  bind("#i-lead", "lead", false); bind("#i-review", "review", false);
  $("#i-service").addEventListener("change", (e) => { P.service = Number(e.target.value); recompute(); });
  $("#i-asp")?.addEventListener("input", (e) => { const v = Number(e.target.value); if (v > 0) { P.asp = v; debRecompute(); } });
  $("#reset").addEventListener("click", load);
}

async function recompute() {
  try {
    out = await api("/api/inventory", { method: "POST", body: {
      forecast_weekly: P.forecast, demand_std: P.std, lead_time_weeks: P.lead, review_period_weeks: P.review,
      service_level_pct: P.service, current_on_hand: P.onhand, incoming_orders: P.onorder, backorders: P.back,
      unit_cost: money() ? 1 : P.asp, is_monetary_scenario: money(),
    } });
    renderResults();
  } catch (e) { $("#results").innerHTML = errorHTML(e); }
}
const debRecompute = debounce(recompute, 220);

function renderResults() {
  if (!out) return;
  const p = out.policy, buy = p.order_quantity > 0;
  const conv = (v) => (money() ? v : v / P.asp);
  $("#results").innerHTML = `
  <div class="kpis k4" style="margin-top:18px">
    ${kpi("Safety stock", show(conv(p.safety_stock)), `Z = ${p.z_score.toFixed(3)} at ${p.service_level_pct}%`)}
    ${kpi("Reorder point", show(conv(p.reorder_point)), `Lead-time demand ${show(conv(p.lead_time_demand))}`)}
    ${kpi("Order-up-to level (S)", show(conv(p.target_inventory)), `Protection ${p.lead_time_weeks + p.review_period_weeks} weeks`, "accent")}
    ${kpi(buy ? "Recommended order" : "Order status", buy ? show(conv(p.order_quantity)) : "Stock sufficient", `Inventory position ${show(conv(p.current_inventory_position))}`, buy ? "bad" : "good")}
  </div>
  <div class="grid g2">
    <section class="panel"><div class="panel-head"><div class="panel-title">Inventory position vs policy levels</div></div><div id="c-pos" class="chart"></div></section>
    <section class="panel"><div class="panel-head"><div class="panel-title">How the service level moves safety stock</div></div><div id="c-svc" class="chart"></div></section>
  </div>
  <div class="grid g2" style="margin-top:18px">
    <section class="panel"><div class="panel-head"><div class="panel-title">Lead-time sensitivity</div></div><div id="c-lead" class="chart"></div></section>
    <section class="panel"><div class="panel-head"><div class="panel-title">Service-level table</div></div>
      ${table([{ label: "Service", get: (r) => `${r.service_level}%` }, { label: "Z", num: true, get: (r) => r.z.toFixed(2) }, { label: "Safety stock", num: true, get: (r) => show(conv(r.safety_stock)) },
        { label: "Reorder point", num: true, get: (r) => show(conv(r.reorder_point)) }, { label: "Order Q", num: true, get: (r) => show(conv(r.order_quantity)) }], out.service_sensitivity, { rowClass: (r) => (r.service_level === p.service_level_pct ? "win" : "") })}</section>
  </div>
  <div class="note"><b>Trade-off:</b> raising the service level from 90% to 99% multiplies safety stock by about ${(2.326 / 1.282).toFixed(1)}×. Longer lead times and review periods grow the buffer with the <i>square root</i> of exposure time; the order-up-to level grows linearly.</div>`;

  const c = colors();
  const pos = [
    ["On-hand", p.current_inventory_position - (P.onorder - P.back), c.rolling], ["On-order", P.onorder, c.sage], ["Backorders", P.back, c.holiday],
    ["Inventory position", p.current_inventory_position, c.actual], ["Reorder point", p.reorder_point, c.baseline], ["Order-up-to (S)", p.target_inventory, c.forecast], ["Order quantity", p.order_quantity, c.holiday],
  ];
  draw("#c-pos", [{ y: pos.map((r) => r[0]).reverse(), x: pos.map((r) => conv(r[1])).reverse(), type: "bar", orientation: "h", marker: { color: pos.map((r) => r[2]).reverse() },
    text: pos.map((r) => show(conv(r[1]))).reverse(), textposition: "outside", cliponaxis: false, hovertemplate: "%{text}<extra></extra>" }],
    layout({ x: unit(), yfmt: "", legend: false, hovermode: "closest", margin: { l: 130, r: 90, t: 10, b: 46 }, xaxis: { tickprefix: money() ? "$" : "" } }));
  const s = out.service_sensitivity;
  draw("#c-svc", [
    { x: s.map((r) => `${r.service_level}%`), y: s.map((r) => conv(r.safety_stock)), name: "Safety stock", type: "bar", marker: { color: c.rolling } },
    { x: s.map((r) => `${r.service_level}%`), y: s.map((r) => conv(r.reorder_point)), name: "Reorder point", mode: "lines+markers", line: { color: c.forecast, width: 2.5 } },
  ], layout({ x: "Service level", y: money() ? "USD" : "Units", yfmt: money() ? "$" : "", hovermode: "x unified", xaxis: { type: "category" } }));
  const l = out.lead_time_sensitivity;
  draw("#c-lead", [
    { x: l.map((r) => r.lead_time), y: l.map((r) => conv(r.safety_stock)), name: "Safety stock", mode: "lines+markers", line: { color: c.rolling, width: 2.5 } },
    { x: l.map((r) => r.lead_time), y: l.map((r) => conv(r.target_inventory)), name: "Order-up-to level S", mode: "lines+markers", line: { color: c.forecast, width: 2.5 } },
  ], layout({ x: "Lead time (weeks)", y: money() ? "USD" : "Units", yfmt: money() ? "$" : "" }));
}

onSeriesChange(load);
onThemeChange(renderResults);
load();
