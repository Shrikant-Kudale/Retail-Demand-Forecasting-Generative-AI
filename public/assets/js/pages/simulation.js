import { $, $$, api, mount, setSubtitle, seriesLabel, state, onSeriesChange, onThemeChange, fmt, kpi, draw, layout, colors, esc, getMeta, sleep, toast } from "../core.js";

const STAGES = [
  ["1 · Ingest", "Load & validate weekly series"],
  ["2 · Engineer", "Lags, rolling stats, seasonality"],
  ["3 · Train", "Fit 8 candidate models live"],
  ["4 · Select", "Rank by rolling-origin CV"],
  ["5 · Replay", "Retrain, forecast, replenish weekly"],
];
const SPEEDS = { fast: 120, normal: 650, slow: 1400 };

await mount({
  page: "simulation", eyebrow: "Real-time simulation", title: "Live weekly replay",
  body: `
  <div class="note"><b>What this does.</b> The server replays the last weeks of history as if they were happening now. Each simulated week it <b>retrains the model on the past only</b>, forecasts ahead, places an order under the (R, S) policy, then reveals the actual demand and scores itself. Nothing is pre-computed - every step is a live call to the Python engine.</div>
  <div class="stepper" id="stepper">${STAGES.map((s, i) => `<div class="stage" id="st${i}"><div class="n">${s[0]}</div><div class="t">${s[1]}</div><div class="d" id="sd${i}"></div></div>`).join("")}</div>
  <section class="panel">
    <div class="controls-row">
      <div class="field" style="margin:0"><label>Replay window: <output id="o-w">26</output> weeks</label><input id="i-w" type="range" min="8" max="40" value="26"></div>
      <div class="field" style="margin:0"><label>Lead time: <output id="o-l">2</output> wk</label><input id="i-l" type="range" min="1" max="6" value="2"></div>
      <div class="field" style="margin:0"><label>Review: <output id="o-r">1</output> wk</label><input id="i-r" type="range" min="1" max="4" value="1"></div>
      <div class="field" style="margin:0"><label>Service level</label><select id="i-s"></select></div>
      <div class="field" style="margin:0"><label>Model</label><select id="i-m"></select></div>
      <div class="field" style="margin:0"><label>Speed</label><select id="i-v"><option value="fast">Fast</option><option value="normal" selected>Normal</option><option value="slow">Slow</option></select></div>
      <label class="hint" style="display:flex;gap:6px;align-items:center"><input type="checkbox" id="i-c" checked> Compare with naive baseline</label>
    </div>
    <div class="controls-row" style="margin-top:14px">
      <button class="btn primary" id="b-start">▶ Run pipeline &amp; start</button>
      <button class="btn" id="b-pause" disabled>❚❚ Pause</button>
      <button class="btn" id="b-step">Step once</button>
      <button class="btn" id="b-reset">Reset</button>
      <span class="hint" id="b-msg"></span>
    </div>
  </section>
  <section class="panel" id="train-panel"><div class="panel-head"><div class="panel-title">Candidate training · live</div><span class="panel-note" id="tr-note">Waiting to start</span></div>
    <div class="progress"><i id="tr-bar"></i></div><div class="model-chips" id="chips"></div></section>
  <div class="kpis" id="kpis"></div>
  <section class="panel"><div class="panel-head"><div class="panel-title">Forecast vs actual · retrained every week</div><span class="panel-note" id="fan-note"></span></div><div id="c-a" class="chart tall"></div></section>
  <div class="grid g2">
    <section class="panel"><div class="panel-head"><div class="panel-title">Running WAPE (lower is better)</div></div><div id="c-b" class="chart short"></div></section>
    <section class="panel"><div class="panel-head"><div class="panel-title">Inventory on hand vs policy</div></div><div id="c-c" class="chart short"></div></section>
  </div>
  <section class="panel"><div class="panel-head"><div class="panel-title">Event log</div></div><div class="terminal" id="log" role="log" aria-live="off"></div></section>`,
});

// ------------------------------------------------------------------ state
let runToken = 0, busy = false, mode = "idle", pipelineDone = false;
let bounds = null, champion = null, origins = [], idx = 0, H = 4, R = null, ctxDates = [], ctxSales = [], lastFan = null;
const modelNames = Object.fromEntries(getMeta().models.map((m) => [m.key, m.name]));

$("#i-s").innerHTML = getMeta().service_levels.map((l) => `<option value="${l}" ${l === 95 ? "selected" : ""}>${l}%</option>`).join("");
$("#i-m").innerHTML = `<option value="auto">Champion (auto-selected)</option>` + getMeta().models.map((m) => `<option value="${m.key}">${esc(m.name)}</option>`).join("");
const cfg = () => ({ W: +$("#i-w").value, L: +$("#i-l").value, Rv: +$("#i-r").value, S: +$("#i-s").value, speed: SPEEDS[$("#i-v").value], compare: $("#i-c").checked, pick: $("#i-m").value });
[["i-w", "o-w"], ["i-l", "o-l"], ["i-r", "o-r"]].forEach(([i, o]) => $("#" + i).addEventListener("input", (e) => { $("#" + o).textContent = e.target.value; }));

const blankTrack = () => ({ state: null, weeks: [], actual: [], f1: [], lo: [], hi: [], wape: [], onhand: [], rop: [], target: [], stock: [], kpis: null });

// ------------------------------------------------------------------ UI helpers
function stage(i, status, detail) {
  const el = $("#st" + i); el.classList.remove("run", "done"); if (status) el.classList.add(status);
  if (detail != null) $("#sd" + i).textContent = detail;
}
function log(html, cls = "") {
  const box = $("#log"), d = document.createElement("div"); d.className = `ln ${cls}`; d.innerHTML = html; box.append(d);
  while (box.children.length > 250) box.firstChild.remove(); box.scrollTop = box.scrollHeight;
}
const ts = (t) => `<span class="t">[${t}]</span>`;
function setButtons() {
  $("#b-start").disabled = mode === "playing" || mode === "pipeline" || mode === "done";
  $("#b-pause").disabled = mode !== "playing";
  $("#b-step").disabled = busy || mode === "playing" || mode === "pipeline" || mode === "done";
  $("#b-start").textContent = mode === "paused" ? "▶ Resume" : "▶ Run pipeline & start";
}
function resetUI(msg = "") {
  runToken++; busy = false; mode = "idle"; pipelineDone = false; champion = null; origins = []; idx = 0; R = null; lastFan = null;
  STAGES.forEach((_, i) => { stage(i, null, ""); });
  $("#chips").innerHTML = ""; $("#tr-bar").style.width = "0"; $("#tr-note").textContent = "Waiting to start";
  $("#log").innerHTML = ""; $("#kpis").innerHTML = ""; $("#fan-note").textContent = ""; $("#b-msg").textContent = msg;
  ["c-a", "c-b", "c-c"].forEach((id) => { const n = $("#" + id); if (window.Plotly && n.data) window.Plotly.purge(n); n.innerHTML = ""; });
  setButtons();
}

// ------------------------------------------------------------------ pipeline (stages 1-4)
async function runPipeline() {
  const token = ++runToken, c = cfg(); mode = "pipeline"; setButtons();
  const alive = () => token === runToken;
  try {
    stage(0, "run"); log(`${ts("ingest")} loading Store ${state.store} / Dept ${state.dept} from the merged weekly dataset…`);
    bounds = await api("/api/simulate/bounds", { params: { store: state.store, dept: state.dept } });
    if (!alive()) return false;
    stage(0, "done", `${bounds.n} weeks`); log(`${ts("ingest")} ${bounds.n} weekly observations ${fmt.date(bounds.dates[0])} → ${fmt.date(bounds.dates.at(-1))}`, "ok");

    stage(1, "run"); await sleep(250);
    const warm = 52; stage(1, "done", `${bounds.n - warm} model rows`);
    log(`${ts("features")} 4 lags + Lag_52 + shifted rolling means + Fourier week terms + holiday & markdown flags (first ${warm} weeks used as warm-up)`, "ok");

    stage(2, "run"); const models = getMeta().models; const done = {};
    $("#chips").innerHTML = models.map((m) => `<span class="chip" id="ch-${m.key}">${esc(m.name)} <b></b></span>`).join("");
    $("#tr-note").textContent = `0 / ${models.length}`;
    for (let i = 0; i < models.length; i++) {
      const m = models[i]; const chip = $("#ch-" + m.key); chip.classList.add("run");
      log(`${ts("train")} fitting ${esc(m.name)} (${esc(m.family)}) + 4-fold rolling-origin validation…`);
      const r = await api("/api/evaluate/model", { params: { store: state.store, dept: state.dept, model: m.key } });
      if (!alive()) return false;
      done[m.key] = r; chip.classList.replace("run", "done"); chip.querySelector("b").textContent = `${fmt.pct(r.cv.wape)} · ${r.elapsed_ms} ms`;
      $("#tr-bar").style.width = `${((i + 1) / models.length) * 100}%`; $("#tr-note").textContent = `${i + 1} / ${models.length}`;
      log(`${ts("train")} ${esc(m.name)}: CV WAPE ${fmt.pct(r.cv.wape)} · holdout ${fmt.pct(r.holdout.wape)} (${r.elapsed_ms} ms)`, r.cv.status === "Valid" ? "ok" : "bad");
    }
    stage(2, "done", `${models.length} models`);

    stage(3, "run"); await sleep(200);
    const valid = Object.values(done).filter((r) => r.cv.status === "Valid" && r.cv.wape != null).sort((a, b) => a.cv.wape - b.cv.wape);
    champion = valid[0].key; stage(3, "done", modelNames[champion]);
    log(`${ts("select")} champion = ${esc(modelNames[champion])} (CV WAPE ${fmt.pct(valid[0].cv.wape)})`, "ok");
    $$(".chip", $("#chips")).forEach((ch) => ch.id === "ch-" + champion && (ch.style.borderColor = "var(--amber)"));

    const start = Math.max(bounds.min_origin, bounds.n - c.W);
    origins = Array.from({ length: bounds.n - start }, (_, k) => start + k); idx = 0;
    ctxDates = bounds.dates.slice(Math.max(0, start - 20), start); ctxSales = bounds.sales.slice(Math.max(0, start - 20), start);
    R = { main: blankTrack(), base: blankTrack() }; H = Math.max(4, c.L + c.Rv);
    pipelineDone = true; stage(4, "run", `0 / ${origins.length} weeks`);
    log(`${ts("replay")} replaying ${origins.length} weeks from ${fmt.date(bounds.dates[start])}; policy L=${c.L}, R=${c.Rv}, service ${c.S}%`, "ok");
    return true;
  } catch (e) {
    if (alive()) { mode = "idle"; setButtons(); log(`${ts("error")} ${esc(e.message)}`, "bad"); toast(e.message, true); }
    return false;
  }
}

// ------------------------------------------------------------------ replay (stage 5)
function push(track, r) {
  track.state = r.state; track.weeks.push(r.date); track.actual.push(r.actual); track.f1.push(r.forecast[0].forecast);
  track.lo.push(r.forecast[0].lower); track.hi.push(r.forecast[0].upper); track.wape.push(r.kpis.running_wape);
  track.onhand.push(r.inventory.on_hand_end); track.rop.push(r.inventory.reorder_point); track.target.push(r.inventory.target_inventory);
  track.stock.push(r.inventory.stockout); track.kpis = r.kpis; track.last = r;
}
async function stepOnce() {
  if (!pipelineDone || idx >= origins.length) return false;
  const token = runToken, c = cfg(), origin = origins[idx], key = c.pick === "auto" ? champion : c.pick;
  const body = (m, st) => ({ store: state.store, dept: state.dept, model: m, origin, horizon: H, lead_time: c.L, review: c.Rv, service_level: c.S, state: st });
  busy = true; setButtons();
  try {
    const calls = [api("/api/simulate/step", { method: "POST", body: body(key, R.main.state) })];
    if (c.compare && key !== "naive") calls.push(api("/api/simulate/step", { method: "POST", body: body("naive", R.base.state) }));
    const [m, b] = await Promise.all(calls);
    if (token !== runToken) return false;
    push(R.main, m); if (b) push(R.base, b); lastFan = m; idx++;
    const i = m.inventory, err = m.pct_error;
    log(`${ts(m.date)} retrained <b>${esc(m.model.name)}</b> on ${m.trained_on_weeks} wks (${m.fit_ms} ms) → forecast ${fmt.usd(m.forecast[0].forecast)} · actual ${fmt.usd(m.actual)} (${err == null ? "-" : fmt.pct(err, 1)} off) · ` +
        (i.order_qty > 0 ? `ordered ${fmt.usd(i.order_qty)} (arrives ${i.order_arrives}) · ` : "no order · ") + `on-hand ${fmt.usd(i.on_hand_end)}` + (i.stockout ? ` · <span class="bad">STOCKOUT, lost ${fmt.usd(i.unmet)}</span>` : ""), i.stockout ? "bad" : "");
    stage(4, "run", `${idx} / ${origins.length} weeks`);
    if (m.done || idx >= origins.length) { mode = "done"; stage(4, "done", `${origins.length} weeks`); log(`${ts("done")} replay complete - fill rate ${fmt.pct(m.kpis.fill_rate, 1)}, running WAPE ${fmt.pct(m.kpis.running_wape)}`, "ok"); }
    drawAll();
    return true;
  } catch (e) {
    if (token === runToken) { mode = "paused"; log(`${ts("error")} ${esc(e.message)}`, "bad"); toast(e.message, true); }
    return false;
  } finally { busy = false; setButtons(); }
}
async function play() {
  const token = runToken; mode = "playing"; setButtons();
  while (token === runToken && mode === "playing" && idx < origins.length) {
    const ok = await stepOnce(); if (!ok) break;
    if (mode === "playing" && idx < origins.length) await sleep(cfg().speed);
  }
  if (token === runToken && mode === "playing") { mode = idx >= origins.length ? "done" : "paused"; setButtons(); }
}

// ------------------------------------------------------------------ drawing
function drawAll() {
  if (!R || !R.main.weeks.length) return;
  // Plotly.react diffs by array identity, so hand it fresh copies of the growing series
  const copy = (t) => ({ ...t, ...Object.fromEntries(["weeks", "actual", "f1", "lo", "hi", "wape", "onhand", "rop", "target", "stock"].map((k) => [k, t[k].slice()])) });
  const c = colors(), M = copy(R.main), B = copy(R.base), cmp = B.weeks.length > 0, k = M.kpis, kb = B.kpis;
  const dlt = (a, b, lowerBetter, f) => (kb && a != null && b != null ? `naive ${f(b)}` : "");
  $("#kpis").innerHTML = [
    kpi("Weeks simulated", `${k.steps} / ${origins.length}`, esc(M.last.model.name), "accent", false),
    kpi("Running WAPE", fmt.pct(k.running_wape), dlt(k.running_wape, kb?.running_wape, true, (v) => fmt.pct(v)), "good"),
    kpi("Fill rate", fmt.pct(k.fill_rate, 1), dlt(k.fill_rate, kb?.fill_rate, false, (v) => fmt.pct(v, 1))),
    kpi("Stockout weeks", String(k.stockout_weeks), `Lost sales ${fmt.usd(k.lost_sales)}` + (kb ? ` · naive ${kb.stockout_weeks}` : ""), k.stockout_weeks ? "bad" : "good"),
    kpi("Avg inventory", fmt.usd(k.avg_inventory), kb ? `naive ${fmt.usd(kb.avg_inventory)}` : ""),
  ].join("");

  const fan = lastFan.forecast, lastW = M.weeks.at(-1);
  $("#fan-note").textContent = `Dashed band: ${fan.length}-week outlook issued at ${fmt.date(lastFan.date)} (σ from ${lastFan.sigma_source})`;
  const traces = [
    { x: ctxDates, y: ctxSales, name: "History (context)", mode: "lines", line: { color: c.baseline, width: 2 }, hovertemplate: "$%{y:,.0f}" },
    { x: M.weeks, y: M.hi, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: M.weeks, y: M.lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.interval, name: "1-wk 95% interval", hoverinfo: "skip" },
    { x: M.weeks, y: M.actual, name: "Actual (revealed)", mode: "lines+markers", line: { color: c.actual, width: 2.4 }, marker: { size: 5 }, hovertemplate: "$%{y:,.0f}" },
    { x: M.weeks, y: M.f1, name: "1-week-ahead forecast", mode: "lines+markers", line: { color: c.forecast, width: 2.2 }, marker: { size: 5, symbol: "diamond" }, hovertemplate: "$%{y:,.0f}" },
  ];
  if (cmp) traces.push({ x: B.weeks, y: B.f1, name: "Naive baseline", mode: "lines", line: { color: c.baseline, width: 1.8, dash: "dot" }, hovertemplate: "$%{y:,.0f}" });
  traces.push(
    { x: fan.map((r) => r.date), y: fan.map((r) => r.upper), mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: fan.map((r) => r.date), y: fan.map((r) => r.lower), mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.band, name: "Current outlook band", hoverinfo: "skip" },
    { x: fan.map((r) => r.date), y: fan.map((r) => r.forecast), name: "Current outlook", mode: "lines+markers", line: { color: c.rolling, width: 2.4, dash: "dash" }, marker: { size: 5 }, hovertemplate: "$%{y:,.0f}" },
  );
  draw("#c-a", traces, layout({ x: "Week", y: "Weekly sales (USD)", shapes: [{ type: "line", xref: "x", yref: "paper", x0: lastW, x1: lastW, y0: 0, y1: 1, line: { color: c.muted, width: 1, dash: "dot" } }] }));

  const xr = { type: "date", range: [bounds.dates[origins[0]], bounds.dates[origins.at(-1)]] };   // fixed replay window
  const wt = [{ x: M.weeks, y: M.wape, name: M.last.model.name, mode: "lines+markers", line: { color: c.forecast, width: 2.6 }, marker: { size: 5 }, hovertemplate: "%{y:.2f}%" }];
  if (cmp) wt.push({ x: B.weeks, y: B.wape, name: "Naive baseline", mode: "lines", line: { color: c.baseline, width: 2, dash: "dot" }, hovertemplate: "%{y:.2f}%" });
  draw("#c-b", wt, layout({ x: "Week", y: "Running WAPE (%)", yfmt: "%", xaxis: xr }));

  const stockX = M.weeks.filter((_, j) => M.stock[j]), stockY = M.onhand.filter((_, j) => M.stock[j]);
  const it = [
    { x: M.weeks, y: M.onhand, name: "On hand (end of week)", mode: "lines", fill: "tozeroy", fillcolor: c.band, line: { color: c.rolling, width: 2.4, shape: "hv" }, hovertemplate: "$%{y:,.0f}" },
    { x: M.weeks, y: M.target, name: "Order-up-to S", mode: "lines", line: { color: c.forecast, width: 1.8, dash: "dash", shape: "hv" }, hovertemplate: "$%{y:,.0f}" },
    { x: M.weeks, y: M.rop, name: "Reorder point", mode: "lines", line: { color: c.actual, width: 1.4, dash: "dot", shape: "hv" }, hovertemplate: "$%{y:,.0f}" },
    { x: stockX, y: stockY, name: "Stockout", mode: "markers", marker: { color: c.holiday, size: 11, symbol: "x" }, hovertemplate: "Stockout" },
  ];
  if (cmp) it.push({ x: B.weeks, y: B.onhand, name: "On hand · naive", mode: "lines", line: { color: c.baseline, width: 1.8, shape: "hv" }, hovertemplate: "$%{y:,.0f}" });
  draw("#c-c", it, layout({ x: "Week", y: "Inventory (USD)", xaxis: xr }));
}

// ------------------------------------------------------------------ wiring
$("#b-start").addEventListener("click", async () => {
  if (mode === "paused") return play();
  resetUI(); if (await runPipeline()) play();
});
$("#b-pause").addEventListener("click", () => { mode = "paused"; setButtons(); $("#b-msg").textContent = "Paused - press Resume or Step once."; });
$("#b-step").addEventListener("click", async () => {
  if (!pipelineDone) { resetUI(); if (!(await runPipeline())) return; mode = "paused"; setButtons(); }
  await stepOnce(); if (mode !== "done") mode = "paused"; setButtons();
});
$("#b-reset").addEventListener("click", () => resetUI());
["i-w", "i-l", "i-r", "i-s", "i-m"].forEach((id) => $("#" + id).addEventListener("change", () => { if (pipelineDone || mode !== "idle") resetUI("Settings changed - press Run to start a new replay."); }));
onSeriesChange(() => resetUI("Selection changed - press Run to start a new replay."));
onThemeChange(() => drawAll());

setSubtitle(`${seriesLabel()} · press <b>Run pipeline &amp; start</b> to watch the full production loop execute step by step.`);
onSeriesChange(() => setSubtitle(`${seriesLabel()} · press <b>Run pipeline &amp; start</b> to watch the full production loop execute step by step.`));
resetUI();
