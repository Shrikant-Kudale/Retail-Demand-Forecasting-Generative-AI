import { $, api, mount, setSubtitle, seriesLabel, state, seriesParams, onSeriesChange, makeGuard, loadingHTML, errorHTML, fmt, esc, toast } from "../core.js";

await mount({
  page: "insights", eyebrow: "Findings · caveats · AI analyst", title: "Insights & takeaways",
  body: `<div id="view"></div>`,
});

const guard = makeGuard();
let ev = null, eda = null, ai = null;

async function load() {
  const id = guard.next();
  $("#view").innerHTML = loadingHTML("Collecting findings from live results…");
  setSubtitle(`${seriesLabel()} · what the evidence says, what it does not, and how to act on it.`);
  try {
    const [e, d, s] = await Promise.all([api("/api/evaluate", { params: seriesParams() }), api("/api/eda", { params: seriesParams() }), api("/api/ai/status")]);
    if (!guard.ok(id)) return;
    ev = e; eda = d; ai = s; render();
  } catch (e) { if (guard.ok(id)) $("#view").innerHTML = errorHTML(e); }
}

function render() {
  const w = ev.winner.cv, cmp = ev.comparison, h = eda.holiday;
  const naive = ev.cv.find((r) => r.key === "naive");
  const holN = ev.residuals[ev.winner.holdout.key]?.holiday_stats?.Holiday?.n ?? 0;
  $("#view").innerHTML = `
  <div class="grid g3">
    <div class="insight amber"><h3>1 · Empirical model superiority</h3>
      <p><b>${esc(w.name)}</b> wins rolling-origin validation with <b>${fmt.pct(w.wape)} WAPE</b> (MAE ${fmt.usd(w.mae)}).</p>
      <ul class="list-clean">${naive ? `<li>${fmt.pct((naive.wape - w.wape) / naive.wape * 100, 0)} fewer errors than repeating last week (${fmt.pct(naive.wape)}).</li>` : ""}${cmp ? `<li>Seasonal features cut the original 4-lag error from ${fmt.pct(cmp.baseline_wape)} to ${fmt.pct(cmp.improved_wape)} (−${cmp.reduction_pct}%).</li>` : ""}<li>Ranking is repeated across 4 folds, not decided by one lucky split.</li></ul></div>
    <div class="insight"><h3>2 · Retail shocks &amp; holidays</h3>
      <p>Holiday weeks average <b>${fmt.signed(h.uplift_pct)}</b> versus normal weeks for this department (${h.holiday_n} weeks observed).</p>
      <ul class="list-clean"><li>Effects differ sharply by department - Thanksgiving, Christmas and Super Bowl weeks behave differently.</li><li>The holdout window holds only <b>${holN}</b> holiday week${holN === 1 ? "" : "s"}: treat holiday-only accuracy as anecdotal.</li></ul></div>
    <div class="insight"><h3>3 · Operational shock management</h3>
      <ul class="list-clean"><li>Hold extra safety stock one to two weeks before known peaks.</li><li>Raise service level only where lost margin outweighs holding cost - the buffer grows with Z.</li><li>Use the weekly re-forecast; prediction intervals widen with horizon.</li></ul></div>
    <div class="insight"><h3>4 · Weekly retraining strategy</h3>
      <ol class="list-clean"><li>Ingest the latest week of sales.</li><li>Refit the champion on all history (seconds).</li><li>Re-forecast and update intervals.</li><li>Recompute reorder point and order size.</li></ol>
      <p class="hint">Watch it run in the <a href="/simulation">Live Simulation</a>.</p></div>
    <div class="insight"><h3>5 · Transferring to other markets</h3>
      <p>The pipeline is calendar-agnostic: replace the US holiday flags with local events (festival and wedding seasons, monsoon, regional sales) and keep the lag + seasonality structure.</p></div>
    <div class="insight amber"><h3>6 · Limitations</h3>
      <ul class="list-clean"><li>Monetary revenue, not units - price changes and inflation blur demand.</li><li>Short histories (≈2.7 years) limit annual-cycle estimation.</li><li>Weather, CPI and fuel are treated as known covariates in future weeks.</li><li>Lost sales assumed (no backorders) in the simulation.</li></ul></div>
  </div>

  <section class="panel" style="margin-top:18px">
    <div class="panel-head"><div class="panel-title">AI analyst</div>
      <span class="panel-note"><span class="status-dot ${ai.configured ? "" : "off"}" style="margin-right:6px"></span>${ai.configured ? `Connected · ${esc(ai.provider)} (${esc(ai.model)})` : "Not connected · showing the built-in rule-based analyst"}</span></div>
    <p class="hint">Answers are grounded only on numbers computed by the forecasting engine for the selected store and department. The API key stays on the server.</p>
    <div class="controls-row" style="margin-bottom:14px"><button class="btn primary" id="brief">Generate briefing</button></div>
    <div id="brief-out"></div>
    <div class="chat-log" id="chat"></div>
    <div class="controls-row" style="align-items:stretch"><textarea id="q" maxlength="500" rows="2" placeholder="${ai.configured ? "Ask: which weeks need extra stock? How reliable is this forecast?" : "Add XAI_API_KEY (Grok) or GEMINI_API_KEY to enable free-form questions"}" ${ai.configured ? "" : "disabled"} style="flex:1"></textarea>
      <button class="btn amber" id="ask" ${ai.configured ? "" : "disabled"}>Ask</button></div>
  </section>`;
  $("#brief").addEventListener("click", briefing);
  $("#ask").addEventListener("click", ask);
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
}

async function briefing() {
  const btn = $("#brief"); btn.disabled = true; btn.textContent = "Analysing…";
  try {
    const r = await api("/api/ai/briefing", { method: "POST", body: { store: state.store, dept: state.dept, metric: state.metric } });
    const box = $("#brief-out"); box.innerHTML = "";
    if (r.ai_text) { const b = document.createElement("div"); b.className = "bubble"; b.textContent = r.ai_text; const t = document.createElement("div"); t.className = "panel-title"; t.textContent = `AI briefing · ${r.ai.provider} (${r.ai.model})`; t.style.margin = "4px 0 8px"; box.append(t, b); }
    const t2 = document.createElement("div"); t2.className = "panel-title"; t2.style.margin = "14px 0 8px"; t2.textContent = "Rule-based analyst notes"; box.append(t2);
    r.rule_based.forEach((n) => { const d = document.createElement("div"); d.className = "bubble"; d.style.marginBottom = "8px"; const b = document.createElement("b"); b.textContent = n.title + ": "; d.append(b, document.createTextNode(n.text)); box.append(d); });
  } catch (e) { toast(e.message, true); }
  btn.disabled = false; btn.textContent = "Generate briefing";
}

async function ask() {
  const q = $("#q"), text = q.value.trim(); if (!text) return;
  const log = $("#chat"); const me = document.createElement("div"); me.className = "bubble me"; me.textContent = text; log.append(me); q.value = "";
  const bot = document.createElement("div"); bot.className = "bubble"; bot.textContent = "Thinking…"; log.append(bot); log.scrollTop = log.scrollHeight;
  try { const r = await api("/api/ai/ask", { method: "POST", body: { store: state.store, dept: state.dept, metric: state.metric, question: text } }); bot.textContent = r.answer; }
  catch (e) { bot.textContent = e.message; }
  log.scrollTop = log.scrollHeight;
}

onSeriesChange(load);
load();
