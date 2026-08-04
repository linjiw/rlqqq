/* RLQQQ public page: charts from site-data.json + browser-verified live signal. */

const $ = (id) => document.getElementById(id);
const fmtPct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmtX = (x) => `${x.toFixed(2)}×`;
const monthFmt = new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" });

/* ---------------- SVG helpers ---------------- */
const NS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function linePath(xs, ys) {
  let d = "";
  for (let i = 0; i < xs.length; i += 1) {
    d += `${i === 0 ? "M" : "L"}${xs[i].toFixed(1)},${ys[i].toFixed(1)}`;
  }
  return d;
}

/* Crosshair + tooltip for a time-series chart. rows: [{label, color, fmt, values}] */
function attachCrosshair(svg, tip, plot, dates, xAt, rows) {
  const hover = el("g", { style: "display:none" }, svg);
  const vline = el("line", {
    y1: plot.y, y2: plot.y + plot.h,
    stroke: css("--text-muted"), "stroke-width": 1, "stroke-dasharray": "3,3",
  }, hover);
  const dots = rows.map((r) =>
    el("circle", { r: 3.5, fill: r.color, stroke: css("--surface-1"), "stroke-width": 2 }, hover));

  function onMove(evt) {
    const rect = svg.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * 840;
    const frac = Math.min(1, Math.max(0, (px - plot.x) / plot.w));
    const i = Math.round(frac * (dates.length - 1));
    const x = xAt(i);
    hover.style.display = "";
    vline.setAttribute("x1", x); vline.setAttribute("x2", x);
    rows.forEach((r, k) => {
      dots[k].setAttribute("cx", x);
      dots[k].setAttribute("cy", r.yAt(r.values[i]));
    });
    tip.style.display = "block";
    tip.innerHTML =
      `<b>${monthFmt.format(new Date(`${dates[i]}T00:00:00Z`))}</b><br>` +
      rows.map((r) => `${r.label}: <b>${r.fmt(r.values[i])}</b>`).join("<br>");
    const host = svg.parentElement.getBoundingClientRect();
    const tx = evt.clientX - host.left + 14;
    tip.style.left = `${Math.min(tx, host.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${evt.clientY - host.top - 10}px`;
  }
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerleave", () => {
    hover.style.display = "none"; tip.style.display = "none";
  });
}

/* ---------------- wealth chart (log scale) ---------------- */
function drawWealth(data) {
  const svg = $("wealth-chart");
  svg.replaceChildren();
  const plot = { x: 56, y: 14, w: 840 - 56 - 62, h: 340 - 14 - 34 };
  const { dates, wealthV10, wealthQqq, wealthVt20 } = data.chart;
  const base = 10000;
  const all = [...wealthV10, ...wealthQqq, ...wealthVt20].map((v) => v * base);
  const lo = Math.min(...all) * 0.95;
  const hi = Math.max(...all) * 1.05;
  const yAt = (mult) => {
    const v = mult * base;
    return plot.y + plot.h - ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * plot.h;
  };
  const xAt = (i) => plot.x + (i / (dates.length - 1)) * plot.w;

  const ticks = [10000, 20000, 50000, 100000, 200000];
  for (const t of ticks) {
    if (t < lo || t > hi) continue;
    const y = yAt(t / base);
    el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline" }, svg);
    el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end" }, svg).textContent =
      t >= 1000 ? `$${t / 1000}k` : `$${t}`;
  }
  const yearStep = Math.ceil(dates.length / 8);
  for (let i = 0; i < dates.length; i += yearStep) {
    el("text", { x: xAt(i), y: plot.y + plot.h + 22, "text-anchor": "middle" }, svg)
      .textContent = dates[i].slice(0, 4);
  }
  el("line", { x1: plot.x, x2: plot.x + plot.w, y1: plot.y + plot.h, y2: plot.y + plot.h, class: "axisline" }, svg);

  const series = [
    { values: wealthVt20, color: css("--series-rule"), w: 1.6, label: "Vol rule", fmt: (v) => `$${Math.round(v * base).toLocaleString()}` },
    { values: wealthQqq, color: css("--series-qqq"), w: 1.8, label: "QQQ", fmt: (v) => `$${Math.round(v * base).toLocaleString()}` },
    { values: wealthV10, color: css("--series-model"), w: 2.4, label: "RLQQQ", fmt: (v) => `$${Math.round(v * base).toLocaleString()}` },
  ];
  for (const s of series) {
    el("path", {
      d: linePath(s.values.map((_, i) => xAt(i)), s.values.map(yAt)),
      fill: "none", stroke: s.color, "stroke-width": s.w, "stroke-linejoin": "round",
    }, svg);
  }
  // direct labels at line ends
  const endLabels = [
    { v: wealthV10.at(-1), color: css("--series-model"), t: "RLQQQ" },
    { v: wealthQqq.at(-1), color: css("--series-qqq"), t: "QQQ" },
    { v: wealthVt20.at(-1), color: css("--series-rule"), t: "Rule" },
  ].sort((a, b) => yAt(a.v) - yAt(b.v));
  let lastY = -Infinity;
  for (const L of endLabels) {
    const y = Math.max(yAt(L.v) + 4, lastY + 13);
    lastY = y;
    const label = el("text", {
      x: plot.x + plot.w + 5, y, "text-anchor": "start",
      style: `fill:${L.color}; font-weight:600`,
    }, svg);
    label.textContent = L.t;
  }

  attachCrosshair(svg, $("wealth-tip"), plot, dates, xAt, series.slice().reverse().map((s) => ({
    label: s.label, color: s.color, values: s.values, yAt, fmt: s.fmt,
  })));
}

/* ---------------- drawdown chart ---------------- */
function drawDrawdown(data) {
  const svg = $("dd-chart");
  svg.replaceChildren();
  const plot = { x: 56, y: 10, w: 840 - 56 - 16, h: 240 - 10 - 34 };
  const { dates, ddV10, ddQqq } = data.chart;
  const lo = Math.min(...ddQqq, ...ddV10) * 1.08;
  const yAt = (v) => plot.y + (v / lo) * plot.h;
  const xAt = (i) => plot.x + (i / (dates.length - 1)) * plot.w;

  for (const t of [0, -0.1, -0.2, -0.3]) {
    if (t < lo) continue;
    const y = yAt(t);
    el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline" }, svg);
    el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end" }, svg).textContent = fmtPct(t, 0);
  }
  const yearStep = Math.ceil(dates.length / 8);
  for (let i = 0; i < dates.length; i += yearStep) {
    el("text", { x: xAt(i), y: plot.y + plot.h + 22, "text-anchor": "middle" }, svg)
      .textContent = dates[i].slice(0, 4);
  }

  // QQQ area (behind), model line (front)
  const qqqPath = linePath(ddQqq.map((_, i) => xAt(i)), ddQqq.map(yAt)) +
    `L${(plot.x + plot.w).toFixed(1)},${yAt(0).toFixed(1)}L${plot.x},${yAt(0).toFixed(1)}Z`;
  el("path", { d: qqqPath, fill: css("--series-qqq"), opacity: 0.18, stroke: "none" }, svg);
  el("path", {
    d: linePath(ddQqq.map((_, i) => xAt(i)), ddQqq.map(yAt)),
    fill: "none", stroke: css("--series-qqq"), "stroke-width": 1.4,
  }, svg);
  el("path", {
    d: linePath(ddV10.map((_, i) => xAt(i)), ddV10.map(yAt)),
    fill: "none", stroke: css("--series-model"), "stroke-width": 2.2, "stroke-linejoin": "round",
  }, svg);

  attachCrosshair(svg, $("dd-tip"), plot, dates, xAt, [
    { label: "RLQQQ", color: css("--series-model"), values: ddV10, yAt, fmt: (v) => fmtPct(v) },
    { label: "QQQ", color: css("--series-qqq"), values: ddQqq, yAt, fmt: (v) => fmtPct(v) },
  ]);
}

/* ---------------- annual bars ---------------- */
function drawAnnual(data) {
  const svg = $("annual-chart");
  svg.replaceChildren();
  const years = Object.keys(data.annual.qqq).sort();
  const model = years.map((y) => data.annual.v10[y] ?? 0);
  const qqq = years.map((y) => data.annual.qqq[y] ?? 0);
  const plot = { x: 56, y: 12, w: 840 - 56 - 16, h: 280 - 12 - 34 };
  const lo = Math.min(0, ...model, ...qqq) * 1.15;
  const hi = Math.max(...model, ...qqq) * 1.15;
  const yAt = (v) => plot.y + ((hi - v) / (hi - lo)) * plot.h;

  for (const t of [-0.3, 0, 0.3, 0.6]) {
    if (t < lo || t > hi) continue;
    const y = yAt(t);
    el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: t === 0 ? "axisline" : "gridline" }, svg);
    el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end" }, svg).textContent = fmtPct(t, 0);
  }

  const group = plot.w / years.length;
  const barW = Math.min(16, group * 0.32);
  const tipRows = [];
  years.forEach((year, i) => {
    const cx = plot.x + group * (i + 0.5);
    const pairs = [
      { v: model[i], color: css("--series-model"), dx: -barW - 1 },
      { v: qqq[i], color: css("--series-qqq"), dx: 1 },
    ];
    for (const p of pairs) {
      const y0 = yAt(Math.max(0, p.v));
      const h = Math.abs(yAt(p.v) - yAt(0));
      el("rect", {
        x: cx + p.dx, y: y0, width: barW, height: Math.max(h, 1),
        fill: p.color, rx: 3,
      }, svg);
    }
    el("text", { x: cx, y: plot.y + plot.h + 22, "text-anchor": "middle" }, svg)
      .textContent = `'${year.slice(2)}`;
    tipRows.push({ year, model: model[i], qqq: qqq[i], cx });
  });

  const tip = $("annual-tip");
  svg.addEventListener("pointermove", (evt) => {
    const rect = svg.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * 840;
    let best = tipRows[0];
    for (const r of tipRows) if (Math.abs(r.cx - px) < Math.abs(best.cx - px)) best = r;
    tip.style.display = "block";
    tip.innerHTML = `<b>${best.year}</b><br>RLQQQ: <b>${fmtPct(best.model)}</b><br>QQQ: <b>${fmtPct(best.qqq)}</b>`;
    const host = svg.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(evt.clientX - host.left + 14, host.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${evt.clientY - host.top - 10}px`;
  });
  svg.addEventListener("pointerleave", () => { tip.style.display = "none"; });

  const tbody = $("annual-table").querySelector("tbody");
  tbody.replaceChildren();
  years.forEach((year, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${year}</td><td>${fmtPct(model[i])}</td><td>${fmtPct(qqq[i])}</td>`;
    tbody.appendChild(tr);
  });
}

/* ---------------- stats, stress, ytd ---------------- */
function renderStats(data) {
  const { v10, qqq } = data.stats;
  $("t-cagr").textContent = fmtPct(v10.cagr);
  $("t-cagr-c").textContent = `QQQ ${fmtPct(qqq.cagr)}`;
  $("t-sharpe").textContent = v10.sharpe.toFixed(2);
  $("t-sharpe-c").textContent = `QQQ ${qqq.sharpe.toFixed(2)}`;
  $("t-dd").textContent = fmtPct(v10.maxDD);
  $("t-dd-c").textContent = `QQQ ${fmtPct(qqq.maxDD)}`;
  $("t-wealth").textContent = `$${Math.round(v10.totalMultiple * 10000 / 1000)}k`;
  $("t-wealth-c").textContent = `QQQ $${Math.round(qqq.totalMultiple * 10000 / 1000)}k`;

  $("s-model-dd").textContent = fmtPct(data.era.v10.maxDD);
  $("s-model-cagr").textContent = fmtPct(data.era.v10.cagr);
  $("s-qqq-dd").textContent = fmtPct(data.era.qqq.maxDD);
  $("s-qqq-cagr").textContent = fmtPct(Math.abs(data.era.qqq.cagr));
  $("s-sig").textContent =
    `${data.era.window}. ${data.era.significant} — the one comparison in this study that clears statistical significance.`;

  const y = data.ytd2026;
  $("ytd-line").textContent =
    `Through ${y.through}: model ${fmtPct(y.v10Return)} vs QQQ ${fmtPct(y.qqqReturn)} ` +
    `(Sharpe ${y.v10.sharpe.toFixed(2)} vs ${y.qqq.sharpe.toFixed(2)}), with a ` +
    `${fmtPct(y.v10.maxDD)} vs ${fmtPct(y.qqq.maxDD)} worst drawdown. A calm bull market ` +
    `is the regime where buy & hold is hardest to beat — the model's job is to keep up ` +
    `here and win when it storms.`;
}

/* ---------------- live signal (ONNX-verified) ---------------- */
async function loadLive() {
  const badge = $("verify-badge");
  try {
    const signalRes = await fetch("assets/live-signal.json", { cache: "no-store" });
    const payload = await signalRes.json();

    // Show market state immediately (display-only fields).
    const m = payload.market;
    $("m-price").textContent = `$${m.price.toFixed(2)}`;
    $("m-vol").textContent = fmtPct(m.realizedVol21);
    $("m-mom").textContent = fmtPct(m.momentum21);
    $("m-dd").textContent = fmtPct(m.drawdown);
    $("m-vix").textContent = m.vix.toFixed(1);
    $("m-anchor").textContent = fmtX(payload.signal.vt10Exposure);
    $("m-tilt").textContent = fmtX(payload.signal.tiltMultiplier);
    const asOfDate = new Date(`${payload.asOf}T21:00:00Z`);
    const ageDays = Math.floor((Date.now() - asOfDate.getTime()) / 86_400_000);
    const freshness = ageDays <= 0 ? "latest completed session"
      : ageDays <= 3 ? "auto-refreshes three times each trading day"
      : "refresh overdue — see research console";
    $("decision-asof").textContent =
      `As of the ${payload.asOf} close · ${freshness} · model ${payload.model.displayName}, frozen ${payload.model.trainCutoff}`;

    // Fail-closed browser verification via the existing module.
    const mod = await import("./browser-inference.mjs");
    const result = await mod.verifyBrowserPolicy({ liveSignal: payload });
    const learned = result.signal.learnedMean;

    $("decision-number").textContent = learned.toFixed(2);
    const stanceEl = $("decision-stance");
    stanceEl.textContent = result.signal.stance;
    stanceEl.classList.toggle("full", learned >= 0.95);
    $("gauge-fill").style.width = `${Math.min(1, learned / 1.5) * 100}%`;
    badge.classList.add("ok");
    $("verify-text").textContent =
      `verified in this browser · ${result.verification.rowCount} sessions replayed, ` +
      `${result.verification.actionMatches}/${result.verification.actionChecks} actions matched`;
  } catch (err) {
    badge.classList.add("fail");
    $("verify-text").textContent = "verification failed — exposure withheld";
    $("decision-number").textContent = "–.––";
    $("decision-stance").textContent = "unavailable";
    console.error("live verification failed:", err);
  }
}

/* ---------------- boot ---------------- */
async function boot() {
  const res = await fetch("assets/site-data.json", { cache: "no-store" });
  const data = await res.json();
  renderStats(data);
  drawWealth(data);
  drawDrawdown(data);
  drawAnnual(data);
  const redraw = () => { drawWealth(data); drawDrawdown(data); drawAnnual(data); };
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  loadLive();
}
boot();
