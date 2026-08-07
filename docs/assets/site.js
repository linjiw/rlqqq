/* RLQQQ public page: charts from site-data.json + browser-verified live signal. */

const $ = (id) => document.getElementById(id);
const fmtPct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmtX = (x) => `${x.toFixed(2)}×`;
const monthFmt = new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
const dayFmt = new Intl.DateTimeFormat("en-US", {
  month: "short", day: "numeric", year: "numeric", timeZone: "UTC",
});
let livePerformanceData = null;
let selectedLivePeriod = "ytd";

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
function attachCrosshair(svg, tip, plot, dates, xAt, rows, dateFormatter = monthFmt) {
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
      `<b>${dateFormatter.format(new Date(`${dates[i]}T00:00:00Z`))}</b><br>` +
      rows.map((r) => `${r.label}: <b>${r.fmt(r.values[i])}</b>`).join("<br>");
    const host = svg.parentElement.getBoundingClientRect();
    const tx = evt.clientX - host.left + 14;
    tip.style.left = `${Math.min(tx, host.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${evt.clientY - host.top - 10}px`;
  }
  svg.onpointermove = onMove;
  svg.onpointerleave = () => {
    hover.style.display = "none"; tip.style.display = "none";
  };
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
  svg.onpointermove = (evt) => {
    const rect = svg.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * 840;
    let best = tipRows[0];
    for (const r of tipRows) if (Math.abs(r.cx - px) < Math.abs(best.cx - px)) best = r;
    tip.style.display = "block";
    tip.innerHTML = `<b>${best.year}</b><br>RLQQQ: <b>${fmtPct(best.model)}</b><br>QQQ: <b>${fmtPct(best.qqq)}</b>`;
    const host = svg.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(evt.clientX - host.left + 14, host.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${evt.clientY - host.top - 10}px`;
  };
  svg.onpointerleave = () => { tip.style.display = "none"; };

  const tbody = $("annual-table").querySelector("tbody");
  tbody.replaceChildren();
  years.forEach((year, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${year}</td><td>${fmtPct(model[i])}</td><td>${fmtPct(qqq[i])}</td>`;
    tbody.appendChild(tr);
  });
}

/* ---------------- live performance ---------------- */
function fmtMaybePct(value, digits = 1) {
  return value == null ? "–" : fmtPct(value, digits);
}

function fmtMaybeNumber(value, digits = 2) {
  return value == null ? "–" : value.toFixed(digits);
}

function fmtPointDifference(value) {
  return `${Math.abs(value * 100).toFixed(1)} percentage points`;
}

function allocationSummary(exposure) {
  if (exposure <= 1) {
    return `${(exposure * 100).toFixed(1)}% QQQ · ${((1 - exposure) * 100).toFixed(1)}% T-bill cash`;
  }
  return `100.0% QQQ · ${((exposure - 1) * 100).toFixed(1)}% financed QQQ`;
}

function formatDate(date) {
  return dayFmt.format(new Date(`${date}T00:00:00Z`));
}

function xLabels(svg, dates, plot, xAt) {
  const positions = [...new Set([0, Math.floor((dates.length - 1) / 2), dates.length - 1])];
  for (const index of positions) {
    el("text", {
      x: xAt(index),
      y: plot.y + plot.h + 22,
      "text-anchor": index === 0 ? "start" : index === dates.length - 1 ? "end" : "middle",
    }, svg).textContent = dayFmt.format(new Date(`${dates[index]}T00:00:00Z`));
  }
}

function drawLiveGrowth(performance, period) {
  const svg = $("live-growth-chart");
  svg.replaceChildren();
  const start = period.startIndex;
  const dates = performance.chart.dates.slice(start);
  const base = 10000;
  const rawSeries = [
    {
      label: "RLQQQ",
      color: css("--series-model"),
      values: performance.chart.rlqqqWealth.slice(start),
      width: 2.6,
    },
    {
      label: "QQQ",
      color: css("--series-qqq"),
      values: performance.chart.qqqWealth.slice(start),
      width: 1.9,
    },
    {
      label: "S&P 500",
      color: css("--series-spy"),
      values: performance.chart.spyWealth.slice(start),
      width: 1.9,
    },
  ];
  const series = rawSeries.map((item) => ({
    ...item,
    values: item.values.map((value) => (value / item.values[0]) * base),
  }));
  const all = series.flatMap((item) => item.values);
  const spread = Math.max(Math.max(...all) - Math.min(...all), 100);
  const lo = Math.min(...all) - spread * 0.12;
  const hi = Math.max(...all) + spread * 0.12;
  const plot = { x: 68, y: 14, w: 840 - 68 - 18, h: 300 - 14 - 38 };
  const xAt = (index) => plot.x + (index / Math.max(1, dates.length - 1)) * plot.w;
  const yAt = (value) => plot.y + ((hi - value) / (hi - lo)) * plot.h;

  for (let tick = 0; tick < 5; tick += 1) {
    const value = lo + ((hi - lo) * tick) / 4;
    const y = yAt(value);
    el("line", {
      x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline",
    }, svg);
    el("text", {
      x: plot.x - 8, y: y + 4, "text-anchor": "end",
    }, svg).textContent = `$${Math.round(value).toLocaleString()}`;
  }
  xLabels(svg, dates, plot, xAt);

  for (const item of [series[2], series[1], series[0]]) {
    el("path", {
      d: linePath(item.values.map((_, index) => xAt(index)), item.values.map(yAt)),
      fill: "none",
      stroke: item.color,
      "stroke-width": item.width,
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
    }, svg);
  }

  attachCrosshair(
    svg,
    $("live-growth-tip"),
    plot,
    dates,
    xAt,
    series.map((item) => ({
      label: item.label,
      color: item.color,
      values: item.values,
      yAt,
      fmt: (value) => `$${Math.round(value).toLocaleString()}`,
    })),
    dayFmt,
  );
}

function drawLiveExposure(performance, period) {
  const svg = $("live-exposure-chart");
  svg.replaceChildren();
  const start = Math.min(period.startIndex, performance.actions.dates.length - 1);
  const dates = performance.actions.dates.slice(start);
  const values = performance.actions.targetExposure.slice(start);
  const plot = { x: 58, y: 14, w: 840 - 58 - 18, h: 300 - 14 - 38 };
  const xAt = (index) => plot.x + (index / Math.max(1, dates.length - 1)) * plot.w;
  const yAt = (value) => plot.y + plot.h - (value / 1.5) * plot.h;

  for (const tick of [0, 0.5, 1.0, 1.5]) {
    const y = yAt(tick);
    el("line", {
      x1: plot.x,
      x2: plot.x + plot.w,
      y1: y,
      y2: y,
      class: tick === 1 ? "axisline" : "gridline",
      "stroke-dasharray": tick === 1 ? "4,4" : "",
    }, svg);
    el("text", {
      x: plot.x - 8, y: y + 4, "text-anchor": "end",
    }, svg).textContent = `${tick.toFixed(1)}×`;
  }
  xLabels(svg, dates, plot, xAt);

  const path = linePath(values.map((_, index) => xAt(index)), values.map(yAt));
  const area = `${path}L${xAt(values.length - 1).toFixed(1)},${yAt(0).toFixed(1)}` +
    `L${xAt(0).toFixed(1)},${yAt(0).toFixed(1)}Z`;
  el("path", {
    d: area, fill: css("--series-model"), opacity: 0.1, stroke: "none",
  }, svg);
  el("path", {
    d: path,
    fill: "none",
    stroke: css("--series-model"),
    "stroke-width": 2.4,
    "stroke-linejoin": "round",
    "stroke-linecap": "round",
  }, svg);

  attachCrosshair(
    svg,
    $("live-exposure-tip"),
    plot,
    dates,
    xAt,
    [{
      label: "Target",
      color: css("--series-model"),
      values,
      yAt,
      fmt: fmtX,
    }],
    dayFmt,
  );
}

function renderLiveComparison(metrics) {
  const rows = [
    { key: "rlqqq", label: "RLQQQ", dot: "" },
    { key: "qqq", label: "QQQ buy & hold", dot: "qqq" },
    { key: "spy", label: "S&P 500 (SPY)", dot: "spy" },
  ];
  const tbody = $("live-comparison-body");
  tbody.replaceChildren();
  for (const row of rows) {
    const values = metrics[row.key];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="series-label"><span class="series-dot ${row.dot}"></span>${row.label}</span></td>
      <td>${fmtMaybePct(values.totalReturn)}</td>
      <td>${fmtMaybePct(values.annualizedVolatility)}</td>
      <td>${fmtMaybeNumber(values.sharpe)}</td>
      <td>${fmtMaybePct(values.maxDrawdown)}</td>
      <td>${fmtX(values.averageExposure)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderLivePeriod(key) {
  if (!livePerformanceData) return;
  const period = livePerformanceData.periods[key] || livePerformanceData.periods.all;
  selectedLivePeriod = key in livePerformanceData.periods ? key : "all";
  const { rlqqq, qqq, spy } = period.metrics;

  document.querySelectorAll("[data-live-period]").forEach((button) => {
    const active = button.dataset.livePeriod === selectedLivePeriod;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });

  $("live-return").textContent = fmtMaybePct(rlqqq.totalReturn);
  $("live-return-c").innerHTML =
    `QQQ <strong>${fmtMaybePct(qqq.totalReturn)}</strong> &middot; ` +
    `S&amp;P 500 <strong>${fmtMaybePct(spy.totalReturn)}</strong>`;
  $("live-vol").textContent = fmtMaybePct(rlqqq.annualizedVolatility);
  $("live-vol-c").innerHTML =
    `QQQ <strong>${fmtMaybePct(qqq.annualizedVolatility)}</strong> &middot; ` +
    `S&amp;P 500 <strong>${fmtMaybePct(spy.annualizedVolatility)}</strong>`;
  $("live-sharpe").textContent = fmtMaybeNumber(rlqqq.sharpe);
  $("live-sharpe-c").innerHTML =
    `QQQ <strong>${fmtMaybeNumber(qqq.sharpe)}</strong> &middot; ` +
    `S&amp;P 500 <strong>${fmtMaybeNumber(spy.sharpe)}</strong>`;
  $("live-dd").textContent = fmtMaybePct(rlqqq.maxDrawdown);
  $("live-dd-c").innerHTML =
    `QQQ <strong>${fmtMaybePct(qqq.maxDrawdown)}</strong> &middot; ` +
    `S&amp;P 500 <strong>${fmtMaybePct(spy.maxDrawdown)}</strong>`;

  const qqqReturnGap = rlqqq.totalReturn - qqq.totalReturn;
  const spyReturnGap = rlqqq.totalReturn - spy.totalReturn;
  const qqqVolGap = rlqqq.annualizedVolatility - qqq.annualizedVolatility;
  $("live-performance-read").textContent =
    `RLQQQ ${qqqReturnGap >= 0 ? "led" : "trailed"} QQQ by ` +
    `${fmtPointDifference(qqqReturnGap)} while running ` +
    `${fmtPointDifference(qqqVolGap)} ${qqqVolGap <= 0 ? "less" : "more"} annualized volatility. ` +
    `It ${spyReturnGap >= 0 ? "led" : "trailed"} the S&P 500 by ` +
    `${fmtPointDifference(spyReturnGap)} with an average ` +
    `${fmtX(rlqqq.averageExposure)} QQQ target.`;

  $("live-period-dates").textContent = period.complete
    ? `${formatDate(period.start)} to ${formatDate(period.end)}`
    : selectedLivePeriod === "ytd"
      ? `YTD from activation · ${formatDate(period.start)} to ${formatDate(period.end)}`
      : `${period.label} requested · available ${formatDate(period.start)} to ${formatDate(period.end)}`;
  $("live-session-count").textContent = `${rlqqq.days} completed sessions`;
  renderLiveComparison(period.metrics);
  drawLiveGrowth(livePerformanceData, period);
  drawLiveExposure(livePerformanceData, period);
}

function renderLivePerformance(performance) {
  if (!performance || performance.schemaVersion !== 1) {
    throw new Error("Live performance payload is missing or unsupported");
  }
  const chartLength = performance.chart.dates.length;
  if (
    chartLength < 2 ||
    performance.chart.rlqqqWealth.length !== chartLength ||
    performance.chart.qqqWealth.length !== chartLength ||
    performance.chart.spyWealth.length !== chartLength
  ) {
    throw new Error("Live performance chart series are not date aligned");
  }
  livePerformanceData = performance;
  $("portfolio-live-since").textContent = `Since ${formatDate(performance.inceptionDate)}`;
  $("live-auto-status").textContent = `Auto-updated through ${formatDate(performance.through)}`;
  $("live-performance-note").textContent =
    `Performance through ${formatDate(performance.through)} scores targets through ` +
    `${formatDate(performance.decisionThrough)}. The ${formatDate(performance.unscoredSignalAsOf)} ` +
    `target remains unscored. Returns include ${performance.accounting.transactionCostBps.toFixed(0)} bp ` +
    `one-way trading costs, T-bill cash, and T-bill + ` +
    `${performance.accounting.borrowSpreadBps.toFixed(0)} bp financing above 1.0×.`;
  document.querySelectorAll("[data-live-period]").forEach((button) => {
    button.onclick = () => renderLivePeriod(button.dataset.livePeriod);
  });
  renderLivePeriod(selectedLivePeriod);
}

/* ---------------- historical stats and stress ---------------- */
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
}

/* ---------------- live signal (ONNX-verified) ---------------- */
async function loadLive() {
  const badge = $("verify-badge");
  try {
    const signalRes = await fetch("assets/live-signal.json", { cache: "no-store" });
    if (!signalRes.ok) throw new Error(`Live data request failed with status ${signalRes.status}`);
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
    $("decision-allocation").textContent = allocationSummary(payload.signal.learnedMean);
    $("decision-explanation").textContent = payload.signal.explanation;
    const asOfDate = new Date(`${payload.asOf}T21:00:00Z`);
    const ageDays = Math.floor((Date.now() - asOfDate.getTime()) / 86_400_000);
    const freshness = ageDays <= 0 ? "latest completed session"
      : ageDays <= 3 ? "auto-refreshes three times each trading day"
      : "refresh overdue — see research console";
    $("decision-asof").textContent =
      `As of the ${payload.asOf} close · ${freshness} · model ${payload.model.displayName}, frozen ${payload.model.trainCutoff}`;
    try {
      renderLivePerformance(payload.performance);
    } catch (performanceError) {
      $("live-auto-status").textContent = "Live performance unavailable";
      $("live-performance-note").textContent = performanceError.message;
      console.error("live performance failed:", performanceError);
    }

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
  const redrawAll = () => {
    redraw();
    if (livePerformanceData) renderLivePeriod(selectedLivePeriod);
  };
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  loadLive();
}
boot();
