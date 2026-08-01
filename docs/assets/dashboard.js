(() => {
  "use strict";

  const COLORS = {
    ink: "#18211e",
    muted: "#68736e",
    faint: "#8b9690",
    border: "#dce2de",
    surface: "#ffffff",
    soft: "#f7f8f6",
    gap: "#eef1ee",
    policy: "#087f5b",
    qqq: "#df6c4f",
    spy: "#3f6fa8",
    learned: "#7a5b9e",
    rule: "#b48627",
    warning: "#9a6210",
  };

  const SERIES_COLORS = {
    composite: COLORS.policy,
    learned: COLORS.learned,
    vt20: COLORS.rule,
    qqq: COLORS.qqq,
    spy: COLORS.spy,
  };

  const state = {
    data: null,
    liveData: null,
    dates: [],
    timestamps: [],
    index: 0,
    position: 0,
    selectedPolicy: "composite",
    chartMode: "wealth",
    playing: false,
    animationFrame: null,
    lastAnimationTime: 0,
    speed: 1,
    pointerActive: false,
    exporting: false,
    wealthDomain: [0.8, 32],
    drawdownFloor: -0.4,
    liveAnimationFrame: null,
    liveAnimationStart: 0,
    liveAnimationProgress: 0,
  };

  const elements = {};
  const dateFormatter = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
  const compactDateFormatter = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
  const dollarFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
  const livePriceFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  const generatedFormatter = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
    timeZoneName: "short",
  });

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    cacheElements();
    initializeIcons();
    loadLiveSignal();

    try {
      const response = await fetch("assets/policy-data.json");
      if (!response.ok) {
        throw new Error(`Data request failed with status ${response.status}`);
      }
      state.data = await response.json();
      state.dates = state.data.dates.map((date) => new Date(`${date}T00:00:00Z`));
      state.timestamps = state.dates.map((date) => date.getTime());
      state.index = state.data.dates.length - 1;
      state.position = state.index;
      calculateDomains();
      hydrateDashboard();
      bindInteractions();
      renderAll();
      setupResizeObservers();
      elements.chartLoading.classList.add("is-hidden");
    } catch (error) {
      showLoadError(error);
    }
  }

  function cacheElements() {
    const ids = [
      "sample-range",
      "sample-days",
      "live-status",
      "live-as-of",
      "live-grid",
      "live-stance",
      "live-learned-target",
      "live-posture",
      "live-gauge-fill",
      "live-gauge-range",
      "live-vt10",
      "live-tilt",
      "live-seed-range",
      "live-composite",
      "live-explanation",
      "live-price",
      "live-daily-change",
      "live-volatility",
      "live-momentum",
      "live-drawdown",
      "live-vix",
      "live-source",
      "live-generated",
      "live-chart",
      "live-chart-message",
      "policy-status",
      "policy-name",
      "policy-description",
      "metric-cagr",
      "metric-cagr-delta",
      "metric-sharpe",
      "metric-sharpe-delta",
      "metric-drawdown",
      "metric-drawdown-delta",
      "metric-wealth",
      "metric-exposure",
      "legend-policy",
      "wealth-chart-title",
      "wealth-chart",
      "exposure-chart",
      "chart-loading",
      "play-button",
      "reset-button",
      "timeline",
      "replay-date",
      "speed-select",
      "video-button",
      "image-button",
      "export-status",
      "decision-title",
      "stance-badge",
      "event-select",
      "target-label",
      "target-value",
      "exposure-gauge-fill",
      "base-label",
      "base-value",
      "tilt-label",
      "tilt-value",
      "turnover-value",
      "state-price",
      "state-volatility",
      "state-momentum",
      "state-drawdown",
      "decision-read",
      "exposure-date",
      "event-strip",
      "integrity-ci",
      "policy-table-body",
      "inference-sharpe",
      "inference-sharpe-ci",
      "inference-cagr",
      "inference-cagr-ci",
      "era-agent-sharpe",
      "era-benchmark-sharpe",
      "era-delta",
      "forward-agent-sharpe",
      "forward-benchmark-sharpe",
      "year-policy-heading",
      "year-table-body",
      "footer-as-of",
    ];
    ids.forEach((id) => {
      elements[toCamelCase(id)] = document.getElementById(id);
    });
  }

  function toCamelCase(value) {
    return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
  }

  function initializeIcons() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  async function loadLiveSignal() {
    try {
      const response = await fetch("assets/live-signal.json", { cache: "no-cache" });
      if (!response.ok) {
        throw new Error(`Live signal request failed with status ${response.status}`);
      }
      const payload = await response.json();
      validateLiveSignal(payload);
      state.liveData = payload;
      renderLiveSignal();
      startLiveChartAnimation();
    } catch (error) {
      showLiveError(error);
    }
  }

  function validateLiveSignal(payload) {
    const history = payload?.history;
    const signal = payload?.signal;
    if (
      payload?.schemaVersion !== 1 ||
      !payload.asOf ||
      !payload.market ||
      !signal ||
      !history ||
      !Array.isArray(history.dates) ||
      history.dates.length < 2 ||
      history.learnedMean.length !== history.dates.length ||
      history.vt10Exposure.length !== history.dates.length
    ) {
      throw new Error("Live signal schema is incomplete");
    }
  }

  function renderLiveSignal() {
    const payload = state.liveData;
    const { market, signal, source } = payload;
    const sourceDate = new Date(`${payload.asOf}T00:00:00Z`);
    const generatedDate = new Date(payload.generatedAt);
    const generatedAgeHours = (Date.now() - generatedDate.getTime()) / 3_600_000;
    const marketAgeDays = (Date.now() - sourceDate.getTime()) / 86_400_000;
    const stale = payload.stale || generatedAgeHours > 120 || marketAgeDays > 6;

    elements.liveGrid.setAttribute("aria-busy", "false");
    elements.liveStatus.className = `live-status${stale ? " is-stale" : ""}`;
    elements.liveStatus.innerHTML =
      `<i data-lucide="${stale ? "clock-alert" : "circle-check"}" aria-hidden="true"></i>` +
      (stale ? "Stale close" : "Latest close");
    elements.liveAsOf.textContent = `As of ${dateFormatter.format(sourceDate)}`;

    elements.liveLearnedTarget.textContent = `${signal.learnedMean.toFixed(2)}x`;
    elements.livePosture.textContent = signal.researchPosture;
    elements.liveVt10.textContent = `${signal.vt10Exposure.toFixed(2)}x`;
    elements.liveTilt.textContent = `${signal.tiltMultiplier.toFixed(2)}x`;
    elements.liveSeedRange.textContent =
      `${signal.learnedMin.toFixed(2)}-${signal.learnedMax.toFixed(2)}x`;
    elements.liveComposite.textContent = `${signal.compositeExposure.toFixed(2)}x`;
    elements.liveExplanation.textContent = signal.explanation;

    elements.liveGaugeFill.style.width =
      `${clamp(signal.learnedMean / 1.5, 0, 1) * 100}%`;
    elements.liveGaugeRange.style.left =
      `${clamp(signal.learnedMin / 1.5, 0, 1) * 100}%`;
    elements.liveGaugeRange.style.width =
      `${clamp((signal.learnedMax - signal.learnedMin) / 1.5, 0, 1) * 100}%`;

    elements.liveStance.textContent = signal.stance;
    elements.liveStance.className = "stance-badge";
    if (signal.stance === "Defensive") {
      elements.liveStance.classList.add("is-defensive");
    } else if (signal.stance === "Reduced risk") {
      elements.liveStance.classList.add("is-reduced");
    }

    elements.livePrice.textContent = livePriceFormatter.format(market.price);
    elements.liveDailyChange.textContent =
      `${market.dailyChange >= 0 ? "+" : ""}${formatPercent(market.dailyChange, 2)}`;
    elements.liveDailyChange.className = returnClass(market.dailyChange);
    elements.liveVolatility.textContent = formatPercent(market.realizedVol21, 1);
    elements.liveMomentum.textContent = formatPercent(market.momentum21, 1);
    elements.liveMomentum.className = returnClass(market.momentum21);
    elements.liveDrawdown.textContent = formatPercent(market.drawdown, 1);
    elements.liveDrawdown.className = returnClass(market.drawdown);
    elements.liveVix.textContent = `VIX ${market.vix.toFixed(2)}`;
    elements.liveSource.textContent =
      `${source.provider} / ${source.frequency.toLowerCase()}`;
    elements.liveGenerated.textContent =
      `Generated ${generatedFormatter.format(generatedDate)}`;
    elements.liveChartMessage.classList.add("is-hidden");
    initializeIcons();
  }

  function showLiveError(error) {
    console.error(error);
    elements.liveGrid.setAttribute("aria-busy", "false");
    elements.liveStatus.className = "live-status is-error";
    elements.liveStatus.innerHTML =
      '<i data-lucide="triangle-alert" aria-hidden="true"></i>Signal unavailable';
    elements.liveAsOf.textContent = "Historical replay remains available";
    elements.liveExplanation.textContent =
      "The latest generated signal could not be loaded.";
    elements.liveChartMessage.textContent = "Latest signal path unavailable";
    elements.liveChartMessage.style.color = COLORS.warning;
    initializeIcons();
  }

  function startLiveChartAnimation() {
    if (state.liveAnimationFrame) {
      cancelAnimationFrame(state.liveAnimationFrame);
    }
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    state.liveAnimationProgress = reduced ? 1 : 0;
    state.liveAnimationStart = performance.now();
    if (reduced) {
      drawLiveChart();
      return;
    }
    state.liveAnimationFrame = requestAnimationFrame(animateLiveChart);
  }

  function animateLiveChart(now) {
    const elapsed = now - state.liveAnimationStart;
    state.liveAnimationProgress = clamp(elapsed / 850, 0, 1);
    drawLiveChart();
    if (state.liveAnimationProgress < 1) {
      state.liveAnimationFrame = requestAnimationFrame(animateLiveChart);
    } else {
      state.liveAnimationFrame = null;
    }
  }

  function drawLiveChart() {
    if (!state.liveData) return;
    const history = state.liveData.history;
    const setup = setupCanvas(elements.liveChart);
    const { context, width, height } = setup;
    const plot = { x: 42, y: 12, width: width - 54, height: height - 38 };
    const count = history.dates.length;
    const throughIndex = Math.max(
      1,
      Math.min(count - 1, Math.floor((count - 1) * state.liveAnimationProgress)),
    );
    const xAt = (index) =>
      plot.x + (index / Math.max(1, count - 1)) * plot.width;
    const yAt = (value) => linearScale(value, [0, 1.5], plot);

    clearCanvas(context, width, height);
    drawHorizontalGrid(
      context,
      plot,
      [0, 0.5, 1, 1.5],
      yAt,
      (value) => `${value.toFixed(1)}x`,
    );

    context.save();
    context.fillStyle = "rgb(122 91 158 / 12%)";
    context.beginPath();
    for (let index = 0; index <= throughIndex; index += 1) {
      const x = xAt(index);
      const y = yAt(history.learnedMax[index]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    for (let index = throughIndex; index >= 0; index -= 1) {
      context.lineTo(xAt(index), yAt(history.learnedMin[index]));
    }
    context.closePath();
    context.fill();
    context.restore();

    drawLiveSeries(
      context,
      history.vt10Exposure,
      throughIndex,
      xAt,
      yAt,
      COLORS.rule,
      1.4,
    );
    drawLiveSeries(
      context,
      history.learnedMean,
      throughIndex,
      xAt,
      yAt,
      COLORS.learned,
      2.3,
    );

    const currentX = xAt(throughIndex);
    drawDot(
      context,
      currentX,
      yAt(history.learnedMean[throughIndex]),
      COLORS.learned,
      3.5,
    );
    drawLiveTimeAxis(context, plot, history.dates, xAt);
  }

  function drawLiveSeries(context, values, throughIndex, xAt, yAt, color, width) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = width;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    for (let index = 0; index <= throughIndex; index += 1) {
      const x = xAt(index);
      const y = yAt(values[index]);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    context.restore();
  }

  function drawLiveTimeAxis(context, plot, dates, xAt) {
    const indices = [0, Math.floor((dates.length - 1) / 2), dates.length - 1];
    context.save();
    context.fillStyle = COLORS.muted;
    context.font = "9px Inter, ui-sans-serif, sans-serif";
    context.textBaseline = "top";
    indices.forEach((index, position) => {
      context.textAlign = position === 0 ? "left" : position === 2 ? "right" : "center";
      context.fillText(
        compactDateFormatter.format(new Date(`${dates[index]}T00:00:00Z`)),
        xAt(index),
        plot.y + plot.height + 7,
      );
    });
    context.restore();
  }

  function calculateDomains() {
    const allSeries = Object.values(state.data.series);
    let wealthMin = Infinity;
    let wealthMax = -Infinity;
    let drawdownMin = 0;
    allSeries.forEach((series) => {
      series.wealth.forEach((value) => {
        wealthMin = Math.min(wealthMin, value);
        wealthMax = Math.max(wealthMax, value);
      });
      series.drawdown.forEach((value) => {
        drawdownMin = Math.min(drawdownMin, value);
      });
    });
    state.wealthDomain = [
      Math.max(0.5, Math.floor(wealthMin * 10) / 10),
      niceLogCeiling(wealthMax),
    ];
    state.drawdownFloor = Math.min(-0.1, Math.floor(drawdownMin * 10) / 10);
  }

  function niceLogCeiling(value) {
    const magnitude = 10 ** Math.floor(Math.log10(value));
    const normalized = value / magnitude;
    const nice = normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
    return nice * magnitude;
  }

  function hydrateDashboard() {
    const { meta, comparison, evidence } = state.data;
    elements.sampleRange.textContent = `${meta.replayStart.slice(0, 4)}-${meta.replayEnd.slice(0, 4)}`;
    elements.sampleDays.textContent = `${meta.sampledDays.toLocaleString("en-US")} OOS days`;
    elements.timeline.max = String(state.data.dates.length - 1);

    populatePolicyTable();
    populateEvents();
    populateAnnualTable();

    elements.inferenceSharpe.textContent = `${formatSigned(comparison.sharpeDelta, 3)} Sharpe`;
    elements.inferenceSharpeCi.textContent =
      `95% CI ${formatSigned(comparison.sharpeCi[0], 3)} to ${formatSigned(comparison.sharpeCi[1], 3)}`;
    elements.inferenceCagr.textContent = `${formatSigned(comparison.cagrDelta * 100, 2)} pp CAGR`;
    elements.inferenceCagrCi.textContent =
      `95% CI ${formatSigned(comparison.cagrCi[0] * 100, 2)} to ${formatSigned(comparison.cagrCi[1] * 100, 2)} pp`;
    elements.integrityCi.textContent =
      `Sharpe CI ${formatSigned(comparison.sharpeCi[0], 2)} to ${formatSigned(comparison.sharpeCi[1], 2)}`;

    elements.eraAgentSharpe.textContent = evidence.eraHoldout.learnedSharpe.toFixed(2);
    elements.eraBenchmarkSharpe.textContent = evidence.eraHoldout.benchmarkSharpe.toFixed(2);
    elements.eraDelta.textContent =
      `${formatSigned(evidence.eraHoldout.sharpeDelta, 2)} ` +
      `[${formatSigned(evidence.eraHoldout.sharpeCi[0], 2)}, ` +
      `${formatSigned(evidence.eraHoldout.sharpeCi[1], 2)}]`;
    elements.forwardAgentSharpe.textContent = evidence.forwardHoldout.blendSharpe.toFixed(2);
    elements.forwardBenchmarkSharpe.textContent =
      evidence.forwardHoldout.benchmarkSharpe.toFixed(2);

    const sourceDate = new Date(`${meta.sourceAsOf}T00:00:00Z`);
    elements.footerAsOf.textContent = `Data through ${dateFormatter.format(sourceDate)}`;

    updateSelectedPolicy();
  }

  function populatePolicyTable() {
    const rows = [
      ["Highest observed return", "composite", "Post-hoc", "posthoc"],
      ["Best robust learned model", "learned", "Era-tested", "robust"],
      ["Best simple policy", "vt20", "No learning", ""],
      ["Primary benchmark", "qqq", "Reference", ""],
      ["Broad-market benchmark", "spy", "Reference", ""],
    ];

    elements.policyTableBody.innerHTML = rows
      .map(([claim, id, status, statusClass]) => {
        const policy = state.data.policies[id];
        const metrics = policy.metrics;
        return `
          <tr>
            <td>${claim}</td>
            <td>
              <strong>${policy.name}</strong>
              <small>${policy.shortName}</small>
            </td>
            <td>${formatPercent(metrics.cagr, 2)}</td>
            <td>${metrics.sharpe.toFixed(3)}</td>
            <td>${formatPercent(metrics.maxDrawdown, 2)}</td>
            <td><span class="table-status ${statusClass}">${status}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  function populateEvents() {
    state.data.events.forEach((event) => {
      const option = document.createElement("option");
      option.value = String(event.index);
      option.textContent = `${event.date.slice(0, 4)} - ${event.label}`;
      elements.eventSelect.append(option);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "event-button";
      button.dataset.index = String(event.index);
      button.innerHTML = `<time>${event.date.slice(0, 4)}</time><span>${event.label}</span>`;
      button.addEventListener("click", () => jumpToIndex(event.index));
      elements.eventStrip.append(button);
    });
  }

  function populateAnnualTable() {
    elements.yearTableBody.innerHTML = state.data.annual
      .map((record) => {
        const partial = record.partial
          ? '<span class="partial-tag">partial</span>'
          : "";
        return `
          <tr tabindex="0" data-year="${record.year}" data-key-index="${record.keyIndex}">
            <td><strong>${record.year}</strong> ${partial}</td>
            <td data-policy-return="${record.year}"></td>
            <td class="${returnClass(record.returns.qqq)}">${formatPercent(record.returns.qqq, 2)}</td>
            <td class="${returnClass(record.returns.spy)}">${formatPercent(record.returns.spy, 2)}</td>
            <td>${record.averageExposure.toFixed(2)}x</td>
            <td class="decision-cell">
              <strong>${formatCompactDate(record.keyDate)} - ${record.keyStance}</strong>
              <span>${record.keyDecision}</span>
            </td>
          </tr>
        `;
      })
      .join("");

    elements.yearTableBody.querySelectorAll("tr").forEach((row) => {
      const activate = () => {
        jumpToIndex(Number(row.dataset.keyIndex));
        document.getElementById("replay").scrollIntoView({ behavior: "smooth" });
      };
      row.addEventListener("click", activate);
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
    });
  }

  function bindInteractions() {
    document.querySelectorAll("[data-policy]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedPolicy = button.dataset.policy;
        setActiveSegment("[data-policy]", button);
        updateSelectedPolicy();
        renderAll();
      });
    });

    document.querySelectorAll("[data-chart-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        state.chartMode = button.dataset.chartMode;
        setActiveSegment("[data-chart-mode]", button);
        elements.wealthChartTitle.textContent =
          state.chartMode === "wealth" ? "Normalized wealth" : "Drawdown from peak";
        drawWealthChart();
      });
    });

    elements.playButton.addEventListener("click", togglePlayback);
    elements.resetButton.addEventListener("click", () => {
      stopPlayback();
      jumpToIndex(0);
    });
    elements.timeline.addEventListener("input", (event) => {
      stopPlayback();
      jumpToIndex(Number(event.target.value));
    });
    elements.speedSelect.addEventListener("change", (event) => {
      state.speed = Number(event.target.value);
    });
    elements.eventSelect.addEventListener("change", (event) => {
      if (event.target.value !== "") {
        jumpToIndex(Number(event.target.value));
      }
    });
    elements.videoButton.addEventListener("click", exportVideo);
    elements.imageButton.addEventListener("click", exportImage);

    bindCanvasScrubbing(elements.wealthChart);
    bindCanvasScrubbing(elements.exposureChart);
  }

  function bindCanvasScrubbing(canvas) {
    const updateFromPointer = (event) => {
      const rect = canvas.getBoundingClientRect();
      const padding = canvas === elements.wealthChart
        ? { left: 56, right: 16 }
        : { left: 56, right: 16 };
      const usableWidth = Math.max(1, rect.width - padding.left - padding.right);
      const ratio = clamp((event.clientX - rect.left - padding.left) / usableWidth, 0, 1);
      const timestamp =
        state.timestamps[0] +
        ratio * (state.timestamps[state.timestamps.length - 1] - state.timestamps[0]);
      jumpToIndex(nearestTimestampIndex(timestamp));
    };

    canvas.addEventListener("pointerdown", (event) => {
      stopPlayback();
      state.pointerActive = true;
      canvas.setPointerCapture(event.pointerId);
      updateFromPointer(event);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (state.pointerActive) {
        updateFromPointer(event);
      }
    });
    const release = (event) => {
      state.pointerActive = false;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
  }

  function setActiveSegment(selector, activeButton) {
    document.querySelectorAll(selector).forEach((button) => {
      const active = button === activeButton;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function setupResizeObservers() {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(() => {
        drawLiveChart();
        drawWealthChart();
        drawExposureChart();
      });
    });
    observer.observe(elements.liveChart.parentElement);
    observer.observe(elements.wealthChart.parentElement);
    observer.observe(elements.exposureChart.parentElement);
  }

  function updateSelectedPolicy() {
    const policy = state.data.policies[state.selectedPolicy];
    const metrics = policy.metrics;
    const benchmark = state.data.policies.qqq.metrics;
    const cagrDelta = metrics.cagr - benchmark.cagr;
    const sharpeDelta = metrics.sharpe - benchmark.sharpe;
    const drawdownImprovement =
      Math.abs(benchmark.maxDrawdown) - Math.abs(metrics.maxDrawdown);

    elements.policyName.textContent = policy.name;
    elements.policyDescription.textContent = policy.description;
    elements.policyStatus.textContent = policy.status;
    elements.policyStatus.className = "classification";
    if (state.selectedPolicy === "learned") {
      elements.policyStatus.classList.add("is-validated");
    } else if (state.selectedPolicy === "vt20") {
      elements.policyStatus.classList.add("is-rule");
    }

    elements.metricCagr.textContent = formatPercent(metrics.cagr, 2);
    elements.metricCagrDelta.textContent =
      `${formatSigned(cagrDelta * 100, 2)} pp vs QQQ`;
    elements.metricSharpe.textContent = metrics.sharpe.toFixed(3);
    elements.metricSharpeDelta.textContent =
      `${formatSigned(sharpeDelta, 3)} vs QQQ`;
    elements.metricDrawdown.textContent = formatPercent(metrics.maxDrawdown, 2);
    elements.metricDrawdownDelta.textContent =
      drawdownImprovement >= 0
        ? `${(drawdownImprovement * 100).toFixed(2)} pp shallower`
        : `${Math.abs(drawdownImprovement * 100).toFixed(2)} pp deeper`;
    elements.metricWealth.textContent = dollarFormatter.format(
      metrics.totalMultiple * 10000,
    );
    elements.metricExposure.textContent = metrics.averageExposure !== undefined
      ? `${metrics.averageExposure.toFixed(2)}x average exposure`
      : "1.00x continuous exposure";
    elements.legendPolicy.textContent = policy.shortName;
    elements.yearPolicyHeading.textContent = policy.shortName;

    state.data.annual.forEach((record) => {
      const cell = document.querySelector(`[data-policy-return="${record.year}"]`);
      const value = record.returns[state.selectedPolicy];
      cell.textContent = formatPercent(value, 2);
      cell.className = returnClass(value);
    });
  }

  function renderAll() {
    if (!state.data) return;
    state.index = clamp(Math.round(state.position), 0, state.data.dates.length - 1);
    state.position = state.index;
    elements.timeline.value = String(state.index);
    renderDecisionInspector();
    updateActiveRecords();
    drawWealthChart();
    drawExposureChart();
  }

  function renderDecisionInspector() {
    const index = state.index;
    const signals = state.data.signals;
    const date = state.dates[index];
    const decision = decisionAt(index, state.selectedPolicy);
    const previous = previousExposure(index, state.selectedPolicy);
    const change = decision.target - previous;
    const marketVol = signals.realizedVol21[index];
    const marketMomentum = signals.momentum21[index];
    const marketDrawdown = signals.marketDrawdown[index];

    elements.replayDate.textContent = compactDateFormatter.format(date);
    elements.decisionTitle.textContent = dateFormatter.format(date);
    elements.exposureDate.textContent = dateFormatter.format(date);
    elements.targetLabel.textContent = decision.targetLabel;
    elements.targetValue.textContent = `${decision.target.toFixed(2)}x`;
    elements.exposureGaugeFill.style.width =
      `${clamp(decision.target / 1.5, 0, 1) * 100}%`;
    elements.baseLabel.textContent = decision.baseLabel;
    elements.baseValue.textContent = `${decision.base.toFixed(2)}x`;
    elements.tiltLabel.textContent = decision.tiltLabel;
    elements.tiltValue.textContent = `${decision.tilt.toFixed(2)}x`;
    elements.turnoverValue.textContent =
      Math.abs(change) < 0.015
        ? "Hold"
        : change > 0
          ? `Add ${change.toFixed(2)}x`
          : `Trim ${Math.abs(change).toFixed(2)}x`;

    elements.statePrice.textContent = dollarFormatter.format(signals.price[index]);
    elements.stateVolatility.textContent = formatPercent(marketVol, 1);
    elements.stateMomentum.textContent = formatPercent(marketMomentum, 1);
    elements.stateDrawdown.textContent = formatPercent(marketDrawdown, 1);
    elements.stateMomentum.className = returnClass(marketMomentum);
    elements.stateDrawdown.className = returnClass(marketDrawdown);
    elements.decisionRead.textContent = decisionNarrative(
      decision,
      marketVol,
      marketDrawdown,
    );

    const stance = stanceForExposure(decision.target);
    elements.stanceBadge.textContent = stance;
    elements.stanceBadge.className = "stance-badge";
    if (stance === "Defensive") {
      elements.stanceBadge.classList.add("is-defensive");
    } else if (stance === "Reduced risk") {
      elements.stanceBadge.classList.add("is-reduced");
    }
  }

  function decisionAt(index, policyId) {
    const signals = state.data.signals;
    if (policyId === "learned") {
      const base = signals.vt10Exposure[index];
      const target = signals.learnedExposure[index];
      return {
        target,
        base,
        tilt: base > 0 ? target / base : 1,
        targetLabel: "Learned core target",
        baseLabel: "10% volatility anchor",
        tiltLabel: "v4 ensemble multiplier",
        policyId,
      };
    }
    if (policyId === "vt20") {
      const target = signals.vt20Exposure[index];
      return {
        target,
        base: target,
        tilt: 1,
        targetLabel: "Rule target",
        baseLabel: "20% volatility target",
        tiltLabel: "Learned adjustment",
        policyId,
      };
    }
    return {
      target: signals.compositeExposure[index],
      base: signals.vt20Exposure[index],
      tilt: signals.multiplier[index],
      targetLabel: "Composite target",
      baseLabel: "20% volatility base",
      tiltLabel: "Transferred v4 tilt",
      policyId,
    };
  }

  function previousExposure(index, policyId) {
    if (index <= 0 || isFoldStart(index)) return 0;
    const signals = state.data.signals;
    const key = policyId === "composite"
      ? "compositeExposure"
      : policyId === "learned"
        ? "learnedExposure"
        : "vt20Exposure";
    return signals[key][index - 1];
  }

  function isFoldStart(index) {
    return state.data.folds.some((fold) => fold.startIndex === index);
  }

  function decisionNarrative(decision, realizedVol, marketDrawdown) {
    const risk = realizedVol >= 0.4
      ? "Extreme volatility"
      : realizedVol >= 0.25
        ? "High volatility"
        : realizedVol >= 0.18
          ? "Elevated volatility"
          : "Contained volatility";
    const drawdownText = marketDrawdown <= -0.2
      ? "deep drawdown"
      : marketDrawdown <= -0.08
        ? "market correction"
        : "market trend";

    if (decision.policyId === "vt20") {
      const sizing = decision.target >= 1.49
        ? "the rule reached its 1.50x cap"
        : `the rule sized exposure at ${decision.target.toFixed(2)}x`;
      return `${risk} and a ${drawdownText}: ${sizing}. No learned signal is used.`;
    }

    const tiltRead = decision.tilt >= 1.1
      ? "added risk above the volatility base"
      : decision.tilt <= 0.9
        ? "trimmed risk below the volatility base"
        : "left the volatility base nearly unchanged";

    if (decision.policyId === "learned") {
      return `${risk} and a ${drawdownText}: the v4 ensemble ${tiltRead}, ending at ${decision.target.toFixed(2)}x.`;
    }
    return `${risk} and a ${drawdownText}: VT20 set ${decision.base.toFixed(2)}x, then the transferred v4 tilt ${tiltRead}.`;
  }

  function stanceForExposure(exposure) {
    if (exposure < 0.65) return "Defensive";
    if (exposure < 0.95) return "Reduced risk";
    if (exposure < 1.2) return "Fully invested";
    return "Levered";
  }

  function updateActiveRecords() {
    const matchingEvent = state.data.events.find((event) => event.index === state.index);
    elements.eventSelect.value = matchingEvent ? String(matchingEvent.index) : "";
    elements.eventStrip.querySelectorAll(".event-button").forEach((button) => {
      button.classList.toggle(
        "is-active",
        Number(button.dataset.index) === state.index,
      );
    });

    const currentYear = state.dates[state.index].getUTCFullYear();
    elements.yearTableBody.querySelectorAll("tr").forEach((row) => {
      row.classList.toggle("is-active", Number(row.dataset.year) === currentYear);
    });
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
      return;
    }
    if (state.index >= state.data.dates.length - 1) {
      state.position = 0;
      state.index = 0;
    }
    state.playing = true;
    state.lastAnimationTime = performance.now();
    updatePlayButton();
    state.animationFrame = requestAnimationFrame(animateReplay);
  }

  function animateReplay(now) {
    if (!state.playing) return;
    const elapsed = Math.min(100, now - state.lastAnimationTime);
    state.lastAnimationTime = now;
    const pointsPerSecond =
      (state.data.dates.length - 1) / 36 * state.speed;
    state.position += (elapsed / 1000) * pointsPerSecond;
    if (state.position >= state.data.dates.length - 1) {
      state.position = state.data.dates.length - 1;
      state.index = Math.round(state.position);
      renderAll();
      stopPlayback();
      return;
    }
    state.index = Math.round(state.position);
    renderAll();
    state.animationFrame = requestAnimationFrame(animateReplay);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.animationFrame) {
      cancelAnimationFrame(state.animationFrame);
      state.animationFrame = null;
    }
    updatePlayButton();
  }

  function updatePlayButton() {
    elements.playButton.setAttribute(
      "aria-label",
      state.playing ? "Pause replay" : "Play replay",
    );
    elements.playButton.dataset.tooltip =
      state.playing ? "Pause replay" : "Play replay";
    elements.playButton.innerHTML =
      `<i data-lucide="${state.playing ? "pause" : "play"}" aria-hidden="true"></i>`;
    initializeIcons();
  }

  function jumpToIndex(index) {
    state.index = clamp(Math.round(index), 0, state.data.dates.length - 1);
    state.position = state.index;
    renderAll();
  }

  function nearestTimestampIndex(timestamp) {
    let low = 0;
    let high = state.timestamps.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (state.timestamps[middle] < timestamp) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    if (
      low > 0 &&
      Math.abs(state.timestamps[low - 1] - timestamp) <
        Math.abs(state.timestamps[low] - timestamp)
    ) {
      return low - 1;
    }
    return low;
  }

  function drawWealthChart() {
    if (!state.data) return;
    const setup = setupCanvas(elements.wealthChart);
    const { context, width, height } = setup;
    const plot = { x: 56, y: 16, width: width - 72, height: height - 48 };

    clearCanvas(context, width, height);
    drawGapBands(context, plot);

    if (state.chartMode === "wealth") {
      drawWealthAxes(context, plot);
      drawLineSeries(
        context,
        state.data.series.spy.wealth,
        state.index,
        plot,
        (value) => logScale(value, state.wealthDomain, plot),
        COLORS.spy,
        1.5,
      );
      drawLineSeries(
        context,
        state.data.series.qqq.wealth,
        state.index,
        plot,
        (value) => logScale(value, state.wealthDomain, plot),
        COLORS.qqq,
        1.6,
      );
      drawLineSeries(
        context,
        state.data.series[state.selectedPolicy].wealth,
        state.index,
        plot,
        (value) => logScale(value, state.wealthDomain, plot),
        SERIES_COLORS[state.selectedPolicy],
        2.5,
      );
      drawCurrentDots(context, plot, "wealth", (value) =>
        logScale(value, state.wealthDomain, plot));
    } else {
      drawDrawdownAxes(context, plot);
      drawLineSeries(
        context,
        state.data.series.spy.drawdown,
        state.index,
        plot,
        (value) => linearScale(value, [state.drawdownFloor, 0], plot),
        COLORS.spy,
        1.5,
      );
      drawLineSeries(
        context,
        state.data.series.qqq.drawdown,
        state.index,
        plot,
        (value) => linearScale(value, [state.drawdownFloor, 0], plot),
        COLORS.qqq,
        1.6,
      );
      drawLineSeries(
        context,
        state.data.series[state.selectedPolicy].drawdown,
        state.index,
        plot,
        (value) => linearScale(value, [state.drawdownFloor, 0], plot),
        SERIES_COLORS[state.selectedPolicy],
        2.5,
      );
      drawCurrentDots(context, plot, "drawdown", (value) =>
        linearScale(value, [state.drawdownFloor, 0], plot));
    }

    drawEventMarkers(context, plot, state.index);
    drawTimeAxis(context, plot);
    drawCursor(context, plot);
  }

  function drawExposureChart() {
    if (!state.data) return;
    const setup = setupCanvas(elements.exposureChart);
    const { context, width, height } = setup;
    const plot = { x: 56, y: 15, width: width - 72, height: height - 46 };

    clearCanvas(context, width, height);
    drawGapBands(context, plot);
    drawExposureAxes(context, plot);
    drawLineSeries(
      context,
      state.data.signals.learnedExposure,
      state.index,
      plot,
      (value) => linearScale(value, [0, 1.5], plot),
      COLORS.learned,
      1.3,
    );
    drawLineSeries(
      context,
      state.data.signals.vt20Exposure,
      state.index,
      plot,
      (value) => linearScale(value, [0, 1.5], plot),
      COLORS.rule,
      1.5,
    );
    drawLineSeries(
      context,
      state.data.signals.compositeExposure,
      state.index,
      plot,
      (value) => linearScale(value, [0, 1.5], plot),
      COLORS.policy,
      2.2,
    );
    drawEventMarkers(context, plot, state.index, false);
    drawTimeAxis(context, plot);
    drawCursor(context, plot);

    [
      ["learnedExposure", COLORS.learned],
      ["vt20Exposure", COLORS.rule],
      ["compositeExposure", COLORS.policy],
    ].forEach(([key, color]) => {
      const x = xForIndex(state.index, plot);
      const y = linearScale(state.data.signals[key][state.index], [0, 1.5], plot);
      drawDot(context, x, y, color, key === "compositeExposure" ? 4 : 3);
    });
  }

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const targetWidth = Math.round(width * ratio);
    const targetHeight = Math.round(height * ratio);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
  }

  function clearCanvas(context, width, height) {
    context.clearRect(0, 0, width, height);
    context.fillStyle = COLORS.surface;
    context.fillRect(0, 0, width, height);
  }

  function drawGapBands(context, plot) {
    context.save();
    context.fillStyle = COLORS.gap;
    state.data.folds.slice(1).forEach((fold, index) => {
      const previous = state.data.folds[index];
      const left = xForTimestamp(
        new Date(`${previous.endDate}T00:00:00Z`).getTime(),
        plot,
      );
      const right = xForTimestamp(
        new Date(`${fold.startDate}T00:00:00Z`).getTime(),
        plot,
      );
      context.fillRect(left, plot.y, Math.max(1, right - left), plot.height);
    });
    context.restore();
  }

  function drawWealthAxes(context, plot) {
    const ticks = [0.5, 1, 2, 5, 10, 20, 50].filter(
      (tick) => tick >= state.wealthDomain[0] && tick <= state.wealthDomain[1],
    );
    drawHorizontalGrid(context, plot, ticks, (value) =>
      logScale(value, state.wealthDomain, plot), (value) =>
      compactDollar(value * 10000));
  }

  function drawDrawdownAxes(context, plot) {
    const ticks = [];
    for (let value = 0; value >= state.drawdownFloor - 0.001; value -= 0.1) {
      ticks.push(Number(value.toFixed(2)));
    }
    drawHorizontalGrid(context, plot, ticks, (value) =>
      linearScale(value, [state.drawdownFloor, 0], plot), (value) =>
      `${Math.round(value * 100)}%`);
  }

  function drawExposureAxes(context, plot) {
    const ticks = [0, 0.5, 1, 1.5];
    drawHorizontalGrid(context, plot, ticks, (value) =>
      linearScale(value, [0, 1.5], plot), (value) => `${value.toFixed(1)}x`);
    const fullY = linearScale(1, [0, 1.5], plot);
    context.save();
    context.strokeStyle = "#8b9690";
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(plot.x, fullY);
    context.lineTo(plot.x + plot.width, fullY);
    context.stroke();
    context.restore();
  }

  function drawHorizontalGrid(context, plot, ticks, scale, label) {
    context.save();
    context.font = "10px Inter, ui-sans-serif, sans-serif";
    context.textAlign = "right";
    context.textBaseline = "middle";
    ticks.forEach((tick) => {
      const y = scale(tick);
      context.strokeStyle = COLORS.border;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(plot.x, y);
      context.lineTo(plot.x + plot.width, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.fillText(label(tick), plot.x - 8, y);
    });
    context.restore();
  }

  function drawTimeAxis(context, plot) {
    context.save();
    context.font = "10px Inter, ui-sans-serif, sans-serif";
    context.fillStyle = COLORS.muted;
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let year = 2010; year <= 2025; year += 2) {
      const timestamp = Date.UTC(year, 0, 1);
      const x = xForTimestamp(timestamp, plot);
      context.strokeStyle = "#edf0ed";
      context.beginPath();
      context.moveTo(x, plot.y);
      context.lineTo(x, plot.y + plot.height);
      context.stroke();
      context.fillText(String(year), x, plot.y + plot.height + 8);
    }
    context.restore();
  }

  function drawLineSeries(
    context,
    values,
    throughIndex,
    plot,
    yScale,
    color,
    lineWidth,
  ) {
    const step = Math.max(1, Math.floor((throughIndex + 1) / Math.max(1, plot.width * 1.7)));
    context.save();
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    let started = false;
    for (let index = 0; index <= throughIndex; index += step) {
      const value = values[index];
      if (value === null || !Number.isFinite(value)) continue;
      const x = xForIndex(index, plot);
      const y = yScale(value);
      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    }
    if (throughIndex % step !== 0 && Number.isFinite(values[throughIndex])) {
      context.lineTo(
        xForIndex(throughIndex, plot),
        yScale(values[throughIndex]),
      );
    }
    context.stroke();
    context.restore();
  }

  function drawCurrentDots(context, plot, seriesKey, yScale) {
    [
      [state.selectedPolicy, SERIES_COLORS[state.selectedPolicy], 4],
      ["qqq", COLORS.qqq, 3],
      ["spy", COLORS.spy, 3],
    ].forEach(([id, color, radius]) => {
      const value = state.data.series[id][seriesKey][state.index];
      drawDot(
        context,
        xForIndex(state.index, plot),
        yScale(value),
        color,
        radius,
      );
    });
  }

  function drawDot(context, x, y, color, radius) {
    context.save();
    context.fillStyle = COLORS.surface;
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  }

  function drawEventMarkers(context, plot, throughIndex, includeLabel = true) {
    context.save();
    state.data.events.forEach((event, eventIndex) => {
      if (event.index > throughIndex) return;
      const x = xForIndex(event.index, plot);
      context.strokeStyle = "#9fa9a4";
      context.lineWidth = 1;
      context.setLineDash([3, 4]);
      context.beginPath();
      context.moveTo(x, plot.y);
      context.lineTo(x, plot.y + plot.height);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = COLORS.surface;
      context.strokeStyle = COLORS.muted;
      context.beginPath();
      context.arc(x, plot.y + 7, 3, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      if (includeLabel && plot.width > 640) {
        context.save();
        context.translate(x + 4, plot.y + 15 + (eventIndex % 2) * 12);
        context.fillStyle = COLORS.muted;
        context.font = "9px Inter, ui-sans-serif, sans-serif";
        context.fillText(event.label, 0, 0);
        context.restore();
      }
    });
    context.restore();
  }

  function drawCursor(context, plot) {
    const x = xForIndex(state.index, plot);
    context.save();
    context.strokeStyle = "#53615b";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, plot.y);
    context.lineTo(x, plot.y + plot.height);
    context.stroke();
    context.restore();
  }

  function xForIndex(index, plot) {
    return xForTimestamp(state.timestamps[index], plot);
  }

  function xForTimestamp(timestamp, plot) {
    const ratio =
      (timestamp - state.timestamps[0]) /
      (state.timestamps[state.timestamps.length - 1] - state.timestamps[0]);
    return plot.x + clamp(ratio, 0, 1) * plot.width;
  }

  function logScale(value, domain, plot) {
    const [minimum, maximum] = domain;
    const ratio =
      (Math.log(value) - Math.log(minimum)) /
      (Math.log(maximum) - Math.log(minimum));
    return plot.y + plot.height - clamp(ratio, 0, 1) * plot.height;
  }

  function linearScale(value, domain, plot) {
    const [minimum, maximum] = domain;
    const ratio = (value - minimum) / (maximum - minimum);
    return plot.y + plot.height - clamp(ratio, 0, 1) * plot.height;
  }

  async function exportVideo() {
    if (state.exporting) return;
    if (!HTMLCanvasElement.prototype.captureStream || !window.MediaRecorder) {
      elements.exportStatus.textContent =
        "Video export is not supported by this browser.";
      return;
    }

    stopPlayback();
    state.exporting = true;
    elements.videoButton.disabled = true;
    elements.imageButton.disabled = true;
    elements.exportStatus.textContent = "Rendering video: 0%";

    const canvas = document.createElement("canvas");
    canvas.width = 1280;
    canvas.height = 720;
    const context = canvas.getContext("2d");
    const stream = canvas.captureStream(30);
    const mimeType = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ].find((type) => MediaRecorder.isTypeSupported(type)) || "";

    try {
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType, videoBitsPerSecond: 6_000_000 } : undefined,
      );
      const chunks = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      });
      const stopped = new Promise((resolve, reject) => {
        recorder.addEventListener("stop", resolve, { once: true });
        recorder.addEventListener("error", reject, { once: true });
      });

      recorder.start(250);
      const duration = 14000;
      const startedAt = performance.now();
      await new Promise((resolve) => {
        const frame = (now) => {
          const progress = clamp((now - startedAt) / duration, 0, 1);
          const index = Math.round(progress * (state.data.dates.length - 1));
          drawExportFrame(context, canvas.width, canvas.height, index);
          elements.exportStatus.textContent =
            `Rendering video: ${Math.round(progress * 100)}%`;
          if (progress < 1) {
            requestAnimationFrame(frame);
          } else {
            window.setTimeout(resolve, 300);
          }
        };
        requestAnimationFrame(frame);
      });

      recorder.stop();
      await stopped;
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunks, { type: mimeType || "video/webm" });
      downloadBlob(blob, `rlqqq-${state.selectedPolicy}-replay.webm`);
      elements.exportStatus.textContent =
        "Video exported as WebM.";
    } catch (error) {
      console.error(error);
      elements.exportStatus.textContent =
        "Video export failed in this browser.";
    } finally {
      state.exporting = false;
      elements.videoButton.disabled = false;
      elements.imageButton.disabled = false;
    }
  }

  function exportImage() {
    const canvas = document.createElement("canvas");
    canvas.width = 1280;
    canvas.height = 720;
    const context = canvas.getContext("2d");
    drawExportFrame(context, canvas.width, canvas.height, state.index);
    canvas.toBlob((blob) => {
      if (!blob) return;
      downloadBlob(
        blob,
        `rlqqq-${state.selectedPolicy}-${state.data.dates[state.index]}.png`,
      );
      elements.exportStatus.textContent = "Current frame exported as PNG.";
    }, "image/png");
  }

  function drawExportFrame(context, width, height, index) {
    const policy = state.data.policies[state.selectedPolicy];
    const decision = decisionAt(index, state.selectedPolicy);
    const date = state.dates[index];
    const signals = state.data.signals;

    context.fillStyle = "#f3f5f2";
    context.fillRect(0, 0, width, height);

    context.fillStyle = COLORS.ink;
    context.fillRect(0, 0, width, 84);
    context.fillStyle = "#ffffff";
    context.font = "700 22px Inter, ui-sans-serif, sans-serif";
    context.fillText("RLQQQ", 42, 38);
    context.fillStyle = "#aeb9b4";
    context.font = "12px Inter, ui-sans-serif, sans-serif";
    context.fillText("POLICY REPLAY / OUT-OF-SAMPLE", 42, 59);
    context.textAlign = "right";
    context.fillStyle = "#ffffff";
    context.font = "600 20px Inter, ui-sans-serif, sans-serif";
    context.fillText(dateFormatter.format(date), width - 42, 42);
    context.textAlign = "left";

    const chartPanel = { x: 36, y: 108, width: 862, height: 548 };
    const detailPanel = { x: 918, y: 108, width: 326, height: 548 };
    drawCanvasPanel(context, chartPanel);
    drawCanvasPanel(context, detailPanel);

    context.fillStyle = COLORS.muted;
    context.font = "700 11px Inter, ui-sans-serif, sans-serif";
    context.fillText("NORMALIZED WEALTH", chartPanel.x + 22, chartPanel.y + 28);
    context.fillStyle = COLORS.ink;
    context.font = "680 19px Inter, ui-sans-serif, sans-serif";
    context.fillText(policy.name, chartPanel.x + 22, chartPanel.y + 54);

    const plot = {
      x: chartPanel.x + 70,
      y: chartPanel.y + 84,
      width: chartPanel.width - 94,
      height: chartPanel.height - 142,
    };
    drawExportPlot(context, plot, index);

    const progress = index / (state.data.dates.length - 1);
    context.fillStyle = "#e3e8e5";
    context.fillRect(chartPanel.x + 22, chartPanel.y + chartPanel.height - 14, chartPanel.width - 44, 4);
    context.fillStyle = SERIES_COLORS[state.selectedPolicy];
    context.fillRect(
      chartPanel.x + 22,
      chartPanel.y + chartPanel.height - 14,
      (chartPanel.width - 44) * progress,
      4,
    );

    const dx = detailPanel.x + 22;
    context.fillStyle = COLORS.muted;
    context.font = "700 11px Inter, ui-sans-serif, sans-serif";
    context.fillText("TARGET EXPOSURE", dx, detailPanel.y + 30);
    context.fillStyle = COLORS.ink;
    context.font = "700 48px Inter, ui-sans-serif, sans-serif";
    context.fillText(`${decision.target.toFixed(2)}x`, dx, detailPanel.y + 82);

    const gaugeX = dx;
    const gaugeY = detailPanel.y + 101;
    const gaugeWidth = detailPanel.width - 44;
    context.fillStyle = "#e6ebe8";
    context.fillRect(gaugeX, gaugeY, gaugeWidth, 10);
    context.fillStyle = SERIES_COLORS[state.selectedPolicy];
    context.fillRect(
      gaugeX,
      gaugeY,
      gaugeWidth * clamp(decision.target / 1.5, 0, 1),
      10,
    );
    context.fillStyle = COLORS.muted;
    context.font = "11px Inter, ui-sans-serif, sans-serif";
    drawDetailRow(context, dx, detailPanel.y + 145, detailPanel.width - 44, decision.baseLabel, `${decision.base.toFixed(2)}x`);
    drawDetailRow(context, dx, detailPanel.y + 181, detailPanel.width - 44, decision.tiltLabel, `${decision.tilt.toFixed(2)}x`);
    drawDetailRow(context, dx, detailPanel.y + 217, detailPanel.width - 44, "21d volatility", formatPercent(signals.realizedVol21[index], 1));
    drawDetailRow(context, dx, detailPanel.y + 253, detailPanel.width - 44, "QQQ from peak", formatPercent(signals.marketDrawdown[index], 1));

    context.fillStyle = "#e5f2ed";
    context.fillRect(dx, detailPanel.y + 286, detailPanel.width - 44, 112);
    context.fillStyle = COLORS.policy;
    context.fillRect(dx, detailPanel.y + 286, 4, 112);
    context.fillStyle = "#42504a";
    context.font = "12px Inter, ui-sans-serif, sans-serif";
    wrapCanvasText(
      context,
      decisionNarrative(
        decision,
        signals.realizedVol21[index],
        signals.marketDrawdown[index],
      ),
      dx + 14,
      detailPanel.y + 310,
      detailPanel.width - 72,
      18,
      5,
    );

    const currentWealth =
      state.data.series[state.selectedPolicy].wealth[index] * 10000;
    context.fillStyle = COLORS.muted;
    context.font = "700 10px Inter, ui-sans-serif, sans-serif";
    context.fillText("$10,000 BECAME", dx, detailPanel.y + 438);
    context.fillStyle = COLORS.ink;
    context.font = "700 26px Inter, ui-sans-serif, sans-serif";
    context.fillText(dollarFormatter.format(currentWealth), dx, detailPanel.y + 470);

    context.fillStyle = COLORS.warning;
    context.font = "700 10px Inter, ui-sans-serif, sans-serif";
    context.fillText(
      state.selectedPolicy === "composite"
        ? "POST-HOC RESEARCH CANDIDATE"
        : policy.status.toUpperCase(),
      dx,
      detailPanel.y + 509,
    );
    context.fillStyle = COLORS.muted;
    context.font = "10px Inter, ui-sans-serif, sans-serif";
    context.fillText("Research result, not investment advice.", dx, detailPanel.y + 531);
  }

  function drawCanvasPanel(context, panel) {
    context.save();
    context.fillStyle = "#ffffff";
    context.strokeStyle = COLORS.border;
    context.lineWidth = 1;
    context.beginPath();
    context.roundRect(panel.x, panel.y, panel.width, panel.height, 6);
    context.fill();
    context.stroke();
    context.restore();
  }

  function drawExportPlot(context, plot, index) {
    const ticks = [1, 2, 5, 10, 20].filter(
      (tick) => tick >= state.wealthDomain[0] && tick <= state.wealthDomain[1],
    );
    context.font = "10px Inter, ui-sans-serif, sans-serif";
    context.textAlign = "right";
    context.textBaseline = "middle";
    ticks.forEach((tick) => {
      const y = logScale(tick, state.wealthDomain, plot);
      context.strokeStyle = COLORS.border;
      context.beginPath();
      context.moveTo(plot.x, y);
      context.lineTo(plot.x + plot.width, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.fillText(compactDollar(tick * 10000), plot.x - 10, y);
    });
    context.textAlign = "left";

    drawLineSeries(
      context,
      state.data.series.spy.wealth,
      index,
      plot,
      (value) => logScale(value, state.wealthDomain, plot),
      COLORS.spy,
      2,
    );
    drawLineSeries(
      context,
      state.data.series.qqq.wealth,
      index,
      plot,
      (value) => logScale(value, state.wealthDomain, plot),
      COLORS.qqq,
      2,
    );
    drawLineSeries(
      context,
      state.data.series[state.selectedPolicy].wealth,
      index,
      plot,
      (value) => logScale(value, state.wealthDomain, plot),
      SERIES_COLORS[state.selectedPolicy],
      3,
    );

    const x = xForIndex(index, plot);
    context.strokeStyle = "#53615b";
    context.beginPath();
    context.moveTo(x, plot.y);
    context.lineTo(x, plot.y + plot.height);
    context.stroke();

    const legendY = plot.y + plot.height + 18;
    [
      [state.data.policies[state.selectedPolicy].shortName, SERIES_COLORS[state.selectedPolicy]],
      ["QQQ", COLORS.qqq],
      ["S&P 500", COLORS.spy],
    ].forEach(([label, color], legendIndex) => {
      const legendX = plot.x + legendIndex * 150;
      context.fillStyle = color;
      context.fillRect(legendX, legendY - 2, 18, 3);
      context.fillStyle = COLORS.muted;
      context.font = "11px Inter, ui-sans-serif, sans-serif";
      context.fillText(label, legendX + 25, legendY);
    });
  }

  function drawDetailRow(context, x, y, width, label, value) {
    context.strokeStyle = COLORS.border;
    context.beginPath();
    context.moveTo(x, y + 15);
    context.lineTo(x + width, y + 15);
    context.stroke();
    context.fillStyle = COLORS.muted;
    context.textAlign = "left";
    context.fillText(label, x, y);
    context.fillStyle = COLORS.ink;
    context.textAlign = "right";
    context.font = "700 11px SFMono-Regular, Consolas, monospace";
    context.fillText(value, x + width, y);
    context.textAlign = "left";
    context.font = "11px Inter, ui-sans-serif, sans-serif";
  }

  function wrapCanvasText(context, text, x, y, maxWidth, lineHeight, maxLines) {
    const words = text.split(" ");
    let line = "";
    let lineCount = 0;
    for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
      const test = line ? `${line} ${words[wordIndex]}` : words[wordIndex];
      if (context.measureText(test).width > maxWidth && line) {
        context.fillText(line, x, y + lineCount * lineHeight);
        line = words[wordIndex];
        lineCount += 1;
        if (lineCount >= maxLines) return;
      } else {
        line = test;
      }
    }
    if (lineCount < maxLines) {
      context.fillText(line, x, y + lineCount * lineHeight);
    }
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function showLoadError(error) {
    console.error(error);
    elements.chartLoading.textContent =
      "The policy data could not be loaded. Serve the docs directory over HTTP.";
    elements.chartLoading.style.color = "#b5473c";
  }

  function formatPercent(value, digits = 1) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatSigned(value, digits = 2) {
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
  }

  function formatCompactDate(dateString) {
    return compactDateFormatter.format(new Date(`${dateString}T00:00:00Z`));
  }

  function compactDollar(value) {
    if (Math.abs(value) >= 1000000) {
      return `$${(value / 1000000).toFixed(value >= 10000000 ? 0 : 1)}m`;
    }
    if (Math.abs(value) >= 1000) {
      return `$${(value / 1000).toFixed(0)}k`;
    }
    return `$${value.toFixed(0)}`;
  }

  function returnClass(value) {
    return value < 0 ? "return-negative" : "return-positive";
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }
})();
