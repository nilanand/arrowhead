const PAGE = document.body?.dataset?.page || "";
const DATA_SYMBOL = (document.body?.dataset?.symbol || "").toUpperCase();

let symbolChart = null;

function qs(id) {
  return document.getElementById(id);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${fmtNum(Number(value) * 100, digits)}%`;
}

function fmtTs(ts) {
  if (!ts) return "-";
  const dt = new Date(Number(ts) * 1000);
  if (Number.isNaN(dt.getTime())) return "-";
  return dt.toLocaleString();
}

function fmtRegimeTag(tag) {
  if (!tag) return "RANGE";
  return String(tag).replaceAll("_", " ");
}

function cssVar(name, fallback) {
  const root = document.documentElement;
  const value = root ? getComputedStyle(root).getPropertyValue(name).trim() : "";
  return value || fallback;
}

function setNavActive() {
  const pageToNav = {
    dashboard: "dashboard",
    screener: "screener",
    symbol: "screener",
    heatmap: "explore",
    correlations: "explore",
    seasonality: "explore",
    events: "events",
    alerts: "alerts",
  };
  const key = pageToNav[PAGE];
  if (!key) return;
  const node = document.querySelector(`.nav-item[data-nav="${key}"]`);
  if (node) node.classList.add("active");
}

function setHealth(meta) {
  const pill = qs("statusPill");
  const dot = qs("statusDot");
  const text = qs("statusText");
  const metaEl = qs("statusMeta");
  if (!pill || !dot || !text || !metaEl) return;

  const poller = meta?.poller || {};
  const lastOkTs = Number(poller.last_ok_ts || 0);
  const nowTs = Math.floor(Date.now() / 1000);
  const age = lastOkTs > 0 ? nowTs - lastOkTs : Infinity;

  let cls = "warn";
  let dotCls = "dot-warn";
  let label = "Degraded";
  if (age <= Number(meta?.config?.poll_seconds || 30) * 3) {
    cls = "ok";
    dotCls = "dot-ok";
    label = "Healthy";
  } else if (age > 600) {
    cls = "bad";
    dotCls = "dot-bad";
    label = "Stale";
  }

  pill.className = `health ${cls}`;
  dot.className = `dot ${dotCls}`;
  text.textContent = label;

  const tracked = Number(poller.tracked_realtime_symbols || 0);
  const universe = Number(poller.universe_size || 0);
  const quotes = Number(poller.last_quotes_seen || 0);
  metaEl.textContent = `provider=${meta.provider} | tracked=${tracked}/${universe} | quotes=${quotes} | last_ok=${fmtTs(lastOkTs)}`;
}

function renderDashboardTableRows(targetId, rows, columns) {
  const body = qs(targetId);
  if (!body) return;
  if (!rows || rows.length === 0) {
    body.innerHTML = `<tr><td colspan="${columns.length}">No rows</td></tr>`;
    return;
  }

  body.innerHTML = rows
    .map((row) => {
      return `<tr>${columns
        .map((col) => {
          const value = typeof col.render === "function" ? col.render(row) : row[col.key];
          return `<td>${value}</td>`;
        })
        .join("")}</tr>`;
    })
    .join("");
}

async function initDashboardPage() {
  async function load() {
    const [health, dashboard] = await Promise.all([fetchJson("/api/health"), fetchJson("/api/dashboard")]);
    setHealth(health);

    const regime = dashboard.regime || {};
    if (qs("gaugeRegime")) qs("gaugeRegime").textContent = fmtRegimeTag(regime.tag);
    if (qs("gaugeBreadth")) qs("gaugeBreadth").textContent = `${fmtNum(regime.breadth_pct_above_ema || 0, 2)}%`;
    if (qs("gaugeVolatility")) qs("gaugeVolatility").textContent = `${fmtNum(regime.median_volatility || 0, 2)}%ile`;
    if (qs("gaugeCorrelation")) qs("gaugeCorrelation").textContent = fmtNum(regime.cross_asset_correlation || 0, 3);
    if (qs("gaugeHighs")) qs("gaugeHighs").textContent = String(regime.breadth_new_highs ?? "0");
    if (qs("gaugeLows")) qs("gaugeLows").textContent = String(regime.breadth_new_lows ?? "0");

    renderDashboardTableRows(
      "dashboardVolRows",
      dashboard.top_volatile || [],
      [
        { render: (row) => `<a href="/symbol/${encodeURIComponent(row.symbol)}">${escapeHtml(row.symbol)}</a>` },
        { render: (row) => escapeHtml(fmtRegimeTag(row.regime_tag)) },
        { render: (row) => fmtNum(row.realized_vol, 4) },
        { render: (row) => `${fmtNum(row.volume_percentile, 1)}%` },
        { render: (row) => `<a class="btn btn-small" href="/symbol/${encodeURIComponent(row.symbol)}">Open</a>` },
      ]
    );

    renderDashboardTableRows(
      "dashboardDrawRows",
      dashboard.top_drawdown || [],
      [
        { render: (row) => `<a href="/symbol/${encodeURIComponent(row.symbol)}">${escapeHtml(row.symbol)}</a>` },
        { render: (row) => escapeHtml(fmtRegimeTag(row.regime_tag)) },
        { render: (row) => fmtPct(row.drawdown_30d, 2) },
        { render: (row) => fmtNum(row.trend_slope, 6) },
        { render: (row) => `<a class="btn btn-small" href="/symbol/${encodeURIComponent(row.symbol)}">Open</a>` },
      ]
    );
  }

  try {
    await load();
  } catch (err) {
    if (qs("statusMeta")) qs("statusMeta").textContent = `Error: ${err.message}`;
  }

  window.setInterval(() => {
    load().catch((err) => {
      if (qs("statusMeta")) qs("statusMeta").textContent = `Error: ${err.message}`;
    });
  }, 30000);
}

async function initScreenerPage() {
  const tfInput = qs("screenerTimeframe");
  const sortInput = qs("screenerSort");
  const refreshBtn = qs("screenerRefreshBtn");
  const meta = qs("screenerMeta");
  const body = qs("screenerRows");

  async function load() {
    const tf = tfInput?.value || "1m";
    const sort = sortInput?.value || "volatility_spike";
    const data = await fetchJson(`/api/screener?timeframe=${encodeURIComponent(tf)}&sort=${encodeURIComponent(sort)}&limit=500`);
    if (meta) {
      meta.textContent = `mode=${data.signals_mode} | timeframe=${data.timeframe} | rows=${data.count}`;
    }

    const rows = data.rows || [];
    if (!rows.length) {
      if (body) body.innerHTML = `<tr><td colspan="10">No rows</td></tr>`;
      return;
    }

    if (body) {
      body.innerHTML = rows
        .map((row) => {
          const action = row.signal_action || "-";
          return `
            <tr>
              <td><a href="/symbol/${encodeURIComponent(row.symbol)}">${escapeHtml(row.symbol)}</a></td>
              <td>${escapeHtml(row.exchange || "-")}</td>
              <td>${escapeHtml(fmtRegimeTag(row.regime_tag))}</td>
              <td>${fmtNum(row.realized_vol, 4)}</td>
              <td>${fmtNum(row.trend_slope, 6)}</td>
              <td>${fmtPct(row.drawdown_30d, 2)}</td>
              <td>${fmtNum(row.volume_percentile, 1)}%</td>
              <td>${fmtNum(row.signal_score, 3)}</td>
              <td>${escapeHtml(action)}</td>
              <td><a class="btn btn-small" href="/symbol/${encodeURIComponent(row.symbol)}">Open</a></td>
            </tr>
          `;
        })
        .join("");
    }
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));
  tfInput?.addEventListener("change", () => refreshBtn?.click());
  sortInput?.addEventListener("change", () => refreshBtn?.click());

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

function drawSymbolChart(plotEl, payload) {
  if (!plotEl || typeof uPlot === "undefined") return;
  const bars = payload.bars || [];
  if (!bars.length) {
    plotEl.innerHTML = "No bars available";
    return;
  }

  const x = bars.map((row) => Number(row.bar_ts));
  const close = bars.map((row) => Number(row.close));
  const ema20 = (payload.overlays?.ema20 || []).map((value) => Number(value));
  const ema100 = (payload.overlays?.ema100 || []).map((value) => Number(value));

  if (symbolChart) {
    symbolChart.destroy();
    symbolChart = null;
  }

  const opts = {
    width: Math.max(plotEl.clientWidth, 640),
    height: 360,
    scales: {
      x: { time: true },
      y: { auto: true },
    },
    series: [
      {},
      { label: "Close", stroke: cssVar("--text-primary", "#ffffff"), width: 2 },
      { label: "EMA20", stroke: cssVar("--text-muted", "#aaaaaa"), width: 1.6, dash: [6, 2] },
      { label: "EMA100", stroke: cssVar("--text-dim", "#666666"), width: 1.6, dash: [2, 2] },
    ],
    axes: [
      {
        stroke: cssVar("--border", "#1c1c1c"),
        grid: { stroke: "#1a1a1a", width: 1 },
        ticks: { stroke: cssVar("--border", "#1c1c1c"), width: 1 },
        font: "12px SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace",
        labelFont: "12px SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace",
        values: (u, vals) => vals.map((v) => new Date(v * 1000).toLocaleDateString()),
      },
      {
        stroke: cssVar("--border", "#1c1c1c"),
        grid: { stroke: "#1a1a1a", width: 1 },
        ticks: { stroke: cssVar("--border", "#1c1c1c"), width: 1 },
        font: "12px SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace",
        labelFont: "12px SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace",
        values: (u, vals) => vals.map((v) => fmtNum(v, 2)),
      },
    ],
    legend: {
      show: true,
      markers: {
        width: 8,
        dash: "solid",
      },
    },
    cursor: {
      x: true,
      y: true,
    },
  };

  symbolChart = new uPlot(opts, [x, close, ema20, ema100], plotEl);
}

async function initSymbolPage() {
  const tfInput = qs("symbolTimeframe");
  const windowInput = qs("symbolWindow");
  const refreshBtn = qs("symbolRefreshBtn");
  const meta = qs("symbolMeta");
  const plot = qs("symbolPlot");
  const featuresBody = qs("symbolFeaturesRows");
  const eventsBody = qs("symbolEventsRows");

  async function load() {
    const timeframe = tfInput?.value || "1m";
    const window = windowInput?.value || "30d";

    const [bars, features, events] = await Promise.all([
      fetchJson(`/api/symbol/${encodeURIComponent(DATA_SYMBOL)}/bars?timeframe=${encodeURIComponent(timeframe)}&window=${encodeURIComponent(window)}&limit=6000`),
      fetchJson(`/api/symbol/${encodeURIComponent(DATA_SYMBOL)}/features`),
      fetchJson(`/api/symbol/${encodeURIComponent(DATA_SYMBOL)}/events?limit=200`),
    ]);

    const nBars = (bars.bars || []).length;
    if (meta) {
      meta.textContent = `${DATA_SYMBOL} | timeframe=${timeframe} | window=${window} | bars=${nBars}`;
    }

    drawSymbolChart(plot, bars);

    const featureRows = features.rows || [];
    if (featuresBody) {
      featuresBody.innerHTML = featureRows.length
        ? featureRows
            .map((row) => `
              <tr>
                <td>${escapeHtml(row.timeframe)}</td>
                <td>${escapeHtml(fmtRegimeTag(row.regime_tag))}</td>
                <td>${fmtNum(row.rsi, 2)}</td>
                <td>${fmtNum(row.atr, 4)}</td>
                <td>${fmtNum(row.realized_vol, 4)}</td>
                <td>${fmtPct(row.drawdown_30d, 2)}</td>
                <td>${escapeHtml(row.signal_action || "-")} (${fmtNum(row.signal_score, 3)})</td>
              </tr>
            `)
            .join("")
        : `<tr><td colspan="7">No features available</td></tr>`;
    }

    const eventRows = events.rows || [];
    if (eventsBody) {
      eventsBody.innerHTML = eventRows.length
        ? eventRows
            .map((row) => `
              <tr>
                <td>${fmtTs(row.ts)}</td>
                <td>${escapeHtml(row.event_type)}</td>
                <td>${escapeHtml(String(row.severity))}</td>
                <td><code>${escapeHtml(JSON.stringify(row.details || {}))}</code></td>
              </tr>
            `)
            .join("")
        : `<tr><td colspan="4">No events available</td></tr>`;
    }
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));
  tfInput?.addEventListener("change", () => refreshBtn?.click());
  windowInput?.addEventListener("change", () => refreshBtn?.click());

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

async function initHeatmapPage() {
  const tfInput = qs("heatmapTf");
  const refreshBtn = qs("heatmapRefreshBtn");
  const body = qs("heatmapRows");
  const meta = qs("heatmapMeta");

  async function load() {
    const tf = tfInput?.value || "1d";
    const data = await fetchJson(`/api/explore/heatmap?timeframe=${encodeURIComponent(tf)}&limit=1000`);
    const rows = data.rows || [];
    if (meta) meta.textContent = `timeframe=${tf} | rows=${rows.length}`;

    if (!body) return;
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="6">No rows</td></tr>`;
      return;
    }

    body.innerHTML = rows
      .map((row) => {
        const score = Number(row.color_score || 0);
        const alpha = Math.min(Math.abs(score) / 4, 1);
        const shade = score >= 0 ? 255 : 170;
        const bg = `rgba(${shade}, ${shade}, ${shade}, ${0.04 + alpha * 0.16})`;

        return `
          <tr style="background:${bg};">
            <td><a href="/symbol/${encodeURIComponent(row.symbol)}">${escapeHtml(row.symbol)}</a></td>
            <td>${escapeHtml(fmtRegimeTag(row.regime_tag))}</td>
            <td>${fmtNum(row.zscore_30d, 3)}</td>
            <td>${fmtPct(row.drawdown_30d, 2)}</td>
            <td>${fmtNum(row.realized_vol, 4)}</td>
            <td>${fmtNum(row.volume_percentile, 1)}%</td>
          </tr>
        `;
      })
      .join("");
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));
  tfInput?.addEventListener("change", () => refreshBtn?.click());

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

async function initCorrelationsPage() {
  const symbolsInput = qs("corrSymbols");
  const refreshBtn = qs("corrRefreshBtn");
  const meta = qs("corrMeta");
  const head = qs("corrHead");
  const body = qs("corrBody");

  async function load() {
    const symbols = symbolsInput?.value || "SPY,QQQ,DIA,IWM,TLT,GLD,USO";
    const data = await fetchJson(`/api/explore/correlations?symbols=${encodeURIComponent(symbols)}&lookback_days=90`);
    const labels = data.symbols || [];
    const matrix = data.matrix || [];

    if (meta) {
      meta.textContent = `symbols=${labels.length} | lookback_days=${data.lookback_days}`;
    }

    if (!head || !body) return;
    if (!labels.length) {
      head.innerHTML = "";
      body.innerHTML = `<tr><td>No correlation rows</td></tr>`;
      return;
    }

    head.innerHTML = `<tr><th>Symbol</th>${labels.map((s) => `<th>${escapeHtml(s)}</th>`).join("")}</tr>`;
    body.innerHTML = labels
      .map((left, i) => {
        const values = matrix[i] || [];
        const cols = values
          .map((value) => {
            const n = Number(value || 0);
            const alpha = Math.min(Math.abs(n), 1);
            const shade = n >= 0 ? 255 : 175;
            const bg = `rgba(${shade}, ${shade}, ${shade}, ${0.05 + alpha * 0.16})`;
            return `<td style="background:${bg};">${fmtNum(n, 3)}</td>`;
          })
          .join("");
        return `<tr><td><strong>${escapeHtml(left)}</strong></td>${cols}</tr>`;
      })
      .join("");
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

async function initSeasonalityPage() {
  const symbolInput = qs("seasonalitySymbol");
  const tfInput = qs("seasonalityTf");
  const refreshBtn = qs("seasonalityRefreshBtn");
  const meta = qs("seasonalityMeta");
  const body = qs("seasonalityRows");

  async function load() {
    const symbol = (symbolInput?.value || "SPY").toUpperCase().trim();
    const tf = tfInput?.value || "1d";
    const data = await fetchJson(`/api/explore/seasonality?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(tf)}`);

    if (meta) {
      meta.textContent = `${data.symbol} | timeframe=${data.timeframe} | bars=${data.bars}`;
    }

    const rows = data.series || [];
    if (!body) return;
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="4">No seasonality rows</td></tr>`;
      return;
    }

    body.innerHTML = rows
      .map((row) => `
        <tr>
          <td>${escapeHtml(row.bucket)}</td>
          <td>${fmtPct(row.mean_return, 3)}</td>
          <td>${fmtPct(row.median_return, 3)}</td>
          <td>${escapeHtml(String(row.samples))}</td>
        </tr>
      `)
      .join("");
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

async function initEventsPage() {
  const symbolInput = qs("eventsSymbol");
  const typeInput = qs("eventsType");
  const refreshBtn = qs("eventsRefreshBtn");
  const meta = qs("eventsMeta");
  const body = qs("eventsRows");

  async function load() {
    const symbol = (symbolInput?.value || "").toUpperCase().trim();
    const eventType = (typeInput?.value || "").trim();
    const params = new URLSearchParams({ limit: "500" });
    if (symbol) params.set("symbol", symbol);
    if (eventType) params.set("event_type", eventType);

    const data = await fetchJson(`/api/events?${params.toString()}`);
    const rows = data.rows || [];
    if (meta) meta.textContent = `events=${rows.length}`;

    if (!body) return;
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="5">No events</td></tr>`;
      return;
    }

    body.innerHTML = rows
      .map((row) => `
        <tr>
          <td>${fmtTs(row.ts)}</td>
          <td>${escapeHtml(row.symbol || "MARKET")}</td>
          <td>${escapeHtml(row.event_type)}</td>
          <td>${escapeHtml(String(row.severity))}</td>
          <td><code>${escapeHtml(JSON.stringify(row.details || {}))}</code></td>
        </tr>
      `)
      .join("");
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

async function initAlertsPage() {
  const refreshBtn = qs("alertsRefreshBtn");
  const meta = qs("alertsMeta");
  const body = qs("alertsRows");

  async function load() {
    const data = await fetchJson(`/api/alerts?limit=500`);
    const rows = data.rows || [];
    if (meta) meta.textContent = `alerts=${rows.length}`;

    if (!body) return;
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="4">No alerts</td></tr>`;
      return;
    }

    body.innerHTML = rows
      .map((row) => `
        <tr>
          <td>${fmtTs(row.ts)}</td>
          <td>${escapeHtml(row.alert_type)}</td>
          <td>${escapeHtml(row.symbol || "MARKET")}</td>
          <td><code>${escapeHtml(JSON.stringify(row.payload || {}))}</code></td>
        </tr>
      `)
      .join("");
  }

  refreshBtn?.addEventListener("click", () => load().catch((err) => {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }));

  try {
    await load();
  } catch (err) {
    if (meta) meta.textContent = `Error: ${err.message}`;
  }
}

setNavActive();

if (PAGE === "dashboard") initDashboardPage();
if (PAGE === "screener") initScreenerPage();
if (PAGE === "symbol") initSymbolPage();
if (PAGE === "heatmap") initHeatmapPage();
if (PAGE === "correlations") initCorrelationsPage();
if (PAGE === "seasonality") initSeasonalityPage();
if (PAGE === "events") initEventsPage();
if (PAGE === "alerts") initAlertsPage();
