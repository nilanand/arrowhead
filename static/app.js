const page = document.body.dataset.page;

function formatCoins(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const n = Number(value || 0);
  const abs = Math.abs(n);
  if (abs >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(n / 1_000).toFixed(2)}k`;
  return `${Math.trunc(n)}`;
}

function formatPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatAgo(ts) {
  if (!ts) return "-";
  const sec = Math.max(Math.floor(Date.now() / 1000 - ts), 0);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

function formatTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sparklineSvg(points, width = 120, height = 28) {
  if (!Array.isArray(points) || points.length < 2) {
    return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="sparkline"><rect x="0" y="0" width="${width}" height="${height}" fill="transparent" /><line x1="0" y1="${height - 1}" x2="${width}" y2="${height - 1}" stroke="#31584f" stroke-width="1" /></svg>`;
  }
  const values = points
    .map((p) => Number(p.price_median))
    .filter((v) => Number.isFinite(v));
  if (values.length < 2) {
    return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="sparkline"><rect x="0" y="0" width="${width}" height="${height}" fill="transparent" /></svg>`;
  }

  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = Math.max(maxV - minV, 1e-9);

  const step = (width - 2) / Math.max(values.length - 1, 1);
  const path = values
    .map((v, i) => {
      const x = 1 + i * step;
      const y = 1 + (height - 3) * (1 - (v - minV) / range);
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return `
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="sparkline">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
      <path d="${path}" fill="none" stroke="#7ac7ab" stroke-width="1.5" vector-effect="non-scaling-stroke" />
    </svg>
  `.trim();
}

function applySparklinesFromPayload(rootEl, sparklineMap) {
  if (!rootEl || !sparklineMap || typeof sparklineMap !== "object") return;
  rootEl.querySelectorAll("[data-sparkline-item-key]").forEach((node) => {
    const itemKey = node.getAttribute("data-sparkline-item-key");
    const points = sparklineMap[itemKey] || [];
    node.innerHTML = sparklineSvg(points);
  });
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let data = {};
  try {
    data = await res.json();
  } catch (err) {
    data = { error: "Invalid JSON response" };
  }

  if (!res.ok) {
    throw new Error(data.error || data.message || `HTTP ${res.status}`);
  }
  return data;
}

function showError(el, message) {
  if (!el) return;
  if (!message) {
    el.style.display = "none";
    el.textContent = "";
    return;
  }
  el.style.display = "block";
  el.textContent = message;
}

let toastTimerHandle = null;
function showToast(message) {
  const el = document.getElementById("graphWatchToast");
  if (!el || !message) return;
  el.textContent = String(message);
  el.style.display = "block";
  clearTimeout(toastTimerHandle);
  toastTimerHandle = setTimeout(() => {
    el.style.display = "none";
    el.textContent = "";
  }, 1800);
}

function initIndexPage() {
  const form = document.getElementById("filtersForm");
  const resultsBody = document.getElementById("resultsBody");
  const statusPill = document.getElementById("statusPill");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const statusMeta = document.getElementById("statusMeta");
  const statusError = document.getElementById("statusError");
  const pageInfo = document.getElementById("pageInfo");
  const prevPageBtn = document.getElementById("prevPageBtn");
  const nextPageBtn = document.getElementById("nextPageBtn");
  const fetchNowBtn = document.getElementById("fetchNowBtn");

  if (
    !form || !resultsBody || !statusPill || !statusDot || !statusText || !statusMeta
    || !statusError || !pageInfo || !prevPageBtn || !nextPageBtn || !fetchNowBtn
  ) {
    return;
  }

  const state = {
    page: 1,
    pageSize: 50,
    total: 0,
  };

  const collectFilters = () => {
    const params = new URLSearchParams();
    const query = document.getElementById("query").value.trim();
    const bin = document.getElementById("bin").value;
    const tier = document.getElementById("tier").value;
    const minPrice = document.getElementById("minPrice").value;
    const maxPrice = document.getElementById("maxPrice").value;
    const sort = document.getElementById("sort").value;
    const endingSoon = document.getElementById("endingSoon").checked;

    if (query) params.set("query", query);
    if (bin !== "") params.set("bin", bin);
    if (tier) params.set("tier", tier);
    if (minPrice) params.set("min_price", minPrice);
    if (maxPrice) params.set("max_price", maxPrice);
    if (sort) params.set("sort", sort);
    if (endingSoon) params.set("ending_soon", "1");

    params.set("page", String(state.page));
    params.set("page_size", String(state.pageSize));
    return params;
  };

  const renderRows = (items) => {
    if (!items.length) {
      resultsBody.innerHTML = `<tr><td colspan="7">No auctions matched your filters.</td></tr>`;
      return;
    }

    resultsBody.innerHTML = items
      .map((row) => {
        const binTag = row.bin ? '<span class="tag">BIN</span>' : '<span class="tag no">AUC</span>';
        const graphItemKey = row.canonical_item_key || "";
        const graphButton = graphItemKey
          ? `<a class="btn btn-small btn-ghost" href="/graph/${encodeURIComponent(graphItemKey)}">Graph</a>`
          : "";
        return `
          <tr>
            <td>${escapeHtml(row.item_name)}</td>
            <td>${binTag}</td>
            <td>${escapeHtml(row.tier)}</td>
            <td>${escapeHtml(row.price_fmt || formatCoins(row.price))}</td>
            <td>${escapeHtml(row.ends_in)}</td>
            <td class="cell-mono">${escapeHtml(row.seller || "-")}</td>
            <td class="col-actions">
              <div class="actions-stack">
                <a class="btn btn-small" href="/auction/${encodeURIComponent(row.uuid)}">Open</a>
                ${graphButton}
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  };

  const renderPagination = () => {
    const maxPage = Math.max(Math.ceil(state.total / state.pageSize), 1);
    pageInfo.textContent = `Page ${state.page} / ${maxPage} (${state.total} results)`;
    prevPageBtn.disabled = state.page <= 1;
    nextPageBtn.disabled = state.page >= maxPage;
  };

  const loadStatus = async () => {
    try {
      const status = await fetchJson("/api/status");
      const hasError = Boolean(status.last_error);
      const dataAge = Number(status.data_age_seconds ?? -1);
      const hasWarn = !hasError && (
        Number(status.consecutive_failures || 0) > 0
        || (Number.isFinite(dataAge) && dataAge > 90)
      );
      const healthState = hasError ? "bad" : (hasWarn ? "warn" : "ok");
      const healthText = hasError ? "ERROR" : (hasWarn ? "WARN" : "OK");

      statusPill.className = `health ${healthState}`;
      statusDot.className = `dot dot-${healthState}`;
      statusText.textContent = healthText;

      statusMeta.innerHTML = `
        <span>Data age: ${status.data_age_seconds ?? "-"}s |</span>
        <span>Last fetch: ${formatAgo(status.last_fetch_ts)} |</span>
        <span>Mode: ${escapeHtml(status.mode)} (${status.light_pages || 1} pages) |</span>
        <span>Auctions: ${status.counts?.auctions ?? 0} |</span>
        <span>Features: ${status.counts?.features ?? 0} |</span>
        <span>Recs: ${status.counts?.recommendations ?? 0} |</span>
        <span>Forecasts: ${status.counts?.forecasts ?? 0} |</span>
        <span>Extremes: ${status.counts?.extremes ?? 0} |</span>
        <span>Open positions: ${status.counts?.open_positions ?? 0} |</span>
        <span>Events: ${status.counts?.events ?? 0} |</span>
        <span>Watchlist: ${status.counts?.favorites ?? 0}</span>
      `;
      showError(statusError, status.last_error || "");
    } catch (err) {
      statusPill.className = "health bad";
      statusDot.className = "dot dot-bad";
      statusText.textContent = "ERROR";
      showError(statusError, err.message);
    }
  };

  const loadAuctions = async () => {
    try {
      const params = collectFilters();
      const data = await fetchJson(`/api/auctions?${params.toString()}`);
      state.total = data.total || 0;
      renderRows(data.items || []);
      renderPagination();
    } catch (err) {
      resultsBody.innerHTML = `<tr><td colspan="7">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    state.page = 1;
    loadAuctions();
  });

  prevPageBtn.addEventListener("click", () => {
    if (state.page <= 1) return;
    state.page -= 1;
    loadAuctions();
  });

  nextPageBtn.addEventListener("click", () => {
    const maxPage = Math.max(Math.ceil(state.total / state.pageSize), 1);
    if (state.page >= maxPage) return;
    state.page += 1;
    loadAuctions();
  });

  fetchNowBtn.addEventListener("click", async () => {
    fetchNowBtn.disabled = true;
    fetchNowBtn.textContent = "Fetching...";
    try {
      const data = await fetchJson("/api/fetch_now", { method: "POST" });
      if (!data.ok) {
        showError(statusError, data.message || data.error || "Fetch failed");
      }
    } catch (err) {
      showError(statusError, err.message);
    } finally {
      fetchNowBtn.disabled = false;
      fetchNowBtn.textContent = "Fetch Now";
      loadStatus();
      loadAuctions();
    }
  });

  loadStatus();
  loadAuctions();

  setInterval(loadStatus, 15_000);
  setInterval(loadAuctions, 20_000);
}

function initSignalsPage() {
  const meta = document.getElementById("signalsMeta");
  const buyBody = document.getElementById("buySignalsBody");
  const sellBody = document.getElementById("sellSignalsBody");
  const onlyLiquidToggle = document.getElementById("signalsOnlyLiquid");

  const liquidityRank = (row) => {
    const direct = row?.liquidity_rank;
    const nested = row?.feature?.liquidity_rank;
    const rank = direct ?? nested;
    return Number.isFinite(Number(rank)) ? Number(rank) : 0;
  };

  const renderBuys = (items) => {
    if (!items.length) {
      buyBody.innerHTML = `<tr><td colspan="13">No BUY recommendations right now.</td></tr>`;
      return;
    }

    buyBody.innerHTML = items
      .map((row) => {
        const f = row.feature || {};
        const bt = row.backtest || {};
        const btSummary = bt.trades
          ? `${formatPct(bt.win_rate)} / ${formatPct(bt.avg_return)} / ${formatPct(bt.max_drawdown)}`
          : "-";
        const graphUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        return `
          <tr>
            <td class="cell-mono"><a href="${graphUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${Number(row.score || 0).toFixed(3)}</td>
            <td>${Number(row.flip_score || 0).toFixed(3)}</td>
            <td>${escapeHtml(f.price_now_fmt || formatCoins(f.price_now))}</td>
            <td class="${Number(row.expected_profit_pct || 0) >= 0 ? "text-good" : "text-bad"}">${formatPct(row.expected_profit_pct)}</td>
            <td>${escapeHtml(f.median_24h_fmt || formatCoins(f.median_24h))}</td>
            <td>${Number(f.zscore_24h || 0).toFixed(3)}</td>
            <td>${Number(f.ema_fast || 0).toFixed(1)} / ${Number(f.ema_slow || 0).toFixed(1)}</td>
            <td>${escapeHtml(f.regime_tag || row.regime_tag || "-")}</td>
            <td>${liquidityRank(row).toFixed(1)}</td>
            <td>${escapeHtml(btSummary)}</td>
            <td><div data-sparkline-item-key="${escapeHtml(row.item_key)}"></div></td>
            <td>${escapeHtml(row.reason || "-")}</td>
          </tr>
        `;
      })
      .join("");
  };

  const renderSells = (items) => {
    if (!items.length) {
      sellBody.innerHTML = `<tr><td colspan="10">No SELL recommendations right now.</td></tr>`;
      return;
    }

    sellBody.innerHTML = items
      .map((row) => {
        const f = row.feature || {};
        const graphUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        return `
          <tr>
            <td class="cell-mono"><a href="${graphUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${Number(row.score || 0).toFixed(3)}</td>
            <td>${escapeHtml(f.price_now_fmt || formatCoins(f.price_now))}</td>
            <td>${Number(f.zscore_24h || 0).toFixed(3)}</td>
            <td>${Number(f.ema_fast || 0).toFixed(1)} / ${Number(f.ema_slow || 0).toFixed(1)}</td>
            <td>${escapeHtml(f.regime_tag || row.regime_tag || "-")}</td>
            <td>${liquidityRank(row).toFixed(1)}</td>
            <td>${Number(row.flip_score || 0).toFixed(3)}</td>
            <td><div data-sparkline-item-key="${escapeHtml(row.item_key)}"></div></td>
            <td>${escapeHtml(row.reason || "-")}</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    try {
      let liqRankMin = 60;
      let sparklinesEnabled = false;
      let sparklinesWindow = "6h";
      let sparklinesTopN = 50;
      try {
        const status = await fetchJson("/api/status");
        liqRankMin = Number(status.liq_rank_min ?? 60);
        sparklinesEnabled = Boolean(status.sparklines_enabled);
        sparklinesWindow = String(status.sparklines_window || "6h");
        sparklinesTopN = Number(status.sparklines_top_n || 50);
      } catch (_err) {
        liqRankMin = 60;
      }

      const onlyLiquid = onlyLiquidToggle ? onlyLiquidToggle.checked : true;
      const onlyLiquidParam = onlyLiquid ? "&only_liquid=1" : "";
      const [buyData, sellData] = await Promise.all([
        fetchJson(`/api/recommendations?action=BUY&limit=20${onlyLiquidParam}`),
        fetchJson(`/api/recommendations?action=SELL&limit=20${onlyLiquidParam}`),
      ]);

      renderBuys(buyData.items || []);
      renderSells(sellData.items || []);
      meta.textContent = `BUY: ${buyData.count || 0} • SELL: ${sellData.count || 0} • min rank ${liqRankMin} • updated ${new Date().toLocaleTimeString()}`;

      if (sparklinesEnabled) {
        const keys = [
          ...new Set(
            [...(buyData.items || []), ...(sellData.items || [])]
              .map((row) => row.item_key)
              .filter(Boolean)
              .slice(0, Math.max(1, sparklinesTopN))
          ),
        ];
        if (keys.length > 0) {
          const params = new URLSearchParams();
          params.set("window", sparklinesWindow);
          params.set("limit", String(Math.max(1, sparklinesTopN)));
          params.set("item_keys", keys.join(","));
          const sparkData = await fetchJson(`/api/sparklines?${params.toString()}`);
          applySparklinesFromPayload(buyBody, sparkData.items || {});
          applySparklinesFromPayload(sellBody, sparkData.items || {});
        }
      }
    } catch (err) {
      meta.textContent = `Error loading signals: ${err.message}`;
      buyBody.innerHTML = `<tr><td colspan="13">${escapeHtml(err.message)}</td></tr>`;
      sellBody.innerHTML = `<tr><td colspan="10">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  if (onlyLiquidToggle) {
    onlyLiquidToggle.addEventListener("change", () => {
      load();
    });
  }

  load();
  setInterval(load, 20_000);
}

function initPortfolioPage() {
  const meta = document.getElementById("portfolioMeta");
  const summaryEl = document.getElementById("portfolioSummary");
  const openBody = document.getElementById("openPositionsBody");
  const closedBody = document.getElementById("closedPositionsBody");

  const pnlClass = (value) => {
    const n = Number(value || 0);
    if (n > 0) return "text-good";
    if (n < 0) return "text-bad";
    return "";
  };

  const renderOpen = (items) => {
    if (!items.length) {
      openBody.innerHTML = `<tr><td colspan="8">No open positions.</td></tr>`;
      return;
    }

    openBody.innerHTML = items
      .map((row) => {
        return `
          <tr>
            <td>${row.id}</td>
            <td class="cell-mono">${escapeHtml(row.item_key)}</td>
            <td>${escapeHtml(row.entry_price_fmt || formatCoins(row.entry_price))}</td>
            <td>${escapeHtml(row.current_price_fmt || formatCoins(row.current_price))}</td>
            <td class="${pnlClass(row.unrealized_pnl)}">${escapeHtml(row.unrealized_pnl_fmt || formatCoins(row.unrealized_pnl))}</td>
            <td>${escapeHtml(formatTime(row.opened_ts))}</td>
            <td>${escapeHtml(row.duration_human || "-")}</td>
            <td>${escapeHtml(row.status)}</td>
          </tr>
        `;
      })
      .join("");
  };

  const renderClosed = (items) => {
    if (!items.length) {
      closedBody.innerHTML = `<tr><td colspan="9">No closed positions yet.</td></tr>`;
      return;
    }

    closedBody.innerHTML = items
      .map((row) => {
        return `
          <tr>
            <td>${row.id}</td>
            <td class="cell-mono">${escapeHtml(row.item_key)}</td>
            <td>${escapeHtml(row.entry_price_fmt || formatCoins(row.entry_price))}</td>
            <td>${escapeHtml(row.exit_price_fmt || formatCoins(row.exit_price))}</td>
            <td class="${pnlClass(row.pnl_after_fee)}">${escapeHtml(row.pnl_after_fee_fmt || formatCoins(row.pnl_after_fee))}</td>
            <td class="${pnlClass(row.profit_per_hour)}">${escapeHtml(row.profit_per_hour_fmt || formatCoins(row.profit_per_hour))}</td>
            <td>${escapeHtml(row.duration_human || "-")}</td>
            <td>${escapeHtml(formatTime(row.opened_ts))}</td>
            <td>${escapeHtml(formatTime(row.closed_ts))}</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    try {
      const [openData, closedData] = await Promise.all([
        fetchJson("/api/positions?status=OPEN&limit=200"),
        fetchJson("/api/positions?status=CLOSED&limit=200"),
      ]);
      renderOpen(openData.items || []);
      renderClosed(closedData.items || []);
      meta.textContent = `Open: ${openData.count || 0} • Closed: ${closedData.count || 0}`;
      const summary = closedData.summary || {};
      if (summaryEl) {
        summaryEl.textContent = `Closed trades: ${summary.closed_trades ?? 0} • Total PnL after fee: ${summary.total_pnl_after_fee_fmt || "-"} • Avg profit/hour: ${summary.avg_profit_per_hour_fmt || "-"} • Median duration: ${summary.median_duration_human || "-"}`;
      }
    } catch (err) {
      meta.textContent = `Error loading positions: ${err.message}`;
      openBody.innerHTML = `<tr><td colspan="8">${escapeHtml(err.message)}</td></tr>`;
      closedBody.innerHTML = `<tr><td colspan="9">${escapeHtml(err.message)}</td></tr>`;
      if (summaryEl) {
        summaryEl.textContent = `Error loading summary: ${err.message}`;
      }
    }
  };

  load();
  setInterval(load, 20_000);
}

function initBacktestsPage() {
  const meta = document.getElementById("backtestMeta");
  const bestBody = document.getElementById("bestBacktestsBody");
  const worstBody = document.getElementById("worstBacktestsBody");
  const allBody = document.getElementById("allBacktestsBody");

  const renderRows = (tbody, rows, colspan, emptyLabel) => {
    if (!rows || rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="${colspan}">${emptyLabel}</td></tr>`;
      return;
    }

    tbody.innerHTML = rows
      .map((row) => {
        return `
          <tr>
            <td class="cell-mono">${escapeHtml(row.item_key)}</td>
            <td>${row.trades}</td>
            <td>${formatPct(row.win_rate)}</td>
            <td class="${Number(row.avg_return || 0) >= 0 ? "text-good" : "text-bad"}">${formatPct(row.avg_return)}</td>
            <td class="text-bad">${formatPct(row.max_drawdown)}</td>
            <td>${row.lookback_hours}h</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    try {
      const data = await fetchJson("/api/backtest");
      renderRows(bestBody, data.best || [], 6, "No backtests yet.");
      renderRows(worstBody, data.worst || [], 6, "No backtests yet.");
      renderRows(allBody, data.items || [], 6, "No backtests yet.");
      meta.textContent = `Items: ${data.count || 0} • Last run: ${formatAgo(data.ts)}`;
    } catch (err) {
      meta.textContent = `Error loading backtests: ${err.message}`;
      bestBody.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
      worstBody.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
      allBody.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  load();
  setInterval(load, 30_000);
}

function initAlertsPage() {
  const meta = document.getElementById("alertsMeta");
  const body = document.getElementById("alertsBody");

  const load = async () => {
    try {
      const data = await fetchJson("/api/alerts?limit=50");
      const items = data.items || [];
      if (!items.length) {
        body.innerHTML = `<tr><td colspan="5">No alerts sent yet.</td></tr>`;
      } else {
        body.innerHTML = items
          .map((row) => {
            return `
              <tr>
                <td>${escapeHtml(formatTime(row.ts))}</td>
                <td>${escapeHtml(row.alert_type)}</td>
                <td class="cell-mono">${escapeHtml(row.item_key)}</td>
                <td class="cell-mono">${escapeHtml(row.uuid || "-")}</td>
                <td>${escapeHtml(row.snippet || "-")}</td>
              </tr>
            `;
          })
          .join("");
      }
      meta.textContent = `Total shown: ${items.length}`;
    } catch (err) {
      meta.textContent = `Error loading alerts: ${err.message}`;
      body.innerHTML = `<tr><td colspan="5">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  load();
  setInterval(load, 20_000);
}

function initSharkPage() {
  const meta = document.getElementById("sharkMeta");
  const error = document.getElementById("sharkError");
  const result = document.getElementById("sharkInjectResult");
  const injectBtn = document.getElementById("injectTestSharkBtn");

  if (!injectBtn) return;
  if (meta) {
    meta.textContent = "DEV tool: inject a test decimal-shark auction and run one scan.";
  }

  injectBtn.addEventListener("click", async () => {
    injectBtn.disabled = true;
    injectBtn.textContent = "Injecting...";
    showError(error, "");
    if (result) result.textContent = "";

    try {
      const data = await fetchJson("/api/shark/test_inject", { method: "POST" });
      if (meta) {
        meta.textContent = `Test shark injected at ${new Date().toLocaleTimeString()}`;
      }
      if (result) {
        result.textContent = `item_key=${data.item_key} | median_24h=${formatCoins(data.median_24h)} | injected_price=${formatCoins(data.injected_price)} | uuid=${data.uuid} | discord_sent=${Boolean(data.discord_sent)}`;
      }
    } catch (err) {
      showError(error, err.message);
      if (meta) {
        meta.textContent = "Inject test failed.";
      }
    } finally {
      injectBtn.disabled = false;
      injectBtn.textContent = "Inject Test Shark (DEV)";
    }
  });
}

function initMarketPage() {
  const meta = document.getElementById("marketMeta");
  const form = document.getElementById("marketForm");
  const itemKeySelect = document.getElementById("marketItemKey");
  const hoursSelect = document.getElementById("marketHours");
  const body = document.getElementById("marketSpreadsBody");

  if (!form || !itemKeySelect || !hoursSelect || !body) return;

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="6">No spread snapshots for the selected item/window.</td></tr>`;
      return;
    }
    body.innerHTML = items
      .map((row) => {
        return `
          <tr>
            <td>${escapeHtml(formatTime(row.ts))}</td>
            <td>${escapeHtml(row.best_bin_fmt || formatCoins(row.best_bin))}</td>
            <td>${escapeHtml(row.second_best_bin_fmt || formatCoins(row.second_best_bin))}</td>
            <td>${escapeHtml(row.spread_abs_fmt || formatCoins(row.spread_abs))}</td>
            <td>${formatPct(row.spread_pct)}</td>
            <td>${Number(row.active_listings || 0)}</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    const params = new URLSearchParams();
    const selectedItem = itemKeySelect.value;
    const selectedHours = hoursSelect.value || "6";
    if (selectedItem) params.set("item_key", selectedItem);
    params.set("hours", selectedHours);

    try {
      const data = await fetchJson(`/api/spreads?${params.toString()}`);
      const keys = data.available_item_keys || [];

      if (keys.length > 0) {
        const current = data.item_key || selectedItem || keys[0];
        itemKeySelect.innerHTML = keys
          .map((key) => {
            const selected = key === current ? "selected" : "";
            return `<option value="${escapeHtml(key)}" ${selected}>${escapeHtml(key)}</option>`;
          })
          .join("");
      } else {
        itemKeySelect.innerHTML = '<option value="">No item keys yet</option>';
      }

      renderRows(data.items || []);
      meta.textContent = `Item: ${data.item_key || "-"} • Rows: ${data.count || 0} • Window: ${data.hours || selectedHours}h`;
    } catch (err) {
      body.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading spreads: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    load();
  });
  itemKeySelect.addEventListener("change", () => load());
  hoursSelect.addEventListener("change", () => load());

  load();
  setInterval(load, 30_000);
}

function initNocomPage() {
  const meta = document.getElementById("nocomMeta");
  const form = document.getElementById("nocomForm");
  const horizonSelect = document.getElementById("nocomHorizon");
  const sortSelect = document.getElementById("nocomSort");
  const body = document.getElementById("nocomBody");

  if (!meta || !form || !horizonSelect || !sortSelect || !body) return;

  const applyHorizonOptions = (status) => {
    const horizons = (status?.nocom_horizons || [30, 120, 360])
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (!horizons.length) return;

    const current = Number(horizonSelect.value || 0);
    const defaultHorizon = Number(status?.nocom_default_horizon || 120);
    const selected = horizons.includes(current)
      ? current
      : (horizons.includes(defaultHorizon) ? defaultHorizon : horizons[0]);

    horizonSelect.innerHTML = horizons
      .map((h) => `<option value="${h}" ${h === selected ? "selected" : ""}>${h}m</option>`)
      .join("");
  };

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="13">No forecasts available for this horizon yet.</td></tr>`;
      return;
    }

    body.innerHTML = items
      .map((row, idx) => {
        const detailId = `nocom-inputs-${idx}`;
        const inputsPretty = escapeHtml(JSON.stringify(row.inputs || {}, null, 2));
        const graphUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        return `
          <tr>
            <td class="cell-mono"><a href="${graphUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${escapeHtml(row.price_now_fmt || formatCoins(row.price_now))}</td>
            <td>${escapeHtml(row.regime_tag || "-")}</td>
            <td>${Number(row.liquidity_rank || 0).toFixed(1)}</td>
            <td class="${Number(row.pred_return || 0) >= 0 ? "text-good" : "text-bad"}">${formatPct(row.pred_return)}</td>
            <td>${formatPct(row.pred_p10)}</td>
            <td>${formatPct(row.pred_p50)}</td>
            <td>${formatPct(row.pred_p90)}</td>
            <td>${formatPct(row.p_up)}</td>
            <td>${formatPct(row.pred_vol)}</td>
            <td>${Number(row.confidence || 0).toFixed(3)}</td>
            <td><div data-sparkline-item-key="${escapeHtml(row.item_key)}"></div></td>
            <td><button type="button" class="btn secondary" data-toggle-inputs="${detailId}">View</button></td>
          </tr>
          <tr id="${detailId}" data-input-row="1" style="display:none;">
            <td colspan="13">
              <pre class="cell-mono" style="margin:0; white-space:pre-wrap;">${inputsPretty}</pre>
            </td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    const horizon = Number(horizonSelect.value || 120);
    const sortBy = String(sortSelect.value || "pred_return");
    try {
      let sparklinesEnabled = false;
      let sparklinesWindow = "6h";
      let sparklinesTopN = 50;
      try {
        const status = await fetchJson("/api/status");
        sparklinesEnabled = Boolean(status.sparklines_enabled);
        sparklinesWindow = String(status.sparklines_window || "6h");
        sparklinesTopN = Number(status.sparklines_top_n || 50);
      } catch (_err) {
        sparklinesEnabled = false;
      }

      const data = await fetchJson(`/api/nocom/forecasts?horizon=${horizon}&limit=300`);
      const items = (data.items || []).slice();
      if (sortBy === "confidence") {
        items.sort(
          (a, b) => Number(b.confidence || 0) - Number(a.confidence || 0)
        );
      } else {
        items.sort(
          (a, b) => Number(b.pred_return || 0) - Number(a.pred_return || 0)
        );
      }
      renderRows(items);
      meta.textContent = `Items: ${items.length} • Horizon: ${horizon}m • Sort: ${sortBy} • Updated ${new Date().toLocaleTimeString()}`;
      if (sparklinesEnabled) {
        const keys = items
          .map((row) => row.item_key)
          .filter(Boolean)
          .slice(0, Math.max(1, sparklinesTopN));
        if (keys.length > 0) {
          const params = new URLSearchParams();
          params.set("window", sparklinesWindow);
          params.set("limit", String(Math.max(1, sparklinesTopN)));
          params.set("item_keys", keys.join(","));
          const sparkData = await fetchJson(`/api/sparklines?${params.toString()}`);
          applySparklinesFromPayload(body, sparkData.items || {});
        }
      }
    } catch (err) {
      body.innerHTML = `<tr><td colspan="13">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading nocom forecasts: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    load();
  });
  horizonSelect.addEventListener("change", () => load());
  sortSelect.addEventListener("change", () => load());

  body.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-toggle-inputs]");
    if (!btn) return;
    const targetId = btn.getAttribute("data-toggle-inputs");
    const row = document.getElementById(targetId);
    if (!row) return;
    const isHidden = row.style.display === "none" || !row.style.display;
    row.style.display = isHidden ? "table-row" : "none";
    btn.textContent = isHidden ? "Hide" : "View";
  });

  (async () => {
    try {
      const status = await fetchJson("/api/status");
      applyHorizonOptions(status);
    } catch (_err) {
      // Keep defaults if status endpoint is unavailable.
    }
    load();
  })();

  setInterval(load, 30_000);
}

function initRankedPage() {
  const meta = document.getElementById("rankedMeta");
  const form = document.getElementById("rankedForm");
  const horizonSelect = document.getElementById("rankedHorizon");
  const minConfInput = document.getElementById("rankedMinConf");
  const body = document.getElementById("rankedBody");

  if (!meta || !form || !horizonSelect || !minConfInput || !body) return;

  const applyHorizonOptions = (status) => {
    const horizons = (status?.nocom_horizons || [30, 120, 360])
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (!horizons.length) return;

    const current = Number(horizonSelect.value || 0);
    const defaultHorizon = Number(status?.nocom_default_horizon || 120);
    const selected = horizons.includes(current)
      ? current
      : (horizons.includes(defaultHorizon) ? defaultHorizon : horizons[0]);

    horizonSelect.innerHTML = horizons
      .map((h) => `<option value="${h}" ${h === selected ? "selected" : ""}>${h}m</option>`)
      .join("");
  };

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="17">No ranked forecasts matched the filters.</td></tr>`;
      return;
    }

    body.innerHTML = items
      .map((row, idx) => {
        const topClass = idx < 10 ? "top-rank" : "";
        const graphUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        return `
          <tr class="${topClass}">
            <td>${idx + 1}</td>
            <td class="cell-mono"><a href="${graphUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${formatCoins(row.price_now)}</td>
            <td>${escapeHtml(row.regime_tag || "-")}</td>
            <td>${Number(row.liquidity_rank || 0).toFixed(1)}</td>
            <td class="${Number(row.pred_return || 0) >= 0 ? "text-good" : "text-bad"}">${formatPct(row.pred_return)}</td>
            <td>${formatPct(row.pred_p10)}</td>
            <td>${formatPct(row.pred_p50)}</td>
            <td>${formatPct(row.pred_p90)}</td>
            <td>${formatPct(row.pred_vol)}</td>
            <td>${formatPct(row.p_up)}</td>
            <td>${Number(row.confidence || 0).toFixed(3)}</td>
            <td>${formatPct(row.ev)}</td>
            <td>${formatPct(row.ev_risk_adj)}</td>
            <td>${formatPct(row.ev_after_fee)}</td>
            <td class="${Number(row.ev_per_hour || 0) >= 0 ? "text-good" : "text-bad"}">${formatPct(row.ev_per_hour)}</td>
            <td><div data-sparkline-item-key="${escapeHtml(row.item_key)}"></div></td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    const horizon = Number(horizonSelect.value || 120);
    const minConf = Number(minConfInput.value || 0.35);
    try {
      let sparklinesEnabled = false;
      let sparklinesWindow = "6h";
      let sparklinesTopN = 50;
      try {
        const status = await fetchJson("/api/status");
        sparklinesEnabled = Boolean(status.sparklines_enabled);
        sparklinesWindow = String(status.sparklines_window || "6h");
        sparklinesTopN = Number(status.sparklines_top_n || 50);
      } catch (_err) {
        sparklinesEnabled = false;
      }

      const data = await fetchJson(
        `/api/nocom/ranked?horizon=${horizon}&limit=200&min_conf=${encodeURIComponent(String(minConf))}`
      );
      const items = data.items || [];
      renderRows(items);
      meta.textContent = `Items: ${items.length} • Horizon: ${horizon}m • Min conf: ${Number(minConf).toFixed(2)} • Updated ${new Date().toLocaleTimeString()}`;

      if (sparklinesEnabled) {
        const keys = items
          .map((row) => row.item_key)
          .filter(Boolean)
          .slice(0, Math.max(1, sparklinesTopN));
        if (keys.length > 0) {
          const params = new URLSearchParams();
          params.set("window", sparklinesWindow);
          params.set("limit", String(Math.max(1, sparklinesTopN)));
          params.set("item_keys", keys.join(","));
          const sparkData = await fetchJson(`/api/sparklines?${params.toString()}`);
          applySparklinesFromPayload(body, sparkData.items || {});
        }
      }
    } catch (err) {
      body.innerHTML = `<tr><td colspan="17">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading ranked EV/hr: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    load();
  });
  horizonSelect.addEventListener("change", () => load());
  minConfInput.addEventListener("change", () => load());

  (async () => {
    try {
      const status = await fetchJson("/api/status");
      applyHorizonOptions(status);
      if (Number.isFinite(Number(status?.nocom_min_conf))) {
        minConfInput.value = Number(status.nocom_min_conf).toFixed(2);
      }
    } catch (_err) {
      // Keep defaults if status endpoint is unavailable.
    }
    load();
  })();

  setInterval(load, 30_000);
}

function initExtremesPage() {
  const meta = document.getElementById("extremesMeta");
  const form = document.getElementById("extremesForm");
  const sortSelect = document.getElementById("extremesSort");
  const orderSelect = document.getElementById("extremesOrder");
  const limitInput = document.getElementById("extremesLimit");
  const body = document.getElementById("extremesBody");

  if (!meta || !form || !sortSelect || !orderSelect || !limitInput || !body) return;

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="13">No item extremes available yet.</td></tr>`;
      return;
    }

    body.innerHTML = items
      .map((row) => {
        const itemUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        return `
          <tr>
            <td class="cell-mono"><a href="${itemUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${formatPct(row.pct_off_ath)}</td>
            <td>${formatPct(row.pct_above_atl)}</td>
            <td>${escapeHtml(row.ath_price_fmt || formatCoins(row.ath_price))}</td>
            <td>${escapeHtml(row.atl_price_fmt || formatCoins(row.atl_price))}</td>
            <td>${escapeHtml(row.high_24h_fmt || formatCoins(row.high_24h))} / ${escapeHtml(row.low_24h_fmt || formatCoins(row.low_24h))}</td>
            <td>${escapeHtml(row.high_7d_fmt || formatCoins(row.high_7d))} / ${escapeHtml(row.low_7d_fmt || formatCoins(row.low_7d))}</td>
            <td>${escapeHtml(row.high_30d_fmt || formatCoins(row.high_30d))} / ${escapeHtml(row.low_30d_fmt || formatCoins(row.low_30d))}</td>
            <td>${formatPct(row.mdd_24h)}</td>
            <td>${formatPct(row.mdd_7d)}</td>
            <td>${formatPct(row.mdd_30d)}</td>
            <td>${formatPct(row.mdd_all)}</td>
            <td>${escapeHtml(formatTime(row.ts))}</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    const sort = String(sortSelect.value || "pct_off_ath");
    const order = String(orderSelect.value || "desc");
    const limit = Math.max(Number(limitInput.value || 200), 1);
    const params = new URLSearchParams();
    params.set("sort", sort);
    params.set("order", order);
    params.set("limit", String(limit));

    try {
      const data = await fetchJson(`/api/extremes?${params.toString()}`);
      const items = data.items || [];
      renderRows(items);
      meta.textContent = `Rows: ${items.length} • sort ${data.sort || sort} ${data.order || order} • updated ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      body.innerHTML = `<tr><td colspan="13">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading extremes: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    load();
  });
  sortSelect.addEventListener("change", () => load());
  orderSelect.addEventListener("change", () => load());

  load();
  setInterval(load, 30_000);
}

function initEventsPage() {
  const meta = document.getElementById("eventsMeta");
  const form = document.getElementById("eventsForm");
  const typeSelect = document.getElementById("eventsType");
  const limitInput = document.getElementById("eventsLimit");
  const body = document.getElementById("eventsBody");

  if (!meta || !form || !typeSelect || !limitInput || !body) return;

  const detailsText = (details) => {
    if (!details || typeof details !== "object") return "-";
    try {
      const compact = JSON.stringify(details);
      if (!compact) return "-";
      return compact.length > 240 ? `${compact.slice(0, 237)}...` : compact;
    } catch (_err) {
      return "-";
    }
  };

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="6">No events for current filter.</td></tr>`;
      return;
    }
    body.innerHTML = items
      .map((row) => {
        const itemUrl = `/item/${encodeURIComponent(row.item_key)}`;
        return `
          <tr>
            <td>${escapeHtml(formatTime(row.ts))}</td>
            <td>${escapeHtml(row.event_type || "-")}</td>
            <td class="cell-mono"><a href="${itemUrl}">${escapeHtml(row.item_key || "-")}</a></td>
            <td>${Number(row.severity || 0)}</td>
            <td>${escapeHtml(row.title || "-")}</td>
            <td class="cell-mono">${escapeHtml(detailsText(row.details))}</td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async () => {
    const eventType = String(typeSelect.value || "").trim();
    const limit = Math.max(Number(limitInput.value || 200), 1);
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (eventType) params.set("type", eventType);
    try {
      const data = await fetchJson(`/api/events?${params.toString()}`);
      const items = data.items || [];
      renderRows(items);
      meta.textContent = `Rows: ${items.length} • type ${eventType || "ALL"} • updated ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      body.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading events: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    load();
  });
  typeSelect.addEventListener("change", () => load());

  load();
  setInterval(load, 30_000);
}

function initGraphsPage() {
  const meta = document.getElementById("graphsMeta");
  const form = document.getElementById("graphsForm");
  const queryInput = document.getElementById("graphsQuery");
  const body = document.getElementById("graphsBody");

  if (!meta || !form || !queryInput || !body) return;

  let lastQuery = "";
  let debounceHandle = null;

  const renderRows = (items) => {
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="7">No canonical items found.</td></tr>`;
      return;
    }
    body.innerHTML = items
      .map((row) => {
        const graphUrl = `/graph/${encodeURIComponent(row.item_key)}`;
        const liqText = (row.liquidity_rank === null || row.liquidity_rank === undefined)
          ? "-"
          : Number(row.liquidity_rank).toFixed(1);
        return `
          <tr>
            <td class="cell-mono"><a href="${graphUrl}">${escapeHtml(row.item_key)}</a></td>
            <td>${escapeHtml(row.display_name || "-")}</td>
            <td>${escapeHtml(row.tier || "-")}</td>
            <td>${escapeHtml(liqText)}</td>
            <td>${escapeHtml(row.price_now_fmt || formatCoins(row.price_now))}</td>
            <td>${escapeHtml(row.regime_tag || "-")}</td>
            <td><a class="btn secondary" href="${graphUrl}">Open</a></td>
          </tr>
        `;
      })
      .join("");
  };

  const load = async (query = "") => {
    const params = new URLSearchParams();
    params.set("limit", "200");
    if (query) params.set("q", query);
    try {
      const data = await fetchJson(`/api/items/search?${params.toString()}`);
      const items = data.items || [];
      renderRows(items);
      meta.textContent = `Rows: ${items.length} • ${query ? `query "${query}"` : "top liquid canonical instruments"} • updated ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      body.innerHTML = `<tr><td colspan="7">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading graph items: ${err.message}`;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const q = String(queryInput.value || "").trim();
    lastQuery = q;
    load(q);
  });

  queryInput.addEventListener("input", () => {
    const q = String(queryInput.value || "").trim();
    if (q === lastQuery) return;
    clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      lastQuery = q;
      load(q);
    }, 250);
  });

  load("");
}

function initGraphItemPage() {
  const itemKey = (document.body.dataset.itemKey || "").trim();
  const defaultHorizon = Number(document.body.dataset.defaultHorizon || 120);
  const meta = document.getElementById("graphMeta");
  const watchToggleBtn = document.getElementById("graphWatchToggle");
  const errorEl = document.getElementById("graphError");
  const plotEl = document.getElementById("graphPlot");
  const hoverMeta = document.getElementById("graphHoverMeta");
  const windowButtons = document.getElementById("graphWindowButtons");
  const horizonSelect = document.getElementById("graphHorizon");
  const eventsBody = document.getElementById("graphEventsBody");

  const showMinMaxEl = document.getElementById("graphShowMinMax");
  const showEmaEl = document.getElementById("graphShowEma");
  const showMedian24El = document.getElementById("graphShowMedian24");
  const showForecastEl = document.getElementById("graphShowForecast");
  const showEventsEl = document.getElementById("graphShowEvents");

  if (
    !itemKey || !meta || !watchToggleBtn || !errorEl || !plotEl || !hoverMeta || !windowButtons
    || !horizonSelect || !eventsBody || !showMinMaxEl || !showEmaEl
    || !showMedian24El || !showForecastEl || !showEventsEl
  ) {
    return;
  }

  if (typeof window.uPlot === "undefined") {
    showError(errorEl, "uPlot failed to load.");
    return;
  }

  const state = {
    window: "24h",
    horizon: Number.isFinite(defaultHorizon) && defaultHorizon > 0 ? defaultHorizon : 120,
    bars: [],
    extremes: null,
    events: [],
    forecast: null,
    plot: null,
    watching: false,
  };
  horizonSelect.value = String(state.horizon);

  const setWatchButton = (watching, pending = false) => {
    state.watching = Boolean(watching);
    watchToggleBtn.classList.toggle("watching", state.watching);
    watchToggleBtn.textContent = state.watching ? "★ Watching" : "☆ Watch";
    watchToggleBtn.disabled = Boolean(pending);
  };

  const refreshWatchState = async () => {
    try {
      const data = await fetchJson("/api/favorites?limit=5003");
      const keys = new Set((data.items || []).map((row) => row.item_key));
      setWatchButton(keys.has(itemKey));
    } catch (_err) {
      setWatchButton(state.watching);
    }
  };

  const median = (values) => {
    if (!values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
    return sorted[mid];
  };

  const emaSeries = (values, span) => {
    if (!values.length) return [];
    const alpha = 2 / (Math.max(Number(span), 1) + 1);
    const out = [Number(values[0])];
    for (let i = 1; i < values.length; i += 1) {
      out.push(alpha * Number(values[i]) + (1 - alpha) * out[i - 1]);
    }
    return out;
  };

  const nearestIndex = (xVals, ts) => {
    const target = Number(ts);
    if (!xVals.length || !Number.isFinite(target)) return -1;
    let lo = 0;
    let hi = xVals.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (Number(xVals[mid]) < target) lo = mid + 1;
      else hi = mid;
    }
    const idx = lo;
    if (idx <= 0) return 0;
    const prev = idx - 1;
    return (Math.abs(Number(xVals[idx]) - target) < Math.abs(Number(xVals[prev]) - target)) ? idx : prev;
  };

  const renderEventsTable = () => {
    const rows = state.events || [];
    if (!rows.length) {
      eventsBody.innerHTML = `<tr><td colspan="4">No events for this item yet.</td></tr>`;
      return;
    }
    eventsBody.innerHTML = rows
      .slice(0, 200)
      .map((row) => {
        return `
          <tr>
            <td>${escapeHtml(formatTime(row.ts))}</td>
            <td>${escapeHtml(row.event_type || "-")}</td>
            <td>${Number(row.severity || 0)}</td>
            <td>${escapeHtml(row.title || "-")}</td>
          </tr>
        `;
      })
      .join("");
  };

  const renderChart = () => {
    const bars = state.bars || [];
    if (!bars.length) {
      if (state.plot) {
        state.plot.destroy();
        state.plot = null;
      }
      plotEl.innerHTML = '<div class="help">No bar history available for this timeframe.</div>';
      hoverMeta.textContent = "No points to inspect.";
      return;
    }

    plotEl.innerHTML = "";
    const x = bars.map((b) => Number(b.bar_ts));
    const medianLine = bars.map((b) => Number(b.price_median || 0));
    const minLine = bars.map((b) => Number(b.price_min || 0));
    const maxLine = bars.map((b) => Number(b.price_max || 0));
    const samples = bars.map((b) => Number(b.n || 0));

    const series = [{ label: "Time" }];
    const data = [x];

    series.push({ label: "Median", stroke: "#7ac7ab", width: 2 });
    data.push(medianLine);

    if (showMinMaxEl.checked) {
      series.push({ label: "Min", stroke: "#3e7767", width: 1 });
      data.push(minLine);
      series.push({ label: "Max", stroke: "#3e7767", width: 1 });
      data.push(maxLine);
    }

    if (showEmaEl.checked) {
      series.push({ label: "EMA10", stroke: "#f4c16b", width: 1.5 });
      data.push(emaSeries(medianLine, 10));
      series.push({ label: "EMA60", stroke: "#e38a5c", width: 1.5 });
      data.push(emaSeries(medianLine, 60));
    }

    if (showMedian24El.checked) {
      const lastTs = Number(x[x.length - 1]);
      const dayCutoff = lastTs - (24 * 3600);
      const med24 = median(
        medianLine.filter((_v, idx) => Number(x[idx]) >= dayCutoff)
      );
      if (med24 !== null) {
        series.push({ label: "Median24h", stroke: "#9ec9ff", width: 1, dash: [6, 4] });
        data.push(new Array(x.length).fill(med24));
      }
    }

    if (showForecastEl.checked && state.forecast) {
      const lastPrice = Number(medianLine[medianLine.length - 1] || 0);
      const p10 = lastPrice * (1 + Number(state.forecast.pred_p10 || 0));
      const p50 = lastPrice * (1 + Number(state.forecast.pred_p50 || 0));
      const p90 = lastPrice * (1 + Number(state.forecast.pred_p90 || 0));
      series.push({ label: "F-P10", stroke: "#6f8ebf", width: 1, dash: [4, 4] });
      data.push(new Array(x.length).fill(p10));
      series.push({ label: "F-P50", stroke: "#7a93d8", width: 1.5 });
      data.push(new Array(x.length).fill(p50));
      series.push({ label: "F-P90", stroke: "#9bb1ef", width: 1, dash: [4, 4] });
      data.push(new Array(x.length).fill(p90));
    }

    if (showEventsEl.checked && (state.events || []).length > 0) {
      const eventPoints = new Array(x.length).fill(null);
      (state.events || []).forEach((evt) => {
        const idx = nearestIndex(x, evt.ts);
        if (idx >= 0 && idx < eventPoints.length) {
          eventPoints[idx] = medianLine[idx];
        }
      });
      series.push({
        label: "Events",
        stroke: "#f45f6e",
        width: 0,
        points: { show: true, size: 5, fill: "#f45f6e", stroke: "#f45f6e" },
      });
      data.push(eventPoints);
    }

    const hoverPlugin = {
      hooks: {
        setCursor: [
          (u) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= x.length) return;
            hoverMeta.textContent = `${formatTime(x[idx])} | median ${formatCoins(medianLine[idx])} | min ${formatCoins(minLine[idx])} | max ${formatCoins(maxLine[idx])} | n ${samples[idx]}`;
          },
        ],
      },
    };

    const width = Math.max(Math.floor(plotEl.clientWidth || 900), 320);
    const height = 380;
    const opts = {
      width,
      height,
      series,
      plugins: [hoverPlugin],
      axes: [
        {
          stroke: "#6c968a",
          grid: { stroke: "#1d3a34" },
          values: (_u, vals) => vals.map((v) => new Date(Number(v) * 1000).toLocaleString()),
        },
        {
          stroke: "#6c968a",
          grid: { stroke: "#1d3a34" },
          values: (_u, vals) => vals.map((v) => formatCoins(v)),
        },
      ],
    };

    if (state.plot) {
      state.plot.destroy();
      state.plot = null;
    }
    state.plot = new window.uPlot(opts, data, plotEl);
  };

  const fetchMaybe = async (url) => {
    try {
      return await fetchJson(url);
    } catch (_err) {
      return null;
    }
  };

  const loadAll = async () => {
    const limits = { "1h": 300, "6h": 800, "24h": 2000, "7d": 4000, "30d": 6000, all: 8000 };
    const window = String(state.window || "24h");
    const limit = Number(limits[window] || 2000);
    showError(errorEl, "");
    try {
      const encoded = encodeURIComponent(itemKey);
      const [history, extremes, events, forecastResp] = await Promise.all([
        fetchJson(`/api/items/${encoded}/history?window=${encodeURIComponent(window)}&limit=${limit}`),
        fetchMaybe(`/api/items/${encoded}/extremes`),
        fetchMaybe(`/api/items/${encoded}/events?limit=200`),
        fetchMaybe(`/api/nocom/forecasts?horizon=${encodeURIComponent(String(state.horizon))}&item_key=${encoded}&limit=1`),
      ]);

      state.bars = history?.bars || [];
      state.extremes = extremes || null;
      state.events = events?.items || [];
      state.forecast = (forecastResp?.items && forecastResp.items.length > 0) ? forecastResp.items[0] : null;

      renderEventsTable();
      renderChart();
      meta.textContent = `Bars: ${state.bars.length} • Window: ${window} • Horizon: ${state.horizon}m • Updated ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      showError(errorEl, err.message);
      meta.textContent = `Error loading chart: ${err.message}`;
      if (state.plot) {
        state.plot.destroy();
        state.plot = null;
      }
      plotEl.innerHTML = "";
      eventsBody.innerHTML = `<tr><td colspan="4">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  windowButtons.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-window]");
    if (!btn) return;
    const nextWindow = String(btn.getAttribute("data-window") || "").trim();
    if (!nextWindow || nextWindow === state.window) return;
    state.window = nextWindow;
    windowButtons.querySelectorAll("[data-window]").forEach((node) => {
      node.classList.toggle("active", node === btn);
    });
    loadAll();
  });

  horizonSelect.addEventListener("change", () => {
    state.horizon = Number(horizonSelect.value || state.horizon || 120);
    loadAll();
  });

  [showMinMaxEl, showEmaEl, showMedian24El, showForecastEl, showEventsEl].forEach((el) => {
    el.addEventListener("change", () => renderChart());
  });

  watchToggleBtn.addEventListener("click", async () => {
    setWatchButton(state.watching, true);
    try {
      if (!state.watching) {
        await fetchJson("/api/favorites", {
          method: "POST",
          body: JSON.stringify({ item_key: itemKey }),
        });
        setWatchButton(true);
        showToast("Added to watchlist");
      } else {
        await fetchJson(`/api/favorites/${encodeURIComponent(itemKey)}`, {
          method: "DELETE",
        });
        setWatchButton(false);
        showToast("Removed from watchlist");
      }
    } catch (err) {
      setWatchButton(state.watching);
      showError(errorEl, err.message);
    } finally {
      watchToggleBtn.disabled = false;
    }
  });

  window.addEventListener("resize", () => {
    renderChart();
  });

  refreshWatchState();
  loadAll();
}

function initWatchlistPage() {
  const meta = document.getElementById("watchlistMeta");
  const body = document.getElementById("watchlistBody");
  const itemForm = document.getElementById("watchlistItemForm");
  const itemQuery = document.getElementById("watchlistItemQuery");
  const itemSelect = document.getElementById("watchlistItemKey");
  const groupForm = document.getElementById("watchlistGroupForm");
  const groupLabel = document.getElementById("watchlistGroupLabel");
  const groupKind = document.getElementById("watchlistGroupKind");
  const groupValue = document.getElementById("watchlistGroupValue");

  if (
    !meta || !body || !itemForm || !itemQuery || !itemSelect
    || !groupForm || !groupLabel || !groupKind || !groupValue
  ) {
    return;
  }

  let searchDebounce = null;

  const renderRows = (targets) => {
    if (!targets.length) {
      body.innerHTML = `<tr><td colspan="14">No watch targets yet. Add an item or group above.</td></tr>`;
      return;
    }

    body.innerHTML = targets
      .map((row) => {
        const type = String(row.type || "ITEM").toUpperCase();
        const sampleKey = String(row.sample_item_key || row.item_key || "").trim();
        const sampleUrl = sampleKey ? `/graph/${encodeURIComponent(sampleKey)}` : "";
        const targetText = type === "ITEM"
          ? String(row.item_key || "-")
          : `${String(row.group_kind || "-")}:${String(row.group_value || "-")}`;
        const actionToggleLabel = Number(row.enabled || 0) === 1 ? "Disable" : "Enable";
        const actionToggleNext = Number(row.enabled || 0) === 1 ? 0 : 1;
        const sparkNode = sampleKey
          ? `<div data-sparkline-item-key="${escapeHtml(sampleKey)}"></div>`
          : "-";

        return `
          <tr>
            <td>${escapeHtml(type)}</td>
            <td>${escapeHtml(row.label || "-")}</td>
            <td class="cell-mono">${escapeHtml(targetText)}</td>
            <td>${Number(row.enabled || 0) === 1 ? "ON" : "OFF"}</td>
            <td>${Number(row.matched_count || 0)}</td>
            <td>${sampleUrl ? `<a class="cell-mono" href="${sampleUrl}">${escapeHtml(sampleKey)}</a>` : "-"}</td>
            <td>${escapeHtml(formatTime(row.created_ts))}</td>
            <td>${escapeHtml(row.price_now_fmt || formatCoins(row.price_now))}</td>
            <td>${row.liquidity_rank === null || row.liquidity_rank === undefined ? "-" : Number(row.liquidity_rank).toFixed(1)}</td>
            <td>${escapeHtml(row.regime_tag || "-")}</td>
            <td>${formatPct(row.pct_off_ath)}</td>
            <td>${formatPct(row.mdd_30d)}</td>
            <td>${sparkNode}</td>
            <td>
              <div class="actions-stack">
                <button class="btn secondary" type="button" data-watch-target-toggle="${Number(row.id)}" data-watch-target-next="${actionToggleNext}">${escapeHtml(actionToggleLabel)}</button>
                <button class="btn secondary" type="button" data-watch-target-remove="${Number(row.id)}">Remove</button>
                ${type === "GROUP" ? `<button class="btn secondary" type="button" data-watch-target-expand="${Number(row.id)}">Expand</button>` : ""}
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  };

  const applySparklines = async (targets) => {
    try {
      const status = await fetchJson("/api/status");
      if (!status.sparklines_enabled) return;
      const keys = targets
        .map((row) => String(row.sample_item_key || row.item_key || "").trim())
        .filter(Boolean)
        .slice(0, Math.max(Number(status.sparklines_top_n || 50), 1));
      if (!keys.length) return;
      const params = new URLSearchParams();
      params.set("window", String(status.sparklines_window || "6h"));
      params.set("limit", String(Math.max(Number(status.sparklines_top_n || 50), 1)));
      params.set("item_keys", [...new Set(keys)].join(","));
      const sparkData = await fetchJson(`/api/sparklines?${params.toString()}`);
      applySparklinesFromPayload(body, sparkData.items || {});
    } catch (_err) {
      // Keep watchlist usable even if sparklines fail.
    }
  };

  const loadTargets = async () => {
    try {
      const data = await fetchJson("/api/watchlist?limit=500");
      const targets = data.targets || data.items || [];
      renderRows(targets);
      meta.textContent = `Targets: ${targets.length} • updated ${new Date().toLocaleTimeString()}`;
      applySparklines(targets);
    } catch (err) {
      body.innerHTML = `<tr><td colspan="14">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading watchlist: ${err.message}`;
    }
  };

  const loadItemSearch = async (query) => {
    const q = String(query || "").trim();
    if (!q) {
      itemSelect.innerHTML = '<option value="">Search to load items</option>';
      return;
    }
    try {
      const data = await fetchJson(`/api/items/search?q=${encodeURIComponent(q)}&limit=50`);
      const items = data.items || [];
      if (!items.length) {
        itemSelect.innerHTML = '<option value="">No matches</option>';
        return;
      }
      itemSelect.innerHTML = items
        .map((row) => {
          const label = `${row.display_name || row.item_key} • ${row.item_key}`;
          return `<option value="${escapeHtml(row.item_key)}">${escapeHtml(label)}</option>`;
        })
        .join("");
    } catch (err) {
      itemSelect.innerHTML = `<option value="">Search failed: ${escapeHtml(err.message)}</option>`;
    }
  };

  itemQuery.addEventListener("input", () => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => {
      loadItemSearch(itemQuery.value || "");
    }, 250);
  });

  itemForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const itemKey = String(itemSelect.value || "").trim();
    if (!itemKey) {
      meta.textContent = "Pick an item_key from search results.";
      return;
    }
    const btn = itemForm.querySelector("#watchlistAddItemBtn");
    if (btn) btn.disabled = true;
    try {
      await fetchJson("/api/watchlist/item", {
        method: "POST",
        body: JSON.stringify({ item_key: itemKey }),
      });
      showToast("Watch item added");
      loadTargets();
    } catch (err) {
      meta.textContent = `Error adding item target: ${err.message}`;
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  groupForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const label = String(groupLabel.value || "").trim();
    const kind = String(groupKind.value || "").trim().toUpperCase();
    const value = String(groupValue.value || "").trim();
    if (!kind || !value) {
      meta.textContent = "Group kind and value are required.";
      return;
    }
    const btn = groupForm.querySelector("#watchlistAddGroupBtn");
    if (btn) btn.disabled = true;
    try {
      await fetchJson("/api/watchlist/group", {
        method: "POST",
        body: JSON.stringify({
          label: label || `${kind}:${value}`,
          group_kind: kind,
          group_value: value,
        }),
      });
      showToast("Watch group added");
      groupValue.value = "";
      if (!label) groupLabel.value = "";
      loadTargets();
    } catch (err) {
      meta.textContent = `Error adding group target: ${err.message}`;
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  body.addEventListener("click", async (event) => {
    const removeBtn = event.target.closest("[data-watch-target-remove]");
    if (removeBtn) {
      const id = Number(removeBtn.getAttribute("data-watch-target-remove") || 0);
      if (!Number.isFinite(id) || id <= 0) return;
      removeBtn.disabled = true;
      try {
        await fetchJson(`/api/watchlist/${id}`, { method: "DELETE" });
        showToast("Watch target removed");
        loadTargets();
      } catch (err) {
        removeBtn.disabled = false;
        meta.textContent = `Error removing target ${id}: ${err.message}`;
      }
      return;
    }

    const toggleBtn = event.target.closest("[data-watch-target-toggle]");
    if (toggleBtn) {
      const id = Number(toggleBtn.getAttribute("data-watch-target-toggle") || 0);
      const nextEnabled = Number(toggleBtn.getAttribute("data-watch-target-next") || 0);
      if (!Number.isFinite(id) || id <= 0) return;
      toggleBtn.disabled = true;
      try {
        await fetchJson(`/api/watchlist/${id}/toggle`, {
          method: "POST",
          body: JSON.stringify({ enabled: nextEnabled ? 1 : 0 }),
        });
        showToast(nextEnabled ? "Target enabled" : "Target disabled");
        loadTargets();
      } catch (err) {
        toggleBtn.disabled = false;
        meta.textContent = `Error toggling target ${id}: ${err.message}`;
      }
      return;
    }

    const expandBtn = event.target.closest("[data-watch-target-expand]");
    if (expandBtn) {
      const id = Number(expandBtn.getAttribute("data-watch-target-expand") || 0);
      if (!Number.isFinite(id) || id <= 0) return;
      expandBtn.disabled = true;
      try {
        const data = await fetchJson(`/api/watchlist/expand/${id}?limit=200`);
        const keys = data.item_keys || [];
        const preview = keys.slice(0, 8).join(", ");
        showToast(`Matches: ${keys.length}${preview ? ` • ${preview}` : ""}`);
      } catch (err) {
        meta.textContent = `Error expanding target ${id}: ${err.message}`;
      } finally {
        expandBtn.disabled = false;
      }
    }
  });

  loadTargets();
  setInterval(loadTargets, 30_000);
}

function initItemPage() {
  const itemKey = (document.body.dataset.itemKey || "").trim();
  const meta = document.getElementById("itemMeta");
  const athEl = document.getElementById("itemAth");
  const atlEl = document.getElementById("itemAtl");
  const pctOffAthEl = document.getElementById("itemPctOffAth");
  const pctAboveAtlEl = document.getElementById("itemPctAboveAtl");
  const highLow24hEl = document.getElementById("itemHighLow24h");
  const highLow7dEl = document.getElementById("itemHighLow7d");
  const highLow30dEl = document.getElementById("itemHighLow30d");
  const drawdownMeta = document.getElementById("itemDrawdownMeta");
  const drawdownBody = document.getElementById("itemDrawdownBody");
  const storyMeta = document.getElementById("itemStoryMeta");
  const storyBody = document.getElementById("itemStoryBody");
  const historyForm = document.getElementById("itemHistoryForm");
  const historyWindow = document.getElementById("itemHistoryWindow");
  const historyLimit = document.getElementById("itemHistoryLimit");
  const historyBody = document.getElementById("itemHistoryBody");

  if (
    !itemKey || !meta || !athEl || !atlEl || !pctOffAthEl || !pctAboveAtlEl || !highLow24hEl
    || !highLow7dEl || !highLow30dEl || !drawdownMeta || !drawdownBody
    || !storyMeta || !storyBody
    || !historyForm || !historyWindow || !historyLimit || !historyBody
  ) {
    return;
  }

  const encodedKey = encodeURIComponent(itemKey);

  const renderExtremes = (ext) => {
    athEl.textContent = `${ext.ath_price_fmt || formatCoins(ext.ath_price)} • ${formatAgo(ext.ath_ts)} (${ext.seconds_since_ath_human || "-"})`;
    atlEl.textContent = `${ext.atl_price_fmt || formatCoins(ext.atl_price)} • ${formatAgo(ext.atl_ts)} (${ext.seconds_since_atl_human || "-"})`;
    pctOffAthEl.textContent = formatPct(ext.pct_off_ath);
    pctAboveAtlEl.textContent = formatPct(ext.pct_above_atl);
    highLow24hEl.textContent = `${ext.high_24h_fmt || formatCoins(ext.high_24h)} / ${ext.low_24h_fmt || formatCoins(ext.low_24h)}`;
    highLow7dEl.textContent = `${ext.high_7d_fmt || formatCoins(ext.high_7d)} / ${ext.low_7d_fmt || formatCoins(ext.low_7d)}`;
    highLow30dEl.textContent = `${ext.high_30d_fmt || formatCoins(ext.high_30d)} / ${ext.low_30d_fmt || formatCoins(ext.low_30d)}`;
    drawdownMeta.textContent = `Bars seen: ${ext.bars_seen || 0} • last update ${formatTime(ext.ts)}`;
    drawdownBody.innerHTML = `
      <tr>
        <td>${formatPct(ext.mdd_all)}</td>
        <td>${formatPct(ext.mdd_30d)}</td>
        <td>${formatPct(ext.mdd_7d)}</td>
        <td>${formatPct(ext.mdd_24h)}</td>
        <td>${escapeHtml(formatTime(ext.mdd_peak_ts))}</td>
        <td>${escapeHtml(formatTime(ext.mdd_trough_ts))}</td>
      </tr>
    `;
  };

  const loadExtremes = async () => {
    try {
      const data = await fetchJson(`/api/items/${encodedKey}/extremes`);
      renderExtremes(data);
      meta.textContent = `Loaded extremes for ${itemKey}`;
    } catch (err) {
      meta.textContent = `Error loading extremes: ${err.message}`;
      drawdownMeta.textContent = `Error loading drawdowns: ${err.message}`;
      drawdownBody.innerHTML = `<tr><td colspan="6">${escapeHtml(err.message)}</td></tr>`;
    }
  };

  const loadEvents = async () => {
    try {
      const data = await fetchJson(`/api/items/${encodedKey}/events?limit=200`);
      const rows = data.items || [];
      if (!rows.length) {
        storyBody.innerHTML = `<tr><td colspan="5">No events for this item yet.</td></tr>`;
      } else {
        storyBody.innerHTML = rows
          .map((row) => {
            let details = "-";
            if (row.details && typeof row.details === "object") {
              try {
                details = JSON.stringify(row.details);
              } catch (_err) {
                details = "-";
              }
            }
            return `
              <tr>
                <td>${escapeHtml(formatTime(row.ts))}</td>
                <td>${escapeHtml(row.event_type || "-")}</td>
                <td>${Number(row.severity || 0)}</td>
                <td>${escapeHtml(row.title || "-")}</td>
                <td class="cell-mono">${escapeHtml(details)}</td>
              </tr>
            `;
          })
          .join("");
      }
      storyMeta.textContent = `Events: ${rows.length}`;
    } catch (err) {
      storyBody.innerHTML = `<tr><td colspan="5">${escapeHtml(err.message)}</td></tr>`;
      storyMeta.textContent = `Error loading events: ${err.message}`;
    }
  };

  const loadHistory = async () => {
    const window = String(historyWindow.value || "24h");
    const limit = Math.max(Number(historyLimit.value || 2000), 1);
    try {
      const data = await fetchJson(
        `/api/items/${encodedKey}/history?window=${encodeURIComponent(window)}&limit=${encodeURIComponent(String(limit))}`
      );
      const bars = data.bars || [];
      if (!bars.length) {
        historyBody.innerHTML = `<tr><td colspan="5">No bars for this item/window.</td></tr>`;
      } else {
        historyBody.innerHTML = bars
          .map((bar) => {
            return `
              <tr>
                <td>${escapeHtml(formatTime(bar.bar_ts))}</td>
                <td>${formatCoins(bar.price_median)}</td>
                <td>${formatCoins(bar.price_min)}</td>
                <td>${formatCoins(bar.price_max)}</td>
                <td>${Number(bar.n || 0)}</td>
              </tr>
            `;
          })
          .join("");
      }
      meta.textContent = `Loaded ${bars.length} bars (${window}) for ${itemKey}`;
    } catch (err) {
      historyBody.innerHTML = `<tr><td colspan="5">${escapeHtml(err.message)}</td></tr>`;
      meta.textContent = `Error loading history: ${err.message}`;
    }
  };

  historyForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadHistory();
  });
  historyWindow.addEventListener("change", () => loadHistory());

  loadExtremes();
  loadEvents();
  loadHistory();
  setInterval(loadEvents, 30_000);
}

function initAuctionPage() {
  const copyBtn = document.getElementById("copyViewAuction");
  if (!copyBtn) return;

  copyBtn.addEventListener("click", async () => {
    const cmd = copyBtn.dataset.command || "";
    try {
      await navigator.clipboard.writeText(cmd);
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = "Copy /viewauction";
      }, 1300);
    } catch (err) {
      copyBtn.textContent = "Copy failed";
    }
  });
}

function initNavState() {
  const navByPage = {
    index: "ah",
    signals: "signals",
    nocom: "signals",
    ranked: "signals",
    graphs: "explore",
    "graph-item": "explore",
    extremes: "explore",
    market: "explore",
    events: "explore",
    alerts: "explore",
    shark: "explore",
    item: "explore",
    auction: "explore",
    portfolio: "trading",
    backtests: "trading",
    watchlist: "watchlist",
  };
  const activeGroup = navByPage[page] || null;
  if (!activeGroup) return;
  document.querySelectorAll(`[data-nav="${activeGroup}"]`).forEach((el) => {
    el.classList.add("active");
  });
}

initNavState();

if (page === "index") initIndexPage();
if (page === "signals") initSignalsPage();
if (page === "nocom") initNocomPage();
if (page === "ranked") initRankedPage();
if (page === "extremes") initExtremesPage();
if (page === "events") initEventsPage();
if (page === "graphs") initGraphsPage();
if (page === "graph-item") initGraphItemPage();
if (page === "watchlist") initWatchlistPage();
if (page === "item") initItemPage();
if (page === "portfolio") initPortfolioPage();
if (page === "backtests") initBacktestsPage();
if (page === "market") initMarketPage();
if (page === "alerts") initAlertsPage();
if (page === "shark") initSharkPage();
if (page === "auction") initAuctionPage();
