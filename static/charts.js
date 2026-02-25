(() => {
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function deepMerge(base, extra) {
    const out = {...base};
    Object.keys(extra || {}).forEach((key) => {
      const baseVal = out[key];
      const extraVal = extra[key];
      if (
        baseVal &&
        extraVal &&
        typeof baseVal === "object" &&
        typeof extraVal === "object" &&
        !Array.isArray(baseVal) &&
        !Array.isArray(extraVal)
      ) {
        out[key] = deepMerge(baseVal, extraVal);
      } else {
        out[key] = extraVal;
      }
    });
    return out;
  }

  const THEME = {
    GRID: cssVar("--grid") || "rgba(255,255,255,0.18)",
    GRID_STRONG: cssVar("--grid-strong") || "rgba(255,255,255,0.28)",
    MUTED: cssVar("--muted") || "rgba(255,255,255,0.70)",
    WHITE: cssVar("--text") || "#ffffff",
    GOOD: cssVar("--good") || "#22c55e",
    BAD: cssVar("--bad") || "#ef4444"
  };
  const FONT_MONO = cssVar("--font-mono") || "SF Mono, SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";

  function makeMonochromeScales(overrides = {}) {
    return deepMerge(
      {
        x: {
          ticks: {
            color: THEME.WHITE,
            font: {
              family: FONT_MONO,
              size: 12,
              weight: "400"
            }
          },
          grid: {color: THEME.GRID}
        },
        y: {
          ticks: {
            color: THEME.WHITE,
            font: {
              family: FONT_MONO,
              size: 12,
              weight: "400"
            }
          },
          grid: {color: THEME.GRID}
        }
      },
      overrides
    );
  }

  function commonOptions(overrides = {}) {
    return deepMerge(
      {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        normalized: true,
        // Keep parsing enabled for category-label + primitive value datasets.
        parsing: true,
        spanGaps: true,
        elements: {
          point: {
            radius: 0
          }
        },
        interaction: {mode: "index", intersect: false},
        plugins: {
          legend: {
            labels: {
              color: THEME.WHITE,
              font: {
                family: FONT_MONO,
                size: 12,
                weight: "400"
              }
            }
          }
        },
        scales: makeMonochromeScales()
      },
      overrides
    );
  }

  // Reuse this manager for all future Chart.js views to avoid lag/stretching from re-creation.
  const chartStore = (window.nocomCharts = window.nocomCharts || {});

  function toFiniteOrNull(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function downsampleEven(labels, seriesList, maxPoints) {
    const safeLabels = Array.isArray(labels) ? labels : [];
    const safeSeries = Array.isArray(seriesList) ? seriesList : [];
    const hardCap = 2000;
    const requested = Number(maxPoints);
    const cap = Math.max(1, Math.min(Number.isFinite(requested) ? requested : hardCap, hardCap));

    if (safeLabels.length <= cap) {
      return {
        labels: safeLabels.slice(),
        series: safeSeries.map((arr) => (Array.isArray(arr) ? arr.slice() : []))
      };
    }

    const step = Math.ceil(safeLabels.length / cap);
    const outLabels = [];
    const outSeries = safeSeries.map(() => []);

    for (let i = 0; i < safeLabels.length; i += step) {
      outLabels.push(safeLabels[i]);
      for (let s = 0; s < safeSeries.length; s += 1) {
        const src = Array.isArray(safeSeries[s]) ? safeSeries[s] : [];
        outSeries[s].push(src[i]);
      }
    }

    const lastIdx = safeLabels.length - 1;
    if (outLabels[outLabels.length - 1] !== safeLabels[lastIdx]) {
      if (outLabels.length >= cap) {
        outLabels[outLabels.length - 1] = safeLabels[lastIdx];
        for (let s = 0; s < safeSeries.length; s += 1) {
          const src = Array.isArray(safeSeries[s]) ? safeSeries[s] : [];
          outSeries[s][outSeries[s].length - 1] = src[lastIdx];
        }
      } else {
        outLabels.push(safeLabels[lastIdx]);
        for (let s = 0; s < safeSeries.length; s += 1) {
          const src = Array.isArray(safeSeries[s]) ? safeSeries[s] : [];
          outSeries[s].push(src[lastIdx]);
        }
      }
    }

    return {
      labels: outLabels,
      series: outSeries
    };
  }

  function getOrCreateChart(canvasId, configFactory) {
    const key = String(canvasId || "").trim();
    if (!key || typeof Chart === "undefined") {
      return null;
    }

    const canvas = document.getElementById(key);
    if (!canvas) {
      return null;
    }

    const existing = chartStore[key];
    if (existing && existing.canvas === canvas) {
      return existing;
    }
    if (existing && typeof existing.destroy === "function") {
      existing.destroy();
      delete chartStore[key];
    }

    const config = typeof configFactory === "function" ? configFactory() : configFactory;
    if (!config) {
      return null;
    }

    const chart = new Chart(canvas, config);
    chartStore[key] = chart;
    console.debug(`[nocomCharts] init chart ${key}`);
    return chart;
  }

  function updateChart(chart, labels, datasetArrays) {
    if (!chart) {
      return;
    }
    chart.data.labels = Array.isArray(labels) ? labels.slice() : [];
    const datasets = chart.data.datasets || [];
    for (let i = 0; i < datasets.length; i += 1) {
      const values = Array.isArray(datasetArrays) && Array.isArray(datasetArrays[i]) ? datasetArrays[i] : [];
      datasets[i].data = values.slice();
    }
    chart.update("none");
  }

  window.ArrowheadCharts = {
    cssVar,
    THEME,
    deepMerge,
    makeMonochromeScales,
    commonOptions,
    toFiniteOrNull,
    downsampleEven,
    getOrCreateChart,
    updateChart
  };
})();
