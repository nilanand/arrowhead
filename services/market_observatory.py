from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import defaultdict

from sqlalchemy import desc, select

from .db import (
    MarketBar,
    MarketFeature,
    MarketRegime,
    Symbol,
    insert_market_events,
    insert_market_regime,
    upsert_market_bars,
    upsert_market_features,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-9
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "1d": 86_400,
}


def align_bar_ts(ts: int, timeframe_seconds: int) -> int:
    step = max(int(timeframe_seconds), 1)
    return int(ts - (ts % step))


def _ema(values: list[float]) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (len(values) + 1.0)
    ema_val = float(values[0])
    for value in values[1:]:
        ema_val = alpha * float(value) + (1.0 - alpha) * ema_val
    return float(ema_val)


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    x_sum = float((n - 1) * n / 2)
    xx_sum = float((n - 1) * n * (2 * n - 1) / 6)
    y_sum = float(sum(values))
    xy_sum = float(sum(i * v for i, v in enumerate(values)))
    denom = n * xx_sum - x_sum * x_sum
    if abs(denom) <= EPSILON:
        return 0.0
    return float((n * xy_sum - x_sum * y_sum) / denom)


def _percentile_value(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clamped = min(max(float(q), 0.0), 1.0)
    idx = clamped * (len(values) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return float(values[lower])
    frac = idx - lower
    return float(values[lower] + (values[upper] - values[lower]) * frac)


def _percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    lo = 0
    hi = len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return float(100.0 * lo / len(sorted_vals))


def _rsi(closes: list[float]) -> float:
    if len(closes) <= 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for prev, cur in zip(closes, closes[1:]):
        delta = float(cur) - float(prev)
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    if losses <= EPSILON:
        return 100.0
    rs = gains / max(losses, EPSILON)
    return float(100.0 - (100.0 / (1.0 + rs)))


def _atr(highs: list[float], lows: list[float], closes: list[float]) -> float:
    if not highs or not lows or not closes:
        return 0.0
    trs: list[float] = []
    prev_close = closes[0]
    for hi, lo, close in zip(highs, lows, closes):
        hi_v = float(hi)
        lo_v = float(lo)
        close_v = float(close)
        tr = max(hi_v - lo_v, abs(hi_v - prev_close), abs(lo_v - prev_close))
        trs.append(tr)
        prev_close = close_v
    if not trs:
        return 0.0
    return float(sum(trs) / len(trs))


def _returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if prev <= 0 or cur <= 0:
            continue
        out.append(math.log(cur / prev))
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n <= 2:
        return 0.0
    x = xs[-n:]
    y = ys[-n:]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / max(n - 1, 1)
    x_var = sum((a - x_mean) ** 2 for a in x) / max(n - 1, 1)
    y_var = sum((b - y_mean) ** 2 for b in y) / max(n - 1, 1)
    denom = math.sqrt(max(x_var, 0.0) * max(y_var, 0.0))
    if denom <= EPSILON:
        return 0.0
    return float(cov / denom)


def _classify_symbol_regime(
    *,
    ema_fast: float,
    ema_slow: float,
    trend_slope: float,
    realized_vol: float,
    vol_threshold: float,
    volume_percentile: float,
) -> str:
    if volume_percentile >= 95.0:
        return "LIQUIDITY_SURGE"
    if realized_vol >= vol_threshold:
        return "VOLATILE"
    if ema_fast > ema_slow and trend_slope > 0:
        return "UPTREND"
    if ema_fast < ema_slow and trend_slope < 0:
        return "DOWNTREND"
    return "RANGE"


def compute_signal(feature: MarketFeature, *, mode: str) -> dict:
    trend = float(feature.trend_slope or 0.0)
    rsi = float(feature.rsi or 50.0)
    drawdown = float(feature.drawdown_30d or 0.0)
    vol = float(feature.realized_vol or 0.0)
    vol_pct = float(feature.volume_percentile or 0.0)

    trend_score = max(min(trend * 125.0, 2.5), -2.5)
    mean_reversion = max(min((50.0 - rsi) / 20.0, 2.0), -2.0)
    drawdown_penalty = max(min(drawdown * 6.0, 0.0), -3.0)
    volatility_penalty = -min(vol * 4.0, 2.5)
    volume_boost = min(vol_pct / 100.0, 1.2)
    score = trend_score + mean_reversion + drawdown_penalty + volatility_penalty + volume_boost

    if str(mode).strip().lower() != "full":
        return {
            "signal_score": round(score, 4),
            "signal_action": None,
        }

    if score >= 1.5:
        action = "BUY"
    elif score <= -1.5:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "signal_score": round(score, 4),
        "signal_action": action,
    }


def aggregate_higher_timeframes(
    session,
    *,
    symbols: list[str],
    closed_minute_ts: int,
    now_ts: int,
    intraday_lookback_days: int,
    daily_lookback_days: int,
) -> dict:
    symbols = [str(s).upper() for s in symbols if s]
    if not symbols:
        return {"rows_5m": 0, "rows_1d": 0}

    rows_to_upsert: list[dict] = []

    five_start = align_bar_ts(closed_minute_ts, TIMEFRAME_SECONDS["5m"])
    five_end = five_start + TIMEFRAME_SECONDS["5m"]
    five_rows = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1m")
            .where(MarketBar.symbol.in_(symbols))
            .where(MarketBar.bar_ts >= five_start)
            .where(MarketBar.bar_ts < five_end)
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    grouped_5m: dict[str, list[MarketBar]] = defaultdict(list)
    for row in five_rows:
        grouped_5m[str(row.symbol)].append(row)

    rows_5m = 0
    for symbol, bars in grouped_5m.items():
        bars_sorted = sorted(bars, key=lambda b: int(b.bar_ts))
        if not bars_sorted:
            continue
        rows_5m += 1
        rows_to_upsert.append(
            {
                "symbol": symbol,
                "timeframe": "5m",
                "bar_ts": int(five_start),
                "open": float(bars_sorted[0].open),
                "high": float(max(b.high for b in bars_sorted)),
                "low": float(min(b.low for b in bars_sorted)),
                "close": float(bars_sorted[-1].close),
                "volume": float(sum(float(b.volume or 0.0) for b in bars_sorted)),
            }
        )

    day_start = align_bar_ts(closed_minute_ts, TIMEFRAME_SECONDS["1d"])
    day_end = day_start + TIMEFRAME_SECONDS["1d"]
    day_rows = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1m")
            .where(MarketBar.symbol.in_(symbols))
            .where(MarketBar.bar_ts >= day_start)
            .where(MarketBar.bar_ts < day_end)
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    grouped_1d: dict[str, list[MarketBar]] = defaultdict(list)
    for row in day_rows:
        grouped_1d[str(row.symbol)].append(row)

    rows_1d = 0
    for symbol, bars in grouped_1d.items():
        bars_sorted = sorted(bars, key=lambda b: int(b.bar_ts))
        if not bars_sorted:
            continue
        rows_1d += 1
        rows_to_upsert.append(
            {
                "symbol": symbol,
                "timeframe": "1d",
                "bar_ts": int(day_start),
                "open": float(bars_sorted[0].open),
                "high": float(max(b.high for b in bars_sorted)),
                "low": float(min(b.low for b in bars_sorted)),
                "close": float(bars_sorted[-1].close),
                "volume": float(sum(float(b.volume or 0.0) for b in bars_sorted)),
            }
        )

    if rows_to_upsert:
        upsert_market_bars(
            session,
            rows_to_upsert,
            now_ts=now_ts,
            intraday_lookback_days=intraday_lookback_days,
            daily_lookback_days=daily_lookback_days,
        )

    return {
        "rows_5m": int(rows_5m),
        "rows_1d": int(rows_1d),
    }


def recompute_market_features(
    session,
    *,
    symbols: list[str],
    now_ts: int,
) -> dict:
    symbols = [str(s).upper() for s in symbols if s]
    if not symbols:
        return {"updated": 0, "by_timeframe": {"1m": 0, "5m": 0, "1d": 0}}

    by_timeframe_rows: dict[str, list[dict]] = {"1m": [], "5m": [], "1d": []}

    for timeframe, tf_seconds in TIMEFRAME_SECONDS.items():
        if timeframe in {"1m", "5m"}:
            cutoff = now_ts - (30 * 86_400)
        else:
            cutoff = now_ts - (2 * 365 * 86_400)

        bars = (
            session.execute(
                select(MarketBar)
                .where(MarketBar.symbol.in_(symbols))
                .where(MarketBar.timeframe == timeframe)
                .where(MarketBar.bar_ts >= cutoff)
                .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
            )
            .scalars()
            .all()
        )

        grouped: dict[str, list[MarketBar]] = defaultdict(list)
        for bar in bars:
            grouped[str(bar.symbol)].append(bar)

        provisional: list[dict] = []
        for symbol, symbol_bars in grouped.items():
            bars_sorted = sorted(symbol_bars, key=lambda row: int(row.bar_ts))
            if not bars_sorted:
                continue

            now_close = float(bars_sorted[-1].close)
            now_volume = float(bars_sorted[-1].volume or 0.0)

            fast_window_start = now_ts - (20 * tf_seconds)
            slow_window_start = now_ts - (100 * tf_seconds)
            rsi_window_start = now_ts - (14 * tf_seconds)
            atr_window_start = now_ts - (14 * tf_seconds)
            roll_30d_start = now_ts - (30 * 86_400)

            bars_fast = [row for row in bars_sorted if int(row.bar_ts) >= fast_window_start] or bars_sorted
            bars_slow = [row for row in bars_sorted if int(row.bar_ts) >= slow_window_start] or bars_sorted
            bars_rsi = [row for row in bars_sorted if int(row.bar_ts) >= rsi_window_start] or bars_sorted
            bars_atr = [row for row in bars_sorted if int(row.bar_ts) >= atr_window_start] or bars_sorted
            bars_30d = [row for row in bars_sorted if int(row.bar_ts) >= roll_30d_start] or bars_sorted

            closes_fast = [float(row.close) for row in bars_fast]
            closes_slow = [float(row.close) for row in bars_slow]
            closes_rsi = [float(row.close) for row in bars_rsi]
            highs_atr = [float(row.high) for row in bars_atr]
            lows_atr = [float(row.low) for row in bars_atr]
            closes_atr = [float(row.close) for row in bars_atr]
            closes_30d = [float(row.close) for row in bars_30d]
            volumes_30d = [float(row.volume or 0.0) for row in bars_30d]

            returns_30d = _returns(closes_30d)
            if timeframe == "1m":
                ann_factor = math.sqrt(252.0 * 390.0)
            elif timeframe == "5m":
                ann_factor = math.sqrt(252.0 * 78.0)
            else:
                ann_factor = math.sqrt(252.0)
            realized_vol = (
                float(statistics.pstdev(returns_30d)) * ann_factor if len(returns_30d) > 1 else 0.0
            )

            mean_30d = float(sum(closes_30d) / max(len(closes_30d), 1))
            std_30d = float(statistics.pstdev(closes_30d)) if len(closes_30d) > 1 else 0.0
            zscore_30d = (now_close - mean_30d) / max(std_30d, EPSILON)

            max_close_30d = float(max(closes_30d)) if closes_30d else now_close
            drawdown_30d = (now_close - max_close_30d) / max(max_close_30d, EPSILON)

            trend_raw = _linear_slope(closes_30d)
            trend_slope = trend_raw / max(abs(mean_30d), EPSILON)

            volume_percentile = _percentile_rank(volumes_30d, now_volume)

            provisional.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ts": int(now_ts),
                    "ema_fast": _ema(closes_fast),
                    "ema_slow": _ema(closes_slow),
                    "rsi": _rsi(closes_rsi),
                    "atr": _atr(highs_atr, lows_atr, closes_atr),
                    "realized_vol": realized_vol,
                    "zscore_30d": zscore_30d,
                    "drawdown_30d": drawdown_30d,
                    "trend_slope": trend_slope,
                    "volume_percentile": volume_percentile,
                    "regime_tag": "RANGE",
                }
            )

        vol_values = sorted([float(row["realized_vol"]) for row in provisional])
        vol_threshold = _percentile_value(vol_values, 0.8) if vol_values else 0.0

        finalized: list[dict] = []
        for row in provisional:
            regime_tag = _classify_symbol_regime(
                ema_fast=float(row["ema_fast"]),
                ema_slow=float(row["ema_slow"]),
                trend_slope=float(row["trend_slope"]),
                realized_vol=float(row["realized_vol"]),
                vol_threshold=float(vol_threshold),
                volume_percentile=float(row["volume_percentile"]),
            )
            row["regime_tag"] = regime_tag
            finalized.append(row)

        by_timeframe_rows[timeframe] = finalized

    rows_to_upsert = [
        row
        for timeframe in ("1m", "5m", "1d")
        for row in by_timeframe_rows.get(timeframe, [])
    ]
    updated = upsert_market_features(session, rows_to_upsert)

    return {
        "updated": int(updated),
        "by_timeframe": {
            "1m": len(by_timeframe_rows.get("1m", [])),
            "5m": len(by_timeframe_rows.get("5m", [])),
            "1d": len(by_timeframe_rows.get("1d", [])),
        },
    }


def recompute_market_regime(session, *, now_ts: int) -> dict:
    active_symbols = (
        session.execute(select(Symbol.symbol).where(Symbol.active == 1).limit(10_000)).scalars().all()
    )
    symbols = [str(s).upper() for s in active_symbols if s]

    if not symbols:
        row = {
            "ts": int(now_ts),
            "breadth_pct_above_ema": 0.0,
            "breadth_new_highs": 0,
            "breadth_new_lows": 0,
            "median_volatility": 0.0,
            "cross_asset_correlation": 0.0,
            "regime_tag": "RANGE",
        }
        insert_market_regime(session, row)
        return row

    daily_cutoff = now_ts - (70 * 86_400)
    bars = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1d")
            .where(MarketBar.symbol.in_(symbols))
            .where(MarketBar.bar_ts >= daily_cutoff)
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    grouped: dict[str, list[MarketBar]] = defaultdict(list)
    for bar in bars:
        grouped[str(bar.symbol)].append(bar)

    total = 0
    above_ema = 0
    new_highs = 0
    new_lows = 0
    for symbol, symbol_bars in grouped.items():
        bars_sorted = sorted(symbol_bars, key=lambda row: int(row.bar_ts))
        if not bars_sorted:
            continue

        latest = bars_sorted[-1]
        close_now = float(latest.close)

        ema_window_start = now_ts - (50 * 86_400)
        high_low_start = now_ts - (20 * 86_400)

        bars_ema = [row for row in bars_sorted if int(row.bar_ts) >= ema_window_start] or bars_sorted
        closes_ema = [float(row.close) for row in bars_ema]
        ema_50 = _ema(closes_ema)

        bars_20d = [row for row in bars_sorted if int(row.bar_ts) >= high_low_start] or bars_sorted
        high_20d = max(float(row.high) for row in bars_20d)
        low_20d = min(float(row.low) for row in bars_20d)

        total += 1
        if close_now > ema_50:
            above_ema += 1
        if close_now >= high_20d:
            new_highs += 1
        if close_now <= low_20d:
            new_lows += 1

    breadth_pct = (100.0 * above_ema / max(total, 1)) if total else 0.0

    vol_features = (
        session.execute(
            select(MarketFeature.realized_vol)
            .where(MarketFeature.timeframe == "1d")
            .where(MarketFeature.realized_vol.is_not(None))
        )
        .scalars()
        .all()
    )
    vol_values = [float(v) for v in vol_features if v is not None]
    vol_percentiles = [_percentile_rank(vol_values, value) for value in vol_values] if vol_values else [0.0]
    median_volatility = float(statistics.median(vol_percentiles)) if vol_percentiles else 0.0

    benchmark_symbols = ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "USO", "XLK", "XLF", "XLE", "XLV"]
    benchmark_rows = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1d")
            .where(MarketBar.symbol.in_(benchmark_symbols))
            .where(MarketBar.bar_ts >= now_ts - (90 * 86_400))
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    by_benchmark: dict[str, list[float]] = defaultdict(list)
    for row in benchmark_rows:
        by_benchmark[str(row.symbol)].append(float(row.close))

    ret_series: dict[str, list[float]] = {}
    for symbol, closes in by_benchmark.items():
        returns = _returns(closes)
        if len(returns) >= 5:
            ret_series[symbol] = returns[-20:]

    corrs: list[float] = []
    series_keys = list(ret_series.keys())
    for i in range(len(series_keys)):
        for j in range(i + 1, len(series_keys)):
            left = ret_series[series_keys[i]]
            right = ret_series[series_keys[j]]
            corrs.append(_corr(left, right))

    cross_asset_corr = float(sum(corrs) / len(corrs)) if corrs else 0.0

    score = 0
    if breadth_pct >= 55.0:
        score += 1
    elif breadth_pct <= 45.0:
        score -= 1

    if new_highs > int(new_lows * 1.2):
        score += 1
    elif new_lows > int(new_highs * 1.2):
        score -= 1

    if median_volatility <= 45.0:
        score += 1
    elif median_volatility >= 70.0:
        score -= 1

    if cross_asset_corr <= 0.45:
        score += 1
    elif cross_asset_corr >= 0.70:
        score -= 1

    if score >= 2:
        regime_tag = "RISK_ON"
    elif score <= -2:
        regime_tag = "RISK_OFF"
    elif breadth_pct >= 55.0:
        regime_tag = "UPTREND"
    elif breadth_pct <= 45.0:
        regime_tag = "DOWNTREND"
    elif median_volatility >= 70.0:
        regime_tag = "VOLATILE"
    else:
        regime_tag = "RANGE"

    row = {
        "ts": int(now_ts),
        "breadth_pct_above_ema": float(round(breadth_pct, 4)),
        "breadth_new_highs": int(new_highs),
        "breadth_new_lows": int(new_lows),
        "median_volatility": float(round(median_volatility, 4)),
        "cross_asset_correlation": float(round(cross_asset_corr, 4)),
        "regime_tag": regime_tag,
    }
    insert_market_regime(session, row)
    return row


def _build_market_event(
    *,
    ts: int,
    symbol: str | None,
    event_type: str,
    severity: int,
    details: dict,
    dedupe_key: str,
) -> dict:
    return {
        "ts": int(ts),
        "symbol": (str(symbol).upper() if symbol else None),
        "event_type": str(event_type),
        "severity": max(min(int(severity), 5), 1),
        "details_json": json.dumps(details, separators=(",", ":")),
        "dedupe_key": str(dedupe_key),
    }


def detect_market_events(
    session,
    *,
    now_ts: int,
    symbols_hint: list[str] | None = None,
) -> dict:
    regime_rows = (
        session.execute(select(MarketRegime).order_by(desc(MarketRegime.ts)).limit(2)).scalars().all()
    )
    latest_regime = regime_rows[0] if regime_rows else None
    prev_regime = regime_rows[1] if len(regime_rows) > 1 else None

    out_rows: list[dict] = []

    if latest_regime and prev_regime:
        if str(latest_regime.regime_tag) != str(prev_regime.regime_tag):
            out_rows.append(
                _build_market_event(
                    ts=now_ts,
                    symbol=None,
                    event_type="REGIME_FLIP",
                    severity=5,
                    details={
                        "from": str(prev_regime.regime_tag),
                        "to": str(latest_regime.regime_tag),
                        "breadth_pct": float(latest_regime.breadth_pct_above_ema or 0.0),
                    },
                    dedupe_key=(
                        f"REGIME_FLIP|{int(latest_regime.ts)}|"
                        f"{str(prev_regime.regime_tag)}|{str(latest_regime.regime_tag)}"
                    ),
                )
            )

        breadth_delta = abs(
            float(latest_regime.breadth_pct_above_ema or 0.0)
            - float(prev_regime.breadth_pct_above_ema or 0.0)
        )
        if breadth_delta >= 20.0:
            out_rows.append(
                _build_market_event(
                    ts=now_ts,
                    symbol=None,
                    event_type="BREADTH_SHOCK",
                    severity=min(5, max(2, int(breadth_delta / 10.0))),
                    details={
                        "breadth_from": float(prev_regime.breadth_pct_above_ema or 0.0),
                        "breadth_to": float(latest_regime.breadth_pct_above_ema or 0.0),
                        "swing": float(round(breadth_delta, 4)),
                    },
                    dedupe_key=f"BREADTH_SHOCK|{int(latest_regime.ts)}|{int(breadth_delta)}",
                )
            )

    features_1m = (
        session.execute(
            select(MarketFeature)
            .where(MarketFeature.timeframe == "1m")
            .where(MarketFeature.ts >= now_ts - 900)
            .order_by(MarketFeature.volume_percentile.desc())
            .limit(2000)
        )
        .scalars()
        .all()
    )
    features_1d = (
        session.execute(
            select(MarketFeature)
            .where(MarketFeature.timeframe == "1d")
            .where(MarketFeature.ts >= now_ts - (2 * 86_400))
            .order_by(MarketFeature.volume_percentile.desc())
            .limit(2000)
        )
        .scalars()
        .all()
    )

    vol_values = [float(row.realized_vol or 0.0) for row in features_1m]
    vol_threshold_98 = _percentile_value(sorted(vol_values), 0.98) if vol_values else 0.0

    watch_symbols = {str(row.symbol).upper() for row in features_1m[:600]}
    if symbols_hint:
        for symbol in symbols_hint:
            watch_symbols.add(str(symbol).upper())

    drawdown_triggered: set[str] = set()
    for row in features_1m:
        symbol = str(row.symbol).upper()

        if float(row.realized_vol or 0.0) >= vol_threshold_98 and vol_threshold_98 > 0:
            out_rows.append(
                _build_market_event(
                    ts=now_ts,
                    symbol=symbol,
                    event_type="VOLATILITY_SHOCK",
                    severity=4,
                    details={
                        "realized_vol": float(row.realized_vol or 0.0),
                        "threshold": float(vol_threshold_98),
                        "timeframe": "1m",
                    },
                    dedupe_key=f"VOLATILITY_SHOCK|{symbol}|{int(row.ts)}",
                )
            )

        if float(row.drawdown_30d or 0.0) <= -0.15:
            drawdown_triggered.add(symbol)
            out_rows.append(
                _build_market_event(
                    ts=now_ts,
                    symbol=symbol,
                    event_type="EXTREME_DRAWDOWN",
                    severity=5,
                    details={
                        "drawdown_30d": float(row.drawdown_30d or 0.0),
                        "threshold": -0.15,
                        "timeframe": "1m",
                    },
                    dedupe_key=f"EXTREME_DRAWDOWN|{symbol}|{int(row.ts)}",
                )
            )

    by_symbol_daily_feature = {str(row.symbol).upper(): row for row in features_1d}
    break_candidates = [
        symbol
        for symbol, row in by_symbol_daily_feature.items()
        if float(row.volume_percentile or 0.0) >= 95.0
    ]

    if break_candidates:
        daily_bars = (
            session.execute(
                select(MarketBar)
                .where(MarketBar.timeframe == "1d")
                .where(MarketBar.symbol.in_(break_candidates))
                .where(MarketBar.bar_ts >= now_ts - (45 * 86_400))
                .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
            )
            .scalars()
            .all()
        )
        grouped_daily: dict[str, list[MarketBar]] = defaultdict(list)
        for row in daily_bars:
            grouped_daily[str(row.symbol).upper()].append(row)

        for symbol, bars in grouped_daily.items():
            bars_sorted = sorted(bars, key=lambda row: int(row.bar_ts))
            if len(bars_sorted) < 2:
                continue
            latest = bars_sorted[-1]
            window = [row for row in bars_sorted if int(row.bar_ts) >= now_ts - (30 * 86_400)]
            if len(window) < 2:
                continue
            prev_window = window[:-1]
            high_ref = max(float(row.high) for row in prev_window)
            low_ref = min(float(row.low) for row in prev_window)
            close_now = float(latest.close)

            if close_now >= high_ref:
                out_rows.append(
                    _build_market_event(
                        ts=now_ts,
                        symbol=symbol,
                        event_type="HIGH_LOW_BREAK_30D",
                        severity=4,
                        details={
                            "direction": "HIGH_BREAK",
                            "close": close_now,
                            "ref_high": high_ref,
                            "volume_percentile": float(
                                by_symbol_daily_feature[symbol].volume_percentile or 0.0
                            ),
                        },
                        dedupe_key=f"HIGH_LOW_BREAK_30D|{symbol}|HIGH|{int(latest.bar_ts)}",
                    )
                )
            elif close_now <= low_ref:
                out_rows.append(
                    _build_market_event(
                        ts=now_ts,
                        symbol=symbol,
                        event_type="HIGH_LOW_BREAK_30D",
                        severity=4,
                        details={
                            "direction": "LOW_BREAK",
                            "close": close_now,
                            "ref_low": low_ref,
                            "volume_percentile": float(
                                by_symbol_daily_feature[symbol].volume_percentile or 0.0
                            ),
                        },
                        dedupe_key=f"HIGH_LOW_BREAK_30D|{symbol}|LOW|{int(latest.bar_ts)}",
                    )
                )

    corr_symbols = list(watch_symbols)[:200]
    if "SPY" not in corr_symbols:
        corr_symbols.append("SPY")
    if "QQQ" not in corr_symbols:
        corr_symbols.append("QQQ")

    corr_bars = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1d")
            .where(MarketBar.symbol.in_(corr_symbols))
            .where(MarketBar.bar_ts >= now_ts - (90 * 86_400))
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )
    corr_grouped: dict[str, list[float]] = defaultdict(list)
    for row in corr_bars:
        corr_grouped[str(row.symbol).upper()].append(float(row.close))

    spy_returns = _returns(corr_grouped.get("SPY", []))
    qqq_returns = _returns(corr_grouped.get("QQQ", []))

    for symbol in corr_symbols:
        if symbol in {"SPY", "QQQ"}:
            continue
        symbol_returns = _returns(corr_grouped.get(symbol, []))
        if len(symbol_returns) < 25:
            continue

        benchmark = spy_returns if len(spy_returns) >= len(qqq_returns) else qqq_returns
        if len(benchmark) < 25:
            continue

        corr_long = _corr(symbol_returns[-60:], benchmark[-60:])
        corr_short = _corr(symbol_returns[-20:], benchmark[-20:])
        if abs(corr_short - corr_long) >= 0.60 and abs(corr_short) <= 0.2:
            out_rows.append(
                _build_market_event(
                    ts=now_ts,
                    symbol=symbol,
                    event_type="CORRELATION_BREAK",
                    severity=4,
                    details={
                        "corr_short": float(round(corr_short, 4)),
                        "corr_long": float(round(corr_long, 4)),
                        "delta": float(round(corr_short - corr_long, 4)),
                    },
                    dedupe_key=f"CORRELATION_BREAK|{symbol}|{now_ts // 300}",
                )
            )

    inserted = insert_market_events(session, out_rows)

    by_type: dict[str, int] = defaultdict(int)
    for row in out_rows:
        by_type[str(row["event_type"])] += 1

    return {
        "inserted": int(inserted),
        "generated": len(out_rows),
        "types": dict(by_type),
    }


def bootstrap_symbol_history(
    provider,
    session,
    *,
    symbol: str,
    now_ts: int,
    intraday_lookback_days: int,
    daily_lookback_days: int,
) -> dict:
    symbol = str(symbol or "").upper().strip()
    if not symbol:
        return {"symbol": symbol, "rows": 0}

    rows_1m = provider.get_historical_bars(
        symbol,
        "1m",
        lookback=max(int(intraday_lookback_days), 1) * 86_400,
    )
    rows_1d = provider.get_historical_bars(
        symbol,
        "1d",
        lookback=max(int(daily_lookback_days), 1) * 86_400,
    )

    merged: list[dict] = rows_1m + rows_1d
    written = upsert_market_bars(
        session,
        merged,
        now_ts=now_ts,
        intraday_lookback_days=intraday_lookback_days,
        daily_lookback_days=daily_lookback_days,
    )

    if rows_1m:
        minute_bars = sorted(rows_1m, key=lambda row: int(row["bar_ts"]))
        touched_minutes = {int(row["bar_ts"]) for row in minute_bars[-500:]}
        for minute_ts in sorted(touched_minutes):
            aggregate_higher_timeframes(
                session,
                symbols=[symbol],
                closed_minute_ts=int(minute_ts),
                now_ts=now_ts,
                intraday_lookback_days=intraday_lookback_days,
                daily_lookback_days=daily_lookback_days,
            )

    return {
        "symbol": symbol,
        "rows": int(written),
        "rows_1m": len(rows_1m),
        "rows_1d": len(rows_1d),
    }
