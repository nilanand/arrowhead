from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass

from sqlalchemy import select

from .db import ItemBar, ItemFeature, save_backtest_results
from .features import EPSILON

logger = logging.getLogger(__name__)


def _ema(values: list[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (max(span, 1) + 1.0)
    ema_val = float(values[0])
    for value in values[1:]:
        ema_val = alpha * float(value) + (1.0 - alpha) * ema_val
    return float(ema_val)


def _slope(values: list[float]) -> float:
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
    return (n * xy_sum - x_sum * y_sum) / denom


def _volatility(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    returns: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if prev <= 0 or cur <= 0:
            continue
        returns.append(math.log(cur / prev))

    if len(returns) <= 1:
        return 0.0
    return float(statistics.pstdev(returns))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    q = min(max(q, 0.0), 1.0)
    idx = (len(sorted_values) - 1) * q
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return float(sorted_values[lower])

    frac = idx - lower
    low = float(sorted_values[lower])
    high = float(sorted_values[upper])
    return low + (high - low) * frac


@dataclass
class BacktestConfig:
    lookback_hours: int = 24
    ah_fee_pct: float = 0.01
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.08
    max_vol_60m: float = 0.06
    min_samples_1h: int = 20
    min_samples_24h: int = 60
    max_items: int = 20
    liq_rank_min: float = 60.0


def _simulate_item(bars: list[ItemBar], cfg: BacktestConfig) -> tuple[int, float, float, float]:
    if len(bars) < 20:
        return 0, 0.0, 0.0, 0.0

    position_entry: float | None = None
    trades: list[float] = []

    equity = 1.0
    peak = 1.0
    min_dd = 0.0

    for idx, bar in enumerate(bars):
        ts = int(bar.bar_ts)
        price_now = float(bar.price_median)

        hist = bars[: idx + 1]
        prices = [float(x.price_median) for x in hist]

        bars_60m = [x for x in hist if x.bar_ts >= ts - 3600]
        bars_1h = [x for x in hist if x.bar_ts >= ts - 3600]
        bars_24h = [x for x in hist if x.bar_ts >= ts - 24 * 3600]
        bars_fast_ema = [x for x in hist if x.bar_ts >= ts - 120 * 60]
        bars_slow_ema = [x for x in hist if x.bar_ts >= ts - 180 * 60]

        samples_1h = sum(int(x.n) for x in bars_1h)
        samples_24h = sum(int(x.n) for x in bars_24h)
        liquidity_ok = samples_1h >= cfg.min_samples_1h and samples_24h >= cfg.min_samples_24h

        prices_24h = [float(x.price_median) for x in bars_24h] or prices
        prices_24h_sorted = sorted(prices_24h)
        p25 = _percentile(prices_24h_sorted, 0.25)
        median = _percentile(prices_24h_sorted, 0.5)
        p75 = _percentile(prices_24h_sorted, 0.75)
        iqr = p75 - p25
        zscore = (price_now - median) / max(iqr, EPSILON)

        prices_60 = [float(x.price_median) for x in bars_60m] or [price_now]
        prices_fast_ema = [float(x.price_median) for x in bars_fast_ema] or prices
        prices_slow_ema = [float(x.price_median) for x in bars_slow_ema] or prices
        ema_fast = _ema(prices_fast_ema, 10)
        ema_slow = _ema(prices_slow_ema, 60)
        trend_slope = _slope(prices_60)
        vol_60m = _volatility(prices_60)

        undervalued = zscore <= -1.0 or price_now <= (median - 1.5 * iqr)
        trend_ok = ema_fast > ema_slow or trend_slope > 0
        vol_ok = vol_60m <= cfg.max_vol_60m

        if position_entry is None:
            if liquidity_ok and undervalued and trend_ok and vol_ok:
                position_entry = price_now
            continue

        stop_loss = price_now <= position_entry * (1 - cfg.stop_loss_pct)
        take_profit = price_now >= position_entry * (1 + cfg.take_profit_pct)
        mean_revert = zscore >= 0.5
        reversal = ema_fast < ema_slow
        should_sell = stop_loss or take_profit or mean_revert or reversal

        if should_sell:
            ret = ((price_now * (1 - cfg.ah_fee_pct)) - position_entry) / max(position_entry, EPSILON)
            trades.append(ret)

            equity *= 1 + ret
            peak = max(peak, equity)
            drawdown = (equity - peak) / peak
            min_dd = min(min_dd, drawdown)
            position_entry = None

    if position_entry is not None and bars:
        last_price = float(bars[-1].price_median)
        ret = ((last_price * (1 - cfg.ah_fee_pct)) - position_entry) / max(position_entry, EPSILON)
        trades.append(ret)

        equity *= 1 + ret
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        min_dd = min(min_dd, drawdown)

    trade_count = len(trades)
    if trade_count == 0:
        return 0, 0.0, 0.0, abs(min_dd)

    wins = sum(1 for r in trades if r > 0)
    return (
        trade_count,
        wins / trade_count,
        sum(trades) / trade_count,
        abs(min_dd),
    )


def run_backtests(session, *, now_ts: int, cfg: BacktestConfig) -> dict:
    liquid_keys = (
        session.execute(
            select(ItemFeature.item_key)
            .where(ItemFeature.liquidity_ok == 1)
            .where(ItemFeature.liquidity_rank >= cfg.liq_rank_min)
            .order_by(ItemFeature.samples_24h.desc())
            .limit(max(cfg.max_items, 1))
        )
        .scalars()
        .all()
    )

    cutoff = now_ts - (max(cfg.lookback_hours, 1) * 3600)
    rows_to_insert: list[dict] = []

    for item_key in liquid_keys:
        bars = (
            session.execute(
                select(ItemBar)
                .where(ItemBar.item_key == item_key)
                .where(ItemBar.bar_ts >= cutoff)
                .order_by(ItemBar.bar_ts.asc())
            )
            .scalars()
            .all()
        )

        trades, win_rate, avg_return, max_dd = _simulate_item(bars, cfg)
        rows_to_insert.append(
            {
                "ts": now_ts,
                "item_key": item_key,
                "lookback_hours": cfg.lookback_hours,
                "trades": trades,
                "win_rate": float(win_rate),
                "avg_return": float(avg_return),
                "max_drawdown": float(max_dd),
            }
        )

    inserted = save_backtest_results(session, rows_to_insert, now_ts=now_ts)
    logger.debug("Backtest summaries updated count=%s", inserted)

    return {
        "updated": inserted,
        "items": len(liquid_keys),
        "ts": now_ts,
    }
