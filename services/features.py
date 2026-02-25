from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import select

from .db import MarketBar, upsert_market_features


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return int(value)
    except Exception:
        return None


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(index=values.index, dtype="float64")

    x = np.arange(window, dtype=float)
    x_mean = float(x.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0:
        return pd.Series(index=values.index, dtype="float64")

    def _slope(raw: np.ndarray) -> float:
        if np.isnan(raw).any():
            return np.nan
        y = raw.astype(float)
        y_mean = float(y.mean())
        num = float(((x - x_mean) * (y - y_mean)).sum())
        return num / denom

    return values.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _rolling_percentile_rank(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(index=values.index, dtype="float64")

    def _pct(raw: np.ndarray) -> float:
        if np.isnan(raw).any():
            return np.nan
        last = float(raw[-1])
        count = float((raw <= last).sum())
        return count / float(len(raw))

    return values.rolling(window=window, min_periods=window).apply(_pct, raw=True)


def _state_from_trend(close: pd.Series, ema_50: pd.Series, trend_slope_20d: pd.Series) -> pd.Series:
    state = pd.Series(index=close.index, dtype="object")
    state[:] = "CHOP"
    uptrend_mask = (close > ema_50) & (trend_slope_20d > 0)
    downtrend_mask = (close < ema_50) & (trend_slope_20d < 0)
    state = state.mask(uptrend_mask, "UPTREND")
    state = state.mask(downtrend_mask, "DOWNTREND")
    return state


def compute_latest_continuation_forecast(
    features_df: pd.DataFrame,
    *,
    lookback_days: int = 504,
    horizon_days: int = 5,
    min_samples: int = 60,
) -> dict:
    if features_df is None or features_df.empty:
        return {
            "cont_prob_5d": None,
            "exp_return_5d": None,
            "cont_samples_5d": None,
            "mom_strength": None,
        }

    df = features_df.sort_index()
    if "state_tag" not in df.columns or "close" not in df.columns:
        return {
            "cont_prob_5d": None,
            "exp_return_5d": None,
            "cont_samples_5d": None,
            "mom_strength": None,
        }

    current_state = str(df["state_tag"].iloc[-1] or "").upper()
    if current_state not in {"UPTREND", "DOWNTREND", "CHOP"}:
        current_state = "CHOP"

    close = pd.to_numeric(df["close"], errors="coerce").where(lambda s: s > 0)
    horizon = max(int(horizon_days), 1)
    next_return = (close.shift(-horizon) / close) - 1.0
    hist = pd.DataFrame(
        {
            "state_tag": df["state_tag"].astype("string"),
            "next_return": next_return,
        },
        index=df.index,
    )
    hist = hist.iloc[:-horizon] if len(hist.index) > horizon else hist.iloc[0:0]
    hist = hist.dropna(subset=["state_tag", "next_return"])

    if lookback_days > 0 and len(hist.index) > int(lookback_days):
        hist = hist.tail(int(lookback_days))

    matched = hist[hist["state_tag"].str.upper() == current_state]
    sample_count = int(len(matched.index))

    cont_prob = None
    exp_return = None
    if sample_count >= max(int(min_samples), 1):
        cont_prob = float((matched["next_return"] > 0).mean())
        exp_return = float(matched["next_return"].mean())

    return_20d = _to_float_or_none(df["return_20d"].iloc[-1] if "return_20d" in df.columns else None)
    vol_20d = _to_float_or_none(df["realized_vol_20d"].iloc[-1] if "realized_vol_20d" in df.columns else None)

    mom_strength = None
    if return_20d is not None and vol_20d is not None:
        denom = max(float(vol_20d) * math.sqrt(20.0), 1e-9)
        mom_strength = float(np.clip(float(return_20d) / denom, -3.0, 3.0))

    return {
        "cont_prob_5d": cont_prob,
        "exp_return_5d": exp_return,
        "cont_samples_5d": sample_count if sample_count > 0 else None,
        "mom_strength": mom_strength,
    }


def compute_market_features_daily(symbol: str, bars_df: pd.DataFrame) -> pd.DataFrame:
    # Daily observatory windows are defined in trading-day counts.
    if bars_df is None or bars_df.empty:
        return pd.DataFrame()

    df = bars_df.copy()
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"bars_df missing required columns: {missing}")

    df = df.sort_index()
    close = pd.to_numeric(df["close"], errors="coerce")
    close = close.where(close > 0)

    log_close = np.log(close)
    log_returns = np.log(close / close.shift(1))

    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    trend_slope_20d = _rolling_slope(log_close, window=20)
    trend_strength_200d = _rolling_slope(log_close, window=200)
    realized_vol_20d = log_returns.rolling(window=20, min_periods=20).std()
    vol_percentile_252d = _rolling_percentile_rank(realized_vol_20d, window=252)

    rolling_max_252d = close.rolling(window=252, min_periods=252).max()
    dist_from_52w_high = (close - rolling_max_252d) / rolling_max_252d
    drawdown_252d = (rolling_max_252d - close) / rolling_max_252d

    return_1d = close.pct_change(1)
    return_5d = close.pct_change(5)
    return_20d = close.pct_change(20)

    state_tag = _state_from_trend(close, ema_50, trend_slope_20d)
    regime_tag = pd.Series(index=df.index, dtype="object")
    regime_tag[:] = "CHOP"
    regime_tag = regime_tag.mask(state_tag == "UPTREND", "BULL")
    regime_tag = regime_tag.mask(state_tag == "DOWNTREND", "BEAR")

    out = pd.DataFrame(
        {
            "symbol": str(symbol).upper(),
            "close": close,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "trend_slope_20d": trend_slope_20d,
            "trend_strength_200d": trend_strength_200d,
            "realized_vol_20d": realized_vol_20d,
            "vol_percentile_252d": vol_percentile_252d,
            "dist_from_52w_high": dist_from_52w_high,
            "drawdown_252d": drawdown_252d,
            "return_1d": return_1d,
            "return_5d": return_5d,
            "return_20d": return_20d,
            "state_tag": state_tag,
            "regime_tag": regime_tag,
        },
        index=df.index,
    )
    return out


def recompute_market_features_daily(
    session,
    *,
    symbols: list[str],
    timeframe: str = "1d",
    lookback_rows: int = 500,
    forecast_lookback_days: int = 504,
    forecast_horizon_days: int = 5,
    forecast_min_samples: int = 60,
) -> dict:
    symbol_list = [str(symbol).upper() for symbol in symbols if symbol]
    if not symbol_list:
        return {"updated": 0, "symbols": 0, "rows": 0}

    bars = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == str(timeframe))
            .where(MarketBar.symbol.in_(symbol_list))
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    grouped: dict[str, list[MarketBar]] = defaultdict(list)
    for bar in bars:
        grouped[str(bar.symbol).upper()].append(bar)

    rows_to_upsert: list[dict] = []
    symbols_updated = 0

    for symbol in symbol_list:
        symbol_bars = grouped.get(symbol, [])
        if not symbol_bars:
            continue

        if lookback_rows > 0 and len(symbol_bars) > lookback_rows:
            symbol_bars = symbol_bars[-lookback_rows:]

        idx = pd.to_datetime([int(row.bar_ts) for row in symbol_bars], unit="s", utc=True).date
        bars_df = pd.DataFrame(
            {
                "open": [float(row.open) for row in symbol_bars],
                "high": [float(row.high) for row in symbol_bars],
                "low": [float(row.low) for row in symbol_bars],
                "close": [float(row.close) for row in symbol_bars],
                "volume": [float(row.volume or 0.0) for row in symbol_bars],
            },
            index=idx,
        )

        feat_df = compute_market_features_daily(symbol, bars_df)
        if feat_df.empty:
            continue

        forecast_metrics = compute_latest_continuation_forecast(
            feat_df,
            lookback_days=forecast_lookback_days,
            horizon_days=forecast_horizon_days,
            min_samples=forecast_min_samples,
        )
        feat_df["cont_prob_5d"] = np.nan
        feat_df["exp_return_5d"] = np.nan
        feat_df["cont_samples_5d"] = np.nan
        feat_df["mom_strength"] = np.nan
        latest_idx = feat_df.index[-1]
        feat_df.loc[latest_idx, "cont_prob_5d"] = forecast_metrics.get("cont_prob_5d")
        feat_df.loc[latest_idx, "exp_return_5d"] = forecast_metrics.get("exp_return_5d")
        feat_df.loc[latest_idx, "cont_samples_5d"] = forecast_metrics.get("cont_samples_5d")
        feat_df.loc[latest_idx, "mom_strength"] = forecast_metrics.get("mom_strength")

        symbols_updated += 1
        for bar_date, row in feat_df.iterrows():
            bar_ts = int(pd.Timestamp(bar_date).tz_localize("UTC").timestamp())
            rows_to_upsert.append(
                {
                    "symbol": symbol,
                    "timeframe": str(timeframe),
                    "bar_ts": bar_ts,
                    "ema_20": _to_float_or_none(row.get("ema_20")),
                    "ema_50": _to_float_or_none(row.get("ema_50")),
                    "ema_200": _to_float_or_none(row.get("ema_200")),
                    "trend_slope_20d": _to_float_or_none(row.get("trend_slope_20d")),
                    "trend_strength_200d": _to_float_or_none(row.get("trend_strength_200d")),
                    "realized_vol_20d": _to_float_or_none(row.get("realized_vol_20d")),
                    "vol_percentile_252d": _to_float_or_none(row.get("vol_percentile_252d")),
                    "dist_from_52w_high": _to_float_or_none(row.get("dist_from_52w_high")),
                    "drawdown_252d": _to_float_or_none(row.get("drawdown_252d")),
                    "return_1d": _to_float_or_none(row.get("return_1d")),
                    "return_5d": _to_float_or_none(row.get("return_5d")),
                    "return_20d": _to_float_or_none(row.get("return_20d")),
                    "regime_tag": str(row.get("regime_tag") or "CHOP"),
                    "cont_prob_5d": _to_float_or_none(row.get("cont_prob_5d")),
                    "exp_return_5d": _to_float_or_none(row.get("exp_return_5d")),
                    "cont_samples_5d": _to_int_or_none(row.get("cont_samples_5d")),
                    "mom_strength": _to_float_or_none(row.get("mom_strength")),
                }
            )

    updated = upsert_market_features(session, rows_to_upsert)
    return {
        "updated": int(updated),
        "symbols": int(symbols_updated),
        "rows": int(len(rows_to_upsert)),
    }
