from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict

from sqlalchemy import select

from .db import MarketBar, insert_market_events

EPSILON = 1e-9


def _build_event_row(
    *,
    ts: int,
    symbol: str,
    event_type: str,
    severity: int,
    details: dict,
    dedupe_key: str,
) -> dict:
    return {
        "ts": int(ts),
        "symbol": str(symbol).upper(),
        "event_type": str(event_type),
        "severity": max(min(int(severity), 5), 1),
        "details_json": json.dumps(details, separators=(",", ":")),
        "dedupe_key": str(dedupe_key),
    }


def detect_market_events_phase0(session, *, symbols: list[str], now_ts: int) -> dict:
    symbol_list = [str(symbol).upper() for symbol in symbols if symbol]
    if not symbol_list:
        return {"generated": 0, "inserted": 0, "events": []}

    rows: list[dict] = []

    # Event 1: NEW_HIGH_WINDOW (30d rolling high breakout)
    cutoff_30d = int(now_ts) - (30 * 86_400)
    bars = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.timeframe == "1m")
            .where(MarketBar.symbol.in_(symbol_list))
            .where(MarketBar.bar_ts >= cutoff_30d)
            .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.asc())
        )
        .scalars()
        .all()
    )

    grouped: dict[str, list[MarketBar]] = defaultdict(list)
    for bar in bars:
        grouped[str(bar.symbol).upper()].append(bar)

    for symbol in symbol_list:
        symbol_bars = grouped.get(symbol, [])
        if not symbol_bars:
            continue
        symbol_bars = sorted(symbol_bars, key=lambda row: int(row.bar_ts))

        latest = symbol_bars[-1]
        prior = symbol_bars[:-1]
        latest_close = float(latest.close)
        prior_high = max([float(row.high) for row in prior] or [latest_close])

        if latest_close >= prior_high:
            rows.append(
                _build_event_row(
                    ts=now_ts,
                    symbol=symbol,
                    event_type="NEW_HIGH_WINDOW",
                    severity=3,
                    details={
                        "window": "30d",
                        "close": latest_close,
                        "prior_high": prior_high,
                        "bar_ts": int(latest.bar_ts),
                    },
                    dedupe_key=f"{symbol}|NEW_HIGH_WINDOW|{int(latest.bar_ts)}",
                )
            )

    # Event 2: VOL_SHOCK (latest absolute return is at upper tail of rolling return distribution)
    for symbol in symbol_list:
        symbol_bars = grouped.get(symbol, [])
        if not symbol_bars:
            continue
        symbol_bars = sorted(symbol_bars, key=lambda row: int(row.bar_ts))
        closes = [float(row.close) for row in symbol_bars]
        abs_returns: list[float] = []
        for prev, cur in zip(closes, closes[1:]):
            if prev <= 0 or cur <= 0:
                continue
            abs_returns.append(abs(math.log(cur / prev)))

        if not abs_returns:
            latest_bar_ts = int(symbol_bars[-1].bar_ts)
            warmup_volume = float(symbol_bars[-1].volume or 0.0)
            if warmup_volume > 0:
                rows.append(
                    _build_event_row(
                        ts=now_ts,
                        symbol=symbol,
                        event_type="VOL_SHOCK",
                        severity=2,
                        details={
                            "window": "30d",
                            "warmup": True,
                            "volume": warmup_volume,
                            "bar_ts": latest_bar_ts,
                        },
                        dedupe_key=f"{symbol}|VOL_SHOCK|{latest_bar_ts}",
                    )
                )
            continue

        latest_abs_ret = float(abs_returns[-1])
        historical = abs_returns[:-1]
        if historical:
            sorted_hist = sorted(historical)
            rank = sum(1 for value in sorted_hist if value <= latest_abs_ret)
            vol_percentile = float(100.0 * rank / max(len(sorted_hist), 1))
            threshold = max(sorted_hist[-1], EPSILON)
        else:
            vol_percentile = 100.0
            threshold = latest_abs_ret

        if vol_percentile >= 80.0 and latest_abs_ret > EPSILON:
            latest_bar_ts = int(symbol_bars[-1].bar_ts)
            severity = 4 if vol_percentile >= 95.0 else 3
            rows.append(
                _build_event_row(
                    ts=now_ts,
                    symbol=symbol,
                    event_type="VOL_SHOCK",
                    severity=severity,
                    details={
                        "window": "30d",
                        "latest_abs_return": latest_abs_ret,
                        "vol_percentile": vol_percentile,
                        "threshold": threshold,
                        "bar_ts": latest_bar_ts,
                    },
                    dedupe_key=f"{symbol}|VOL_SHOCK|{latest_bar_ts}",
                )
            )

    inserted = insert_market_events(session, rows)

    events_out = [
        {
            "ts": int(row["ts"]),
            "symbol": str(row["symbol"]),
            "event_type": str(row["event_type"]),
            "severity": int(row["severity"]),
            "details_json": str(row["details_json"]),
            "dedupe_key": str(row["dedupe_key"]),
        }
        for row in rows
    ]

    return {
        "generated": len(rows),
        "inserted": int(inserted),
        "events": events_out,
    }
