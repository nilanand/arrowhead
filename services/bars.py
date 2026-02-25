from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import IntradayBar

logger = logging.getLogger(__name__)
_INTRADAY_BATCH_SIZE = 500


def align_bar_ts(ts: int, bar_seconds: int = 60) -> int:
    size = max(int(bar_seconds), 1)
    return int(ts - (ts % size))


@dataclass
class MarketBarBuilder:
    bar_seconds: int = 60
    _minute_buckets: dict[int, dict[str, dict]] = field(default_factory=dict)

    def ingest_quotes(self, quotes: list[dict]) -> int:
        accepted = 0
        for quote in quotes:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if not symbol:
                continue

            ts = int(quote.get("ts") or 0)
            price = float(quote.get("price") or 0.0)
            volume = float(quote.get("volume") or 0.0)
            if ts <= 0 or price <= 0:
                continue

            minute_ts = align_bar_ts(ts, self.bar_seconds)
            bucket = self._minute_buckets.setdefault(minute_ts, {})
            agg = bucket.get(symbol)
            if agg is None:
                bucket[symbol] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": max(volume, 0.0),
                }
            else:
                agg["high"] = max(float(agg["high"]), price)
                agg["low"] = min(float(agg["low"]), price)
                agg["close"] = price
                agg["volume"] = float(agg["volume"]) + max(volume, 0.0)
            accepted += 1

        return accepted

    def flush_closed_bars(self, *, now_ts: int) -> tuple[list[dict], int | None]:
        current_minute = align_bar_ts(int(now_ts), self.bar_seconds)
        closable = sorted(minute_ts for minute_ts in self._minute_buckets.keys() if minute_ts < current_minute)
        if not closable:
            return [], None

        rows: list[dict] = []
        for minute_ts in closable:
            bucket = self._minute_buckets.pop(minute_ts, {})
            for symbol, agg in bucket.items():
                rows.append(
                    {
                        "symbol": str(symbol).upper(),
                        "timeframe": "1m",
                        "bar_ts": int(minute_ts),
                        "open": float(agg["open"]),
                        "high": float(agg["high"]),
                        "low": float(agg["low"]),
                        "close": float(agg["close"]),
                        "volume": float(agg["volume"]),
                    }
                )

        return rows, int(closable[-1])


def upsert_intraday_bars(session, symbol: str, df_1m: pd.DataFrame) -> int:
    symbol_upper = str(symbol or "").strip().upper()
    if not symbol_upper:
        return 0

    fetched_count = 0
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if df_1m is not None and not df_1m.empty:
        frame = df_1m.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
        frame = frame[~frame.index.isna()]
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in frame.columns:
                frame[col] = pd.Series(dtype="float64")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame["volume"] = frame["volume"].fillna(0.0)
        fetched_count = int(len(frame.index))

    last_ts = session.execute(
        select(func.max(IntradayBar.bar_ts)).where(IntradayBar.symbol == symbol_upper)
    ).scalar_one_or_none()

    if last_ts is not None and not frame.empty:
        frame = frame[frame.index > pd.Timestamp(last_ts)]

    if frame.empty:
        logger.info(
            "intraday upserted 0 new bars for %s (fetched=%s, last_ts=%s)",
            symbol_upper,
            int(fetched_count),
            (last_ts.isoformat() if last_ts is not None else None),
        )
        return 0

    rows = [
        {
            "symbol": symbol_upper,
            "bar_ts": pd.Timestamp(idx_dt).to_pydatetime(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
        }
        for idx_dt, row in frame.iterrows()
    ]
    if not rows:
        logger.info(
            "intraday upserted 0 new bars for %s (fetched=%s, last_ts=%s)",
            symbol_upper,
            int(fetched_count),
            (last_ts.isoformat() if last_ts is not None else None),
        )
        return 0

    before_count = int(
        session.execute(
            select(func.count()).select_from(IntradayBar).where(IntradayBar.symbol == symbol_upper)
        ).scalar_one()
        or 0
    )

    insert_stmt = sqlite_insert(IntradayBar).on_conflict_do_nothing(
        index_elements=[IntradayBar.symbol, IntradayBar.bar_ts]
    )
    for i in range(0, len(rows), _INTRADAY_BATCH_SIZE):
        session.execute(insert_stmt, rows[i:i + _INTRADAY_BATCH_SIZE])

    after_count = int(
        session.execute(
            select(func.count()).select_from(IntradayBar).where(IntradayBar.symbol == symbol_upper)
        ).scalar_one()
        or 0
    )
    inserted_count = max(after_count - before_count, 0)

    logger.info(
        "intraday upserted %s new bars for %s (fetched=%s, last_ts=%s)",
        int(inserted_count),
        symbol_upper,
        int(fetched_count),
        (last_ts.isoformat() if last_ts is not None else None),
    )
    return int(inserted_count)
