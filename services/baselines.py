from __future__ import annotations

import math
import time
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import AuctionHistory, ItemBaseline


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    q = max(0.0, min(1.0, q))
    idx = (len(sorted_values) - 1) * q
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return float(sorted_values[lower])

    lower_val = float(sorted_values[lower])
    upper_val = float(sorted_values[upper])
    fraction = idx - lower
    return lower_val + (upper_val - lower_val) * fraction


def recompute_baselines(session, baseline_hours: int) -> int:
    now_ts = int(time.time())
    cutoff = now_ts - (baseline_hours * 3600)

    rows = session.execute(
        select(AuctionHistory.item_key, AuctionHistory.price)
        .where(AuctionHistory.bin == 1)
        .where(AuctionHistory.ts >= cutoff)
    ).all()

    grouped: dict[str, list[float]] = defaultdict(list)
    for item_key, price in rows:
        grouped[item_key].append(float(price))

    baseline_rows: list[dict] = []
    for item_key, prices in grouped.items():
        prices.sort()
        p25 = _percentile(prices, 0.25)
        median = _percentile(prices, 0.5)
        p75 = _percentile(prices, 0.75)
        iqr = p75 - p25
        baseline_rows.append(
            {
                "item_key": item_key,
                "ts": now_ts,
                "sample_n": len(prices),
                "p25": p25,
                "median": median,
                "p75": p75,
                "iqr": iqr,
                "min": float(prices[0]),
                "max": float(prices[-1]),
            }
        )

    if baseline_rows:
        upsert_stmt = sqlite_insert(ItemBaseline).values(baseline_rows)
        update_columns = {
            c.name: upsert_stmt.excluded[c.name]
            for c in ItemBaseline.__table__.columns
            if c.name != "item_key"
        }
        session.execute(
            upsert_stmt.on_conflict_do_update(index_elements=[ItemBaseline.item_key], set_=update_columns)
        )

    keys = set(grouped.keys())
    if keys:
        session.execute(delete(ItemBaseline).where(ItemBaseline.item_key.not_in(keys)))
    else:
        session.execute(delete(ItemBaseline))

    return len(baseline_rows)
