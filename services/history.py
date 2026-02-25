from __future__ import annotations

import logging
import time
from collections import defaultdict

from sqlalchemy import desc, func, select

from .db import BATCH_SIZE, ItemBar, ItemExtreme, ItemFeature, upsert_item_extremes as _db_upsert_item_extremes

logger = logging.getLogger(__name__)

EPSILON = 1e-9
WINDOW_TO_SECONDS: dict[str, int | None] = {
    "1h": 3600,
    "6h": 6 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
    "all": None,
}


def parse_window_to_seconds(window_str: str) -> int | None:
    key = (window_str or "").strip().lower()
    if key not in WINDOW_TO_SECONDS:
        raise ValueError(f"Unsupported window: {window_str}")
    return WINDOW_TO_SECONDS[key]


def get_item_history(
    session,
    item_key: str,
    window_seconds: int | None,
    limit: int,
    *,
    ascending: bool = True,
    now_ts: int | None = None,
) -> list[dict]:
    current_ts = int(now_ts or time.time())
    bounded_limit = max(int(limit), 1)

    query = select(ItemBar).where(ItemBar.item_key == item_key)
    if window_seconds is not None:
        cutoff = current_ts - max(int(window_seconds), 0)
        query = query.where(ItemBar.bar_ts >= cutoff)

    order_clause = ItemBar.bar_ts.asc() if ascending else ItemBar.bar_ts.desc()
    rows = session.execute(query.order_by(order_clause).limit(bounded_limit)).scalars().all()
    return [
        {
            "bar_ts": int(row.bar_ts),
            "n": int(row.n),
            "price_median": float(row.price_median),
            "price_min": float(row.price_min),
            "price_max": float(row.price_max),
        }
        for row in rows
    ]


def compute_max_drawdown_from_series(points: list[tuple[int, float]]) -> tuple[float, int | None, int | None]:
    max_drawdown = 0.0
    peak_so_far = 0.0
    peak_ts_so_far: int | None = None
    worst_peak_ts: int | None = None
    worst_trough_ts: int | None = None

    for ts, value in points:
        price = float(value)
        if price <= 0:
            continue
        if peak_so_far <= 0 or price > peak_so_far:
            peak_so_far = price
            peak_ts_so_far = int(ts)

        drawdown = (peak_so_far - price) / max(peak_so_far, EPSILON)
        if drawdown > max_drawdown:
            max_drawdown = float(drawdown)
            worst_peak_ts = peak_ts_so_far
            worst_trough_ts = int(ts)

    return float(max_drawdown), worst_peak_ts, worst_trough_ts


def _calc_window_high_low(
    rows: list[tuple[int, float, float, float]],
    *,
    now_ts: int,
    window_seconds: int,
) -> tuple[float | None, float | None]:
    cutoff = now_ts - int(window_seconds)
    high: float | None = None
    low: float | None = None
    for ts, _median, price_min, price_max in rows:
        if int(ts) < cutoff:
            continue
        pmax = float(price_max)
        pmin = float(price_min)
        if high is None or pmax > high:
            high = pmax
        if low is None or pmin < low:
            low = pmin
    return high, low


def _calc_window_mdd(
    rows: list[tuple[int, float, float, float]],
    *,
    now_ts: int,
    window_seconds: int,
) -> float | None:
    cutoff = now_ts - int(window_seconds)
    points = [(int(ts), float(price_median)) for ts, price_median, _min, _max in rows if int(ts) >= cutoff]
    if not points:
        return None
    mdd, _peak_ts, _trough_ts = compute_max_drawdown_from_series(points)
    return float(mdd)


def _compute_extremes_from_rows(
    item_key: str,
    rows: list[tuple[int, float, float, float]],
    *,
    now_ts: int,
    min_bars: int,
    price_now: float | None,
) -> dict | None:
    if not rows:
        return None

    bars_seen = len(rows)

    ath_price: float | None = None
    ath_ts: int | None = None
    atl_price: float | None = None
    atl_ts: int | None = None

    for ts, _median, price_min, price_max in rows:
        pmax = float(price_max)
        pmin = float(price_min)
        if ath_price is None or pmax > ath_price:
            ath_price = pmax
            ath_ts = int(ts)
        if atl_price is None or pmin < atl_price:
            atl_price = pmin
            atl_ts = int(ts)

    high_24h, low_24h = _calc_window_high_low(rows, now_ts=now_ts, window_seconds=24 * 3600)
    high_7d, low_7d = _calc_window_high_low(rows, now_ts=now_ts, window_seconds=7 * 24 * 3600)
    high_30d, low_30d = _calc_window_high_low(rows, now_ts=now_ts, window_seconds=30 * 24 * 3600)

    latest_median = float(rows[-1][1])
    current_price = float(price_now) if price_now is not None else latest_median

    pct_off_ath = None
    if ath_price is not None and ath_price > 0:
        pct_off_ath = (ath_price - current_price) / max(float(ath_price), EPSILON)

    pct_above_atl = None
    if atl_price is not None and atl_price > 0:
        pct_above_atl = (current_price - atl_price) / max(float(atl_price), EPSILON)

    seconds_since_ath = max(now_ts - int(ath_ts), 0) if ath_ts is not None else None
    seconds_since_atl = max(now_ts - int(atl_ts), 0) if atl_ts is not None else None

    mdd_all: float | None = None
    mdd_peak_ts: int | None = None
    mdd_trough_ts: int | None = None
    mdd_30d: float | None = None
    mdd_7d: float | None = None
    mdd_24h: float | None = None
    if bars_seen >= max(int(min_bars), 1):
        all_points = [(int(ts), float(price_median)) for ts, price_median, _min, _max in rows]
        mdd_all, mdd_peak_ts, mdd_trough_ts = compute_max_drawdown_from_series(all_points)
        mdd_30d = _calc_window_mdd(rows, now_ts=now_ts, window_seconds=30 * 24 * 3600)
        mdd_7d = _calc_window_mdd(rows, now_ts=now_ts, window_seconds=7 * 24 * 3600)
        mdd_24h = _calc_window_mdd(rows, now_ts=now_ts, window_seconds=24 * 3600)

    return {
        "item_key": item_key,
        "ts": int(now_ts),
        "bars_seen": int(bars_seen),
        "ath_price": ath_price,
        "ath_ts": ath_ts,
        "atl_price": atl_price,
        "atl_ts": atl_ts,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "high_7d": high_7d,
        "low_7d": low_7d,
        "high_30d": high_30d,
        "low_30d": low_30d,
        "pct_off_ath": pct_off_ath,
        "pct_above_atl": pct_above_atl,
        "seconds_since_ath": seconds_since_ath,
        "seconds_since_atl": seconds_since_atl,
        "mdd_all": mdd_all,
        "mdd_peak_ts": mdd_peak_ts,
        "mdd_trough_ts": mdd_trough_ts,
        "mdd_30d": mdd_30d,
        "mdd_7d": mdd_7d,
        "mdd_24h": mdd_24h,
    }


def compute_extremes_for_item(
    session,
    item_key: str,
    now_ts: int,
    *,
    min_bars: int = 90,
    price_now_hint: float | None = None,
    bars_rows: list[tuple[int, float, float, float]] | None = None,
) -> dict | None:
    if bars_rows is None:
        raw_rows = session.execute(
            select(ItemBar.bar_ts, ItemBar.price_median, ItemBar.price_min, ItemBar.price_max)
            .where(ItemBar.item_key == item_key)
            .order_by(ItemBar.bar_ts.asc())
        ).all()
        rows = [(int(ts), float(median), float(price_min), float(price_max)) for ts, median, price_min, price_max in raw_rows]
    else:
        rows = [(int(ts), float(median), float(price_min), float(price_max)) for ts, median, price_min, price_max in bars_rows]

    if not rows:
        return None

    if price_now_hint is not None:
        price_now = float(price_now_hint)
    else:
        feature_price = session.execute(
            select(ItemFeature.price_now).where(ItemFeature.item_key == item_key).limit(1)
        ).scalar_one_or_none()
        price_now = float(feature_price) if feature_price is not None else float(rows[-1][1])

    return _compute_extremes_from_rows(
        item_key,
        rows,
        now_ts=int(now_ts),
        min_bars=max(int(min_bars), 1),
        price_now=price_now,
    )


def upsert_item_extremes(session, rows: list[dict]) -> int:
    return _db_upsert_item_extremes(session, rows)


def _select_extremes_candidates(
    session,
    *,
    now_ts: int,
    top_n: int,
    liq_rank_min: float,
) -> list[str]:
    bounded_top_n = max(int(top_n), 1)
    primary_rows = session.execute(
        select(ItemFeature.item_key)
        .where(ItemFeature.liquidity_ok == 1)
        .where(ItemFeature.liquidity_rank >= float(liq_rank_min))
        .order_by(ItemFeature.liquidity_rank.desc(), ItemFeature.samples_24h.desc())
        .limit(bounded_top_n)
    ).scalars().all()
    if primary_rows:
        return [str(item_key) for item_key in primary_rows if item_key]

    cutoff = int(now_ts) - 24 * 3600
    fallback_rows = session.execute(
        select(ItemBar.item_key, func.count().label("bar_count"))
        .where(ItemBar.bar_ts >= cutoff)
        .group_by(ItemBar.item_key)
        .order_by(desc("bar_count"), ItemBar.item_key.asc())
        .limit(bounded_top_n)
    ).all()
    return [str(item_key) for item_key, _count in fallback_rows if item_key]


def recompute_item_extremes(
    session,
    *,
    now_ts: int,
    top_n: int,
    liq_rank_min: float,
    min_bars: int,
) -> dict:
    candidate_keys = _select_extremes_candidates(
        session,
        now_ts=now_ts,
        top_n=top_n,
        liq_rank_min=liq_rank_min,
    )
    if not candidate_keys:
        return {"updated": 0, "candidates": 0}

    feature_price_rows = session.execute(
        select(ItemFeature.item_key, ItemFeature.price_now).where(ItemFeature.item_key.in_(set(candidate_keys)))
    ).all()
    feature_price_map = {
        str(item_key): float(price_now)
        for item_key, price_now in feature_price_rows
        if item_key and price_now is not None
    }

    bar_rows = session.execute(
        select(ItemBar.item_key, ItemBar.bar_ts, ItemBar.price_median, ItemBar.price_min, ItemBar.price_max)
        .where(ItemBar.item_key.in_(set(candidate_keys)))
        .order_by(ItemBar.item_key.asc(), ItemBar.bar_ts.asc())
    ).all()

    grouped_rows: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for item_key, bar_ts, price_median, price_min, price_max in bar_rows:
        grouped_rows[str(item_key)].append((int(bar_ts), float(price_median), float(price_min), float(price_max)))

    out_rows: list[dict] = []
    for item_key in candidate_keys:
        computed = compute_extremes_for_item(
            session,
            item_key=item_key,
            now_ts=now_ts,
            min_bars=min_bars,
            price_now_hint=feature_price_map.get(item_key),
            bars_rows=grouped_rows.get(item_key, []),
        )
        if computed is None:
            continue
        out_rows.append(computed)

    updated = upsert_item_extremes(session, out_rows)
    logger.debug(
        "Extremes updated: candidates=%s rows=%s updated=%s batch_size=%s",
        len(candidate_keys),
        len(out_rows),
        updated,
        BATCH_SIZE,
    )
    return {
        "updated": int(updated),
        "candidates": int(len(candidate_keys)),
    }
