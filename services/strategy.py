from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select

from .db import (
    ItemFeature,
    PortfolioPosition,
    close_position,
    get_open_position_map,
    open_position,
    upsert_recommendations,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.08
    ah_fee_pct: float = 0.01
    max_positions: int = 3
    max_vol_60m: float = 0.06
    top_buy_n: int = 50
    liq_rank_min: float = 60.0


@dataclass
class _ProvisionalRec:
    item_key: str
    action: str
    score: float
    reason: str
    target_buy: float | None
    target_sell: float | None


def _build_sell_signal(position: PortfolioPosition, feature: ItemFeature, cfg: StrategyConfig) -> tuple[str, float, str, float | None]:
    entry = float(position.entry_price)
    price_now = float(feature.price_now)

    if price_now <= entry * (1 - cfg.stop_loss_pct):
        return "SELL", 4.0, "Stop loss hit", entry * (1 - cfg.stop_loss_pct)

    if price_now >= entry * (1 + cfg.take_profit_pct):
        return "SELL", 3.0, "Take profit reached", entry * (1 + cfg.take_profit_pct)

    if float(feature.zscore_24h) >= 0.5:
        return "SELL", 2.0, "Mean reversion exit (z>=0.5)", float(feature.median_24h)

    if float(feature.ema_fast) < float(feature.ema_slow):
        return "SELL", 1.5, "Trend reversal (EMA fast below slow)", float(feature.ema_slow)

    return "HOLD", 0.2, "Position still valid", entry * (1 + cfg.take_profit_pct)


def _build_buy_candidate(feature: ItemFeature, cfg: StrategyConfig) -> tuple[bool, float, str]:
    liquidity_rank = float(feature.liquidity_rank or 0.0)
    if liquidity_rank < float(cfg.liq_rank_min):
        return (
            False,
            0.0,
            f"Liquidity rank {liquidity_rank:.1f} below min {float(cfg.liq_rank_min):.1f}",
        )

    if int(feature.liquidity_ok) != 1:
        return False, 0.0, "Insufficient liquidity"

    price_now = float(feature.price_now)
    median = float(feature.median_24h)
    iqr = float(feature.iqr_24h)
    zscore = float(feature.zscore_24h)

    undervalued = zscore <= -1.0 or price_now <= (median - 1.5 * iqr)
    trend_ok = float(feature.ema_fast) > float(feature.ema_slow) or float(feature.trend_slope_60m) > 0
    vol_ok = float(feature.vol_60m) <= cfg.max_vol_60m

    if not undervalued:
        return False, 0.0, "Not undervalued"
    if not trend_ok:
        return False, 0.0, "Undervalued but trend not improving"
    if not vol_ok:
        return False, 0.0, "Volatility too high"

    discount = 0.0 if median <= 0 else max((median - price_now) / median, 0.0)
    raw_buy_score = max(-zscore, 0.0) * 1.6 + discount * 4.5
    if float(feature.ema_fast) > float(feature.ema_slow):
        raw_buy_score += 0.75
    if float(feature.trend_slope_60m) > 0:
        raw_buy_score += min(float(feature.trend_slope_60m) / max(price_now, 1.0), 1.0)

    score = raw_buy_score * (1.0 + liquidity_rank / 100.0)

    reason = (
        f"Undervalued ({zscore:.2f} IQR) + trend improving + liquidity rank {liquidity_rank:.1f}"
    )
    return True, score, reason


def recompute_recommendations_and_portfolio(
    session,
    *,
    now_ts: int,
    cfg: StrategyConfig,
) -> dict:
    features = session.execute(select(ItemFeature)).scalars().all()
    open_positions = get_open_position_map(session)

    provisional: list[_ProvisionalRec] = []
    buy_pool: list[tuple[str, float, str, ItemFeature]] = []

    for feature in features:
        held_pos = open_positions.get(feature.item_key)
        if held_pos is not None:
            action, score, reason, target_sell = _build_sell_signal(held_pos, feature, cfg)
            provisional.append(
                _ProvisionalRec(
                    item_key=feature.item_key,
                    action=action,
                    score=score,
                    reason=reason,
                    target_buy=None,
                    target_sell=target_sell,
                )
            )
            continue

        is_buy, score, reason = _build_buy_candidate(feature, cfg)
        if is_buy:
            buy_pool.append((feature.item_key, score, reason, feature))
            provisional.append(
                _ProvisionalRec(
                    item_key=feature.item_key,
                    action="BUY",
                    score=score,
                    reason=reason,
                    target_buy=float(feature.price_now),
                    target_sell=float(feature.median_24h),
                )
            )
        else:
            action = "IGNORE" if int(feature.liquidity_ok) == 0 else "HOLD"
            if reason.startswith("Liquidity rank"):
                action = "IGNORE"
            provisional.append(
                _ProvisionalRec(
                    item_key=feature.item_key,
                    action=action,
                    score=0.0,
                    reason=reason,
                    target_buy=None,
                    target_sell=None,
                )
            )

    buy_pool.sort(key=lambda row: row[1], reverse=True)
    top_buy = buy_pool[: max(int(cfg.top_buy_n), 1)]
    top_buy_keys = {row[0] for row in top_buy}

    recommendation_rows: list[dict] = []
    for rec in provisional:
        action = rec.action
        score = rec.score
        reason = rec.reason
        target_buy = rec.target_buy
        target_sell = rec.target_sell

        if action == "BUY" and rec.item_key not in top_buy_keys:
            action = "HOLD"
            score = 0.0
            reason = "Signal valid but below top buy cutoff"
            target_buy = None

        recommendation_rows.append(
            {
                "item_key": rec.item_key,
                "ts": now_ts,
                "action": action,
                "score": float(score),
                "reason": reason,
                "target_buy": target_buy,
                "target_sell": target_sell,
            }
        )

    rec_count = upsert_recommendations(session, recommendation_rows)

    closed_positions = 0
    open_position_map = get_open_position_map(session)
    feature_map = {feature.item_key: feature for feature in features}

    for item_key, position in list(open_position_map.items()):
        feature = feature_map.get(item_key)
        if feature is None:
            continue
        action, _, reason, _ = _build_sell_signal(position, feature, cfg)
        if action != "SELL":
            continue

        closed = close_position(
            session,
            position,
            closed_ts=now_ts,
            exit_price=float(feature.price_now),
            fee_pct=cfg.ah_fee_pct,
        )
        closed_positions += 1
        logger.debug(
            "Portfolio close: item_key=%s duration=%s pnl_after_fee=%.3f profit_per_hour=%.3f",
            item_key,
            int(closed.duration_seconds or 0),
            float(closed.pnl_after_fee or 0.0),
            float(closed.profit_per_hour or 0.0),
        )
        logger.debug(
            "Paper trade closed: item=%s entry=%.0f exit=%.0f reason=%s",
            item_key,
            float(position.entry_price),
            float(feature.price_now),
            reason,
        )

    current_open_positions = session.execute(
        select(PortfolioPosition).where(PortfolioPosition.status == "OPEN")
    ).scalars().all()
    open_keys = {row.item_key for row in current_open_positions}
    slots = max(int(cfg.max_positions) - len(current_open_positions), 0)

    opened_positions = 0
    for item_key, _, reason, feature in top_buy:
        if slots <= 0:
            break
        if item_key in open_keys:
            continue

        open_position(
            session,
            item_key=item_key,
            opened_ts=now_ts,
            entry_price=float(feature.price_now),
            qty=1,
        )
        open_keys.add(item_key)
        slots -= 1
        opened_positions += 1
        logger.debug(
            "Paper trade opened: item=%s entry=%.0f reason=%s",
            item_key,
            float(feature.price_now),
            reason,
        )

    buy_count = sum(1 for row in recommendation_rows if row["action"] == "BUY")
    sell_count = sum(1 for row in recommendation_rows if row["action"] == "SELL")

    logger.debug(
        "Recommendations computed count=%s buys=%s sells=%s opened=%s closed=%s",
        rec_count,
        buy_count,
        sell_count,
        opened_positions,
        closed_positions,
    )

    return {
        "recommendations": rec_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "opened_positions": opened_positions,
        "closed_positions": closed_positions,
    }
