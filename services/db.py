from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time
from contextlib import contextmanager

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    desc,
    func,
    literal,
    select,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, scoped_session, sessionmaker

from .normalization import canonicalize_auction

BATCH_SIZE = 200
MC_FORMAT_CODE_RE = re.compile(r"§.")
WS_RE = re.compile(r"\s+")

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Auction(Base):
    __tablename__ = "auctions"

    uuid: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    item_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    tier: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    starting_bid: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    highest_bid_amount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bin: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    bids_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    seller: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    last_updated_ts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_item_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    canonical_item_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)


class AuctionHistory(Base):
    __tablename__ = "auctions_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    item_name: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    bin: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    uuid: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    raw_item_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    canonical_item_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)


class ItemBar(Base):
    __tablename__ = "item_bars"

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    bar_ts: Mapped[int] = mapped_column(Integer, primary_key=True)
    n: Mapped[int] = mapped_column(Integer, nullable=False)
    price_median: Mapped[float] = mapped_column(Float, nullable=False)
    price_min: Mapped[float] = mapped_column(Float, nullable=False)
    price_max: Mapped[float] = mapped_column(Float, nullable=False)


class ItemFeature(Base):
    __tablename__ = "item_features"

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    price_now: Mapped[float] = mapped_column(Float, nullable=False)
    ema_fast: Mapped[float] = mapped_column(Float, nullable=False)
    ema_slow: Mapped[float] = mapped_column(Float, nullable=False)
    trend_slope_60m: Mapped[float] = mapped_column(Float, nullable=False)
    vol_60m: Mapped[float] = mapped_column(Float, nullable=False)
    median_24h: Mapped[float] = mapped_column(Float, nullable=False)
    p25_24h: Mapped[float] = mapped_column(Float, nullable=False)
    p75_24h: Mapped[float] = mapped_column(Float, nullable=False)
    iqr_24h: Mapped[float] = mapped_column(Float, nullable=False)
    zscore_24h: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown_6h: Mapped[float] = mapped_column(Float, nullable=False)
    samples_1h: Mapped[int] = mapped_column(Integer, nullable=False)
    samples_24h: Mapped[int] = mapped_column(Integer, nullable=False)
    liquidity_ok: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    liquidity_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_tag: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)


class ItemSpreadSnapshot(Base):
    __tablename__ = "item_spread_snapshots"
    __table_args__ = (
        Index("idx_item_spread_ts", "ts"),
        Index("idx_item_spread_key_ts", "item_key", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    best_bin: Mapped[int] = mapped_column(Integer, nullable=False)
    second_best_bin: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spread_abs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spread_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    active_listings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ItemForecast(Base):
    __tablename__ = "item_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "item_key",
            "ts",
            "horizon_minutes",
            "model_name",
            name="uq_item_forecast_key_ts_horizon_model",
        ),
        Index("idx_forecasts_item_ts", "item_key", "ts"),
        Index("idx_forecasts_ts", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    pred_return: Mapped[float] = mapped_column(Float, nullable=False)
    pred_p10: Mapped[float] = mapped_column(Float, nullable=False)
    pred_p50: Mapped[float] = mapped_column(Float, nullable=False)
    pred_p90: Mapped[float] = mapped_column(Float, nullable=False)
    p_up: Mapped[float] = mapped_column(Float, nullable=False)
    pred_vol: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    features_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class NocomModelState(Base):
    __tablename__ = "nocom_model_state"
    __table_args__ = (
        UniqueConstraint(
            "horizon_minutes",
            "quantile",
            "model_name",
            "feature_schema_version",
            name="uq_nocom_model_state_hqmv",
        ),
        Index("idx_model_state_hq", "horizon_minutes", "quantile"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    quantile: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    feature_schema_version: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    weights_json: Mapped[str] = mapped_column(Text, nullable=False)
    n_updates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_update_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_loss: Mapped[float | None] = mapped_column(Float, nullable=True)


class NocomTrainPair(Base):
    __tablename__ = "nocom_train_pairs"
    __table_args__ = (
        UniqueConstraint(
            "item_key",
            "t0_bar_ts",
            "horizon_minutes",
            name="uq_nocom_train_pair_item_t0_h",
        ),
        Index("idx_pairs_horizon_t0", "horizon_minutes", "t0_bar_ts"),
        Index("idx_pairs_item_horizon", "item_key", "horizon_minutes"),
        Index("idx_pairs_unlabeled", "horizon_minutes", "y_realized"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    t0_bar_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    x_json: Mapped[str] = mapped_column(Text, nullable=False)
    y_realized: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)


class ItemExtreme(Base):
    __tablename__ = "item_extremes"
    __table_args__ = (
        Index("idx_extremes_ts", "ts"),
    )

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    bars_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ath_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ath_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    atl_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    atl_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)

    high_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_30d: Mapped[float | None] = mapped_column(Float, nullable=True)

    pct_off_ath: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_above_atl: Mapped[float | None] = mapped_column(Float, nullable=True)
    seconds_since_ath: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seconds_since_atl: Mapped[int | None] = mapped_column(Integer, nullable=True)

    mdd_all: Mapped[float | None] = mapped_column(Float, nullable=True)
    mdd_peak_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mdd_trough_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mdd_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    mdd_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    mdd_24h: Mapped[float | None] = mapped_column(Float, nullable=True)


class ItemEvent(Base):
    __tablename__ = "item_events"
    __table_args__ = (
        Index("idx_events_item_ts", "item_key", "ts"),
        Index("idx_events_type_ts", "event_type", "ts"),
        Index("idx_events_dedupe", "dedupe_key", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)


class FavoriteItem(Base):
    __tablename__ = "favorite_items"
    __table_args__ = (
        Index("idx_favorites_created", "created_ts"),
    )

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    created_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    pinned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class WatchTarget(Base):
    __tablename__ = "watch_targets"
    __table_args__ = (
        UniqueConstraint(
            "target_type",
            "item_key",
            "group_kind",
            "group_value",
            name="uq_watch_targets_target_item_group",
        ),
        Index("idx_watch_targets_enabled", "enabled"),
        Index("idx_watch_targets_type", "target_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    item_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    group_kind: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    group_value: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1, index=True)
    created_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)


class ItemKeyMeta(Base):
    __tablename__ = "item_key_meta"
    __table_args__ = (
        Index("idx_meta_name", "display_name"),
        Index("idx_meta_base", "base_id"),
    )

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    base_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    item_key: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    action: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    target_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_sell: Mapped[float | None] = mapped_column(Float, nullable=True)


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    opened_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    closed_ts: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pnl_after_fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_per_hour: Mapped[float | None] = mapped_column(Float, nullable=True)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    lookback_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class AlertSent(Base):
    __tablename__ = "alerts_sent"
    __table_args__ = (
        Index("idx_alerts_item_type_ts", "item_key", "alert_type", "ts"),
        Index("idx_alerts_uuid", "uuid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    item_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    uuid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)


class Snapshot(Base):
    __tablename__ = "snapshots"

    ts: Mapped[int] = mapped_column(Integer, primary_key=True)
    ok: Mapped[int] = mapped_column(Integer, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    auction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Symbol(Base):
    __tablename__ = "symbols"
    __table_args__ = (
        Index("idx_symbols_exchange_active", "exchange", "active"),
        Index("idx_symbols_last_seen", "last_seen_ts"),
    )

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    exchange: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    asset_type: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1, index=True)
    last_seen_ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)


class MarketBar(Base):
    __tablename__ = "market_bars"
    __table_args__ = (
        Index("idx_market_bars_tf_ts", "timeframe", "bar_ts"),
        Index("idx_market_bars_symbol_tf_ts", "symbol", "timeframe", "bar_ts"),
    )

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    timeframe: Mapped[str] = mapped_column(Text, primary_key=True)
    bar_ts: Mapped[int] = mapped_column(Integer, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


# --- INTRADAY: intraday_bars model (v0.16.0) ---
class IntradayBar(Base):
    __tablename__ = "intraday_bars"
    __table_args__ = (
        Index("idx_intraday_bars_symbol_ts", "symbol", "bar_ts"),
    )

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    bar_ts: Mapped[dt.datetime] = mapped_column(DateTime, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


# --- END INTRADAY ---
class MarketFeature(Base):
    __tablename__ = "market_features"
    __table_args__ = (
        Index("idx_market_features_tf_bar_ts", "timeframe", "bar_ts"),
        Index("idx_market_features_symbol_tf_bar_ts", "symbol", "timeframe", "bar_ts"),
        Index("idx_market_features_regime_bar_ts", "regime_tag", "bar_ts"),
    )

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    timeframe: Mapped[str] = mapped_column(Text, primary_key=True)
    bar_ts: Mapped[int] = mapped_column(Integer, primary_key=True)
    ema_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_slope_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_strength_200d: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_vol_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_percentile_252d: Mapped[float | None] = mapped_column(Float, nullable=True)
    dist_from_52w_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown_252d: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_tag: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    cont_prob_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    exp_return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    cont_samples_5d: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mom_strength: Mapped[float | None] = mapped_column(Float, nullable=True)


class MarketRegime(Base):
    __tablename__ = "market_regime"
    __table_args__ = (
        Index("idx_market_regime_tag_ts", "regime_tag", "ts"),
    )

    ts: Mapped[int] = mapped_column(Integer, primary_key=True)
    breadth_pct_above_ema20: Mapped[float | None] = mapped_column(Float, nullable=True)
    breadth_pct_above_ema50: Mapped[float | None] = mapped_column(Float, nullable=True)
    breadth_pct_above_ema200: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_with_positive_20d_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_20d_high_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    new_20d_low_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_volatility_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_tag: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    computed_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)


class MarketEvent(Base):
    __tablename__ = "market_events"
    __table_args__ = (
        Index("idx_market_events_ts", "ts"),
        Index("idx_market_events_symbol_ts", "symbol", "ts"),
        Index("idx_market_events_type_ts", "event_type", "ts"),
        Index("idx_market_events_dedupe", "dedupe_key", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)


class AlertSentEquity(Base):
    __tablename__ = "alerts_sent_equities"
    __table_args__ = (
        Index("idx_alerts_equities_type_symbol_ts", "alert_type", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)


_ENGINE = None
SessionLocal = None


def get_default_db_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///nocom.db")


def _ensure_sqlite_columns(engine) -> None:
    if not str(engine.url).startswith("sqlite"):
        return

    migrations: dict[str, list[tuple[str, str]]] = {
        "auctions": [
            ("price", "INTEGER NOT NULL DEFAULT 0"),
            ("item_id", "TEXT"),
            ("last_updated_ts", "INTEGER NOT NULL DEFAULT 0"),
            ("raw_item_key", "TEXT"),
            ("canonical_item_key", "TEXT"),
        ],
        "auctions_history": [
            ("bin", "INTEGER NOT NULL DEFAULT 1"),
            ("raw_item_key", "TEXT"),
            ("canonical_item_key", "TEXT"),
        ],
        "recommendations": [
            ("target_buy", "REAL"),
            ("target_sell", "REAL"),
        ],
        "item_features": [
            ("liquidity_rank", "REAL"),
            ("regime_tag", "TEXT"),
        ],
        "portfolio_positions": [
            ("duration_seconds", "INTEGER"),
            ("pnl_after_fee", "REAL"),
            ("return_pct", "REAL"),
            ("profit_per_hour", "REAL"),
        ],
        "item_extremes": [
            ("ts", "INTEGER NOT NULL DEFAULT 0"),
            ("bars_seen", "INTEGER NOT NULL DEFAULT 0"),
            ("ath_price", "REAL"),
            ("ath_ts", "INTEGER"),
            ("atl_price", "REAL"),
            ("atl_ts", "INTEGER"),
            ("high_24h", "REAL"),
            ("low_24h", "REAL"),
            ("high_7d", "REAL"),
            ("low_7d", "REAL"),
            ("high_30d", "REAL"),
            ("low_30d", "REAL"),
            ("pct_off_ath", "REAL"),
            ("pct_above_atl", "REAL"),
            ("seconds_since_ath", "INTEGER"),
            ("seconds_since_atl", "INTEGER"),
            ("mdd_all", "REAL"),
            ("mdd_peak_ts", "INTEGER"),
            ("mdd_trough_ts", "INTEGER"),
            ("mdd_30d", "REAL"),
            ("mdd_7d", "REAL"),
            ("mdd_24h", "REAL"),
        ],
        "market_regime": [
            ("breadth_pct_above_ema20", "REAL"),
            ("breadth_pct_above_ema50", "REAL"),
            ("breadth_pct_above_ema200", "REAL"),
            ("pct_with_positive_20d_return", "REAL"),
            ("new_20d_high_pct", "REAL"),
            ("new_20d_low_pct", "REAL"),
            ("median_volatility_20d", "REAL"),
            ("computed_ts", "INTEGER"),
        ],
        "market_features": [
            ("ema_200", "REAL"),
            ("trend_strength_200d", "REAL"),
            ("vol_percentile_252d", "REAL"),
            ("dist_from_52w_high", "REAL"),
            ("cont_prob_5d", "REAL"),
            ("exp_return_5d", "REAL"),
            ("cont_samples_5d", "INTEGER"),
            ("mom_strength", "REAL"),
        ],
    }

    with engine.begin() as conn:
        for table_name, columns in migrations.items():
            table_exists = conn.execute(
                text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": table_name},
            ).scalar_one_or_none()
            if not table_exists:
                continue

            current_cols = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            }
            for col_name, ddl in columns:
                if col_name in current_cols:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {ddl}"))

        # --- INTRADAY: intraday_bars model (v0.16.0) ---
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS intraday_bars ("
                "symbol TEXT NOT NULL,"
                "bar_ts DATETIME NOT NULL,"
                "open REAL NOT NULL,"
                "high REAL NOT NULL,"
                "low REAL NOT NULL,"
                "close REAL NOT NULL,"
                "volume REAL NOT NULL DEFAULT 0.0,"
                "PRIMARY KEY(symbol, bar_ts)"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_intraday_bars_symbol_ts ON intraday_bars(symbol, bar_ts DESC)"
            )
        )
        # --- END INTRADAY ---

        market_features_exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_features'")
        ).scalar_one_or_none()
        if market_features_exists:
            mf_cols = conn.execute(text("PRAGMA table_info(market_features)")).fetchall()
            mf_col_names = {str(row[1]) for row in mf_cols}
            mf_pk = [str(row[1]) for row in mf_cols if int(row[5] or 0) > 0]
            expected_pk = ["symbol", "timeframe", "bar_ts"]
            needs_rebuild = ("bar_ts" not in mf_col_names) or (mf_pk != expected_pk)

            if needs_rebuild:
                legacy_name = f"market_features_legacy_{int(time.time())}"
                conn.execute(text(f"ALTER TABLE market_features RENAME TO {legacy_name}"))
                conn.execute(
                    text(
                        "CREATE TABLE market_features ("
                        "symbol TEXT NOT NULL,"
                        "timeframe TEXT NOT NULL,"
                        "bar_ts INTEGER NOT NULL,"
                        "ema_20 REAL,"
                        "ema_50 REAL,"
                        "ema_200 REAL,"
                        "trend_slope_20d REAL,"
                        "trend_strength_200d REAL,"
                        "realized_vol_20d REAL,"
                        "vol_percentile_252d REAL,"
                        "dist_from_52w_high REAL,"
                        "drawdown_252d REAL,"
                        "return_1d REAL,"
                        "return_5d REAL,"
                        "return_20d REAL,"
                        "regime_tag TEXT,"
                        "cont_prob_5d REAL,"
                        "exp_return_5d REAL,"
                        "cont_samples_5d INTEGER,"
                        "mom_strength REAL,"
                        "PRIMARY KEY(symbol, timeframe, bar_ts)"
                        ")"
                    )
                )

                legacy_cols = {str(row[1]) for row in conn.execute(text(f"PRAGMA table_info({legacy_name})")).fetchall()}
                if {"symbol", "timeframe"}.issubset(legacy_cols):
                    source_bar_ts = "bar_ts" if "bar_ts" in legacy_cols else ("ts" if "ts" in legacy_cols else "CAST(strftime('%s','now') AS INTEGER)")
                    source_ema_20 = "ema_20" if "ema_20" in legacy_cols else ("ema_fast" if "ema_fast" in legacy_cols else "NULL")
                    source_ema_50 = "ema_50" if "ema_50" in legacy_cols else ("ema_slow" if "ema_slow" in legacy_cols else "NULL")
                    source_ema_200 = "ema_200" if "ema_200" in legacy_cols else "NULL"
                    source_slope = "trend_slope_20d" if "trend_slope_20d" in legacy_cols else ("trend_slope" if "trend_slope" in legacy_cols else "NULL")
                    source_trend_200 = "trend_strength_200d" if "trend_strength_200d" in legacy_cols else "NULL"
                    source_vol = "realized_vol_20d" if "realized_vol_20d" in legacy_cols else ("realized_vol" if "realized_vol" in legacy_cols else "NULL")
                    source_vol_pct = "vol_percentile_252d" if "vol_percentile_252d" in legacy_cols else "NULL"
                    source_dist_52w = "dist_from_52w_high" if "dist_from_52w_high" in legacy_cols else "NULL"
                    source_dd = "drawdown_252d" if "drawdown_252d" in legacy_cols else ("drawdown_30d" if "drawdown_30d" in legacy_cols else "NULL")
                    source_ret_1d = "return_1d" if "return_1d" in legacy_cols else "NULL"
                    source_ret_5d = "return_5d" if "return_5d" in legacy_cols else "NULL"
                    source_ret_20d = "return_20d" if "return_20d" in legacy_cols else "NULL"
                    source_regime = "regime_tag" if "regime_tag" in legacy_cols else "NULL"
                    source_cont_prob = "cont_prob_5d" if "cont_prob_5d" in legacy_cols else "NULL"
                    source_exp_return = "exp_return_5d" if "exp_return_5d" in legacy_cols else "NULL"
                    source_cont_samples = "cont_samples_5d" if "cont_samples_5d" in legacy_cols else "NULL"
                    source_mom_strength = "mom_strength" if "mom_strength" in legacy_cols else "NULL"
                    conn.execute(
                        text(
                            "INSERT OR IGNORE INTO market_features("
                            "symbol,timeframe,bar_ts,ema_20,ema_50,ema_200,trend_slope_20d,trend_strength_200d,realized_vol_20d,vol_percentile_252d,dist_from_52w_high,drawdown_252d,return_1d,return_5d,return_20d,regime_tag,cont_prob_5d,exp_return_5d,cont_samples_5d,mom_strength"
                            ") "
                            "SELECT "
                            f"symbol,timeframe,{source_bar_ts},{source_ema_20},{source_ema_50},{source_ema_200},{source_slope},{source_trend_200},{source_vol},{source_vol_pct},{source_dist_52w},{source_dd},{source_ret_1d},{source_ret_5d},{source_ret_20d},{source_regime},{source_cont_prob},{source_exp_return},{source_cont_samples},{source_mom_strength} "
                            f"FROM {legacy_name}"
                        )
                    )

                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_market_features_tf_bar_ts ON market_features(timeframe, bar_ts DESC)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_market_features_symbol_tf_bar_ts ON market_features(symbol, timeframe, bar_ts DESC)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_market_features_regime_bar_ts ON market_features(regime_tag, bar_ts DESC)"))

        market_regime_exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_regime'")
        ).scalar_one_or_none()
        if market_regime_exists:
            mr_cols = conn.execute(text("PRAGMA table_info(market_regime)")).fetchall()
            mr_col_names = {str(row[1]) for row in mr_cols}
            mr_pk = [str(row[1]) for row in mr_cols if int(row[5] or 0) > 0]
            expected_mr_pk = ["ts"]
            legacy_markers = {"breadth_pct_above_ema", "breadth_new_highs", "breadth_new_lows", "median_volatility", "cross_asset_correlation"}
            needs_regime_rebuild = (mr_pk != expected_mr_pk) or bool(mr_col_names.intersection(legacy_markers))

            if needs_regime_rebuild:
                legacy_name = f"market_regime_legacy_{int(time.time())}"
                conn.execute(text(f"ALTER TABLE market_regime RENAME TO {legacy_name}"))
                conn.execute(
                    text(
                        "CREATE TABLE market_regime ("
                        "ts INTEGER NOT NULL PRIMARY KEY,"
                        "breadth_pct_above_ema20 REAL,"
                        "breadth_pct_above_ema50 REAL,"
                        "breadth_pct_above_ema200 REAL,"
                        "pct_with_positive_20d_return REAL,"
                        "new_20d_high_pct REAL,"
                        "new_20d_low_pct REAL,"
                        "median_volatility_20d REAL,"
                        "regime_tag TEXT,"
                        "computed_ts INTEGER"
                        ")"
                    )
                )

                legacy_cols = {str(row[1]) for row in conn.execute(text(f"PRAGMA table_info({legacy_name})")).fetchall()}
                if "ts" in legacy_cols:
                    source_breadth_legacy = "breadth_pct_above_ema" if "breadth_pct_above_ema" in legacy_cols else "NULL"
                    source_breadth_20 = "breadth_pct_above_ema20" if "breadth_pct_above_ema20" in legacy_cols else (
                        "breadth_pct_above_ema50" if "breadth_pct_above_ema50" in legacy_cols else source_breadth_legacy
                    )
                    source_breadth_50 = "breadth_pct_above_ema50" if "breadth_pct_above_ema50" in legacy_cols else source_breadth_legacy
                    source_breadth_200 = "breadth_pct_above_ema200" if "breadth_pct_above_ema200" in legacy_cols else "NULL"
                    source_pct_pos_20d = "pct_with_positive_20d_return" if "pct_with_positive_20d_return" in legacy_cols else "NULL"
                    source_new_20d_high = "new_20d_high_pct" if "new_20d_high_pct" in legacy_cols else "NULL"
                    source_new_20d_low = "new_20d_low_pct" if "new_20d_low_pct" in legacy_cols else "NULL"
                    source_median_vol = "median_volatility_20d" if "median_volatility_20d" in legacy_cols else (
                        "median_volatility" if "median_volatility" in legacy_cols else "NULL"
                    )
                    source_regime = "regime_tag" if "regime_tag" in legacy_cols else "NULL"
                    source_computed = "computed_ts" if "computed_ts" in legacy_cols else "NULL"
                    conn.execute(
                        text(
                            "INSERT OR IGNORE INTO market_regime("
                            "ts,breadth_pct_above_ema20,breadth_pct_above_ema50,breadth_pct_above_ema200,pct_with_positive_20d_return,new_20d_high_pct,new_20d_low_pct,median_volatility_20d,regime_tag,computed_ts"
                            ") "
                            "SELECT "
                            f"ts,{source_breadth_20},{source_breadth_50},{source_breadth_200},{source_pct_pos_20d},{source_new_20d_high},{source_new_20d_low},{source_median_vol},{source_regime},{source_computed} "
                            f"FROM {legacy_name}"
                        )
                    )

                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_market_regime_tag_ts ON market_regime(regime_tag, ts DESC)"
                    )
                )

        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_alerts_item_type_ts ON alerts_sent(item_key, alert_type, ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_alerts_uuid ON alerts_sent(uuid)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_forecasts_item_ts ON item_forecasts(item_key, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_forecasts_ts ON item_forecasts(ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_item_forecast_key_ts_horizon_model ON item_forecasts(item_key, ts, horizon_minutes, model_name)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_nocom_model_state_hqmv ON nocom_model_state(horizon_minutes, quantile, model_name, feature_schema_version)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_model_state_hq ON nocom_model_state(horizon_minutes, quantile)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_nocom_train_pair_item_t0_h ON nocom_train_pairs(item_key, t0_bar_ts, horizon_minutes)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_pairs_horizon_t0 ON nocom_train_pairs(horizon_minutes, t0_bar_ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_pairs_item_horizon ON nocom_train_pairs(item_key, horizon_minutes)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_pairs_unlabeled ON nocom_train_pairs(horizon_minutes, y_realized)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_extremes_ts ON item_extremes(ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_auctions_canonical ON auctions(canonical_item_key)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_hist_canonical_ts ON auctions_history(canonical_item_key, ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_events_item_ts ON item_events(item_key, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_events_type_ts ON item_events(event_type, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedupe ON item_events(dedupe_key)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_favorites_created ON favorite_items(created_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_watch_targets_enabled ON watch_targets(enabled)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_watch_targets_type ON watch_targets(target_type)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_watch_targets_target_item_group ON watch_targets(target_type, item_key, group_kind, group_value)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_watch_targets_item ON watch_targets(target_type, item_key) WHERE target_type = 'ITEM'"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_watch_targets_group ON watch_targets(target_type, group_kind, group_value) WHERE target_type = 'GROUP'"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_meta_name ON item_key_meta(display_name)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_meta_base ON item_key_meta(base_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_symbols_exchange_active ON symbols(exchange, active)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_symbols_last_seen ON symbols(last_seen_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_bars_tf_ts ON market_bars(timeframe, bar_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_tf_ts ON market_bars(symbol, timeframe, bar_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_features_tf_bar_ts ON market_features(timeframe, bar_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_features_symbol_tf_bar_ts ON market_features(symbol, timeframe, bar_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_features_regime_bar_ts ON market_features(regime_tag, bar_ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_regime_tag_ts ON market_regime(regime_tag, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_events_ts ON market_events(ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_events_symbol_ts ON market_events(symbol, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_market_events_type_ts ON market_events(event_type, ts DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_market_events_dedupe ON market_events(dedupe_key)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_alerts_equities_type_symbol_ts ON alerts_sent_equities(alert_type, symbol, ts DESC)"
            )
        )

        # Backfill canonical/raw keys for pre-migration rows with best-effort fallback.
        conn.execute(
            text(
                "UPDATE auctions "
                "SET raw_item_key = COALESCE(raw_item_key, lower(item_name) || '|' || tier || '|1') "
                "WHERE raw_item_key IS NULL OR raw_item_key = ''"
            )
        )
        conn.execute(
            text(
                "UPDATE auctions "
                "SET canonical_item_key = COALESCE(canonical_item_key, lower(item_name) || '|' || tier || '|1') "
                "WHERE canonical_item_key IS NULL OR canonical_item_key = ''"
            )
        )
        conn.execute(
            text(
                "UPDATE auctions_history "
                "SET canonical_item_key = COALESCE(canonical_item_key, item_key) "
                "WHERE canonical_item_key IS NULL OR canonical_item_key = ''"
            )
        )
        conn.execute(
            text(
                "UPDATE auctions_history "
                "SET raw_item_key = COALESCE(raw_item_key, item_key) "
                "WHERE raw_item_key IS NULL OR raw_item_key = ''"
            )
        )
        conn.execute(
            text(
                "INSERT OR IGNORE INTO watch_targets(target_type, item_key, label, enabled, created_ts) "
                "SELECT 'ITEM', fi.item_key, fi.item_key, 1, COALESCE(fi.created_ts, CAST(strftime('%s','now') AS INTEGER)) "
                "FROM favorite_items fi "
                "WHERE fi.item_key IS NOT NULL AND TRIM(fi.item_key) <> ''"
            )
        )
        conn.execute(
            text(
                "INSERT OR REPLACE INTO item_key_meta(item_key, base_id, display_name, tier, category, last_seen_ts) "
                "SELECT a.canonical_item_key, NULLIF(a.item_id, ''), a.item_name, a.tier, a.category, a.ts "
                "FROM auctions a "
                "JOIN ("
                "  SELECT canonical_item_key, MAX(ts) AS max_ts "
                "  FROM auctions "
                "  WHERE canonical_item_key IS NOT NULL AND canonical_item_key <> '' "
                "  GROUP BY canonical_item_key"
                ") latest "
                "ON latest.canonical_item_key = a.canonical_item_key AND latest.max_ts = a.ts "
                "WHERE a.canonical_item_key IS NOT NULL AND a.canonical_item_key <> ''"
            )
        )


def init_db(db_url: str | None = None) -> None:
    global _ENGINE, SessionLocal
    if _ENGINE is not None:
        return

    _ENGINE = create_engine(
        db_url or get_default_db_url(),
        connect_args={"check_same_thread": False},
        future=True,
    )
    SessionLocal = scoped_session(
        sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, expire_on_commit=False)
    )

    Base.metadata.create_all(bind=_ENGINE)
    _ensure_sqlite_columns(_ENGINE)


def get_session():
    if SessionLocal is None:
        init_db()
    return SessionLocal()


@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def normalize_item_name(item_name: str | None) -> str:
    cleaned = MC_FORMAT_CODE_RE.sub("", item_name or "")
    cleaned = cleaned.replace("§", "")
    cleaned = WS_RE.sub(" ", cleaned).strip()
    return cleaned or "unknown"


def normalize_item_key(item_name: str | None, tier: str | None, bin_flag: int | bool) -> str:
    safe_name = normalize_item_name(item_name).lower()
    safe_tier = (tier or "COMMON").strip().upper()
    bin_bit = 1 if bool(bin_flag) else 0
    return f"{safe_name}|{safe_tier}|{bin_bit}"


def display_name_from_canonical_key(item_key: str | None, fallback_name: str | None = None) -> str:
    canonical = str(item_key or "").strip()
    base = canonical.split("|", 1)[0].strip() if canonical else ""
    if base.upper().startswith("PET_"):
        base = base[4:]
    text = base.replace("_", " ").strip()
    if text and text.upper() != "UNKNOWN":
        return text.title()

    fallback = normalize_item_name(fallback_name or "").strip()
    return fallback if fallback else "unknown"


def extract_price(auction: dict) -> int:
    if int(bool(auction.get("bin", False))) == 1:
        return int(auction.get("starting_bid") or 0)
    highest = auction.get("highest_bid_amount")
    if highest is None:
        highest = auction.get("starting_bid") or 0
    return int(highest)


def _auction_row(auction: dict, ts: int) -> dict | None:
    uuid = auction.get("uuid")
    if not uuid:
        return None

    item_name = normalize_item_name(auction.get("item_name"))
    tier = (auction.get("tier") or "COMMON").strip().upper()
    start_bid = int(auction.get("starting_bid") or 0)
    highest_bid = int(auction.get("highest_bid_amount") or start_bid)
    bin_flag = int(bool(auction.get("bin", False)))
    price = start_bid if bin_flag == 1 else (highest_bid or start_bid)
    canonical = canonicalize_auction(auction)

    return {
        "uuid": uuid,
        "ts": ts,
        "item_name": item_name,
        "item_id": auction.get("item_id"),
        "tier": tier,
        "category": auction.get("category"),
        "starting_bid": start_bid,
        "highest_bid_amount": highest_bid,
        "bin": bin_flag,
        "bids_count": len(auction.get("bids", [])),
        "end_ts": int((auction.get("end") or 0) / 1000),
        "seller": auction.get("auctioneer") or auction.get("seller"),
        "price": int(price),
        "last_updated_ts": ts,
        "raw_item_key": canonical.get("raw_item_key"),
        "canonical_item_key": canonical.get("canonical_item_key"),
    }


def upsert_item_key_meta(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    deduped_by_key: dict[str, dict] = {}
    for row in rows:
        item_key = str(row.get("item_key") or "").strip()
        if not item_key:
            continue
        prior = deduped_by_key.get(item_key)
        if prior is None or int(row.get("last_seen_ts") or 0) >= int(prior.get("last_seen_ts") or 0):
            deduped_by_key[item_key] = {
                "item_key": item_key,
                "base_id": (str(row.get("base_id")).strip().upper() if row.get("base_id") else None),
                "display_name": str(row.get("display_name") or item_key),
                "tier": (str(row.get("tier")).strip().upper() if row.get("tier") else None),
                "category": (str(row.get("category")).strip().upper() if row.get("category") else None),
                "last_seen_ts": int(row.get("last_seen_ts") or 0),
            }

    payload = list(deduped_by_key.values())
    if not payload:
        return 0

    insert_stmt = sqlite_insert(ItemKeyMeta)
    update_columns = {
        "base_id": insert_stmt.excluded.base_id,
        "display_name": insert_stmt.excluded.display_name,
        "tier": insert_stmt.excluded.tier,
        "category": insert_stmt.excluded.category,
        "last_seen_ts": insert_stmt.excluded.last_seen_ts,
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[ItemKeyMeta.item_key],
        set_=update_columns,
    )
    for i in range(0, len(payload), BATCH_SIZE):
        session.execute(upsert_stmt, payload[i:i + BATCH_SIZE])
    return len(payload)


def upsert_auctions(
    session,
    auction_payloads: list[dict],
    *,
    ts: int,
    full_scan: bool,
    lookback_hours: int,
) -> dict:
    rows: list[dict] = []
    history_rows: list[dict] = []
    meta_rows: list[dict] = []
    now_ts = int(time.time())

    for payload in auction_payloads:
        row = _auction_row(payload, ts)
        if not row:
            continue

        rows.append(row)

        if row["bin"] == 1:
            canonical_key = str(row.get("canonical_item_key") or normalize_item_key(row["item_name"], row["tier"], 1))
            raw_key = str(row.get("raw_item_key") or normalize_item_key(row["item_name"], row["tier"], 1))
            history_rows.append(
                {
                    "ts": ts,
                    "item_key": canonical_key,
                    "item_name": row["item_name"],
                    "tier": row["tier"],
                    "bin": 1,
                    "price": int(row["price"]),
                    "uuid": row["uuid"],
                    "raw_item_key": raw_key,
                    "canonical_item_key": canonical_key,
                }
            )

        canonical_key = str(row.get("canonical_item_key") or normalize_item_key(row["item_name"], row["tier"], 1))
        base_id = str(row.get("item_id") or "").strip().upper() or None
        if not base_id:
            base_id = canonical_key.split("|", 1)[0].strip().upper() or None
        meta_rows.append(
            {
                "item_key": canonical_key,
                "base_id": base_id,
                "display_name": display_name_from_canonical_key(canonical_key, row["item_name"]),
                "tier": row["tier"],
                "category": row.get("category"),
                "last_seen_ts": ts,
            }
        )

    if full_scan:
        session.execute(delete(Auction))

    auctions_total = len(rows)
    auctions_batches = (auctions_total + BATCH_SIZE - 1) // BATCH_SIZE if auctions_total else 0
    if rows:
        insert_stmt = sqlite_insert(Auction)
        update_columns = {
            c.name: insert_stmt.excluded[c.name]
            for c in Auction.__table__.columns
            if c.name != "uuid"
        }
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[Auction.uuid],
            set_=update_columns,
        )
        for i in range(0, auctions_total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.execute(upsert_stmt, batch)

    history_total = len(history_rows)
    history_batches = (history_total + BATCH_SIZE - 1) // BATCH_SIZE if history_total else 0
    if history_rows:
        history_insert = sqlite_insert(AuctionHistory).on_conflict_do_nothing(
            index_elements=[AuctionHistory.uuid]
        )
        for i in range(0, history_total, BATCH_SIZE):
            batch = history_rows[i:i + BATCH_SIZE]
            session.execute(history_insert, batch)

    meta_total = upsert_item_key_meta(session, meta_rows)

    session.execute(delete(Auction).where(Auction.end_ts < now_ts - 60))

    retention_cutoff = ts - (max(lookback_hours, 1) * 3600)
    session.execute(delete(AuctionHistory).where(AuctionHistory.ts < retention_cutoff))

    logger.debug(
        "DB upsert: auctions total=%s batches=%s batch_size=%s",
        auctions_total,
        auctions_batches,
        BATCH_SIZE,
    )
    logger.debug(
        "DB upsert: history total=%s batches=%s batch_size=%s",
        history_total,
        history_batches,
        BATCH_SIZE,
    )

    return {
        "auctions_total": auctions_total,
        "history_total": history_total,
        "meta_total": meta_total,
    }


def snapshot_item_spreads(
    session,
    *,
    ts: int,
    lookback_hours: int,
    top_n: int,
) -> dict:
    capped_top_n = max(int(top_n), 1)
    fallback_expr = (func.lower(Auction.item_name) + literal("|") + Auction.tier + literal("|1"))
    item_key_expr = func.coalesce(Auction.canonical_item_key, fallback_expr).label("item_key")

    active_rows = session.execute(
        select(item_key_expr, func.count(Auction.uuid).label("active_listings"))
        .where(Auction.bin == 1)
        .where(Auction.end_ts > ts)
        .where(item_key_expr.is_not(None))
        .group_by(item_key_expr)
        .order_by(desc("active_listings"))
        .limit(capped_top_n)
    ).all()

    if not active_rows:
        retention_cutoff = ts - (max(lookback_hours, 1) * 3600)
        session.execute(delete(ItemSpreadSnapshot).where(ItemSpreadSnapshot.ts < retention_cutoff))
        logger.debug("Spread snapshots: keys=%s rows=%s", 0, 0)
        return {"keys": 0, "rows": 0}

    selected_keys = [str(row.item_key) for row in active_rows if row.item_key]
    active_map = {str(row.item_key): int(row.active_listings or 0) for row in active_rows if row.item_key}

    price_rows = session.execute(
        select(item_key_expr, Auction.price)
        .where(Auction.bin == 1)
        .where(Auction.end_ts > ts)
        .where(item_key_expr.in_(selected_keys))
        .order_by(item_key_expr.asc(), Auction.price.asc())
    ).all()

    top_prices: dict[str, list[int]] = {}
    for item_key, price in price_rows:
        key = str(item_key)
        bucket = top_prices.setdefault(key, [])
        if len(bucket) < 2:
            bucket.append(int(price or 0))

    rows: list[dict] = []
    for item_key in selected_keys:
        prices = top_prices.get(item_key, [])
        if not prices:
            continue

        best_bin = int(prices[0])
        second_best_bin = int(prices[1]) if len(prices) > 1 else None
        spread_abs = (second_best_bin - best_bin) if second_best_bin is not None else None
        spread_pct = (float(spread_abs) / max(best_bin, 1)) if spread_abs is not None else None

        rows.append(
            {
                "ts": ts,
                "item_key": item_key,
                "best_bin": best_bin,
                "second_best_bin": second_best_bin,
                "spread_abs": spread_abs,
                "spread_pct": spread_pct,
                "active_listings": int(active_map.get(item_key, 0)),
            }
        )

    if rows:
        insert_stmt = sqlite_insert(ItemSpreadSnapshot)
        for i in range(0, len(rows), BATCH_SIZE):
            session.execute(insert_stmt, rows[i:i + BATCH_SIZE])

    retention_cutoff = ts - (max(lookback_hours, 1) * 3600)
    session.execute(delete(ItemSpreadSnapshot).where(ItemSpreadSnapshot.ts < retention_cutoff))

    logger.debug("Spread snapshots: keys=%s rows=%s", len(selected_keys), len(rows))
    return {"keys": len(selected_keys), "rows": len(rows)}


def upsert_item_bars(session, rows: list[dict], *, lookback_hours: int, now_ts: int) -> int:
    total = len(rows)
    if rows:
        insert_stmt = sqlite_insert(ItemBar)
        update_columns = {
            "n": insert_stmt.excluded.n,
            "price_median": insert_stmt.excluded.price_median,
            "price_min": insert_stmt.excluded.price_min,
            "price_max": insert_stmt.excluded.price_max,
        }
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[ItemBar.item_key, ItemBar.bar_ts],
            set_=update_columns,
        )
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.execute(upsert_stmt, batch)

    cutoff = now_ts - (max(lookback_hours, 1) * 3600)
    session.execute(delete(ItemBar).where(ItemBar.bar_ts < cutoff))
    return total


def upsert_item_features(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(ItemFeature)
    update_columns = {
        c.name: insert_stmt.excluded[c.name]
        for c in ItemFeature.__table__.columns
        if c.name != "item_key"
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[ItemFeature.item_key],
        set_=update_columns,
    )
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(upsert_stmt, batch)
    return total


def upsert_recommendations(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(Recommendation)
    update_columns = {
        c.name: insert_stmt.excluded[c.name]
        for c in Recommendation.__table__.columns
        if c.name != "item_key"
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[Recommendation.item_key],
        set_=update_columns,
    )
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(upsert_stmt, batch)
    return total


def upsert_item_extremes(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(ItemExtreme)
    update_columns = {
        c.name: insert_stmt.excluded[c.name]
        for c in ItemExtreme.__table__.columns
        if c.name != "item_key"
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[ItemExtreme.item_key],
        set_=update_columns,
    )
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(upsert_stmt, batch)
    return total


def save_backtest_results(session, rows: list[dict], *, now_ts: int) -> int:
    total = len(rows)
    if not rows:
        return 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(sqlite_insert(BacktestResult), batch)

    # Keep backtest table bounded while preserving transparency.
    cutoff = now_ts - 7 * 24 * 3600
    session.execute(delete(BacktestResult).where(BacktestResult.ts < cutoff))
    return total


def get_open_positions(session) -> list[PortfolioPosition]:
    return (
        session.execute(
            select(PortfolioPosition)
            .where(PortfolioPosition.status == "OPEN")
            .order_by(PortfolioPosition.opened_ts.asc())
        )
        .scalars()
        .all()
    )


def get_open_position_map(session) -> dict[str, PortfolioPosition]:
    rows = get_open_positions(session)
    return {row.item_key: row for row in rows}


def open_position(session, *, item_key: str, opened_ts: int, entry_price: float, qty: int = 1) -> PortfolioPosition:
    pos = PortfolioPosition(
        item_key=item_key,
        opened_ts=opened_ts,
        entry_price=float(entry_price),
        qty=max(int(qty), 1),
        status="OPEN",
    )
    session.add(pos)
    session.flush()
    return pos


def close_position(
    session,
    position: PortfolioPosition,
    *,
    closed_ts: int,
    exit_price: float,
    fee_pct: float,
) -> PortfolioPosition:
    epsilon = 1e-9
    qty = max(int(position.qty), 1)
    entry_price = float(position.entry_price)
    close_ts = int(closed_ts)
    exit_price_value = float(exit_price)

    position.status = "CLOSED"
    position.closed_ts = close_ts
    position.exit_price = exit_price_value

    gross_entry = entry_price * qty
    gross_exit = exit_price_value * qty
    exit_fee = gross_exit * max(float(fee_pct), 0.0)
    pnl_after_fee = gross_exit - gross_entry - exit_fee

    duration_seconds = max(close_ts - int(position.opened_ts), 0)
    duration_hours = max(duration_seconds / 3600.0, epsilon)

    position.duration_seconds = duration_seconds
    position.pnl_after_fee = pnl_after_fee
    position.return_pct = pnl_after_fee / max(gross_entry, epsilon)
    position.profit_per_hour = pnl_after_fee / duration_hours
    position.pnl = pnl_after_fee
    session.add(position)
    return position


def insert_alerts_sent(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(AlertSent)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(insert_stmt, batch)
    return total


def insert_item_events(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(ItemEvent).prefix_with("OR IGNORE")
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.execute(insert_stmt, batch)
    return total


def get_recent_events(
    session,
    *,
    limit: int = 200,
    event_type: str | None = None,
    item_key: str | None = None,
) -> list[ItemEvent]:
    bounded_limit = min(max(int(limit), 1), 2000)
    query = select(ItemEvent)
    if event_type:
        query = query.where(ItemEvent.event_type == str(event_type))
    if item_key:
        query = query.where(ItemEvent.item_key == str(item_key))
    return (
        session.execute(query.order_by(ItemEvent.ts.desc(), ItemEvent.id.desc()).limit(bounded_limit))
        .scalars()
        .all()
    )


def get_favorites(session, *, limit: int = 500) -> list[FavoriteItem]:
    bounded_limit = min(max(int(limit), 1), 5003)
    return (
        session.execute(
            select(FavoriteItem)
            .order_by(FavoriteItem.pinned.desc(), FavoriteItem.created_ts.desc())
            .limit(bounded_limit)
        )
        .scalars()
        .all()
    )


WATCH_TARGET_ITEM = "ITEM"
WATCH_TARGET_GROUP = "GROUP"
WATCH_GROUP_KIND_NAME_PREFIX = "NAME_PREFIX"
WATCH_GROUP_KIND_BASE_ID = "BASE_ID"
SUPPORTED_WATCH_GROUP_KINDS = {WATCH_GROUP_KIND_NAME_PREFIX, WATCH_GROUP_KIND_BASE_ID}


def normalize_watch_target_type(raw: str | None) -> str:
    value = str(raw or WATCH_TARGET_ITEM).strip().upper()
    if value not in {WATCH_TARGET_ITEM, WATCH_TARGET_GROUP}:
        return WATCH_TARGET_ITEM
    return value


def normalize_watch_group_kind(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().upper()
    if not value:
        return None
    if value not in SUPPORTED_WATCH_GROUP_KINDS:
        return None
    return value


def get_watch_targets(
    session,
    *,
    limit: int = 500,
    enabled_only: bool = False,
) -> list[WatchTarget]:
    bounded_limit = min(max(int(limit), 1), 5003)
    query = select(WatchTarget)
    if enabled_only:
        query = query.where(WatchTarget.enabled == 1)
    return (
        session.execute(
            query.order_by(WatchTarget.enabled.desc(), WatchTarget.created_ts.desc(), WatchTarget.id.desc()).limit(
                bounded_limit
            )
        )
        .scalars()
        .all()
    )


def get_watch_target_by_id(session, *, target_id: int) -> WatchTarget | None:
    return session.get(WatchTarget, int(target_id))


def get_item_watch_target(session, *, item_key: str) -> WatchTarget | None:
    normalized = str(item_key or "").strip()
    if not normalized:
        return None
    return (
        session.execute(
            select(WatchTarget)
            .where(WatchTarget.target_type == WATCH_TARGET_ITEM)
            .where(WatchTarget.item_key == normalized)
            .order_by(WatchTarget.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )


def add_watch_item_target(
    session,
    *,
    item_key: str,
    label: str | None,
    created_ts: int,
    enabled: int = 1,
) -> WatchTarget:
    normalized_key = str(item_key or "").strip()
    if not normalized_key:
        raise ValueError("item_key is required")

    existing = get_item_watch_target(session, item_key=normalized_key)
    if existing is not None:
        if enabled in (0, 1) and int(existing.enabled) != int(enabled):
            existing.enabled = int(enabled)
            session.add(existing)
        return existing

    target = WatchTarget(
        target_type=WATCH_TARGET_ITEM,
        item_key=normalized_key,
        label=(str(label or "").strip() or normalized_key),
        group_kind=None,
        group_value=None,
        enabled=1 if int(enabled) else 0,
        created_ts=int(created_ts),
    )
    session.add(target)
    session.flush()
    return target


def add_watch_group_target(
    session,
    *,
    label: str,
    group_kind: str,
    group_value: str,
    created_ts: int,
    enabled: int = 1,
) -> WatchTarget:
    normalized_kind = normalize_watch_group_kind(group_kind)
    if normalized_kind is None:
        raise ValueError("Unsupported group_kind")
    normalized_value = str(group_value or "").strip()
    if not normalized_value:
        raise ValueError("group_value is required")
    safe_label = str(label or "").strip() or f"{normalized_kind}:{normalized_value}"

    existing = (
        session.execute(
            select(WatchTarget)
            .where(WatchTarget.target_type == WATCH_TARGET_GROUP)
            .where(WatchTarget.group_kind == normalized_kind)
            .where(WatchTarget.group_value == normalized_value)
            .order_by(WatchTarget.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if existing is not None:
        if safe_label and existing.label != safe_label:
            existing.label = safe_label
        if enabled in (0, 1) and int(existing.enabled) != int(enabled):
            existing.enabled = int(enabled)
        session.add(existing)
        return existing

    target = WatchTarget(
        target_type=WATCH_TARGET_GROUP,
        item_key=None,
        label=safe_label,
        group_kind=normalized_kind,
        group_value=normalized_value,
        enabled=1 if int(enabled) else 0,
        created_ts=int(created_ts),
    )
    session.add(target)
    session.flush()
    return target


def remove_watch_target(session, *, target_id: int) -> int:
    result = session.execute(delete(WatchTarget).where(WatchTarget.id == int(target_id)))
    return int(result.rowcount or 0)


def set_watch_target_enabled(session, *, target_id: int, enabled: int) -> bool:
    result = session.execute(
        update(WatchTarget)
        .where(WatchTarget.id == int(target_id))
        .values(enabled=1 if int(enabled) else 0)
    )
    return int(result.rowcount or 0) > 0


def _expand_name_prefix_keys(session, *, prefix: str, limit: int) -> list[str]:
    normalized_prefix = str(prefix or "").strip().lower()
    if not normalized_prefix:
        return []

    meta_rows = session.execute(
        select(ItemKeyMeta.item_key)
        .where(ItemKeyMeta.display_name.is_not(None))
        .where(func.lower(ItemKeyMeta.display_name).like(f"{normalized_prefix}%"))
        .order_by(ItemKeyMeta.last_seen_ts.desc())
        .limit(limit)
    ).all()
    keys = [str(row.item_key) for row in meta_rows if row.item_key]
    if keys:
        return keys

    fallback_rows = session.execute(
        select(Auction.canonical_item_key)
        .where(Auction.canonical_item_key.is_not(None))
        .where(func.lower(Auction.item_name).like(f"{normalized_prefix}%"))
        .group_by(Auction.canonical_item_key)
        .order_by(func.max(Auction.ts).desc())
        .limit(limit)
    ).all()
    return [str(row.canonical_item_key) for row in fallback_rows if row.canonical_item_key]


def _expand_base_id_keys(session, *, base_id: str, limit: int) -> list[str]:
    normalized_base_id = str(base_id or "").strip().upper()
    if not normalized_base_id:
        return []

    meta_rows = session.execute(
        select(ItemKeyMeta.item_key)
        .where(ItemKeyMeta.base_id.is_not(None))
        .where(func.upper(ItemKeyMeta.base_id) == normalized_base_id)
        .order_by(ItemKeyMeta.last_seen_ts.desc())
        .limit(limit)
    ).all()
    keys = [str(row.item_key) for row in meta_rows if row.item_key]
    if keys:
        return keys

    fallback_rows = session.execute(
        select(Auction.canonical_item_key)
        .where(Auction.canonical_item_key.is_not(None))
        .where(Auction.item_id.is_not(None))
        .where(func.upper(Auction.item_id) == normalized_base_id)
        .group_by(Auction.canonical_item_key)
        .order_by(func.max(Auction.ts).desc())
        .limit(limit)
    ).all()
    return [str(row.canonical_item_key) for row in fallback_rows if row.canonical_item_key]


def expand_watch_target_to_item_keys(
    session,
    target: WatchTarget,
    *,
    limit: int = 2000,
) -> list[str]:
    bounded_limit = min(max(int(limit), 1), 20_000)
    if target is None:
        return []

    if normalize_watch_target_type(target.target_type) == WATCH_TARGET_ITEM:
        key = str(target.item_key or "").strip()
        return [key] if key else []

    group_kind = normalize_watch_group_kind(target.group_kind)
    if group_kind == WATCH_GROUP_KIND_NAME_PREFIX:
        return _expand_name_prefix_keys(
            session,
            prefix=str(target.group_value or ""),
            limit=bounded_limit,
        )
    if group_kind == WATCH_GROUP_KIND_BASE_ID:
        return _expand_base_id_keys(
            session,
            base_id=str(target.group_value or ""),
            limit=bounded_limit,
        )
    return []


def add_favorite_item(
    session,
    *,
    item_key: str,
    created_ts: int,
    note: str | None = None,
    pinned: int = 0,
) -> None:
    add_watch_item_target(
        session,
        item_key=str(item_key),
        label=(note or str(item_key)),
        created_ts=int(created_ts),
        enabled=1,
    )


def remove_favorite_item(session, *, item_key: str) -> int:
    target = get_item_watch_target(session, item_key=str(item_key))
    if target is None:
        return 0
    return remove_watch_target(session, target_id=int(target.id))


def get_watchlist_rows(session, *, limit: int = 500) -> list[dict]:
    targets = get_watch_targets(session, limit=limit, enabled_only=False)
    if not targets:
        return []

    expanded_by_target: dict[int, list[str]] = {}
    all_item_keys: set[str] = set()
    for target in targets:
        keys = expand_watch_target_to_item_keys(session, target, limit=2000)
        deduped_keys = list(dict.fromkeys([str(key) for key in keys if key]))
        expanded_by_target[int(target.id)] = deduped_keys
        all_item_keys.update(deduped_keys)

    feature_map: dict[str, ItemFeature] = {}
    extreme_map: dict[str, ItemExtreme] = {}
    if all_item_keys:
        feature_rows = (
            session.execute(select(ItemFeature).where(ItemFeature.item_key.in_(all_item_keys))).scalars().all()
        )
        feature_map = {str(row.item_key): row for row in feature_rows}
        extreme_rows = (
            session.execute(select(ItemExtreme).where(ItemExtreme.item_key.in_(all_item_keys))).scalars().all()
        )
        extreme_map = {str(row.item_key): row for row in extreme_rows}

    out: list[dict] = []
    for target in targets:
        target_type = normalize_watch_target_type(target.target_type)
        expanded_keys = expanded_by_target.get(int(target.id), [])
        sample_key = str(target.item_key or "") if target_type == WATCH_TARGET_ITEM else ""
        if not sample_key:
            if expanded_keys:
                sample_key = max(
                    expanded_keys,
                    key=lambda key: float((feature_map.get(key).liquidity_rank if feature_map.get(key) else 0.0) or 0.0),
                )
            else:
                sample_key = ""

        feature = feature_map.get(sample_key) if sample_key else None
        extreme = extreme_map.get(sample_key) if sample_key else None
        out.append(
            {
                "id": int(target.id),
                "target_type": target_type,
                "item_key": (str(target.item_key) if target.item_key else None),
                "label": str(target.label or ""),
                "group_kind": (str(target.group_kind) if target.group_kind else None),
                "group_value": (str(target.group_value) if target.group_value else None),
                "enabled": int(target.enabled or 0),
                "created_ts": int(target.created_ts),
                "matched_count": (1 if target_type == WATCH_TARGET_ITEM and target.item_key else len(expanded_keys)),
                "sample_item_key": (sample_key or None),
                "price_now": (float(feature.price_now) if feature and feature.price_now is not None else None),
                "liquidity_rank": (
                    float(feature.liquidity_rank) if feature and feature.liquidity_rank is not None else None
                ),
                "regime_tag": (feature.regime_tag if feature else None),
                "pct_off_ath": (float(extreme.pct_off_ath) if extreme and extreme.pct_off_ath is not None else None),
                "mdd_30d": (float(extreme.mdd_30d) if extreme and extreme.mdd_30d is not None else None),
            }
        )
    return out


def get_latest_alert_ts_map(session, *, alert_type: str, item_keys: set[str]) -> dict[str, int]:
    if not item_keys:
        return {}
    rows = session.execute(
        select(AlertSent.item_key, func.max(AlertSent.ts))
        .where(AlertSent.alert_type == alert_type)
        .where(AlertSent.item_key.in_(item_keys))
        .group_by(AlertSent.item_key)
    ).all()
    return {str(item_key): int(latest_ts) for item_key, latest_ts in rows if latest_ts is not None}


def get_existing_alert_uuids(session, uuids: set[str]) -> set[str]:
    if not uuids:
        return set()
    rows = session.execute(
        select(AlertSent.uuid)
        .where(AlertSent.uuid.is_not(None))
        .where(AlertSent.uuid.in_(uuids))
    ).all()
    return {str(row[0]) for row in rows if row[0]}


def get_recent_alerts(session, *, limit: int = 50) -> list[AlertSent]:
    bounded_limit = min(max(int(limit), 1), 500)
    return (
        session.execute(select(AlertSent).order_by(AlertSent.ts.desc(), AlertSent.id.desc()).limit(bounded_limit))
        .scalars()
        .all()
    )


def upsert_symbols(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    deduped: dict[str, dict] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        previous = deduped.get(symbol)
        if previous is None or int(row.get("last_seen_ts") or 0) >= int(previous.get("last_seen_ts") or 0):
            deduped[symbol] = {
                "symbol": symbol,
                "exchange": (str(row.get("exchange")).strip().upper() if row.get("exchange") else None),
                "asset_type": (str(row.get("asset_type")).strip().upper() if row.get("asset_type") else None),
                "active": 1 if int(row.get("active") or 0) else 0,
                "last_seen_ts": int(row.get("last_seen_ts") or 0),
            }

    payload = list(deduped.values())
    if not payload:
        return 0

    insert_stmt = sqlite_insert(Symbol)
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[Symbol.symbol],
        set_={
            "exchange": insert_stmt.excluded.exchange,
            "asset_type": insert_stmt.excluded.asset_type,
            "active": insert_stmt.excluded.active,
            "last_seen_ts": insert_stmt.excluded.last_seen_ts,
        },
    )
    for i in range(0, len(payload), BATCH_SIZE):
        session.execute(upsert_stmt, payload[i:i + BATCH_SIZE])
    return len(payload)


def get_active_symbols(session, *, limit: int = 5003) -> list[str]:
    bounded_limit = min(max(int(limit), 1), 20_000)
    rows = (
        session.execute(
            select(Symbol.symbol)
            .where(Symbol.active == 1)
            .order_by(Symbol.last_seen_ts.desc(), Symbol.symbol.asc())
            .limit(bounded_limit)
        )
        .scalars()
        .all()
    )
    return [str(symbol).upper() for symbol in rows if symbol]


def upsert_market_bars(
    session,
    rows: list[dict],
    *,
    now_ts: int,
    intraday_lookback_days: int = 30,
    daily_lookback_days: int = 730,
    prune_old: bool = True,
) -> int:
    total = len(rows)
    if rows:
        insert_stmt = sqlite_insert(MarketBar)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[MarketBar.symbol, MarketBar.timeframe, MarketBar.bar_ts],
            set_={
                "open": insert_stmt.excluded.open,
                "high": insert_stmt.excluded.high,
                "low": insert_stmt.excluded.low,
                "close": insert_stmt.excluded.close,
                "volume": insert_stmt.excluded.volume,
            },
        )
        for i in range(0, total, BATCH_SIZE):
            session.execute(upsert_stmt, rows[i:i + BATCH_SIZE])

    if prune_old:
        intraday_cutoff = int(now_ts) - (max(int(intraday_lookback_days), 1) * 86_400)
        daily_cutoff = int(now_ts) - (max(int(daily_lookback_days), 30) * 86_400)
        session.execute(
            delete(MarketBar)
            .where(MarketBar.timeframe.in_(["1m", "5m"]))
            .where(MarketBar.bar_ts < intraday_cutoff)
        )
        session.execute(
            delete(MarketBar)
            .where(MarketBar.timeframe == "1d")
            .where(MarketBar.bar_ts < daily_cutoff)
        )

    return total


def upsert_market_features(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0

    insert_stmt = sqlite_insert(MarketFeature)
    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[MarketFeature.symbol, MarketFeature.timeframe, MarketFeature.bar_ts],
        set_={
            "ema_20": insert_stmt.excluded.ema_20,
            "ema_50": insert_stmt.excluded.ema_50,
            "ema_200": insert_stmt.excluded.ema_200,
            "trend_slope_20d": insert_stmt.excluded.trend_slope_20d,
            "trend_strength_200d": insert_stmt.excluded.trend_strength_200d,
            "realized_vol_20d": insert_stmt.excluded.realized_vol_20d,
            "vol_percentile_252d": insert_stmt.excluded.vol_percentile_252d,
            "dist_from_52w_high": insert_stmt.excluded.dist_from_52w_high,
            "drawdown_252d": insert_stmt.excluded.drawdown_252d,
            "return_1d": insert_stmt.excluded.return_1d,
            "return_5d": insert_stmt.excluded.return_5d,
            "return_20d": insert_stmt.excluded.return_20d,
            "regime_tag": insert_stmt.excluded.regime_tag,
            "cont_prob_5d": insert_stmt.excluded.cont_prob_5d,
            "exp_return_5d": insert_stmt.excluded.exp_return_5d,
            "cont_samples_5d": insert_stmt.excluded.cont_samples_5d,
            "mom_strength": insert_stmt.excluded.mom_strength,
        },
    )
    for i in range(0, total, BATCH_SIZE):
        session.execute(upsert_stmt, rows[i:i + BATCH_SIZE])
    return total


def insert_market_regime(session, row: dict) -> None:
    stmt = sqlite_insert(MarketRegime).values(
        ts=int(row.get("ts") or 0),
        breadth_pct_above_ema20=(
            float(row.get("breadth_pct_above_ema20")) if row.get("breadth_pct_above_ema20") is not None else None
        ),
        breadth_pct_above_ema50=(
            float(row.get("breadth_pct_above_ema50")) if row.get("breadth_pct_above_ema50") is not None else None
        ),
        breadth_pct_above_ema200=(
            float(row.get("breadth_pct_above_ema200")) if row.get("breadth_pct_above_ema200") is not None else None
        ),
        pct_with_positive_20d_return=(
            float(row.get("pct_with_positive_20d_return"))
            if row.get("pct_with_positive_20d_return") is not None
            else None
        ),
        new_20d_high_pct=(
            float(row.get("new_20d_high_pct")) if row.get("new_20d_high_pct") is not None else None
        ),
        new_20d_low_pct=(
            float(row.get("new_20d_low_pct")) if row.get("new_20d_low_pct") is not None else None
        ),
        median_volatility_20d=(
            float(row.get("median_volatility_20d")) if row.get("median_volatility_20d") is not None else None
        ),
        regime_tag=(str(row.get("regime_tag")) if row.get("regime_tag") else None),
        computed_ts=(int(row.get("computed_ts")) if row.get("computed_ts") is not None else None),
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[MarketRegime.ts],
            set_={
                "breadth_pct_above_ema20": stmt.excluded.breadth_pct_above_ema20,
                "breadth_pct_above_ema50": stmt.excluded.breadth_pct_above_ema50,
                "breadth_pct_above_ema200": stmt.excluded.breadth_pct_above_ema200,
                "pct_with_positive_20d_return": stmt.excluded.pct_with_positive_20d_return,
                "new_20d_high_pct": stmt.excluded.new_20d_high_pct,
                "new_20d_low_pct": stmt.excluded.new_20d_low_pct,
                "median_volatility_20d": stmt.excluded.median_volatility_20d,
                "regime_tag": stmt.excluded.regime_tag,
                "computed_ts": stmt.excluded.computed_ts,
            },
        )
    )


def insert_market_events(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0
    stmt = sqlite_insert(MarketEvent).prefix_with("OR IGNORE")
    for i in range(0, total, BATCH_SIZE):
        session.execute(stmt, rows[i:i + BATCH_SIZE])
    return total


def insert_alerts_sent_equities(session, rows: list[dict]) -> int:
    total = len(rows)
    if not rows:
        return 0
    stmt = sqlite_insert(AlertSentEquity)
    for i in range(0, total, BATCH_SIZE):
        session.execute(stmt, rows[i:i + BATCH_SIZE])
    return total


def get_recent_market_events(
    session,
    *,
    limit: int = 200,
    symbol: str | None = None,
    event_type: str | None = None,
) -> list[MarketEvent]:
    bounded_limit = min(max(int(limit), 1), 5003)
    query = select(MarketEvent)
    if symbol:
        query = query.where(MarketEvent.symbol == str(symbol).upper())
    if event_type:
        query = query.where(MarketEvent.event_type == str(event_type))
    return (
        session.execute(query.order_by(MarketEvent.ts.desc(), MarketEvent.id.desc()).limit(bounded_limit))
        .scalars()
        .all()
    )


def get_recent_alerts_equities(session, *, limit: int = 200) -> list[AlertSentEquity]:
    bounded_limit = min(max(int(limit), 1), 5003)
    return (
        session.execute(
            select(AlertSentEquity).order_by(AlertSentEquity.ts.desc(), AlertSentEquity.id.desc()).limit(bounded_limit)
        )
        .scalars()
        .all()
    )


def get_latest_market_regime(session) -> MarketRegime | None:
    return (
        session.execute(select(MarketRegime).order_by(MarketRegime.ts.desc()).limit(1))
        .scalars()
        .first()
    )


def get_market_regime_history(session, *, limit: int = 200) -> list[MarketRegime]:
    bounded_limit = min(max(int(limit), 1), 5003)
    return (
        session.execute(select(MarketRegime).order_by(MarketRegime.ts.desc()).limit(bounded_limit))
        .scalars()
        .all()
    )


def record_snapshot(
    session,
    *,
    ok: bool,
    error: str | None,
    page_count: int,
    auction_count: int,
    ts: int | None = None,
) -> None:
    snapshot_ts = ts or int(time.time())
    stmt = sqlite_insert(Snapshot).values(
        ts=snapshot_ts,
        ok=1 if ok else 0,
        error=error,
        page_count=page_count,
        auction_count=auction_count,
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Snapshot.ts],
            set_={
                "ok": stmt.excluded.ok,
                "error": stmt.excluded.error,
                "page_count": stmt.excluded.page_count,
                "auction_count": stmt.excluded.auction_count,
            },
        )
    )


def get_counts(session) -> dict:
    auctions_count = session.execute(select(func.count()).select_from(Auction)).scalar_one()
    history_count = session.execute(select(func.count()).select_from(AuctionHistory)).scalar_one()
    bars_count = session.execute(select(func.count()).select_from(ItemBar)).scalar_one()
    features_count = session.execute(select(func.count()).select_from(ItemFeature)).scalar_one()
    recs_count = session.execute(select(func.count()).select_from(Recommendation)).scalar_one()
    open_positions = session.execute(
        select(func.count()).select_from(PortfolioPosition).where(PortfolioPosition.status == "OPEN")
    ).scalar_one()
    total_positions = session.execute(select(func.count()).select_from(PortfolioPosition)).scalar_one()
    backtests_count = session.execute(select(func.count()).select_from(BacktestResult)).scalar_one()
    alerts_count = session.execute(select(func.count()).select_from(AlertSent)).scalar_one()
    events_count = session.execute(select(func.count()).select_from(ItemEvent)).scalar_one()
    favorites_count = session.execute(select(func.count()).select_from(WatchTarget)).scalar_one()
    spreads_count = session.execute(select(func.count()).select_from(ItemSpreadSnapshot)).scalar_one()
    forecasts_count = session.execute(select(func.count()).select_from(ItemForecast)).scalar_one()
    nocom_state_count = session.execute(select(func.count()).select_from(NocomModelState)).scalar_one()
    nocom_pairs_count = session.execute(select(func.count()).select_from(NocomTrainPair)).scalar_one()
    extremes_count = session.execute(select(func.count()).select_from(ItemExtreme)).scalar_one()
    symbols_count = session.execute(select(func.count()).select_from(Symbol)).scalar_one()
    market_bars_count = session.execute(select(func.count()).select_from(MarketBar)).scalar_one()
    market_features_count = session.execute(select(func.count()).select_from(MarketFeature)).scalar_one()
    market_regime_count = session.execute(select(func.count()).select_from(MarketRegime)).scalar_one()
    market_events_count = session.execute(select(func.count()).select_from(MarketEvent)).scalar_one()
    market_alerts_count = session.execute(select(func.count()).select_from(AlertSentEquity)).scalar_one()

    return {
        "auctions": int(auctions_count),
        "history": int(history_count),
        "bars": int(bars_count),
        "features": int(features_count),
        "recommendations": int(recs_count),
        "open_positions": int(open_positions),
        "positions": int(total_positions),
        "backtests": int(backtests_count),
        "alerts": int(alerts_count),
        "events": int(events_count),
        "favorites": int(favorites_count),
        "watch_targets": int(favorites_count),
        "spreads": int(spreads_count),
        "forecasts": int(forecasts_count),
        "nocom_model_state": int(nocom_state_count),
        "nocom_train_pairs": int(nocom_pairs_count),
        "extremes": int(extremes_count),
        "symbols": int(symbols_count),
        "market_bars": int(market_bars_count),
        "market_features": int(market_features_count),
        "market_regime": int(market_regime_count),
        "market_events": int(market_events_count),
        "market_alerts_equities": int(market_alerts_count),
    }
