from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request
import pandas as pd
from sqlalchemy import and_, func, select

load_dotenv()

from services.db import (  # noqa: E402
    IntradayBar,
    MarketBar,
    MarketEvent,
    MarketFeature,
    MarketRegime,
    Symbol,
    get_latest_market_regime,
    init_db,
    session_scope,
)
from services.market_data import build_market_data_provider, parse_intraday_symbols  # noqa: E402
from services.poller import MarketPoller  # noqa: E402
from services.features import compute_market_features_daily  # noqa: E402
from services.discord_sender import send_discord_message  # noqa: E402
from services.intraday import resample_ohlcv  # noqa: E402

os.environ.setdefault("FLASK_RUN_PORT", "5003")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ts_to_date_label(ts: int | None) -> str:
    if ts is None:
        return "-"
    try:
        value = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date()
    except Exception:
        return "-"
    return value.isoformat()


def _normalize_tf(raw: str | None) -> str:
    tf = str(raw or "1d").strip().lower()
    if tf in {"1m", "5m", "15m", "1w"}:
        return tf
    return "1d"


def _serialize_history_row(*, ts: int, close: float | None, ema_20: float | None, ema_50: float | None, ema_200: float | None,
                           vol_20d: float | None, drawdown_252d: float | None, trend_strength_200d: float | None,
                           vol_percentile_252d: float | None, dist_from_52w_high: float | None, regime_tag: str | None) -> dict:
    return {
        "date": _ts_to_date_label(int(ts)),
        "close": (float(close) if close is not None else None),
        "ema_20": (float(ema_20) if ema_20 is not None else None),
        "ema_50": (float(ema_50) if ema_50 is not None else None),
        "ema_200": (float(ema_200) if ema_200 is not None else None),
        "vol_20d": (float(vol_20d) if vol_20d is not None else None),
        "drawdown_252d": (float(drawdown_252d) if drawdown_252d is not None else None),
        "trend_strength_200d": (float(trend_strength_200d) if trend_strength_200d is not None else None),
        "vol_percentile_252d": (float(vol_percentile_252d) if vol_percentile_252d is not None else None),
        "dist_from_52w_high": (float(dist_from_52w_high) if dist_from_52w_high is not None else None),
        "regime_tag": (str(regime_tag) if regime_tag else "CHOP"),
    }


def _dt_to_iso_label(value: dt.datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _downsample_frame_even(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    limit = max(int(max_points), 1)
    if df.empty or len(df.index) <= limit:
        return df

    step = max((len(df.index) + limit - 1) // limit, 1)
    sampled = df.iloc[::step].copy()
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.iloc[[-1]]], axis=0)
    sampled = sampled[~sampled.index.duplicated(keep="last")]
    if len(sampled.index) > limit:
        sampled = sampled.iloc[-limit:]
    return sampled


def _history_rows_intraday(
    session,
    *,
    symbol_upper: str,
    tf: str,
    lookback_days: int,
    max_points: int,
) -> list[dict]:
    cutoff_dt = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=max(int(lookback_days), 1))
    rows = session.execute(
        select(
            IntradayBar.bar_ts,
            IntradayBar.open,
            IntradayBar.high,
            IntradayBar.low,
            IntradayBar.close,
            IntradayBar.volume,
        )
        .where(IntradayBar.symbol == symbol_upper)
        .where(IntradayBar.bar_ts >= cutoff_dt)
        .order_by(IntradayBar.bar_ts.asc())
    ).all()
    if not rows:
        return []

    idx = pd.to_datetime([row[0] for row in rows], errors="coerce", utc=True)
    df = pd.DataFrame(
        {
            "open": [float(row[1]) for row in rows],
            "high": [float(row[2]) for row in rows],
            "low": [float(row[3]) for row in rows],
            "close": [float(row[4]) for row in rows],
            "volume": [float(row[5] or 0.0) for row in rows],
        },
        index=idx,
    )
    df = df[~df.index.isna()]
    if df.empty:
        return []
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = df.index.tz_convert("UTC").tz_localize(None)

    if tf == "5m":
        df = resample_ohlcv(df, "5T")
    elif tf == "15m":
        df = resample_ohlcv(df, "15T")

    if df.empty:
        return []

    df = _downsample_frame_even(df, max_points=max_points)
    out: list[dict] = []
    for idx_dt, row in df.iterrows():
        out.append(
            {
                "date": _dt_to_iso_label(pd.Timestamp(idx_dt)),
                "open": (float(row.get("open")) if pd.notna(row.get("open")) else None),
                "high": (float(row.get("high")) if pd.notna(row.get("high")) else None),
                "low": (float(row.get("low")) if pd.notna(row.get("low")) else None),
                "close": (float(row.get("close")) if pd.notna(row.get("close")) else None),
                "volume": (float(row.get("volume")) if pd.notna(row.get("volume")) else None),
            }
        )
    return out


def _history_rows_1d(session, *, symbol_upper: str, limit: int) -> list[dict]:
    rows = session.execute(
        select(
            MarketBar.bar_ts,
            MarketBar.close,
            MarketFeature.ema_20,
            MarketFeature.ema_50,
            MarketFeature.ema_200,
            MarketFeature.realized_vol_20d,
            MarketFeature.drawdown_252d,
            MarketFeature.trend_strength_200d,
            MarketFeature.vol_percentile_252d,
            MarketFeature.dist_from_52w_high,
            MarketFeature.regime_tag,
        )
        .select_from(MarketBar)
        .outerjoin(
            MarketFeature,
            and_(
                MarketFeature.symbol == MarketBar.symbol,
                MarketFeature.timeframe == MarketBar.timeframe,
                MarketFeature.bar_ts == MarketBar.bar_ts,
            ),
        )
        .where(MarketBar.symbol == symbol_upper)
        .where(MarketBar.timeframe == TIMEFRAME)
        .order_by(MarketBar.bar_ts.desc())
        .limit(limit)
    ).all()

    return [
        _serialize_history_row(
            ts=int(bar_ts),
            close=(float(close_v) if close_v is not None else None),
            ema_20=(float(ema_20) if ema_20 is not None else None),
            ema_50=(float(ema_50) if ema_50 is not None else None),
            ema_200=(float(ema_200) if ema_200 is not None else None),
            vol_20d=(float(vol_20d) if vol_20d is not None else None),
            drawdown_252d=(float(dd_252d) if dd_252d is not None else None),
            trend_strength_200d=(float(trend_200d) if trend_200d is not None else None),
            vol_percentile_252d=(float(vol_pct) if vol_pct is not None else None),
            dist_from_52w_high=(float(dist_52w) if dist_52w is not None else None),
            regime_tag=(str(regime_tag) if regime_tag else None),
        )
        for (
            bar_ts,
            close_v,
            ema_20,
            ema_50,
            ema_200,
            vol_20d,
            dd_252d,
            trend_200d,
            vol_pct,
            dist_52w,
            regime_tag,
        ) in reversed(rows)
    ]


def _history_rows_1w(session, *, symbol_upper: str, limit: int) -> list[dict]:
    daily_fetch_limit = min(max(int(limit) * 10, 600), 5003)
    bars = session.execute(
        select(
            MarketBar.bar_ts,
            MarketBar.open,
            MarketBar.high,
            MarketBar.low,
            MarketBar.close,
            MarketBar.volume,
        )
        .where(MarketBar.symbol == symbol_upper)
        .where(MarketBar.timeframe == TIMEFRAME)
        .order_by(MarketBar.bar_ts.desc())
        .limit(daily_fetch_limit)
    ).all()
    if not bars:
        return []

    bars_asc = list(reversed(bars))
    idx = pd.to_datetime([int(row[0]) for row in bars_asc], unit="s", utc=True)
    df = pd.DataFrame(
        {
            "open": [float(row[1]) for row in bars_asc],
            "high": [float(row[2]) for row in bars_asc],
            "low": [float(row[3]) for row in bars_asc],
            "close": [float(row[4]) for row in bars_asc],
            "volume": [float(row[5] or 0.0) for row in bars_asc],
        },
        index=idx,
    )
    weekly = (
        df.resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    if weekly.empty:
        return []

    feat = compute_market_features_daily(symbol_upper, weekly)
    merged = weekly.join(
        feat[
            [
                "ema_20",
                "ema_50",
                "ema_200",
                "realized_vol_20d",
                "drawdown_252d",
                "trend_strength_200d",
                "vol_percentile_252d",
                "dist_from_52w_high",
                "regime_tag",
            ]
        ],
        how="left",
    )
    merged = merged.tail(limit)
    out: list[dict] = []
    for idx_dt, row in merged.iterrows():
        bar_ts = int(pd.Timestamp(idx_dt).tz_convert("UTC").timestamp())
        out.append(
            _serialize_history_row(
                ts=bar_ts,
                close=(float(row.get("close")) if row.get("close") is not None else None),
                ema_20=(float(row.get("ema_20")) if pd.notna(row.get("ema_20")) else None),
                ema_50=(float(row.get("ema_50")) if pd.notna(row.get("ema_50")) else None),
                ema_200=(float(row.get("ema_200")) if pd.notna(row.get("ema_200")) else None),
                vol_20d=(float(row.get("realized_vol_20d")) if pd.notna(row.get("realized_vol_20d")) else None),
                drawdown_252d=(float(row.get("drawdown_252d")) if pd.notna(row.get("drawdown_252d")) else None),
                trend_strength_200d=(float(row.get("trend_strength_200d")) if pd.notna(row.get("trend_strength_200d")) else None),
                vol_percentile_252d=(float(row.get("vol_percentile_252d")) if pd.notna(row.get("vol_percentile_252d")) else None),
                dist_from_52w_high=(float(row.get("dist_from_52w_high")) if pd.notna(row.get("dist_from_52w_high")) else None),
                regime_tag=(str(row.get("regime_tag")) if pd.notna(row.get("regime_tag")) else None),
            )
        )
    return out


PORT = env_int("PORT", 5003)
DEBUG = env_bool("FLASK_DEBUG", False)
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()

DATA_PROVIDER = (os.getenv("DATA_PROVIDER") or "yahoo").strip().lower()
TIMEFRAME = (os.getenv("TIMEFRAME") or "1d").strip().lower()
if TIMEFRAME != "1d":
    TIMEFRAME = "1d"
POLL_SECONDS = env_int("POLL_SECONDS", 600)
UNIVERSE_REFRESH_SECONDS = env_int("UNIVERSE_REFRESH_SECONDS", 1800)
DAILY_LOOKBACK_DAYS = env_int("DAILY_LOOKBACK_DAYS", 1825)
DAILY_BACKFILL_DAYS = env_int("DAILY_BACKFILL_DAYS", 420)
DAILY_PATCH_BUFFER_DAYS = env_int("DAILY_PATCH_BUFFER_DAYS", 7)
REQUIRED_TRADING_DAYS = env_int("REQUIRED_TRADING_DAYS", 520)
FORECAST_LOOKBACK_DAYS = env_int("FORECAST_LOOKBACK_DAYS", 504)
FORECAST_HORIZON_DAYS = env_int("FORECAST_HORIZON_DAYS", 5)
FORECAST_MIN_SAMPLES = env_int("FORECAST_MIN_SAMPLES", 60)
INTRADAY_ENABLED = env_bool("INTRADAY_ENABLED", False)
INTRADAY_SYMBOLS = parse_intraday_symbols(os.getenv("INTRADAY_SYMBOLS"))
INTRADAY_LOOKBACK_DAYS = env_int("INTRADAY_LOOKBACK_DAYS", 7)
DISCORD_SENDER_ENABLED = env_bool("DISCORD_SENDER_ENABLED", True)
DISCORD_SENDER_MAX_CHARS = max(1, env_int("DISCORD_SENDER_MAX_CHARS", 500))
DISCORD_SENDER_COOLDOWN_SECONDS = max(0, env_int("DISCORD_SENDER_COOLDOWN_SECONDS", 5))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("arrowhead")

init_db()
provider = build_market_data_provider(DATA_PROVIDER)
poller = MarketPoller(
    provider=provider,
    poll_seconds=POLL_SECONDS,
    universe_refresh_seconds=UNIVERSE_REFRESH_SECONDS,
    daily_lookback_days=DAILY_LOOKBACK_DAYS,
    daily_backfill_days=DAILY_BACKFILL_DAYS,
    daily_patch_buffer_days=DAILY_PATCH_BUFFER_DAYS,
    required_trading_days=REQUIRED_TRADING_DAYS,
    timeframe=TIMEFRAME,
    forecast_lookback_days=FORECAST_LOOKBACK_DAYS,
    forecast_horizon_days=FORECAST_HORIZON_DAYS,
    forecast_min_samples=FORECAST_MIN_SAMPLES,
    intraday_enabled=INTRADAY_ENABLED,
    intraday_symbols=INTRADAY_SYMBOLS,
    intraday_lookback_days=INTRADAY_LOOKBACK_DAYS,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
_poller_started = False
_discord_sender_lock = threading.Lock()
_discord_sender_last_send_ts = 0.0


def _discord_sender_available() -> bool:
    webhook_url = (os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    return bool(DISCORD_SENDER_ENABLED and webhook_url)


def start_poller_if_needed() -> None:
    global _poller_started
    if _poller_started:
        return
    poller.start()
    _poller_started = True
    logger.info("arrowhead daily poller started")


@app.before_request
def _ensure_background_poller() -> None:
    start_poller_if_needed()


def _dashboard_payload(session) -> dict:
    latest_subq = (
        select(
            MarketBar.symbol.label("symbol"),
            func.max(MarketBar.bar_ts).label("latest_bar_ts"),
        )
        .where(MarketBar.timeframe == TIMEFRAME)
        .group_by(MarketBar.symbol)
        .subquery()
    )

    query_rows = session.execute(
        select(
            Symbol.symbol,
            MarketBar.bar_ts,
            MarketBar.close,
            MarketFeature.ema_20,
            MarketFeature.ema_50,
            MarketFeature.ema_200,
            MarketFeature.trend_slope_20d,
            MarketFeature.trend_strength_200d,
            MarketFeature.return_20d,
            MarketFeature.realized_vol_20d,
            MarketFeature.vol_percentile_252d,
            MarketFeature.dist_from_52w_high,
            MarketFeature.drawdown_252d,
            MarketFeature.regime_tag,
            MarketFeature.cont_prob_5d,
            MarketFeature.exp_return_5d,
            MarketFeature.cont_samples_5d,
            MarketFeature.mom_strength,
        )
        .select_from(Symbol)
        .outerjoin(latest_subq, latest_subq.c.symbol == Symbol.symbol)
        .outerjoin(
            MarketBar,
            and_(
                MarketBar.symbol == Symbol.symbol,
                MarketBar.timeframe == TIMEFRAME,
                MarketBar.bar_ts == latest_subq.c.latest_bar_ts,
            ),
        )
        .outerjoin(
            MarketFeature,
            and_(
                MarketFeature.symbol == Symbol.symbol,
                MarketFeature.timeframe == TIMEFRAME,
                MarketFeature.bar_ts == latest_subq.c.latest_bar_ts,
            ),
        )
        .where(Symbol.active == 1)
        .order_by(Symbol.symbol.asc())
    ).all()

    rows: list[dict] = []
    for (
        symbol,
        bar_ts,
        close,
        ema_20,
        ema_50,
        ema_200,
        slope_20d,
        trend_200d,
        return_20d,
        vol_20d,
        vol_pct_252d,
        dist_52w,
        drawdown_252d,
        regime_tag,
        cont_prob_5d,
        exp_return_5d,
        cont_samples_5d,
        mom_strength,
    ) in query_rows:
        rows.append(
            {
                "symbol": str(symbol),
                "bar_ts": (int(bar_ts) if bar_ts is not None else None),
                "bar_date": _ts_to_date_label(int(bar_ts)) if bar_ts is not None else "-",
                "close": (float(close) if close is not None else None),
                "ema_20": (float(ema_20) if ema_20 is not None else None),
                "ema_50": (float(ema_50) if ema_50 is not None else None),
                "ema_200": (float(ema_200) if ema_200 is not None else None),
                "trend_slope_20d": (float(slope_20d) if slope_20d is not None else None),
                "trend_strength_200d": (float(trend_200d) if trend_200d is not None else None),
                "return_20d": (float(return_20d) if return_20d is not None else None),
                "realized_vol_20d": (float(vol_20d) if vol_20d is not None else None),
                "vol_percentile_252d": (float(vol_pct_252d) if vol_pct_252d is not None else None),
                "dist_from_52w_high": (float(dist_52w) if dist_52w is not None else None),
                "drawdown_252d": (float(drawdown_252d) if drawdown_252d is not None else None),
                "regime_tag": (str(regime_tag) if regime_tag else "CHOP"),
                "cont_prob_5d": (float(cont_prob_5d) if cont_prob_5d is not None else None),
                "exp_return_5d": (float(exp_return_5d) if exp_return_5d is not None else None),
                "cont_samples_5d": (int(cont_samples_5d) if cont_samples_5d is not None else None),
                "mom_strength": (float(mom_strength) if mom_strength is not None else None),
            }
        )

    latest_regime = get_latest_market_regime(session)
    last_daily_bar_ts = (
        session.execute(select(func.max(MarketBar.bar_ts)).where(MarketBar.timeframe == TIMEFRAME)).scalar_one_or_none()
    )

    return {
        "rows": rows,
        "latest_regime": latest_regime,
        "last_daily_bar_ts": (int(last_daily_bar_ts) if last_daily_bar_ts is not None else None),
    }


def _symbol_payload(session, symbol: str) -> dict:
    symbol_upper = str(symbol).upper()

    latest_bar = (
        session.execute(
            select(MarketBar)
            .where(MarketBar.symbol == symbol_upper)
            .where(MarketBar.timeframe == TIMEFRAME)
            .order_by(MarketBar.bar_ts.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )

    latest_feature = None
    if latest_bar is not None:
        latest_feature = (
            session.execute(
                select(MarketFeature)
                .where(MarketFeature.symbol == symbol_upper)
                .where(MarketFeature.timeframe == TIMEFRAME)
                .where(MarketFeature.bar_ts == int(latest_bar.bar_ts))
                .limit(1)
            )
            .scalars()
            .first()
        )

    return {
        "symbol": symbol_upper,
        "latest_bar": latest_bar,
        "latest_feature": latest_feature,
    }


@app.get("/")
def home() -> object:
    return redirect("/dashboard", code=302)


@app.get("/font_test")
def font_test() -> object:
    return render_template(
        "font_test.html",
        page_title="arrowhead | font test",
        page_id="font_test",
    )


@app.get("/dashboard")
def dashboard() -> object:
    with session_scope() as session:
        payload = _dashboard_payload(session)

    return render_template(
        "dashboard.html",
        page_title="arrowhead | dashboard",
        page_id="dashboard",
        status=poller.get_status(),
        provider_name=str(getattr(provider, "name", "unknown")),
        timeframe=TIMEFRAME,
        payload=payload,
        ts_to_date=_ts_to_date_label,
        discord_sender_available=_discord_sender_available(),
        sender_max_chars=DISCORD_SENDER_MAX_CHARS,
    )


@app.get("/symbol/<symbol>")
def symbol_page(symbol: str) -> object:
    selected_tf = _normalize_tf(request.args.get("tf"))
    with session_scope() as session:
        payload = _symbol_payload(session, symbol)

    return render_template(
        "symbol.html",
        page_title=f"arrowhead | symbol {payload['symbol']}",
        page_id="symbol",
        symbol=payload["symbol"],
        latest_bar=payload["latest_bar"],
        latest_feature=payload["latest_feature"],
        status=poller.get_status(),
        timeframe=TIMEFRAME,
        selected_tf=selected_tf,
        ts_to_date=_ts_to_date_label,
    )


@app.get("/api/status")
def api_status() -> object:
    status = poller.get_status()
    with session_scope() as session:
        symbols_count = int(
            session.execute(select(func.count()).select_from(Symbol).where(Symbol.active == 1)).scalar_one() or 0
        )
        bars_1d_count = int(
            session.execute(select(func.count()).select_from(MarketBar).where(MarketBar.timeframe == TIMEFRAME)).scalar_one() or 0
        )
        features_1d_count = int(
            session.execute(select(func.count()).select_from(MarketFeature).where(MarketFeature.timeframe == TIMEFRAME)).scalar_one() or 0
        )
        regime_rows = int(session.execute(select(func.count()).select_from(MarketRegime)).scalar_one() or 0)
        events_count = int(session.execute(select(func.count()).select_from(MarketEvent)).scalar_one() or 0)
        last_daily_bar_ts = session.execute(
            select(func.max(MarketBar.bar_ts)).where(MarketBar.timeframe == TIMEFRAME)
        ).scalar_one_or_none()
        latest_feature_subq = (
            select(
                MarketFeature.symbol.label("symbol"),
                func.max(MarketFeature.bar_ts).label("latest_bar_ts"),
            )
            .where(MarketFeature.timeframe == TIMEFRAME)
            .group_by(MarketFeature.symbol)
            .subquery()
        )
        forecasts_ready_symbols = int(
            session.execute(
                select(func.count())
                .select_from(MarketFeature)
                .join(
                    latest_feature_subq,
                    and_(
                        latest_feature_subq.c.symbol == MarketFeature.symbol,
                        latest_feature_subq.c.latest_bar_ts == MarketFeature.bar_ts,
                    ),
                )
                .where(MarketFeature.timeframe == TIMEFRAME)
                .where(MarketFeature.cont_prob_5d.is_not(None))
            ).scalar_one()
            or 0
        )

    return jsonify(
        {
            "provider": str(getattr(provider, "name", "unknown")),
            "provider_name": str(getattr(provider, "name", "unknown")),
            "timeframe": TIMEFRAME,
            "port": int(os.getenv("PORT", 5003)),
            "poller_running": bool(status.get("running")),
            "last_bar_ts_written": status.get("last_daily_bar_ts"),
            "last_daily_bar_ts": (int(last_daily_bar_ts) if last_daily_bar_ts is not None else None),
            "last_ok_ts": status.get("last_ok_ts"),
            "last_poll_ok_ts": status.get("last_poll_ok_ts"),
            "last_error": status.get("last_error"),
            "last_provider_error": status.get("last_provider_error"),
            "provider_endpoints_in_use": status.get("provider_endpoints_in_use") or [],
            "forecasts_ready_symbols": forecasts_ready_symbols,
            "forecast_lookback_days": int(status.get("forecast_lookback_days") or FORECAST_LOOKBACK_DAYS),
            "forecast_horizon_days": int(status.get("forecast_horizon_days") or FORECAST_HORIZON_DAYS),
            "counts": {
                "symbols": symbols_count,
                "bars": bars_1d_count,
                "features": features_1d_count,
                "bars_1d": bars_1d_count,
                "features_1d": features_1d_count,
                "market_regime_rows": regime_rows,
                "events": events_count,
                "forecasts_ready_symbols": forecasts_ready_symbols,
            },
            "poller": status,
        }
    )


@app.get("/api/market/regime/latest")
def api_market_regime_latest() -> object:
    with session_scope() as session:
        row = get_latest_market_regime(session)

    if row is None:
        return jsonify({"ok": True, "regime": None})

    return jsonify(
        {
            "ok": True,
            "regime": {
                "ts": int(row.ts),
                "date": _ts_to_date_label(int(row.ts)),
                "breadth_pct_above_ema20": (
                    float(row.breadth_pct_above_ema20) if row.breadth_pct_above_ema20 is not None else None
                ),
                "breadth_pct_above_ema50": (
                    float(row.breadth_pct_above_ema50) if row.breadth_pct_above_ema50 is not None else None
                ),
                "breadth_pct_above_ema200": (
                    float(row.breadth_pct_above_ema200) if row.breadth_pct_above_ema200 is not None else None
                ),
                "pct_with_positive_20d_return": (
                    float(row.pct_with_positive_20d_return) if row.pct_with_positive_20d_return is not None else None
                ),
                "new_20d_high_pct": (
                    float(row.new_20d_high_pct) if row.new_20d_high_pct is not None else None
                ),
                "new_20d_low_pct": (
                    float(row.new_20d_low_pct) if row.new_20d_low_pct is not None else None
                ),
                "median_volatility_20d": (
                    float(row.median_volatility_20d) if row.median_volatility_20d is not None else None
                ),
                "regime_tag": (str(row.regime_tag) if row.regime_tag else None),
            },
        }
    )


@app.get("/api/market/regime/history")
def api_market_regime_history() -> object:
    raw_limit = request.args.get("limit", "400")
    try:
        limit = min(max(int(raw_limit), 1), 400)
    except ValueError:
        limit = 400

    with session_scope() as session:
        rows = (
            session.execute(
                select(
                    MarketRegime.ts,
                    MarketRegime.breadth_pct_above_ema20,
                    MarketRegime.breadth_pct_above_ema50,
                    MarketRegime.breadth_pct_above_ema200,
                    MarketRegime.pct_with_positive_20d_return,
                    MarketRegime.new_20d_high_pct,
                    MarketRegime.new_20d_low_pct,
                    MarketRegime.median_volatility_20d,
                    MarketRegime.regime_tag,
                )
                .order_by(MarketRegime.ts.desc())
                .limit(limit)
            )
            .all()
        )

    out = [
        {
            "date": _ts_to_date_label(int(ts)),
            "breadth_pct_above_ema20": (
                float(breadth_20) if breadth_20 is not None else None
            ),
            "breadth_pct_above_ema50": (
                float(breadth_50) if breadth_50 is not None else None
            ),
            "breadth_pct_above_ema200": (
                float(breadth_200) if breadth_200 is not None else None
            ),
            "pct_with_positive_20d_return": (
                float(pct_pos_20d) if pct_pos_20d is not None else None
            ),
            "new_20d_high_pct": (
                float(high_20d) if high_20d is not None else None
            ),
            "new_20d_low_pct": (
                float(low_20d) if low_20d is not None else None
            ),
            "median_volatility_20d": (
                float(median_vol) if median_vol is not None else None
            ),
            "regime_tag": (str(regime_tag) if regime_tag else "CHOP"),
        }
        for ts, breadth_20, breadth_50, breadth_200, pct_pos_20d, high_20d, low_20d, median_vol, regime_tag in reversed(rows)
    ]
    return jsonify(out)


@app.get("/api/market_structure")
def api_market_structure() -> object:
    raw_days = request.args.get("days", "60")
    try:
        days = min(max(int(raw_days), 1), 400)
    except ValueError:
        days = 60

    with session_scope() as session:
        rows = (
            session.execute(
                select(
                    MarketRegime.ts,
                    MarketRegime.breadth_pct_above_ema20,
                    MarketRegime.breadth_pct_above_ema50,
                    MarketRegime.breadth_pct_above_ema200,
                )
                .order_by(MarketRegime.ts.desc())
                .limit(days)
            )
            .all()
        )

    ordered = list(reversed(rows))
    labels = [_ts_to_date_label(int(ts)) for ts, *_rest in ordered]
    pct_above_ema20 = [float(v) if v is not None else None for _ts, v, _v50, _v200 in ordered]
    pct_above_ema50 = [float(v) if v is not None else None for _ts, _v20, v, _v200 in ordered]
    pct_above_ema200 = [float(v) if v is not None else None for _ts, _v20, _v50, v in ordered]

    message = None
    if len(labels) < 2:
        message = "warming up: need more history"

    return jsonify(
        {
            "labels": labels,
            "pct_above_ema20": pct_above_ema20,
            "pct_above_ema50": pct_above_ema50,
            "pct_above_ema200": pct_above_ema200,
            "message": message,
        }
    )


@app.get("/api/symbol/<symbol>/history")
def api_symbol_history(symbol: str) -> object:
    tf = _normalize_tf(request.args.get("tf"))
    raw_limit = request.args.get("limit")
    try:
        requested_limit = int(raw_limit) if raw_limit is not None else None
    except ValueError:
        requested_limit = None
    raw_days = request.args.get("days")
    try:
        lookback_days = max(int(raw_days), 1) if raw_days is not None else INTRADAY_LOOKBACK_DAYS
    except ValueError:
        lookback_days = INTRADAY_LOOKBACK_DAYS

    symbol_upper = str(symbol).upper()
    with session_scope() as session:
        if tf == "1m" or tf == "5m" or tf == "15m":
            max_points = min(max(int(requested_limit or 1200), 1), 1200)
            out = _history_rows_intraday(
                session,
                symbol_upper=symbol_upper,
                tf=tf,
                lookback_days=lookback_days,
                max_points=max_points,
            )
        elif tf == "1w":
            limit = min(max(int(requested_limit or 400), 1), 400)
            out = _history_rows_1w(session, symbol_upper=symbol_upper, limit=limit)
        else:
            limit = min(max(int(requested_limit or 400), 1), 400)
            out = _history_rows_1d(session, symbol_upper=symbol_upper, limit=limit)
    return jsonify(out)


@app.post("/api/discord/send")
def api_discord_send() -> object:
    global _discord_sender_last_send_ts

    if not _discord_sender_available():
        return jsonify({"ok": False, "error": "discord sender not configured"}), 400
    if not request.is_json:
        return jsonify({"ok": False, "error": "json body required"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "invalid json payload"}), 400

    message = str(payload.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "message is required"}), 400
    if len(message) > DISCORD_SENDER_MAX_CHARS:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"message exceeds max length ({DISCORD_SENDER_MAX_CHARS})",
                }
            ),
            400,
        )

    now = time.time()
    with _discord_sender_lock:
        elapsed = now - _discord_sender_last_send_ts
        if elapsed < DISCORD_SENDER_COOLDOWN_SECONDS:
            return jsonify({"ok": False, "error": "cooldown"}), 429

        webhook_url = (os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
        ok, err = send_discord_message(webhook_url=webhook_url, content=message)
        if not ok:
            return jsonify({"ok": False, "error": (err or "discord send failed")}), 400

        _discord_sender_last_send_ts = now

    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    start_poller_if_needed()
    port = int(os.getenv("PORT", 5003))
    print(f"Starting arrowhead flask runtime on port={port}")
    app.run(host="0.0.0.0", port=port, debug=DEBUG)
