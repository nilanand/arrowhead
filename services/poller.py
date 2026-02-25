from __future__ import annotations

import datetime as dt
import logging
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import and_, func, select, update

from .db import (
    IntradayBar,
    MarketBar,
    MarketFeature,
    MarketRegime,
    Symbol,
    insert_market_regime,
    session_scope,
    upsert_market_bars,
    upsert_symbols,
)
from .bars import upsert_intraday_bars
from .features import recompute_market_features_daily
from .market_data import parse_daily_symbols, parse_intraday_symbols

logger = logging.getLogger(__name__)


@dataclass
class PollerStatus:
    running: bool = False
    provider: str = "unknown"
    timeframe: str = "1d"

    last_ok_ts: int | None = None
    last_poll_ok_ts: int | None = None
    last_error: str | None = None
    last_provider_error: str | None = None

    consecutive_failures: int = 0
    backoff_until_ts: int | None = None

    last_universe_refresh_ts: int | None = None
    last_daily_bar_ts: int | None = None
    last_features_ts: int | None = None
    last_regime_ts: int | None = None
    last_noop_ts: int | None = None

    symbols_tracked: int = 0
    bars_fetched_last: int = 0
    bars_inserted_last: int = 0
    features_rows_upserted_last: int = 0
    provider_endpoints_in_use: list[str] = field(default_factory=list)


class MarketPoller:
    def __init__(
        self,
        *,
        provider,
        poll_seconds: int = 600,
        universe_refresh_seconds: int = 1800,
        daily_lookback_days: int = 1825,
        daily_backfill_days: int = 180,
        daily_patch_buffer_days: int = 7,
        required_trading_days: int = 520,
        timeframe: str = "1d",
        forecast_lookback_days: int = 504,
        forecast_horizon_days: int = 5,
        forecast_min_samples: int = 60,
        intraday_enabled: bool = False,
        intraday_symbols: list[str] | None = None,
        intraday_lookback_days: int = 7,
    ) -> None:
        self.provider = provider
        self.poll_seconds = max(int(poll_seconds), 60)
        self.universe_refresh_seconds = max(int(universe_refresh_seconds), 300)
        self.daily_lookback_days = max(int(daily_lookback_days), 365)
        self.daily_backfill_days = max(int(daily_backfill_days), 30)
        self.daily_patch_buffer_days = max(int(daily_patch_buffer_days), 0)
        self.required_trading_days = max(int(required_trading_days), 260)
        self.timeframe = str(timeframe or "1d")
        self.forecast_lookback_days = max(int(forecast_lookback_days), 20)
        self.forecast_horizon_days = max(int(forecast_horizon_days), 1)
        self.forecast_min_samples = max(int(forecast_min_samples), 1)
        self.intraday_enabled = bool(intraday_enabled)
        self.intraday_symbols = [str(s).upper() for s in (intraday_symbols or parse_intraday_symbols(os.getenv("INTRADAY_SYMBOLS"))) if s]
        self.intraday_lookback_days = max(int(intraday_lookback_days), 1)

        self._symbols: list[str] = []
        self._next_universe_refresh_ts = 0
        self._next_poll_ts = time.time() + 1.0

        self._status = PollerStatus(
            provider=str(getattr(provider, "name", "unknown")),
            timeframe=self.timeframe,
        )
        self._status_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="arrowhead-poller", daemon=True)
        self._thread.start()
        self._set_status(running=True)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._set_status(running=False)

    def is_running(self) -> bool:
        with self._status_lock:
            return bool(self._status.running)

    def get_status(self) -> dict:
        with self._status_lock:
            return {
                "running": bool(self._status.running),
                "provider": str(self._status.provider),
                "timeframe": str(self._status.timeframe),
                "last_ok_ts": self._status.last_ok_ts,
                "last_poll_ok_ts": self._status.last_poll_ok_ts,
                "last_error": self._status.last_error,
                "last_provider_error": self._status.last_provider_error,
                "consecutive_failures": int(self._status.consecutive_failures),
                "backoff_until_ts": self._status.backoff_until_ts,
                "last_universe_refresh_ts": self._status.last_universe_refresh_ts,
                "last_daily_bar_ts": self._status.last_daily_bar_ts,
                "last_features_ts": self._status.last_features_ts,
                "last_regime_ts": self._status.last_regime_ts,
                "last_noop_ts": self._status.last_noop_ts,
                "symbols_tracked": int(self._status.symbols_tracked),
                "bars_fetched_last": int(self._status.bars_fetched_last),
                "bars_inserted_last": int(self._status.bars_inserted_last),
                "features_rows_upserted_last": int(self._status.features_rows_upserted_last),
                "provider_endpoints_in_use": list(self._status.provider_endpoints_in_use),
                "poll_seconds": self.poll_seconds,
                "universe_refresh_seconds": self.universe_refresh_seconds,
                "daily_backfill_days": self.daily_backfill_days,
                "daily_patch_buffer_days": self.daily_patch_buffer_days,
                "required_trading_days": self.required_trading_days,
                "forecast_lookback_days": self.forecast_lookback_days,
                "forecast_horizon_days": self.forecast_horizon_days,
                "forecast_min_samples": self.forecast_min_samples,
                "intraday_enabled": bool(self.intraday_enabled),
                "intraday_symbols": list(self.intraday_symbols),
                "intraday_lookback_days": self.intraday_lookback_days,
            }

    def _set_status(self, **kwargs) -> None:
        with self._status_lock:
            for key, value in kwargs.items():
                setattr(self._status, key, value)

    @staticmethod
    def _date_to_ts(trading_date: dt.date) -> int:
        return int(dt.datetime.combine(trading_date, dt.time.min, tzinfo=dt.timezone.utc).timestamp())

    @staticmethod
    def _ts_to_date(raw_ts: int) -> dt.date:
        return dt.datetime.fromtimestamp(int(raw_ts), tz=dt.timezone.utc).date()

    @staticmethod
    def _safe_float(raw: object) -> float:
        try:
            return float(raw)
        except Exception:
            return 0.0

    def _configured_symbols(self) -> list[str]:
        return parse_daily_symbols(os.getenv("DAILY_SYMBOLS"))

    def _refresh_universe_if_needed(self, session, *, now_ts: int) -> None:
        if int(now_ts) < int(self._next_universe_refresh_ts) and self._symbols:
            return

        symbols = self._configured_symbols()
        rows = [
            {
                "symbol": symbol,
                "exchange": "US",
                "asset_type": "ETF" if symbol in {"SPY", "QQQ", "IWM", "TLT", "GLD"} else "EQUITY",
                "active": 1,
                "last_seen_ts": int(now_ts),
            }
            for symbol in symbols
        ]
        if symbols:
            session.execute(
                update(Symbol)
                .where(Symbol.symbol.not_in(symbols))
                .values(active=0)
            )
        upsert_symbols(session, rows)
        self._symbols = list(symbols)
        self._next_universe_refresh_ts = int(now_ts) + self.universe_refresh_seconds
        self._set_status(
            last_universe_refresh_ts=int(now_ts),
            symbols_tracked=len(self._symbols),
        )

    def _latest_bar_ts_map(self, session, *, symbols: list[str]) -> dict[str, int]:
        if not symbols:
            return {}
        rows = (
            session.execute(
                select(
                    MarketBar.symbol,
                    func.max(MarketBar.bar_ts).label("max_bar_ts"),
                )
                .where(MarketBar.timeframe == self.timeframe)
                .where(MarketBar.symbol.in_(symbols))
                .group_by(MarketBar.symbol)
            )
            .all()
        )
        return {
            str(symbol).upper(): int(max_bar_ts)
            for symbol, max_bar_ts in rows
            if symbol is not None and max_bar_ts is not None
        }

    def _bar_count_map(self, session, *, symbols: list[str]) -> dict[str, int]:
        if not symbols:
            return {}
        rows = (
            session.execute(
                select(
                    MarketBar.symbol,
                    func.count().label("row_count"),
                )
                .where(MarketBar.timeframe == self.timeframe)
                .where(MarketBar.symbol.in_(symbols))
                .group_by(MarketBar.symbol)
            )
            .all()
        )
        return {
            str(symbol).upper(): int(row_count or 0)
            for symbol, row_count in rows
            if symbol is not None
        }

    def _feature_count_map(self, session, *, symbols: list[str]) -> dict[str, int]:
        if not symbols:
            return {}
        rows = (
            session.execute(
                select(
                    MarketFeature.symbol,
                    func.count().label("row_count"),
                )
                .where(MarketFeature.timeframe == self.timeframe)
                .where(MarketFeature.symbol.in_(symbols))
                .group_by(MarketFeature.symbol)
            )
            .all()
        )
        return {
            str(symbol).upper(): int(row_count or 0)
            for symbol, row_count in rows
            if symbol is not None
        }

    def _existing_bar_keys(self, session, *, rows: list[dict]) -> set[tuple[str, int]]:
        keys_by_symbol: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            bar_ts = int(row.get("bar_ts") or 0)
            if not symbol or bar_ts <= 0:
                continue
            keys_by_symbol[symbol].append(bar_ts)

        existing: set[tuple[str, int]] = set()
        for symbol, ts_values in keys_by_symbol.items():
            unique_ts = sorted(set(ts_values))
            if not unique_ts:
                continue
            chunk_size = 300
            for i in range(0, len(unique_ts), chunk_size):
                chunk = unique_ts[i:i + chunk_size]
                rows_found = (
                    session.execute(
                        select(MarketBar.symbol, MarketBar.bar_ts)
                        .where(MarketBar.timeframe == self.timeframe)
                        .where(MarketBar.symbol == symbol)
                        .where(MarketBar.bar_ts.in_(chunk))
                    )
                    .all()
                )
                for existing_symbol, existing_ts in rows_found:
                    if existing_symbol is None or existing_ts is None:
                        continue
                    existing.add((str(existing_symbol).upper(), int(existing_ts)))
        return existing

    def _compute_market_regime_for_ts(
        self,
        session,
        *,
        symbols: list[str],
        target_ts: int,
        now_ts: int,
        include_extremes: bool,
    ) -> bool:
        if not symbols:
            return False

        rows = (
            session.execute(
                select(
                    MarketFeature.symbol,
                    MarketFeature.ema_20,
                    MarketFeature.ema_50,
                    MarketFeature.ema_200,
                    MarketFeature.trend_slope_20d,
                    MarketFeature.return_20d,
                    MarketFeature.realized_vol_20d,
                    MarketBar.close,
                )
                .select_from(MarketFeature)
                .join(
                    MarketBar,
                    and_(
                        MarketBar.symbol == MarketFeature.symbol,
                        MarketBar.timeframe == MarketFeature.timeframe,
                        MarketBar.bar_ts == MarketFeature.bar_ts,
                    ),
                )
                .where(MarketFeature.timeframe == self.timeframe)
                .where(MarketFeature.bar_ts == int(target_ts))
                .where(MarketFeature.symbol.in_(symbols))
            )
            .all()
        )
        if not rows:
            return False

        breadth_20_flags: list[int] = []
        breadth_50_flags: list[int] = []
        breadth_200_flags: list[int] = []
        positive_20d_flags: list[int] = []
        vol_values: list[float] = []
        structural_bull_flags: list[int] = []
        structural_bear_flags: list[int] = []
        latest_close_by_symbol: dict[str, float] = {}

        for symbol, ema_20, ema_50, ema_200, slope_20d, ret_20d, vol_20d, close in rows:
            if close is None:
                continue
            close_f = float(close)
            symbol_upper = str(symbol).upper()
            latest_close_by_symbol[symbol_upper] = close_f
            if ema_20 is not None:
                breadth_20_flags.append(1 if close_f > float(ema_20) else 0)
            if ema_50 is not None:
                breadth_50_flags.append(1 if close_f > float(ema_50) else 0)
            if ema_200 is not None:
                breadth_200_flags.append(1 if close_f > float(ema_200) else 0)
            if ret_20d is not None:
                positive_20d_flags.append(1 if float(ret_20d) > 0 else 0)
            if ema_50 is not None and slope_20d is not None:
                is_bull = close_f > float(ema_50) and float(slope_20d) > 0
                is_bear = close_f < float(ema_50) and float(slope_20d) < 0
                structural_bull_flags.append(1 if is_bull else 0)
                structural_bear_flags.append(1 if is_bear else 0)
            if vol_20d is not None:
                vol_values.append(float(vol_20d))

        def _pct(flags: list[int]) -> float | None:
            if not flags:
                return None
            return float(sum(flags) / len(flags))

        pct_new_high = None
        pct_new_low = None
        if include_extremes:
            recent_bar_rows = (
                session.execute(
                    select(MarketBar.symbol, MarketBar.close, MarketBar.bar_ts)
                    .where(MarketBar.timeframe == self.timeframe)
                    .where(MarketBar.symbol.in_(symbols))
                    .where(MarketBar.bar_ts <= int(target_ts))
                    .where(MarketBar.bar_ts >= int(target_ts) - (90 * 86_400))
                    .order_by(MarketBar.symbol.asc(), MarketBar.bar_ts.desc())
                )
                .all()
            )
            close_windows: dict[str, list[float]] = defaultdict(list)
            for symbol, close, _bar_ts in recent_bar_rows:
                if symbol is None or close is None:
                    continue
                symbol_upper = str(symbol).upper()
                if len(close_windows[symbol_upper]) >= 20:
                    continue
                close_windows[symbol_upper].append(float(close))

            new_20d_high_flags: list[int] = []
            new_20d_low_flags: list[int] = []
            for symbol, closes in close_windows.items():
                current_close = latest_close_by_symbol.get(symbol)
                if current_close is None or len(closes) < 20:
                    continue
                high_20 = max(closes)
                low_20 = min(closes)
                new_20d_high_flags.append(1 if current_close >= high_20 else 0)
                new_20d_low_flags.append(1 if current_close <= low_20 else 0)
            pct_new_high = _pct(new_20d_high_flags)
            pct_new_low = _pct(new_20d_low_flags)

        breadth_20 = _pct(breadth_20_flags)
        breadth_50 = _pct(breadth_50_flags)
        breadth_200 = _pct(breadth_200_flags)
        pct_positive_20d = _pct(positive_20d_flags)
        median_vol = float(statistics.median(vol_values)) if vol_values else None
        bull_share = _pct(structural_bull_flags) or 0.0
        bear_share = _pct(structural_bear_flags) or 0.0

        if bull_share >= 0.55 and bull_share > bear_share:
            regime_tag = "BULL"
        elif bear_share >= 0.55 and bear_share > bull_share:
            regime_tag = "BEAR"
        else:
            regime_tag = "CHOP"

        insert_market_regime(
            session,
            {
                "ts": int(target_ts),
                "breadth_pct_above_ema20": breadth_20,
                "breadth_pct_above_ema50": breadth_50,
                "breadth_pct_above_ema200": breadth_200,
                "pct_with_positive_20d_return": pct_positive_20d,
                "new_20d_high_pct": pct_new_high,
                "new_20d_low_pct": pct_new_low,
                "median_volatility_20d": median_vol,
                "regime_tag": regime_tag,
                "computed_ts": int(now_ts),
            },
        )
        return True

    def _compute_market_regime(self, session, *, symbols: list[str], now_ts: int) -> int | None:
        if not symbols:
            return None

        feature_ts_rows = (
            session.execute(
                select(MarketFeature.bar_ts)
                .where(MarketFeature.timeframe == self.timeframe)
                .where(MarketFeature.symbol.in_(symbols))
                .group_by(MarketFeature.bar_ts)
                .order_by(MarketFeature.bar_ts.asc())
            )
            .all()
        )
        feature_dates = [int(bar_ts) for (bar_ts,) in feature_ts_rows if bar_ts is not None]
        if not feature_dates:
            return None

        latest_ts = int(feature_dates[-1])
        earliest_ts = int(feature_dates[0])

        existing_ts_rows = (
            session.execute(
                select(MarketRegime.ts)
                .where(MarketRegime.ts >= int(earliest_ts))
                .where(MarketRegime.ts <= int(latest_ts))
            )
            .all()
        )
        existing_ts = {int(ts) for (ts,) in existing_ts_rows if ts is not None}

        dates_to_compute = [ts for ts in feature_dates if ts not in existing_ts]
        if latest_ts not in dates_to_compute:
            dates_to_compute.append(latest_ts)

        computed_rows = 0
        for ts in dates_to_compute:
            did_write = self._compute_market_regime_for_ts(
                session,
                symbols=symbols,
                target_ts=int(ts),
                now_ts=now_ts,
                include_extremes=(int(ts) == int(latest_ts)),
            )
            if did_write:
                computed_rows += 1

        if computed_rows > 0:
            logger.info(
                "Arrowhead market regime upserted rows=%s range=%s..%s latest=%s",
                int(computed_rows),
                int(dates_to_compute[0]),
                int(dates_to_compute[-1]),
                int(latest_ts),
            )

        return int(latest_ts)

    def _build_daily_rows(self, *, symbol: str, frame) -> list[dict]:
        out: list[dict] = []
        if frame is None or getattr(frame, "empty", True):
            return out

        for trading_date, row in frame.iterrows():
            if isinstance(trading_date, dt.datetime):
                trade_date = trading_date.date()
            else:
                trade_date = trading_date

            if not isinstance(trade_date, dt.date):
                continue

            open_v = self._safe_float(row.get("open"))
            high_v = self._safe_float(row.get("high"))
            low_v = self._safe_float(row.get("low"))
            close_v = self._safe_float(row.get("close"))
            volume_v = max(self._safe_float(row.get("volume")), 0.0)

            if min(open_v, high_v, low_v, close_v) <= 0.0:
                continue

            out.append(
                {
                    "symbol": str(symbol).upper(),
                    "timeframe": self.timeframe,
                    "bar_ts": self._date_to_ts(trade_date),
                    "open": open_v,
                    "high": high_v,
                    "low": low_v,
                    "close": close_v,
                    "volume": volume_v,
                }
            )

        out.sort(key=lambda row: int(row["bar_ts"]))
        return out

    # --- INTRADAY: polling section (v0.16.0) ---
    def _run_intraday_cycle(self, *, now_ts: int, session=None) -> dict:
        result = {
            "enabled": bool(self.intraday_enabled),
            "symbols": {},
            "inserted_total": 0,
            "errors": [],
        }
        if not self.intraday_enabled:
            return result

        symbols = [str(symbol).upper() for symbol in self.intraday_symbols if symbol]
        if not symbols:
            return result

        fetch_intraday = getattr(self.provider, "fetch_intraday_1m", None)
        if not callable(fetch_intraday):
            error_msg = "provider does not implement fetch_intraday_1m"
            logger.error("Intraday polling skipped: %s", error_msg)
            result["errors"].append(error_msg)
            return result

        if session is None:
            with session_scope() as managed_session:
                return self._run_intraday_cycle(now_ts=now_ts, session=managed_session)

        for symbol in symbols:
            fetched_rows = 0
            inserted_rows = 0
            last_stored = None
            try:
                frame = fetch_intraday(symbol, lookback_days=self.intraday_lookback_days)
                fetched_rows = int(len(frame.index)) if frame is not None else 0
                inserted_rows = int(upsert_intraday_bars(session, symbol, frame))
                last_stored = session.execute(
                    select(func.max(IntradayBar.bar_ts)).where(IntradayBar.symbol == str(symbol).upper())
                ).scalar_one_or_none()
                logger.info(
                    "intraday poll symbol=%s fetched=%s inserted=%s last_bar_ts=%s",
                    str(symbol).upper(),
                    int(fetched_rows),
                    int(inserted_rows),
                    (last_stored.isoformat() if last_stored is not None else None),
                )
            except Exception as exc:
                err_text = str(exc)
                logger.error("intraday poll failed for symbol=%s: %s", str(symbol).upper(), err_text)
                result["errors"].append(f"{symbol}:{err_text}")
            result["symbols"][str(symbol).upper()] = {
                "fetched_rows": int(fetched_rows),
                "inserted_rows": int(inserted_rows),
                "last_bar_ts": (last_stored.isoformat() if last_stored is not None else None),
            }
            result["inserted_total"] += int(inserted_rows)

        return result

    # --- END INTRADAY ---

    def _run_cycle(self) -> dict:
        now_ts = int(time.time())
        today = dt.datetime.now(dt.timezone.utc).date()

        with session_scope() as session:
            self._refresh_universe_if_needed(session, now_ts=now_ts)
            symbols = list(self._symbols)
            if not symbols:
                intraday_result = self._run_intraday_cycle(now_ts=now_ts, session=session)
                provider_endpoints = ["yahoo_download_1d"]
                if intraday_result.get("enabled"):
                    provider_endpoints.append("yahoo_download_1m")
                self._set_status(
                    last_ok_ts=now_ts,
                    last_poll_ok_ts=now_ts,
                    last_error=None,
                    last_provider_error=None,
                    bars_fetched_last=0,
                    bars_inserted_last=0,
                    features_rows_upserted_last=0,
                    last_noop_ts=now_ts,
                    provider_endpoints_in_use=provider_endpoints,
                )
                return {"ok": True, "ts": now_ts, "noop": True, "reason": "no symbols configured"}

            before_map = self._latest_bar_ts_map(session, symbols=symbols)
            bar_count_before = self._bar_count_map(session, symbols=symbols)
            feature_count_before = self._feature_count_map(session, symbols=symbols)

            # Convert trading-day target to a safe calendar-day span for Yahoo fetch windows.
            ensured_calendar_days = max(self.daily_backfill_days, self.required_trading_days * 2)

            fetch_groups: dict[dt.date, list[str]] = defaultdict(list)
            for symbol in symbols:
                history_rows = int(bar_count_before.get(symbol, 0))
                needs_history = history_rows < self.required_trading_days
                if needs_history:
                    start_date = today - dt.timedelta(days=ensured_calendar_days)
                else:
                    last_bar_ts = before_map.get(symbol)
                    if last_bar_ts is None:
                        start_date = today - dt.timedelta(days=self.daily_backfill_days)
                    else:
                        start_date = self._ts_to_date(int(last_bar_ts)) - dt.timedelta(days=self.daily_patch_buffer_days)
                if start_date > today:
                    start_date = today
                fetch_groups[start_date].append(symbol)

            bars_by_symbol: dict[str, object] = {}
            fetch_debug: dict[str, dict] = {}
            last_provider_error = None
            for start_date, grouped_symbols in sorted(fetch_groups.items(), key=lambda item: item[0]):
                try:
                    fetched = self.provider.fetch_daily_bars(
                        grouped_symbols,
                        start_date=start_date,
                        end_date=today,
                    )
                except Exception as exc:
                    last_provider_error = str(exc)
                    logger.error(
                        "Daily provider fetch failed for symbols=%s start=%s end=%s: %s",
                        ",".join(grouped_symbols),
                        start_date,
                        today,
                        exc,
                    )
                    continue

                for symbol_key, frame in (fetched or {}).items():
                    symbol_upper = str(symbol_key).upper()
                    bars_by_symbol[symbol_upper] = frame
                    if frame is None or getattr(frame, "empty", True):
                        fetch_debug[symbol_upper] = {
                            "rows": 0,
                            "min_date": None,
                            "max_date": None,
                        }
                        continue
                    try:
                        idx_values = list(frame.index)
                        min_date = str(idx_values[0]) if idx_values else None
                        max_date = str(idx_values[-1]) if idx_values else None
                    except Exception:
                        min_date = None
                        max_date = None
                    fetch_debug[symbol_upper] = {
                        "rows": int(len(frame.index)),
                        "min_date": min_date,
                        "max_date": max_date,
                    }

            all_rows: list[dict] = []
            fetched_rows_by_symbol: dict[str, int] = {}
            for symbol in symbols:
                built_rows = self._build_daily_rows(symbol=symbol, frame=bars_by_symbol.get(symbol))
                fetched_rows_by_symbol[symbol] = int(len(built_rows))
                all_rows.extend(built_rows)

            deduped: dict[tuple[str, str, int], dict] = {}
            for row in all_rows:
                key = (str(row["symbol"]), str(row["timeframe"]), int(row["bar_ts"]))
                deduped[key] = row
            bar_rows = sorted(deduped.values(), key=lambda row: (str(row["symbol"]), int(row["bar_ts"])))

            existing_before = self._existing_bar_keys(session, rows=bar_rows)
            if bar_rows:
                upsert_market_bars(
                    session,
                    bar_rows,
                    now_ts=now_ts,
                    intraday_lookback_days=30,
                    daily_lookback_days=self.daily_lookback_days,
                    prune_old=False,
                )

            inserted_keys = [
                (str(row["symbol"]).upper(), int(row["bar_ts"]))
                for row in bar_rows
                if (str(row["symbol"]).upper(), int(row["bar_ts"])) not in existing_before
            ]
            inserted_bar_count = len(inserted_keys)
            symbols_with_new_bars = sorted(set(symbol for symbol, _ in inserted_keys))

            after_map = self._latest_bar_ts_map(session, symbols=symbols)
            bar_count_after = self._bar_count_map(session, symbols=symbols)
            feature_count_after = self._feature_count_map(session, symbols=symbols)

            features_result = {"updated": 0, "rows": 0}
            regime_ts = None

            # Recompute features for all symbols that changed OR are missing feature coverage.
            dirty_symbols: set[str] = set(symbols_with_new_bars)
            for symbol in symbols:
                bars_n = int(bar_count_after.get(symbol, 0))
                features_n = int(feature_count_after.get(symbol, 0))
                if bars_n < self.required_trading_days:
                    dirty_symbols.add(symbol)
                # Keep features aligned to available bars so every symbol has full-history rows.
                elif features_n < bars_n:
                    dirty_symbols.add(symbol)

            if dirty_symbols:
                features_result = recompute_market_features_daily(
                    session,
                    symbols=sorted(dirty_symbols),
                    timeframe=self.timeframe,
                    lookback_rows=0,
                    forecast_lookback_days=self.forecast_lookback_days,
                    forecast_horizon_days=self.forecast_horizon_days,
                    forecast_min_samples=self.forecast_min_samples,
                )
                feature_count_after = self._feature_count_map(session, symbols=symbols)

            # Keep market_regime as a date series: fill missing trading dates and refresh latest.
            regime_ts = self._compute_market_regime(session, symbols=symbols, now_ts=now_ts)

            if not dirty_symbols:
                intraday_result = self._run_intraday_cycle(now_ts=now_ts, session=session)
                provider_endpoints = ["yahoo_download_1d"]
                if intraday_result.get("enabled"):
                    provider_endpoints.append("yahoo_download_1m")
                latest_ts = max(after_map.values()) if after_map else None
                logger.info(
                    "Arrowhead daily cycle no-op: required_days=%s bars=%s features=%s fetched=%s windows=%s",
                    self.required_trading_days,
                    {s: int(bar_count_after.get(s, 0)) for s in symbols},
                    {s: int(feature_count_after.get(s, 0)) for s in symbols},
                    {s: int(fetched_rows_by_symbol.get(s, 0)) for s in symbols},
                    {
                        s: fetch_debug.get(s, {"rows": 0, "min_date": None, "max_date": None})
                        for s in symbols
                    },
                )
                prior = self.get_status()
                self._set_status(
                    last_ok_ts=now_ts,
                    last_poll_ok_ts=now_ts,
                    last_error=None,
                    last_provider_error=last_provider_error,
                    bars_fetched_last=len(bar_rows),
                    bars_inserted_last=0,
                    features_rows_upserted_last=0,
                    last_daily_bar_ts=(int(latest_ts) if latest_ts is not None else None),
                    last_regime_ts=(int(regime_ts) if regime_ts is not None else prior.get("last_regime_ts")),
                    last_noop_ts=now_ts,
                    symbols_tracked=len(symbols),
                    provider_endpoints_in_use=provider_endpoints,
                )
                return {
                    "ok": True,
                    "ts": now_ts,
                    "noop": True,
                    "bars_fetched": len(bar_rows),
                    "bars_inserted": 0,
                }

            latest_ts = max(after_map.values()) if after_map else None

        intraday_result = self._run_intraday_cycle(now_ts=now_ts)
        provider_endpoints = ["yahoo_download_1d"]
        if intraday_result.get("enabled"):
            provider_endpoints.append("yahoo_download_1m")

        logger.info(
            "Arrowhead daily cycle: required_days=%s bars=%s features=%s fetched=%s windows=%s dirty=%s inserted=%s",
            self.required_trading_days,
            {s: int(bar_count_after.get(s, 0)) for s in symbols},
            {s: int(feature_count_after.get(s, 0)) for s in symbols},
            {s: int(fetched_rows_by_symbol.get(s, 0)) for s in symbols},
            {
                s: fetch_debug.get(s, {"rows": 0, "min_date": None, "max_date": None})
                for s in symbols
            },
            sorted(dirty_symbols),
            int(inserted_bar_count),
        )

        prior = self.get_status()
        self._set_status(
            last_ok_ts=now_ts,
            last_poll_ok_ts=now_ts,
            last_error=None,
            last_provider_error=last_provider_error,
            consecutive_failures=0,
            backoff_until_ts=None,
            bars_fetched_last=len(bar_rows),
            bars_inserted_last=int(inserted_bar_count),
            features_rows_upserted_last=int(features_result.get("updated") or 0),
            last_daily_bar_ts=(int(latest_ts) if latest_ts is not None else prior.get("last_daily_bar_ts")),
            last_features_ts=(
                now_ts if int(features_result.get("updated") or 0) > 0 else prior.get("last_features_ts")
            ),
            last_regime_ts=(int(regime_ts) if regime_ts is not None else prior.get("last_regime_ts")),
            symbols_tracked=len(symbols),
            provider_endpoints_in_use=provider_endpoints,
        )

        return {
            "ok": True,
            "ts": now_ts,
            "noop": False,
            "bars_fetched": len(bar_rows),
            "bars_inserted": int(inserted_bar_count),
            "symbols_with_new_bars": list(symbols_with_new_bars),
            "features": features_result,
            "regime_ts": regime_ts,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._status_lock:
                backoff_until = self._status.backoff_until_ts

            if backoff_until and int(now) < int(backoff_until):
                time.sleep(1)
                continue

            if now >= self._next_poll_ts:
                try:
                    result = self._run_cycle()
                    logger.debug(
                        "Arrowhead cycle ok: noop=%s bars_fetched=%s inserted=%s features=%s regime_ts=%s",
                        result.get("noop"),
                        result.get("bars_fetched"),
                        result.get("bars_inserted"),
                        result.get("features", {}).get("updated"),
                        result.get("regime_ts"),
                    )
                    self._next_poll_ts = time.time() + self.poll_seconds
                except Exception as exc:
                    fail_ts = int(time.time())
                    with self._status_lock:
                        self._status.consecutive_failures += 1
                        fail_count = int(self._status.consecutive_failures)
                    backoff_seconds = min(self.poll_seconds * (2 ** min(fail_count, 4)), 3600)
                    self._set_status(
                        last_error=str(exc),
                        backoff_until_ts=fail_ts + int(backoff_seconds),
                    )
                    logger.error("Arrowhead poll cycle failed: %s", exc)
                    self._next_poll_ts = time.time() + max(backoff_seconds, self.poll_seconds)

            time.sleep(1)
