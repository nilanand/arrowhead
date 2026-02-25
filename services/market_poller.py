from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from .db import get_active_symbols, session_scope, upsert_market_bars, upsert_symbols
from .market_alerts import EquitiesAlertsService
from .market_observatory import (
    aggregate_higher_timeframes,
    align_bar_ts,
    bootstrap_symbol_history,
    detect_market_events,
    recompute_market_features,
    recompute_market_regime,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketPollerStatus:
    last_ok_ts: int | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    backoff_until_ts: int | None = None

    last_universe_refresh_ts: int | None = None
    last_quote_poll_ts: int | None = None
    last_bar_close_ts: int | None = None
    last_feature_ts: int | None = None
    last_regime_ts: int | None = None
    last_events_ts: int | None = None
    last_alerts_ts: int | None = None

    universe_size: int = 0
    tracked_realtime_symbols: int = 0
    last_quotes_seen: int = 0
    last_closed_bars: int = 0
    last_bootstrap_symbols: int = 0


class MarketPoller:
    def __init__(
        self,
        *,
        provider,
        alerts_service: EquitiesAlertsService,
        poll_seconds: int = 30,
        universe_refresh_seconds: int = 6 * 3600,
        universe_symbol_limit: int = 8000,
        realtime_symbol_limit: int = 1500,
        history_bootstrap_batch: int = 2,
        history_refresh_seconds: int = 12 * 3600,
        intraday_lookback_days: int = 30,
        daily_lookback_days: int = 730,
        signals_mode: str = "observatory",
    ) -> None:
        self.provider = provider
        self.alerts_service = alerts_service

        self.poll_seconds = max(int(poll_seconds), 10)
        self.universe_refresh_seconds = max(int(universe_refresh_seconds), 300)
        self.universe_symbol_limit = max(int(universe_symbol_limit), 100)
        self.realtime_symbol_limit = max(int(realtime_symbol_limit), 50)
        self.history_bootstrap_batch = max(int(history_bootstrap_batch), 0)
        self.history_refresh_seconds = max(int(history_refresh_seconds), 1800)
        self.intraday_lookback_days = max(int(intraday_lookback_days), 1)
        self.daily_lookback_days = max(int(daily_lookback_days), 30)
        self.signals_mode = str(signals_mode or "observatory").strip().lower()

        self._status = MarketPollerStatus()
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._next_run_ts = time.time() + 2.0
        self._next_universe_refresh_ts = 0.0

        self._symbols: list[str] = []
        self._history_cursor = 0
        self._last_history_sync: dict[str, int] = {}

        self._minute_buckets: dict[int, dict[str, dict]] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="market-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def get_status(self) -> dict:
        with self._status_lock:
            return {
                "last_ok_ts": self._status.last_ok_ts,
                "last_error": self._status.last_error,
                "consecutive_failures": self._status.consecutive_failures,
                "backoff_until_ts": self._status.backoff_until_ts,
                "last_universe_refresh_ts": self._status.last_universe_refresh_ts,
                "last_quote_poll_ts": self._status.last_quote_poll_ts,
                "last_bar_close_ts": self._status.last_bar_close_ts,
                "last_feature_ts": self._status.last_feature_ts,
                "last_regime_ts": self._status.last_regime_ts,
                "last_events_ts": self._status.last_events_ts,
                "last_alerts_ts": self._status.last_alerts_ts,
                "universe_size": self._status.universe_size,
                "tracked_realtime_symbols": self._status.tracked_realtime_symbols,
                "last_quotes_seen": self._status.last_quotes_seen,
                "last_closed_bars": self._status.last_closed_bars,
                "last_bootstrap_symbols": self._status.last_bootstrap_symbols,
                "poll_seconds": self.poll_seconds,
                "universe_refresh_seconds": self.universe_refresh_seconds,
                "universe_symbol_limit": self.universe_symbol_limit,
                "realtime_symbol_limit": self.realtime_symbol_limit,
                "history_bootstrap_batch": self.history_bootstrap_batch,
                "intraday_lookback_days": self.intraday_lookback_days,
                "daily_lookback_days": self.daily_lookback_days,
                "signals_mode": self.signals_mode,
            }

    def _set_status(self, **kwargs) -> None:
        with self._status_lock:
            for key, value in kwargs.items():
                setattr(self._status, key, value)

    def _refresh_universe_if_needed(self, session, *, now_ts: int) -> dict:
        refreshed = False
        refresh_rows = 0

        if now_ts >= int(self._next_universe_refresh_ts) or not self._symbols:
            universe_rows = self.provider.get_universe()
            if universe_rows:
                refresh_rows = upsert_symbols(session, universe_rows)
                refreshed = True
            self._next_universe_refresh_ts = float(now_ts + self.universe_refresh_seconds)

        active_symbols = get_active_symbols(session, limit=self.universe_symbol_limit)
        self._symbols = [str(symbol).upper() for symbol in active_symbols if symbol]

        if refreshed:
            self._set_status(last_universe_refresh_ts=now_ts)

        return {
            "refreshed": refreshed,
            "rows": int(refresh_rows),
            "universe_size": len(self._symbols),
        }

    def _add_quote(self, quote: dict, *, now_ts: int) -> None:
        symbol = str(quote.get("symbol") or "").upper().strip()
        if not symbol:
            return

        quote_ts = int(quote.get("ts") or now_ts)
        price = float(quote.get("price") or 0.0)
        if price <= 0:
            return
        volume = float(quote.get("volume") or 0.0)

        minute_ts = align_bar_ts(quote_ts, 60)
        bucket = self._minute_buckets.setdefault(minute_ts, {})
        current = bucket.get(symbol)
        if current is None:
            bucket[symbol] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": max(volume, 0.0),
            }
            return

        current["high"] = max(float(current["high"]), price)
        current["low"] = min(float(current["low"]), price)
        current["close"] = price
        current["volume"] = float(current["volume"]) + max(volume, 0.0)

    def _close_finished_minutes(self, session, *, now_ts: int) -> dict:
        current_minute = align_bar_ts(now_ts, 60)
        close_minutes = sorted([minute_ts for minute_ts in self._minute_buckets.keys() if minute_ts < current_minute])
        if not close_minutes:
            return {
                "closed_bars": 0,
                "touched_symbols": [],
                "last_closed_minute": None,
                "aggregated_5m": 0,
                "aggregated_1d": 0,
            }

        touched_symbols: set[str] = set()
        bars_to_write: list[dict] = []
        for minute_ts in close_minutes:
            bucket = self._minute_buckets.pop(minute_ts, {})
            for symbol, agg in bucket.items():
                touched_symbols.add(str(symbol).upper())
                bars_to_write.append(
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

        upsert_market_bars(
            session,
            bars_to_write,
            now_ts=now_ts,
            intraday_lookback_days=self.intraday_lookback_days,
            daily_lookback_days=self.daily_lookback_days,
        )

        aggregated_5m = 0
        aggregated_1d = 0
        touched_list = sorted(touched_symbols)
        for minute_ts in close_minutes:
            agg_result = aggregate_higher_timeframes(
                session,
                symbols=touched_list,
                closed_minute_ts=int(minute_ts),
                now_ts=now_ts,
                intraday_lookback_days=self.intraday_lookback_days,
                daily_lookback_days=self.daily_lookback_days,
            )
            aggregated_5m += int(agg_result.get("rows_5m") or 0)
            aggregated_1d += int(agg_result.get("rows_1d") or 0)

        return {
            "closed_bars": len(bars_to_write),
            "touched_symbols": touched_list,
            "last_closed_minute": int(close_minutes[-1]),
            "aggregated_5m": int(aggregated_5m),
            "aggregated_1d": int(aggregated_1d),
        }

    def _run_bootstrap(self, session, *, now_ts: int) -> dict:
        if self.history_bootstrap_batch <= 0 or not self._symbols:
            return {"processed": 0, "rows": 0}

        processed = 0
        rows_written = 0
        attempts = 0
        max_attempts = max(self.history_bootstrap_batch * 3, 3)

        while processed < self.history_bootstrap_batch and attempts < max_attempts and self._symbols:
            attempts += 1
            symbol = self._symbols[self._history_cursor % len(self._symbols)]
            self._history_cursor += 1
            last_sync_ts = int(self._last_history_sync.get(symbol, 0) or 0)
            if last_sync_ts and (now_ts - last_sync_ts) < self.history_refresh_seconds:
                continue

            try:
                result = bootstrap_symbol_history(
                    self.provider,
                    session,
                    symbol=symbol,
                    now_ts=now_ts,
                    intraday_lookback_days=self.intraday_lookback_days,
                    daily_lookback_days=self.daily_lookback_days,
                )
            except Exception as exc:
                logger.warning("History bootstrap failed for %s: %s", symbol, exc)
                continue

            self._last_history_sync[symbol] = now_ts
            processed += 1
            rows_written += int(result.get("rows") or 0)

        return {
            "processed": int(processed),
            "rows": int(rows_written),
        }

    def _run_cycle(self) -> dict:
        now_ts = int(time.time())

        with session_scope() as session:
            universe_result = self._refresh_universe_if_needed(session, now_ts=now_ts)
            symbols_for_quotes = self._symbols[: self.realtime_symbol_limit]
            quotes = self.provider.poll_latest_quotes(symbols_for_quotes)
            for quote in quotes:
                self._add_quote(quote, now_ts=now_ts)

            closed_result = self._close_finished_minutes(session, now_ts=now_ts)

            feature_result = {"updated": 0}
            regime_result = {}
            events_result = {"inserted": 0}
            alerts_result = {"sent": 0}

            if int(closed_result.get("closed_bars") or 0) > 0:
                feature_result = recompute_market_features(
                    session,
                    symbols=list(closed_result.get("touched_symbols") or []),
                    now_ts=now_ts,
                )
                regime_result = recompute_market_regime(session, now_ts=now_ts)
                events_result = detect_market_events(
                    session,
                    now_ts=now_ts,
                    symbols_hint=list(closed_result.get("touched_symbols") or []),
                )
                alerts_result = self.alerts_service.run(session, now_ts=now_ts)

            bootstrap_result = self._run_bootstrap(session, now_ts=now_ts)

        self._set_status(
            last_ok_ts=now_ts,
            last_error=None,
            consecutive_failures=0,
            backoff_until_ts=None,
            last_quote_poll_ts=now_ts,
            last_bar_close_ts=(
                int(closed_result.get("last_closed_minute")) if closed_result.get("last_closed_minute") else None
            ),
            last_feature_ts=(now_ts if int(feature_result.get("updated") or 0) > 0 else self._status.last_feature_ts),
            last_regime_ts=(now_ts if regime_result else self._status.last_regime_ts),
            last_events_ts=(now_ts if int(events_result.get("inserted") or 0) > 0 else self._status.last_events_ts),
            last_alerts_ts=(now_ts if int(alerts_result.get("sent") or 0) > 0 else self._status.last_alerts_ts),
            universe_size=int(universe_result.get("universe_size") or len(self._symbols)),
            tracked_realtime_symbols=len(symbols_for_quotes),
            last_quotes_seen=len(quotes),
            last_closed_bars=int(closed_result.get("closed_bars") or 0),
            last_bootstrap_symbols=int(bootstrap_result.get("processed") or 0),
        )

        return {
            "ok": True,
            "now_ts": now_ts,
            "universe": universe_result,
            "quotes": {
                "requested_symbols": len(symbols_for_quotes),
                "quotes_seen": len(quotes),
            },
            "bars": closed_result,
            "features": feature_result,
            "regime": regime_result,
            "events": events_result,
            "alerts": alerts_result,
            "bootstrap": bootstrap_result,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._status_lock:
                backoff_until = self._status.backoff_until_ts

            if backoff_until and int(now) < int(backoff_until):
                time.sleep(1)
                continue

            if now >= self._next_run_ts:
                try:
                    result = self._run_cycle()
                    logger.debug(
                        "Market poller cycle: quotes=%s bars=%s features=%s events=%s alerts=%s",
                        result.get("quotes", {}).get("quotes_seen"),
                        result.get("bars", {}).get("closed_bars"),
                        result.get("features", {}).get("updated"),
                        result.get("events", {}).get("inserted"),
                        result.get("alerts", {}).get("sent"),
                    )
                    self._next_run_ts = time.time() + self.poll_seconds
                except Exception as exc:
                    fail_ts = int(time.time())
                    with self._status_lock:
                        self._status.consecutive_failures += 1
                        fail_count = self._status.consecutive_failures
                    backoff_seconds = min(self.poll_seconds * (2 ** min(fail_count, 5)), 600)
                    self._set_status(
                        last_error=str(exc),
                        backoff_until_ts=fail_ts + int(backoff_seconds),
                    )
                    logger.error("Market poller cycle failed: %s", exc)
                    self._next_run_ts = time.time() + max(backoff_seconds, self.poll_seconds)

            time.sleep(1)
