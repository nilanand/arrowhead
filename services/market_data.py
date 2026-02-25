from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_DAILY_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
DEFAULT_INTRADAY_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


def parse_daily_symbols(raw: str | None) -> list[str]:
    tokens = [
        str(token).strip().upper()
        for token in str(raw or "").split(",")
        if str(token).strip()
    ]
    deduped = list(dict.fromkeys(tokens))
    return deduped if deduped else list(DEFAULT_DAILY_SYMBOLS)


def parse_intraday_symbols(raw: str | None) -> list[str]:
    tokens = [
        str(token).strip().upper()
        for token in str(raw or "").split(",")
        if str(token).strip()
    ]
    deduped = list(dict.fromkeys(tokens))
    return deduped if deduped else list(DEFAULT_INTRADAY_SYMBOLS)


@dataclass
class MarketDataProviderConfig:
    provider_name: str = "yahoo"
    symbols: list[str] | None = None
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.5


class MarketDataProvider(Protocol):
    name: str

    def fetch_daily_bars(
        self,
        symbols: list[str],
        *,
        start_date: dt.date,
        end_date: dt.date,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def fetch_intraday_1m(self, symbol: str, *, lookback_days: int) -> pd.DataFrame:
        raise NotImplementedError


class YahooMarketDataProvider:
    name = "yahoo"

    def __init__(self, cfg: MarketDataProviderConfig) -> None:
        self.cfg = cfg
        self._symbols = cfg.symbols or list(DEFAULT_DAILY_SYMBOLS)

    def _download(self, tickers: str, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        attempts = max(int(self.cfg.retry_attempts), 1)
        delay_base = max(float(self.cfg.retry_backoff_seconds), 0.1)
        end_plus_one = end_date + dt.timedelta(days=1)

        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                frame = yf.download(
                    tickers=tickers,
                    start=start_date.isoformat(),
                    end=end_plus_one.isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                if isinstance(frame, pd.DataFrame):
                    return frame
                return pd.DataFrame()
            except Exception as exc:  # pragma: no cover - network/provider variability
                last_err = exc
                logger.warning(
                    "Yahoo download attempt %s/%s failed for tickers=%s: %s",
                    attempt,
                    attempts,
                    tickers,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(delay_base * attempt)

        if last_err is not None:
            raise last_err
        return pd.DataFrame()

    @staticmethod
    def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        lower_map: dict[str, str] = {}
        for col in frame.columns:
            key = str(col).strip().lower().replace(" ", "")
            lower_map[key] = str(col)

        required = {
            "open": lower_map.get("open"),
            "high": lower_map.get("high"),
            "low": lower_map.get("low"),
            "close": lower_map.get("close"),
            "volume": lower_map.get("volume"),
        }
        if any(src is None for src in required.values()):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        out = pd.DataFrame(index=frame.index)
        for dst, src in required.items():
            out[dst] = pd.to_numeric(frame[str(src)], errors="coerce")

        dt_idx = pd.to_datetime(out.index, errors="coerce", utc=True)
        out.index = dt_idx
        out = out[~out.index.isna()]
        out = out.sort_index()

        out = out.dropna(subset=["open", "high", "low", "close"])
        out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
        out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

        # Treat each bar as a trading-date label.
        out.index = out.index.date
        out = out[~out.index.duplicated(keep="last")]
        return out[["open", "high", "low", "close", "volume"]]

    # --- INTRADAY: yfinance 1m fetcher (v0.16.0) ---
    @staticmethod
    def _normalize_intraday_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        lower_map: dict[str, str] = {}
        for col in frame.columns:
            key = str(col).strip().lower().replace(" ", "")
            lower_map[key] = str(col)

        required = {
            "open": lower_map.get("open"),
            "high": lower_map.get("high"),
            "low": lower_map.get("low"),
            "close": lower_map.get("close"),
            "volume": lower_map.get("volume"),
        }
        if any(src is None for src in required.values()):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        out = pd.DataFrame(index=frame.index)
        for dst, src in required.items():
            out[dst] = pd.to_numeric(frame[str(src)], errors="coerce")

        dt_idx = pd.to_datetime(out.index, errors="coerce", utc=True)
        out.index = dt_idx
        out = out[~out.index.isna()]
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]

        out = out.dropna(subset=["open", "high", "low", "close"])
        out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
        out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

        # Store as naive UTC timestamps for consistent SQLite DateTime serialization.
        out.index = out.index.tz_convert("UTC").tz_localize(None)
        return out[["open", "high", "low", "close", "volume"]]

    def fetch_intraday_1m(self, symbol: str, *, lookback_days: int) -> pd.DataFrame:
        symbol_upper = str(symbol or "").strip().upper()
        if not symbol_upper:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        attempts = max(int(self.cfg.retry_attempts), 1)
        delay_base = max(float(self.cfg.retry_backoff_seconds), 0.1)
        period_days = max(int(lookback_days), 1)

        for attempt in range(1, attempts + 1):
            try:
                frame = yf.download(
                    tickers=symbol_upper,
                    period=f"{period_days}d",
                    interval="1m",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                raw = self._extract_symbol_frame(frame, symbol_upper, symbol_count=1)
                normalized = self._normalize_intraday_ohlcv(raw)
                if normalized.empty:
                    logger.warning(
                        "Yahoo 1m returned no rows for symbol=%s lookback_days=%s",
                        symbol_upper,
                        period_days,
                    )
                return normalized
            except Exception as exc:  # pragma: no cover - provider/network variability
                logger.warning(
                    "Yahoo 1m attempt %s/%s failed for symbol=%s: %s",
                    attempt,
                    attempts,
                    symbol_upper,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(delay_base * attempt)

        logger.warning("Yahoo 1m fetch failed for symbol=%s after retries", symbol_upper)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # --- END INTRADAY ---

    @staticmethod
    def _extract_symbol_frame(raw: pd.DataFrame, symbol: str, *, symbol_count: int) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(str(v).upper() for v in raw.columns.get_level_values(0))
            level1 = set(str(v).upper() for v in raw.columns.get_level_values(1))

            if str(symbol).upper() in level0:
                return raw[str(symbol).upper()].copy()
            if str(symbol).upper() in level1:
                return raw.xs(str(symbol).upper(), axis=1, level=1, drop_level=True).copy()
            return pd.DataFrame()

        if int(symbol_count) == 1:
            return raw.copy()
        return pd.DataFrame()

    def fetch_daily_bars(
        self,
        symbols: list[str],
        *,
        start_date: dt.date,
        end_date: dt.date,
    ) -> dict[str, pd.DataFrame]:
        active_symbols = [str(symbol).upper() for symbol in (symbols or self._symbols) if symbol]
        if not active_symbols:
            return {}

        tickers = " ".join(active_symbols)
        output: dict[str, pd.DataFrame] = {}

        try:
            batch = self._download(tickers, start_date=start_date, end_date=end_date)
        except Exception as exc:
            logger.error("Yahoo batch download failed for %s: %s", tickers, exc)
            batch = pd.DataFrame()

        for symbol in active_symbols:
            raw_symbol = self._extract_symbol_frame(batch, symbol, symbol_count=len(active_symbols))
            normalized = self._normalize_ohlcv(raw_symbol)
            if not normalized.empty:
                output[symbol] = normalized
                continue

            # Per-symbol retry fallback so one bad symbol never breaks the cycle.
            try:
                single = self._download(symbol, start_date=start_date, end_date=end_date)
                single_frame = self._extract_symbol_frame(single, symbol, symbol_count=1)
                normalized_single = self._normalize_ohlcv(single_frame)
                if not normalized_single.empty:
                    output[symbol] = normalized_single
                    continue
            except Exception as exc:
                logger.warning("Yahoo single-symbol download failed for %s: %s", symbol, exc)

            logger.warning("Yahoo returned no daily bars for symbol=%s in range %s..%s", symbol, start_date, end_date)

        return output


def build_market_data_provider(provider_name: str | None) -> MarketDataProvider:
    configured_symbols = parse_daily_symbols(os.getenv("DAILY_SYMBOLS"))
    normalized = str(provider_name or os.getenv("DATA_PROVIDER") or "yahoo").strip().lower()

    if normalized != "yahoo":
        logger.warning(
            "DATA_PROVIDER=%s is not enabled in arrowhead default mode. Falling back to yahoo.",
            normalized,
        )
        normalized = "yahoo"

    cfg = MarketDataProviderConfig(
        provider_name=normalized,
        symbols=configured_symbols,
        retry_attempts=int(os.getenv("YAHOO_RETRY_ATTEMPTS", "3")),
        retry_backoff_seconds=float(os.getenv("YAHOO_RETRY_BACKOFF_SECONDS", "1.5")),
    )

    logger.info("Market data provider selected: yahoo")
    return YahooMarketDataProvider(cfg)
