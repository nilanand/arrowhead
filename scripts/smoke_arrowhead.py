from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

# Ensure default provider in smoke context.
os.environ.setdefault("DATA_PROVIDER", "yahoo")
os.environ.setdefault("POLL_SECONDS", "600")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app, poller  # noqa: E402
from services.db import session_scope, upsert_market_bars, upsert_symbols  # noqa: E402
from services.features import recompute_market_features_daily  # noqa: E402
from services.market_data import build_market_data_provider  # noqa: E402


def _date_to_ts(value: dt.date) -> int:
    return int(dt.datetime.combine(value, dt.time.min, tzinfo=dt.timezone.utc).timestamp())


def main() -> None:
    provider = build_market_data_provider("yahoo")
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = end_date - dt.timedelta(days=420)

    bars_by_symbol = provider.fetch_daily_bars(
        ["SPY"],
        start_date=start_date,
        end_date=end_date,
    )
    spy_df = bars_by_symbol.get("SPY")
    assert spy_df is not None and not spy_df.empty, "Yahoo provider returned no SPY bars"

    rows = []
    for bar_date, row in spy_df.iterrows():
        rows.append(
            {
                "symbol": "SPY",
                "timeframe": "1d",
                "bar_ts": _date_to_ts(bar_date),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    with session_scope() as session:
        upsert_symbols(
            session,
            [
                {
                    "symbol": "SPY",
                    "exchange": "US",
                    "asset_type": "ETF",
                    "active": 1,
                    "last_seen_ts": int(dt.datetime.now(dt.timezone.utc).timestamp()),
                }
            ],
        )
        upsert_market_bars(
            session,
            rows,
            now_ts=int(dt.datetime.now(dt.timezone.utc).timestamp()),
            intraday_lookback_days=30,
            daily_lookback_days=1825,
        )
        recompute_market_features_daily(session, symbols=["SPY"], timeframe="1d", lookback_rows=500)
        poller._compute_market_regime(
            session,
            symbols=["SPY"],
            now_ts=int(dt.datetime.now(dt.timezone.utc).timestamp()),
        )

    # Run one poller cycle to compute market regime/status wiring.
    poller._run_cycle()

    with app.test_client() as client:
        resp = client.get("/api/status")
        assert resp.status_code == 200, f"Unexpected /api/status status: {resp.status_code}"
        payload = resp.get_json() or {}
        hist_resp = client.get("/api/symbol/SPY/history?limit=5")
        assert hist_resp.status_code == 200, f"Unexpected /api/symbol/SPY/history status: {hist_resp.status_code}"
        hist_payload = hist_resp.get_json() or []
        hist_weekly_resp = client.get("/api/symbol/SPY/history?limit=12&tf=1w")
        assert hist_weekly_resp.status_code == 200, f"Unexpected weekly /api/symbol/SPY/history status: {hist_weekly_resp.status_code}"
        hist_weekly_payload = hist_weekly_resp.get_json() or []
        regime_hist_resp = client.get("/api/market/regime/history?limit=5")
        assert regime_hist_resp.status_code == 200, f"Unexpected /api/market/regime/history status: {regime_hist_resp.status_code}"
        regime_hist_payload = regime_hist_resp.get_json() or []

    assert payload.get("provider_name") == "yahoo", "Expected provider_name=yahoo"
    assert payload.get("last_daily_bar_ts") is not None, "Expected non-null last_daily_bar_ts"
    assert "forecast_lookback_days" in payload, "Expected forecast_lookback_days in /api/status"
    assert "forecasts_ready_symbols" in payload, "Expected forecasts_ready_symbols in /api/status"
    rows = hist_payload if isinstance(hist_payload, list) else []
    if rows:
        first_row = rows[0]
        assert "date" in first_row, "Expected date in symbol history payload"
        assert "close" in first_row, "Expected close in symbol history payload"
        assert "ema_20" in first_row, "Expected ema_20 in symbol history payload"
        assert "ema_200" in first_row, "Expected ema_200 in symbol history payload"
        assert "vol_20d" in first_row, "Expected vol_20d in symbol history payload"
        assert "dist_from_52w_high" in first_row, "Expected dist_from_52w_high in symbol history payload"
        dates = [str(row.get("date") or "") for row in rows]
        assert dates == sorted(dates), "Expected symbol history sorted ascending by date"
    weekly_rows = hist_weekly_payload if isinstance(hist_weekly_payload, list) else []
    if weekly_rows:
        weekly_dates = [str(row.get("date") or "") for row in weekly_rows]
        assert weekly_dates == sorted(weekly_dates), "Expected weekly symbol history sorted ascending by date"
    regime_rows = regime_hist_payload if isinstance(regime_hist_payload, list) else []
    if regime_rows:
        first_regime = regime_rows[0]
        assert "date" in first_regime, "Expected date in market regime history payload"
        assert "breadth_pct_above_ema20" in first_regime, "Expected breadth ema20 in regime history payload"
        assert "breadth_pct_above_ema50" in first_regime, "Expected breadth in regime history payload"
        assert "breadth_pct_above_ema200" in first_regime, "Expected breadth ema200 in regime history payload"
        regime_dates = [str(row.get("date") or "") for row in regime_rows]
        assert regime_dates == sorted(regime_dates), "Expected regime history sorted ascending by date"

    print("smoke ok: provider=yahoo last_daily_bar_ts present")


if __name__ == "__main__":
    main()
