# Arrowhead

## overview
Arrowhead is a macro telescope for observing market structure and regime context.
The UI uses a command-terminal style that is primarily monochrome (black/white) with limited semantic green/red states.

It is observational only:

- no broker connections
- no order placement
- no portfolio management
- no execution logic
- no trading recommendations

Default operation uses free Yahoo Finance daily bars via `yfinance`.
Optional intraday mode adds 1-minute bar ingestion (rolling 7 days) and dynamic 5m/15m resampling for symbol history views.

## governance policy
If code changes schema, config, feature logic, routes, or poller behavior:

- `README.md` must be updated in the same change.
- changelog must append a new minor-version entry with date, summary, and technical details.

## architecture
Core modules:

- `services/market_data.py`
- `services/poller.py`
- `services/features.py`
- `services/db.py`
- `services/intraday.py`
- `app.py`

Daily poll flow:

1. load configured `DAILY_SYMBOLS`
2. enforce per-symbol history floor (`REQUIRED_TRADING_DAYS`) before feature updates
3. compute per-symbol fetch window:
   first run: `today - DAILY_BACKFILL_DAYS`
   incremental run: `last_bar_date - DAILY_PATCH_BUFFER_DAYS`
   history-short symbols: extended backfill window until they meet the history floor
4. fetch daily OHLCV (`1d`) from Yahoo Finance with retry/backoff
5. upsert into `market_bars` using PK dedupe (`symbol,timeframe,bar_ts`)
6. recompute daily features for all dirty symbols (new bars and/or insufficient feature coverage)
7. upsert `market_regime` per trading date (`ts`) so breadth history is always available for charts

Poll cadence can be frequent (default 10 minutes); most cycles no-op outside new daily prints.

Intraday poll flow (optional):

1. read `INTRADAY_ENABLED` and `INTRADAY_SYMBOLS`
2. fetch rolling `1m` bars from Yahoo (`period=INTRADAY_LOOKBACK_DAYS d`)
3. upsert only new rows into `intraday_bars` (PK dedupe on `symbol,bar_ts`)
4. keep 5m/15m virtual only (computed at API time from stored 1m rows)

## data provider
Default provider: `yahoo`

- no API keys required
- uses `yfinance.download(..., interval="1d")`
- handles single and multi-ticker responses
- missing symbol data is skipped with warning (no crash)

If `DATA_PROVIDER` is set to a non-yahoo value, Arrowhead falls back to yahoo in default mode.

## configuration
Use `.env` (or `.env.example` as template):

- `DATA_PROVIDER=yahoo`
- `DAILY_SYMBOLS=SPY,QQQ,IWM,TLT,GLD`
- `TIMEFRAME=1d`
- `POLL_SECONDS=600`
- `UNIVERSE_REFRESH_SECONDS=1800`
- `DAILY_LOOKBACK_DAYS=1825`
- `DAILY_BACKFILL_DAYS=420`
- `DAILY_PATCH_BUFFER_DAYS=7`
- `REQUIRED_TRADING_DAYS=520`
- `FORECAST_LOOKBACK_DAYS=504`
- `FORECAST_HORIZON_DAYS=5`
- `FORECAST_MIN_SAMPLES=60`
- `YAHOO_RETRY_ATTEMPTS=3`
- `YAHOO_RETRY_BACKOFF_SECONDS=1.5`
- `INTRADAY_ENABLED=false`
- `INTRADAY_SYMBOLS=SPY,QQQ,IWM,TLT,GLD`
- `INTRADAY_LOOKBACK_DAYS=7`
- `DISCORD_WEBHOOK_URL=` (optional)
- `DISCORD_SENDER_ENABLED=true`
- `DISCORD_SENDER_MAX_CHARS=500`
- `DISCORD_SENDER_COOLDOWN_SECONDS=5`

Backfill behavior:

- on a fresh DB, Arrowhead backfills historical daily bars (default 420 days) so long-window context features populate quickly.
- after initial fill, incremental polling fetches only a small patch window before the last known bar to avoid gaps.
- if a symbol is still below `REQUIRED_TRADING_DAYS`, Arrowhead automatically re-enters deep backfill for that symbol and recomputes features until the floor is met.

Port behavior:

- `PORT` is loaded from `.env`
- runtime fallback/default port is `5003`

## schema
### `market_bars`
- PK: (`symbol`, `timeframe`, `bar_ts`)
- fields: `open`, `high`, `low`, `close`, `volume`
- timeframe used by Arrowhead: `1d`

### `intraday_bars`
- PK: (`symbol`, `bar_ts`)
- fields: `open`, `high`, `low`, `close`, `volume`
- cadence used by Arrowhead intraday mode: `1m` only
- retention source constraint: Yahoo `1m` bars are provider-limited to a rolling ~7 day window

### `market_features`
- PK: (`symbol`, `timeframe`, `bar_ts`)
- fields:
  - `ema_20`, `ema_50`, `ema_200`
  - `trend_slope_20d`
  - `trend_strength_200d`
  - `realized_vol_20d`
  - `vol_percentile_252d`
  - `dist_from_52w_high`
  - `drawdown_252d`
  - `return_1d`, `return_5d`, `return_20d`
  - `regime_tag` (`BULL|BEAR|CHOP`)
  - `cont_prob_5d`, `exp_return_5d`, `cont_samples_5d`
  - `mom_strength`

### `market_regime`
- PK: `ts` (daily trading date label stored as midnight-UTC timestamp)
- fields:
  - `breadth_pct_above_ema20`
  - `breadth_pct_above_ema50`
  - `breadth_pct_above_ema200`
  - `pct_with_positive_20d_return`
  - `new_20d_high_pct`
  - `new_20d_low_pct`
  - `median_volatility_20d`
  - `regime_tag` (`BULL|BEAR|CHOP`)

Market regime rules:

- symbol-level structure: `BULL` when `close > ema_50` and `trend_slope_20d > 0`
- symbol-level structure: `BEAR` when `close < ema_50` and `trend_slope_20d < 0`
- market `regime_tag` is aggregated from symbol structure shares (otherwise `CHOP`)

## feature definitions
Feature windows are trading-day windows (`1d` cadence).

- `trend_slope_20d`: slope of `log(close)` over trailing 20 trading days
- `trend_strength_200d`: slope of `log(close)` over trailing 200 trading days
- `realized_vol_20d`: standard deviation of daily log returns over trailing 20 trading days (non-annualized)
- `vol_percentile_252d`: percentile rank of current `vol_20d` in trailing 252 trading days
- `dist_from_52w_high`: `(close - rolling_252d_high) / rolling_252d_high`
- `drawdown_252d`: `(rolling_max_252d - close) / rolling_max_252d`
- `return_1d/5d/20d`: close-to-close percent returns (fractional)
- `cont_prob_5d`: empirical continuation estimate
  - `P(next_5d_return > 0 | current_state)` where state is `UPTREND|DOWNTREND|CHOP`
  - state is defined by `close vs ema_50` and `trend_slope_20d` sign
  - estimated from symbol-local historical matches over `FORECAST_LOOKBACK_DAYS`
  - requires at least `FORECAST_MIN_SAMPLES`, else `NULL`
- `exp_return_5d`: mean of matched historical next-5d returns
- `mom_strength`: `clip(return_20d / max(realized_vol_20d*sqrt(20), eps), -3, 3)`

Insufficient history leaves affected feature fields as `NULL`.

Daily feature windows and forecast lookback are row-based trading-day windows at `1d` cadence.

## routes
UI:

- `GET /dashboard`
- `GET /symbol/<symbol>`

Dashboard panels:

- status bar
- discord sender (manual message only)
- market weather KPIs
- breadth depth KPIs
- market structure chart
- symbol trend table

Symbol panels:

- price + trend (EMA20/50/200 with subtle regime shading)
- context (52w distance, vol percentile, 200d trend strength, drawdown)
- risk (volatility + drawdown chart)
- historical continuation (descriptive only)

API:

- `GET /api/status`
- `GET /api/market/regime/latest`
- `GET /api/market/regime/history?limit=400`
- `GET /api/market_structure?days=60`
- `GET /api/symbol/<symbol>/history?limit=400&tf=1d|1w|1m|5m|15m`
- `POST /api/discord/send`

`POST /api/discord/send` request:

- JSON body: `{ "message": "..." }`
- requires `DISCORD_WEBHOOK_URL` and `DISCORD_SENDER_ENABLED=true`
- validation: non-empty message, max length `DISCORD_SENDER_MAX_CHARS`
- cooldown: `DISCORD_SENDER_COOLDOWN_SECONDS` (returns HTTP `429` with `{"ok": false, "error": "cooldown"}` when hit)
- behavior: manual-only sender; no automated alerts are triggered by this route

`/api/symbol/<symbol>/history?limit=400&tf=...` response (ascending by date):

- `date` (`YYYY-MM-DD`)
- `close`
- `ema_20`
- `ema_50`
- `ema_200`
- `vol_20d`
- `drawdown_252d`
- `trend_strength_200d`
- `vol_percentile_252d`
- `dist_from_52w_high`
- `regime_tag`

For intraday `tf=1m|5m|15m`, response rows use:

- `date` (`YYYY-MM-DDTHH:MM:SSZ`)
- `open`, `high`, `low`, `close`, `volume`

Timeframe toggle:

- `tf=1d` reads persisted daily bars/features.
- `tf=1w` dynamically resamples weekly bars from daily history and computes weekly EMA20/EMA50/EMA200 on demand.
- weekly data is not persisted.
- `tf=1m` reads persisted `intraday_bars` rows (ascending, capped/downsampled for UI safety).
- `tf=5m` and `tf=15m` are dynamically resampled from stored `1m` bars via pandas (`first/max/min/last/sum`) and are not persisted.

Intraday API examples:

- `/api/symbol/SPY/history?tf=1m`
- `/api/symbol/SPY/history?tf=5m`
- `/api/symbol/SPY/history?tf=15m`

`/api/market/regime/history?limit=400` response (ascending by date):

- `date` (`YYYY-MM-DD`)
- `breadth_pct_above_ema20`
- `breadth_pct_above_ema50`
- `breadth_pct_above_ema200`
- `pct_with_positive_20d_return`
- `new_20d_high_pct`
- `new_20d_low_pct`
- `median_volatility_20d`
- `regime_tag`

`/api/market_structure?days=60` response (ascending by date):

- `labels`: date labels (`YYYY-MM-DD`)
- `pct_above_ema20`
- `pct_above_ema50`
- `pct_above_ema200`
- `message`: `"warming up: need more history"` when fewer than 2 points are available

Charts:

- Chart.js loaded via CDN in base template.
- `/symbol/<symbol>` renders price+EMA20/50/200 with subtle regime shading and a risk chart.
- `/dashboard` renders market structure breadth lines.

`/api/status` includes:

- provider name
- timeframe
- `last_daily_bar_ts`
- `last_poll_ok_ts`
- `last_error`
- counts: symbols, bars_1d, features_1d, market_regime_rows
- forecast fields: `forecasts_ready_symbols`, `forecast_lookback_days`, `forecast_horizon_days`

## run
1. install dependencies:

```bash
pip install -r requirements.txt
```

2. start app:

```bash
python3 app.py
```

3. open:

- `http://localhost:5003/dashboard`
- `http://localhost:5003/api/status`

## smoke test
Run the lightweight Arrowhead smoke script:

```bash
python3 scripts/smoke_arrowhead.py
```

It verifies:

- Yahoo provider fetch for SPY (multi-month daily window)
- bar upsert + feature recompute
- `/api/status` reports provider `yahoo`
- `/api/status` has non-null `last_daily_bar_ts`

## changelog
### v0.16.0 - 2026-02-25
Summary:
- Added lightweight intraday mode with 1-minute storage and dynamic 5m/15m resampling, isolated from daily systems.

Technical details:
- Added new `intraday_bars` table (`symbol,bar_ts` PK) for isolated intraday persistence.
- Added Yahoo `1m` intraday fetcher with retry/backoff and normalization/dedupe to UTC-sorted OHLCV.
- Added idempotent intraday upsert path that inserts only new rows per symbol and logs fetched/inserted/last-ts.
- Added poller integration gated by `INTRADAY_ENABLED` with configurable `INTRADAY_SYMBOLS` and `INTRADAY_LOOKBACK_DAYS`.
- Extended `/api/symbol/<symbol>/history` to support `tf=1m|5m|15m`; 5m/15m are computed dynamically from stored 1m bars and are not persisted.

### v0.16.4 - 2026-02-25
Summary:
- Fixed empty Market Structure Chart by persisting breadth history by trading date and exposing a dedicated chart series endpoint.

Technical details:
- Poller now upserts `market_regime` across missing feature dates (not just latest), then refreshes the latest regime row each cycle.
- Added `GET /api/market_structure?days=60` returning chart-ready arrays for `% > EMA20/50/200`.
- Updated dashboard chart fetch to use `/api/market_structure`; when fewer than 2 points exist, UI shows `warming up: need more history`.

### v0.16.3 - 2026-02-25
Summary:
- Fixed daily warmup so backfill and feature recompute are enforced for every configured symbol, not just the first symbol with updates.

Technical details:
- Poller now enforces a per-symbol history floor via `REQUIRED_TRADING_DAYS` (default `520`) before treating the symbol as warmed up.
- Added per-symbol ensure-history fetch windows and dirty-symbol recompute logic for missing feature coverage.
- Feature recompute now runs for all dirty symbols and persists full available daily feature history for those symbols.
- Added per-cycle proof logs with per-symbol fetched window, bar counts, and feature counts.

### v0.16.2 - 2026-02-25
Summary:
- Added a manual Discord sender panel so Arrowhead can send user-written messages via webhook.

Technical details:
- Added `services/discord_sender.py` with `send_discord_message(webhook_url, content)` helper.
- Added `POST /api/discord/send` with JSON validation, max-length guard, and cooldown throttling.
- Added `/dashboard` Discord Sender panel with textarea + send button (enabled only when configured).
- Added new env controls: `DISCORD_SENDER_ENABLED`, `DISCORD_SENDER_MAX_CHARS`, and `DISCORD_SENDER_COOLDOWN_SECONDS`.
- Sender remains manual-only; no new automated Discord behavior was introduced.

### v0.16.1 - 2026-02-25
Summary:
- Switched Arrowhead UI to a black & white terminal theme with limited semantic color.

Technical details:
- Removed neon green accents from panel chrome, headings, badges, and chart styling.
- Standardized borders and gridlines to shared white grid variables.
- Updated badges/buttons/tables to monochrome defaults with semantic `good`/`bad` states.
- Added shared `static/charts.js` helpers so Chart.js ticks/gridlines follow monochrome CSS variables.

### v0.16.0 - 2026-02-25
Summary:
- Refined Arrowhead into telescope-mode macro observatory.
- Added 200-day context, 52-week distance, volatility percentile, expanded breadth metrics, weekly toggle.

Technical details:
- Extended `market_features` with `ema_200`, `dist_from_52w_high`, `vol_percentile_252d`, and `trend_strength_200d`.
- Expanded `market_regime` breadth calculations (`>ema20/50/200`, positive 20d, new 20d highs/lows).
- Added EMA200 overlays and regime shading on symbol charts.
- Added dynamic weekly resampling for `/api/symbol/<symbol>/history?tf=1w`.
- Reordered dashboard/symbol panels for telescope-mode structure and de-emphasized continuation output.

### v0.15.3 - 2026-02-25
Summary:
- Added time-series charts to symbol and dashboard pages.

Technical details:
- Introduced `/api/symbol/<symbol>/history` endpoint with ascending daily chart payload.
- Introduced `/api/market/regime/history` endpoint for daily breadth/volatility series.
- Integrated Chart.js via CDN.
- Rendered price, EMA, volatility, drawdown, and market breadth charts in monochrome style.

### v0.15.2 - 2026-02-25
Summary:
- Dashboard refreshed into a command-terminal layout with clearer framed panels.
- Added empirical 5-day momentum continuation forecasts (probability + expected return) for each symbol.

Technical details:
- UI: introduced reusable panel/grid components with monochrome terminal styling and compact status badges.
- Features: added `cont_prob_5d`, `exp_return_5d`, `cont_samples_5d`, and `mom_strength` computed from historical state-matched outcomes.
- Schema: extended `market_features` with forecast columns and idempotent SQLite migration support.
- API: extended `/api/status` and symbol history payloads with forecast readiness and parameters.

### v0.15.1 - 2026-02-25
Summary:
- Centered and enlarged Arrowhead brand header; removed subtitle.
- Added automatic historical backfill for daily bars to ensure robust feature computation.

Technical details:
- UI header now uses centered `Arrowhead` brand layout with balanced left/right columns.
- Poller now fetches up to `DAILY_BACKFILL_DAYS` on first run.
- Incremental patch fetch uses `DAILY_PATCH_BUFFER_DAYS` to avoid gaps.
- Daily bars rely on PK dedupe and are no longer pruned during Arrowhead polling.
- Features recompute only when new bars are inserted.

### v0.15.0 - 2026-02-25
Summary:
- Rebuilt nocom into Arrowhead daily market trend observatory using free Yahoo Finance data.

Technical details:
- Provider: added yfinance-based Yahoo provider and made it default (no keys required).
- Pipeline: daily OHLCV persistence with idempotent dedupe and no-op polling when no new bar exists.
- Features: added daily trend/vol/drawdown/return features and per-symbol regime tags.
- Market weather: added market_regime daily summary (breadth + median vol + risk tag).
- UI/API: added `/dashboard` and `/symbol/<symbol>` for trend viewing; extended `/api/status`.

### v0.14.0 - 2026-02-24
Summary:
- switched phase 0 Polygon ingestion from snapshot to aggregates endpoints for plan compatibility.

Technical details:
- removed phase 0 usage of `/v2/snapshot/locale/us/markets/stocks/tickers`.
- added per-symbol aggregate polling via `/v2/aggs/ticker/{symbol}/range/1/minute/{from}/{to}`.
- added `/v2/aggs/ticker/{symbol}/prev` fallback when range returns 401/403.
- enforced closed-minute bar gating before feature/event recompute.
- recompute now runs only when new closed 1m bars are inserted.
- added poller status telemetry: `provider_endpoints_in_use`, `last_poll_ok_ts`, `last_provider_error`.
- added temporary per-symbol disable windows for repeated permission/rate-limit failures.

### v0.13.0 - 2026-02-24
Summary:
- fixed `.env` loading during startup.
- standardized runtime port handling to `PORT` with a single fallback.
- removed hardcoded legacy port usage across runtime/docs.
- enforced `5003` as the only fallback/default port.
