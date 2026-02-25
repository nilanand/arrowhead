from __future__ import annotations

import json
import logging
import math
from bisect import bisect_left
from dataclasses import dataclass

from sqlalchemy import bindparam, case, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import (
    BATCH_SIZE,
    ItemBar,
    ItemFeature,
    ItemForecast,
    ItemSpreadSnapshot,
    NocomModelState,
    NocomTrainPair,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-9
Q10_Z = 1.2816
Q_SPREAD_SIGMA_DENOM = 2.5632
ONLINE_QUANTILES = (0.1, 0.5, 0.9)


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _clamp_01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _sigmoid(x: float) -> float:
    x = _clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-x))


def _safe_int_list(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        n = max(int(value), 1)
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return tuple(sorted(out))


def _normalize_mode(value: str) -> str:
    mode = (value or "auto").strip().lower()
    if mode in {"auto", "baseline", "online"}:
        return mode
    return "auto"


def _logret_to_frac(y_hat: float) -> float:
    clipped = _clip(float(y_hat), -2.0, 2.0)
    return float(math.exp(clipped) - 1.0)


@dataclass
class NocomConfig:
    enabled: bool = True
    top_n: int = 300
    horizons: tuple[int, ...] = (30, 120, 360)
    min_bars: int = 90
    risk_k: float = 0.5
    min_conf: float = 0.35
    sigma_mult: float = 1.0
    mr_k: float = 0.25
    trend_clip: float = 0.05
    spread_penalty_denom: float = 0.02
    w_trend_up: float = 0.70
    w_mr_up: float = 0.30
    w_trend_range: float = 0.40
    w_mr_range: float = 0.60
    w_trend_vol: float = 0.35
    w_mr_vol: float = 0.65
    conf_vol_penalty: float = 0.25
    liq_rank_min: float = 60.0
    max_vol_60m: float = 0.06

    # Mode control.
    mode: str = "auto"

    # Online training config.
    train_enabled: bool = True
    train_lookback_days: int = 14
    train_retention_days: int = 30
    train_batch: int = 500
    min_train_samples: int = 500
    min_updates: int = 200
    train_lr: float = 0.05
    lr_decay: float = 2000.0
    l2: float = 0.001
    w_clip: float = 10.0
    b_clip: float = 1.0
    cleanup_every_bars: int = 60
    bar_seconds: int = 60

    # Names/versions.
    baseline_model_name: str = "nocom_v1"
    online_model_name: str = "nocom_online_v1"
    feature_schema_version: str = "nocom_x_v1"
    baseline_features_version: str = "features_v1"
    baseline_model_version: str = "nocom_v1"
    online_model_version: str = "nocom_online_v1"

    # Backward-compat placeholder used by older callsites.
    model_name: str = "auto"

    def normalized(self) -> "NocomConfig":
        mode = _normalize_mode(self.mode)
        return NocomConfig(
            enabled=bool(self.enabled),
            top_n=max(int(self.top_n), 1),
            horizons=_safe_int_list(self.horizons),
            min_bars=max(int(self.min_bars), 1),
            risk_k=float(self.risk_k),
            min_conf=_clamp_01(float(self.min_conf)),
            sigma_mult=max(float(self.sigma_mult), 0.0),
            mr_k=float(self.mr_k),
            trend_clip=max(float(self.trend_clip), 0.0),
            spread_penalty_denom=max(float(self.spread_penalty_denom), EPSILON),
            w_trend_up=float(self.w_trend_up),
            w_mr_up=float(self.w_mr_up),
            w_trend_range=float(self.w_trend_range),
            w_mr_range=float(self.w_mr_range),
            w_trend_vol=float(self.w_trend_vol),
            w_mr_vol=float(self.w_mr_vol),
            conf_vol_penalty=max(float(self.conf_vol_penalty), 0.0),
            liq_rank_min=float(self.liq_rank_min),
            max_vol_60m=max(float(self.max_vol_60m), EPSILON),
            mode=mode,
            train_enabled=bool(self.train_enabled),
            train_lookback_days=max(int(self.train_lookback_days), 1),
            train_retention_days=max(int(self.train_retention_days), 1),
            train_batch=max(int(self.train_batch), 1),
            min_train_samples=max(int(self.min_train_samples), 1),
            min_updates=max(int(self.min_updates), 1),
            train_lr=max(float(self.train_lr), EPSILON),
            lr_decay=max(float(self.lr_decay), EPSILON),
            l2=max(float(self.l2), 0.0),
            w_clip=max(float(self.w_clip), 1e-6),
            b_clip=max(float(self.b_clip), 1e-6),
            cleanup_every_bars=max(int(self.cleanup_every_bars), 1),
            bar_seconds=max(int(self.bar_seconds), 1),
            baseline_model_name=str(self.baseline_model_name or "nocom_v1"),
            online_model_name=str(self.online_model_name or "nocom_online_v1"),
            feature_schema_version=str(self.feature_schema_version or "nocom_x_v1"),
            baseline_features_version=str(self.baseline_features_version or "features_v1"),
            baseline_model_version=str(self.baseline_model_version or "nocom_v1"),
            online_model_version=str(self.online_model_version or "nocom_online_v1"),
            model_name=mode,
        )


def get_latest_bar_ts(session) -> int | None:
    value = session.execute(select(func.max(ItemBar.bar_ts))).scalar_one_or_none()
    return int(value) if value is not None else None


def _get_latest_spread_map(session, item_keys: set[str]) -> dict[str, float]:
    if not item_keys:
        return {}
    rows = session.execute(
        select(
            ItemSpreadSnapshot.item_key,
            ItemSpreadSnapshot.ts,
            ItemSpreadSnapshot.spread_pct,
        )
        .where(ItemSpreadSnapshot.item_key.in_(item_keys))
        .order_by(ItemSpreadSnapshot.item_key.asc(), ItemSpreadSnapshot.ts.desc())
    ).all()

    spread_map: dict[str, float] = {}
    for item_key, _ts, spread_pct in rows:
        key = str(item_key)
        if key in spread_map:
            continue
        spread_map[key] = float(spread_pct or 0.0)
    return spread_map


def _get_candidate_features(session, *, cfg: NocomConfig) -> list[ItemFeature]:
    return (
        session.execute(
            select(ItemFeature)
            .where(ItemFeature.liquidity_ok == 1)
            .where(ItemFeature.liquidity_rank >= float(cfg.liq_rank_min))
            .order_by(ItemFeature.liquidity_rank.desc())
            .limit(int(cfg.top_n))
        )
        .scalars()
        .all()
    )


def _get_bar_count_map(session, item_keys: set[str]) -> dict[str, int]:
    if not item_keys:
        return {}
    bar_counts = session.execute(
        select(ItemBar.item_key, func.count())
        .where(ItemBar.item_key.in_(item_keys))
        .group_by(ItemBar.item_key)
    ).all()
    return {str(item_key): int(count) for item_key, count in bar_counts}


def _build_feature_vector(feature: ItemFeature, *, spread_pct: float, cfg: NocomConfig) -> dict[str, float]:
    price_now = max(float(feature.price_now or 0.0), EPSILON)
    median_24h = max(float(feature.median_24h or 0.0), EPSILON)
    iqr_24h = max(float(feature.iqr_24h or 0.0), 0.0)
    regime = str(feature.regime_tag or "RANGE").strip().upper()

    trend_strength = _clip(
        (float(feature.ema_fast or 0.0) - float(feature.ema_slow or 0.0)) / price_now,
        -float(cfg.trend_clip),
        float(cfg.trend_clip),
    )

    out = {
        "zscore_24h_clipped": _clip(float(feature.zscore_24h or 0.0), -5.0, 5.0),
        "trend_strength": trend_strength,
        "slope_hourly_clipped": _clip(float(feature.trend_slope_60m or 0.0), -0.05, 0.05),
        "vol_60m_clipped": _clip(float(feature.vol_60m or 0.0), 0.0, 0.20),
        "spread_pct_clipped": _clip(float(spread_pct or 0.0), 0.0, 0.30),
        "liquidity_rank_norm": _clip(float(feature.liquidity_rank or 0.0) / 100.0, 0.0, 1.0),
        "spread_ratio_clipped": _clip(iqr_24h / max(median_24h, EPSILON), 0.0, 1.0),
        "regime_onehot_UPTREND": 1.0 if regime == "UPTREND" else 0.0,
        "regime_onehot_RANGE": 1.0 if regime == "RANGE" else 0.0,
        "regime_onehot_VOLATILE": 1.0 if regime == "VOLATILE" else 0.0,
        "regime_onehot_DOWNTREND": 1.0 if regime == "DOWNTREND" else 0.0,
        "samples_24h_log": _clip(math.log1p(max(float(feature.samples_24h or 0.0), 0.0)), 0.0, 12.0),
    }
    return out


def _compute_regime_weights(regime: str, cfg: NocomConfig) -> tuple[float, float]:
    regime = (regime or "RANGE").strip().upper()
    if regime == "UPTREND":
        return float(cfg.w_trend_up), float(cfg.w_mr_up)
    if regime == "VOLATILE":
        return float(cfg.w_trend_vol), float(cfg.w_mr_vol)
    return float(cfg.w_trend_range), float(cfg.w_mr_range)


def _baseline_forecast(
    *,
    feature: ItemFeature,
    x: dict[str, float],
    horizon_minutes: int,
    cfg: NocomConfig,
) -> dict[str, float]:
    horizon_scale = float(horizon_minutes) / 60.0
    slope_scaled = float(x["slope_hourly_clipped"]) * horizon_scale
    spread_penalty = _clip(
        float(x["spread_pct_clipped"]) / max(float(cfg.spread_penalty_denom), EPSILON),
        0.0,
        1.0,
    )
    reversion = -float(cfg.mr_k) * float(x["zscore_24h_clipped"]) * float(x["spread_ratio_clipped"])

    regime = str(feature.regime_tag or "RANGE").strip().upper()
    w_trend, w_mr = _compute_regime_weights(regime, cfg)

    base_return = w_trend * (float(x["trend_strength"]) + slope_scaled) + w_mr * reversion - spread_penalty * 0.01
    vol_60m = float(x["vol_60m_clipped"])
    pred_vol = vol_60m * math.sqrt(max(horizon_scale, EPSILON))
    sigma = pred_vol * float(cfg.sigma_mult)
    pred_p50 = base_return
    pred_p10 = base_return - Q10_Z * sigma
    pred_p90 = base_return + Q10_Z * sigma
    p_up = _sigmoid(base_return / max(pred_vol, EPSILON))

    return {
        "pred_return": float(base_return),
        "pred_p10": float(pred_p10),
        "pred_p50": float(pred_p50),
        "pred_p90": float(pred_p90),
        "pred_vol": float(max(pred_vol, 0.0)),
        "p_up": float(_clamp_01(p_up)),
        "spread_penalty": float(spread_penalty),
        "reversion": float(reversion),
        "slope_scaled": float(slope_scaled),
    }


def _base_confidence(*, x: dict[str, float], cfg: NocomConfig) -> float:
    vol_penalty = float(cfg.conf_vol_penalty) * (
        float(x["vol_60m_clipped"]) / max(float(cfg.max_vol_60m), EPSILON)
    )
    return _clamp_01(
        0.6 * float(x["liquidity_rank_norm"])
        + 0.2 * (1.0 / (1.0 + float(x["spread_ratio_clipped"])))
        + 0.2 * abs(float(x["trend_strength"]))
        - vol_penalty
    )


def _empty_state() -> dict:
    return {"bias": 0.0, "w": {}, "n_updates": 0, "last_loss": None}


def _load_model_state_map(session, *, cfg: NocomConfig) -> dict[tuple[int, float], dict]:
    rows = (
        session.execute(
            select(NocomModelState)
            .where(NocomModelState.horizon_minutes.in_(cfg.horizons))
            .where(NocomModelState.quantile.in_(ONLINE_QUANTILES))
            .where(NocomModelState.model_name == cfg.online_model_name)
            .where(NocomModelState.feature_schema_version == cfg.feature_schema_version)
        )
        .scalars()
        .all()
    )

    state_map: dict[tuple[int, float], dict] = {
        (int(h), float(q)): _empty_state() for h in cfg.horizons for q in ONLINE_QUANTILES
    }
    for row in rows:
        key = (int(row.horizon_minutes), float(row.quantile))
        payload = _empty_state()
        try:
            parsed = json.loads(row.weights_json or "{}")
            if isinstance(parsed, dict):
                payload["bias"] = float(parsed.get("bias", 0.0))
                raw_w = parsed.get("w", {})
                if isinstance(raw_w, dict):
                    payload["w"] = {str(k): float(v) for k, v in raw_w.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = _empty_state()
        payload["n_updates"] = int(row.n_updates or 0)
        payload["last_loss"] = float(row.last_loss) if row.last_loss is not None else None
        state_map[key] = payload
    return state_map


def _predict_log_return(state: dict, x: dict[str, float]) -> float:
    bias = float(state.get("bias", 0.0))
    weights = state.get("w") or {}
    total = bias
    for feat_name, feat_value in x.items():
        total += float(weights.get(feat_name, 0.0)) * float(feat_value)
    return float(total)


def _insert_training_pairs(
    session,
    *,
    now_ts: int,
    now_bar_ts: int,
    feature_rows: list[ItemFeature],
    spread_map: dict[str, float],
    bar_count_map: dict[str, int],
    cfg: NocomConfig,
) -> int:
    rows: list[dict] = []
    for feature in feature_rows:
        item_key = str(feature.item_key)
        if int(bar_count_map.get(item_key, 0)) <= 0:
            continue

        x = _build_feature_vector(feature, spread_pct=float(spread_map.get(item_key, 0.0)), cfg=cfg)
        x_json = json.dumps(x, separators=(",", ":"))

        for horizon_minutes in cfg.horizons:
            rows.append(
                {
                    "item_key": item_key,
                    "t0_bar_ts": int(now_bar_ts),
                    "horizon_minutes": int(horizon_minutes),
                    "x_json": x_json,
                    "y_realized": None,
                    "y_ts": None,
                    "created_ts": int(now_ts),
                }
            )

    if rows:
        insert_stmt = sqlite_insert(NocomTrainPair).prefix_with("OR IGNORE")
        for i in range(0, len(rows), BATCH_SIZE):
            session.execute(insert_stmt, rows[i:i + BATCH_SIZE])

    return len(rows)


def _label_due_pairs(session, *, now_bar_ts: int, cfg: NocomConfig) -> dict:
    label_limit = max(int(cfg.train_batch) * 2, int(cfg.top_n))
    total_due = 0
    total_labeled = 0

    for horizon_minutes in cfg.horizons:
        horizon_seconds = int(horizon_minutes) * 60
        due_cutoff = int(now_bar_ts) - horizon_seconds

        due_rows = session.execute(
            select(
                NocomTrainPair.id,
                NocomTrainPair.item_key,
                NocomTrainPair.t0_bar_ts,
            )
            .where(NocomTrainPair.horizon_minutes == int(horizon_minutes))
            .where(NocomTrainPair.y_realized.is_(None))
            .where(NocomTrainPair.t0_bar_ts <= due_cutoff)
            .order_by(NocomTrainPair.t0_bar_ts.asc())
            .limit(label_limit)
        ).all()

        if not due_rows:
            continue

        total_due += len(due_rows)
        by_item: dict[str, list[tuple[int, int]]] = {}
        for pair_id, item_key, t0_bar_ts in due_rows:
            key = str(item_key)
            by_item.setdefault(key, []).append((int(pair_id), int(t0_bar_ts)))

        updates: list[dict] = []
        for item_key, bucket in by_item.items():
            min_t0 = min(t0 for _pair_id, t0 in bucket)
            max_target = max(t0 + horizon_seconds for _pair_id, t0 in bucket)

            bars = session.execute(
                select(ItemBar.bar_ts, ItemBar.price_median)
                .where(ItemBar.item_key == item_key)
                .where(ItemBar.bar_ts >= min_t0)
                .where(ItemBar.bar_ts <= max_target)
                .order_by(ItemBar.bar_ts.asc())
            ).all()
            if not bars:
                continue

            ts_arr = [int(row.bar_ts) for row in bars]
            price_arr = [float(row.price_median or 0.0) for row in bars]

            for pair_id, t0 in bucket:
                idx0 = bisect_left(ts_arr, t0)
                if idx0 >= len(ts_arr) or int(ts_arr[idx0]) != int(t0):
                    continue
                price0 = float(price_arr[idx0])
                if price0 <= 0.0:
                    continue

                target_ts = int(t0) + horizon_seconds
                idx_h = bisect_left(ts_arr, target_ts)
                if idx_h >= len(ts_arr):
                    continue
                price_h = float(price_arr[idx_h])
                if price_h <= 0.0:
                    continue

                y_realized = math.log(price_h / price0)
                updates.append(
                    {
                        "pair_id": int(pair_id),
                        "y_realized": float(y_realized),
                        "y_ts": int(ts_arr[idx_h]),
                    }
                )

        if not updates:
            continue

        pairs_table = NocomTrainPair.__table__
        update_stmt = (
            pairs_table.update()
            .where(pairs_table.c.id == bindparam("pair_id"))
            .values(
                y_realized=bindparam("y_realized"),
                y_ts=bindparam("y_ts"),
            )
        )
        for i in range(0, len(updates), BATCH_SIZE):
            session.execute(update_stmt, updates[i:i + BATCH_SIZE])
        total_labeled += len(updates)

    return {"due": int(total_due), "labeled": int(total_labeled)}


def _pinball_loss(error: float, q: float) -> float:
    e = float(error)
    qq = float(q)
    return (qq * e) if e >= 0.0 else ((qq - 1.0) * e)


def _train_online_models(session, *, now_bar_ts: int, cfg: NocomConfig) -> dict:
    if not cfg.train_enabled:
        return {
            "enabled": False,
            "trained_pairs": 0,
            "ready_by_horizon": {int(h): False for h in cfg.horizons},
            "labeled_counts": {int(h): 0 for h in cfg.horizons},
        }

    state_map = _load_model_state_map(session, cfg=cfg)
    lookback_cutoff = int(now_bar_ts) - int(cfg.train_lookback_days) * 86400

    trained_pairs_total = 0

    for horizon_minutes in cfg.horizons:
        train_rows = session.execute(
            select(NocomTrainPair.x_json, NocomTrainPair.y_realized)
            .where(NocomTrainPair.horizon_minutes == int(horizon_minutes))
            .where(NocomTrainPair.y_realized.is_not(None))
            .where(NocomTrainPair.t0_bar_ts >= int(lookback_cutoff))
            .order_by(NocomTrainPair.t0_bar_ts.desc())
            .limit(int(cfg.train_batch))
        ).all()

        if not train_rows:
            continue

        dataset: list[tuple[dict[str, float], float]] = []
        for x_json, y_realized in train_rows:
            if y_realized is None:
                continue
            try:
                parsed = json.loads(x_json or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            x = {str(k): float(v) for k, v in parsed.items()}
            dataset.append((x, float(y_realized)))

        if not dataset:
            continue

        for quantile in ONLINE_QUANTILES:
            key = (int(horizon_minutes), float(quantile))
            state = state_map.get(key) or _empty_state()
            bias = float(state.get("bias", 0.0))
            weights = {str(k): float(v) for k, v in (state.get("w") or {}).items()}
            n_updates = int(state.get("n_updates", 0))
            rolling_loss = state.get("last_loss")
            rolling_loss_f = float(rolling_loss) if rolling_loss is not None else None

            for x, y_realized in dataset:
                y_hat = bias
                for feat_name, feat_value in x.items():
                    y_hat += float(weights.get(feat_name, 0.0)) * float(feat_value)

                error = float(y_realized) - float(y_hat)
                g = (float(quantile) - 1.0) if error < 0.0 else float(quantile)
                lr_t = float(cfg.train_lr) / math.sqrt(1.0 + (float(n_updates) / float(cfg.lr_decay)))

                bias = _clip(bias + lr_t * g, -float(cfg.b_clip), float(cfg.b_clip))
                for feat_name, feat_value in x.items():
                    prev_w = float(weights.get(feat_name, 0.0))
                    updated = prev_w + lr_t * (g * float(feat_value) - float(cfg.l2) * prev_w)
                    weights[feat_name] = _clip(updated, -float(cfg.w_clip), float(cfg.w_clip))

                loss = _pinball_loss(error, float(quantile))
                if rolling_loss_f is None:
                    rolling_loss_f = float(loss)
                else:
                    rolling_loss_f = 0.95 * rolling_loss_f + 0.05 * float(loss)

                n_updates += 1

            state_map[key] = {
                "bias": float(bias),
                "w": weights,
                "n_updates": int(n_updates),
                "last_loss": (float(rolling_loss_f) if rolling_loss_f is not None else None),
            }

        trained_pairs_total += len(dataset)

    labeled_counts = {
        int(h): 0 for h in cfg.horizons
    }
    count_rows = session.execute(
        select(NocomTrainPair.horizon_minutes, func.count())
        .where(NocomTrainPair.horizon_minutes.in_(cfg.horizons))
        .where(NocomTrainPair.y_realized.is_not(None))
        .group_by(NocomTrainPair.horizon_minutes)
    ).all()
    for horizon_minutes, count in count_rows:
        labeled_counts[int(horizon_minutes)] = int(count)

    ready_by_horizon: dict[int, bool] = {}
    for horizon_minutes in cfg.horizons:
        per_q_ready = []
        for quantile in ONLINE_QUANTILES:
            st = state_map.get((int(horizon_minutes), float(quantile))) or _empty_state()
            per_q_ready.append(int(st.get("n_updates", 0)) >= int(cfg.min_updates))
        ready_by_horizon[int(horizon_minutes)] = (
            int(labeled_counts.get(int(horizon_minutes), 0)) >= int(cfg.min_train_samples)
            and all(per_q_ready)
        )

    state_rows: list[dict] = []
    for horizon_minutes in cfg.horizons:
        for quantile in ONLINE_QUANTILES:
            state = state_map.get((int(horizon_minutes), float(quantile))) or _empty_state()
            state_rows.append(
                {
                    "horizon_minutes": int(horizon_minutes),
                    "quantile": float(quantile),
                    "model_name": cfg.online_model_name,
                    "feature_schema_version": cfg.feature_schema_version,
                    "weights_json": json.dumps(
                        {
                            "bias": float(state.get("bias", 0.0)),
                            "w": state.get("w") or {},
                        },
                        separators=(",", ":"),
                    ),
                    "n_updates": int(state.get("n_updates", 0)),
                    "last_update_ts": int(now_bar_ts),
                    "last_loss": (
                        float(state.get("last_loss"))
                        if state.get("last_loss") is not None
                        else None
                    ),
                }
            )

    if state_rows:
        insert_stmt = sqlite_insert(NocomModelState)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[
                NocomModelState.horizon_minutes,
                NocomModelState.quantile,
                NocomModelState.model_name,
                NocomModelState.feature_schema_version,
            ],
            set_={
                "weights_json": insert_stmt.excluded.weights_json,
                "n_updates": insert_stmt.excluded.n_updates,
                "last_update_ts": insert_stmt.excluded.last_update_ts,
                "last_loss": insert_stmt.excluded.last_loss,
            },
        )
        for i in range(0, len(state_rows), BATCH_SIZE):
            session.execute(upsert_stmt, state_rows[i:i + BATCH_SIZE])

    return {
        "enabled": True,
        "trained_pairs": int(trained_pairs_total),
        "ready_by_horizon": ready_by_horizon,
        "labeled_counts": labeled_counts,
        "state_map": state_map,
    }


def _cleanup_old_pairs(session, *, now_bar_ts: int, cfg: NocomConfig) -> int:
    retention_cutoff = int(now_bar_ts) - int(cfg.train_retention_days) * 86400
    result = session.execute(
        delete(NocomTrainPair).where(NocomTrainPair.t0_bar_ts < int(retention_cutoff))
    )
    return int(result.rowcount or 0)


def _resolve_mode_for_horizon(*, cfg: NocomConfig, horizon_minutes: int, ready_by_horizon: dict[int, bool]) -> str:
    mode = _normalize_mode(cfg.mode)
    ready = bool(ready_by_horizon.get(int(horizon_minutes), False))
    if mode == "baseline":
        return "baseline"
    if mode == "online":
        return "online" if ready else "baseline"
    return "online" if ready else "baseline"


def _online_forecast(
    *,
    x: dict[str, float],
    horizon_minutes: int,
    cfg: NocomConfig,
    state_map: dict[tuple[int, float], dict],
) -> dict[str, float] | None:
    states = {
        float(q): state_map.get((int(horizon_minutes), float(q)))
        for q in ONLINE_QUANTILES
    }
    if any(state is None for state in states.values()):
        return None

    # Predict log-return quantiles, then convert to fractional returns.
    y10 = _predict_log_return(states[0.1], x)
    y50 = _predict_log_return(states[0.5], x)
    y90 = _predict_log_return(states[0.9], x)

    p10 = _logret_to_frac(y10)
    p50 = _logret_to_frac(y50)
    p90 = _logret_to_frac(y90)

    ordered = sorted([p10, p50, p90])
    pred_p10 = float(ordered[0])
    pred_p50 = float(ordered[1])
    pred_p90 = float(ordered[2])

    pred_return = float(pred_p50)
    pred_vol = max((pred_p90 - pred_p10) / Q_SPREAD_SIGMA_DENOM, EPSILON)
    p_up = _sigmoid(pred_return / max(pred_vol, EPSILON))

    base_conf = _base_confidence(x=x, cfg=cfg)
    quantile_spread = max(pred_p90 - pred_p10, 0.0)
    spread_penalty = 0.35 * _clip(quantile_spread / 0.30, 0.0, 1.0)
    q50_loss = states[0.5].get("last_loss")
    loss_penalty = 0.0
    if q50_loss is not None:
        loss_penalty = 0.15 * _clip(float(q50_loss) / 0.20, 0.0, 1.0)
    confidence = _clamp_01(base_conf - spread_penalty - loss_penalty)

    return {
        "pred_return": float(pred_return),
        "pred_p10": float(pred_p10),
        "pred_p50": float(pred_p50),
        "pred_p90": float(pred_p90),
        "pred_vol": float(pred_vol),
        "p_up": float(_clamp_01(p_up)),
        "confidence": float(confidence),
        "y10_log_return": float(y10),
        "y50_log_return": float(y50),
        "y90_log_return": float(y90),
        "quantile_spread": float(quantile_spread),
    }


def compute_nocom_forecasts(session, *, now_ts: int, cfg: NocomConfig) -> dict:
    cfg = cfg.normalized()
    if not cfg.enabled:
        return {
            "enabled": False,
            "candidates": 0,
            "eligible": 0,
            "attempted": 0,
            "mode": cfg.mode,
        }

    features = _get_candidate_features(session, cfg=cfg)
    if not features:
        return {
            "enabled": True,
            "candidates": 0,
            "eligible": 0,
            "attempted": 0,
            "mode": cfg.mode,
        }

    item_keys = {row.item_key for row in features}
    bar_count_map = _get_bar_count_map(session, item_keys)
    spread_pct_map = _get_latest_spread_map(session, item_keys)
    latest_bar_ts = get_latest_bar_ts(session)

    inserted_pairs = 0
    labeled_stats = {"due": 0, "labeled": 0}
    train_stats = {
        "enabled": False,
        "trained_pairs": 0,
        "ready_by_horizon": {int(h): False for h in cfg.horizons},
        "labeled_counts": {int(h): 0 for h in cfg.horizons},
        "state_map": {},
    }

    if latest_bar_ts is not None and cfg.train_enabled:
        inserted_pairs = _insert_training_pairs(
            session,
            now_ts=int(now_ts),
            now_bar_ts=int(latest_bar_ts),
            feature_rows=features,
            spread_map=spread_pct_map,
            bar_count_map=bar_count_map,
            cfg=cfg,
        )
        labeled_stats = _label_due_pairs(session, now_bar_ts=int(latest_bar_ts), cfg=cfg)
        train_stats = _train_online_models(session, now_bar_ts=int(latest_bar_ts), cfg=cfg)

        if cfg.cleanup_every_bars > 0:
            bucket = int(latest_bar_ts) // max(int(cfg.bar_seconds), 1)
            if bucket % int(cfg.cleanup_every_bars) == 0:
                cleaned = _cleanup_old_pairs(session, now_bar_ts=int(latest_bar_ts), cfg=cfg)
                logger.debug("nocom train-pair cleanup: deleted=%s cutoff_days=%s", cleaned, cfg.train_retention_days)

    state_map = train_stats.get("state_map") or _load_model_state_map(session, cfg=cfg)
    ready_by_horizon = {
        int(h): bool((train_stats.get("ready_by_horizon") or {}).get(int(h), False)) for h in cfg.horizons
    }
    labeled_counts = {
        int(h): int((train_stats.get("labeled_counts") or {}).get(int(h), 0)) for h in cfg.horizons
    }

    rows: list[dict] = []
    eligible_items = 0

    for feature in features:
        item_key = str(feature.item_key)
        bars_n = int(bar_count_map.get(item_key, 0))
        if bars_n < int(cfg.min_bars):
            continue
        eligible_items += 1

        spread_pct = float(spread_pct_map.get(item_key, 0.0))
        x = _build_feature_vector(feature, spread_pct=spread_pct, cfg=cfg)
        base_conf = _base_confidence(x=x, cfg=cfg)

        for horizon_minutes in cfg.horizons:
            forecast_mode = _resolve_mode_for_horizon(
                cfg=cfg,
                horizon_minutes=int(horizon_minutes),
                ready_by_horizon=ready_by_horizon,
            )

            if forecast_mode == "online":
                online_payload = _online_forecast(
                    x=x,
                    horizon_minutes=int(horizon_minutes),
                    cfg=cfg,
                    state_map=state_map,
                )
            else:
                online_payload = None

            if online_payload is None:
                baseline_payload = _baseline_forecast(
                    feature=feature,
                    x=x,
                    horizon_minutes=int(horizon_minutes),
                    cfg=cfg,
                )
                confidence = _base_confidence(x=x, cfg=cfg)
                pred_return = float(baseline_payload["pred_return"])
                pred_p10 = float(baseline_payload["pred_p10"])
                pred_p50 = float(baseline_payload["pred_p50"])
                pred_p90 = float(baseline_payload["pred_p90"])
                pred_vol = float(max(baseline_payload["pred_vol"], 0.0))
                p_up = float(_clamp_01(baseline_payload["p_up"]))
                model_name = cfg.baseline_model_name
                model_version = cfg.baseline_model_version
                features_version = cfg.baseline_features_version
                inputs_payload = {
                    "mode": "baseline",
                    "feature_schema": cfg.feature_schema_version,
                    "base_confidence": base_conf,
                    "features": x,
                    "slope_scaled": baseline_payload["slope_scaled"],
                    "reversion": baseline_payload["reversion"],
                    "spread_penalty": baseline_payload["spread_penalty"],
                    "bars_n": bars_n,
                    "regime": str(feature.regime_tag or "RANGE"),
                    "train_ready": bool(ready_by_horizon.get(int(horizon_minutes), False)),
                    "labeled_samples": int(labeled_counts.get(int(horizon_minutes), 0)),
                }
            else:
                pred_return = float(online_payload["pred_return"])
                pred_p10 = float(online_payload["pred_p10"])
                pred_p50 = float(online_payload["pred_p50"])
                pred_p90 = float(online_payload["pred_p90"])
                pred_vol = float(max(online_payload["pred_vol"], EPSILON))
                p_up = float(_clamp_01(online_payload["p_up"]))
                confidence = float(_clamp_01(online_payload["confidence"]))
                model_name = cfg.online_model_name
                model_version = cfg.online_model_version
                features_version = cfg.feature_schema_version

                n_updates_snapshot = {
                    str(q): int((state_map.get((int(horizon_minutes), float(q))) or {}).get("n_updates", 0))
                    for q in ONLINE_QUANTILES
                }
                inputs_payload = {
                    "mode": "online",
                    "feature_schema": cfg.feature_schema_version,
                    "base_confidence": base_conf,
                    "features": x,
                    "yhat_log_return": {
                        "q10": online_payload["y10_log_return"],
                        "q50": online_payload["y50_log_return"],
                        "q90": online_payload["y90_log_return"],
                    },
                    "quantile_spread": online_payload["quantile_spread"],
                    "labeled_samples": int(labeled_counts.get(int(horizon_minutes), 0)),
                    "n_updates": n_updates_snapshot,
                }

            rows.append(
                {
                    "ts": int(now_ts),
                    "item_key": item_key,
                    "horizon_minutes": int(horizon_minutes),
                    "model_name": model_name,
                    "pred_return": float(pred_return),
                    "pred_p10": float(pred_p10),
                    "pred_p50": float(pred_p50),
                    "pred_p90": float(pred_p90),
                    "p_up": float(_clamp_01(p_up)),
                    "pred_vol": float(max(pred_vol, EPSILON)),
                    "confidence": float(_clamp_01(confidence)),
                    "features_version": features_version,
                    "model_version": model_version,
                    "inputs_json": json.dumps(inputs_payload, separators=(",", ":")),
                }
            )

    if rows:
        insert_stmt = sqlite_insert(ItemForecast).prefix_with("OR IGNORE")
        for i in range(0, len(rows), BATCH_SIZE):
            session.execute(insert_stmt, rows[i:i + BATCH_SIZE])

    logger.debug(
        "nocom forecasts computed: mode=%s candidates=%s eligible=%s rows=%s inserted_pairs=%s labeled=%s trained_pairs=%s horizons=%s",
        cfg.mode,
        len(features),
        eligible_items,
        len(rows),
        inserted_pairs,
        labeled_stats.get("labeled", 0),
        train_stats.get("trained_pairs", 0),
        ",".join(str(h) for h in cfg.horizons),
    )

    return {
        "enabled": True,
        "mode": cfg.mode,
        "candidates": len(features),
        "eligible": eligible_items,
        "attempted": len(rows),
        "horizons": list(cfg.horizons),
        "training": {
            "enabled": bool(cfg.train_enabled),
            "pairs_created": int(inserted_pairs),
            "pairs_due": int(labeled_stats.get("due", 0)),
            "pairs_labeled": int(labeled_stats.get("labeled", 0)),
            "trained_pairs": int(train_stats.get("trained_pairs", 0)),
            "ready_by_horizon": {int(k): bool(v) for k, v in (train_stats.get("ready_by_horizon") or {}).items()},
            "labeled_counts": {int(k): int(v) for k, v in (train_stats.get("labeled_counts") or {}).items()},
        },
    }


def get_latest_forecasts(
    session,
    horizon_minutes: int,
    limit: int,
    *,
    model_name: str = "nocom_v1",
    item_keys: set[str] | None = None,
) -> list[ItemForecast]:
    bounded_limit = max(int(limit), 1)
    horizon = max(int(horizon_minutes), 1)
    model_selector = (model_name or "auto").strip().lower()

    latest_q = select(ItemForecast.item_key, func.max(ItemForecast.ts).label("max_ts")).where(
        ItemForecast.horizon_minutes == horizon
    )

    if model_selector not in {"auto", ""}:
        latest_q = latest_q.where(ItemForecast.model_name == model_name)
    if item_keys:
        latest_q = latest_q.where(ItemForecast.item_key.in_(item_keys))

    latest_q = latest_q.group_by(ItemForecast.item_key).subquery()

    query = (
        select(ItemForecast)
        .join(
            latest_q,
            (ItemForecast.item_key == latest_q.c.item_key)
            & (ItemForecast.ts == latest_q.c.max_ts),
        )
        .where(ItemForecast.horizon_minutes == horizon)
    )

    if model_selector not in {"auto", ""}:
        query = query.where(ItemForecast.model_name == model_name)
    if item_keys:
        query = query.where(ItemForecast.item_key.in_(item_keys))

    rows = session.execute(
        query.order_by(
            ItemForecast.item_key.asc(),
            ItemForecast.ts.desc(),
            case((ItemForecast.model_name.like("%online%"), 1), else_=0).desc(),
        )
    ).scalars().all()

    deduped: list[ItemForecast] = []
    seen: set[str] = set()
    for row in rows:
        item_key = str(row.item_key)
        if item_key in seen:
            continue
        seen.add(item_key)
        deduped.append(row)

    deduped.sort(key=lambda row: (float(row.confidence), float(row.pred_return)), reverse=True)
    return deduped[:bounded_limit]


def rank_items_with_ev(
    session,
    *,
    horizon_minutes: int,
    limit: int,
    min_conf: float,
    risk_k: float,
    ah_fee_pct: float,
    model_name: str = "nocom_v1",
) -> list[dict]:
    forecasts = get_latest_forecasts(
        session,
        horizon_minutes=horizon_minutes,
        limit=max(int(limit) * 20, 500),
        model_name=model_name,
    )
    if not forecasts:
        return []

    keys = {row.item_key for row in forecasts}
    feature_rows = (
        session.execute(select(ItemFeature).where(ItemFeature.item_key.in_(keys))).scalars().all()
    )
    feature_map = {row.item_key: row for row in feature_rows}

    rows: list[dict] = []
    horizon_hours = max(float(horizon_minutes) / 60.0, EPSILON)
    conf_floor = _clamp_01(float(min_conf))
    risk_k = float(risk_k)
    fee_pct = float(ah_fee_pct)

    for fc in forecasts:
        if float(fc.confidence) < conf_floor:
            continue
        feature = feature_map.get(fc.item_key)
        if feature is None:
            continue

        ev = float(fc.pred_return)
        ev_risk_adj = ev - risk_k * float(fc.pred_vol)
        ev_after_fee = ev_risk_adj - fee_pct
        ev_per_hour = ev_after_fee / horizon_hours

        rows.append(
            {
                "item_key": fc.item_key,
                "price_now": float(feature.price_now),
                "regime_tag": feature.regime_tag,
                "liquidity_rank": float(feature.liquidity_rank or 0.0),
                "pred_return": float(fc.pred_return),
                "pred_p10": float(fc.pred_p10),
                "pred_p50": float(fc.pred_p50),
                "pred_p90": float(fc.pred_p90),
                "pred_vol": float(fc.pred_vol),
                "p_up": float(fc.p_up),
                "confidence": float(fc.confidence),
                "horizon_minutes": int(fc.horizon_minutes),
                "model_name": fc.model_name,
                "ev": ev,
                "ev_risk_adj": ev_risk_adj,
                "ev_after_fee": ev_after_fee,
                "ev_per_hour": ev_per_hour,
            }
        )

    rows.sort(key=lambda row: float(row["ev_per_hour"]), reverse=True)
    return rows[: max(int(limit), 1)]
