from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import func, select

from .db import AlertSentEquity, MarketEvent, insert_alerts_sent_equities
from .discord import send_webhook

logger = logging.getLogger(__name__)

ALERTABLE_EVENT_TYPES = {
    "REGIME_FLIP",
    "BREADTH_SHOCK",
    "HIGH_LOW_BREAK_30D",
    "VOLATILITY_SHOCK",
    "CORRELATION_BREAK",
    "EXTREME_DRAWDOWN",
}


@dataclass
class EquitiesAlertsConfig:
    enabled: bool = True
    cooldown_minutes: int = 30
    max_alerts_per_hour: int = 12


class EquitiesAlertsService:
    def __init__(self, cfg: EquitiesAlertsConfig) -> None:
        self.cfg = cfg

    def _load_recent_candidates(self, session, *, now_ts: int) -> list[MarketEvent]:
        window_start = int(now_ts) - (3 * 3600)
        return (
            session.execute(
                select(MarketEvent)
                .where(MarketEvent.ts >= window_start)
                .where(MarketEvent.event_type.in_(ALERTABLE_EVENT_TYPES))
                .order_by(MarketEvent.severity.desc(), MarketEvent.ts.desc(), MarketEvent.id.desc())
            )
            .scalars()
            .all()
        )

    def _latest_sent_map(self, session) -> dict[tuple[str, str], int]:
        rows = (
            session.execute(
                select(
                    AlertSentEquity.alert_type,
                    AlertSentEquity.symbol,
                    func.max(AlertSentEquity.ts).label("latest_ts"),
                ).group_by(AlertSentEquity.alert_type, AlertSentEquity.symbol)
            )
            .all()
        )
        out: dict[tuple[str, str], int] = {}
        for alert_type, symbol, latest_ts in rows:
            key = (str(alert_type or ""), str(symbol or ""))
            if latest_ts is not None:
                out[key] = int(latest_ts)
        return out

    def run(self, session, *, now_ts: int) -> dict:
        if not self.cfg.enabled:
            return {
                "enabled": False,
                "sent": 0,
                "candidates": 0,
                "skipped_hourly_cap": 0,
                "skipped_cooldown": 0,
                "skipped_webhook": 0,
            }

        hourly_sent = (
            session.execute(
                select(func.count())
                .select_from(AlertSentEquity)
                .where(AlertSentEquity.ts >= int(now_ts) - 3600)
            ).scalar_one()
            or 0
        )
        remaining_budget = max(int(self.cfg.max_alerts_per_hour) - int(hourly_sent), 0)

        candidates = self._load_recent_candidates(session, now_ts=now_ts)
        latest_map = self._latest_sent_map(session)

        cooldown_seconds = max(int(self.cfg.cooldown_minutes), 0) * 60
        sent_rows: list[dict] = []
        skipped_cooldown = 0
        skipped_webhook = 0

        if remaining_budget <= 0:
            return {
                "enabled": True,
                "sent": 0,
                "candidates": len(candidates),
                "skipped_hourly_cap": len(candidates),
                "skipped_cooldown": 0,
                "skipped_webhook": 0,
            }

        for event in candidates:
            if remaining_budget <= 0:
                break

            alert_type = str(event.event_type)
            symbol = str(event.symbol or "")
            key = (alert_type, symbol)
            last_ts = latest_map.get(key)
            if last_ts is not None and cooldown_seconds > 0 and (int(now_ts) - int(last_ts)) < cooldown_seconds:
                skipped_cooldown += 1
                continue

            details = {}
            try:
                details = json.loads(str(event.details_json or "{}"))
                if not isinstance(details, dict):
                    details = {}
            except json.JSONDecodeError:
                details = {}

            direction = details.get("direction")
            direction_suffix = f" {direction}" if direction else ""
            content = (
                f"[nocom] {alert_type}{direction_suffix} | "
                f"severity={int(event.severity)} | "
                f"symbol={symbol or 'MARKET'}"
            )
            sent_ok = send_webhook(content)
            if not sent_ok:
                skipped_webhook += 1
                continue

            payload = {
                "event_id": int(event.id),
                "event_ts": int(event.ts),
                "severity": int(event.severity),
                "symbol": (symbol or None),
                "details": details,
            }
            sent_rows.append(
                {
                    "ts": int(now_ts),
                    "alert_type": alert_type,
                    "symbol": symbol or None,
                    "payload": json.dumps(payload, separators=(",", ":")),
                }
            )
            latest_map[key] = int(now_ts)
            remaining_budget -= 1

        inserted = insert_alerts_sent_equities(session, sent_rows)
        logger.info(
            "Equities alerts run: candidates=%s sent=%s cooldown_skips=%s webhook_skips=%s",
            len(candidates),
            inserted,
            skipped_cooldown,
            skipped_webhook,
        )

        return {
            "enabled": True,
            "sent": int(inserted),
            "candidates": len(candidates),
            "skipped_hourly_cap": max(len(candidates) - int(inserted) - skipped_cooldown - skipped_webhook, 0),
            "skipped_cooldown": int(skipped_cooldown),
            "skipped_webhook": int(skipped_webhook),
        }
