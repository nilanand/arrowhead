from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from sqlalchemy import func, select

from .db import AlertSentEquity, insert_alerts_sent_equities
from .discord import send_webhook

logger = logging.getLogger(__name__)


@dataclass
class AlertsConfig:
    enabled: bool = True
    cooldown_minutes: int = 30
    severity_threshold: int = 3


class AlertSink:
    def __init__(self, *, webhook_url: str | None) -> None:
        self.webhook_url = str(webhook_url or "").strip()

    @property
    def is_noop(self) -> bool:
        return self.webhook_url == ""

    def emit(self, message: str) -> bool:
        if self.is_noop:
            return True
        return bool(send_webhook(message))


class AlertsService:
    def __init__(self, cfg: AlertsConfig) -> None:
        self.cfg = cfg
        self.sink = AlertSink(webhook_url=os.getenv("DISCORD_WEBHOOK_URL"))
        self.app_base_url = (os.getenv("NOCOM_BASE_URL") or "http://localhost:5003").strip().rstrip("/")

    def _latest_alert_ts_map(self, session) -> dict[tuple[str, str], int]:
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

    def run(self, session, *, events: list[dict], now_ts: int) -> dict:
        if not self.cfg.enabled:
            return {
                "enabled": False,
                "processed": len(events),
                "sent": 0,
                "skipped_cooldown": 0,
                "skipped_severity": 0,
                "skipped_sink": 0,
                "sink": "noop" if self.sink.is_noop else "discord",
            }

        cooldown_seconds = max(int(self.cfg.cooldown_minutes), 0) * 60
        severity_threshold = max(int(self.cfg.severity_threshold), 1)
        latest_map = self._latest_alert_ts_map(session)

        rows_to_insert: list[dict] = []
        skipped_cooldown = 0
        skipped_severity = 0
        skipped_sink = 0

        for event in events:
            event_type = str(event.get("event_type") or "")
            symbol = str(event.get("symbol") or "").upper()
            severity = int(event.get("severity") or 0)
            if severity < severity_threshold:
                skipped_severity += 1
                continue

            key = (event_type, symbol)
            last_ts = latest_map.get(key)
            if last_ts is not None and cooldown_seconds > 0 and (int(now_ts) - int(last_ts)) < cooldown_seconds:
                skipped_cooldown += 1
                continue

            details = {}
            raw_details = event.get("details_json")
            if isinstance(raw_details, str) and raw_details:
                try:
                    parsed = json.loads(raw_details)
                    if isinstance(parsed, dict):
                        details = parsed
                except json.JSONDecodeError:
                    details = {}

            metrics_parts = []
            for key, value in details.items():
                metrics_parts.append(f"{key}={value}")
                if len(metrics_parts) >= 3:
                    break
            metrics_text = " | ".join(metrics_parts) if metrics_parts else "metrics=none"
            symbol_path = f"/symbol/{symbol}" if symbol else "/dashboard"
            symbol_link = f"{self.app_base_url}{symbol_path}"
            content = (
                f"[nocom]\\n"
                f"symbol={symbol or 'MARKET'}\\n"
                f"event_type={event_type}\\n"
                f"severity={severity}\\n"
                f"{metrics_text}\\n"
                f"link={symbol_link}"
            )
            emitted = self.sink.emit(content)
            if not emitted:
                skipped_sink += 1
                continue

            payload = {
                "event_ts": int(event.get("ts") or now_ts),
                "event_type": event_type,
                "severity": severity,
                "dedupe_key": str(event.get("dedupe_key") or ""),
                "details": details,
                "sink": ("noop" if self.sink.is_noop else "discord"),
            }
            rows_to_insert.append(
                {
                    "ts": int(now_ts),
                    "alert_type": event_type,
                    "symbol": (symbol or None),
                    "payload": json.dumps(payload, separators=(",", ":")),
                }
            )
            latest_map[key] = int(now_ts)

        inserted = insert_alerts_sent_equities(session, rows_to_insert)
        logger.info(
            "Phase0 alerts run: events=%s sent=%s skipped(severity=%s cooldown=%s sink=%s) sink=%s",
            len(events),
            inserted,
            skipped_severity,
            skipped_cooldown,
            skipped_sink,
            ("noop" if self.sink.is_noop else "discord"),
        )

        return {
            "enabled": True,
            "processed": len(events),
            "sent": int(inserted),
            "skipped_cooldown": int(skipped_cooldown),
            "skipped_severity": int(skipped_severity),
            "skipped_sink": int(skipped_sink),
            "sink": ("noop" if self.sink.is_noop else "discord"),
        }
