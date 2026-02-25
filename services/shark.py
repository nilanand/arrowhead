from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

from sqlalchemy import func, select

from .db import Auction, ItemBaseline, SharkAlert, normalize_item_key
from .discord import send_webhook
from .format import format_coins, human_ends_in

logger = logging.getLogger(__name__)


def _bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(val: str | None, default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val: str | None, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@dataclass
class SharkConfig:
    enabled: bool = _bool(os.getenv("SHARK_ENABLED"), False)
    threshold_pct_below_median: float = _float(os.getenv("SHARK_PCT_BELOW_MEDIAN"), 0.35)
    threshold_iqr_multiplier: float = _float(os.getenv("SHARK_IQR_MULT"), 1.5)
    min_price: int = _int(os.getenv("SHARK_MIN_PRICE"), 100_000)
    cooldown_minutes: int = _int(os.getenv("SHARK_COOLDOWN_MINUTES"), 10)


class SharkService:
    def __init__(self, *, webhook_url: str | None, min_baseline_samples: int = 30) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.min_baseline_samples = max(min_baseline_samples, 1)
        self._cfg = SharkConfig()
        self._lock = threading.Lock()
        self._current_deals: list[dict] = []
        self._last_scan_ts: int | None = None

    def get_state(self) -> dict:
        with self._lock:
            return {
                "enabled": self._cfg.enabled,
                "threshold_pct_below_median": self._cfg.threshold_pct_below_median,
                "threshold_iqr_multiplier": self._cfg.threshold_iqr_multiplier,
                "min_price": self._cfg.min_price,
                "cooldown_minutes": self._cfg.cooldown_minutes,
                "webhook_configured": bool(self.webhook_url),
                "deals": list(self._current_deals),
                "last_scan_ts": self._last_scan_ts,
            }

    def toggle(self, enabled: bool) -> None:
        with self._lock:
            self._cfg.enabled = bool(enabled)

    def update_config(
        self,
        *,
        threshold_pct_below_median: float | None = None,
        threshold_iqr_multiplier: float | None = None,
        min_price: int | None = None,
        cooldown_minutes: int | None = None,
        enabled: bool | None = None,
    ) -> dict:
        with self._lock:
            if threshold_pct_below_median is not None:
                self._cfg.threshold_pct_below_median = min(max(float(threshold_pct_below_median), 0.01), 0.95)
            if threshold_iqr_multiplier is not None:
                self._cfg.threshold_iqr_multiplier = min(max(float(threshold_iqr_multiplier), 0.1), 10.0)
            if min_price is not None:
                self._cfg.min_price = max(int(min_price), 0)
            if cooldown_minutes is not None:
                self._cfg.cooldown_minutes = max(int(cooldown_minutes), 0)
            if enabled is not None:
                self._cfg.enabled = bool(enabled)

            return {
                "enabled": self._cfg.enabled,
                "threshold_pct_below_median": self._cfg.threshold_pct_below_median,
                "threshold_iqr_multiplier": self._cfg.threshold_iqr_multiplier,
                "min_price": self._cfg.min_price,
                "cooldown_minutes": self._cfg.cooldown_minutes,
            }

    def _discord_content(self, deal: dict) -> str:
        return (
            "SHARK DEAL DETECTED\n"
            f"Item: {deal['item_name']} [{deal['tier']}]\n"
            f"Price: {format_coins(deal['price'])}\n"
            f"Median(24h): {format_coins(deal['median'])}\n"
            f"Below median: {deal['pct_below'] * 100:.1f}%\n"
            f"Ends in: {deal['ends_in']}\n"
            f"UUID: {deal['uuid']}\n"
            f"/viewauction {deal['uuid']}"
        )

    def _fetch_recent_cooldowns(self, session, now_ts: int, cooldown_seconds: int) -> set[str]:
        if cooldown_seconds <= 0:
            return set()

        cutoff = now_ts - cooldown_seconds
        rows = session.execute(
            select(SharkAlert.item_key)
            .where(SharkAlert.ts >= cutoff)
            .group_by(SharkAlert.item_key)
        ).all()
        return {row[0] for row in rows}

    def _insert_alert(self, session, item_key: str, uuid: str, now_ts: int) -> None:
        # Insert once per auction UUID; duplicates are ignored by caller via checks.
        session.add(SharkAlert(item_key=item_key, uuid=uuid, ts=now_ts))

    def scan_and_alert(self, session) -> dict:
        now_ts = int(time.time())

        with self._lock:
            cfg = SharkConfig(
                enabled=self._cfg.enabled,
                threshold_pct_below_median=self._cfg.threshold_pct_below_median,
                threshold_iqr_multiplier=self._cfg.threshold_iqr_multiplier,
                min_price=self._cfg.min_price,
                cooldown_minutes=self._cfg.cooldown_minutes,
            )

        baseline_rows = session.execute(select(ItemBaseline)).scalars().all()
        baseline_map = {row.item_key: row for row in baseline_rows}

        auctions = session.execute(
            select(Auction).where(Auction.bin == 1).where(Auction.end_ts > now_ts)
        ).scalars().all()

        cooldown_seconds = cfg.cooldown_minutes * 60
        recently_alerted_keys = self._fetch_recent_cooldowns(session, now_ts, cooldown_seconds)

        deals: list[dict] = []
        alerts_sent = 0

        for auction in auctions:
            item_key = normalize_item_key(auction.item_name, auction.tier, 1)
            baseline = baseline_map.get(item_key)
            if not baseline or baseline.sample_n < self.min_baseline_samples:
                continue

            price = int(auction.starting_bid)
            if price < cfg.min_price:
                continue

            median = float(baseline.median)
            p25 = float(baseline.p25)
            iqr = float(baseline.iqr)

            threshold_by_median = median * (1 - cfg.threshold_pct_below_median)
            threshold_by_iqr = p25 - (cfg.threshold_iqr_multiplier * iqr)
            if price > threshold_by_median:
                continue
            if price > threshold_by_iqr:
                continue

            pct_below = 0.0 if median <= 0 else max((median - price) / median, 0.0)
            z_like = 0.0 if iqr <= 0 else (median - price) / iqr

            deal = {
                "uuid": auction.uuid,
                "item_key": item_key,
                "item_name": auction.item_name,
                "tier": auction.tier,
                "price": price,
                "median": median,
                "p25": p25,
                "iqr": iqr,
                "pct_below": pct_below,
                "z_like": z_like,
                "end_ts": auction.end_ts,
                "ends_in": human_ends_in(auction.end_ts, now_ts),
                "seller": auction.seller,
            }
            deals.append(deal)

            should_alert = cfg.enabled and item_key not in recently_alerted_keys
            if should_alert:
                sent = False
                if self.webhook_url:
                    sent, _ = send_webhook(self.webhook_url, self._discord_content(deal))
                self._insert_alert(session, item_key=item_key, uuid=auction.uuid, now_ts=now_ts)
                recently_alerted_keys.add(item_key)
                alerts_sent += 1
                logger.debug(
                    "SHARK DEAL: %s price=%s median=%s (%%below=%.2f) uuid=%s sent_to_discord=%s",
                    item_key,
                    price,
                    int(median),
                    pct_below * 100,
                    auction.uuid,
                    "yes" if sent else "no",
                )

        deals.sort(key=lambda row: (row["pct_below"], row["z_like"]), reverse=True)
        deals = deals[:250]

        with self._lock:
            self._current_deals = deals
            self._last_scan_ts = now_ts

        return {
            "deals": deals,
            "last_scan_ts": now_ts,
            "enabled": cfg.enabled,
            "alerts_sent": alerts_sent,
            "webhook_configured": bool(self.webhook_url),
            "threshold_pct_below_median": cfg.threshold_pct_below_median,
            "threshold_iqr_multiplier": cfg.threshold_iqr_multiplier,
            "min_price": cfg.min_price,
            "cooldown_minutes": cfg.cooldown_minutes,
        }

    def test_webhook(self) -> tuple[bool, str]:
        if not self.webhook_url:
            return False, "DISCORD_WEBHOOK_URL is not configured"

        return send_webhook(
            self.webhook_url,
            "nocom Shark Mode test ping successful. /viewauction <uuid>",
        )
