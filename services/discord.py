from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


def send_webhook(content: str) -> bool:
    webhook_url = (os.getenv("DISCORD_WEBHOOK_URL") or "").strip()
    if not webhook_url:
        logger.debug("Discord webhook missing; skipping alert send")
        return False

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(webhook_url, json={"content": content}, timeout=8)
            if response.status_code < 300:
                return True
            last_err = RuntimeError(f"Discord returned HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_err = exc

        if attempt < 3:
            time.sleep(0.4 * (2 ** (attempt - 1)))

    if last_err is not None:
        logger.error("Discord webhook send failed: %s", last_err)
    return False
