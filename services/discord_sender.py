from __future__ import annotations

from typing import Tuple

import requests


def send_discord_message(webhook_url: str, content: str) -> Tuple[bool, str | None]:
    target = str(webhook_url or "").strip()
    if not target:
        return False, "webhook url is required"

    try:
        response = requests.post(target, json={"content": content}, timeout=5)
    except requests.RequestException as exc:
        return False, f"request failed: {exc.__class__.__name__}"

    if 200 <= response.status_code < 300:
        return True, None
    return False, f"discord returned status {response.status_code}"
