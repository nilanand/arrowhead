from __future__ import annotations

import math
import time


def format_coins(value: int | float | None) -> str:
    if value is None:
        return "-"
    n = float(value)
    sign = "-" if n < 0 else ""
    n = abs(n)

    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.2f}k"
    if n.is_integer():
        return f"{sign}{int(n)}"
    return f"{sign}{n:.2f}"


def format_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.1f}%"


def now_ts() -> int:
    return int(time.time())


def human_ends_in(end_ts: int | None, ref_ts: int | None = None) -> str:
    if not end_ts:
        return "-"
    now = ref_ts or now_ts()
    remaining = end_ts - now
    if remaining <= 0:
        return "Ended"

    hours = remaining // 3600
    minutes = (remaining % 3600) // 60
    seconds = remaining % 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def human_duration(duration_seconds: int | None) -> str:
    if duration_seconds is None:
        return "-"
    seconds = max(int(duration_seconds), 0)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
