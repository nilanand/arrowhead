from __future__ import annotations

import os
import re
from dataclasses import dataclass

MC_FORMAT_CODE_RE = re.compile(r"§.")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9_ ]+")
PET_LEVEL_RE = re.compile(r"(?i)(?:\[\s*lvl\s*(\d{1,3})\s*\]|(?:lvl|level)\s*[: ]\s*(\d{1,3}))")

KNOWN_REFORGES = {
    "ANCIENT",
    "AWKWARD",
    "BIZARRE",
    "BLESSED",
    "BLOODY",
    "BULKY",
    "CHEAP",
    "CLEAN",
    "COLOSSAL",
    "DEADLY",
    "DUNGEON",
    "EPIC",
    "FAIR",
    "FAST",
    "FABLED",
    "FIERCE",
    "FABLED",
    "FABLED",
    "FABLED",
    "FINE",
    "FLASHY",
    "FLEET",
    "FRUITFUL",
    "GENTLE",
    "GILDED",
    "GIANT",
    "GRAND",
    "HASTY",
    "HEROIC",
    "HEAVY",
    "HURTFUL",
    "KIND",
    "LUSH",
    "LEGENDARY",
    "LOVING",
    "LUCKY",
    "MITHRAIC",
    "NEAT",
    "ODD",
    "PAINFUL",
    "PRECISE",
    "RAPID",
    "RUGGED",
    "SALTY",
    "SHARP",
    "SILKY",
    "SMART",
    "SPIKED",
    "SPIRIT",
    "STRANGE",
    "STRONG",
    "SUPERIOR",
    "SUSPICIOUS",
    "SWIFT",
    "TITANIC",
    "TREACHEROUS",
    "UNPLEASANT",
    "UNREAL",
    "VERY",
    "WARPED",
    "WISE",
    "WITHERED",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CanonicalizationConfig:
    canonicalize_enabled: bool = _env_bool("CANONICALIZE_ENABLED", True)
    pet_level_equiv_min: int = _env_int("PET_LEVEL_EQUIV_MIN", 99)
    reforge_strip_enabled: bool = _env_bool("REFORGE_STRIP_ENABLED", True)


def normalize_item_name(name: str) -> str:
    cleaned = MC_FORMAT_CODE_RE.sub("", name or "")
    cleaned = cleaned.replace("✪", " ")
    cleaned = cleaned.replace("★", " ")
    cleaned = cleaned.replace("§", "")
    cleaned = WS_RE.sub(" ", cleaned).strip().upper()
    cleaned = NON_ALNUM_RE.sub(" ", cleaned)
    cleaned = WS_RE.sub(" ", cleaned).strip()
    return cleaned or "UNKNOWN"


def _to_slug(value: str) -> str:
    return WS_RE.sub("_", value.strip().upper()) if value else "UNKNOWN"


def _strip_reforge_prefix(name: str, *, enabled: bool) -> str:
    if not enabled:
        return name
    parts = name.split(" ", 1)
    if not parts:
        return name
    first = parts[0].strip().upper()
    if first in KNOWN_REFORGES and len(parts) > 1:
        return parts[1].strip()
    return name


def _strip_pet_level_markers(name: str) -> str:
    stripped = PET_LEVEL_RE.sub(" ", name or "")
    stripped = stripped.replace("[", " ").replace("]", " ")
    stripped = WS_RE.sub(" ", stripped).strip()
    return stripped or name


def _detect_pet_level(name: str, payload: dict) -> int | None:
    raw_level = payload.get("pet_level")
    if raw_level is not None:
        try:
            return max(int(raw_level), 0)
        except (TypeError, ValueError):
            pass

    for match in PET_LEVEL_RE.finditer(name):
        for group in match.groups():
            if not group:
                continue
            try:
                return max(int(group), 0)
            except ValueError:
                continue
    return None


def _is_pet(payload: dict, normalized_name: str) -> bool:
    category = str(payload.get("category") or "").strip().upper()
    if category == "PET":
        return True
    if normalized_name.startswith("[LVL ") or " PET" in normalized_name:
        return True
    return False


def _pet_bucket(level: int | None, *, equiv_min: int) -> str:
    if level is None:
        return "LUNKNOWN"
    if level >= int(equiv_min):
        return f"L{int(equiv_min)}PLUS"
    if 81 <= level <= 98:
        return "L81_98"
    if 1 <= level <= 80:
        return "L1_80"
    return "LUNKNOWN"


def _raw_item_key(name: str, tier: str, bin_flag: int = 1) -> str:
    safe_name = normalize_item_name(name).lower()
    safe_tier = (tier or "COMMON").strip().upper()
    return f"{safe_name}|{safe_tier}|{1 if bin_flag else 0}"


def canonicalize_auction(auction_payload: dict) -> dict:
    cfg = CanonicalizationConfig()

    item_name = str(auction_payload.get("item_name") or "UNKNOWN")
    normalized_name = normalize_item_name(item_name)
    tier = str(auction_payload.get("tier") or "COMMON").strip().upper() or "COMMON"
    category = str(auction_payload.get("category") or "UNKNOWN").strip().upper() or "UNKNOWN"
    item_id = str(auction_payload.get("item_id") or "").strip().upper()

    is_pet = _is_pet(auction_payload, normalized_name)
    base_name = normalized_name
    if is_pet:
        base_name = _strip_pet_level_markers(base_name)
    base_name = _strip_reforge_prefix(base_name, enabled=cfg.reforge_strip_enabled)
    base = _to_slug(item_id or base_name)

    tags: list[str] = []
    pet_level_bucket = None
    if is_pet:
        level = _detect_pet_level(normalized_name, auction_payload)
        pet_level_bucket = _pet_bucket(level, equiv_min=cfg.pet_level_equiv_min)
        tags.append(f"PET_{pet_level_bucket}")

    if not tags:
        tags.append("BASE")

    if not cfg.canonicalize_enabled:
        canonical = _raw_item_key(item_name, tier, 1)
    else:
        canonical = f"{base}|{tier}|{_to_slug(category)}|{','.join(tags)}"

    return {
        "raw_item_key": _raw_item_key(item_name, tier, 1),
        "canonical_item_key": canonical,
        "normalized_name": normalized_name,
        "pet_level_bucket": pet_level_bucket,
    }
