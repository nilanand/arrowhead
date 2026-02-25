from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

AUCTIONS_URL = "https://api.hypixel.net/skyblock/auctions"
AUCTION_BY_UUID_URL = "https://api.hypixel.net/skyblock/auction"


class HypixelApiError(RuntimeError):
    pass


class MissingApiKeyError(HypixelApiError):
    pass


@dataclass
class FetchResult:
    auctions: list[dict]
    page_count: int
    cached: bool = False


class HypixelAuctionsClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        cache_ttl_seconds: int = 5,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.cache_ttl_seconds = max(cache_ttl_seconds, 0)
        self.max_retries = max(max_retries, 1)
        self.backoff_base_seconds = max(backoff_base_seconds, 0.1)
        self.session = requests.Session()
        self._cache: dict[str, tuple[float, FetchResult]] = {}

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise MissingApiKeyError("HYPIXEL_API_KEY is missing. Set it in env or .env.")

    def _request_json(self, url: str, params: dict) -> dict:
        self._ensure_key()
        req_params = dict(params)
        req_params["key"] = self.api_key

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=req_params, timeout=10)
                if response.status_code != 200:
                    message = f"Hypixel HTTP {response.status_code}"
                    if response.status_code >= 500 or response.status_code == 429:
                        raise HypixelApiError(message)
                    raise HypixelApiError(f"{message}. Not retrying.")

                payload = response.json()
                if not payload.get("success", False):
                    cause = payload.get("cause") or payload.get("error") or "success=false"
                    raise HypixelApiError(f"Hypixel API failure: {cause}")
                return payload
            except (requests.RequestException, ValueError, HypixelApiError) as exc:
                last_err = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(min(sleep_seconds, 12))

        raise HypixelApiError(str(last_err) if last_err else "Unknown Hypixel API error")

    def fetch_auctions(
        self,
        *,
        full_scan: bool,
        use_cache: bool = True,
        page_limit: int = 1,
    ) -> FetchResult:
        normalized_limit = max(int(page_limit), 1)
        cache_key = "full" if full_scan else f"light:{normalized_limit}"
        now = time.time()

        if use_cache and self.cache_ttl_seconds > 0:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > now:
                return FetchResult(
                    auctions=list(cached[1].auctions),
                    page_count=cached[1].page_count,
                    cached=True,
                )

        first_page = self._request_json(AUCTIONS_URL, {"page": 0})
        auctions = list(first_page.get("auctions") or [])
        total_pages = int(first_page.get("totalPages") or 1)

        pages_fetched = 1
        if full_scan and total_pages > 1:
            for page in range(1, total_pages):
                payload = self._request_json(AUCTIONS_URL, {"page": page})
                auctions.extend(payload.get("auctions") or [])
                pages_fetched += 1
        elif not full_scan and total_pages > 1 and normalized_limit > 1:
            for page in range(1, min(total_pages, normalized_limit)):
                payload = self._request_json(AUCTIONS_URL, {"page": page})
                auctions.extend(payload.get("auctions") or [])
                pages_fetched += 1

        result = FetchResult(auctions=auctions, page_count=pages_fetched, cached=False)
        if self.cache_ttl_seconds > 0:
            self._cache[cache_key] = (now + self.cache_ttl_seconds, result)
        return result

    def fetch_auction_by_uuid(self, uuid: str) -> dict:
        payload = self._request_json(AUCTION_BY_UUID_URL, {"uuid": uuid})
        return payload.get("auction") or {}
