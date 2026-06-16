"""Thread-safe token-bucket rate limiter + 429-aware GET for scraping vendors.

Ported from the alpha-pulse project's ``RateBucket``. KR data vendors (Naver /
KRX) hit undocumented endpoints with undisclosed rate limits; a module-level
shared bucket caps aggregate requests-per-second across threads so concurrent
ticker analyses stay under the host's threshold and avoid 429/IP blocks.

``requests`` is already a project dependency, so this adds no new package.
"""

from __future__ import annotations

import logging
import random
import threading
import time

import requests

logger = logging.getLogger(__name__)


class RateBucket:
    """Token-bucket rate limiter, thread-safe.

    Refills ``rate`` tokens per second up to ``capacity`` (burst allowance);
    :meth:`acquire` blocks until a token is available.
    """

    def __init__(self, rate: float = 8.0, capacity: int | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else int(rate)
        self._tokens = float(self.capacity)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Acquire ``tokens``, sleeping (outside the lock) if the bucket is dry."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last_update = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self.rate
            time.sleep(wait)


# Shared module-level bucket so all KR vendors/threads draw from one budget.
_SHARED_BUCKET = RateBucket(rate=8.0)

# A descriptive UA; Naver/KRX serve these and block generic/empty tokens.
DEFAULT_UA = "Mozilla/5.0 (compatible; tradingagents/0.2; +https://github.com/TauricResearch/TradingAgents)"


def safe_get(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
    bucket: RateBucket | None = None,
) -> requests.Response:
    """Rate-limited GET with exponential-backoff retry on HTTP 429 / errors.

    Acquires a token from the shared (or supplied) bucket before each attempt,
    retries 429s and transient request errors with ``2**attempt + jitter``
    backoff, and raises the last exception if all attempts fail — callers in a
    vendor module let that propagate so :func:`route_to_vendor` skips to the
    next vendor.
    """
    bucket = bucket or _SHARED_BUCKET
    hdrs = {"User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        bucket.acquire()
        try:
            resp = requests.get(url, headers=hdrs, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("429 from %s; backing off %.1fs (attempt %d)", url, wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("GET %s failed (attempt %d): %s; retrying in %.1fs", url, attempt + 1, exc, wait)
            time.sleep(wait)

    raise last_exc or RuntimeError(f"safe_get exhausted retries for {url}")
