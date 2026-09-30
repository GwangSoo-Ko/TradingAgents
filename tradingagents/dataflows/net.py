"""HTTP helpers shared by the vendors.

``get_scrubbed`` / ``vendor_reachable`` serve the ``requests``-based vendors.
``default_ssl_context`` serves the stdlib-``urllib`` vendors (reddit,
stocktwits): macOS Python.framework installs ship without a linked OpenSSL CA
bundle, so the default ``ssl`` context can't verify certificates and every
``urlopen`` to an HTTPS host fails with ``CERTIFICATE_VERIFY_FAILED``.
``requests`` bundles certifi; ``urllib`` has no such fallback, so build its TLS
context from certifi's bundle explicitly. certifi is already a transitive
dependency (via ``requests``/``yfinance``); if it is somehow absent we fall back
to the OS default rather than fail hard.
"""

from __future__ import annotations

import functools
import ssl

import requests


def get_scrubbed(url: str, *, params: dict, timeout: float, secret: str, passthrough=()):
    """``requests.get`` plus ``raise_for_status``, with ``secret`` kept out of errors.

    Vendors that authenticate with a query parameter put the key in the URL, and
    requests quotes the full URL in HTTP, connection and timeout errors, so any
    log or traceback that records one would carry the key (#1324). A requests
    error is re-raised as the same class with the key replaced and nothing
    attached: no request or response (both hold the URL) and no exception chain,
    which is why this raises after the ``except`` block rather than inside it.
    Statuses in ``passthrough`` are returned for the caller to handle.
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code not in passthrough:
            response.raise_for_status()
        return response
    except requests.RequestException as exc:
        error = type(exc)(str(exc).replace(secret, "***")) if secret else exc
    raise error


def vendor_reachable(url: str, timeout: float = 5.0) -> bool:
    """Whether the vendor answers at all, for telling silence from an outage.

    A client that returns an empty result instead of raising leaves those two
    cases indistinguishable. Called only when a result is empty.
    """
    try:
        requests.head(url, timeout=timeout, allow_redirects=True)
        return True
    except requests.RequestException:
        return False


@functools.lru_cache(maxsize=1)
def default_ssl_context() -> ssl.SSLContext:
    """A verifying TLS context backed by certifi's CA bundle (cached).

    Verification (``CERT_REQUIRED`` + ``check_hostname``) is never disabled — the
    fix is to point at a CA bundle that exists, not to skip the check.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — certifi missing/broken; use the OS default
        return ssl.create_default_context()
