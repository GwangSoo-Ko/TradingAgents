"""Shared HTTPS helper for the stdlib-``urllib`` data vendors (reddit, stocktwits).

macOS Python.framework installs ship without a linked OpenSSL CA bundle, so the
default ``ssl`` context can't verify certificates and every ``urlopen`` to an
HTTPS host fails with ``CERTIFICATE_VERIFY_FAILED``. ``requests``-based vendors
(yfinance, Alpha Vantage, …) are unaffected because ``requests`` bundles certifi.
The stdlib-``urllib`` vendors have no such fallback, so build their TLS context
from certifi's bundle explicitly. certifi is already a transitive dependency
(via ``requests``/``yfinance``); if it is somehow absent we fall back to the OS
default rather than fail hard.
"""

from __future__ import annotations

import functools
import ssl


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
