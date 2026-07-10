"""Shared HTTPS helper for the stdlib-urllib vendors (reddit, stocktwits).

Regression guard for CERTIFICATE_VERIFY_FAILED on macOS Python.framework installs
that ship without a linked CA bundle: default_ssl_context() must be a verifying
context whose CA store is populated (from certifi) regardless of the OS bundle.
"""
import ssl

import pytest

from tradingagents.dataflows.net import default_ssl_context


@pytest.mark.unit
class TestDefaultSSLContext:
    def test_returns_verifying_context(self):
        ctx = default_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_ca_store_is_populated(self):
        # certifi-backed => CAs load even when the OS bundle is missing (the
        # macOS case that produced CERTIFICATE_VERIFY_FAILED). A plain
        # create_default_context() on such a box loads zero CAs.
        assert default_ssl_context().cert_store_stats()["x509_ca"] > 0

    def test_context_is_cached(self):
        assert default_ssl_context() is default_ssl_context()


@pytest.mark.unit
class TestVendorWiring:
    """The stdlib-urllib vendors must hand a verifying context to urlopen."""

    def test_stocktwits_passes_ssl_context(self):
        from unittest.mock import patch

        from tradingagents.dataflows import stocktwits
        captured = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["context"] = context
            raise OSError("stop")  # caught -> placeholder; we only need the arg

        with patch.object(stocktwits, "urlopen", side_effect=fake_urlopen):
            stocktwits.fetch_stocktwits_messages("AAPL")
        assert isinstance(captured.get("context"), ssl.SSLContext)

    def test_reddit_passes_ssl_context(self):
        from unittest.mock import patch

        from tradingagents.dataflows import reddit
        captured = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["context"] = context
            raise OSError("stop")

        with patch.object(reddit, "urlopen", side_effect=fake_urlopen):
            reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
        assert isinstance(captured.get("context"), ssl.SSLContext)
