"""Tier-2 Stage 1 foundation: KR ticker normalization + rate-limit utility.

Prerequisites shared by every Korean data vendor (Naver / KRX / DART). Fully
network-free: the rate bucket is exercised on its in-memory token math and
safe_get is driven by a mocked requests.get with time.sleep patched out.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.dataflows.kr_utils import is_kr_ticker, to_krx_code
from tradingagents.dataflows import rate_limit
from tradingagents.dataflows.rate_limit import RateBucket, safe_get


@pytest.mark.unit
class TestKrTickerNormalization:
    def test_to_krx_code_from_qualified(self):
        assert to_krx_code("005930.KS") == "005930"
        assert to_krx_code("247540.KQ") == "247540"
        assert to_krx_code("005930.ks") == "005930"  # case-insensitive

    def test_to_krx_code_from_bare(self):
        assert to_krx_code("005930") == "005930"

    def test_to_krx_code_rejects_non_kr(self):
        for bad in ("AAPL", "7203.T", "BRK.B", "", "12345", "ABCDEF.KS"):
            with pytest.raises(ValueError):
                to_krx_code(bad)

    def test_is_kr_ticker(self):
        assert is_kr_ticker("005930.KS")
        assert is_kr_ticker("247540.KQ")
        assert is_kr_ticker("005930")
        assert not is_kr_ticker("AAPL")
        assert not is_kr_ticker("7203.T")
        assert not is_kr_ticker("")


@pytest.mark.unit
class TestRateBucket:
    def test_starts_full_no_block(self):
        # A fresh bucket has `capacity` tokens, so the first calls don't sleep.
        b = RateBucket(rate=8.0, capacity=8)
        with patch("tradingagents.dataflows.rate_limit.time.sleep") as slept:
            for _ in range(8):
                b.acquire()
            slept.assert_not_called()

    def test_blocks_when_drained(self):
        # 9th token on a capacity-8 bucket must wait for a refill.
        b = RateBucket(rate=8.0, capacity=8)
        with patch("tradingagents.dataflows.rate_limit.time.sleep") as slept:
            # Drain, forcing the refill path; sleep is mocked so the test is fast.
            # After sleep returns, the loop re-checks and (with mocked sleep not
            # advancing the clock) would spin — so make the first sleep "grant"
            # tokens by stopping after it's called once.
            slept.side_effect = StopIteration
            for _ in range(8):
                b.acquire()
            with pytest.raises(StopIteration):
                b.acquire()
            assert slept.called


@pytest.mark.unit
class TestSafeGet:
    def _resp(self, status=200):
        return SimpleNamespace(status_code=status, raise_for_status=lambda: None)

    def test_returns_on_200(self):
        with patch.object(rate_limit.requests, "get", return_value=self._resp(200)) as g, \
             patch("tradingagents.dataflows.rate_limit.time.sleep"):
            resp = safe_get("https://example.com")
            assert resp.status_code == 200
            g.assert_called_once()
            # default UA header is always sent
            assert "User-Agent" in g.call_args.kwargs["headers"]

    def test_retries_on_429_then_succeeds(self):
        seq = [self._resp(429), self._resp(200)]
        with patch.object(rate_limit.requests, "get", side_effect=seq) as g, \
             patch("tradingagents.dataflows.rate_limit.time.sleep"):
            resp = safe_get("https://example.com", max_retries=3)
            assert resp.status_code == 200
            assert g.call_count == 2

    def test_raises_after_exhausting_retries(self):
        with patch.object(rate_limit.requests, "get",
                          side_effect=rate_limit.requests.RequestException("boom")), \
             patch("tradingagents.dataflows.rate_limit.time.sleep"):
            with pytest.raises(rate_limit.requests.RequestException):
                safe_get("https://example.com", max_retries=2)
