"""The Korean-only vendors decline a non-Korean ticker without deciding the verdict.

main.py routes news through "naver,yfinance" and fundamentals through
"wisereport,yfinance" for every ticker, US included. The Korean vendors refuse a
US ticker before any request, and that refusal used to count as the vendor's
"no data" verdict: when Yahoo was then unreachable, the router reported
NO_DATA ("wisereport only serves Korean tickers ... may be invalid, delisted")
instead of upstream's DATA_UNAVAILABLE ("says nothing about the instrument").
The refusal is now a VendorOutOfScopeError that the router passes over.
"""

import pytest

from tradingagents.dataflows import (
    naver_news,
    opendart_common,
    opendart_fundamentals,
    router,
    wisereport,
)
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorOutOfScopeError,
    VendorRateLimitError,
)

pytestmark = pytest.mark.unit

# main.py's chains (build_config).
_KR_CHAINS = {"data_vendors": {"news_data": "naver,yfinance",
                               "fundamental_data": "wisereport,yfinance"}}


def _raise(exc):
    def impl(*a, **k):
        raise exc
    return impl


@pytest.mark.parametrize("call", [
    lambda: naver_news.get_news("NVDA", "2026-09-01", "2026-09-08"),
    lambda: wisereport.get_fundamentals("NVDA", "2026-09-08"),
    lambda: opendart_fundamentals.get_fundamentals("NVDA", "2026-09-08"),
])
def test_a_korean_vendor_declines_a_us_ticker_before_any_request(monkeypatch, call):
    for module in (naver_news, wisereport, opendart_common):
        monkeypatch.setattr(module, "safe_get", _raise(AssertionError("requested")))
    with pytest.raises(VendorOutOfScopeError) as caught:
        call()
    # Still a NoMarketDataError, so any caller that caught that keeps working.
    assert isinstance(caught.value, NoMarketDataError)


def test_a_yahoo_outage_behind_a_korean_vendor_reads_as_unavailable(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS["get_fundamentals"], "yfinance", _raise(
        VendorRateLimitError("Yahoo Finance is unreachable; no fundamentals was retrieved")))
    with run_config(_KR_CHAINS):
        out = router.route_to_vendor("get_fundamentals", "NVDA", "2026-09-08")
    assert out.startswith("DATA_UNAVAILABLE"), out
    assert "Korean" not in out


def test_a_us_absence_behind_a_korean_vendor_carries_the_covering_vendors_reason(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", _raise(
        NoMarketDataError("NVDA", "NVDA", "news unavailable: timed out")))
    with run_config(_KR_CHAINS):
        out = router.route_to_vendor("get_news", "NVDA", "2026-09-01", "2026-09-08")
    assert out.startswith("NO_DATA_AVAILABLE"), out
    assert "news unavailable: timed out" in out and "Korean" not in out


def test_a_us_ticker_still_gets_the_covering_vendors_data(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS["get_fundamentals"], "yfinance",
                        lambda *a, **k: "NVDA fundamentals")
    with run_config(_KR_CHAINS):
        assert router.route_to_vendor("get_fundamentals", "NVDA", "2026-09-08") == "NVDA fundamentals"


def test_a_chain_of_korean_vendors_alone_still_reports_no_data_for_a_us_ticker():
    with run_config({"tool_vendors": {"get_news": "naver"}}):
        out = router.route_to_vendor("get_news", "NVDA", "2026-09-01", "2026-09-08")
    assert out.startswith("NO_DATA_AVAILABLE"), out
    assert "only serves Korean" in out


def test_an_erroring_covering_vendor_still_yields_the_sentinel_not_a_raise(monkeypatch):
    """Before the change the Korean refusal turned a covering vendor's crash into
    NO_DATA; that stays so -- passing over the refusal must not make a core tool
    raise where it returned text."""
    monkeypatch.setitem(router.VENDOR_METHODS["get_fundamentals"], "yfinance",
                        _raise(RuntimeError("boom")))
    with run_config(_KR_CHAINS):
        out = router.route_to_vendor("get_fundamentals", "NVDA", "2026-09-08")
    assert out.startswith("NO_DATA_AVAILABLE"), out


def test_a_korean_ticker_is_still_served_by_the_korean_vendor(monkeypatch):
    monkeypatch.setitem(router.VENDOR_METHODS["get_fundamentals"], "wisereport",
                        lambda *a, **k: "wisereport data")
    with run_config(_KR_CHAINS):
        out = router.route_to_vendor("get_fundamentals", "005930.KS", "2026-09-08")
    assert out == "wisereport data"
