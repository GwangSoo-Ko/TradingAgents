"""Tier-0 Korean-market coverage fixes.

These are fork-local correctness fixes for KOSPI/KOSDAQ tickers; the tests
lock them in so a future upstream merge can't silently regress them. All
network-free: StockTwits guards short-circuit before any HTTP call, and the
currency anchor is rendered from a synthetic identity dict.
"""

import pytest

from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.agents.utils.agent_utils import build_instrument_context


@pytest.mark.unit
class TestStockTwitsNonUSGuard:
    """StockTwits is a US cashtag service — non-US / numeric symbols must be
    short-circuited, never sent to the endpoint (no network in these tests)."""

    def test_korean_suffix_short_circuits(self):
        out = fetch_stocktwits_messages("005930.KS")
        assert "no coverage for non-US ticker 005930.KS" in out

    def test_kosdaq_suffix_short_circuits(self):
        assert "no coverage for non-US ticker 247540.KQ" in fetch_stocktwits_messages("247540.KQ")

    def test_other_non_us_suffix_short_circuits(self):
        # Tokyo .T would 404 on StockTwits too.
        assert "no coverage for non-US ticker 7203.T" in fetch_stocktwits_messages("7203.T")

    def test_bare_numeric_code_short_circuits(self):
        # Bare 005930 would be misresolved by StockTwits to internal id 5930
        # (IWM), injecting unrelated US data — must be refused.
        out = fetch_stocktwits_messages("005930")
        assert "no coverage for numeric-code ticker 005930" in out

    def test_non_ascii_name_degrades_gracefully(self):
        # A Korean company name must not crash the node with UnicodeEncodeError.
        out = fetch_stocktwits_messages("삼성전자")
        assert out.startswith("<")  # a placeholder, not an exception


@pytest.mark.unit
class TestCurrencyAnchor:
    """build_instrument_context surfaces the listing currency so agents don't
    assume USD for a KRW-denominated instrument."""

    def test_krw_currency_rendered(self):
        ctx = build_instrument_context(
            "005930.KS",
            identity={"company_name": "Samsung Electronics Co., Ltd.", "currency": "KRW"},
        )
        assert "quoted in KRW" in ctx

    def test_usd_currency_rendered(self):
        ctx = build_instrument_context(
            "AAPL", identity={"company_name": "Apple Inc.", "currency": "USD"}
        )
        assert "quoted in USD" in ctx

    def test_no_currency_no_clause(self):
        # Absent currency must not emit a dangling "quoted in" clause.
        ctx = build_instrument_context("AAPL", identity={"company_name": "Apple Inc."})
        assert "quoted in" not in ctx
