"""Tier-0 + Tier-1 Korean-market coverage fixes.

These are fork-local correctness fixes for KOSPI/KOSDAQ tickers; the tests
lock them in so a future upstream merge can't silently regress them. All
network-free: StockTwits guards short-circuit before any HTTP call, the
currency anchor is rendered from a synthetic identity dict, the news fallback
and fundamentals derivation are driven by mocked yfinance objects, and region
selection is pure config logic.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tradingagents.agents.context import build_instrument_context
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages


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


@pytest.mark.unit
class TestNameTickerDisplay:
    """Reports should read 'Company Name (TICKER)' rather than a bare ticker."""

    def test_prompt_instructs_name_ticker_form(self):
        ctx = build_instrument_context(
            "005930.KS", identity={"company_name": "Samsung Electronics Co., Ltd."}
        )
        assert 'Samsung Electronics Co., Ltd. (005930.KS)' in ctx
        # still keep the exact ticker for tool calls / FINAL line
        assert "exact ticker" in ctx

    def test_no_name_no_display_instruction(self):
        # Without a resolved name, no name(ticker) instruction is emitted.
        ctx = build_instrument_context("ZZZZ")
        assert "report headings and prose" not in ctx

    def test_display_label(self):
        from unittest.mock import patch

        import tradingagents.agents.context as au
        with patch.object(au, "resolve_instrument_identity",
                          return_value={"company_name": "Apple Inc."}):
            assert au.instrument_display_label("AAPL") == "Apple Inc. (AAPL)"
        with patch.object(au, "resolve_instrument_identity", return_value={}):
            assert au.instrument_display_label("ZZZZ") == "ZZZZ"


@pytest.mark.unit
class TestNewsRegion:
    """Tier-1 Fix #1: macro-news region selection by exchange suffix."""

    def test_region_for_ticker(self):
        from tradingagents.default_config import news_region_for_ticker
        assert news_region_for_ticker("005930.KS") == "KR"
        assert news_region_for_ticker("247540.KQ") == "KR"
        assert news_region_for_ticker("AAPL") is None
        assert news_region_for_ticker("7203.T") is None  # mapped to benchmark, not news-region

    def test_region_query_selection(self):
        from tradingagents.default_config import DEFAULT_CONFIG, news_region_for_ticker
        by_region = DEFAULT_CONFIG["global_news_queries_by_region"]
        # KR run selects Korea macro; unmapped falls back to the US/default list.
        kr = by_region.get(news_region_for_ticker("005930.KS")) or DEFAULT_CONFIG["global_news_queries"]
        us = by_region.get(news_region_for_ticker("AAPL")) or DEFAULT_CONFIG["global_news_queries"]
        assert any("Korea" in q for q in kr)
        assert us == DEFAULT_CONFIG["global_news_queries"]
        assert any("Federal Reserve" in q for q in kr)  # keeps a global driver


@pytest.mark.unit
class TestNewsDateWindowFallback:
    """Tier-1 Fix #2: surface most-recent PRE-window articles when in-window
    coverage is sparse, never future-dated (look-ahead safety)."""

    def _patch(self, monkeypatch, articles):
        import tradingagents.dataflows.vendors.yahoo.news as yn
        monkeypatch.setattr(yn, "_extract_article_data", lambda a: a)
        monkeypatch.setattr(yn, "yf_retry", lambda f: f())
        stock = SimpleNamespace(get_news=lambda count: articles)
        monkeypatch.setattr(yn, "yf", SimpleNamespace(Ticker=lambda t: stock))
        return yn

    def test_sparse_window_surfaces_older_not_future(self, monkeypatch):
        articles = [
            {"pub_date": datetime(2026, 1, 1), "title": "OldA", "publisher": "P", "summary": "s", "link": "l"},
            {"pub_date": datetime(2026, 1, 2), "title": "OldB", "publisher": "P", "summary": "s", "link": "l"},
            {"pub_date": datetime(2026, 5, 1), "title": "FutureX", "publisher": "P", "summary": "s", "link": "l"},
        ]
        yn = self._patch(monkeypatch, articles)
        out = yn.get_news_yfinance("005930.KS", "2026-03-01", "2026-03-07")
        assert "BEFORE the window" in out      # fallback note present
        assert "OldB" in out and "OldA" in out  # older articles surfaced
        assert "FutureX" not in out             # future-dated never surfaced

    def test_kst_morning_article_on_window_start_is_not_dropped(self, monkeypatch):
        # An article stamped 08:00 KST on the FIRST day of the window is
        # 2026-02-28T23:00Z -- just outside the UTC window. It must be surfaced
        # as pre-window context, never silently discarded.
        #
        # Regression: the pre-window test truncated the offset
        # (``pub_date.replace(tzinfo=None)`` -> 03-01 08:00, "not older than
        # start") while the window check (``in_window``) converted it (-> 02-28
        # 23:00Z, "outside"), so the article fell between both nets and was
        # dropped as if future-dated -- despite being in the past. Korean
        # morning news on day one of a window vanished entirely.
        kst = timezone(timedelta(hours=9))
        articles = [
            {"pub_date": datetime(2026, 3, 1, 8, 0, tzinfo=kst), "title": "SeoulMorning",
             "publisher": "P", "summary": "s", "link": "l"},
        ]
        yn = self._patch(monkeypatch, articles)
        out = yn.get_news_yfinance("005930.KS", "2026-03-01", "2026-03-07")
        assert "SeoulMorning" in out
        assert "BEFORE the window" in out  # correctly labeled, not silently dropped

    def test_no_articles_at_all_returns_none_message(self, monkeypatch):
        # An empty feed never observed this past window, so it is reported as
        # unavailable rather than as "no news happened" (upstream coverage_gap).
        yn = self._patch(monkeypatch, [])
        out = yn.get_news_yfinance("005930.KS", "2026-03-01", "2026-03-07")
        assert "unavailable" in out and "not an absence" in out
        assert "###" not in out  # no fabricated article body

    def test_empty_window_gives_the_verdict_then_the_older_context(self, monkeypatch):
        # Coverage reaches back past the window start, so the empty window is a
        # real absence: the verdict comes first and the pre-window articles
        # follow as clearly labeled context.
        articles = [
            {"pub_date": datetime(2026, 2, 20), "title": "OlderA", "publisher": "P", "summary": "s", "link": "l"},
        ]
        yn = self._patch(monkeypatch, articles)
        out = yn.get_news_yfinance("005930.KS", "2026-03-01", "2026-03-07")
        assert out.startswith("No news found for 005930.KS between 2026-03-01 and 2026-03-07")
        assert "unavailable" not in out
        assert out.index("No news found") < out.index("BEFORE the window") < out.index("OlderA")

    def test_undated_article_never_leaks_into_a_historical_window(self, monkeypatch):
        # A backtest can't prove an undated article predates its as-of date, so
        # it must not be surfaced -- neither in the window nor as pre-window
        # context (#992/#1007). The pre-window branch once let it through.
        articles = [
            {"pub_date": None, "title": "UndatedX", "publisher": "P", "summary": "s", "link": "l"},
            {"pub_date": datetime(2026, 3, 3), "title": "InWindow", "publisher": "P", "summary": "s", "link": "l"},
            {"pub_date": datetime(2026, 2, 20), "title": "OlderA", "publisher": "P", "summary": "s", "link": "l"},
        ]
        yn = self._patch(monkeypatch, articles)
        out = yn.get_news_yfinance("005930.KS", "2026-03-01", "2026-03-07")
        assert "InWindow" in out and "OlderA" in out
        assert "UndatedX" not in out

    def test_undated_article_is_kept_in_a_live_window(self, monkeypatch):
        # A window reaching the present can't hold a future article, so an
        # undated one is kept there.
        today = datetime.now(timezone.utc).date()
        articles = [
            {"pub_date": None, "title": "UndatedLive", "publisher": "P", "summary": "s", "link": "l"},
        ]
        yn = self._patch(monkeypatch, articles)
        out = yn.get_news_yfinance(
            "005930.KS", (today - timedelta(days=7)).isoformat(), today.isoformat()
        )
        assert "UndatedLive" in out


@pytest.mark.unit
class TestDerivedFundamentals:
    """Tier-1 Fix #3: derive PE/EPS/PB/BookValue + reporting currency when
    Yahoo omits them for KR tickers; leave US tickers untouched."""

    def _patch(self, monkeypatch, info, bs=None):
        import tradingagents.dataflows.vendors.yahoo.fundamentals as yf_mod
        monkeypatch.setattr(yf_mod, "normalize_symbol", lambda s: s)
        monkeypatch.setattr(yf_mod, "yf_retry", lambda f: f())
        ticker_obj = SimpleNamespace(info=info, quarterly_balance_sheet=bs)
        monkeypatch.setattr(yf_mod, "yf", SimpleNamespace(Ticker=lambda s: ticker_obj))
        return yf_mod

    def test_kr_missing_ratios_are_derived(self, monkeypatch):
        import pandas as pd
        info = {
            "longName": "Samsung Electronics Co., Ltd.",
            "currency": "KRW", "financialCurrency": "KRW",
            "currentPrice": 349000.0, "sharesOutstanding": 5_764_191_903,
            "netIncomeToCommon": 83_333_740_494_848,
            "trailingPE": None, "trailingEps": None,
            "priceToBook": None, "bookValue": None,
        }
        bs = pd.DataFrame(
            {pd.Timestamp("2026-03-31"): [4.7398e14, 6.598e9]},
            index=["Stockholders Equity", "Ordinary Shares Number"],
        )
        yf_mod = self._patch(monkeypatch, info, bs)
        out = yf_mod.get_fundamentals("005930.KS")
        assert "Reporting currency: KRW" in out
        assert "EPS (TTM): 14457.14 (derived)" in out
        assert "PE Ratio (TTM): 24.14 (derived)" in out
        # Book value comes from the balance sheet: a silently failed lookup would
        # still print both labels, as "N/A (vendor)".
        assert "Book Value: 71836.92 (derived)" in out
        assert "Price to Book: 4.86 (derived)" in out
        assert "N/A (vendor)" not in out
        assert "Data retrieved on" not in out  # no wall-clock stamp on the live path (#1300)

    _KR_INFO = {
        "longName": "KR Co", "currency": "KRW", "currentPrice": 10000.0,
        "sharesOutstanding": 1_000_000, "netIncomeToCommon": 5e8,
        "trailingPE": None, "trailingEps": None, "priceToBook": None, "bookValue": None,
    }

    def test_kr_book_value_skips_a_quarter_yahoo_has_not_filled_in(self, monkeypatch):
        """The newest balance-sheet column stays blank (NaN) for a while after a
        quarter closes -- the KR runs that fall back to Yahoo when wisereport
        fails. NaN is truthy, so the first-column read derived 'Book Value: nan'
        and handed the fundamentals analyst a NaN as a valuation anchor."""
        import re

        import pandas as pd
        bs = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [float("nan"), float("nan")],
             pd.Timestamp("2026-03-31"): [9.0e9, 1_000_000.0]},
            index=["Stockholders Equity", "Ordinary Shares Number"],
        )
        yf_mod = self._patch(monkeypatch, dict(self._KR_INFO), bs)
        out = yf_mod.get_fundamentals("123456.KQ")
        assert not re.search(r"\bnan\b", out, re.IGNORECASE), out
        assert "Book Value: 9000.0 (derived)" in out
        assert "Price to Book: 1.11 (derived)" in out

    def test_kr_book_value_with_no_reported_quarter_is_marked_na(self, monkeypatch):
        import re

        import pandas as pd
        bs = pd.DataFrame(
            {pd.Timestamp("2026-06-30"): [float("nan"), float("nan")]},
            index=["Stockholders Equity", "Ordinary Shares Number"],
        )
        yf_mod = self._patch(monkeypatch, dict(self._KR_INFO), bs)
        out = yf_mod.get_fundamentals("123456.KQ")
        assert not re.search(r"\bnan\b", out, re.IGNORECASE), out
        assert "Book Value: N/A (vendor)" in out
        assert "Price to Book: N/A (vendor)" in out
        assert "EPS (TTM): 500.0 (derived)" in out  # the income side is unaffected

    def test_kr_underivable_marks_na(self, monkeypatch):
        # KOSDAQ name where even inputs are missing -> N/A (vendor), not dropped.
        info = {
            "longName": "Some KOSDAQ Co.", "currency": "KRW",
            "regularMarketPrice": 41350.0,
            "trailingPE": None, "trailingEps": None,
            "priceToBook": None, "bookValue": None,
            "netIncomeToCommon": None, "sharesOutstanding": None,
        }
        yf_mod = self._patch(monkeypatch, info, None)
        out = yf_mod.get_fundamentals("035720.KQ")
        assert "N/A (vendor)" in out  # criticals surfaced, not silently dropped

    def test_us_ticker_uses_vendor_values_no_derived(self, monkeypatch):
        info = {
            "longName": "Apple Inc.", "currency": "USD",
            "currentPrice": 250.0, "trailingPE": 37.7, "trailingEps": 8.27,
            "priceToBook": 43.0, "bookValue": 7.26,
        }
        yf_mod = self._patch(monkeypatch, info, None)
        out = yf_mod.get_fundamentals("AAPL")
        assert "(derived)" not in out          # vendor values used directly
        assert "PE Ratio (TTM): 37.7" in out
