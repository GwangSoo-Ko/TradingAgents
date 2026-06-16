"""Hybrid ticker/name resolver (name -> real ticker candidates).

Network-free by default (mocked yf.Search + mocked LLM). Two opt-in live tests
hit the real yfinance Search and run only when TA_LIVE_RESOLVER=1.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.dataflows import ticker_resolver as tr
from tradingagents.dataflows.ticker_resolver import (
    Candidate,
    _llm_normalize,
    looks_like_ticker,
    resolve_query,
)


@pytest.mark.unit
class TestLooksLikeTicker:
    def test_ticker_shapes(self):
        for t in ("AAPL", "BRK.B", "BTC-USD", "005930.KS", "7203.T", "005930"):
            assert looks_like_ticker(t), t

    def test_name_shapes(self):
        for n in ("Samsung Electronics", "apple inc", "the iphone maker", "", "삼성 전자"):
            assert not looks_like_ticker(n), n


@pytest.mark.unit
class TestResolveQuery:
    def test_empty_returns_empty(self):
        assert resolve_query("") == []

    def test_direct_ticker_fast_path(self):
        # Ticker-shaped + yfinance recognises it -> single direct candidate, no search.
        with patch("tradingagents.agents.utils.agent_utils.resolve_instrument_identity",
                   return_value={"company_name": "Apple Inc.", "exchange": "NMS", "quote_type": "EQUITY"}), \
             patch.object(tr, "_search_yf") as search:
            out = resolve_query("AAPL")
            assert len(out) == 1 and out[0].symbol == "AAPL" and out[0].source == "direct"
            search.assert_not_called()

    def test_name_uses_grounded_search(self):
        with patch.object(tr, "_search_yf", return_value=[
            Candidate("AAPL", "Apple Inc.", "NMS", "EQUITY", "search"),
            Candidate("APLE", "Apple Hospitality REIT", "NYQ", "EQUITY", "search"),
        ]):
            out = resolve_query("Apple")
            assert [c.symbol for c in out] == ["AAPL", "APLE"]

    def test_fuzzy_empty_then_llm_normalize(self):
        # Raw search empty (typo) -> LLM normalizes -> re-search finds the ticker.
        def fake_search(term, limit):
            if term == "samsng":
                return []
            return [Candidate("005930.KS", "Samsung Electronics", "KSC", "EQUITY", "search")]
        llm = SimpleNamespace(invoke=lambda p: SimpleNamespace(content='["Samsung Electronics"]'))
        with patch.object(tr, "_search_yf", side_effect=fake_search):
            out = resolve_query("samsng", llm=llm)
        assert out and out[0].symbol == "005930.KS"

    def test_no_llm_no_fallback(self):
        # Without an LLM, an unmatched fuzzy query just returns [].
        with patch.object(tr, "_search_yf", return_value=[]):
            assert resolve_query("samsng") == []

    def test_dedup_by_symbol(self):
        with patch.object(tr, "_search_yf", return_value=[
            Candidate("AAPL", "Apple Inc.", "NMS", "EQUITY", "search"),
            Candidate("AAPL", "Apple Inc.", "NMS", "EQUITY", "search"),
        ]):
            out = resolve_query("Apple")
            assert len(out) == 1


@pytest.mark.unit
class TestLlmNormalize:
    def test_parses_json_array(self):
        llm = SimpleNamespace(invoke=lambda p: SimpleNamespace(content='Here: ["Samsung Electronics", "Samsung"]'))
        assert _llm_normalize("samsng", llm) == ["Samsung Electronics", "Samsung"]

    def test_bad_output_returns_empty(self):
        llm = SimpleNamespace(invoke=lambda p: SimpleNamespace(content="no json here"))
        assert _llm_normalize("???", llm) == []

    def test_llm_raises_returns_empty(self):
        def boom(p): raise RuntimeError("llm down")
        assert _llm_normalize("x", SimpleNamespace(invoke=boom)) == []

    def test_candidate_label(self):
        assert Candidate("AAPL", "Apple Inc.", "NMS", "EQUITY", "search").label() == "Apple Inc. (AAPL · NMS)"


@pytest.mark.unit
class TestCliResolverLlm:
    def test_no_provider_key_returns_none(self, monkeypatch):
        from cli.utils import _build_resolver_llm
        for k in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        assert _build_resolver_llm() is None

    def test_uses_first_available_key(self, monkeypatch):
        import cli.utils as u
        for k in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        sentinel = object()
        captured = {}
        def fake_create(provider, model):
            captured["provider"] = provider
            return SimpleNamespace(get_llm=lambda: sentinel)
        monkeypatch.setattr("tradingagents.llm_clients.create_llm_client", fake_create)
        assert u._build_resolver_llm() is sentinel
        assert captured["provider"] == "google"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TA_LIVE_RESOLVER") != "1",
                    reason="set TA_LIVE_RESOLVER=1 to hit the real yfinance Search")
class TestResolverLive:
    def test_live_name_to_ticker(self):
        out = resolve_query("Apple")
        assert any(c.symbol == "AAPL" for c in out)

    def test_live_korean_name(self):
        # yfinance Search surfaces the KR ticker for this Korean name (no LLM).
        out = resolve_query("SK하이닉스")
        assert any(c.symbol == "000660.KS" for c in out)
