"""Naver per-ticker Korean news vendor (Tier-2 Stage 1).

Mostly network-free (mocked safe_get). One opt-in live test hits the real
Naver endpoint and is skipped unless TA_LIVE_NAVER=1, so CI stays hermetic.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.dataflows import naver_news
from tradingagents.dataflows.naver_news import get_news, _parse_naver_datetime, _flatten_items
from tradingagents.dataflows.symbol_utils import NoMarketDataError


# A faithful sample of the real endpoint shape (list of clusters -> items).
def _sample_payload(date="20260601"):
    return [
        {"total": 2, "items": [
            {"datetime": f"{date}1635", "officeName": "한국경제",
             "titleFull": "삼성전자, AI 메모리 투자 확대", "title": "삼성전자 AI 투자",
             "body": "삼성전자가 AI 메모리 생산능력을 늘린다.",
             "mobileNewsUrl": "https://n.news.naver.com/mnews/article/015/0005293690"},
        ]},
        {"total": 1, "items": [
            {"datetime": f"{date}0900", "officeName": "이데일리",
             "titleFull": "코스피 사상 최고", "body": "지수가 사상 최고를 경신했다.",
             "mobileNewsUrl": "https://n.news.naver.com/mnews/article/018/0006295517"},
        ]},
    ]


def _mock_resp(payload):
    return SimpleNamespace(json=lambda: payload)


@pytest.mark.unit
class TestNaverNewsParsing:
    def test_parse_datetime(self):
        assert _parse_naver_datetime("202606011635").strftime("%Y-%m-%d") == "2026-06-01"
        assert _parse_naver_datetime("bad") is None
        assert _parse_naver_datetime(None) is None

    def test_flatten_items(self):
        items = _flatten_items(_sample_payload())
        assert len(items) == 2
        assert items[0]["officeName"] == "한국경제"


@pytest.mark.unit
class TestNaverNewsVendor:
    def test_non_kr_ticker_raises_for_fallthrough(self):
        # Must raise (not return) so route_to_vendor tries the next vendor.
        with pytest.raises(ValueError):
            get_news("AAPL", "2026-05-25", "2026-06-02")

    def test_returns_korean_news_in_window(self):
        with patch.object(naver_news, "safe_get", return_value=_mock_resp(_sample_payload("20260601"))):
            out = get_news("005930.KS", "2026-05-25", "2026-06-02")
        assert "Korean News (Naver)" in out
        assert "삼성전자, AI 메모리 투자 확대" in out
        assert "source: 한국경제" in out
        assert "n.news.naver.com" in out

    def test_filters_out_of_window_and_future(self):
        # Article dated 2026-06-01; window ends 2026-05-30 -> nothing in window.
        with patch.object(naver_news, "safe_get", return_value=_mock_resp(_sample_payload("20260601"))):
            with pytest.raises(NoMarketDataError):
                get_news("005930.KS", "2026-05-20", "2026-05-30")

    def test_empty_payload_raises_no_market_data(self):
        with patch.object(naver_news, "safe_get", return_value=_mock_resp([])):
            with pytest.raises(NoMarketDataError):
                get_news("247540.KQ", "2026-05-25", "2026-06-02")

    def test_bare_six_digit_code_accepted(self):
        with patch.object(naver_news, "safe_get", return_value=_mock_resp(_sample_payload("20260601"))):
            out = get_news("005930", "2026-05-25", "2026-06-02")
        assert "Korean News (Naver)" in out


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TA_LIVE_NAVER") != "1",
                    reason="set TA_LIVE_NAVER=1 to hit the real Naver endpoint (needs KR network)")
class TestNaverNewsLive:
    def test_live_samsung_news(self):
        # Wide window so recent articles land inside it.
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=14)
        out = get_news("005930.KS", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        assert "Korean News (Naver)" in out
        assert "###" in out  # at least one article rendered
