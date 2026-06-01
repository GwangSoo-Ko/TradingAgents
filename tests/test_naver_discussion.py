"""Naver 종목토론방 retail-sentiment source (Tier-2 Stage 3).

Network-free by default (mocked safe_get). One opt-in live test (TA_LIVE_NAVER=1).
Includes a PIPA guard: author identities must never appear in the output.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.dataflows import naver_discussion as nd
from tradingagents.dataflows.naver_discussion import fetch_discussion_sentiment


def _payload(*posts):
    return SimpleNamespace(json=lambda: {"result": {"posts": list(posts)}})


def _post(nick, title, body, agree=0, disagree=0, depth=0, cleanbot=True, when="2026-06-01T17:00:00"):
    return {
        "id": "1", "writtenAt": when, "title": title,
        "contentSwReplacedButImg": body, "recommendCount": agree,
        "notRecommendCount": disagree, "replyDepth": depth, "isCleanbotPassed": cleanbot,
        "writer": {"nickname": nick, "profileId": "pid-" + nick},
    }


@pytest.mark.unit
class TestDiscussionSentiment:
    def test_non_kr_placeholder(self):
        out = fetch_discussion_sentiment("AAPL")
        assert "not applicable" in out

    def test_aggregates_volume_and_engagement(self):
        posts = [
            _post("핑구칭", "삼전 오른다", "상승 기대", agree=9, disagree=3),
            _post("삼존마", "조정 온다", "하락 주의", agree=6, disagree=0),
        ]
        with patch.object(nd, "safe_get", return_value=_payload(*posts)):
            out = fetch_discussion_sentiment("005930.KS")
        assert "Discussion volume: 2" in out
        assert "추천/agree: 15" in out and "비추천/disagree: 3" in out
        assert "삼전 오른다" in out  # public post text included

    def test_no_author_identity_leak(self):
        # PIPA: nickname / profileId must never appear in the rendered block.
        posts = [_post("감각있는차트맛아이돌", "제목", "본문", agree=5)]
        with patch.object(nd, "safe_get", return_value=_payload(*posts)):
            out = fetch_discussion_sentiment("005930.KS")
        assert "감각있는차트맛아이돌" not in out
        assert "pid-" not in out

    def test_filters_replies_and_cleanbot(self):
        posts = [
            _post("a", "top-level", "ok", depth=0, cleanbot=True),
            _post("b", "reply", "x", depth=1, cleanbot=True),
            _post("c", "spam", "x", depth=0, cleanbot=False),
        ]
        with patch.object(nd, "safe_get", return_value=_payload(*posts)):
            out = fetch_discussion_sentiment("005930.KS")
        assert "Discussion volume: 1" in out
        assert "top-level" in out and "reply" not in out and "spam" not in out

    def test_empty_placeholder(self):
        with patch.object(nd, "safe_get", return_value=_payload()):
            assert "no Naver 종목토론방 posts" in fetch_discussion_sentiment("247540.KQ")

    def test_fetch_failure_degrades(self):
        with patch.object(nd, "safe_get", side_effect=RuntimeError("boom")):
            out = fetch_discussion_sentiment("005930.KS")
        assert out.startswith("<Naver 종목토론방 unavailable")


@pytest.mark.unit
class TestSentimentAnalystGating:
    def test_off_by_default(self, monkeypatch):
        from tradingagents.agents.analysts.sentiment_analyst import _maybe_fetch_kr_discussion
        from tradingagents.dataflows.config import set_config
        from tradingagents.default_config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["enable_kr_discussion_sentiment"] = False
        set_config(cfg)
        assert _maybe_fetch_kr_discussion("005930.KS") == ""

    def test_on_only_for_kr(self, monkeypatch):
        import tradingagents.agents.analysts.sentiment_analyst as sa
        from tradingagents.dataflows.config import set_config
        from tradingagents.default_config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["enable_kr_discussion_sentiment"] = True
        set_config(cfg)
        # non-KR -> still skipped (empty)
        assert sa._maybe_fetch_kr_discussion("AAPL") == ""
        # KR -> delegates to fetcher (mock it to avoid network)
        monkeypatch.setattr(
            "tradingagents.dataflows.naver_discussion.fetch_discussion_sentiment",
            lambda t: "MOCK_KR_BLOCK",
        )
        assert sa._maybe_fetch_kr_discussion("005930.KS") == "MOCK_KR_BLOCK"
        set_config(DEFAULT_CONFIG.copy())  # restore


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TA_LIVE_NAVER") != "1",
                    reason="set TA_LIVE_NAVER=1 to hit the real Naver discussion endpoint")
class TestDiscussionLive:
    def test_live_samsung(self):
        out = fetch_discussion_sentiment("005930.KS")
        assert "Discussion volume:" in out
