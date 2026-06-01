"""Naver 종목토론방 (retail discussion board) sentiment source for KR tickers.

Korean retail sentiment lives on Naver's per-stock discussion board, which
StockTwits and English Reddit structurally cannot cover. This pulls the recent
post listing from Naver's mobile front-api JSON (plain HTTP, no API key, no
headless browser) and renders an aggregate signal for the sentiment analyst.

PRIVACY (PIPA): author identities (nickname / profileId) are deliberately
dropped — never returned, never persisted. Only the discussion volume, public
post text (title + short body), and community engagement (추천/비추천) are
surfaced, transiently, into the analyst prompt. There is no bull/bear label on
the board, so this is presented as a qualitative attention/mood signal for the
LLM to read — NOT a fabricated ratio.

OPT-IN: gated behind config['enable_kr_discussion_sentiment'] (default False),
since it queries an undocumented Naver endpoint. Off by default.
"""

from __future__ import annotations

import logging

from .kr_utils import is_kr_ticker, to_krx_code
from .rate_limit import safe_get

logger = logging.getLogger(__name__)

_LIST_API = "https://m.stock.naver.com/front-api/discussion/list"


def _clean(text: str, limit: int = 80) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:limit] + "…") if len(t) > limit else t


def fetch_discussion_sentiment(ticker: str, limit: int = 20) -> str:
    """Return an aggregate 종목토론방 sentiment block for a KR ticker.

    Degrades gracefully to a placeholder string on any failure (the sentiment
    analyst always gets a string). Author identities are never included.
    """
    if not is_kr_ticker(ticker):
        return f"<Naver 종목토론방 not applicable for non-Korean ticker {ticker}>"
    try:
        code = to_krx_code(ticker)
        resp = safe_get(
            _LIST_API,
            params={"discussionType": "domesticStock", "itemCode": code, "pageSize": limit},
        )
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — graceful, never crash the node
        logger.warning("Naver discussion fetch failed for %s: %s", ticker, exc)
        return f"<Naver 종목토론방 unavailable: {type(exc).__name__}>"

    result = payload.get("result") or {}
    posts = [
        p for p in (result.get("posts") or [])
        if p.get("replyDepth", 0) == 0 and p.get("isCleanbotPassed", True)
    ]
    if not posts:
        return f"<no Naver 종목토론방 posts found for {ticker}>"

    total_agree = sum(int(p.get("recommendCount") or 0) for p in posts)
    total_disagree = sum(int(p.get("notRecommendCount") or 0) for p in posts)

    # Show the most-engaged posts (agree+disagree) so the LLM reads what the
    # community is actually reacting to. Author identities are omitted.
    ranked = sorted(
        posts,
        key=lambda p: int(p.get("recommendCount") or 0) + int(p.get("notRecommendCount") or 0),
        reverse=True,
    )
    lines = [
        f"Discussion volume: {len(posts)} recent top-level posts "
        f"(retail attention proxy). Community engagement totals across these "
        f"posts — 추천/agree: {total_agree}, 비추천/disagree: {total_disagree}.",
        "",
        "Recent posts by engagement (추천/비추천 are community reactions to the "
        "post, not a bull/bear label — read the Korean text for mood; author "
        "identities omitted):",
    ]
    for p in ranked[:limit]:
        written = (p.get("writtenAt") or "")[:16].replace("T", " ")
        agree = int(p.get("recommendCount") or 0)
        disagree = int(p.get("notRecommendCount") or 0)
        title = _clean(p.get("title"), 60)
        body = _clean(p.get("contentSwReplacedButImg") or p.get("contentSwReplaced"), 100)
        lines.append(f"- [{written} · 👍{agree} 👎{disagree}] {title}" + (f" — {body}" if body else ""))

    return "\n".join(lines)
