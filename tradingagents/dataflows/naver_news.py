"""Naver Finance per-ticker Korean-language news vendor.

yfinance returns sparse, English-only news for KOSPI/KOSDAQ (.KS/.KQ) tickers,
so this vendor pulls native Korean headlines from Naver's mobile finance news
endpoint (``m.stock.naver.com/api/news/stock/{code}``), which returns clean
JSON without an API key.

OPT-IN: registered for ``get_news`` but NOT a default in ``data_vendors``, so it
only runs when a user sets ``news_data`` (or ``tool_vendors['get_news']``) to
include ``"naver"`` — e.g. ``"naver,yfinance"``. For non-Korean tickers it
raises so :func:`route_to_vendor` falls through to the next vendor.

NOTE: this queries an undocumented Naver endpoint. It is not enabled by default;
review Naver's ToS before turning it on as a category default, especially for a
redistributed deployment.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .config import get_config
from .kr_utils import is_kr_ticker, to_krx_code
from .rate_limit import safe_get
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

_NEWS_API = "https://m.stock.naver.com/api/news/stock/{code}"


def _parse_naver_datetime(raw) -> datetime | None:
    """Naver article ``datetime`` is ``"YYYYMMDDHHMM"`` — take the date part."""
    try:
        return datetime.strptime(str(raw)[:8], "%Y%m%d")
    except (ValueError, TypeError):
        return None


def _flatten_items(payload) -> list[dict]:
    """The endpoint returns a list of clusters, each ``{total, items:[...]}``."""
    items: list[dict] = []
    for cluster in payload or []:
        if isinstance(cluster, dict):
            items.extend(cluster.get("items", []) or [])
    return items


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Per-ticker Korean news from Naver Finance for a KOSPI/KOSDAQ ticker.

    Returns the same markdown shape as the yfinance/alpha_vantage news vendors.
    Raises ``ValueError`` for non-Korean tickers (dispatcher tries the next
    vendor) and ``NoMarketDataError`` when no article falls within
    ``[start_date, end_date]`` (so a ``"naver,yfinance"`` chain falls back to
    yfinance). Articles dated after ``end_date`` are never surfaced
    (look-ahead safety).
    """
    if not is_kr_ticker(ticker):
        raise ValueError(f"naver vendor only serves Korean tickers, got {ticker!r}")
    code = to_krx_code(ticker)

    limit = get_config().get("news_article_limit", 20)
    resp = safe_get(_NEWS_API.format(code=code), params={"pageSize": limit, "page": 1})

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    blocks = []
    for art in _flatten_items(resp.json()):
        art_dt = _parse_naver_datetime(art.get("datetime"))
        if art_dt is not None and not (start_dt <= art_dt <= end_dt):
            continue  # outside window (incl. future-dated) -> skip
        title = (art.get("titleFull") or art.get("title") or "").replace("\n", " ").strip()
        if not title:
            continue
        office = (art.get("officeName") or "").strip()
        body = (art.get("body") or "").replace("\n", " ").strip()
        url = art.get("mobileNewsUrl") or ""
        block = f"### {title} (source: {office})\n"
        if body:
            block += f"{body}\n"
        if url:
            block += f"Link: {url}\n"
        blocks.append(block)

    if not blocks:
        # Let the dispatcher fall back to the next vendor (e.g. yfinance).
        raise NoMarketDataError(ticker, code, "no Naver news in the requested window")

    header = f"## {ticker} Korean News (Naver), from {start_date} to {end_date}:\n\n"
    return header + "\n".join(blocks)
