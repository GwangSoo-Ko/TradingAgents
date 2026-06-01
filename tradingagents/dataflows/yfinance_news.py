"""yfinance-based news data fetching functions."""

from typing import Optional

import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .config import get_config
from .stockstats_utils import yf_retry


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    # Handle nested content structure
    if "content" in article:
        content = article["content"]
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")

        # Get URL from canonicalUrl or clickThroughUrl
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "")

        # Get publish date
        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "link": link,
            "pub_date": pub_date,
        }
    else:
        # Fallback for flat structure
        return {
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "pub_date": None,
        }


# When fewer than this many articles fall inside the requested window, surface
# up to _FALLBACK_N most-recent PRE-window articles for context. This keeps
# sparse-coverage tickers (e.g. KOSPI/KOSDAQ names on short windows) from
# returning "No news found" when relevant-but-slightly-older articles exist.
_MIN_IN_WINDOW = 3
_FALLBACK_N = 3


def _format_article(data: dict) -> str:
    out = f"### {data['title']} (source: {data['publisher']})\n"
    if data["summary"]:
        out += f"{data['summary']}\n"
    if data["link"]:
        out += f"Link: {data['link']}\n"
    return out + "\n"


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    article_limit = get_config()["news_article_limit"]
    try:
        stock = yf.Ticker(ticker)
        news = yf_retry(lambda: stock.get_news(count=article_limit))

        if not news:
            return f"No news found for {ticker}"

        # Parse date range for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_str = ""
        filtered_count = 0
        pre_window = []  # (date, data) for articles OLDER than the window

        for article in news:
            data = _extract_article_data(article)

            # Filter by date if publish time is available
            if data["pub_date"]:
                pub_date_naive = data["pub_date"].replace(tzinfo=None)
                if pub_date_naive < start_dt:
                    # Older than the window — keep as a fallback candidate.
                    pre_window.append((pub_date_naive, data))
                    continue
                if pub_date_naive > end_dt + relativedelta(days=1):
                    continue  # future-dated: drop (never surface — look-ahead safety)

            news_str += _format_article(data)
            filtered_count += 1

        # Sparse in-window coverage (common for non-US tickers on short windows):
        # surface the most-recent PRE-window articles, clearly labeled as
        # predating the window, so the analyst still has grounded context.
        note = ""
        if filtered_count < _MIN_IN_WINDOW and pre_window:
            pre_window.sort(key=lambda x: x[0], reverse=True)
            extra = pre_window[:_FALLBACK_N]
            note = (
                f"\n### Note: only {filtered_count} article(s) within "
                f"{start_date}..{end_date}; showing {len(extra)} most-recent "
                f"article(s) from BEFORE the window for context (these predate "
                f"the analysis window):\n\n"
            )
            for _, data in extra:
                note += _format_article(data)

        if filtered_count == 0 and not note:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}{note}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    # Region-aware macro queries: the run stashes the analysed ticker's region
    # in config (news_region, set in propagate); fall back to the US/default set.
    region = config.get("news_region")
    by_region = config.get("global_news_queries_by_region") or {}
    search_queries = by_region.get(region) or config["global_news_queries"]

    all_news = []
    seen_titles = set()

    try:
        for query in search_queries:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                news_count=limit,
                enable_fuzzy_query=True,
            ))

            if search.news:
                for article in search.news:
                    # Handle both flat and nested structures
                    if "content" in article:
                        data = _extract_article_data(article)
                        title = data["title"]
                    else:
                        title = article.get("title", "")

                    # Deduplicate by title
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append(article)

            if len(all_news) >= limit:
                break

        if not all_news:
            return f"No global news found for {curr_date}"

        # Calculate date range
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        news_str = ""
        for article in all_news[:limit]:
            # Handle both flat and nested structures
            if "content" in article:
                data = _extract_article_data(article)
                # Skip articles published after curr_date (look-ahead guard)
                if data.get("pub_date"):
                    pub_naive = data["pub_date"].replace(tzinfo=None) if hasattr(data["pub_date"], "replace") else data["pub_date"]
                    if pub_naive > curr_dt + relativedelta(days=1):
                        continue
                title = data["title"]
                publisher = data["publisher"]
                link = data["link"]
                summary = data["summary"]
            else:
                title = article.get("title", "No title")
                publisher = article.get("publisher", "Unknown")
                link = article.get("link", "")
                summary = ""

            news_str += f"### {title} (source: {publisher})\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching global news: {str(e)}"
