"""Reddit search fetcher for ticker-specific discussion posts.

Uses Reddit's public JSON endpoints (``reddit.com/r/{sub}/search.json``)
which do not require an API key. Public throughput is ~10 requests per
minute per IP, well within budget for a single agent run that queries
a handful of finance subreddits per ticker.

Returns formatted plaintext blocks ready for prompt injection. Degrades
gracefully — returns a placeholder string rather than raising, so callers
never have to special-case missing data.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# Corporate-form suffixes stripped from a company name before using it as a
# Reddit search term. Reddit posts almost never include the legal form
# ("Apple Inc", "Natera, Inc.") so keeping it would actively hurt recall.
# The (?:...)+ form chews through stacked suffixes like "Co., Ltd." or
# "Company Limited" in one pass.
_CORP_SUFFIXES = re.compile(
    r"(?:,?\s+(?:inc|incorporated|corp|corporation|co|ltd|limited|plc|sa|nv|ag|holdings|group|company)\.?)+$",
    re.IGNORECASE,
)


def _sanitize_company_name(name: str) -> str:
    """Strip legal-form suffixes and trailing punctuation for a search term."""
    cleaned = _CORP_SUFFIXES.sub("", name.strip()).strip(" ,.")
    return cleaned


def _build_query(ticker: str, company_name: Optional[str]) -> tuple[str, Optional[str]]:
    """Build the Reddit search ``q`` value plus the sanitized company name.

    Returns ``(query, sanitized_name_or_None)``. The second value is the
    cleaned company name when it was actually OR'd into the query, or
    ``None`` when the call effectively falls back to a ticker-only search
    (no name supplied, or sanitization left something redundant). Callers
    use it to build a human-readable label without re-sanitizing.
    """
    if not company_name:
        return ticker, None
    cleaned = _sanitize_company_name(company_name)
    # If sanitization left a name that is identical to the ticker (case-insensitive)
    # or empty, drop the OR branch to avoid noise.
    if not cleaned or cleaned.upper() == ticker.upper():
        return ticker, None
    return f'{ticker} OR "{cleaned}"', cleaned


def _fetch_subreddit(
    query: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    qs = urlencode({
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })
    url = _API.format(sub=sub, qs=qs)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Reddit fetch failed for r/%s · q=%r: %s", sub, query, exc)
        return []
    children = (payload.get("data") or {}).get("children") or []
    return [c.get("data", {}) for c in children if isinstance(c, dict)]


def fetch_reddit_posts(
    ticker: str,
    company_name: Optional[str] = None,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` (and optionally
    ``company_name``) across finance subreddits and return them as a
    formatted plaintext block.

    Passing ``company_name`` (e.g. ``"Natera"`` for ticker ``"NTRA"``)
    expands recall: many retail posts spell out the company instead of
    using the cashtag. Results from the OR'd query are deduped within
    each subreddit by Reddit post id, so the same thread isn't reported
    twice. Falls back to the bare ticker when ``company_name`` is ``None``
    or sanitization leaves nothing useful.

    ``inter_request_delay`` keeps us under Reddit's public rate limit
    (~10 req/min per IP) even if the caller queries many subreddits.
    """
    query, sanitized_name = _build_query(ticker, company_name)
    label_terms = (
        f"{ticker.upper()} / {sanitized_name}" if sanitized_name else ticker.upper()
    )

    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(query, sub, limit_per_sub, timeout)
        # Dedupe by Reddit post id in case the OR'd query catches the same
        # thread under multiple terms. Order-preserving so newest stays first.
        seen_ids: set[str] = set()
        unique_posts: list[dict] = []
        for p in posts:
            pid = p.get("id") or p.get("name")
            if pid and pid in seen_ids:
                continue
            if pid:
                seen_ids.add(pid)
            unique_posts.append(p)
        total_posts += len(unique_posts)
        if not unique_posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {label_terms} in the past 7 days>")
            continue

        lines = [f"r/{sub} — {len(unique_posts)} recent posts mentioning {label_terms}:"]
        for p in unique_posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {label_terms} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
