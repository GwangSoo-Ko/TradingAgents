"""Reddit search fetcher for ticker-specific discussion posts.

Primary path is Reddit's public JSON search endpoint
(``reddit.com/r/{sub}/search.json``), which carries the richest data
(score, comment count, body). Reddit's WAF increasingly returns
``HTTP 403 Blocked`` on that endpoint (issue #862), so when the JSON request
fails we transparently fall back to the public Atom/RSS search feed
(``/search.rss``). The RSS feed is gated less aggressively and serves the
same descriptive User-Agent we already send; the fallback lacks score /
comment counts, so RSS-sourced posts are marked and the formatter omits those
metrics rather than printing fake zeros.

Recall is widened by OR-ing the ticker with its company name when the caller
supplies one (e.g. ``NTRA OR "Natera"``): short or non-obvious tickers rarely
appear verbatim in retail posts, so the company name catches the same threads
under their conversational spelling.

No API key required either way. Returns formatted plaintext blocks ready for
prompt injection and degrades gracefully — returns a placeholder string
rather than raising, so callers never special-case missing data.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

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


def _search_qs(query: str, limit: int) -> str:
    return urlencode({
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: Optional[str]) -> Optional[float]:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _fetch_subreddit_rss(
    query: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Fallback path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(query, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.warning("Reddit RSS fetch failed for r/%s · q=%r: %s", sub, query, exc)
        return []

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit(
    query: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    url = _API.format(sub=sub, qs=_search_qs(query, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · q=%r: %s — falling back to RSS feed.",
            sub, query, exc,
        )
        return _fetch_subreddit_rss(query, sub, limit, timeout)


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

    The JSON search endpoint is tried first; on an HTTP 403 (or any other
    fetch error) it transparently falls back to the Atom/RSS feed, which
    lacks score / comment counts — those are then omitted rather than shown
    as zeros, and the subreddit header is marked accordingly.

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
        # RSS-sourced posts carry no id/name, so they are simply never deduped
        # (harmless: the RSS path returns a single feed with no OR-duplication).
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

        via_rss = any(p.get("source") == "rss" for p in unique_posts)
        header = f"r/{sub} — {len(unique_posts)} recent posts mentioning {label_terms}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in unique_posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {label_terms} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
