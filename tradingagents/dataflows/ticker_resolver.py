"""Resolve a user's free-text input (company name or ticker) to real tickers.

Hybrid, data-grounded design — the LLM helps interpret fuzzy/foreign input but
NEVER invents a ticker (a hallucinated ticker means analysing the wrong
company). Truth comes only from a grounded search (yfinance Search):

  1. Ticker-shaped input that yfinance recognises -> used directly.
  2. Otherwise grounded search on the raw query returns real candidates.
  3. If that finds nothing and an LLM is supplied, the LLM normalises the
     input into clean company-name search terms (typos -> "Samsung
     Electronics", "애플" -> "Apple", "the iphone maker" -> "Apple"), which are
     re-searched. The LLM only produces SEARCH TERMS; tickers always come from
     the grounded search.

The caller (CLI) presents the returned candidates — name + ticker + exchange —
for the user to choose, so an imprecise entry never silently analyses the wrong
instrument.
"""

from __future__ import annotations

import json
import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Candidate(NamedTuple):
    symbol: str
    name: str
    exchange: str
    quote_type: str
    source: str  # "direct" | "search"

    def label(self) -> str:
        """Human-readable picker line, e.g. 'Apple Inc. (AAPL · NMS)'."""
        bits = self.symbol
        if self.exchange:
            bits += f" · {self.exchange}"
        return f"{self.name} ({bits})" if self.name else self.symbol


# Ticker shapes we treat as "looks like a symbol, not a name":
#   AAPL, BRK.B, BTC-USD, 005930.KS, 7203.T, 005930
_TICKER_RE = re.compile(r"^[A-Za-z0-9]{1,6}([.\-][A-Za-z0-9]{1,5})?$")


def looks_like_ticker(query: str) -> bool:
    """True if ``query`` is shaped like a ticker (no spaces, ticker charset)."""
    q = (query or "").strip()
    return bool(q) and " " not in q and bool(_TICKER_RE.match(q))


def _search_yf(term: str, limit: int) -> list[Candidate]:
    """Grounded candidate search via yfinance. Returns [] on any failure."""
    try:
        import yfinance as yf
        res = yf.Search(term, max_results=limit)
        quotes = getattr(res, "quotes", None) or []
    except Exception as exc:  # noqa: BLE001 — search is best-effort
        logger.warning("yfinance Search failed for %r: %s", term, exc)
        return []

    out: list[Candidate] = []
    for q in quotes:
        symbol = (q.get("symbol") or "").strip()
        if not symbol:
            continue
        name = (q.get("shortname") or q.get("longname") or "").strip()
        out.append(Candidate(
            symbol=symbol,
            name=name,
            exchange=(q.get("exchange") or "").strip(),
            quote_type=(q.get("quoteType") or "").strip(),
            source="search",
        ))
    return out


def _llm_normalize(query: str, llm) -> list[str]:
    """Ask the LLM to turn fuzzy/foreign input into clean company-name search
    terms. Returns [] on any failure. The LLM is told to emit NAMES, not
    tickers — tickers are resolved by the grounded search afterwards.
    """
    prompt = (
        "A user is searching for a publicly traded company but may have typed a "
        "misspelling, a non-English name, or a description. Output ONLY a JSON "
        "array of up to 3 likely official company NAMES (in English) to search "
        "for — never ticker symbols, never commentary.\n"
        f'User input: "{query}"\n'
        'Example: input "samsng electronics" -> ["Samsung Electronics"]; '
        'input "애플" -> ["Apple"]; input "the iphone maker" -> ["Apple"].'
    )
    try:
        content = llm.invoke(prompt).content
        if isinstance(content, list):  # some providers return content blocks
            content = " ".join(str(c) for c in content)
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return []
        terms = json.loads(match.group(0))
        return [str(t).strip() for t in terms if str(t).strip()][:3]
    except Exception as exc:  # noqa: BLE001 — normalization is best-effort
        logger.warning("LLM query normalization failed for %r: %s", query, exc)
        return []


def _dedup(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in cands:
        key = c.symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def resolve_query(query: str, llm=None, limit: int = 8) -> list[Candidate]:
    """Resolve free-text ``query`` to a list of real ticker candidates.

    With ``llm`` supplied, fuzzy/foreign input that the grounded search can't
    match directly is normalized into search terms and re-searched. Returns a
    possibly-empty, de-duplicated candidate list (best first). Never fabricates
    a symbol.
    """
    q = (query or "").strip()
    if not q:
        return []

    # Fast path: ticker-shaped input that yfinance recognises -> use directly.
    if looks_like_ticker(q):
        from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
        ident = resolve_instrument_identity(q)
        if ident.get("company_name"):
            return [Candidate(
                symbol=q.upper(),
                name=ident["company_name"],
                exchange=ident.get("exchange", ""),
                quote_type=ident.get("quote_type", ""),
                source="direct",
            )]

    # Grounded search on the raw query.
    candidates = _search_yf(q, limit)

    # Nothing matched (typo / non-English / description): let the LLM normalize
    # the query into clean search terms, then re-search the grounded source.
    if not candidates and llm is not None:
        for term in _llm_normalize(q, llm):
            candidates.extend(_search_yf(term, limit))

    return _dedup(candidates)[:limit]
