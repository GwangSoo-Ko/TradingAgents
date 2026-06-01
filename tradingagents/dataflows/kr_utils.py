"""Korean-market helpers shared across KR data vendors.

KOSPI/KOSDAQ tickers reach the pipeline in Yahoo's exchange-qualified form
(``005930.KS`` / ``247540.KQ``), but KRX / Naver / DART endpoints want the bare
six-digit code (``005930``). These helpers translate between the two while the
agent-facing symbol stays the qualified form everywhere (per the CLAUDE.md
exchange-qualified-ticker round-trip rule).
"""

from __future__ import annotations

import re

# Yahoo exchange suffixes for the two Korea Exchange boards.
_KR_SUFFIXES = (".KS", ".KQ")
# A KRX issue code is exactly six digits (e.g. 005930 Samsung, 247540 Ecopro BM).
_KRX_CODE_RE = re.compile(r"^\d{6}$")


def is_kr_ticker(ticker: str) -> bool:
    """True if ``ticker`` is a Korea Exchange listing.

    Accepts both the Yahoo-qualified form (``005930.KS`` / ``247540.KQ``) and a
    bare six-digit KRX code (``005930``). Used to gate KR-only vendors so they
    short-circuit (and fall through to the next vendor) for non-KR symbols.
    """
    t = (ticker or "").upper().strip()
    return t.endswith(_KR_SUFFIXES) or bool(_KRX_CODE_RE.match(t))


def to_krx_code(ticker: str) -> str:
    """Return the bare six-digit KRX code for a Korean ticker.

    ``005930.KS`` / ``247540.KQ`` / ``005930`` all map to the six-digit code.
    Raises ``ValueError`` for anything that isn't a KR ticker so callers can
    let the dispatcher skip to the next vendor rather than querying a KR
    endpoint with a US symbol.
    """
    t = (ticker or "").upper().strip()
    for suffix in _KR_SUFFIXES:
        if t.endswith(suffix):
            base = t[: -len(suffix)]
            if _KRX_CODE_RE.match(base):
                return base
            raise ValueError(f"not a six-digit KRX code: {ticker!r}")
    if _KRX_CODE_RE.match(t):
        return t
    raise ValueError(f"not a Korean ticker: {ticker!r}")
