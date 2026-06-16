"""wisereport (Naver-affiliated) consensus & forward-estimate vendor for KR tickers.

Complements the OpenDART vendor: OpenDART gives FSS-audited *past* statements,
while wisereport adds what filings never contain — forward analyst ESTIMATES
(year+1/+2), the consensus TARGET PRICE and investment opinion, and valuation
multiples (PER/PBR/ROE/EV-EBITDA). Empirically the overlap with OpenDART is only
the income-statement actuals; the rest is complementary.

All data is in the plain-HTTP responses (verified against a headless-browser
render: the consensus target price, opinion, forward estimates and ratios are
all server-side — crawl4ai/Playwright is NOT needed; the browser-only extra was
redundant detail tables).

This vendor MERGES OpenDART (audited actuals + balance sheet, best-effort) with
the wisereport consensus/estimates into one fundamentals report, for the most
complete KR picture. Opt-in via fundamental_data="wisereport,yfinance".
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from .kr_utils import is_kr_ticker, to_krx_code
from .rate_limit import safe_get
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

_FIN_URL = "https://navercomp.wisereport.co.kr/v2/company/cF1002.aspx"
_MAIN_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx"
# wisereport serves these only to a browser-like UA + a Referer to its own host.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://navercomp.wisereport.co.kr/",
}
_YEAR_RE = re.compile(r"^\d{4}\([AE]\)$")


def _parse_financials(html: str) -> list[dict]:
    """Parse the cF1002 financial-series table (3 actual + 2 estimate years).

    Each data row is 12 cells: year | revenue | rev_YoY | OP | NI | EPS | PER |
    PBR | ROE | EV/EBITDA | net_debt | statement_basis.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) >= 12 and _YEAR_RE.match(cells[0]):
            rows.append({
                "year": cells[0], "revenue": cells[1], "op_income": cells[3],
                "net_income": cells[4], "eps": cells[5], "per": cells[6],
                "pbr": cells[7], "roe": cells[8], "ev_ebitda": cells[9],
            })
    return rows


def _parse_consensus(html: str) -> dict | None:
    """Parse the analyst-consensus block (opinion, target price, fwd EPS/PER, #firms)."""
    text = re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    m = re.search(
        r"추정기관수\s+([\d.]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+(\d+)", text
    )
    if not m:
        return None
    return {
        "opinion": m.group(1), "target_price": m.group(2),
        "fwd_eps": m.group(3), "fwd_per": m.group(4), "n_institutions": m.group(5),
    }


def _wisereport_section(code: str) -> str:
    fin_html = safe_get(f"{_FIN_URL}?cmp_cd={code}&finGubun=MAIN", headers=_HEADERS).text
    main_html = safe_get(f"{_MAIN_URL}?cmp_cd={code}", headers=_HEADERS).text
    rows = _parse_financials(fin_html)
    consensus = _parse_consensus(main_html)
    if not rows and not consensus:
        raise NoMarketDataError(code, code, "wisereport returned no parseable data")

    out = ["## wisereport — forward estimates, consensus & valuation (억원 / 원)"]
    if consensus:
        out.append(
            f"Analyst consensus: opinion {consensus['opinion']} (1=strong sell … 5=strong buy), "
            f"target price {consensus['target_price']} KRW, forward EPS {consensus['fwd_eps']}, "
            f"forward PER {consensus['fwd_per']}, from {consensus['n_institutions']} institutions."
        )
    if rows:
        out.append("Financial series (A=actual, E=estimate):")
        for r in rows:
            out.append(
                f"  {r['year']}: revenue {r['revenue']} (억원), OP {r['op_income']}, "
                f"net income {r['net_income']}, EPS {r['eps']}, PER {r['per']}, "
                f"PBR {r['pbr']}, ROE {r['roe']}%, EV/EBITDA {r['ev_ebitda']}"
            )
    return "\n".join(out)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Complete KR fundamentals: OpenDART audited actuals + wisereport forward
    estimates / consensus. Raises NoMarketDataError for non-KR (dispatcher
    falls through); NoMarketDataError when neither source yields data.
    """
    if not is_kr_ticker(ticker):
        raise NoMarketDataError(ticker, detail="wisereport only serves Korean tickers")
    code = to_krx_code(ticker)

    # Audited actuals + balance sheet from OpenDART (best-effort; needs DART_API_KEY).
    opendart_section = ""
    try:
        from .opendart_fundamentals import get_fundamentals as _opendart
        opendart_section = _opendart(ticker, curr_date)
    except Exception as exc:  # noqa: BLE001 — OpenDART is a bonus, not required
        logger.info("OpenDART section unavailable for %s, using wisereport only: %s", ticker, exc)

    wise_section = _wisereport_section(code)  # raises NoMarketDataError if empty

    parts = [p for p in (opendart_section, wise_section) if p]
    return "\n\n".join(parts)
