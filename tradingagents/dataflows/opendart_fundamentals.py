"""OpenDART fundamentals vendor — audited KR financial statements.

yfinance drops trailing valuation fields and gives shallow statements for many
.KS/.KQ listings; OpenDART provides the FSS-audited annual figures. This vendor
returns the key balance-sheet and income-statement headline figures for the
most recent annual report available as of ``curr_date`` (look-ahead safe).

Opt-in: registered for ``get_fundamentals`` but not a default; enable via
fundamental_data="opendart,yfinance" (or tool_vendors["get_fundamentals"]).
"""

from __future__ import annotations

import logging
from datetime import datetime

from .kr_utils import is_kr_ticker
from .opendart_common import corp_code_for, dart_get
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

# Annual report (사업보고서). KR annual reports are filed by end of March of the
# following year, so a fiscal year is only safely "available" after that.
_ANNUAL = "11011"

# account_nm -> (English label, statement division) for the headline figures.
_WANTED = {
    "자산총계": ("Total Assets", "BS"),
    "부채총계": ("Total Liabilities", "BS"),
    "자본총계": ("Total Equity", "BS"),
    "매출액": ("Revenue", "IS"),
    "영업이익": ("Operating Income", "IS"),
    "당기순이익": ("Net Income", "IS"),
}


def _latest_available_fiscal_year(curr_date: str | None) -> int:
    """Most recent fiscal year whose annual report is filed by ``curr_date``.

    Annual reports land by ~end of March; before April we can't yet rely on the
    prior year's filing, so step back one more year. Without curr_date (live
    use) assume the prior fiscal year is available.
    """
    if not curr_date:
        return datetime.now().year - 1
    d = datetime.strptime(curr_date, "%Y-%m-%d")
    return d.year - 1 if d.month >= 4 else d.year - 2


def _fmt(amount: str) -> str:
    """OpenDART amounts are comma-formatted strings; keep them readable."""
    return (amount or "").strip() or "N/A"


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Audited KR fundamentals (key BS/IS figures) for a Korean ticker.

    Raises NoMarketDataError for non-KR tickers (dispatcher falls through) and
    NoMarketDataError when OpenDART has no filing (so a 'opendart,yfinance'
    chain falls back to yfinance).
    """
    if not is_kr_ticker(ticker):
        raise NoMarketDataError(ticker, detail="OpenDART only serves Korean tickers")
    corp = corp_code_for(ticker)  # raises NoMarketDataError if unmapped
    year = _latest_available_fiscal_year(curr_date)

    # Try consolidated (CFS) first, then separate (OFS); step back a year once
    # if the chosen year has no filing yet.
    data = None
    for attempt_year in (year, year - 1):
        for fs_div in ("CFS", "OFS"):
            try:
                data = dart_get(
                    "fnlttSinglAcntAll.json",
                    corp_code=corp, bsns_year=str(attempt_year),
                    reprt_code=_ANNUAL, fs_div=fs_div,
                )
                year = attempt_year
                break
            except NoMarketDataError:
                data = None
                continue
        if data is not None:
            break
    if data is None:
        raise NoMarketDataError(ticker, corp, "no OpenDART annual filing found")

    items = data.get("list") or []
    figures = {}
    currency = None
    for it in items:
        nm = it.get("account_nm")
        if nm in _WANTED and _WANTED[nm][1] == it.get("sj_div"):
            label = _WANTED[nm][0]
            if label not in figures:  # first (consolidated total) wins
                figures[label] = (nm, _fmt(it.get("thstrm_amount")))
                currency = currency or it.get("currency")

    if not figures:
        raise NoMarketDataError(ticker, corp, "OpenDART filing had no headline figures")

    lines = [f"{label} ({nm}): {amt}" for label, (nm, amt) in figures.items()]
    header = (
        f"# Company Fundamentals (OpenDART) for {ticker}\n"
        f"# Source: OpenDART 사업보고서 (annual report) FY{year}, corp_code {corp}\n"
    )
    if currency:
        header += f"# Reporting currency: {currency}\n"
    header += "# Audited figures from the FSS electronic disclosure system.\n\n"
    return header + "\n".join(lines)
