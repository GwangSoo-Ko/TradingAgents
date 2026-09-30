import math
from typing import Annotated

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import withhold_live_profile
from tradingagents.dataflows.errors import NoMarketDataError, VendorError, VendorRateLimitError
from tradingagents.dataflows.net import vendor_reachable
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import (
    YAHOO_HOST,
    raise_for_empty,
    yf_retry,
)


def _latest_reported(row) -> float | None:
    """A balance-sheet row's most recent finite figure (columns run newest first).

    Yahoo leaves a quarter's cells blank (NaN) until it fills them in, and NaN is
    truthy, so reading the first column blindly derived "Book Value: nan".
    """
    for value in row:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None
):
    """Get company fundamentals overview from yfinance.

    ``Ticker.info`` is a present-day snapshot with no historical vintage, so a
    past ``curr_date`` withholds it through the shared point-in-time guard
    (``date_window.withhold_live_profile``, #1300).
    """
    canonical = normalize_symbol(ticker)

    # Guard before the request: the response would only be discarded, and the
    # answer does not depend on it.
    withheld = withhold_live_profile(curr_date, canonical)
    if withheld:
        return withheld

    try:
        ticker_obj = yf.Ticker(canonical)
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            raise_for_empty(ticker, canonical, "fundamentals")

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        # Yahoo omits trailing PE/EPS/Price-to-Book/Book Value for many Korean
        # (.KS/.KQ) listings even when the inputs exist, so the fundamentals
        # analyst loses every valuation anchor. Derive them from data already on
        # hand (and the balance sheet for book value). Derived values are tagged
        # "(derived)" so downstream agents don't treat them as vendor-authoritative.
        derived: dict[str, float] = {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares = info.get("sharesOutstanding")
        net_income = info.get("netIncomeToCommon")

        eps = info.get("trailingEps")
        if eps is None and net_income and shares:
            eps = net_income / shares
            derived["EPS (TTM)"] = round(eps, 2)
        if info.get("trailingPE") is None and price and eps and eps > 0:
            derived["PE Ratio (TTM)"] = round(price / eps, 2)
        if info.get("priceToBook") is None or info.get("bookValue") is None:
            # Book value per share isn't in info for KR tickers; pull equity
            # from the balance sheet. Only fetched when info lacks PB/bookValue,
            # so US tickers (which usually have them) skip this extra call. Uses
            # its own Ticker handle so it does not depend on how `info` was read.
            try:
                bs = yf_retry(lambda: yf.Ticker(canonical).quarterly_balance_sheet)
                equity = None
                for row in ("Stockholders Equity", "Common Stock Equity"):
                    if row in bs.index:
                        equity = _latest_reported(bs.loc[row])
                        if equity is not None:
                            break
                bvps_shares = shares
                if "Ordinary Shares Number" in bs.index:
                    bvps_shares = _latest_reported(bs.loc["Ordinary Shares Number"]) or shares
                if equity and bvps_shares:
                    bvps = equity / bvps_shares
                    if info.get("bookValue") is None:
                        derived["Book Value"] = round(bvps, 2)
                    if info.get("priceToBook") is None and price and bvps > 0:
                        derived["Price to Book"] = round(price / bvps, 2)
            except Exception:  # noqa: BLE001 — balance sheet unavailable -> "N/A (vendor)"
                pass

        # Labels that anchor valuation; surface them explicitly (vendor value ->
        # derived -> "N/A (vendor)") instead of silently dropping a None, so the
        # agent knows the figure is unavailable rather than overlooked.
        _critical = {"PE Ratio (TTM)", "EPS (TTM)", "Price to Book", "Book Value"}
        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")
            elif label in derived:
                lines.append(f"{label}: {derived[label]} (derived)")
            elif label in _critical:
                lines.append(f"{label}: N/A (vendor)")

        # yfinance returns a stub dict (e.g. {"trailingPegRatio": None}) for
        # unknown symbols, so `info` is truthy but every field is empty. Treat
        # "no usable fields" as no data rather than emitting a bare header the
        # agent might fabricate around. (The N/A criticals alone don't count.)
        if not any(not ln.endswith("N/A (vendor)") for ln in lines):
            raise NoMarketDataError(ticker, canonical, "no fundamental fields returned")

        header = f"# Company Fundamentals for {canonical}\n"
        currency = info.get("financialCurrency") or info.get("currency")
        if currency:
            header += f"# Reporting currency: {currency}\n"
        header += "\n"

        return header + "\n".join(lines)

    except VendorError:
        raise
    except Exception as e:
        raise NoMarketDataError(ticker, canonical, f"fundamentals unavailable: {e}") from e


# This vendor dates a statement by the period it covers, not by the day it was
# filed, and carries no filing date to do better. A company files weeks after its
# period ends, so a run dated in that gap can be served figures that were not yet
# public. Say so rather than implying the stricter guarantee (SEC EDGAR, which
# does carry filing dates, serves US filers as filed).
_PERIOD_END_VINTAGE = (
    "# Periods are cut at the fiscal period end; this vendor does not report "
    "filing dates, so the most recent period may not have been published yet.\n\n"
)


def _statement(ticker, freq, curr_date, title, quarterly_attr, annual_attr) -> str:
    """One financial statement as CSV, cut at ``curr_date`` by period end."""
    canonical = normalize_symbol(ticker)
    what = title.lower()
    try:
        ticker_obj = yf.Ticker(canonical)
        attr = quarterly_attr if freq.lower() == "quarterly" else annual_attr
        data = filter_financials_by_date(yf_retry(lambda: getattr(ticker_obj, attr)), curr_date)
        if data.empty:
            raise_for_empty(ticker, canonical, f"{what} data")
        return f"# {title} data for {canonical} ({freq})\n" + _PERIOD_END_VINTAGE + data.to_csv()
    except VendorError:
        raise
    except Exception as e:
        raise NoMarketDataError(ticker, canonical, f"{what} unavailable: {e}") from e


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    return _statement(ticker, freq, curr_date, "Balance Sheet", "quarterly_balance_sheet", "balance_sheet")


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    return _statement(ticker, freq, curr_date, "Cash Flow", "quarterly_cashflow", "cashflow")


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    return _statement(ticker, freq, curr_date, "Income Statement", "quarterly_income_stmt", "income_stmt")


# Rows are dated by the transaction, which is when the insider traded, not when
# the market learned of it: a Form 4 is filed up to two business days later and
# this vendor reports no filing date, so the most recent rows may not have been
# public on the analysis date.
_TRANSACTION_DATE_VINTAGE = (
    "# Rows are dated by transaction date. A trade becomes public when its Form 4 "
    "is filed, up to two business days later, so the newest rows may not have been "
    "known on this date.\n\n"
)


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str | None, "only transactions on or before this date, yyyy-mm-dd"] = None,
):
    """Get insider transactions data from yfinance."""
    canonical = normalize_symbol(ticker)
    try:
        ticker_obj = yf.Ticker(canonical)
        data = yf_retry(lambda: ticker_obj.insider_transactions)

        # Empty is normal here (many valid symbols have no insider filings),
        # so report it plainly rather than treating the symbol as invalid.
        if data is None or data.empty:
            if not vendor_reachable(YAHOO_HOST):
                raise VendorRateLimitError("Yahoo Finance is unreachable; insider filings were not retrieved")
            return f"No insider transactions reported for symbol '{canonical}'"

        if curr_date:
            traded = data["Start Date"]
            kept = data[traded <= pd.Timestamp(curr_date)]
            if kept.empty:
                return (
                    f"<insider transactions unavailable for {canonical} as of {curr_date}: "
                    "Yahoo serves recent transactions only>"
                )
            data = kept

        return f"# Insider Transactions data for {canonical}\n" + _TRANSACTION_DATE_VINTAGE + data.to_csv()

    except VendorError:
        raise
    except Exception as e:
        raise NoMarketDataError(ticker, canonical, f"insider transactions unavailable: {e}") from e


def get_company_profile(ticker: str) -> dict:
    """Yahoo's current profile for ``ticker``: name, sector, industry and the like."""
    canonical = normalize_symbol(ticker)
    try:
        return yf_retry(lambda: yf.Ticker(canonical).info) or {}
    except Exception as e:
        raise NoMarketDataError(ticker, canonical, f"profile unavailable: {e}") from e


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]
