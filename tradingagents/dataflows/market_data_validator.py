"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct_diff(a: float, b: float) -> float:
    return abs(a - b) / abs(b) * 100.0 if b else 0.0


def _latest_price_crosscheck(symbol: str, curr_date: str, yf_date: str, yf_close) -> str:
    """Best-effort latest-close cross-check against Alpha Vantage.

    Returns a markdown section, or '' when unavailable (cross-check disabled, no
    API key, rate-limited, network/parse error, or the verified close is not
    numeric). NEVER raises — verification must not be blocked by a second vendor.
    Look-ahead safety is delegated to the Alpha Vantage fetch, which only returns
    closes on or before ``curr_date``.
    """
    from tradingagents.dataflows.config import get_config

    if not get_config().get("enable_alpha_vantage_price_crosscheck", True):
        return ""
    try:
        yf_close_f = float(yf_close)
    except (TypeError, ValueError):
        return ""
    try:
        from tradingagents.dataflows.alpha_vantage_stock import (
            get_latest_close_on_or_before,
        )
        result = get_latest_close_on_or_before(symbol, curr_date)
    except Exception:  # noqa: BLE001 — fail open; the cross-check is best-effort
        return ""
    if not result:
        return ""
    av_date, av_close = result

    header = ["", "### Latest-price cross-check (Alpha Vantage)", ""]
    if av_date > yf_date:
        body = [
            "⚠️ The primary feed's latest close may be STALE — Alpha Vantage has a "
            "more recent close on or before the analysis date:",
            "",
            "| Source | Latest date | Close |",
            "|---|---|---:|",
            f"| yfinance (primary) | {yf_date} | {yf_close_f:.2f} |",
            f"| Alpha Vantage (cross-check) | {av_date} | {av_close:.2f} |",
            "",
            f"Treat **{av_close:.2f} ({av_date})** as the most recent close; the "
            "primary feed appears to be lagging the latest session.",
        ]
    elif av_date < yf_date:
        body = [
            f"Note: Alpha Vantage's latest available close ({av_date}, "
            f"{av_close:.2f}) is older than the primary feed ({yf_date}, "
            f"{yf_close_f:.2f}); using the primary feed.",
        ]
    else:
        pct = _pct_diff(av_close, yf_close_f)
        if pct <= 0.5:
            body = [
                f"✓ Latest close confirmed by Alpha Vantage: {yf_close_f:.2f} on "
                f"{yf_date} (Δ {pct:.2f}%)."
            ]
        else:
            body = [
                f"⚠️ Vendor discrepancy on {yf_date}: yfinance {yf_close_f:.2f} vs "
                f"Alpha Vantage {av_close:.2f} (Δ {pct:.2f}%). Verify before relying "
                "on the exact level.",
            ]
    return "\n".join(header + body)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    crosscheck = _latest_price_crosscheck(symbol, curr_date, latest_date, latest.get("Close"))
    if crosscheck:
        lines.append(crosscheck)

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
