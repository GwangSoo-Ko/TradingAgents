from datetime import datetime

from .alpha_vantage_common import _filter_csv_by_date_range, _make_api_request


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values, adjusted close values, and historical split/dividend events
    filtered to the specified date range.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily adjusted time series data filtered to the date range.
    """
    # Parse dates to determine the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 days
    # Compact returns latest 100 data points, so check if start_date is recent enough
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)


def get_latest_close_on_or_before(symbol: str, on_or_before: str):
    """Latest daily close on or before ``on_or_before`` from Alpha Vantage.

    Uses TIME_SERIES_DAILY (compact, free tier) and filters out rows after
    ``on_or_before``, so it is look-ahead-safe for historical analysis dates too.
    Returns ``(date_str 'YYYY-MM-DD', close float)`` or ``None`` when no usable
    row is available. Raises ``AlphaVantageNotConfiguredError`` when no API key is
    set (callers treat that as "cross-check unavailable").
    """
    from io import StringIO

    import pandas as pd

    params = {"symbol": symbol, "outputsize": "compact", "datatype": "csv"}
    csv_data = _make_api_request("TIME_SERIES_DAILY", params)
    if not csv_data or not str(csv_data).strip():
        return None
    df = pd.read_csv(StringIO(csv_data))
    if df.empty or "close" not in df.columns:
        return None
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[df[date_col] <= pd.to_datetime(on_or_before)].sort_values(date_col)
    if df.empty:
        return None
    last = df.iloc[-1]
    return last[date_col].strftime("%Y-%m-%d"), float(last["close"])
