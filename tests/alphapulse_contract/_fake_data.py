"""Library-level fakes for the harness subprocess: data, network, clock, .env.

Everything here replaces a THIRD-PARTY or OS boundary, never fork code, so the
fork's vendor modules, router, validators and caches all run for real:

* ``yfinance``: ``Ticker`` / ``download`` / ``Search`` / ``Tickers`` are replaced
  on the ``yfinance`` module with offline fakes serving ``fixtures/instruments.json``
  plus a deterministic synthetic price path (``close_on``).
* ``requests``: ``HTTPAdapter.send`` answers from ``fixtures/http_routes.json``
  (plus the scenario's ``http_rules``); nothing reaches urllib3.
* ``urllib``: ``OpenerDirector.open`` answers from the same routes (StockTwits,
  Reddit use ``urlopen``).
* sockets and ``curl_cffi`` (which bypasses Python sockets): any attempt is
  refused and recorded in ``captures.network.blocked``.
* ``dotenv.find_dotenv``: fenced so a developer's ``.env`` above the checkout is
  never loaded; a scenario may supply its own ``.env`` text instead.
* ``time.sleep``: recorded and skipped (vendor pacing/backoff), unless the
  scenario sets ``real_sleep``.

Faults are injected at this same layer (``yfinance.faults``, ``http_rules`` with
``status``/``raise``) so failure paths also run through real vendor code.
"""

from __future__ import annotations

import datetime as _dt
import email.message
import fnmatch
import hashlib
import io
import json
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.response
from typing import Any
from zoneinfo import ZoneInfo

from . import _runtime as rt

_INSTRUMENTS: dict[str, Any] = {}
_ROUTES: list[dict[str, Any]] = []
_REAL_SLEEP = time.sleep
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_fault_lock = threading.Lock()
_fault_hits: dict[int, int] = {}
_route_hits: dict[int, int] = {}


class HarnessNetworkBlocked(OSError):
    """Raised for any real network attempt inside the harness subprocess."""


# ============================================================================ data

def load_fixtures() -> None:
    raw = json.loads((rt.FIXTURES / "instruments.json").read_text(encoding="utf-8"))
    _INSTRUMENTS.clear()
    _INSTRUMENTS.update({k: v for k, v in raw.items() if not k.startswith("_")})
    for sym, override in ((rt.SPEC.get("yfinance") or {}).get("info_overrides") or {}).items():
        entry = _INSTRUMENTS.setdefault(sym, {"base_price": 100.0, "price_decimals": 2,
                                              "volume_base": 100000,
                                              "exchange_tz": "America/New_York", "info": {}})
        entry["info"] = {**entry.get("info", {}), **override}
    routes = json.loads((rt.FIXTURES / "http_routes.json").read_text(encoding="utf-8"))["rules"]
    _ROUTES.clear()
    _ROUTES.extend(list(rt.SPEC.get("http_rules") or []) + routes)


def instrument(symbol: str) -> dict[str, Any] | None:
    return _INSTRUMENTS.get(str(symbol).upper()) or _INSTRUMENTS.get(str(symbol))


def close_on(inst: dict[str, Any], day: _dt.date) -> float:
    """Deterministic synthetic close: a two-wave path within about +-3% of base_price."""
    x = day.toordinal()
    wave = 0.018 * math.sin(2 * math.pi * x / 23.0) + 0.011 * math.sin(2 * math.pi * x / 61.0 + 1.3)
    return round(float(inst["base_price"]) * (1.0 + wave), int(inst.get("price_decimals", 2)))


def _exchange_today(inst: dict[str, Any]) -> _dt.date:
    tz = ZoneInfo(inst.get("exchange_tz") or "UTC")
    return _dt.datetime.now(tz).date()


def _business_days(start: _dt.date, end_exclusive: _dt.date) -> list[_dt.date]:
    out, d = [], start
    while d < end_exclusive:
        if d.weekday() < 5:
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


def _to_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value)
    try:
        return _dt.date.fromisoformat(text[:10])
    except ValueError:
        import pandas as pd
        return pd.Timestamp(text).date()


def _period_start(period: str | None, today: _dt.date) -> _dt.date:
    p = (period or "1mo").strip().lower()
    if p == "max":
        return today - _dt.timedelta(days=365 * 20)
    if p == "ytd":
        return _dt.date(today.year, 1, 1)
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", p)
    if not m:
        return today - _dt.timedelta(days=31)
    n, unit = int(m.group(1)), m.group(2)
    days = {"d": 1, "wk": 7, "mo": 31, "y": 366}[unit] * n
    return today - _dt.timedelta(days=days)


def bars(symbol: str, start: Any = None, end: Any = None, period: str | None = None):
    """[(date, open, high, low, close, volume)] for business days in [start, end)."""
    inst = instrument(symbol)
    if inst is None:
        return []
    today = _exchange_today(inst)
    s, e = _to_date(start), _to_date(end)
    if s is None:
        s = _period_start(period, today)
    if e is None:
        e = today + _dt.timedelta(days=1)
    e = min(e, today + _dt.timedelta(days=1))
    rows = []
    dec = int(inst.get("price_decimals", 2))
    for d in _business_days(s, e):
        close = close_on(inst, d)
        prev = d - _dt.timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= _dt.timedelta(days=1)
        opn = close_on(inst, prev)
        high = round(max(opn, close) * 1.004, dec)
        low = round(min(opn, close) * 0.996, dec)
        vol = int(float(inst.get("volume_base", 100000)) * (1 + 0.3 * math.sin(2 * math.pi * d.toordinal() / 17.0)))
        rows.append((d, opn, high, low, close, vol))
    return rows


# ------------------------------------------------------------------ fault rules

def _yf_fault(api: str, symbol: str) -> dict[str, Any] | None:
    faults = (rt.SPEC.get("yfinance") or {}).get("faults") or []
    for i, f in enumerate(faults):
        if not fnmatch.fnmatch(api, f.get("api", "*")):
            continue
        if not fnmatch.fnmatch(str(symbol), f.get("symbol", "*")):
            continue
        with _fault_lock:
            hits = _fault_hits.get(i, 0)
            times = f.get("times")
            if times is not None and hits >= int(times):
                continue
            _fault_hits[i] = hits + 1
        return f
    return None


def make_exception(name: str, message: str) -> BaseException:
    name = name or "RuntimeError"
    if name == "YFRateLimitError":
        from yfinance.exceptions import YFRateLimitError
        return YFRateLimitError()
    if name in ("ConnectionError", "HTTPError", "Timeout", "ReadTimeout", "ConnectTimeout"):
        import requests
        return getattr(requests.exceptions, name)(message)
    if name == "URLError":
        return urllib.error.URLError(message)
    if name == "CurlError":
        from curl_cffi import CurlError
        return CurlError(message)
    builtin = {"TimeoutError": TimeoutError, "OSError": OSError, "RuntimeError": RuntimeError,
               "ValueError": ValueError, "KeyError": KeyError}
    return builtin.get(name, RuntimeError)(message)


def _apply_yf_fault(api: str, symbol: str) -> dict[str, Any] | None:
    fault = _yf_fault(api, symbol)
    if fault is not None and fault.get("raise"):
        rt.record_data_call({"lib": "yfinance", "api": api, "symbol": symbol,
                             "fault": fault.get("raise")})
        raise make_exception(fault["raise"], fault.get("message") or f"harness fault on {api}")
    return fault


# ============================================================================ yfinance

def _frame(rows, *, tz: str | None, adjusted_columns: bool):
    import pandas as pd
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not rows:
        empty = pd.DataFrame(columns=cols)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="Date")
    if tz:
        idx = idx.tz_localize(tz)
    df = pd.DataFrame([r[1:] for r in rows], index=idx, columns=cols)
    df["Volume"] = df["Volume"].astype("int64")
    if adjusted_columns:
        df = df[["Close", "High", "Low", "Open", "Volume"]]
    return df


def _row_faults(rows: list, fault: dict[str, Any] | None) -> list:
    """Apply row-level faults.

    'empty' (no rows); 'drop_dates' (ISO dates with no bar, e.g. the trade date on a
    lagging feed); 'drop_last_bars' (the newest N bars of what was requested — for a
    download that is today's end of the 5-year window, so it models a lag on live runs).
    """
    if not fault:
        return rows
    if fault.get("empty"):
        return []
    missing = {str(d) for d in fault.get("drop_dates") or []}
    if missing:
        rows = [r for r in rows if r[0].isoformat() not in missing]
    drop = int(fault.get("drop_last_bars") or 0)
    return rows[:-drop] if drop > 0 else rows


def _nan_last_close(df):
    if len(df):
        df = df.copy()
        df["Close"] = df["Close"].astype("float64")
        df.iloc[-1, df.columns.get_loc("Close")] = float("nan")
    return df


def _statement(symbol: str, key: str):
    import pandas as pd
    inst = instrument(symbol) or {}
    st = inst.get("statements") or {}
    data = st.get(key)
    if not data:
        return pd.DataFrame()
    ends = st["quarterly_ends"] if key.startswith("quarterly_") else st["annual_ends"]
    cols = pd.to_datetime(ends[: len(next(iter(data.values())))])
    return pd.DataFrame({c: [vals[i] for vals in data.values()] for i, c in enumerate(cols)},
                        index=list(data.keys())).astype("float64")


def _news_nested(symbol: str) -> list[dict[str, Any]]:
    inst = instrument(symbol) or {}
    out = []
    for i, item in enumerate(inst.get("news") or []):
        when = rt.trade_date() - _dt.timedelta(days=int(item.get("days_before", 1)))
        out.append({
            "id": f"harness-news-{symbol}-{i}",
            "content": {
                "id": f"harness-news-{symbol}-{i}",
                "contentType": "STORY",
                "title": item["title"],
                "summary": item.get("summary", ""),
                "pubDate": f"{when.isoformat()}T01:00:00Z",
                "provider": {"displayName": item.get("publisher", "Harness Wire")},
                "canonicalUrl": {"url": f"https://finance.yahoo.com/news/harness-{i}.html"},
                "clickThroughUrl": {"url": f"https://finance.yahoo.com/news/harness-{i}.html"},
            },
        })
    return out


class FakeTicker:
    """Offline stand-in for ``yfinance.Ticker`` (the attributes the fork reads)."""

    _INSIDER_COLUMNS = ["Shares", "Value", "URL", "Text", "Insider", "Position",
                        "Transaction", "Start Date", "Ownership"]

    def __init__(self, ticker: str, session: Any = None, **_kw: Any) -> None:
        self.ticker = str(ticker).upper()

    def _log(self, api: str, **extra: Any) -> None:
        rt.record_data_call({"lib": "yfinance", "api": api, "symbol": self.ticker, **extra})

    # -- identity -----------------------------------------------------------
    @property
    def info(self) -> dict[str, Any]:
        fault = _apply_yf_fault("info", self.ticker)
        self._log("Ticker.info")
        if fault and fault.get("empty"):
            return {"trailingPegRatio": None}
        inst = instrument(self.ticker)
        if inst is None:
            return {"trailingPegRatio": None}
        return json.loads(json.dumps(inst.get("info") or {}))

    def get_info(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.info

    @property
    def fast_info(self) -> dict[str, Any]:
        info = self.info
        return {"currency": info.get("currency"), "exchange": info.get("exchange"),
                "last_price": info.get("regularMarketPrice"), "timezone": info.get("exchangeTimezoneName")}

    @property
    def isin(self) -> str:
        return "-"

    # -- prices ---------------------------------------------------------------
    def history(self, period: str | None = None, interval: str = "1d", start: Any = None,
                end: Any = None, **_kw: Any):
        fault = _apply_yf_fault("history", self.ticker)
        inst = instrument(self.ticker)
        rows = _row_faults(bars(self.ticker, start, end, period), fault)
        tz = (inst or {}).get("exchange_tz")
        df = _frame(rows, tz=tz, adjusted_columns=False)
        if len(df):
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
        if fault and fault.get("nan_last_close"):
            df = _nan_last_close(df)
        self._log("Ticker.history", start=str(start) if start is not None else None,
                  end=str(end) if end is not None else None, period=period, rows=len(df),
                  fault=(fault or {}).get("name"))
        return df

    def get_history_metadata(self, *a: Any, **k: Any) -> dict[str, Any]:
        info = self.info
        return {"currency": info.get("currency"), "exchangeTimezoneName": info.get("exchangeTimezoneName")}

    # -- statements -----------------------------------------------------------
    def _stmt(self, key: str):
        _apply_yf_fault("statements", self.ticker)
        df = _statement(self.ticker, key)
        self._log(f"Ticker.{key}", rows=len(df))
        return df

    balance_sheet = property(lambda self: self._stmt("balance_sheet"))
    balancesheet = balance_sheet
    quarterly_balance_sheet = property(lambda self: self._stmt("quarterly_balance_sheet"))
    quarterly_balancesheet = quarterly_balance_sheet
    cashflow = property(lambda self: self._stmt("cashflow"))
    quarterly_cashflow = property(lambda self: self._stmt("quarterly_cashflow"))
    income_stmt = property(lambda self: self._stmt("income_stmt"))
    quarterly_income_stmt = property(lambda self: self._stmt("quarterly_income_stmt"))
    financials = income_stmt
    quarterly_financials = quarterly_income_stmt

    def _by_freq(self, stem: str, freq: str = "yearly", **_kw: Any):
        return self._stmt(f"quarterly_{stem}" if str(freq).startswith("q") else stem)

    def get_balance_sheet(self, as_dict: bool = False, pretty: bool = False, freq: str = "yearly"):
        return self._by_freq("balance_sheet", freq)

    get_balancesheet = get_balance_sheet

    def get_cashflow(self, as_dict: bool = False, pretty: bool = False, freq: str = "yearly"):
        return self._by_freq("cashflow", freq)

    def get_income_stmt(self, as_dict: bool = False, pretty: bool = False, freq: str = "yearly"):
        return self._by_freq("income_stmt", freq)

    get_financials = get_income_stmt

    @property
    def insider_transactions(self):
        import pandas as pd
        _apply_yf_fault("insider_transactions", self.ticker)
        rows = (instrument(self.ticker) or {}).get("insider_transactions") or []
        df = pd.DataFrame(rows, columns=self._INSIDER_COLUMNS) if rows else pd.DataFrame(columns=self._INSIDER_COLUMNS)
        self._log("Ticker.insider_transactions", rows=len(df))
        return df

    def get_insider_transactions(self, *a: Any, **k: Any):
        return self.insider_transactions

    # -- news -----------------------------------------------------------------
    @property
    def news(self) -> list[dict[str, Any]]:
        return self.get_news()

    def get_news(self, count: int = 10, tab: str = "news") -> list[dict[str, Any]]:
        fault = _apply_yf_fault("get_news", self.ticker)
        items = [] if (fault and fault.get("empty")) else _news_nested(self.ticker)[: int(count)]
        self._log("Ticker.get_news", count=count, items=len(items))
        return items

    # -- rarely used, modelled as empty ----------------------------------------
    def _empty(self, api: str):
        import pandas as pd
        self._log(f"Ticker.{api}")
        return pd.DataFrame()

    dividends = property(lambda self: self._empty("dividends"))
    splits = property(lambda self: self._empty("splits"))
    actions = property(lambda self: self._empty("actions"))
    recommendations = property(lambda self: self._empty("recommendations"))
    major_holders = property(lambda self: self._empty("major_holders"))
    institutional_holders = property(lambda self: self._empty("institutional_holders"))
    earnings_dates = property(lambda self: self._empty("earnings_dates"))

    @property
    def calendar(self) -> dict[str, Any]:
        self._log("Ticker.calendar")
        return {}

    @property
    def options(self) -> tuple:
        return ()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        rt.note(f"yfinance.Ticker.{name} is not modelled by the harness (called for {self.ticker})")
        rt.record_data_call({"lib": "yfinance", "api": f"Ticker.{name}", "symbol": self.ticker,
                             "unmodelled": True})
        raise AttributeError(f"harness: yfinance.Ticker.{name} is not modelled")


class FakeTickers:
    def __init__(self, tickers: Any, session: Any = None) -> None:
        names = tickers.split() if isinstance(tickers, str) else list(tickers)
        self.symbols = [str(t).upper() for t in names]
        self.tickers = {s: FakeTicker(s) for s in self.symbols}


def fake_download(tickers: Any, start: Any = None, end: Any = None, actions: bool = False,
                  threads: bool = True, ignore_tz: Any = None, group_by: str = "column",
                  auto_adjust: bool = True, back_adjust: bool = False, repair: bool = False,
                  keepna: bool = False, progress: bool = True, period: str | None = None,
                  interval: str = "1d", prepost: bool = False, rounding: bool = False,
                  timeout: int = 10, session: Any = None, multi_level_index: bool = True,
                  **_kw: Any):
    import pandas as pd
    names = tickers.split() if isinstance(tickers, str) else list(tickers)
    names = [str(n).upper() for n in names]
    frames = {}
    for sym in names:
        fault = _apply_yf_fault("download", sym)
        rows = _row_faults(bars(sym, start, end, period), fault)
        df = _frame(rows, tz=None, adjusted_columns=bool(auto_adjust))
        if fault and fault.get("nan_last_close"):
            df = _nan_last_close(df)
        rt.record_data_call({"lib": "yfinance", "api": "download", "symbol": sym,
                             "start": str(start) if start is not None else None,
                             "end": str(end) if end is not None else None,
                             "period": period, "rows": len(df),
                             "fault": (fault or {}).get("name")})
        frames[sym] = df
    if len(names) == 1 and not multi_level_index:
        return frames[names[0]]
    parts = []
    for sym, df in frames.items():
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples([(c, sym) for c in df.columns], names=["Price", "Ticker"])
        parts.append(df)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


class FakeSearch:
    """Offline stand-in for ``yfinance.Search`` (news + quotes)."""

    def __init__(self, query: str, max_results: int = 8, news_count: int = 8,
                 lists_count: int = 8, include_cb: bool = True, include_nav_links: bool = False,
                 include_research: bool = False, include_cultural_assets: bool = False,
                 enable_fuzzy_query: bool = False, recommended: int = 8, session: Any = None,
                 timeout: int = 30, raise_errors: bool = True, **_kw: Any) -> None:
        self.query = str(query)
        fault = _apply_yf_fault("Search", self.query)
        empty = bool(fault and fault.get("empty"))
        q = self.query.lower()
        self.quotes = [] if empty else [
            {"symbol": sym, "shortname": inst["info"].get("shortName"),
             "longname": inst["info"].get("longName"), "exchange": inst["info"].get("exchange"),
             "quoteType": inst["info"].get("quoteType")}
            for sym, inst in _INSTRUMENTS.items()
            if q and (q in sym.lower() or q in str(inst["info"].get("longName", "")).lower())
        ][: int(max_results)]
        self.news = [] if empty else self._news(int(news_count))
        self.lists, self.research, self.nav = [], [], []
        self.response = {"quotes": self.quotes, "news": self.news}
        self.all = {"quotes": self.quotes, "news": self.news, "lists": [], "research": [], "nav": []}
        rt.record_data_call({"lib": "yfinance", "api": "Search", "query": self.query,
                             "news": len(self.news), "quotes": len(self.quotes),
                             "fault": (fault or {}).get("name")})

    def _news(self, count: int) -> list[dict[str, Any]]:
        out = []
        # hashlib, not hash(): str hashes are salted per process (PYTHONHASHSEED), and
        # two runs of one scenario must see identical data.
        query_id = hashlib.sha256(self.query.encode("utf-8")).hexdigest()[:8]
        for i, days in enumerate((1, 2, 4)[: max(0, count)]):
            when = _dt.datetime.combine(rt.trade_date() - _dt.timedelta(days=days),
                                        _dt.time(0, 30), tzinfo=_dt.timezone.utc)
            out.append({
                "uuid": f"harness-search-{query_id}-{i}",
                "title": f"{self.query} — market brief {i + 1}",
                "publisher": ("Reuters", "Bloomberg", "Yonhap")[i % 3],
                "link": f"https://finance.yahoo.com/news/harness-search-{i}.html",
                "providerPublishTime": int(when.timestamp()),
                "type": "STORY",
            })
        return out

    def search(self) -> FakeSearch:
        return self


def install_yfinance() -> None:
    import yfinance
    for mod in (yfinance,):
        mod.Ticker = FakeTicker
        mod.Tickers = FakeTickers
        mod.download = fake_download
        mod.Search = FakeSearch
    # Submodule bindings used by some callers (``from yfinance.ticker import Ticker``).
    for dotted, attr, fake in (("yfinance.ticker", "Ticker", FakeTicker),
                               ("yfinance.tickers", "Tickers", FakeTickers),
                               ("yfinance.search", "Search", FakeSearch),
                               ("yfinance.multi", "download", fake_download)):
        try:
            import importlib
            sub = importlib.import_module(dotted)
            setattr(sub, attr, fake)
        except Exception:  # noqa: BLE001
            pass


# ============================================================================ HTTP

def _render_tokens(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            base, sign, n, fmt = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            delta = _dt.timedelta(days=n if sign == "+" else -n)
            if base == "D":
                moment = _dt.datetime.combine(rt.trade_date(), _dt.time(1, 0),
                                              tzinfo=_dt.timezone.utc) + delta
            else:
                moment = _dt.datetime.now(_dt.timezone.utc) + delta
            if fmt == "iso":
                return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
            if fmt == "epoch":
                return str(int(moment.timestamp()))
            return moment.strftime(fmt)
        return re.sub(r"\{\{(D|NOW)([+-])(\d+)\|([^}]+)\}\}", _sub, value)
    if isinstance(value, list):
        return [_render_tokens(v) for v in value]
    if isinstance(value, dict):
        return {k: _render_tokens(v) for k, v in value.items()}
    return value


def _alpha_vantage_daily_csv(url: str) -> str:
    from urllib.parse import parse_qs, urlsplit
    qs = parse_qs(urlsplit(url).query)
    symbol = (qs.get("symbol") or [""])[0]
    compact = (qs.get("outputsize") or ["compact"])[0] == "compact"
    rows = bars(symbol, period="max")
    rows = rows[-100:] if compact else rows
    extra = (rt.SPEC.get("alpha_vantage") or {}).get("extra_rows") or []
    lines = ["timestamp,open,high,low,close,volume"]
    table = {r[0].isoformat(): r for r in rows}
    for e in extra:
        table[e["date"]] = (_dt.date.fromisoformat(e["date"]), e["close"], e["close"], e["close"],
                            e["close"], 1000)
    for key in sorted(table, reverse=True):
        d, o, h, lo, c, v = table[key]
        lines.append(f"{key},{o},{h},{lo},{c},{v}")
    return "\n".join(lines) + "\n"


_GENERATORS = {"alpha_vantage_daily_csv": _alpha_vantage_daily_csv}


def route(url: str, method: str = "GET") -> dict[str, Any]:
    """Resolve a URL to {'status','body','headers','raise','rule'}; records the call.

    A rule with ``times: N`` answers only its first N matching requests; later ones
    fall through to the next matching rule (e.g. one 429, then the fixture).
    """
    for index, rule in enumerate(_ROUTES):
        needles = rule.get("match") or []
        if isinstance(needles, str):
            needles = [needles]
        if all(n in url for n in needles):
            if rule.get("times") is not None:
                with _fault_lock:
                    hits = _route_hits.get(index, 0)
                    if hits >= int(rule["times"]):
                        continue
                    _route_hits[index] = hits + 1
            name = rule.get("name") or "+".join(needles)
            if rule.get("raise"):
                rt.record_data_call({"lib": "http", "method": method, "url": url, "rule": name,
                                     "fault": rule["raise"]})
                return {"raise": rule["raise"], "message": rule.get("message") or f"harness: {name}"}
            if "generator" in rule:
                body = _GENERATORS[rule["generator"]](url).encode("utf-8")
                ctype = "text/csv"
            elif "json" in rule:
                body = json.dumps(_render_tokens(rule["json"]), ensure_ascii=False).encode("utf-8")
                ctype = "application/json"
            else:
                body = str(_render_tokens(rule.get("text", ""))).encode("utf-8")
                ctype = "text/html; charset=utf-8"
            headers = {"Content-Type": ctype, **(rule.get("headers") or {})}
            status = int(rule.get("status", 200))
            rt.record_data_call({"lib": "http", "method": method, "url": url, "rule": name,
                                 "status": status})
            return {"status": status, "body": body, "headers": headers, "rule": name}
    with rt._lock:
        rt.CAPTURES["network"]["unrouted"].append({"method": method, "url": url,
                                                   "stack": rt.short_stack()})
    rt.record_data_call({"lib": "http", "method": method, "url": url, "rule": None, "status": 404})
    return {"status": 404, "body": b"harness: no route for this URL",
            "headers": {"Content-Type": "text/plain"}, "rule": None}


_REASONS = {200: "OK", 403: "Forbidden", 404: "Not Found", 429: "Too Many Requests",
            500: "Internal Server Error", 503: "Service Unavailable"}


def install_requests() -> None:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.structures import CaseInsensitiveDict

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        res = route(request.url, request.method or "GET")
        if res.get("raise"):
            exc = make_exception(res["raise"], res["message"])
            if isinstance(exc, requests.exceptions.RequestException):
                exc.request = request
            raise exc
        resp = requests.models.Response()
        resp.status_code = res["status"]
        resp._content = res["body"]
        resp._content_consumed = True
        resp.headers = CaseInsensitiveDict(res["headers"])
        resp.url = request.url
        resp.request = request
        resp.encoding = "utf-8"
        resp.reason = _REASONS.get(res["status"], "Harness")
        resp.connection = self
        return resp

    HTTPAdapter.send = send


def install_urllib() -> None:
    def open_(self, fullurl, data=None, timeout=None):
        url = fullurl if isinstance(fullurl, str) else fullurl.full_url
        method = "GET" if isinstance(fullurl, str) else fullurl.get_method()
        res = route(url, method)
        if res.get("raise"):
            raise make_exception(res["raise"], res["message"])
        msg = email.message.Message()
        for k, v in res["headers"].items():
            msg[k] = v
        if res["status"] >= 400:
            raise urllib.error.HTTPError(url, res["status"], _REASONS.get(res["status"], "Harness"),
                                         msg, io.BytesIO(res["body"]))
        return urllib.response.addinfourl(io.BytesIO(res["body"]), msg, url, res["status"])

    urllib.request.OpenerDirector.open = open_


# ============================================================================ guards

def _blocked(api: str, target: Any) -> HarnessNetworkBlocked:
    with rt._lock:
        rt.CAPTURES["network"]["blocked"].append({"api": api, "target": repr(target)[:200],
                                                  "stack": rt.short_stack()})
    return HarnessNetworkBlocked(f"harness: network access blocked ({api} {target!r})")


def install_socket_guard() -> None:
    def connect(self, address):
        if getattr(self, "family", None) == getattr(socket, "AF_UNIX", object()):
            return _REAL_CONNECT(self, address)
        raise _blocked("socket.connect", address)

    def connect_ex(self, address):
        if getattr(self, "family", None) == getattr(socket, "AF_UNIX", object()):
            return _REAL_CONNECT_EX(self, address)
        raise _blocked("socket.connect_ex", address)

    def create_connection(address, *a, **k):
        raise _blocked("socket.create_connection", address)

    def getaddrinfo(host, *a, **k):
        raise _blocked("socket.getaddrinfo", host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo


def install_curl_guard() -> None:
    """curl_cffi talks to libcurl directly (no Python socket), so block it at its API."""
    try:
        import curl_cffi
        from curl_cffi import requests as cr
    except Exception:  # noqa: BLE001 — not installed: nothing to guard
        return

    def perform(self, *a, **k):
        raise _blocked("curl_cffi.Curl.perform", getattr(self, "_url", "?"))

    def request(self, method, url, *a, **k):
        raise _blocked("curl_cffi.requests.Session.request", url)

    async def arequest(self, method, url, *a, **k):
        raise _blocked("curl_cffi.requests.AsyncSession.request", url)

    curl_cffi.Curl.perform = perform
    cr.Session.request = request
    cr.AsyncSession.request = arequest


def install_dotenv_fence() -> None:
    """Serve only the scenario's .env (if any); never a developer's real one.

    tradingagents/__init__.py calls ``load_dotenv(find_dotenv(usecwd=True))``,
    which walks up from the repo root — in a nested worktree that reaches the
    main checkout's .env with real keys. alpha-pulse runs a clone with no .env above
    it, so '' is the faithful answer.
    """
    try:
        import dotenv
        import dotenv.main
    except Exception:  # noqa: BLE001
        return
    real = dotenv.main.find_dotenv
    provided = rt.RUN_DIR / "scenario.env" if rt.RUN_DIR else None

    def find_dotenv(filename: str = ".env", raise_error_if_not_found: bool = False,
                    usecwd: bool = False) -> str:
        try:
            would = real(filename, raise_error_if_not_found=False, usecwd=usecwd)
        except Exception as exc:  # noqa: BLE001
            would = f"<error {exc}>"
        served = str(provided) if (filename == ".env" and provided and provided.exists()) else ""
        with rt._lock:
            rt.CAPTURES["dotenv"].append({"filename": filename, "usecwd": usecwd,
                                          "served": served, "fenced_off": would})
        return served

    dotenv.find_dotenv = find_dotenv
    dotenv.main.find_dotenv = find_dotenv


def install_sleep_recorder() -> None:
    if rt.SPEC.get("real_sleep"):
        return

    def sleep(seconds: float) -> None:
        with rt._lock:
            if len(rt.CAPTURES["sleeps"]) < 500:
                rt.CAPTURES["sleeps"].append({"seconds": seconds, "at": (rt.short_stack(2) or ["?"])[-1]})
        _REAL_SLEEP(0)

    time.sleep = sleep


def install_all() -> None:
    load_fixtures()
    install_socket_guard()
    install_curl_guard()
    install_requests()
    install_urllib()
    install_sleep_recorder()
    install_dotenv_fence()
    install_yfinance()
