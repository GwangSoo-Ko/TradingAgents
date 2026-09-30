"""alpha-pulse data-layer contract: which vendor answers which tool, and what the market
analyst's verification tools hand the model.

alpha-pulse runs ``main.py`` for its KR holdings and candidates every night (KST
``YYYYMMDD`` date) and for KR or US names from the web and discovery (ISO date).
``main.build_config()`` puts the fork's KR-only vendors first in two chains -- news
``naver,yfinance`` and fundamentals ``wisereport,yfinance`` -- and switches on the Naver
discussion board; for a US symbol the KR vendors must step aside so Yahoo serves it
unchanged. These tests watch that through the real main.py (``harness.run_main``) with
data faked only at the library layer (yfinance / requests / urllib, ``_fake_data.py``), so
the fork's vendor code, router, retries and caches all run for real. Expected values are
written out from the reviewed fixtures (``fixtures/instruments.json``,
``fixtures/http_routes.json``); nothing is recomputed by the code under test.

The runs use today's KST date -- what the nightly batch and discovery send. Upstream
v0.5.x withholds Yahoo's live company profile for past dates (an accepted drift), and a
routing lock must not go red for that reason.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
import requests.adapters

from . import scenarios
from ._compat import get_symbol, import_first
from .harness import REPO_ROOT, RunResult, kst_today, run_main

TODAY = kst_today()
KR_TICKER = "417310.KS"  # 하나글로벌리츠, the KR holding of scenario s1
US_TICKER = "AAPL"
KR_ARGV = [KR_TICKER, TODAY.strftime("%Y%m%d")]  # nightly batch form (KST YYYYMMDD)
US_ARGV = [US_TICKER, TODAY.isoformat()]  # web / discovery form
KR_HOSTS = ("m.stock.naver.com", "navercomp.wisereport.co.kr", "opendart.fss.or.kr")

# Where the fork keeps its KR-only vendors today, then where a merge could move them.
KR_VENDOR_MODULES = {
    "wisereport": ("tradingagents.dataflows.wisereport",
                   "tradingagents.dataflows.vendors.wisereport"),
    "naver_news": ("tradingagents.dataflows.naver_news",
                   "tradingagents.dataflows.vendors.naver_news"),
    "naver_discussion": ("tradingagents.dataflows.naver_discussion",
                         "tradingagents.dataflows.vendors.naver_discussion"),
    "opendart_fundamentals": ("tradingagents.dataflows.opendart_fundamentals",
                              "tradingagents.dataflows.vendors.opendart_fundamentals"),
}

# fixtures/http_routes.json "naver_discussion": the authors behind the posts. The vendor
# promises never to pass them on (PIPA); the harness serves them so that promise is tested.
DISCUSSION_AUTHORS = (
    "harness_nick_alpha", "harness_nick_beta", "harness_nick_gamma",
    "harness-profile-001", "harness-profile-002", "harness-profile-003",
    "harness-user-001", "harness-user-002", "harness-user-003",
)

# Yahoo answering HTTP 429 to everything, the way yfinance 1.7 (production's version)
# surfaces it: Ticker.history / .info / Search / statements raise YFRateLimitError
# (yfinance/scrapers/history.py keeps rate limits as a special case), while yf.download
# swallows every per-ticker error into an empty frame (yfinance/multi.py _download_one).
# No "times": the throttle outlasts yf_retry's three retries.
YAHOO_THROTTLED = [
    {"api": "history", "raise": "YFRateLimitError", "name": "yahoo-429"},
    {"api": "download", "empty": True, "name": "yahoo-429-swallowed-by-download"},
    {"api": "info", "raise": "YFRateLimitError", "name": "yahoo-429"},
    {"api": "Search", "raise": "YFRateLimitError", "name": "yahoo-429"},
    {"api": "get_news", "raise": "YFRateLimitError", "name": "yahoo-429"},
    {"api": "statements", "raise": "YFRateLimitError", "name": "yahoo-429"},
    {"api": "insider_transactions", "raise": "YFRateLimitError", "name": "yahoo-429"},
]
# Nothing in the fork requests Yahoo over HTTP (yfinance is faked above), so this rule is
# inert today. Upstream's loader tells "no rows" from an outage with a HEAD to Yahoo
# (net.vendor_reachable); a throttled Yahoo still answers it, with a 429.
YAHOO_THROTTLED_HTTP = [
    {"name": "yahoo-429", "match": ["finance.yahoo.com"], "status": 429,
     "text": "Too Many Requests"},
]

# Alpha Vantage's close for today in the cross-check test: outside AAPL's synthetic
# Yahoo path (base 232.14, within about +-3%), so it can only come from Alpha Vantage.
AV_CLOSE = 241.37


def _kr_spec(name: str, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    """s1's KR holding (nightly shape) without its memory-log seed: no settlement noise."""
    return scenarios.derive(scenarios.get("s1_nightly_kr_holding_sell"), patch or {},
                            name=name, memory_log_seed=None)


def _us_spec(name: str, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    """s4's US discovery run (position context explicitly '')."""
    return scenarios.derive(scenarios.get("s4_us_not_held_discovery"), patch or {}, name=name)


KR_LIVE = _kr_spec("dl_kr_live")
US_LIVE = _us_spec("dl_us_live")


def _tool_output(res: RunResult, role: str, tool: str) -> str:
    """What the real ``tool`` returned to ``role``'s model during the run."""
    outputs = [str(t["content"]) for t in res.tool_results(role) if t["name"] == tool]
    seen = sorted({t["name"] for t in res.tool_results(role)})
    assert outputs, f"{role} never got a {tool!r} result back (got {seen})\n{res.describe()}"
    return "\n".join(outputs)


def _http_calls(res: RunResult, *needles: str) -> list[dict[str, Any]]:
    """HTTP requests of the run whose full URL contains every needle."""
    return [d for d in res.data_calls
            if d.get("lib") == "http" and all(n in str(d.get("url")) for n in needles)]


def _urls(res: RunResult) -> list[str]:
    return [f"{d.get('status')} {d.get('url')}" for d in res.data_calls if d.get("lib") == "http"]


def _line(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(label):
            return line.strip()
    raise AssertionError(f"no line starting with {label!r} in:\n{text}")


def _search_queries(res: RunResult) -> list[str]:
    return [str(d.get("query")) for d in res.data_calls
            if d.get("lib") == "yfinance" and d.get("api") == "Search"]


class _NetworkRefused(BaseException):
    """BaseException on purpose: no vendor's ``except Exception`` can hide the attempt."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process vendor calls must decline before any I/O; refuse it loudly if not."""
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise _NetworkRefused("a KR vendor reached for the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# ---------------------------------------------------------------------- vendor routing


def test_kr_run_takes_fundamentals_from_wisereport_and_news_from_naver(ap_run):
    """Breaks if a KR run's chains stop reaching the KR vendors: upstream's router has no
    'naver'/'wisereport' rows and an explicit chain silently keeps only the vendors the
    router knows ('naver,yfinance' becomes 'yfinance'), or ``main.build_config()`` loses
    its KR ``data_vendors`` override while main.py's conflict is resolved. Either way every
    KR nightly run analyses Yahoo's sparse English headlines and info-only fundamentals
    (no consensus, no estimates) with rc 0, and nothing in alpha-pulse notices."""
    res = ap_run(KR_LIVE, argv=KR_ARGV).assert_ok()

    fundamentals = _tool_output(res, "fundamentals_analyst", "get_fundamentals")
    # wisereport's consensus for 417310 (http_routes.json wisereport_main_417310 and
    # wisereport_financials_417310): target price 11,850 KRW, forward PER 18.56.
    assert "11,850" in fundamentals and "18.56" in fundamentals, fundamentals
    # Yahoo's info block would print the market cap (instruments.json 417310.KS).
    assert "297000000000" not in fundamentals, fundamentals
    wisereport = _http_calls(res, "navercomp.wisereport.co.kr", "cmp_cd=417310")
    assert wisereport, _urls(res)

    news = _tool_output(res, "news_analyst", "get_news")
    assert "리츠 섹터, 차입금 차환 앞두고 배당 여력 점검" in news, news  # Naver-only headline
    assert "Korean REITs weigh refinancing costs as loans come due" not in news, news  # Yahoo's
    naver = _http_calls(res, "m.stock.naver.com/api/news/stock/417310")
    assert naver, _urls(res)
    # Every per-ticker news request of the run went through the KR chain (the sentiment
    # analyst pre-fetches through it too): Yahoo was never asked for this ticker's news.
    yahoo_news = [d for d in res.data_calls if d.get("lib") == "yfinance"
                  and d.get("api") == "Ticker.get_news" and d.get("symbol") == KR_TICKER]
    assert yahoo_news == [], yahoo_news


def test_us_run_is_served_by_yahoo_and_never_reaches_korean_sources(ap_run):
    """Breaks if the KR-only vendors stop stepping aside for a US symbol under the same
    chains -- a KR vendor that queries Naver / wisereport / OpenDART with 'AAPL' before
    declining -- so a US web/discovery run sends traffic to Korean hosts, or gets their
    empty answers, a NO_DATA sentinel or a crash instead of Yahoo's data."""
    res = ap_run(US_LIVE, argv=US_ARGV).assert_ok()

    fundamentals = _tool_output(res, "fundamentals_analyst", "get_fundamentals")
    # Yahoo's info for AAPL (instruments.json): market cap and trailing P/E.
    assert "3452000000000" in fundamentals and "35.23" in fundamentals, fundamentals
    news = _tool_output(res, "news_analyst", "get_news")
    assert "Apple suppliers ramp iPhone 18 production ahead of September event" in news, news
    korean = [u for u in _urls(res) if any(host in u for host in KR_HOSTS)]
    assert korean == [], korean


def test_macro_news_queries_follow_the_market_of_the_analysed_ticker(ap_run):
    """Breaks if the ticker's news region stops reaching the data layer. Upstream runs the
    graph inside ``with run_config(self.config)`` (96daaf1), which snapshots the config; a
    merge that leaves the fork's ``news_region`` assignment after that snapshot sends every
    KR run the US/EU macro queries (Fed, S&P 500) instead of Bank of Korea / KOSPI context
    -- and alpha-pulse's nightly batch is KR-only."""
    kr = _search_queries(ap_run(KR_LIVE, argv=KR_ARGV).assert_ok())
    us = _search_queries(ap_run(US_LIVE, argv=US_ARGV).assert_ok())
    korea = ("Korea", "KOSPI", "KOSDAQ")
    assert kr and any(term in q for q in kr for term in korea), kr
    assert us and not any(term in q for q in us for term in korea), us


@pytest.mark.parametrize("key", sorted(KR_VENDOR_MODULES))
def test_kr_vendor_module_imports_from_this_checkout(key):
    """Breaks if a fork-only KR vendor module no longer imports. Upstream deleted
    ``dataflows/symbol_utils.py`` (it is ``symbols.py`` now) while wisereport, naver_news
    and opendart_* still say ``from .symbol_utils import NoMarketDataError``: main.py then
    dies at import with rc 1 before any LLM call, for every ticker (every nightly run
    fails), and the lazily imported discussion module would fail without a sound."""
    module = import_first(*KR_VENDOR_MODULES[key])
    assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT.resolve()), module.__file__


def test_kr_vendors_decline_a_us_ticker_with_the_routers_no_data_error(no_network):
    """Breaks if a KR vendor stops declining a non-KR ticker the way the router's
    fall-through expects: with the error taxonomy's NoMarketDataError, before any I/O. A
    stand-in class (written to paper over the stale ``symbol_utils`` import) is not "no
    data here" to the router -- every US run then logs the KR vendor as broken, and a chain
    whose remaining vendors also error surfaces the KR refusal as the tool's exception
    (ToolNode re-raises, rc 1) instead of a sentinel. A vendor that does its I/O first
    sends US symbols to Korean hosts."""
    no_data = get_symbol("NoMarketDataError", "tradingagents.dataflows.errors")
    wisereport = import_first(*KR_VENDOR_MODULES["wisereport"])
    naver_news = import_first(*KR_VENDOR_MODULES["naver_news"])
    opendart = import_first(*KR_VENDOR_MODULES["opendart_fundamentals"])
    with pytest.raises(no_data):
        wisereport.get_fundamentals(US_TICKER, TODAY.isoformat())
    with pytest.raises(no_data):
        naver_news.get_news(US_TICKER, "2026-09-01", TODAY.isoformat())
    with pytest.raises(no_data):
        opendart.get_fundamentals(US_TICKER, TODAY.isoformat())


def test_kr_fundamentals_fall_back_to_yahoo_with_derived_valuation(tmp_path):
    """Breaks if, while wisereport is down, (a) the fork's KR-derived valuation (d6a4f54)
    is gone -- Yahoo omits trailing P/E, EPS, P/B and book value for most .KS/.KQ listings
    and upstream's yahoo fundamentals prints only what Yahoo returns, so the fundamentals
    analyst of a KR nightly run gets no valuation anchor at all (rc 0, silent) -- or (b)
    the outage no longer falls through to Yahoo (rc 1, or a sentinel instead of data)."""
    spec = _kr_spec("dl_kr_wisereport_down", {"http_rules": [
        {"name": "wisereport_down", "match": ["navercomp.wisereport.co.kr/"], "status": 500,
         "text": "Internal Server Error"},
    ]})
    res = run_main(tmp_path, spec, argv=KR_ARGV).assert_ok()
    # wisereport is still first in the chain: it was asked, and it failed.
    down = [d for d in _http_calls(res, "navercomp.wisereport.co.kr") if d.get("status") == 500]
    assert down, _urls(res)

    fundamentals = _tool_output(res, "fundamentals_analyst", "get_fundamentals")
    # Hand-derived from instruments.json 417310.KS (its info has no trailingEps,
    # trailingPE, priceToBook or bookValue):
    #   EPS  = netIncomeToCommon 15,800,000,000 / sharesOutstanding 27,500,000 = 574.545..
    #   P/E  = currentPrice 10,800 / 574.545.. = 18.797..
    #   BVPS = quarterly Stockholders Equity 261,000,000,000
    #          / Ordinary Shares Number 27,500,000 = 9,490.909..
    #   P/B  = 10,800 / 9,490.909.. = 1.1379..
    for label, value in (("EPS (TTM)", "574.55"), ("PE Ratio (TTM)", "18.8"),
                         ("Book Value", "9490.91"), ("Price to Book", "1.14")):
        line = _line(fundamentals, label)
        assert value in line and "(derived)" in line, (label, line)


def test_kr_discussion_board_reaches_the_sentiment_analyst_without_its_authors(ap_run):
    """Breaks if a KR run loses the Naver 종목토론방 source: the merge's sentiment_analyst.py
    conflict hunk swallows ``kr_discussion_block = _maybe_fetch_kr_discussion(ticker)``
    (a known merge hazard), and the module is imported lazily inside a fail-open try, so a
    lost line or a broken import is silent (rc 0). Also breaks if
    post authors (nickname / profileId / userId) start flowing into prompts, the saved
    report or the memory log alpha-pulse archives (PIPA: the vendor must drop them)."""
    res = ap_run(KR_LIVE, argv=KR_ARGV).assert_ok()
    board = _http_calls(res, "m.stock.naver.com/front-api/discussion/list", "itemCode=417310")
    assert board, _urls(res)
    # A top-level post of http_routes.json "naver_discussion".
    prompt = res.prompt_for("sentiment_analyst")
    assert "배당 보고 기준일까지 들고 간다" in prompt, prompt[-3000:]

    places = {"stdout": res.stdout, "memory log": res.memory_log or ""}
    for call in res.llm_calls:
        places[f"LLM call {call.get('seq')} ({call.get('role')})"] = RunResult.prompt_text(call)
    for rel in res.report_files:
        places[f"report {rel}"] = res.report_file(rel)
    leaks = [(where, who) for where, text in places.items()
             for who in DISCUSSION_AUTHORS if who in text]
    assert leaks == [], leaks


# ------------------------------------------------------------ market verification tools


def test_verified_snapshot_flags_a_lagging_yahoo_close_with_alpha_vantage(tmp_path):
    """Breaks if the latest-close cross-check turns into a silent no-op. Upstream moved
    ``alpha_vantage_stock`` to ``vendors/alpha_vantage/stock.py``; the snapshot imports it
    lazily inside the cross-check's fail-open try, so after a merge the ImportError is
    swallowed, Alpha Vantage is never asked and the section never appears (rc 0, no error)
    -- and the market analyst anchors on Yahoo's stale close. Also breaks if the merge
    takes upstream's snapshot, which has no cross-check at all."""
    spec = _us_spec("dl_us_av_crosscheck", {
        # Yahoo lags the latest session: no bar for today on any price endpoint.
        "yfinance": {"faults": [{"api": "*", "symbol": US_TICKER, "name": "yahoo-lags",
                                 "drop_dates": [TODAY.isoformat()]}]},
        # Alpha Vantage already has today's close (served through the requests fake).
        "alpha_vantage": {"extra_rows": [{"date": TODAY.isoformat(), "close": AV_CLOSE}]},
    })
    res = run_main(tmp_path, spec, argv=US_ARGV,
                   env_overrides={"ALPHA_VANTAGE_API_KEY": "placeholder"}).assert_ok()
    asked = _http_calls(res, "www.alphavantage.co/query", "function=TIME_SERIES_DAILY",
                        "symbol=AAPL")
    assert asked, f"the snapshot never asked Alpha Vantage for the latest close: {_urls(res)}"

    snapshot = _tool_output(res, "market_analyst", "get_verified_market_snapshot")
    header = "Latest-price cross-check (Alpha Vantage)"
    assert header in snapshot, snapshot
    section = snapshot.split(header, 1)[1]
    assert "STALE" in section, section
    assert TODAY.isoformat() in section and f"{AV_CLOSE:.2f}" in section, section


def test_a_yahoo_throttle_that_outlasts_the_retries_fails_the_run(ap_run, tmp_path):
    """Breaks if a Yahoo rate limit that outlasts the retries stops failing the run.

    Today the price tools the market analyst must call raise (Ticker.history ->
    YFRateLimitError after yf_retry's three retries; the verified snapshot ->
    NoMarketDataError on the empty frame yf.download returns), the ToolNode re-raises and
    main.py exits non-zero; upstream v0.5.1 behaves the same. alpha-pulse learns of the outage
    only through that: a non-zero exit is a failed run, and a failed run keeps no plan.
    Upstream v0.5.2 turns the throttle into DATA_UNAVAILABLE tool text
    (``vendor_unavailable`` and the snapshot tool's except): the run would finish rc 0 with a
    TRADE_PLAN_JSON drafted without verified prices, which alpha-pulse would take as a normal
    plan. Adopting that needs a consumer-side guard first (e.g. no plan when the snapshot was
    unavailable); change this test only together with it.
    """
    control = ap_run(KR_LIVE, argv=KR_ARGV).assert_ok()
    assert control.plan is not None, "the same run without the throttle printed no plan"

    spec = _kr_spec("dl_kr_yahoo_throttled", {"yfinance": {"faults": YAHOO_THROTTLED},
                                              "http_rules": YAHOO_THROTTLED_HTTP})
    res = run_main(tmp_path, spec, argv=KR_ARGV)
    # Not vacuous: main.py reached the market analyst and Yahoo did throttle it.
    analyst_calls = res.calls_for("market_analyst")
    throttled = [d for d in res.data_calls if d.get("fault")]
    assert analyst_calls and throttled, res.describe()

    rc = res.rc
    assert rc != 0, "a persistent Yahoo 429 no longer fails the run\n" + res.describe()
