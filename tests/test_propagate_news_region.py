"""propagate() gives each run the macro-news region of its own ticker.

get_global_news picks Bank of Korea / KOSPI queries instead of Fed / S&P when the
config it reads carries ``news_region == "KR"``. propagate() sets that key from
the ticker, and it has to do so before ``run_config()`` snapshots the graph's
config for the run: set afterwards, a Korean run's data tools read the previous
run's region (or none on a fresh graph) and silently fall back to US macro news.
"""

import copy

import pytest

from tradingagents.dataflows.config import get_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _graph():
    g = object.__new__(TradingAgentsGraph)
    g.config = copy.deepcopy(DEFAULT_CONFIG)
    g._checkpointer_ctx = None
    return g


def _region_seen_by_a_run(graph, ticker):
    """The ``news_region`` a data tool would read while propagate() runs."""
    seen = []

    def _run(*a, **k):
        seen.append(get_config().get("news_region"))
        return {}, "Hold"

    graph._run_graph = _run
    graph.propagate(ticker, "2026-09-01")
    return seen


@pytest.mark.unit
@pytest.mark.parametrize("ticker, region", [
    ("005930.KS", "KR"),
    ("247540.KQ", "KR"),
    ("AAPL", None),
])
def test_the_run_sees_its_tickers_news_region(ticker, region):
    assert _region_seen_by_a_run(_graph(), ticker) == [region]


@pytest.mark.unit
def test_a_reused_graph_never_serves_the_previous_tickers_region():
    """A region set after the snapshot shows up one run late: the first KR run
    reads None, and the US run after it reads KR."""
    graph = _graph()
    assert _region_seen_by_a_run(graph, "005930.KS") == ["KR"]
    assert _region_seen_by_a_run(graph, "AAPL") == [None]
    assert _region_seen_by_a_run(graph, "247540.KQ") == ["KR"]
