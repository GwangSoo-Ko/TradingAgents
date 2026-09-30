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


@pytest.mark.unit
def test_the_region_stays_in_the_runs_own_scope():
    """The region is a fact about one run. Written into the graph's config it
    edits the caller's dict -- DEFAULT_CONFIG itself when the graph was built
    without one -- and written into the process-wide config it reaches whatever
    reads that next outside a run: the CLI's stream, another graph's tools."""
    graph = _graph()
    callers_config = graph.config
    before = copy.deepcopy(callers_config)

    assert _region_seen_by_a_run(graph, "005930.KS") == ["KR"]

    assert graph.config is callers_config and callers_config == before
    assert get_config().get("news_region") is None


# --- the interactive CLI ---------------------------------------------------------
# The CLI streams the graph itself instead of calling propagate(), so it has to put
# the region into the config it builds the graph with; the graph publishes that
# config to the data tools.


class _StreamingGraph:
    """The lifecycle run_analysis drives, with a two-chunk stream."""

    def __init__(self):
        self.graph = self
        self.propagator = self
        self._resuming = False

    def create_run_state(self, *a, **k):
        return {"messages": []}

    def get_graph_args(self, callbacks=None):
        return {}

    def begin_checkpoint(self, *a, **k):
        return None

    def checkpoint_input(self, state):
        return state

    def stream(self, graph_input, **kwargs):
        yield {"messages": [], "market_report": "M"}
        yield {"messages": [], "final_trade_decision": "Rating: Hold"}

    def record_decision(self, *a, **k):
        pass

    def clear_checkpoint_on_success(self, *a, **k):
        pass

    def end_checkpoint(self):
        pass

    def process_signal(self, text):
        return "Hold"


class _Buffer:
    def __init__(self):
        self.messages, self.tool_calls = [], []
        self.report_sections, self.agent_status = {}, {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        self.selected_analysts = [a.lower() for a in selected_analysts]

    def add_message(self, kind, content):
        self.messages.append((0.0, kind, content))

    def add_tool_call(self, name, args):
        self.tool_calls.append((0.0, name, args))

    def update_report_section(self, *a):
        pass

    def update_agent_status(self, agent, status):
        self.agent_status[agent] = status


class _NullLive:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _cli_run_config(monkeypatch, tmp_path, ticker):
    """The config run_analysis builds its graph with, for ``ticker``."""
    import cli.run as cli_run
    from cli.models import AnalystType

    built = []

    def _graph(*a, config=None, **k):
        built.append(config)
        return _StreamingGraph()

    monkeypatch.setattr(cli_run, "get_user_selections", lambda: {
        "ticker": ticker, "analysis_date": "2026-09-01", "asset_type": "stock",
        "analysts": [AnalystType.MARKET], "research_depth": 1,
        "quick_think_llm": "q", "deep_think_llm": "d", "backend_url": None,
        "llm_provider": "openai",
    })
    # The dict cli.run itself copies: a test that reloads default_config leaves
    # the module-level DEFAULT_CONFIG imported here a different object, and the
    # run would then write under the user's real results directory.
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", _graph)
    monkeypatch.setattr(cli_run, "message_buffer", _Buffer())
    monkeypatch.setattr(cli_run, "create_layout", lambda: None)
    monkeypatch.setattr(cli_run, "update_display", lambda *a, **k: None)
    monkeypatch.setattr(cli_run, "Live", _NullLive)
    monkeypatch.setattr(cli_run, "instrument_display_label", lambda t: t)
    monkeypatch.setattr(cli_run.typer, "prompt", lambda *a, **k: "N")
    cli_run.run_analysis()
    assert len(built) == 1
    return built[0]


@pytest.mark.unit
@pytest.mark.parametrize("ticker, region", [("005930.KS", "KR"), ("NVDA", None)])
def test_the_cli_run_gets_its_tickers_news_region(monkeypatch, tmp_path, ticker, region):
    assert _cli_run_config(monkeypatch, tmp_path, ticker)["news_region"] == region
