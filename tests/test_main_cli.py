"""main.py CLI arg parsing: ticker required, date defaults to today.

Imports main (guarded by ``if __name__ == "__main__"``) without running a graph.
"""
import datetime
import json
import os
import time
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import main as m


@pytest.mark.unit
class TestMainArgs:
    def test_ticker_is_required(self):
        with pytest.raises(SystemExit):
            m.parse_args([])

    def test_ticker_parsed(self):
        assert m.parse_args(["MU"]).ticker == "MU"

    def test_date_defaults_to_today(self):
        assert m.parse_args(["MU"]).date == datetime.date.today().isoformat()

    def test_date_explicit_passthrough(self):
        assert m.parse_args(["MU", "2026-01-15"]).date == "2026-01-15"

    def test_bad_date_is_rejected(self):
        with pytest.raises(SystemExit):
            m.parse_args(["MU", "not-a-date"])

    def test_exchange_qualified_ticker_roundtrips(self):
        # KR/exchange-qualified tickers must pass through unchanged.
        assert m.parse_args(["005930.KS"]).ticker == "005930.KS"

    @pytest.mark.parametrize(("argv_date", "iso"), [
        ("20260819", "2026-08-19"),
        ("20240229", "2024-02-29"),
    ])
    def test_basic_date_form_is_normalized_to_iso(self, argv_date, iso):
        # The nightly caller passes the KST date as YYYYMMDD; propagate() and the
        # report tree take YYYY-MM-DD.
        assert m.parse_args(["005930.KS", argv_date]).date == iso

    @pytest.mark.parametrize("bad", ["20260230", "20261301", "2026081", "202608190"])
    def test_malformed_basic_dates_are_rejected(self, bad):
        with pytest.raises(SystemExit):
            m.parse_args(["005930.KS", bad])

    def test_basic_date_form_does_not_rely_on_311_fromisoformat(self, monkeypatch):
        """``date.fromisoformat`` reads YYYYMMDD only from Python 3.11 on. Under a
        3.10-strict fromisoformat the nightly argv must still parse, not exit 2
        before the run starts -- requires-python allows 3.10."""

        class _Date310(datetime.date):
            @classmethod
            def fromisoformat(cls, value):
                if len(value) != 10 or value[4] != "-" or value[7] != "-":
                    raise ValueError(f"Invalid isoformat string: {value!r}")
                return super().fromisoformat(value)

        monkeypatch.setattr(m, "datetime", types.SimpleNamespace(
            date=_Date310, datetime=datetime.datetime))
        assert m.parse_args(["005930.KS", "20260819"]).date == "2026-08-19"
        assert m.parse_args(["005930.KS", "2026-08-19"]).date == "2026-08-19"


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is POSIX-only")
def test_the_default_date_is_the_process_local_date():
    """With no DATE, main.py analyses today in the process's own time zone (TZ):
    the consumer runs it with TZ=Asia/Seoul and expects the KST date, which
    differs from the UTC date every morning before 09:00. Checked under two zones
    26 hours apart, so a UTC-based default fails on at least one of them."""
    saved = os.environ.get("TZ")
    seen = {}
    try:
        for zone in ("Etc/GMT-14", "Etc/GMT+12"):   # UTC+14 and UTC-12
            os.environ["TZ"] = zone
            time.tzset()
            before = datetime.datetime.now(ZoneInfo(zone)).date().isoformat()
            got = m.parse_args(["005930.KS"]).date
            after = datetime.datetime.now(ZoneInfo(zone)).date().isoformat()
            seen[zone] = (got, {before, after})
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()
    for zone, (got, expected) in seen.items():
        assert got in expected, (zone, got, expected)
    assert seen["Etc/GMT-14"][0] != seen["Etc/GMT+12"][0]


@pytest.mark.unit
class TestMainWritesReports:
    def test_main_writes_rich_report_tree_and_prints_its_path_last(
            self, monkeypatch, tmp_path, capsys):
        # main() writes the report tree with the CLI's rich header (company label
        # + a per-role model table rendered from the run config) under
        # results_dir/reports/<TICKER>_<stamp>/, and its last stdout line points
        # at complete_report.md. Runs the real writer and header: only the graph
        # and the company-name lookup are stubbed.
        import tradingagents.agents.context as context

        real_build_config = m.build_config

        class FakeGraph:
            def __init__(self, *a, **k):
                pass

            def propagate(self, ticker, date):
                return {
                    "market_report": "MKT",
                    "final_trade_decision": "**Rating**: Buy",
                    "risk_debate_state": {"judge_decision": "**Rating**: Buy"},
                }, "Buy"

        monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
        monkeypatch.setattr(m, "build_config", lambda: {
            **real_build_config(), "results_dir": str(tmp_path / "results")})
        monkeypatch.setattr(context, "resolve_instrument_identity",
                            lambda ticker: {"company_name": "Micron Technology, Inc."})
        m.main(["MU", "2026-01-15"])

        lines = capsys.readouterr().out.splitlines()
        assert lines[-2] == "Buy"  # no typed plan: the parsed decision
        assert lines[-1].startswith("Report saved: ")
        report = Path(lines[-1][len("Report saved: "):])
        assert report.name == "complete_report.md"
        assert report.parent.parent == tmp_path / "results" / "reports"
        assert report.parent.name.startswith("MU_")
        text = report.read_text(encoding="utf-8")
        assert text.startswith("# Trading Analysis Report: Micron Technology, Inc. (MU)\n\n"
                               "Generated: ")
        # The model table comes from main.build_config(): Opus judges, Sonnet elsewhere.
        assert "| portfolio_manager | `vertex_anthropic` | `claude-opus-5-5` |" in text
        assert "| trader *(tier default)* | `vertex_anthropic` | `claude-sonnet-5-5` |" in text
        assert (report.parent / "1_analysts" / "market.md").read_text(encoding="utf-8") == "MKT"
        assert (report.parent / "5_portfolio" / "decision.md").read_text(
            encoding="utf-8") == "**Rating**: Buy"


_DECISION_WORDS = {"buy", "overweight", "hold", "underweight", "sell", "review"}


def _read_like_the_consumer(stdout: str) -> tuple[str | None, dict | None, str | None]:
    """(decision, plan, report) as a subprocess consumer reads main.py's stdout.

    The grammar docs/INTEGRATION.md §1b documents: the report is the LAST
    ``Report saved:`` line, the decision the nearest rating word on a line of its
    own above it, the plan the LAST ``TRADE_PLAN_JSON:`` line. Everything the run
    printed before (the debug trace of model messages) comes first and may
    contain any of those shapes.
    """
    lines = stdout.splitlines()
    saved = [i for i, line in enumerate(lines) if line.startswith("Report saved:")]
    if not saved:
        return None, None, None
    last = saved[-1]
    report = lines[last][len("Report saved:"):].strip()
    decision = next((line.strip() for line in reversed(lines[:last])
                     if line.strip().lower() in _DECISION_WORDS), None)
    plans = [line for line in lines if line.startswith("TRADE_PLAN_JSON:")]
    plan = json.loads(plans[-1][len("TRADE_PLAN_JSON:"):]) if plans else None
    return decision, plan, report


@pytest.mark.unit
def test_the_result_lines_come_last_after_a_trace_that_mimics_them(
        monkeypatch, tmp_path, capsys):
    """main.py runs the graph with debug=True, so every model message is printed
    before the result, and a message can hold a rating word on a line of its own,
    a ``TRADE_PLAN_JSON:`` line or a ``Report saved:`` line. The run's own three
    lines -- decision, plan, report path -- must follow everything the run
    printed, with nothing between them and nothing after, or the consumer reads
    the trace's values instead of the run's."""
    from tradingagents.agents import context
    from tradingagents.agents.schemas import PortfolioDecision

    typed = PortfolioDecision(rating="Underweight", executive_summary="s",
                              investment_thesis="t", stop_loss=95.0)

    class TracingGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date):
            # What a debug trace can look like: rating words, a plan line and a
            # report line written by a model, all before the result.
            print("================================ Ai Message ================================")
            print("Hold")
            print('TRADE_PLAN_JSON: {"rating": "Buy", "stop_loss": 1.0}')
            print("Report saved: /decoy/complete_report.md")
            print("Buy")
            state = {"final_trade_decision": "**Rating**: Underweight",
                     "portfolio_decision_obj": typed}
            return state, "Underweight"

    real_build_config = m.build_config
    monkeypatch.setattr(m, "TradingAgentsGraph", TracingGraph)
    monkeypatch.setattr(m, "build_config", lambda: {
        **real_build_config(), "results_dir": str(tmp_path / "results")})
    monkeypatch.setattr(context, "resolve_instrument_identity", lambda ticker: {})
    m.main(["MU", "2026-01-15"])

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[-3] == "Underweight"
    assert lines[-2].startswith("TRADE_PLAN_JSON: ")
    assert lines[-1].startswith("Report saved: ")
    decision, plan, report = _read_like_the_consumer(out)
    assert decision == plan["rating"] == "Underweight"
    assert plan["stop_loss"] == 95.0
    assert report == lines[-1][len("Report saved: "):]
    assert Path(report).name == "complete_report.md" and Path(report).is_file()
    assert Path(report).resolve().is_relative_to((tmp_path / "results").resolve())


@pytest.mark.unit
def test_build_config_is_the_korean_vertex_run_the_consumer_relies_on():
    """The runner's config is part of what the consumer gets: Korean output, the
    KR vendor chains (they refuse non-KR tickers, so US runs fall through to
    yfinance), 종목토론방 sentiment, the tiered Vertex Claude models. A merge that
    re-bases build_config on upstream's defaults changes every run silently."""
    cfg = m.build_config()
    assert cfg["llm_provider"] == "vertex_anthropic"
    assert (cfg["deep_think_llm"], cfg["quick_think_llm"]) == ("claude-opus-5-5", "claude-sonnet-5-5")
    assert (cfg["anthropic_thinking"], cfg["anthropic_max_tokens"], cfg["anthropic_effort"]) == (
        "adaptive", 32000, "high")
    assert cfg["output_language"] == "Korean"
    assert cfg["data_vendors"]["news_data"] == "naver,yfinance"
    assert cfg["data_vendors"]["fundamental_data"] == "wisereport,yfinance"
    assert cfg["enable_kr_discussion_sentiment"] is True
    assert cfg["position_context_from_env"] is True  # the PM-only account channel
    assert cfg["role_models"] == {
        role: {"provider": "vertex_anthropic", "model": "claude-opus-5-5",
               "anthropic_effort": "xhigh"}
        for role in ("research_manager", "portfolio_manager")
    }


_JUDGES = {"research_manager", "portfolio_manager"}


@pytest.mark.unit
@pytest.mark.parametrize("generic_max_tokens", [None, 8000])
def test_every_role_gets_the_runners_model_and_effort(monkeypatch, tmp_path, generic_max_tokens):
    """What each graph role is built with under main.build_config(), through the
    real resolver (tier defaults + role_models + client dedup): the two judges on
    Opus 5.5 at effort xhigh, every other role and the reflector on Sonnet 5.5 at
    effort high, all with max_tokens 32000 and adaptive thinking -- and nothing else
    (no temperature, no retry override). A merge that routes a judge through the
    deep tier, drops a per-role effort or lets the generic max_tokens replace the
    Vertex cap changes the production run without failing anything else."""
    import tradingagents.graph.trading_graph as tg

    built = {}

    def record(**kwargs):
        llm = object()
        built[id(llm)] = kwargs
        return types.SimpleNamespace(get_llm=lambda: llm)

    monkeypatch.setattr(tg, "create_llm_client", record)
    cfg = {**m.build_config(), "results_dir": str(tmp_path / "results"),
           "data_cache_dir": str(tmp_path / "cache"),
           "memory_log_path": str(tmp_path / "memory.md"),
           "max_tokens": generic_max_tokens}
    graph = tg.TradingAgentsGraph(debug=True, config=cfg)

    def spec(model, effort):
        return {"provider": "vertex_anthropic", "model": model, "base_url": None,
                "effort": effort, "max_tokens": 32000, "thinking": "adaptive",
                "project": None, "location": None}

    for role in sorted(tg.ROLE_KEYS):
        want = spec("claude-opus-5-5", "xhigh") if role in _JUDGES else spec("claude-sonnet-5-5", "high")
        assert built[id(graph._llm_for(role))] == want, role
    assert built[id(graph.reflector.quick_thinking_llm)] == spec("claude-sonnet-5-5", "high")
    assert len(tg.ROLE_KEYS) == 12


@pytest.mark.unit
def test_build_config_keeps_the_portfolio_notice_out_of_the_prompts():
    # main.py passes no portfolio; the account context reaches the Portfolio
    # Manager only (TRADINGAGENTS_POSITION_CONTEXT). The gate keeps upstream's
    # "Portfolio context: not provided" notice out of every prompt of its runs.
    assert m.build_config()["portfolio_notice_when_absent"] is False
