"""main.py CLI arg parsing: ticker required, date defaults to today.

Imports main (guarded by ``if __name__ == "__main__"``) without running a graph.
"""
import datetime
from pathlib import Path

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
        assert "| portfolio_manager | `vertex_anthropic` | `claude-opus-5` |" in text
        assert "| trader *(tier default)* | `vertex_anthropic` | `claude-sonnet-5` |" in text
        assert (report.parent / "1_analysts" / "market.md").read_text(encoding="utf-8") == "MKT"
        assert (report.parent / "5_portfolio" / "decision.md").read_text(
            encoding="utf-8") == "**Rating**: Buy"


@pytest.mark.unit
def test_build_config_keeps_the_portfolio_notice_out_of_the_prompts():
    # main.py passes no portfolio; the account context reaches the Portfolio
    # Manager only (TRADINGAGENTS_POSITION_CONTEXT). The gate keeps upstream's
    # "Portfolio context: not provided" notice out of every prompt of its runs.
    assert m.build_config()["portfolio_notice_when_absent"] is False
