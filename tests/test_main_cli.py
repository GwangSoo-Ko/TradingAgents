"""main.py CLI arg parsing: ticker required, date defaults to today.

Imports main (guarded by ``if __name__ == "__main__"``) without running a graph.
"""
import datetime

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
    def test_main_writes_rich_report_via_cli_writer(self, monkeypatch, tmp_path):
        # main() writes the CLI's rich-header report tree (company label + a
        # per-role model table), so it calls cli.main.save_report_to_disk WITH
        # the run config (the config is what renders the model table).
        import cli.main as cli_main
        calls = {}

        class FakeGraph:
            def __init__(self, *a, **k):
                pass

            def propagate(self, ticker, date):
                return {"final_trade_decision": "Buy"}, "Buy"

        def fake_save(final_state, ticker, save_path, config):
            calls["args"] = (final_state, ticker, save_path, config)
            return tmp_path / "complete_report.md"

        monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
        monkeypatch.setattr(cli_main, "save_report_to_disk", fake_save)
        m.main(["MU", "2026-01-15"])

        fs, ticker, save_path, config = calls["args"]
        assert ticker == "MU"
        assert fs == {"final_trade_decision": "Buy"}
        assert config["llm_provider"] == "vertex_anthropic"  # renders the model table
