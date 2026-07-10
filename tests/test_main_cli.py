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
    def test_main_propagates_then_saves_report_tree(self, monkeypatch, tmp_path):
        calls = {}

        class FakeGraph:
            def __init__(self, *a, **k):
                pass

            def propagate(self, ticker, date):
                calls["propagate"] = (ticker, date)
                return {"final_trade_decision": "Buy"}, "Buy"

            def save_reports(self, final_state, ticker):
                calls["save_reports"] = (final_state, ticker)
                return tmp_path / "reports" / "MU_x" / "complete_report.md"

        monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
        m.main(["MU", "2026-01-15"])

        assert calls["propagate"] == ("MU", "2026-01-15")
        # save_reports gets the graph's final_state and the ticker.
        assert calls["save_reports"][1] == "MU"
        assert calls["save_reports"][0] == {"final_trade_decision": "Buy"}
