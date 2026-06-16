"""wisereport consensus/estimates vendor (Tier-2 Stage 3b), merged with OpenDART.

Network-free by default (mocked safe_get + mocked OpenDART). One opt-in live
test (TA_LIVE_WISEREPORT=1, needs KR network).
"""

import os
from types import SimpleNamespace

import pytest

from tradingagents.dataflows import wisereport as wr
from tradingagents.dataflows.symbol_utils import NoMarketDataError
from tradingagents.dataflows.wisereport import (
    _parse_consensus,
    _parse_financials,
    get_fundamentals,
)

_FIN_HTML = """
<table>
<tr><th>재무년월</th><th>매출액</th><th>영업이익</th><th>당기순이익</th><th>EPS</th><th>PER</th><th>PBR</th><th>ROE</th><th>EV/EBITDA</th><th>순부채</th><th>기준</th></tr>
<tr><td>2025(A)</td><td>3,336,059</td><td>10.88</td><td>436,011</td><td>442,610</td><td>6,564</td><td>18.27</td><td>1.87</td><td>10.85</td><td>7.53</td><td>-23.06</td><td>IFRS연결</td></tr>
<tr><td>2026(E)</td><td>6,846,861</td><td>105.24</td><td>3,516,619</td><td>2,874,804</td><td>43,098</td><td>7.36</td><td>3.00</td><td>51.46</td><td>4.32</td><td>-40.21</td><td>IFRS연결</td></tr>
</table>
"""
_MAIN_HTML = "<div>투자의견 목표주가 (원) EPS (원) PER (배) 추정기관수 4.04 396,667 43,098 7.36 24</div>"


@pytest.mark.unit
class TestParsers:
    def test_parse_financials(self):
        rows = _parse_financials(_FIN_HTML)
        assert len(rows) == 2
        assert rows[0]["year"] == "2025(A)" and rows[0]["revenue"] == "3,336,059"
        assert rows[0]["op_income"] == "436,011" and rows[0]["net_income"] == "442,610"
        assert rows[1]["year"] == "2026(E)" and rows[1]["per"] == "7.36"

    def test_parse_consensus(self):
        c = _parse_consensus(_MAIN_HTML)
        assert c == {"opinion": "4.04", "target_price": "396,667",
                     "fwd_eps": "43,098", "fwd_per": "7.36", "n_institutions": "24"}

    def test_parse_consensus_absent(self):
        assert _parse_consensus("<div>no consensus here</div>") is None


@pytest.mark.unit
class TestGetFundamentals:
    def _mock_fetch(self, monkeypatch):
        # safe_get called twice: financials then main page.
        monkeypatch.setattr(wr, "safe_get", lambda url, **kw: SimpleNamespace(
            text=_FIN_HTML if "cF1002" in url else _MAIN_HTML))

    def test_non_kr_raises(self):
        with pytest.raises(ValueError):
            get_fundamentals("AAPL")

    def test_merges_opendart_and_wisereport(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        monkeypatch.setattr(
            "tradingagents.dataflows.opendart_fundamentals.get_fundamentals",
            lambda t, c=None: "# OpenDART audited\nTotal Assets (자산총계): 566942110000000",
        )
        out = get_fundamentals("005930.KS", "2026-05-29")
        assert "OpenDART audited" in out                 # audited section present
        assert "자산총계" in out
        assert "target price 396,667" in out             # consensus
        assert "2026(E)" in out and "EV/EBITDA" in out    # forward estimates + ratios

    def test_opendart_failure_degrades_to_wisereport_only(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        def boom(t, c=None): raise RuntimeError("no DART key")
        monkeypatch.setattr("tradingagents.dataflows.opendart_fundamentals.get_fundamentals", boom)
        out = get_fundamentals("005930.KS", "2026-05-29")
        assert "OpenDART audited" not in out
        assert "wisereport" in out and "target price 396,667" in out  # still useful

    def test_no_data_raises(self, monkeypatch):
        monkeypatch.setattr(wr, "safe_get", lambda url, **kw: SimpleNamespace(text="<html></html>"))
        monkeypatch.setattr("tradingagents.dataflows.opendart_fundamentals.get_fundamentals",
                            lambda t, c=None: (_ for _ in ()).throw(RuntimeError("x")))
        with pytest.raises(NoMarketDataError):
            get_fundamentals("005930.KS", "2026-05-29")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TA_LIVE_WISEREPORT") != "1",
                    reason="set TA_LIVE_WISEREPORT=1 to hit the real wisereport endpoint")
class TestWisereportLive:
    def test_live_samsung(self):
        out = get_fundamentals("005930.KS", "2026-05-29")
        assert "target price" in out and "2026(E)" in out
