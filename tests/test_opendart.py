"""OpenDART fundamentals vendor (Tier-2 Stage 2).

Network-free by default (mocked dart_get / corp map). One opt-in live test hits
the real OpenDART API and runs only when TA_LIVE_DART=1 (and DART_API_KEY set).
"""

import os

import pytest

from tradingagents.dataflows import opendart_common as common, opendart_fundamentals as fund
from tradingagents.dataflows.opendart_common import OpenDartNotConfiguredError, get_api_key
from tradingagents.dataflows.opendart_fundamentals import (
    _latest_available_fiscal_year,
    get_fundamentals,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _sample_list():
    return {"status": "000", "list": [
        {"account_nm": "자산총계", "sj_div": "BS", "thstrm_amount": "566,942,110,000,000", "currency": "KRW"},
        {"account_nm": "부채총계", "sj_div": "BS", "thstrm_amount": "130,621,773,000,000", "currency": "KRW"},
        {"account_nm": "자본총계", "sj_div": "BS", "thstrm_amount": "436,320,337,000,000", "currency": "KRW"},
        {"account_nm": "매출액", "sj_div": "IS", "thstrm_amount": "333,605,938,000,000", "currency": "KRW"},
        {"account_nm": "영업이익", "sj_div": "IS", "thstrm_amount": "43,601,051,000,000", "currency": "KRW"},
        {"account_nm": "당기순이익", "sj_div": "IS", "thstrm_amount": "45,206,805,000,000", "currency": "KRW"},
    ]}


@pytest.mark.unit
class TestApiKeyAndYear:
    def test_get_api_key_raises_when_absent(self, monkeypatch):
        monkeypatch.delenv("DART_API_KEY", raising=False)
        with pytest.raises(OpenDartNotConfiguredError):
            get_api_key()

    def test_latest_available_fiscal_year(self):
        # Annual reports filed by end-March: before April -> year-2, else year-1.
        assert _latest_available_fiscal_year("2026-05-29") == 2025
        assert _latest_available_fiscal_year("2026-02-15") == 2024
        assert _latest_available_fiscal_year(None) >= 2024


@pytest.mark.unit
class TestCorpCodeMapping:
    def test_non_kr_raises_valueerror(self, monkeypatch):
        monkeypatch.setattr(common, "_load_corp_map", lambda: {"005930": "00126380"})
        with pytest.raises(ValueError):
            common.corp_code_for("AAPL")

    def test_kr_resolves(self, monkeypatch):
        monkeypatch.setattr(common, "_load_corp_map", lambda: {"005930": "00126380"})
        assert common.corp_code_for("005930.KS") == "00126380"

    def test_unmapped_kr_raises_no_data(self, monkeypatch):
        monkeypatch.setattr(common, "_load_corp_map", lambda: {})
        with pytest.raises(NoMarketDataError):
            common.corp_code_for("999999.KQ")

    def test_doctype_guard_rejects_entity_xml(self, monkeypatch, tmp_path):
        # _load_corp_map must reject a DTD/ENTITY payload (XXE/billion-laughs).
        import io
        import zipfile
        malicious = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "boom">]><result></result>'
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as z:
            z.writestr("CORPCODE.xml", malicious)
        zpath = tmp_path / "opendart_corpcode.zip"
        zpath.write_bytes(zbuf.getvalue())
        common._CORP_MAP = None
        monkeypatch.setattr(common, "_corp_code_zip_path", lambda: zpath)
        with pytest.raises(ValueError):
            common._load_corp_map()
        common._CORP_MAP = None


@pytest.mark.unit
class TestGetFundamentals:
    def test_non_kr_raises_for_fallthrough(self, monkeypatch):
        # Upfront is_kr_ticker guard fires before corp_code_for; must raise
        # NoMarketDataError so route_to_vendor quiet-skips to the next vendor.
        with pytest.raises(NoMarketDataError):
            get_fundamentals("AAPL")

    def test_renders_headline_figures(self, monkeypatch):
        monkeypatch.setattr(fund, "corp_code_for", lambda t: "00126380")
        monkeypatch.setattr(fund, "dart_get", lambda path, **kw: _sample_list())
        out = get_fundamentals("005930.KS", "2026-05-29")
        assert "OpenDART" in out
        assert "Reporting currency: KRW" in out
        assert "Revenue (매출액): 333,605,938,000,000" in out
        assert "Net Income (당기순이익): 45,206,805,000,000" in out
        assert "Total Assets (자산총계):" in out

    def test_no_filing_raises_no_data(self, monkeypatch):
        monkeypatch.setattr(fund, "corp_code_for", lambda t: "00126380")
        def always_no_data(path, **kw):
            raise NoMarketDataError("005930.KS", "00126380", "013")
        monkeypatch.setattr(fund, "dart_get", always_no_data)
        with pytest.raises(NoMarketDataError):
            get_fundamentals("005930.KS", "2026-05-29")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("TA_LIVE_DART") != "1",
                    reason="set TA_LIVE_DART=1 (and DART_API_KEY) to hit the real OpenDART API")
class TestOpenDartLive:
    def test_live_samsung_fundamentals(self):
        out = get_fundamentals("005930.KS", "2026-05-29")
        assert "OpenDART" in out
        assert "자산총계" in out and "매출액" in out
