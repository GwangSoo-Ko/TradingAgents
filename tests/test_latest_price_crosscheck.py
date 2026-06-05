"""Alpha Vantage latest-close cross-check in the verified market snapshot.

All network is mocked: the AV fetch and (where needed) load_ohlcv are patched,
so these tests are hermetic and prove the cross-check logic + look-ahead safety.
"""
from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.market_data_validator as validator


def _patch_av(monkeypatch, result):
    monkeypatch.setattr(
        "tradingagents.dataflows.alpha_vantage_stock.get_latest_close_on_or_before",
        lambda symbol, on_or_before: result,
    )


@pytest.mark.unit
class TestCrosscheckHelper:
    def test_flags_stale_primary_feed(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-04", 369.36))
        out = validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68)
        assert "STALE" in out
        assert "369.36" in out and "2026-06-04" in out
        assert "most recent close" in out

    def test_confirms_when_vendors_agree(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-03", 355.70))
        out = validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68)
        assert "confirmed" in out.lower()

    def test_flags_discrepancy_same_day(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-03", 380.00))
        out = validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68)
        assert "discrepancy" in out.lower()

    def test_notes_when_av_older(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-02", 358.39))
        out = validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68)
        assert "older than the primary feed" in out

    def test_empty_when_av_returns_none(self, monkeypatch):
        _patch_av(monkeypatch, None)
        assert validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68) == ""

    def test_empty_when_av_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("network down")
        monkeypatch.setattr(
            "tradingagents.dataflows.alpha_vantage_stock.get_latest_close_on_or_before", boom
        )
        assert validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68) == ""

    def test_empty_when_disabled_by_config(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-04", 369.36))
        monkeypatch.setattr(
            "tradingagents.dataflows.config.get_config",
            lambda: {"enable_alpha_vantage_price_crosscheck": False},
        )
        assert validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", 355.68) == ""

    def test_empty_when_yf_close_not_numeric(self, monkeypatch):
        _patch_av(monkeypatch, ("2026-06-04", 369.36))
        assert validator._latest_price_crosscheck("GOOG", "2026-06-05", "2026-06-03", "N/A") == ""


@pytest.mark.unit
class TestCrosscheckInSnapshot:
    def test_snapshot_includes_stale_flag(self, monkeypatch):
        dates = pd.bdate_range("2026-05-01", "2026-06-03")
        closes = [300 + i for i in range(len(dates))]
        df = pd.DataFrame({
            "Date": dates, "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": [1_000_000] * len(dates),
        })
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: df)
        _patch_av(monkeypatch, ("2026-06-04", 369.36))
        snap = validator.build_verified_market_snapshot("GOOG", "2026-06-05")
        assert "Latest-price cross-check (Alpha Vantage)" in snap
        assert "STALE" in snap and "369.36" in snap


@pytest.mark.unit
class TestAlphaVantageFetch:
    def test_filters_on_or_before_and_returns_latest(self, monkeypatch):
        csv = (
            "timestamp,open,high,low,close,volume\n"
            "2026-06-04,355.48,369.85,354.80,369.36,37645382\n"
            "2026-06-03,358.33,362.50,354.38,355.68,43031000\n"
            "2026-06-02,363.16,369.79,355.00,358.39,34648600\n"
        )
        monkeypatch.setattr(
            "tradingagents.dataflows.alpha_vantage_stock._make_api_request",
            lambda fn, params: csv,
        )
        from tradingagents.dataflows.alpha_vantage_stock import get_latest_close_on_or_before
        assert get_latest_close_on_or_before("GOOG", "2026-06-05") == ("2026-06-04", 369.36)
        # historical cutoff excludes 06-04 -> 06-03 (look-ahead safe)
        assert get_latest_close_on_or_before("GOOG", "2026-06-03") == ("2026-06-03", 355.68)

    def test_none_when_empty_response(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.dataflows.alpha_vantage_stock._make_api_request",
            lambda fn, params: "",
        )
        from tradingagents.dataflows.alpha_vantage_stock import get_latest_close_on_or_before
        assert get_latest_close_on_or_before("GOOG", "2026-06-05") is None
