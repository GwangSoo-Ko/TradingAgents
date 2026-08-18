"""PortfolioDecision 이 집행 가능한 매매계획을 담는다.

진입 밴드·손절·분할은 지금까지 executive_summary 산문 안에만 있었다. 주문가로 쓰려면
구조화되어야 한다 — 보고서에는 폐기된 원안 숫자와 확정 숫자가 같은 문장에 섞여 있어
나중에 파싱하면 잘못된 값을 집는다.
"""
from __future__ import annotations

import json

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    Tranche,
    render_pm_decision,
)

pytestmark = pytest.mark.unit


def _decision(**kw) -> PortfolioDecision:
    base = {
        "rating": PortfolioRating.OVERWEIGHT,
        "executive_summary": "3분할 진입.",
        "investment_thesis": "밸류에이션 갭.",
    }
    base.update(kw)
    return PortfolioDecision(**base)


def test_trade_plan_fields_default_to_empty():
    """계획 필드는 선택이다 — 모델이 못 채워도 결정 자체는 유효해야 한다."""
    d = _decision()
    assert d.total_weight_pct is None
    assert d.stop_loss is None
    assert d.tranches == []


def test_tranches_round_trip_through_json():
    """AlphaPulse 가 stdout 에서 받는 것은 model_dump_json 결과다."""
    d = _decision(
        total_weight_pct=3.0,
        stop_loss=12050,
        tranches=[
            Tranche(seq=1, pct=30, price_low=13200, price_high=13400, trigger="immediate"),
            Tranche(seq=2, pct=30, price_low=12900, price_high=13000,
                    trigger="price", condition="볼린저 하단 접근 시"),
            Tranche(seq=3, pct=40, trigger="event", condition="8월 실적 확인"),
        ],
    )

    payload = json.loads(d.model_dump_json())

    assert payload["total_weight_pct"] == 3.0
    assert payload["stop_loss"] == 12050
    assert [t["trigger"] for t in payload["tranches"]] == ["immediate", "price", "event"]
    assert payload["tranches"][2]["price_low"] is None


def test_tranche_rejects_unknown_trigger():
    """trigger 는 셋뿐이다 — 오타가 조용히 통과하면 초안 판정이 무너진다."""
    with pytest.raises(ValueError):
        Tranche(seq=1, pct=100, trigger="asap")


def test_render_keeps_existing_markdown_shape():
    """마크다운은 다운스트림(메모리 로그·CLI·리포트)이 읽는다 — 헤더가 바뀌면 안 된다."""
    md = render_pm_decision(_decision(price_target=14000, time_horizon="6-12개월"))

    assert "**Rating**: Overweight" in md
    assert "**Executive Summary**:" in md
    assert "**Investment Thesis**:" in md
    assert "**Price Target**: 14000" in md


def test_render_shows_plan_when_present():
    """계획이 있으면 사람이 읽는 보고서에도 보여야 한다."""
    md = render_pm_decision(_decision(
        total_weight_pct=3.0,
        stop_loss=12050,
        tranches=[Tranche(seq=1, pct=100, price_low=13200, price_high=13400,
                          trigger="immediate")],
    ))

    assert "**Position Size**: 3.0%" in md
    assert "**Stop Loss**: 12050" in md
    assert "**Entry Plan**:" in md
    assert "13200" in md


def test_render_appends_plan_after_the_existing_headers():
    """신규 블록은 기존 헤더 '뒤'에 온다.

    부분문자열만 보면 Entry Plan 을 Executive Summary 위로 올리는 회귀가 green 으로
    통과한다 — 다운스트림(메모리 로그·CLI·리포트 작성기)은 순서를 전제로 읽는다.
    """
    md = render_pm_decision(_decision(
        price_target=14000,
        time_horizon="6-12개월",
        total_weight_pct=3.0,
        stop_loss=12050,
        tranches=[Tranche(seq=1, pct=100, price_low=13200, price_high=13400,
                          trigger="immediate")],
    ))

    order = [
        "**Rating**",
        "**Executive Summary**",
        "**Investment Thesis**",
        "**Price Target**",
        "**Time Horizon**",
        "**Position Size**",
        "**Stop Loss**",
        "**Entry Plan**",
    ]
    positions = [md.index(header) for header in order]
    assert positions == sorted(positions), dict(zip(order, positions, strict=True))
    # The tranche lines belong under the Entry Plan header, not above it.
    assert md.index("**Entry Plan**") < md.index("- #1 ")


def test_invoke_structured_returns_object_alongside_markdown():
    """렌더 결과만 반환하면 객체가 버려져 stdout 으로 꺼낼 수 없다."""
    from unittest.mock import MagicMock

    from tradingagents.agents.utils.structured import invoke_structured_or_freetext

    obj = _decision(total_weight_pct=3.0)
    structured = MagicMock()
    structured.invoke.return_value = obj

    text, returned = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", render_pm_decision, "Portfolio Manager",
    )

    assert "**Rating**: Overweight" in text
    assert returned is obj


def test_freetext_fallback_returns_none_object():
    """구조화가 실패하면 계획이 없다 — AlphaPulse 는 이때 초안을 만들지 않는다."""
    from unittest.mock import MagicMock

    from tradingagents.agents.utils.structured import invoke_structured_or_freetext

    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("boom")
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content="자유 텍스트 결론")

    text, returned = invoke_structured_or_freetext(
        structured, plain, "prompt", render_pm_decision, "Portfolio Manager",
    )

    assert text == "자유 텍스트 결론"
    assert returned is None


def test_state_declares_portfolio_decision_obj():
    """langgraph 는 state 스키마에 없는 키를 조용히 버린다(예외 없음).

    실측(probe): AgentState 에 없는 키를 노드가 반환하면 invoke 는 성공하지만
    최종 state 에서 사라진다. 선언이 빠지면 main.py 의 TRADE_PLAN_JSON 이
    영구히 안 나오는데 아무도 못 알아챈다.
    """
    from tradingagents.agents.utils.agent_states import AgentState

    assert "portfolio_decision_obj" in AgentState.__annotations__


def test_portfolio_manager_node_puts_object_in_state():
    """PM 노드가 구조화 객체를 state 에 실어야 main.py 가 stdout 으로 꺼낼 수 있다."""
    from unittest.mock import MagicMock

    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    obj = _decision(total_weight_pct=3.0, stop_loss=12050)
    structured = MagicMock()
    structured.invoke.return_value = obj
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    node = create_portfolio_manager(llm)
    result = node(_pm_state())

    assert result["portfolio_decision_obj"] is obj
    assert "**Rating**: Overweight" in result["final_trade_decision"]


def test_portfolio_manager_node_carries_none_on_freetext_fallback():
    """폴백이면 계획이 없다 — 마크다운은 살고 객체는 None 이다."""
    from unittest.mock import MagicMock

    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("boom")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content="자유 텍스트 결론")

    node = create_portfolio_manager(llm)
    result = node(_pm_state())

    assert result["portfolio_decision_obj"] is None
    assert result["final_trade_decision"] == "자유 텍스트 결론"


def _pm_state() -> dict:
    empty_risk_debate = {
        "history": "risk debate",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "count": 3,
    }
    return {
        "company_of_interest": "005830.KS",
        "trade_date": "2026-08-18",
        "instrument_context": "KR listed equity",
        "risk_debate_state": empty_risk_debate,
        "investment_plan": "research plan",
        "trader_investment_plan": "trader plan",
    }


def test_main_prints_trade_plan_json_line(monkeypatch, tmp_path, capsys):
    """Task 2 의 AlphaPulse 파서가 읽는 것은 stdout 의 이 한 줄이다."""
    import cli.main as cli_main
    import main as m

    obj = _decision(
        total_weight_pct=3.0,
        stop_loss=12050,
        tranches=[Tranche(seq=1, pct=100, price_low=13200, price_high=13400,
                          trigger="immediate")],
    )

    class FakeGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date):
            return {"final_trade_decision": "MD", "portfolio_decision_obj": obj}, "MD"

    monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(cli_main, "save_report_to_disk",
                        lambda *a, **k: tmp_path / "complete_report.md")
    m.main(["005830.KS", "2026-08-18"])

    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("TRADE_PLAN_JSON: ")]
    assert len(lines) == 1
    payload = json.loads(lines[0][len("TRADE_PLAN_JSON: "):])
    assert payload["total_weight_pct"] == 3.0
    assert payload["stop_loss"] == 12050
    assert payload["tranches"][0]["trigger"] == "immediate"


def test_main_omits_trade_plan_json_when_structured_failed(monkeypatch, tmp_path, capsys):
    """폴백이면 줄이 아예 없다 — 소비자는 계획을 지어내면 안 된다."""
    import cli.main as cli_main
    import main as m

    class FakeGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date):
            return {"final_trade_decision": "MD", "portfolio_decision_obj": None}, "MD"

    monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(cli_main, "save_report_to_disk",
                        lambda *a, **k: tmp_path / "complete_report.md")
    m.main(["005830.KS", "2026-08-18"])

    assert "TRADE_PLAN_JSON:" not in capsys.readouterr().out
