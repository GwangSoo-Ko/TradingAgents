"""PortfolioDecision 이 집행 가능한 매매계획을 담는다.

진입 밴드·손절·분할은 지금까지 executive_summary 산문 안에만 있었다. 주문가로 쓰려면
구조화되어야 한다 — 보고서에는 폐기된 원안 숫자와 확정 숫자가 같은 문장에 섞여 있어
나중에 파싱하면 잘못된 값을 집는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    Tranche,
    TrancheTrigger,
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
                    trigger="conditional", condition="볼린저 하단 접근 시",
                    triggers=[TrancheTrigger(kind="stop", price=12900)]),
            Tranche(seq=3, pct=40, trigger="conditional", condition="8월 실적 확인",
                    triggers=[TrancheTrigger(kind="event", condition="8월 실적 확인")]),
        ],
    )

    payload = json.loads(d.model_dump_json())

    assert payload["total_weight_pct"] == 3.0
    assert payload["stop_loss"] == 12050
    assert [t["trigger"] for t in payload["tranches"]] == [
        "immediate", "conditional", "conditional",
    ]
    assert payload["tranches"][2]["price_low"] is None
    # The triggers list has to survive the dump too -- it is where the sell-side
    # conditions live now, and the consumer reads only this JSON.
    assert payload["tranches"][0]["triggers"] == []
    assert payload["tranches"][1]["triggers"] == [{
        "kind": "stop", "price": 12900.0, "trail_pct": None,
        "reference_price": None, "reference_label": None, "condition": None,
    }]


def test_tranche_rejects_unknown_trigger():
    """trigger 는 둘뿐이다 — 오타가 조용히 통과하면 초안 판정이 무너진다."""
    with pytest.raises(ValueError):
        Tranche(seq=1, pct=100, trigger="asap")


def test_tranche_rejects_the_retired_trigger_vocabulary():
    """'price'/'event' 는 trigger 가 아니라 triggers[].kind 로 옮겨갔다.

    옛 값이 계속 통과하면 조건이 트랜치 사이로 흩어져 pct 분할 계약이 깨진다 —
    같은 물량을 두 번 세고도 합계 100 검증만 통과한다.
    """
    for retired in ("price", "event"):
        with pytest.raises(ValueError):
            Tranche(seq=1, pct=100, trigger=retired)


def test_immediate_tranche_defaults_to_no_triggers():
    """즉시 집행분에는 조건이 없다 — 소비자가 주문으로 바꾸는 것은 이쪽뿐이다."""
    assert Tranche(seq=1, pct=100, trigger="immediate").triggers == []


def test_null_triggers_do_not_discard_the_whole_plan():
    """모델이 생략 대신 null 을 쓰면 검증이 터져 계획 전체가 자유텍스트로 폴백한다."""
    assert Tranche(seq=1, pct=100, trigger="immediate", triggers=None).triggers == []


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

    from tradingagents.agents.structured import invoke_structured_or_freetext

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

    from tradingagents.agents.structured import invoke_structured_or_freetext

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
    from tradingagents.agents.state import AgentState

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


def test_pm_output_section_names_the_plan_fields():
    """upstream 486dec1 의 PM '## Output' 은 Rating/Executive Summary/Investment
    Thesis 세 섹션만 나열한다. 계획 필드가 목록에 없으면 구조화 호출이 선택 필드를
    덜 채울 수 있다 -- tranches 가 비면 AlphaPulse 는 델타 전량을 현재가로 한 번에
    주문하고, 비중이 비면 초안이 0건이다. 포크는 그 목록에 계획 필드와 revision
    지시를 얹는다."""
    from unittest.mock import MagicMock

    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or _decision()
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    create_portfolio_manager(llm)(_pm_state())

    assert "## Output" in captured["prompt"]
    section = captured["prompt"].split("## Output", 1)[1]
    for field in ("total_weight_pct", "tranches", "stop_loss", "exit_target",
                  "kill_switch", "revision"):
        assert f"`{field}`" in section, field
    # The rating still leads the list (free-text readers take the first labelled rating).
    assert section.index("**Rating**") < section.index("`total_weight_pct`")


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


def _run_main(monkeypatch, tmp_path, capsys, plan_obj) -> list[str]:
    """main.main() 를 가짜 그래프로 돌리고 stdout 줄을 돌려준다.

    보고서 작성기는 가짜로 바꾸지 않는다 -- main.py 의 post-run import(보고서
    작성기, safe_ticker_component, 헤더 헬퍼)는 유료 run 이 전부 끝난 뒤에야
    실행되므로, 그 경로가 깨지면 계획을 찍고도 rc 1 이 되어 AlphaPulse 가 계획을
    버린다. 작성기만 monkeypatch 하면 그 파손이 가려진다. 실제 작성기가 tmp_path
    아래에 쓰고, 오프라인 유지를 위해 헤더의 회사명 조회(identity)만 스텁한다.

    results_dir 는 main.build_config 의 결과에 직접 덮어쓴다. DEFAULT_CONFIG 를
    고치면 안 된다 -- 다른 테스트가 default_config 를 reload 하면 main 이 쥔
    DEFAULT_CONFIG 는 다른 dict 라서, 보고서가 사용자의 실제 results 디렉터리에
    써진다.
    """
    import main as m
    from tradingagents.agents import context

    class FakeGraph:
        def __init__(self, *a, **k):
            pass

        def propagate(self, ticker, date):
            return {"final_trade_decision": "MD", "portfolio_decision_obj": plan_obj}, "MD"

    real_build_config = m.build_config
    monkeypatch.setattr(m, "build_config",
                        lambda: {**real_build_config(), "results_dir": str(tmp_path)})
    monkeypatch.setattr(m, "TradingAgentsGraph", FakeGraph)
    monkeypatch.setattr(context, "resolve_instrument_identity", lambda ticker: {})
    m.main(["005830.KS", "2026-08-18"])
    return capsys.readouterr().out.splitlines()


def _assert_report_saved_last(lines: list[str], tmp_path) -> None:
    """AlphaPulse 는 마지막 'Report saved:' 줄로 run 완료를 판정한다."""
    assert lines and lines[-1].startswith("Report saved: "), lines[-3:]
    report = Path(lines[-1][len("Report saved: "):])
    assert report.name == "complete_report.md" and report.is_file(), report
    assert report.resolve().is_relative_to(tmp_path.resolve()), report


def test_main_prints_trade_plan_json_line(monkeypatch, tmp_path, capsys):
    """Task 2 의 AlphaPulse 파서가 읽는 것은 stdout 의 이 한 줄이다."""
    obj = _decision(
        total_weight_pct=3.0,
        stop_loss=12050,
        tranches=[Tranche(seq=1, pct=100, price_low=13200, price_high=13400,
                          trigger="immediate")],
    )

    out = _run_main(monkeypatch, tmp_path, capsys, obj)

    lines = [ln for ln in out if ln.startswith("TRADE_PLAN_JSON: ")]
    assert len(lines) == 1
    payload = json.loads(lines[0][len("TRADE_PLAN_JSON: "):])
    assert payload["total_weight_pct"] == 3.0
    assert payload["stop_loss"] == 12050
    assert payload["tranches"][0]["trigger"] == "immediate"
    # The decision line right above it is the typed rating -- the same value as
    # TRADE_PLAN_JSON.rating -- not the graph's text-parsed signal ("MD" here).
    assert out[out.index(lines[0]) - 1] == "Overweight"
    _assert_report_saved_last(out, tmp_path)


def test_main_omits_trade_plan_json_when_structured_failed(monkeypatch, tmp_path, capsys):
    """폴백이면 줄이 아예 없다 — 소비자는 계획을 지어내면 안 된다."""
    out = _run_main(monkeypatch, tmp_path, capsys, None)

    assert not [ln for ln in out if "TRADE_PLAN_JSON:" in ln]
    assert "MD" in out  # no typed plan: the graph's signal is the decision line
    _assert_report_saved_last(out, tmp_path)


def test_portfolio_decision_carries_revision():
    """revision 은 TRADE_PLAN_JSON 에 실려야 소비자(AlphaPulse)가 검증할 수 있다."""
    from tradingagents.agents.schemas import PlanRevision, PortfolioDecision

    d = PortfolioDecision(
        rating="Underweight",
        executive_summary="reduce",
        investment_thesis="thesis",
        revision=PlanRevision(kind="thesis_error", note="8월 랠리의 근거였던 업종 후광이 소멸"),
    )

    payload = json.loads(d.model_dump_json(exclude={"executive_summary", "investment_thesis"}))

    assert payload["revision"] == {
        "kind": "thesis_error",
        "note": "8월 랠리의 근거였던 업종 후광이 소멸",
    }


def test_revision_defaults_to_none_for_fresh_entries():
    """신규 진입에는 화해할 논지가 없다 — 기본값이 없으면 모든 매수 계획이 깨진다."""
    from tradingagents.agents.schemas import PortfolioDecision

    d = PortfolioDecision(rating="Overweight", executive_summary="buy", investment_thesis="t")

    assert d.revision is None


def test_revision_rejects_unknown_kind():
    """분류가 자유 문자열이면 집계가 불가능해진다 — Literal 이 그것을 막는다."""
    import pydantic

    from tradingagents.agents.schemas import PortfolioDecision

    with pytest.raises(pydantic.ValidationError):
        PortfolioDecision(
            rating="Sell",
            executive_summary="s",
            investment_thesis="t",
            revision={"kind": "vibes", "note": "n"},
        )


def test_render_pm_decision_includes_revision():
    """번복 사유는 저장되는 리포트 마크다운에도 남아야 사람이 사후에 읽는다."""
    from tradingagents.agents.schemas import PlanRevision, render_pm_decision

    md = render_pm_decision(
        _decision(rating="Underweight", revision=PlanRevision(kind="tactical", note="비중만 축소"))
    )

    assert "**Revision** (tactical): 비중만 축소" in md


@pytest.mark.parametrize(("typed", "extra", "quoted"), [
    ("Sell",
     {"revision": {"kind": "thesis_error",
                   "note": "The founding rating - Overweight - rested on the sector halo."}},
     "rating - Overweight"),
    ("Hold", {"executive_summary": "Most recent prior rating: Buy. 이번에는 관망한다."},
     "rating: Buy"),
    ("Sell", {"kill_switch": {"condition": "consensus rating: Hold 로 돌아서면 전량 청산"}},
     "rating: Hold"),
    ("Underweight",
     {"revision": {"kind": "tactical", "note": "진입 당시 등급(Rating: Buy)의 근거가 약해졌다."}},
     "Rating: Buy"),
])
def test_the_rendered_decision_reads_back_as_its_typed_rating(typed, extra, quoted):
    """메모리 로그 태그(decision_log.store_decision)와 propagate 의 신호는 렌더된 PM
    마크다운에서 등급을 다시 읽는다. PM 산문은 Founding Thesis 의 옛 등급이나 revision
    사유 속 다른 등급을 인용한다. upstream v0.5.1 의 '마지막 라벨' 파서는 그 인용을
    결정으로 읽어 typed Sell 을 'Overweight' 로 태그했다(b690dc7 백포트로 수정). 자기
    줄을 여는 첫 라벨('**Rating**: X')이 결정이어야 태그가 TRADE_PLAN_JSON.rating 과 같다."""
    from tradingagents.agents.rating import parse_rating

    md = render_pm_decision(_decision(rating=typed, **extra))

    assert quoted in md  # the prose really quotes a different rating
    assert parse_rating(md) == typed
