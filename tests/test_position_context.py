"""position_context 가 propagate() 의 실제 호출 경로 각 이음매에 닿는지 확인한다.

기존 past_context 테스트들은 전부 어딘가를 우회한다(propagator 반환 dict 만
검사 / 노드에 dict 직접 주입 / graph.invoke 를 MagicMock). 이 파일도 컴파일된
langgraph StateGraph 자체를 실제로 invoke 하지는 않는다(LLM 호출 없이는 전체
그래프를 끝까지 돌릴 수 없다) — 대신 이음매를 하나씩 실측한다:

- AgentState 선언 vs create_initial_state 반환 키 (langgraph silent-drop 가드)
- Propagator.create_initial_state 가 파라미터를 그대로 실어 나르는지
- TradingAgentsGraph._run_graph 가 실제로 position_context=_read_position_context()
  를 호출하는지 — MagicMock propagator/graph 에 진짜 ``_run_graph`` 를 바인딩해
  AlphaPulse 가 의존하는 그 배선 한 줄(trading_graph.py 의 create_initial_state
  호출부)이 살아있는지를 본다
- _read_position_context 자체의 env 파싱 계약
"""

import contextlib
import functools
import inspect
import json
from unittest.mock import MagicMock

import tradingagents.agents.analysts.fundamentals_analyst as fundamentals
import tradingagents.agents.analysts.market_analyst as market
import tradingagents.agents.analysts.news_analyst as news
import tradingagents.agents.analysts.sentiment_analyst as sentiment
import tradingagents.agents.analysts.social_media_analyst as social
import tradingagents.agents.researchers.bear_researcher as bear
import tradingagents.agents.researchers.bull_researcher as bull
import tradingagents.agents.risk_mgmt.aggressive_debator as aggressive
import tradingagents.agents.risk_mgmt.conservative_debator as conservative
import tradingagents.agents.risk_mgmt.neutral_debator as neutral
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import build_position_block
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import (
    TradingAgentsGraph,
    _read_position_context,
    scrub_account_numbers,
)


def test_state_declares_every_key_create_initial_state_returns():
    """특정 필드가 아니라 결함 클래스를 막는다 — 미선언 키는 langgraph 가 버린다."""
    state = Propagator().create_initial_state(
        "AAPL", "2026-08-19", position_context='{"held_qty": 1}'
    )
    declared = set(AgentState.__annotations__)
    undeclared = set(state) - declared
    assert not undeclared, f"AgentState 에 선언되지 않은 키: {undeclared}"


def test_create_initial_state_carries_position_context():
    state = Propagator().create_initial_state(
        "AAPL", "2026-08-19", position_context='{"held_qty": 2697}'
    )
    assert state["position_context"] == '{"held_qty": 2697}'


def test_position_context_defaults_to_empty():
    state = Propagator().create_initial_state("AAPL", "2026-08-19")
    assert state["position_context"] == ""


def test_read_position_context_returns_raw_json(monkeypatch):
    payload = json.dumps({"held_qty": 2697, "avg_price": 10900})
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", payload)
    assert _read_position_context() == payload


def test_read_position_context_empty_when_unset(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_POSITION_CONTEXT", raising=False)
    assert _read_position_context() == ""


def test_read_position_context_empty_when_blank(monkeypatch, capsys):
    """빈 문자열 주입은 '상속을 끊는다'는 뜻이다 -- AlphaPulse 가 발굴 deep 경로에서 쓴다.

    값이 "" 인 것뿐 아니라, 이 경로가 unset 과 동일하게 조용해야 한다는 것도
    함께 잠근다 -- 경고를 찍고도 우연히 "" 를 반환하는 회귀는 반환값만
    보면 통과해버린다.
    """
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", "")
    assert _read_position_context() == ""
    assert capsys.readouterr().err == ""


def test_read_position_context_survives_broken_json(monkeypatch, capsys):
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", "{not json")
    assert _read_position_context() == ""
    assert "not valid JSON" in capsys.readouterr().err


def _bind_real_run_graph(mock_graph, final_state):
    """MagicMock 위에 진짜 TradingAgentsGraph._run_graph 를 바인딩한다.

    tests/test_memory_log.py 의 test_full_pipeline_no_regression 과 동일한
    패턴 -- LLM/그래프 컴파일 없이, propagate() 가 실제로 호출하는 그 메서드
    본문(그리고 그 안의 create_initial_state 호출부)만 실행시킨다.
    """
    mock_graph.memory_log.get_past_context.return_value = ""
    mock_graph.resolve_instrument_context.return_value = ""
    mock_graph.config = {}
    mock_graph.debug = False
    mock_graph.propagator.create_initial_state.return_value = final_state
    mock_graph.propagator.get_graph_args.return_value = {}
    mock_graph.graph.invoke.return_value = final_state
    mock_graph._run_graph = functools.partial(TradingAgentsGraph._run_graph, mock_graph)
    return mock_graph


def test_run_graph_forwards_position_context_from_env(monkeypatch):
    """_run_graph 가 env 를 읽어 create_initial_state 에 실제로 실어 보내는지 확인한다.

    이것이 AlphaPulse 가 의존하는 실제 배선 지점(trading_graph.py 의
    ``position_context=_read_position_context(),`` 한 줄)이다. 위의
    create_initial_state 테스트들은 그 함수가 파라미터를 '받으면' 나른다는
    것만 증명할 뿐, _run_graph 가 실제로 그 파라미터를 넘긴다는 것은
    증명하지 못한다 -- 그 한 줄이 지워져도 위 테스트들은 전부 초록이다.
    """
    payload = json.dumps({"held_qty": 500, "avg_price": 123.45})
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", payload)
    mock_graph = _bind_real_run_graph(
        MagicMock(), {"final_trade_decision": "Rating: Buy\nBuy AAPL."}
    )

    mock_graph._run_graph("AAPL", "2026-08-19")

    call_kwargs = mock_graph.propagator.create_initial_state.call_args.kwargs
    assert call_kwargs["position_context"] == payload


def test_run_graph_forwards_empty_position_context_when_unset(monkeypatch):
    """env 미설정 시 _run_graph 가 '' 를 넘긴다 -- 뭔가를 넘긴다는 것과 올바른
    값을 넘긴다는 것은 다른 주장이라, 이 음성 케이스가 따로 필요하다."""
    monkeypatch.delenv("TRADINGAGENTS_POSITION_CONTEXT", raising=False)
    mock_graph = _bind_real_run_graph(
        MagicMock(), {"final_trade_decision": "Rating: Hold\nHold AAPL."}
    )

    mock_graph._run_graph("AAPL", "2026-08-19")

    call_kwargs = mock_graph.propagator.create_initial_state.call_args.kwargs
    assert call_kwargs["position_context"] == ""


# ---------------------------------------------------------------------------
# Task 3: 결정 단계 프롬프트 주입 + 아카이브 스크럽
#
# 위 테스트들이 보는 것은 "채널이 선언됐고 _run_graph 가 실어 보낸다"까지다.
# 아래는 그 다음 세 축이다:
#   1. build_position_block 렌더링 계약 (있음/없음/깨짐/미보유)
#   2. 실제로 그 블록을 받는 노드가 PM·Trader 뿐이라는 것 (프롬프트 캡처 + 소스 검사)
#   3. 아카이브(store_decision) 로 계좌 숫자가 새지 않는다는 것
# ---------------------------------------------------------------------------

_CTX = json.dumps({
    "held_qty": 2697, "avg_price": 10900, "current_price": 10950,
    "unrealized_pnl_pct": 0.46, "current_weight_pct": 5.90,
    "cash": 456535870, "total_nav": 500769000, "currency": "KRW",
})


def test_position_block_renders_when_context_present():
    block = build_position_block({"position_context": _CTX})
    assert "2697" in block
    assert "10900" in block or "10,900" in block
    assert "do not quote" in block.lower() or "인용" in block


def test_position_block_is_empty_string_when_absent():
    """머리말째 사라져야 한다. '**Current Position:**\\n(빈칸)' 은 모델에게
    '뭔가 있어야 하는데 없다'는 잘못된 신호가 된다."""
    assert build_position_block({}) == ""
    assert build_position_block({"position_context": ""}) == ""


def test_position_block_survives_broken_json():
    assert build_position_block({"position_context": "{broken"}) == ""


def test_position_block_survives_non_dict_json():
    """JSON 으로는 유효하지만 dict 가 아닌 값(리스트·스칼라)도 조용히 무시한다."""
    assert build_position_block({"position_context": "[1, 2]"}) == ""
    assert build_position_block({"position_context": "42"}) == ""


def test_position_block_says_not_held_when_qty_zero():
    ctx = json.dumps({"held_qty": 0, "avg_price": None, "total_nav": 500769000,
                      "cash": 456535870, "currency": "KRW"})
    block = build_position_block({"position_context": ctx})
    assert block
    assert "not currently held" in block.lower() or "미보유" in block


def test_scrub_removes_injected_account_numbers():
    text = "Deploy 5,000,000 KRW of the 456535870 cash against the 2697 shares held."
    out = scrub_account_numbers(text, _CTX)
    assert "456535870" not in out
    assert "2697" not in out


def test_scrub_removes_comma_formatted_account_numbers():
    """모델은 사람이 읽는 형태로 되쓴다 -- 자릿점 형태를 놓치면 스크럽은 무의미하다."""
    text = "Cash on hand is 456,535,870 KRW against a 500,769,000 KRW NAV."
    out = scrub_account_numbers(text, _CTX)
    assert "456,535,870" not in out
    assert "500,769,000" not in out


def test_scrub_is_noop_without_context():
    text = "Deploy 456535870."
    assert scrub_account_numbers(text, "") == text


def test_scrub_is_noop_on_broken_context():
    text = "Deploy 456535870."
    assert scrub_account_numbers(text, "{broken") == text
    assert scrub_account_numbers(text, "[1, 2]") == text


def test_scrub_keeps_percentages_and_prices():
    """% 사이징과 가격 수준은 남아야 한다 -- 프롬프트가 그 형태로 쓰라고 시킨다."""
    text = "Trim to 4.0% of NAV; unrealised P&L is 0.46%. Stop below 10,500."
    out = scrub_account_numbers(text, _CTX)
    assert out == text


def test_run_graph_scrubs_account_numbers_before_archiving(monkeypatch):
    """배선 축 -- store_decision 에 닿는 문자열에서 숫자가 지워졌는지 실측한다.

    scrub_account_numbers 단위 테스트만으로는 '함수는 맞는데 _run_graph 가
    부르지 않는다'를 못 잡는다. 그 상태로 유출은 계속되고, get_past_context
    (n_same=5) 가 다음 다섯 번의 실행을 오염시킨다.
    """
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", _CTX)
    decision = "Rating: Overweight\nAdd using the 456535870 cash; we hold 2697 shares."
    mock_graph = _bind_real_run_graph(
        MagicMock(), {"final_trade_decision": decision, "position_context": _CTX}
    )

    mock_graph._run_graph("417310.KS", "2026-08-19")

    archived = mock_graph.memory_log.store_decision.call_args.kwargs["final_trade_decision"]
    assert "456535870" not in archived
    assert "2697" not in archived


def test_run_graph_does_not_scrub_the_state_the_operator_sees(monkeypatch):
    """디스크 리포트·UI 는 모델의 원문을 봐야 한다 -- 스크럽은 아카이브 사본 한정."""
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", _CTX)
    decision = "Rating: Overweight\nAdd using the 456535870 cash; we hold 2697 shares."
    final_state = {"final_trade_decision": decision, "position_context": _CTX}
    mock_graph = _bind_real_run_graph(MagicMock(), final_state)

    returned_state, _ = mock_graph._run_graph("417310.KS", "2026-08-19")

    assert returned_state["final_trade_decision"] == decision


def test_analysts_never_read_position_context():
    """처분 효과 차단이 코드로 지켜지는지 -- 주석이 아니라 소스로 확인한다."""
    for mod in (market, news, sentiment, social, fundamentals,
                bull, bear, aggressive, conservative, neutral):
        src = inspect.getsource(mod)
        assert "position_context" not in src, f"{mod.__name__} 이 보유를 읽는다"
        assert "build_position_block" not in src, f"{mod.__name__} 이 보유를 읽는다"


class _CapturingLLM:
    """프롬프트를 잡아두는 스텁. structured 바인딩도 자기 자신을 돌려준다.

    invoke 가 항상 예외를 던지는 것은 의도다 -- 유효한 구조화 응답을 흉내 내면
    스키마 검증까지 테스트로 끌고 들어와야 한다. 우리가 보는 것은 프롬프트뿐이다.
    """

    def __init__(self):
        self.prompts: list[str] = []

    def with_structured_output(self, *_a, **_k):
        return self

    def bind_tools(self, *_a, **_k):
        return self

    def invoke(self, prompt, *_a, **_k):
        self.prompts.append(str(prompt))
        raise RuntimeError("stub: force free-text fallback")


def _decision_state(position_context: str) -> dict:
    return {
        "company_of_interest": "417310.KS",
        "instrument_context": "test instrument",
        "position_context": position_context,
        "investment_plan": "plan",
        "trader_investment_plan": "proposal",
        "past_context": "",
        "risk_debate_state": {
            "history": "h", "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "latest_speaker": "", "current_aggressive_response": "",
            "current_conservative_response": "", "current_neutral_response": "", "count": 0,
        },
    }


def _capture_prompt(node, position_context: str) -> str:
    llm = _CapturingLLM()
    # 스텁이 structured·free-text 두 호출을 모두 실패시킨다 -- 우리가 보는 건 프롬프트다.
    with contextlib.suppress(Exception):
        node(llm)(_decision_state(position_context))
    assert llm.prompts, "노드가 LLM 을 부르지 않았다"
    return llm.prompts[0]


def test_portfolio_manager_prompt_carries_the_position_block():
    prompt = _capture_prompt(create_portfolio_manager, _CTX)
    assert "2697" in prompt, "PM 프롬프트에 보유 수량이 없다"
    assert "Current Position" in prompt


def test_portfolio_manager_prompt_has_no_heading_when_context_absent():
    prompt = _capture_prompt(create_portfolio_manager, "")
    assert "Current Position" not in prompt


def test_trader_prompt_carries_the_position_block():
    prompt = _capture_prompt(create_trader, _CTX)
    assert "2697" in prompt, "Trader 프롬프트에 보유 수량이 없다"
    assert "Current Position" in prompt


def test_trader_prompt_has_no_heading_when_context_absent():
    prompt = _capture_prompt(create_trader, "")
    assert "Current Position" not in prompt
