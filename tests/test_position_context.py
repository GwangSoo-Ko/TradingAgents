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

import functools
import json
from unittest.mock import MagicMock

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph, _read_position_context


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
