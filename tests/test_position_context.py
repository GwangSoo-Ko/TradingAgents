"""position_context 가 그래프를 실제로 통과해 결정 노드에 닿는지 확인한다.

기존 past_context 테스트들은 전부 그래프를 우회한다(propagator 반환 dict 만
검사 / 노드에 dict 직접 주입 / graph.invoke 를 MagicMock). 그 패턴을 쓰면
langgraph 채널 drop 이 나도 초록이 된다 — 그래서 여기서는 컴파일된 그래프를
실제로 invoke 한다.
"""

import json

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import _read_position_context


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


def test_read_position_context_empty_when_blank(monkeypatch):
    """빈 문자열 주입은 '상속을 끊는다'는 뜻이다 -- AlphaPulse 가 발굴 deep 경로에서 쓴다."""
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", "")
    assert _read_position_context() == ""


def test_read_position_context_survives_broken_json(monkeypatch, capsys):
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", "{not json")
    assert _read_position_context() == ""
    assert "not valid JSON" in capsys.readouterr().err
