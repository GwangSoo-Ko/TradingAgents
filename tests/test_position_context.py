"""position_context 가 propagate() 의 실제 호출 경로 각 이음매에 닿는지 확인한다.

기존 past_context 테스트들은 전부 어딘가를 우회한다(propagator 반환 dict 만
검사 / 노드에 dict 직접 주입 / graph.invoke 를 MagicMock). 이 파일도 컴파일된
langgraph StateGraph 자체를 실제로 invoke 하지는 않는다(LLM 호출 없이는 전체
그래프를 끝까지 돌릴 수 없다) — 대신 이음매를 하나씩 실측한다:

- AgentState 선언 vs create_initial_state 반환 키 (langgraph silent-drop 가드)
- Propagator.create_initial_state 가 파라미터를 그대로 실어 나르는지
- TradingAgentsGraph._run_graph 가 실제로 position_context=_read_position_context()
  를 호출하는지 — MagicMock propagator/graph 에 진짜 ``_run_graph`` 와 그것이 거치는
  ``create_run_state``/``record_decision`` 을 바인딩해 AlphaPulse 가 의존하는 그
  배선 한 줄(trading_graph.py 의 create_initial_state 호출부)이 살아있는지를 본다
- _read_position_context 자체의 env 파싱 계약
"""

import ast
import contextlib
import functools
import io
import json
import tokenize
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tradingagents.agents as agents_pkg
from tradingagents.agents.context import build_position_block
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.state import AgentState
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
    본문만 실행시킨다. _run_graph 는 초기 상태를 ``create_run_state`` 로 만들고
    결정을 ``record_decision`` 으로 남기므로 그 둘도 진짜로 바인딩한다 -- MagicMock
    이 자동 생성한 가짜가 대신 불리면 create_initial_state 호출부와 스크럽을 전혀
    거치지 않아, 아래 배선 테스트들이 무의미하게 통과하거나 call_args 없이 실패한다.
    정산(settle_pending)·identity 조회·as_of 계산은 스텁으로 둔다.
    """
    mock_graph.memory_log.get_past_context.return_value = ""
    mock_graph.resolve_instrument_context.return_value = ""
    mock_graph.settle_pending.return_value = None
    mock_graph._memory_as_of.return_value = None
    mock_graph.config = {}
    mock_graph.debug = False
    mock_graph.propagator.create_initial_state.return_value = final_state
    mock_graph.propagator.get_graph_args.return_value = {}
    mock_graph.graph.invoke.return_value = final_state
    mock_graph._run_graph = functools.partial(TradingAgentsGraph._run_graph, mock_graph)
    mock_graph.create_run_state = functools.partial(
        TradingAgentsGraph.create_run_state, mock_graph
    )
    mock_graph.record_decision = functools.partial(
        TradingAgentsGraph.record_decision, mock_graph
    )
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


def test_record_decision_scrubs_account_numbers_on_its_own():
    """The CLI records through record_decision() without _run_graph, so the scrub
    has to live in record_decision itself -- a scrub kept only in _run_graph would
    leave the CLI's archive (and the next five same-ticker PM prompts) with the
    raw balances."""
    graph = MagicMock()
    decision = "Rating: Overweight\nAdd using the 456535870 cash; we hold 2697 shares."

    TradingAgentsGraph.record_decision(
        graph, "417310.KS", "2026-08-19",
        {"final_trade_decision": decision, "position_context": _CTX},
    )

    archived = graph.memory_log.store_decision.call_args.kwargs["final_trade_decision"]
    assert "456535870" not in archived
    assert "2697" not in archived
    assert "[redacted]" in archived


# The modules that must never see the account: every agent node but the Portfolio
# Manager. Listed only to prove the glob scan below really covered them.
_NON_PM_NODE_MODULES = frozenset({
    "analysts/market_analyst.py", "analysts/sentiment_analyst.py",
    "analysts/news_analyst.py", "analysts/fundamentals_analyst.py",
    "researchers/bull_researcher.py", "researchers/bear_researcher.py",
    "managers/research_manager.py", "trader/trader.py",
    "risk_mgmt/aggressive_debator.py", "risk_mgmt/conservative_debator.py",
    "risk_mgmt/neutral_debator.py",
})
_ACCOUNT_CHANNEL_NAMES = ("position_context", "build_position_block")


def _lines_allowed_to_name_the_channel(rel: str, src: str) -> set[int] | None:
    """Lines of ``tradingagents/agents/<rel>`` that may name the account channel.

    ``None`` means the whole module (the Portfolio Manager, the one reader). The
    state schema may only declare the channel, and context.py may only name it
    inside ``build_position_block`` (plus exporting that name). Every other
    module under tradingagents/agents gets no line at all.
    """
    if rel == "managers/portfolio_manager.py":
        return None
    tree = ast.parse(src)
    allowed: set[int] = set()
    if rel == "state.py":
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == "AgentState":
                for node in cls.body:
                    if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                            and node.target.id == "position_context"):
                        allowed.update(range(node.lineno, node.end_lineno + 1))
    elif rel == "context.py":
        for node in tree.body:
            is_renderer = (isinstance(node, ast.FunctionDef)
                           and node.name == "build_position_block")
            is_export_list = isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            if is_renderer or is_export_list:
                allowed.update(range(node.lineno, node.end_lineno + 1))
    return allowed


def _docstring_lines(src: str) -> set[int]:
    """Lines holding a module/class/function docstring (prose, not a read)."""
    lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def test_only_the_portfolio_manager_reads_position_context():
    """처분 효과 차단이 코드로 지켜지는지 -- 주석이 아니라 소스로 확인한다.

    Trader 와 Research Manager 가 이 목록에 있는 이유는 미묘하다. 둘 다 사이징을
    하지만, 둘 다 **마지막 노드가 아니다**:

    - Trader 의 TraderProposal 은 render_trader_proposal 로 position_sizing·
      reasoning 을 trader_investment_plan 에 렌더한다. aggressive/conservative/
      neutral 세 토론자가 그 문자열을 **그대로** 프롬프트에 박는다
      (setup.py: Trader -> Aggressive -> ... -> PM). Trader 에 주입하면 계좌가
      바로 그 세 모듈에 도달한다 -- 이 테스트가 지키려는 그 모듈들에.
    - Research Manager 의 investment_plan 은 PM 이 되읽는 사이징 입력이라
      가장 매력적인 미래 주입 지점이다.

    소스 스캔은 '다른 필드에 실려 들어오는 보유 정보'를 볼 수 없다. 그래서
    유일한 안전한 주입 지점은 그래프의 마지막 노드인 PM 하나뿐이다.

    Scanned by glob over every module under tradingagents/agents (not a fixed
    list), so a module added or moved by an upstream sync is covered the day it
    lands. Code counts -- names, attributes, string literals such as a state key;
    comments and docstrings do not (prose cannot read state).
    """
    agents_dir = Path(agents_pkg.__file__).resolve().parent
    scanned: set[str] = set()
    readers: list[str] = []
    for path in sorted(agents_dir.rglob("*.py")):
        rel = path.relative_to(agents_dir).as_posix()
        src = path.read_text(encoding="utf-8")
        scanned.add(rel)
        allowed = _lines_allowed_to_name_the_channel(rel, src)
        if allowed is None:
            continue
        prose = _docstring_lines(src)
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and tok.start[0] in prose:
                continue
            if any(name in tok.string for name in _ACCOUNT_CHANNEL_NAMES) \
                    and tok.start[0] not in allowed:
                readers.append(f"{rel}:{tok.start[0]}: {tok.string[:80]!r}")

    missing = _NON_PM_NODE_MODULES - scanned
    assert not missing, f"스캔이 노드 모듈을 놓쳤다(이동/개명?): {sorted(missing)}"
    assert "managers/portfolio_manager.py" in scanned
    assert not readers, "PM 이 아닌 곳이 보유를 읽는다:\n" + "\n".join(readers)


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



# --- 리뷰 후속: 스크럽이 실제 문장에서 새던 구멍들 ---------------------------


@pytest.mark.parametrize("text", [
    "Cash of 456535870.",
    "Cash 456535870, plus room.",
    "NAV is 456,535,870.",
    "We hold 2697.",
])
def test_scrub_catches_figures_at_sentence_end_and_before_a_comma(text):
    """가장 흔한 어법이 정확히 새던 자리다.

    이전 lookahead `(?![\\d,.])` 는 마침표·쉼표를 '숫자의 연속'으로 봐서, 문장
    끝(`...870.`)과 쉼표 앞(`...870,`)의 숫자를 통째로 통과시켰다. 브리프가 준
    두 테스트는 둘 다 숫자 뒤에 공백을 두는 바람에 이 구멍을 못 봤다 -- 즉
    스위트는 이 태스크가 막으려는 바로 그 유출에 눈이 멀어 있었다.
    """
    out = scrub_account_numbers(text, _CTX)
    assert "456535870" not in out
    assert "456,535,870" not in out
    assert "2697" not in out
    assert "[redacted]" in out


@pytest.mark.parametrize("text,unchanged_because", [
    ("Stop below 109000.", "109000 은 avg_price 10900 을 품고 있을 뿐 다른 숫자다"),
    ("Entry at 110900.", "110900 도 마찬가지 -- 앞쪽 경계"),
    ("Average cost basis 10900.50 per share.", "10900.50 은 10900 의 연속이다"),
])
def test_scrub_still_refuses_to_eat_part_of_a_longer_number(text, unchanged_because):
    assert scrub_account_numbers(text, _CTX) == text, unchanged_because


def test_scrub_redacts_integral_floats_written_as_bare_integers():
    """생산자가 10900.0 을 주고 모델이 10900 이라고 쓰면 -- 그 조합이 새면 안 된다.

    JSON 에는 숫자 타입이 하나뿐이고 position_context 생산자(후속 태스크)가
    int 로 줄지 float 로 줄지는 아직 못 박히지 않았다. 즉 이건 동전던지기다.
    """
    ctx = json.dumps({"held_qty": 2697.0, "avg_price": 10900.0,
                      "cash": 456535870.0, "total_nav": 500769000.0})
    text = "Average cost 10900 KRW; cash 456535870; NAV 500,769,000; 2697 shares."
    out = scrub_account_numbers(text, ctx)
    for leaked in ("10900", "456535870", "500,769,000", "2697"):
        assert leaked not in out, f"{leaked} 가 살아남았다: {out}"


def test_scrub_leaves_one_and_two_digit_figures_alone():
    """1주 보유가 결정문의 모든 '1' 을 지우면 아카이브가 망가진다.

    3자리 미만은 계좌 정보라고 부를 만한 게 못 된다(추측 가능하고, 잔고가
    아니다). 반대로 현실적인 현금·NAV 는 전부 3자리를 넘는다.
    """
    ctx = json.dumps({"held_qty": 1, "avg_price": 12, "cash": 5000, "total_nav": 100000})
    text = "R:R 1 to 3; Phase 1 entry; 12 month view."
    assert scrub_account_numbers(text, ctx) == text
    # 같은 맥락의 큰 값들은 여전히 지운다 -- 문턱이 스크럽 자체를 끄면 안 된다.
    assert "5000" not in scrub_account_numbers("Cash 5000.", ctx)
    assert "100,000" not in scrub_account_numbers("NAV 100,000.", ctx)


def test_langgraph_preserves_position_context_through_to_the_final_state():
    """리뷰가 제기한 미검증 가정: 실제 그래프의 final_state 가 이 키를 나르는가?

    나르지 않으면 _run_graph 의 스크럽은 `final_state.get("position_context", "")`
    에서 "" 를 받아 **조용한 무동작**이 된다 -- 예외도 로그도 없이 유출이 계속된다.
    위의 _run_graph 테스트들은 MagicMock 이 final_state 를 돌려주므로 이 가정을
    증명하지 못한다. 그래서 진짜 StateGraph 를 컴파일해서 실측한다.

    (position_context 는 리듀서 없는 Annotated[str, "..."] 로 선언돼 있어 langgraph
    기본 LastValue 채널이 된다. 어떤 노드도 이 키를 반환하지 않으므로 초기값이
    끝까지 살아남는다 -- 이 테스트가 그 동작을 langgraph 업그레이드에 대해 못박는다.)
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AgentState)
    # PM 이 하는 일만 흉내 낸다: 다른 키를 쓰고 position_context 는 건드리지 않는다.
    builder.add_node("decide", lambda state: {"final_trade_decision": "Rating: Hold"})
    builder.add_edge(START, "decide")
    builder.add_edge("decide", END)
    graph = builder.compile()

    initial = Propagator().create_initial_state("AAPL", "2026-08-19", position_context=_CTX)
    final_state = graph.invoke(initial)

    assert final_state["position_context"] == _CTX
    assert final_state["final_trade_decision"] == "Rating: Hold"


_FOUNDING = {
    "held_qty": 366, "avg_price": 8631.0, "current_price": 8250.0,
    "unrealized_pnl_pct": -4.41, "current_weight_pct": 3.1,
    "cash": 1000.0, "total_nav": 50000.0, "currency": "KRW",
    "founding_thesis": {
        "as_of": "20260903", "rating": "Overweight", "price_target": 10500.0,
        "time_horizon": "6-12 months", "kill_switch_price": 6300.0,
        "entry_price": 8718.0, "entry_date": "20260904",
        "trading_days_held": 1, "last_night_rating": "Underweight",
    },
    "founding_thesis_absent_reason": "",
}


def test_position_block_renders_founding_thesis():
    """PM 이 '그 포지션을 만든 계획'을 봐야 번복이 독립 추첨이 아니게 된다.

    바레 서브스트링(``"10500" in block``)은 값을 슬롯에 묶지 못한다 --
    price_target 과 entry_price 가 서로 뒤바뀌어 렌더돼도 두 숫자 다 어딘가에는
    나타나므로 통과해 버린다. 그래서 각 값을 그 값이 속한 레이블과 함께 한
    문자열로 묶어 단언한다: as_of+rating, entry_price+entry_date 처럼 뒤바뀔 수
    있는 쌍은 반드시 같이 박아, 뒤바뀜이 생기면 그 문자열 자체가 사라지게 한다.
    """
    block = build_position_block({"position_context": json.dumps(_FOUNDING)})

    assert "Founding Thesis" in block
    assert "revision" in block          # 무엇을 채워야 하는지 지시
    assert "On 20260903 you rated this **Overweight**" in block   # 작성일 + 진입 등급
    assert "price target 10500.0 KRW" in block                    # 목표가
    assert "horizon 6-12 months" in block                         # 호라이즌
    assert "Entered at 8718.0 KRW on 20260904" in block           # 진입가 + 진입일
    assert "That plan's own kill switch: 6300.0 KRW" in block     # kill switch
    assert "Most recent prior rating: Underweight" in block       # 어젯밤 등급


def test_position_block_states_trading_days_held_even_when_zero():
    """0 은 '오늘 사서 오늘 밤 뒤집는다' 다 — 가장 극단적인 사례가 사라지면 안 된다."""
    ctx = json.loads(json.dumps(_FOUNDING))
    ctx["founding_thesis"]["trading_days_held"] = 0

    block = build_position_block({"position_context": json.dumps(ctx)})

    assert "Trading days held since entry: 0" in block


def test_position_block_distinguishes_lookup_failure_from_no_plan():
    """조회 실패가 '계획 없음'으로 위장되면 인프라 장애가 수동 매수처럼 보인다."""
    failed = json.loads(json.dumps(_FOUNDING))
    failed["founding_thesis"] = None
    failed["founding_thesis_absent_reason"] = "lookup_failed"

    absent = json.loads(json.dumps(_FOUNDING))
    absent["founding_thesis"] = None
    absent["founding_thesis_absent_reason"] = "no_source_plan"

    b_failed = build_position_block({"position_context": json.dumps(failed)})
    b_absent = build_position_block({"position_context": json.dumps(absent)})

    assert "could not be retrieved" in b_failed
    assert "not opened from a plan" in b_absent
    assert b_failed != b_absent


def test_position_block_without_founding_keys_is_unchanged():
    """옛 페이로드(키 없음)로도 죽지 않아야 배포 순서가 어긋나도 안전하다."""
    block = build_position_block({"position_context": _CTX})

    assert "Founding Thesis" not in block
    assert block  # 기존 렌더는 그대로
