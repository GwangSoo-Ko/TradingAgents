"""직접 Anthropic API 경로(``anthropic`` 공급자)가 실제로 보내는 요청을 고정한다.

main.py 의 Claude 호출이 Vertex 에서 ANTHROPIC_API_KEY 의 직접 API 로
옮겨졌다. Vertex 경로와 같은 동작(강제하지 않는 tool 호출, effort·thinking·max_tokens)을
직접 경로에서도 지키는지, 설치된 진짜 ``langchain_anthropic.ChatAnthropic`` 과
``anthropic`` SDK 로 확인한다. 바꾸는 것은 HTTP 전송층(``HTTPTransport.handle_request``)
하나뿐이다: 요청 본문을 잡아 두고 미리 만든 Messages 응답을 돌려준다. 네트워크도
실제 키도 쓰지 않는다(conftest 가 키를 ``placeholder`` 로 채운다).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import PortfolioRating, ResearchPlan

PLAN_ARGS = {
    "recommendation": "Overweight",
    "rationale": "Services margin expansion outweighs the valuation premium.",
    "strategic_actions": "Build to 1.2x a standard allocation over two tranches.",
}


def _http_lib():
    """anthropic SDK 가 쓰는 HTTP 라이브러리 (0.x: httpx, 1.x: httpx2)."""
    import anthropic

    return pytest.importorskip("httpx2" if anthropic.__version__.startswith("1.") else "httpx")


def _message(content, stop_reason="end_turn", model="claude-opus-5-5"):
    return {"id": "msg_direct_test", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


TEXT_REPLY = _message([{"type": "text", "text": "OK"}])
TOOL_REPLY = _message(
    [{"type": "thinking", "thinking": "", "signature": "sig"},
     {"type": "tool_use", "id": "toolu_1", "name": "ResearchPlan", "input": PLAN_ARGS}],
    stop_reason="tool_use")


@pytest.fixture
def wire(monkeypatch):
    """가짜 전송층: 보낸 요청을 ``wire.requests`` 에 남기고 ``wire.replies`` 를 차례로 돌려준다."""
    http = _http_lib()

    class Wire:
        requests: list = []
        replies: list = []

    Wire.requests, Wire.replies = [], []

    def handle_request(self, request):
        raw = request.read()
        Wire.requests.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "timeout": request.extensions.get("timeout"),
            "body": json.loads(raw.decode("utf-8")) if raw else None,
        })
        reply = Wire.replies.pop(0) if Wire.replies else TEXT_REPLY
        return http.Response(200, json=reply, request=request)

    monkeypatch.setattr(http.HTTPTransport, "handle_request", handle_request)
    return Wire


def _direct_llm(model="claude-opus-5-5", **kwargs):
    from tradingagents.llm_clients.anthropic_client import AnthropicClient

    return AnthropicClient(model, **kwargs).get_llm()


# ============================================================================ R1: 구조화 출력


@pytest.mark.unit
class TestDirectClaudeStructuredOutput:
    def test_structured_call_lets_the_model_choose_the_tool(self, wire):
        """Opus 5.5 / Sonnet 5.5 는 강제 tool_choice({"type": "tool"} / "any")를 400 으로
        돌려준다. 직접 경로도 Vertex 와 같이 스키마를 유일한 tool 로 두고 auto 로 보낸다."""
        wire.replies.append(TOOL_REPLY)
        llm = _direct_llm(effort="xhigh")
        plan = llm.with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        assert isinstance(plan, ResearchPlan)
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        (request,) = wire.requests
        body = request["body"]
        assert body["tool_choice"] == {"type": "auto"}
        assert [t["name"] for t in body["tools"]] == ["ResearchPlan"]
        # 답을 tool 로 하라는 지시가 마지막 사용자 메시지로 붙는다 (Vertex 와 같은 문구)
        assert "`ResearchPlan`" in json.dumps(body["messages"][-1], ensure_ascii=False)

    def test_no_request_forces_the_tool_also_on_the_re_ask(self, wire):
        """산문으로 답하면 한 번 더 묻는다: 두 요청 모두 tool_choice 는 auto 다."""
        wire.replies.extend([_message([{"type": "text", "text": "Overweight, because..."}]),
                             TOOL_REPLY])
        plan = _direct_llm().with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        assert [r["body"]["tool_choice"] for r in wire.requests] == [{"type": "auto"}] * 2

    def test_include_raw_and_validation_errors_behave_as_on_vertex(self, wire):
        wire.replies.append(TOOL_REPLY)
        out = _direct_llm().with_structured_output(ResearchPlan, include_raw=True).invoke("x")
        assert isinstance(out["parsed"], ResearchPlan) and out["parsing_error"] is None
        bad = _message([{"type": "tool_use", "id": "toolu_2", "name": "ResearchPlan",
                         "input": {"recommendation": "Overweight"}}], stop_reason="tool_use")
        wire.replies.append(bad)
        with pytest.raises(ValidationError):  # 다시 묻지 않고 곧바로 올린다 (fail-closed)
            _direct_llm().with_structured_output(ResearchPlan).invoke("x")
        assert len(wire.requests) == 2

    def test_direct_and_vertex_share_one_structured_output_implementation(self):
        """두 공급자가 같은 구현을 쓴다: 한 곳을 고치면 둘 다 바뀐다(복사본이 갈라지지 않는다)."""
        from tradingagents.llm_clients import claude_structured, vertex_clients
        from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic

        mixin = claude_structured.ClaudeStructuredOutput
        assert issubclass(NormalizedChatAnthropic, mixin)
        assert vertex_clients._ClaudeStructuredOutput is mixin
        # mixin 이 ChatAnthropic 보다 MRO 앞에 있어야 그 with_structured_output 이 이긴다
        assert (NormalizedChatAnthropic.with_structured_output
                is mixin.with_structured_output)

    def test_invoke_still_flattens_thinking_blocks_to_text(self, wire):
        wire.replies.append(_message([{"type": "thinking", "thinking": "", "signature": "s"},
                                      {"type": "text", "text": "hi"}]))
        assert _direct_llm().invoke("x").content == "hi"


# ============================================================================ R2: thinking·effort·max_tokens

MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _production_kwargs(effort: str) -> dict:
    """main.build_config() 가 만드는 클라이언트 kwargs (build_llm_kwargs / 역할 경로의 결과)."""
    return {"effort": effort, "max_tokens": 32000, "thinking": "adaptive"}


@pytest.mark.unit
class TestDirectClaudeRequestShape:
    @pytest.mark.parametrize("model,effort", [("claude-opus-5-5", "xhigh"),
                                              ("claude-sonnet-5-5", "high")])
    def test_production_settings_reach_the_request_body(self, wire, model, effort):
        """effort 는 ChatAnthropic 의 ``effort`` 필드(= reasoning_effort 의 alias)로 넘기고,
        SDK 요청에서 ``output_config.effort`` 가 된다(최상위 effort 는 API 필드가 아니다).
        thinking 의 "adaptive" 약식은 {"type": "adaptive"} 로 감싼다."""
        _direct_llm(model, **_production_kwargs(effort)).invoke("ping")
        (request,) = wire.requests
        assert request["method"] == "POST" and request["url"] == MESSAGES_URL
        body = request["body"]
        assert body["model"] == model
        assert body["max_tokens"] == 32000
        assert body["output_config"] == {"effort": effort}
        assert body["thinking"] == {"type": "adaptive"}
        assert "effort" not in body and "reasoning_effort" not in body
        assert not {"temperature", "top_p", "top_k"} & set(body), sorted(body)
        assert request["headers"].get("x-api-key") == "placeholder"  # conftest 가 채운 ANTHROPIC_API_KEY

    def test_structured_call_carries_the_same_settings(self, wire):
        wire.replies.append(TOOL_REPLY)
        llm = _direct_llm("claude-opus-5-5", **_production_kwargs("xhigh"))
        llm.with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        body = wire.requests[0]["body"]
        assert body["tool_choice"] == {"type": "auto"}
        assert (body["max_tokens"], body["output_config"], body["thinking"]) == (
            32000, {"effort": "xhigh"}, {"type": "adaptive"})

    @pytest.mark.parametrize("thinking,expected", [
        ("adaptive", {"type": "adaptive"}),
        ("disabled", {"type": "disabled"}),
        ({"type": "adaptive", "display": "summarized"},
         {"type": "adaptive", "display": "summarized"}),
    ])
    def test_thinking_shorthand_and_dict_like_vertex(self, wire, thinking, expected):
        # disabled 는 Opus 5.5 가 거절하므로 Sonnet 5 로 확인한다 (모양만 본다)
        _direct_llm("claude-sonnet-5", thinking=thinking, max_tokens=1024).invoke("x")
        assert wire.requests[0]["body"]["thinking"] == expected

    def test_effort_gate_still_drops_effort_for_unsupported_models(self, wire):
        _direct_llm("claude-haiku-4-5", effort="high", max_tokens=1024).invoke("x")
        body = wire.requests[0]["body"]
        assert "output_config" not in body and "effort" not in body

    def test_unset_knobs_leave_the_request_untouched(self, wire):
        _direct_llm("claude-opus-5-5").invoke("x")
        body = wire.requests[0]["body"]
        assert not {"thinking", "output_config", "effort"} & set(body), sorted(body)


# ============================================================================ R3: SDK 비스트리밍 가드


@pytest.mark.unit
class TestNonStreamingGuard:
    """anthropic SDK 는 클라이언트 timeout 이 기본값(600s, connect 5s)일 때 예상 시간이
    10분을 넘는 비스트리밍 요청(max_tokens > ~21.3k)을 보내기 전에 ValueError
    ("Streaming is required ...")로 거절한다. ChatAnthropic 은 SDK 클라이언트를
    timeout=None 으로 만들므로 이 가드가 걸리지 않는다 — Vertex 경로와 같다."""

    GUARD = "Streaming is required"

    def test_installed_sdk_guard_rejects_32000_with_its_default_timeout(self, wire):
        """대조군: 설치된 SDK 에 가드가 살아 있다 (아래 시험이 헛돌지 않음을 보인다)."""
        import anthropic

        client = anthropic.Anthropic(api_key="placeholder")
        with pytest.raises(ValueError, match=self.GUARD):
            client.messages.create(model="claude-opus-5-5", max_tokens=32000,
                                   messages=[{"role": "user", "content": "x"}])
        assert wire.requests == []

    @pytest.mark.parametrize("model,effort", [("claude-opus-5-5", "xhigh"),
                                              ("claude-sonnet-5-5", "high")])
    def test_production_32000_non_streaming_request_is_sent(self, wire, model, effort):
        llm = _direct_llm(model, **_production_kwargs(effort))
        assert llm.streaming is False
        assert llm._client.timeout is None  # SDK 기본 timeout 이 아니다 -> 가드 없음
        llm.invoke("ping")
        (request,) = wire.requests
        assert request["body"]["max_tokens"] == 32000
        assert request["body"].get("stream") in (None, False)

    def test_structured_judge_call_is_sent_too(self, wire):
        wire.replies.append(TOOL_REPLY)
        llm = _direct_llm("claude-opus-5-5", **_production_kwargs("xhigh"))
        plan = llm.with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        assert len(wire.requests) == 1


# ============================================================================ main.py 의 모델


@pytest.mark.unit
@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_main_py_models_are_known_to_the_anthropic_catalog(model):
    """main.py 는 두 모델을 anthropic 공급자로 만든다. 카탈로그에 없으면 클라이언트를
    만들 때마다 stderr 에 'not in the known model list' 경고가 나온다."""
    import warnings

    from tradingagents.llm_clients.validators import validate_model

    assert validate_model("anthropic", model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _direct_llm(model)
