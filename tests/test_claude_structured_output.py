"""Structured output for Claude without a forced tool call.

Claude Opus 5.5 / Sonnet 5.5 answer a forced ``tool_choice`` with a 400
('tool_choice: type "tool" and "any" are not supported for this model.'), which is
the call ``ChatAnthropicVertex.with_structured_output`` makes. The fork binds the
schema as the only tool with ``tool_choice="auto"``, tells the model to answer
through it, and asks once more when a reply carries no call. These tests pin that
path with a scripted chat model — no SDK, no network.
"""
import sys
import types

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, ValidationError

from tradingagents.agents.schemas import PortfolioRating, ResearchPlan

PLAN_ARGS = {
    "recommendation": "Overweight",
    "rationale": "Services margin expansion outweighs the valuation premium.",
    "strategic_actions": "Build to 1.2x a standard allocation over two tranches.",
}


def _call(args, name="ResearchPlan", call_id="toolu_1"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedClaude(BaseChatModel):
    """A chat model that replays scripted replies and records every request."""

    model_config = ConfigDict(extra="allow")
    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-claude"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": kwargs.get("tools"),
                           "tool_choice": kwargs.get("tool_choice")})
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def _structured(replies, **kw):
    from tradingagents.llm_clients.claude_structured import auto_tool_structured_output
    llm = ScriptedClaude(replies=list(replies))
    return llm, auto_tool_structured_output(llm, ResearchPlan, **kw)


@pytest.mark.unit
class TestAutoToolStructuredOutput:
    def test_binds_the_schema_alone_and_never_forces_the_call(self):
        llm, runnable = _structured([_call(PLAN_ARGS)])
        runnable.invoke("Weigh the debate.")
        (call,) = llm.calls
        assert call["tools"] == [ResearchPlan]
        assert call["tool_choice"] == "auto"

    def test_parses_the_tool_call_through_the_schema(self):
        _, runnable = _structured([_call(PLAN_ARGS)])
        plan = runnable.invoke("Weigh the debate.")
        assert isinstance(plan, ResearchPlan)
        assert plan.recommendation is PortfolioRating.OVERWEIGHT

    def test_keeps_the_prompt_and_tells_the_model_to_answer_through_the_tool(self):
        llm, runnable = _structured([_call(PLAN_ARGS)])
        runnable.invoke("Weigh the debate.")
        messages = llm.calls[0]["messages"]
        assert messages[0] == HumanMessage("Weigh the debate.")
        assert isinstance(messages[-1], HumanMessage)
        assert "`ResearchPlan`" in messages[-1].content

    def test_accepts_a_message_list_prompt(self):
        llm, runnable = _structured([_call(PLAN_ARGS)])
        runnable.invoke([{"role": "system", "content": "You are the RM."},
                         {"role": "user", "content": "Weigh the debate."}])
        messages = llm.calls[0]["messages"]
        assert [m.type for m in messages] == ["system", "human", "human"]

    def test_asks_once_more_when_the_reply_has_no_tool_call(self):
        llm, runnable = _structured([AIMessage("I recommend Overweight."), _call(PLAN_ARGS)])
        plan = runnable.invoke("Weigh the debate.")
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        assert len(llm.calls) == 2
        retry = llm.calls[1]["messages"]
        assert retry[-2] == AIMessage("I recommend Overweight.")
        assert isinstance(retry[-1], HumanMessage) and "`ResearchPlan`" in retry[-1].content
        assert llm.calls[1]["tool_choice"] == "auto"

    def test_an_empty_reply_is_not_asked_again(self):
        # No text and no call: the thinking spent max_tokens, or the model declined.
        # The same request at the same effort ends the same way (live, Opus 5.5 at
        # effort max: ~200 s per attempt), so it goes straight to the caller's fallback.
        llm, runnable = _structured([AIMessage(""), _call(PLAN_ARGS)])
        with pytest.raises(ValueError, match="no answer"):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 1

    def test_a_reply_cut_off_at_max_tokens_is_a_miss_without_a_re_ask(self):
        # Live, Opus 5.5 at effort max: every judge reply was thinking only, stopped at
        # max_tokens. Asking again at the same cap ends the same way.
        cut = AIMessage("", response_metadata={"stop_reason": "max_tokens"},
                        usage_metadata={"input_tokens": 14000, "output_tokens": 20000,
                                        "total_tokens": 34000})
        llm, runnable = _structured([cut, _call(PLAN_ARGS)])
        with pytest.raises(ValueError, match=r"max_tokens.*20000"):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 1

    def test_a_tool_call_cut_off_at_max_tokens_is_not_used(self):
        # Its arguments may be a truncated prefix that happens to validate: a plan with
        # tranches missing must not reach TRADE_PLAN_JSON.
        cut = _call(PLAN_ARGS)
        cut.response_metadata = {"stop_reason": "max_tokens"}
        llm, runnable = _structured([cut])
        with pytest.raises(ValueError, match="max_tokens"):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 1

    def test_a_refusal_is_a_miss_without_a_re_ask(self):
        refused = AIMessage("", response_metadata={"stop_reason": "refusal"})
        llm, runnable = _structured([refused, _call(PLAN_ARGS)])
        with pytest.raises(ValueError, match="refusal"):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 1

    def test_raises_when_the_second_reply_still_has_no_tool_call(self):
        llm, runnable = _structured([AIMessage("Overweight."), AIMessage("Still prose.")])
        with pytest.raises(ValueError, match="ResearchPlan"):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 2

    def test_a_call_that_fails_validation_raises_without_a_retry(self):
        # Fail-closed: an unreadable answer is a miss for the caller's free-text
        # fallback, not something to re-ask until it passes.
        llm, runnable = _structured([_call({**PLAN_ARGS, "recommendation": "Strong Buy"})])
        with pytest.raises(ValidationError):
            runnable.invoke("Weigh the debate.")
        assert len(llm.calls) == 1

    def test_ignores_a_call_to_another_tool(self):
        llm, runnable = _structured([_call({"query": "AAPL"}, name="web_search"), _call(PLAN_ARGS)])
        assert runnable.invoke("Weigh the debate.").recommendation is PortfolioRating.OVERWEIGHT
        assert len(llm.calls) == 2

    def test_include_raw_reports_raw_parsed_and_error(self):
        _, runnable = _structured([_call(PLAN_ARGS)], include_raw=True)
        out = runnable.invoke("Weigh the debate.")
        assert isinstance(out["raw"], AIMessage) and out["raw"].tool_calls
        assert isinstance(out["parsed"], ResearchPlan) and out["parsing_error"] is None

        _, runnable = _structured([AIMessage("a"), AIMessage("b")], include_raw=True)
        out = runnable.invoke("Weigh the debate.")
        assert out["parsed"] is None and isinstance(out["parsing_error"], ValueError)

    def test_the_freetext_fallback_still_runs_after_a_miss(self):
        from tradingagents.agents.structured import invoke_structured_or_freetext
        llm, runnable = _structured([AIMessage("a"), AIMessage("b")])
        plain = ScriptedClaude(replies=[AIMessage("free text plan")])
        text, obj = invoke_structured_or_freetext(runnable, plain, "Weigh the debate.",
                                                  lambda p: p.rationale, "Research Manager")
        assert (text, obj) == ("free text plan", None)


def _install_scripted_vertex(monkeypatch):
    built = {}

    class ChatAnthropicVertex(ScriptedClaude):  # the real class's name: Normalized<name>
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built["kwargs"] = kwargs
            built["llm"] = self

    garden = types.ModuleType("langchain_google_vertexai.model_garden")
    garden.ChatAnthropicVertex = ChatAnthropicVertex
    pkg = types.ModuleType("langchain_google_vertexai")
    pkg.model_garden = garden
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", pkg)
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai.model_garden", garden)
    return built


@pytest.mark.unit
class TestVertexClaudeStructuredOutput:
    def test_vertex_claude_structured_output_never_forces_the_tool(self, monkeypatch):
        built = _install_scripted_vertex(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        llm = VertexAnthropicClient("claude-opus-5-5", project="p", location="global",
                                    effort="max", thinking="adaptive").get_llm()
        llm.replies.append(_call(PLAN_ARGS))
        plan = llm.with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        (call,) = built["llm"].calls
        assert call["tool_choice"] == "auto"
        # effort/thinking still travel in model_kwargs, untouched by the binding
        assert built["kwargs"]["model_kwargs"] == {
            "output_config": {"effort": "max"}, "thinking": {"type": "adaptive"}}

    def test_real_chat_anthropic_vertex_request_lets_the_model_choose_the_tool(self, monkeypatch):
        """The request the real SDK sends: tool_choice auto, effort/thinking intact."""
        garden = pytest.importorskip("langchain_google_vertexai.model_garden")
        import json

        import anthropic
        http = pytest.importorskip("httpx2" if anthropic.__version__.startswith("1.") else "httpx")
        from tradingagents.llm_clients.vertex_clients import (
            _ClaudeStructuredOutput,
            _normalized_subclass,
        )

        bodies = []
        reply = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                 "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                             {"type": "tool_use", "id": "toolu_1", "name": "ResearchPlan",
                              "input": PLAN_ARGS}],
                 "stop_reason": "tool_use", "stop_sequence": None,
                 "usage": {"input_tokens": 1, "output_tokens": 1}}

        def handle_request(self, request):
            bodies.append(json.loads(request.read().decode("utf-8")))
            return http.Response(200, json=reply, request=request)

        monkeypatch.setattr(http.HTTPTransport, "handle_request", handle_request)
        cls = _normalized_subclass(garden.ChatAnthropicVertex, _ClaudeStructuredOutput)
        llm = cls(model_name="claude-opus-5-5", project="tpmn-dev", location="global",
                  max_tokens=20000, access_token="test-token",
                  model_kwargs={"output_config": {"effort": "max"},
                                "thinking": {"type": "adaptive"}})
        plan = llm.with_structured_output(ResearchPlan).invoke("Weigh the debate.")
        assert plan.recommendation is PortfolioRating.OVERWEIGHT
        (body,) = bodies
        assert body["tool_choice"] == {"type": "auto"}
        assert [t["name"] for t in body["tools"]] == ["ResearchPlan"]
        assert body["output_config"] == {"effort": "max"}
        assert body["thinking"] == {"type": "adaptive"}
        assert body["max_tokens"] == 20000

    def test_vertex_claude_keeps_normalizing_invoke_output(self, monkeypatch):
        _install_scripted_vertex(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        llm = VertexAnthropicClient("claude-sonnet-5-5", project="p", location="global").get_llm()
        llm.replies.append(AIMessage(content=[{"type": "thinking", "thinking": ""},
                                              {"type": "text", "text": "hi"}]))
        assert llm.invoke("x").content == "hi"
        assert type(llm).__name__ == "NormalizedChatAnthropicVertex"
