"""Structured output for Claude without a forced tool call.

Claude Opus 5.5 and Sonnet 5.5 answer a forced ``tool_choice`` (``{"type": "tool"}``
or ``"any"``) with a 400 -- 'tool_choice: type "tool" and "any" are not supported
for this model.' -- and that forced call is exactly what
``ChatAnthropicVertex.with_structured_output`` sends. The API's native
alternative, ``output_config.format``, fails on ``PortfolioDecision`` with
"Grammar compilation timed out" (tpmn-dev, 2026-10-01), so the schema stays a
tool: bound alone with ``tool_choice="auto"``, the model is told to answer
through it, and asked once more when its reply carries no call.

The tool definition and the parse (``Schema.model_validate`` on the call's
arguments) are the ones the forced path used, so the fork's fail-closed plan
validators see the same input. A call that fails validation is raised at once,
not re-asked: the caller's free-text fallback treats it as a miss, as before.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

ANSWER_THROUGH_TOOL = (
    "Give your final answer by calling the `{name}` tool exactly once, with every "
    "required field filled in. Do not answer in plain text."
)
CALL_THE_TOOL_NOW = (
    "Your reply did not call the `{name}` tool. Call it now with that answer."
)


def _as_messages(value: Any) -> list[BaseMessage]:
    if isinstance(value, PromptValue):
        return value.to_messages()
    if isinstance(value, str):
        return [HumanMessage(value)]
    return list(convert_to_messages(value))


def _call_to(message: Any, name: str) -> dict | None:
    for call in getattr(message, "tool_calls", None) or []:
        if call.get("name") == name:
            return call
    return None


# A reply that stopped here is not an answer: its thinking may have spent max_tokens
# (live, Opus 5.5 at effort max: thinking only, every time), a tool input may be a
# truncated prefix, or the model declined. Asking again ends the same way.
_CUT_SHORT = ("max_tokens", "refusal")


def _cut_short(message: Any, name: str) -> str | None:
    stop = (getattr(message, "response_metadata", None) or {}).get("stop_reason")
    if stop not in _CUT_SHORT:
        return None
    out = (getattr(message, "usage_metadata", None) or {}).get("output_tokens")
    return f"{name}: the reply stopped at {stop} (output tokens: {out}); not asked again"


def auto_tool_structured_output(llm: Any, schema: Any, *, include_raw: bool = False) -> Runnable:
    """``llm.with_structured_output(schema)`` without forcing the tool call.

    ``llm`` must implement ``bind_tools``. ``schema`` is a Pydantic model (the
    result is an instance) or a JSON-schema dict (the result is the arguments).
    With ``include_raw`` the result is ``{"raw", "parsed", "parsing_error"}`` and
    a miss is reported there instead of raised.
    """
    name = convert_to_openai_tool(schema)["function"]["name"]
    bound = llm.bind_tools([schema], tool_choice="auto")

    def _parse(call: dict) -> Any:
        if isinstance(schema, type) and hasattr(schema, "model_validate"):
            return schema.model_validate(call["args"])
        return call["args"]

    def _run(value: Any, config: RunnableConfig | None = None) -> Any:
        first = [*_as_messages(value), HumanMessage(ANSWER_THROUGH_TOOL.format(name=name))]
        raw = bound.invoke(first, config=config)
        miss = _cut_short(raw, name)
        call = None if miss else _call_to(raw, name)
        if call is None and miss is None:
            text = raw.content if isinstance(raw.content, str) else ""
            if text.strip() or getattr(raw, "tool_calls", None):
                # Answered in prose: ask it to put that answer through the tool. Reached
                # for another tool (one it was never given): ask afresh, as a call
                # cannot be replayed without a result.
                retry = ([*first, AIMessage(text), HumanMessage(CALL_THE_TOOL_NOW.format(name=name))]
                         if text.strip() else first)
                raw = bound.invoke(retry, config=config)
                cut = _cut_short(raw, name)
                call = None if cut else _call_to(raw, name)
                miss = cut or f"{name}: the model did not call the tool, also when asked again"
            else:
                # No text and no call: nothing to build on, and the same request ends
                # the same way, so the caller's fallback takes it from here.
                miss = f"{name}: the model returned no answer (no text, no tool call)"
        if not include_raw:
            if call is None:
                raise ValueError(miss)
            return _parse(call)
        try:
            if call is None:
                raise ValueError(miss)
            return {"raw": raw, "parsed": _parse(call), "parsing_error": None}
        except Exception as exc:  # noqa: BLE001 - reported, as LangChain's include_raw does
            return {"raw": raw, "parsed": None, "parsing_error": exc}

    return RunnableLambda(_run, name=f"{name}AutoToolStructuredOutput")
