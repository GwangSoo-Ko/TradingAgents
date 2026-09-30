"""Scripted chat model for the harness subprocess — the only LLM the fork ever sees.

``HarnessChatModel`` is a real langchain ``BaseChatModel``: ``invoke``, message
coercion, ``bind``/``RunnableBinding`` and callbacks are langchain's own code.
Only ``_generate`` is scripted. Every call is attributed to the LangGraph node
that made it (via the runnable-config contextvar LangGraph sets per task) and
recorded in ``captures.llm_calls`` with the exact messages the node sent.

Replies come from the scenario's ``roles`` scripts:

* analyst tool loops: ``tool_rounds[i]`` is returned as an AIMessage with
  ``tool_calls`` while ``i`` (= AI tool-call messages already in the prompt) is in
  range, filtered to the tools the node actually bound; then ``text``.
* structured calls: ``with_structured_output(Schema)`` mirrors
  ``ChatAnthropicVertex.with_structured_output`` (the production path): bind the
  schema as the only tool, then parse the first tool call with
  ``Schema.model_validate(args)`` — the fork's real validators/coercers run, and a
  ValidationError propagates exactly like the real parser's, so the fork's real
  free-text fallback runs. ``structured`` may be a raw dict (the tool args),
  ``"fail"`` (the model answers in prose, no tool call -> parser returns None) or
  ``"raise"`` (a provider error).
* anything else: ``text`` (a string, or a list consumed call by call).

Two installation boundaries (scenario ``llm_boundary``):

* ``"factory"`` (default): ``create_llm_client`` is replaced wherever the fork
  bound it; the construction record holds (provider, model, base_url, kwargs) as
  the graph passed them.
* ``"sdk"``: ``langchain_google_vertexai.model_garden.ChatAnthropicVertex`` is
  replaced, so the fork's real client classes translate the config into SDK
  kwargs; the record holds the SDK constructor kwargs. This fake SDK module is
  installed in both modes as a backstop, so a code path that bypasses the
  factory still cannot reach the network.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough

from . import _runtime as rt

_MAX_TOOL_ROUNDS = 8


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            return str(tool["function"].get("name"))
        return str(tool.get("name") or tool.get("title") or "?")
    if isinstance(tool, type):
        return tool.__name__
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or repr(tool))


def _message_record(m: BaseMessage) -> dict[str, Any]:
    rec: dict[str, Any] = {"type": m.type, "content": m.content}
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        rec["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                             for tc in tool_calls]
    for attr in ("tool_call_id", "name"):
        val = getattr(m, attr, None)
        if val:
            rec[attr] = val
    return rec


def record_construction(boundary: str, **fields: Any) -> int:
    with rt._lock:
        rec_id = len(rt.CAPTURES["llm_constructions"])
        rt.CAPTURES["llm_constructions"].append(
            rt.json_safe({"id": rec_id, "boundary": boundary, **fields, "stack": rt.short_stack(6)})
        )
    return rec_id


class HarnessChatModel(BaseChatModel):
    """Deterministic, scripted chat model (see module docstring)."""

    harness_llm_id: int = -1

    @property
    def _llm_type(self) -> str:
        return "alphapulse-contract-harness"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"harness_llm_id": self.harness_llm_id}

    # -- langchain surface ------------------------------------------------------
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,
                               method: Any = None, **kwargs: Any):
        name = _tool_name(schema)
        with rt._lock:
            rt.CAPTURES["structured_bindings"].append(rt.json_safe({
                "llm_id": self.harness_llm_id, "schema": name, "method": method,
                "include_raw": include_raw, "kwargs": kwargs,
            }))
        bound = self.bind(tools=[schema], tool_choice=name, harness_structured=name)

        def _parse(message: Any) -> Any:
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return None
            args = tool_calls[0].get("args")
            if isinstance(schema, type) and hasattr(schema, "model_validate"):
                return schema.model_validate(args)
            return args

        parser = RunnableLambda(_parse)
        if include_raw:
            assign = RunnablePassthrough.assign(parsed=lambda d: _parse(d["raw"]),
                                                parsing_error=lambda _: None)
            fallback = RunnablePassthrough.assign(parsed=lambda _: None)
            return RunnableMap(raw=bound) | assign.with_fallbacks([fallback],
                                                                   exception_key="parsing_error")
        return bound | parser

    # -- scripted generation ------------------------------------------------------
    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        top, inner = rt.current_node()
        role = rt.role_for_node(top) if top else rt.role_outside_graph()
        if top is not None:
            rt.note_node_config(top)
        tools = kwargs.get("tools") or []
        tool_names = [_tool_name(t) for t in tools]
        schema = kwargs.get("harness_structured")
        kind = "structured" if schema else ("tools" if tools else "chat")
        script = rt.role_script(role)
        index = rt.bump_counter(role, kind)
        record: dict[str, Any] = {
            "seq": rt.next_seq(), "node": top, "inner_node": inner, "role": role,
            "llm_id": self.harness_llm_id, "kind": kind, "schema": schema,
            "tools": tool_names if kind == "tools" else None, "call_index": index,
            "messages": [_message_record(m) for m in messages],
        }
        try:
            reply = self._reply(kind, script, schema, tool_names, messages, index, record)
        except Exception as exc:
            record["reply"] = {"raised": f"{type(exc).__name__}: {exc}"}
            with rt._lock:
                rt.CAPTURES["llm_calls"].append(rt.json_safe(record))
            raise
        record["reply"] = _message_record(reply)
        with rt._lock:
            rt.CAPTURES["llm_calls"].append(rt.json_safe(record))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _reply(self, kind: str, script: dict[str, Any], schema: str | None,
               tool_names: list[str], messages: list[BaseMessage], index: int,
               record: dict[str, Any]) -> AIMessage:
        seq = record["seq"]
        if kind == "structured":
            spec = script.get("structured", "fail")
            if spec == "raise":
                raise RuntimeError(f"harness: scripted provider error for {schema}")
            if isinstance(spec, dict):
                return AIMessage(content="", tool_calls=[
                    {"name": schema, "args": rt.render(spec), "id": f"toolu_harness_{seq}",
                     "type": "tool_call"}])
            return AIMessage(content=rt.render(self._text(script, "structured_fail_text", 0)
                                               or "(harness) answered in prose instead of the tool"))
        if kind == "tools":
            rounds = script.get("tool_rounds") or []
            done = sum(1 for m in messages if m.type == "ai" and getattr(m, "tool_calls", None))
            if done < len(rounds) and done < _MAX_TOOL_ROUNDS:
                wanted = rounds[done]
                calls = [c for c in wanted if c.get("name") in tool_names]
                skipped = [c.get("name") for c in wanted if c.get("name") not in tool_names]
                if skipped:
                    record["skipped_tool_calls"] = skipped
                if calls:
                    return AIMessage(content="", tool_calls=[
                        {"name": c["name"], "args": rt.render(c.get("args") or {}),
                         "id": f"call_harness_{seq}_{i}", "type": "tool_call"}
                        for i, c in enumerate(calls)])
            index = rt.bump_counter(record["role"], "tools-final") if done else index
        return AIMessage(content=rt.render(self._text(script, "text", index)
                                           or f"(harness) no scripted reply for {record['role']}"))

    @staticmethod
    def _text(script: dict[str, Any], key: str, index: int) -> str | None:
        value = script.get(key)
        if isinstance(value, list):
            return value[min(index, len(value) - 1)] if value else None
        return value


# ============================================================================ boundaries

class _FakeClient:
    """What ``create_llm_client`` returns: a client whose ``get_llm`` is scripted."""

    def __init__(self, rec_id: int, provider: str, model: str, base_url: Any, kwargs: dict):
        self.rec_id = rec_id
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_llm(self) -> HarnessChatModel:
        return HarnessChatModel(harness_llm_id=self.rec_id)

    def validate_model(self) -> bool:
        return True

    def warn_if_unknown_model(self) -> None:
        return None

    def get_provider_name(self) -> str:
        return str(self.provider)


def fake_create_llm_client(provider: str, model: str | None = None, base_url: Any = None,
                           **kwargs: Any) -> _FakeClient:
    rec_id = record_construction("factory", provider=provider, model=model, base_url=base_url,
                                 kwargs=kwargs)
    return _FakeClient(rec_id, provider, model or "", base_url, kwargs)


class FakeChatAnthropicVertex(HarnessChatModel):
    """Stand-in for langchain_google_vertexai.model_garden.ChatAnthropicVertex."""

    def __init__(self, **kwargs: Any) -> None:
        rec_id = record_construction("sdk", sdk_class="ChatAnthropicVertex", kwargs=kwargs)
        passthrough = {k: kwargs[k] for k in ("callbacks", "tags", "metadata") if k in kwargs}
        super().__init__(harness_llm_id=rec_id, **passthrough)


def install_sdk_fakes() -> None:
    """Put fake ``langchain_google_vertexai`` modules in sys.modules (both modes)."""
    pkg = types.ModuleType("langchain_google_vertexai")
    pkg.__path__ = []  # a package, so ``import langchain_google_vertexai.model_garden`` resolves
    garden = types.ModuleType("langchain_google_vertexai.model_garden")
    garden.ChatAnthropicVertex = FakeChatAnthropicVertex
    pkg.model_garden = garden
    pkg.__harness_fake__ = True
    sys.modules["langchain_google_vertexai"] = pkg
    sys.modules["langchain_google_vertexai.model_garden"] = garden


def install_factory_fake() -> list[str]:
    """Replace create_llm_client in every already-imported fork module that bound it."""
    import importlib

    factory = importlib.import_module("tradingagents.llm_clients.factory")
    real = factory.create_llm_client
    patched = []
    for name, mod in list(sys.modules.items()):
        if not (name in ("tradingagents", "cli") or name.startswith(("tradingagents.", "cli."))):
            continue
        if getattr(mod, "create_llm_client", None) is real:
            mod.create_llm_client = fake_create_llm_client
            patched.append(name)
    return patched
