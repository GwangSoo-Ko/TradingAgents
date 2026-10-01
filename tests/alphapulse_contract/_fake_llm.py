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
* structured calls: at the ``"sdk"`` boundary the fake ChatAnthropicVertex's own
  ``with_structured_output`` does what the real one does (forces the schema tool),
  and the fork's ``NormalizedChatAnthropicVertex`` override replaces it with
  ``tradingagents.llm_clients.claude_structured`` — so losing that override shows up
  as the 400 below. At the ``"factory"`` boundary no fork client wraps the fake, so
  ``HarnessChatModel.with_structured_output`` runs that same fork path itself: the
  schema bound as the only tool with ``tool_choice="auto"``, an
  answer-through-the-tool instruction, one re-ask on a prose reply, then
  ``Schema.model_validate(args)`` — the fork's real validators/coercers run, and a
  ValidationError propagates exactly like the real parser's, so the fork's real
  free-text fallback runs. A request whose only tool is a Pydantic schema is
  scripted from ``structured``: a raw dict (the tool args), ``"fail"`` (the model
  answers in prose, no tool call) or ``"raise"`` (a provider error).
* like the real Claude Opus 5.5 / Sonnet 5.5, a request that forces a tool call
  (``tool_choice`` ``{"type": "tool"}`` / ``"any"`` / a tool name) on one of those
  models is answered with the API's 400 — so a path that forces one breaks the run
  here before it breaks production.
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
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from . import _runtime as rt

_MAX_TOOL_ROUNDS = 8

# Models whose API answers a forced tool call with a 400 (live on Vertex, 2026-10-01).
REJECTS_FORCED_TOOL_CHOICE = frozenset(
    {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"})
FORCED_TOOL_CHOICE_ERROR = 'tool_choice: type "tool" and "any" are not supported for this model.'


class HarnessBadRequest(RuntimeError):
    """The 400 the real API returns (invalid_request_error)."""


def _forces_a_tool(tool_choice: Any) -> bool:
    if tool_choice in (None, "auto", "none"):
        return False
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") in ("tool", "any")
    return True  # "any", or a tool name (LangChain's shorthand for {"type": "tool"})


def _schema_tool(tools: list[Any]) -> str | None:
    """The schema name when the request's only tool is a Pydantic schema (a structured call)."""
    if len(tools) == 1 and isinstance(tools[0], type) and issubclass(tools[0], BaseModel):
        return tools[0].__name__
    return None


def model_of(llm_id: int) -> str | None:
    """The model name the LLM with this construction id was built for."""
    records = rt.CAPTURES["llm_constructions"]
    if not 0 <= llm_id < len(records):
        return None
    rec = records[llm_id]
    return rec.get("model") or (rec.get("kwargs") or {}).get("model_name")


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

    def _record_binding(self, schema: Any, include_raw: bool, method: Any,
                        kwargs: dict[str, Any]) -> None:
        with rt._lock:
            rt.CAPTURES["structured_bindings"].append(rt.json_safe({
                "llm_id": self.harness_llm_id, "schema": _tool_name(schema), "method": method,
                "include_raw": include_raw, "kwargs": kwargs,
            }))

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,
                               method: Any = None, **kwargs: Any):
        """Factory boundary: no fork client wraps this model, so it answers like the
        object production builds (NormalizedChatAnthropicVertex): the fork's auto-tool path."""
        self._record_binding(schema, include_raw, method, kwargs)
        # Imported here, not at module level: importing ``tradingagents`` runs its
        # load_dotenv(), which must happen when main.py imports it, not at harness boot.
        from tradingagents.llm_clients.claude_structured import auto_tool_structured_output

        return auto_tool_structured_output(self, schema, include_raw=include_raw)

    # -- scripted generation ------------------------------------------------------
    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        top, inner = rt.current_node()
        role = rt.role_for_node(top) if top else rt.role_outside_graph()
        if top is not None:
            rt.note_node_config(top)
        tools = kwargs.get("tools") or []
        tool_names = [_tool_name(t) for t in tools]
        schema = _schema_tool(tools)
        kind = "structured" if schema else ("tools" if tools else "chat")
        script = rt.role_script(role)
        index = rt.bump_counter(role, kind)
        record: dict[str, Any] = {
            "seq": rt.next_seq(), "node": top, "inner_node": inner, "role": role,
            "llm_id": self.harness_llm_id, "kind": kind, "schema": schema,
            "tools": tool_names if kind == "tools" else None, "call_index": index,
            "tool_choice": kwargs.get("tool_choice"),
            "messages": [_message_record(m) for m in messages],
        }
        try:
            if (tools and _forces_a_tool(kwargs.get("tool_choice"))
                    and model_of(self.harness_llm_id) in REJECTS_FORCED_TOOL_CHOICE):
                raise HarnessBadRequest(f"Error code: 400 - {FORCED_TOOL_CHOICE_ERROR}")
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

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,
                               method: Any = None, **kwargs: Any):
        """What the real ChatAnthropicVertex does: force the schema tool, parse the first
        call. The 5.5 models answer that with the 400, so at this boundary only the
        fork's own override (``_ClaudeStructuredOutput``) keeps a structured call alive."""
        self._record_binding(schema, include_raw, method, kwargs)
        bound = self.bind_tools([schema], tool_choice=_tool_name(schema))

        def _first_call(message: Any) -> Any:
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                return None
            if isinstance(schema, type) and hasattr(schema, "model_validate"):
                return schema.model_validate(calls[0].get("args"))
            return calls[0].get("args")

        return bound | RunnableLambda(_first_call)


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
