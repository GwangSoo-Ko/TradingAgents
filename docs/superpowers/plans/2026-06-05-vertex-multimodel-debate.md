# Vertex Model Garden Multi-Model Debate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing debate roles run on three different Vertex-hosted foundation models (Gemini / Claude / Grok) instead of one model playing every side, reached exclusively through Vertex AI Model Garden.

**Architecture:** Add a `vertex_*` provider layer to the LLM client factory (3 small client classes + a shared Vertex auth helper, all lazy-imported), generalize the hardcoded quick/deep tier wiring into a `role_models`-driven per-role LLM resolver with client dedup (confined to `trading_graph.py` + `setup.py`, node factory signatures unchanged), and expose it via a CLI "Vertex Model Garden" preset. When `role_models` is unset the system is byte-for-byte identical to today.

**Tech Stack:** Python ≥3.10, LangChain (`langchain-google-vertexai`, `langchain-openai`), LangGraph, pytest, Google ADC auth.

**Spec:** `docs/superpowers/specs/2026-06-05-vertex-multimodel-debate-design.md`

**Branch:** `feature/vertex-multimodel-debate` (already created; the spec is committed there). Commit each task locally. Do **not** push to origin until the user asks.

**Test command (no Vertex SDK needed — all mocked):**
```bash
DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest -q
```

**Key design decisions (read before starting):**
- Vertex clients forward a **minimal, SDK-safe kwarg set** in v1 (`temperature`, `max_retries`, `callbacks`, plus `max_tokens` for Claude). Thinking-config (`thinking_level`/`effort`) param names for the Vertex SDK are **deferred** to live verification — do not forward them.
- Vertex clients' `validate_model()` returns `True` (any model accepted, like ollama/openrouter). No `model_catalog.py` change.
- Client **dedup** keys on `(provider, model, location, kwargs)`, so a full run builds exactly **3** Vertex clients: one Gemini (bull + conservative + analysts + trader), one Grok (bear + aggressive, single OAuth token), one Claude (neutral + both judges).
- Grok is reached via `ChatOpenAI` pointed at the Vertex `endpoints/openapi` URL with a Google OAuth access token as `api_key`.

---

## File Structure

**Create:**
- `tradingagents/llm_clients/vertex_auth.py` — Vertex auth/config helpers (lazy `google.*` imports).
- `tradingagents/llm_clients/vertex_clients.py` — `VertexGeminiClient`, `VertexAnthropicClient`, `VertexGrokClient`.
- `cli/presets.py` — `VERTEX_DEBATE_PRESET` + `apply_vertex_multimodel_config()`.
- `tests/test_vertex_clients.py`, `tests/test_role_model_resolver.py`, `tests/test_vertex_cli_preset.py`.

**Modify:**
- `tradingagents/llm_clients/factory.py` — dispatch `vertex_*` providers.
- `tradingagents/default_config.py` — add `role_models`, `vertex_project`, `vertex_location`.
- `tradingagents/graph/trading_graph.py` — `DEEP_ROLES`/`ROLE_KEYS` constants + resolver methods; replace 2-LLM construction and the `GraphSetup` call.
- `tradingagents/graph/setup.py` — `GraphSetup` takes `llm_for` and wires each node per role.
- `cli/utils.py` — provider-table row + `ask_vertex_config()`.
- `cli/main.py` — Vertex branch in `get_user_selections` + `apply_vertex_multimodel_config` in `run_analysis`.
- `CLAUDE.md` — document the Vertex multi-model mode + required env/auth.

---

## Task 1: Vertex auth helper

**Files:**
- Create: `tradingagents/llm_clients/vertex_auth.py`
- Test: `tests/test_vertex_clients.py`

- [ ] **Step 1: Write the failing tests** (create `tests/test_vertex_clients.py` with just the auth tests for now)

```python
"""Vertex provider layer: auth helper + the three Vertex client classes.

All Vertex SDK imports are lazy, so these tests install lightweight fake
modules / mock the token fetch — no langchain-google-vertexai, google-auth,
or network required.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestVertexAuth:
    def test_openapi_base_url_global_drops_location_prefix(self):
        from tradingagents.llm_clients import vertex_auth
        assert vertex_auth.openapi_base_url("tpmn-dev", "global") == (
            "https://aiplatform.googleapis.com/v1/projects/tpmn-dev/"
            "locations/global/endpoints/openapi"
        )

    def test_openapi_base_url_regional_keeps_prefix(self):
        from tradingagents.llm_clients import vertex_auth
        assert vertex_auth.openapi_base_url("p", "us-east5") == (
            "https://us-east5-aiplatform.googleapis.com/v1/projects/p/"
            "locations/us-east5/endpoints/openapi"
        )

    def test_resolve_location_prefers_explicit_then_env_then_global(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        assert vertex_auth.resolve_location(None) == "global"
        assert vertex_auth.resolve_location("us-central1") == "us-central1"
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        assert vertex_auth.resolve_location(None) == "europe-west1"

    def test_resolve_project_prefers_explicit_then_env(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "envproj")
        assert vertex_auth.resolve_project(None) == "envproj"
        assert vertex_auth.resolve_project("explicit") == "explicit"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_clients.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.llm_clients.vertex_auth'`

- [ ] **Step 3: Create `tradingagents/llm_clients/vertex_auth.py`**

```python
"""Shared authentication and endpoint helpers for Vertex AI Model Garden.

All `google.*` imports are function-local (lazy) so importing this module never
pulls google-auth / the Vertex SDK during test collection or in environments
that do not use Vertex. Authentication is via Google Application Default
Credentials (ADC: `gcloud auth application-default login`) or a service-account
JSON pointed at by GOOGLE_APPLICATION_CREDENTIALS — never a vendor API key.
"""

from __future__ import annotations

import os
from typing import Optional

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_LOCATION = "global"


def resolve_project(explicit: Optional[str]) -> Optional[str]:
    """Vertex project: explicit arg, else GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT."""
    return (
        explicit
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )


def resolve_location(explicit: Optional[str]) -> str:
    """Vertex location: explicit arg, else GOOGLE_CLOUD_LOCATION, else 'global'."""
    return explicit or os.environ.get("GOOGLE_CLOUD_LOCATION") or _DEFAULT_LOCATION


def get_access_token(credentials=None) -> str:
    """Return a fresh Google OAuth access token for the OpenAI-compatible path.

    Used only by the Grok client (Vertex MaaS via the OpenAI-compatible
    endpoint), which passes the token as the OpenAI ``api_key``. Tokens expire
    in ~1 hour; this fetches/refreshes one at client-build time.
    """
    import google.auth
    import google.auth.transport.requests

    if credentials is None:
        credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def openapi_base_url(project: str, location: str) -> str:
    """Build the Vertex OpenAI-compatible chat-completions base URL.

    The ``global`` endpoint uses the host ``aiplatform.googleapis.com`` with no
    ``{location}-`` prefix; regional endpoints prepend the region.
    """
    host = (
        "https://aiplatform.googleapis.com"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com"
    )
    return f"{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_clients.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/vertex_auth.py tests/test_vertex_clients.py
git commit -m "feat(vertex): add Vertex AI auth + endpoint helper (ADC, openapi base url)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Vertex client classes + factory dispatch

**Files:**
- Create: `tradingagents/llm_clients/vertex_clients.py`
- Modify: `tradingagents/llm_clients/factory.py` (after line 49, the `google` branch)
- Test: `tests/test_vertex_clients.py` (append)

- [ ] **Step 1: Append the failing client tests to `tests/test_vertex_clients.py`**

```python
def _install_fake_vertexai(monkeypatch):
    """Install lightweight fake langchain-google-vertexai modules.

    Real classes (not MagicMocks) so the Normalized* subclasses can subclass
    them. Each records its constructor kwargs and returns a normalizable
    response from invoke().
    """
    captured = {}

    mod = types.ModuleType("langchain_google_vertexai")

    class FakeChatVertexAI:
        def __init__(self, **kw):
            captured["gemini"] = kw

        def invoke(self, input, config=None, **kw):
            r = MagicMock()
            r.content = [{"type": "text", "text": "hi"}]
            return r

    mod.ChatVertexAI = FakeChatVertexAI

    sub = types.ModuleType("langchain_google_vertexai.model_garden")

    class FakeChatAnthropicVertex:
        def __init__(self, **kw):
            captured["claude"] = kw

        def invoke(self, input, config=None, **kw):
            r = MagicMock()
            r.content = "hi"
            return r

    sub.ChatAnthropicVertex = FakeChatAnthropicVertex
    mod.model_garden = sub

    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", mod)
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai.model_garden", sub)
    return captured


@pytest.mark.unit
class TestVertexClients:
    def test_gemini_client_builds_with_project_and_location(self, monkeypatch):
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexGeminiClient
        llm = VertexGeminiClient(
            "gemini-3.5-flash", project="tpmn-dev", location="global"
        ).get_llm()
        assert captured["gemini"]["model"] == "gemini-3.5-flash"
        assert captured["gemini"]["project"] == "tpmn-dev"
        assert captured["gemini"]["location"] == "global"
        # invoke() normalizes list-of-blocks content to a string
        assert llm.invoke("x").content == "hi"

    def test_gemini_client_forwards_only_safe_kwargs(self, monkeypatch):
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexGeminiClient
        VertexGeminiClient(
            "gemini-3.5-flash", project="p", location="global",
            temperature=0.0, thinking_level="high",  # thinking_level must be dropped in v1
        ).get_llm()
        assert captured["gemini"]["temperature"] == 0.0
        assert "thinking_level" not in captured["gemini"]
        assert "thinking_budget" not in captured["gemini"]

    def test_anthropic_client_builds_with_model_name(self, monkeypatch):
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient(
            "claude-opus-4-8", project="tpmn-dev", location="global"
        ).get_llm()
        assert captured["claude"]["model_name"] == "claude-opus-4-8"
        assert captured["claude"]["project"] == "tpmn-dev"
        assert captured["claude"]["location"] == "global"

    def test_grok_client_builds_openai_compat_with_token(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.setattr(
            vertex_auth, "get_access_token", lambda credentials=None: "FAKE_TOKEN"
        )
        from tradingagents.llm_clients.vertex_clients import VertexGrokClient
        llm = VertexGrokClient(
            "xai/grok-4.20-reasoning", project="tpmn-dev", location="global"
        ).get_llm()
        assert llm.model_name == "xai/grok-4.20-reasoning"
        base = str(llm.openai_api_base)
        assert "endpoints/openapi" in base
        assert base.startswith("https://aiplatform.googleapis.com/")  # global host
        assert llm.openai_api_key.get_secret_value() == "FAKE_TOKEN"

    def test_validate_model_accepts_any(self, monkeypatch):
        _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import (
            VertexGeminiClient, VertexAnthropicClient, VertexGrokClient,
        )
        assert VertexGeminiClient("anything").validate_model() is True
        assert VertexAnthropicClient("anything").validate_model() is True
        assert VertexGrokClient("anything").validate_model() is True

    def test_factory_dispatches_vertex_providers(self, monkeypatch):
        _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.factory import create_llm_client
        from tradingagents.llm_clients.vertex_clients import (
            VertexGeminiClient, VertexAnthropicClient, VertexGrokClient,
        )
        assert isinstance(
            create_llm_client("vertex_gemini", "gemini-3.5-flash", project="p"),
            VertexGeminiClient,
        )
        assert isinstance(
            create_llm_client("vertex_anthropic", "claude-opus-4-8", project="p"),
            VertexAnthropicClient,
        )
        assert isinstance(
            create_llm_client("vertex_grok", "xai/grok-4.20-reasoning", project="p"),
            VertexGrokClient,
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_clients.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.llm_clients.vertex_clients'`

- [ ] **Step 3: Create `tradingagents/llm_clients/vertex_clients.py`**

```python
"""LLM clients for foundation models hosted on Vertex AI Model Garden.

Three providers, two integration patterns:
  - vertex_gemini   -> ChatVertexAI            (langchain_google_vertexai)
  - vertex_anthropic -> ChatAnthropicVertex    (langchain_google_vertexai.model_garden)
  - vertex_grok     -> ChatOpenAI against the Vertex OpenAI-compatible endpoint
                       (langchain_openai), Google OAuth token as api_key

All SDK imports are lazy (inside get_llm) so the default install / test suite
runs without the optional ``[vertex]`` extra. v1 forwards a minimal, SDK-safe
kwarg set; thinking-config (thinking_level / effort) plumbing for the Vertex SDK
is deferred until param names are confirmed against a live project.
"""

from __future__ import annotations

from typing import Any, Optional

from . import vertex_auth
from .base_client import BaseLLMClient, normalize_content

# Cache of dynamically-created Normalized<Base> subclasses, keyed on the base
# class, so the subclass identity is stable across calls (and lazy imports).
_NORMALIZED_CLASSES: dict = {}


def _normalized_subclass(base_cls: type) -> type:
    """Return a cached subclass of ``base_cls`` that normalizes invoke() output.

    Mirrors the NormalizedChat* pattern in the other clients: providers that
    return list-of-blocks content (Gemini 3 reasoning blocks, Claude tool/think
    blocks) are flattened to a plain string for downstream agents.
    """
    cached = _NORMALIZED_CLASSES.get(base_cls)
    if cached is None:
        class _Normalized(base_cls):  # type: ignore[misc, valid-type]
            def invoke(self, input, config=None, **kwargs):
                return normalize_content(super().invoke(input, config, **kwargs))

        _Normalized.__name__ = f"Normalized{base_cls.__name__}"
        _NORMALIZED_CLASSES[base_cls] = _Normalized
        cached = _Normalized
    return cached


class _VertexClientBase(BaseLLMClient):
    """Shared ctor for Vertex clients: project/location alongside model/base_url."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.project = project
        self.location = location

    def validate_model(self) -> bool:
        # Vertex model IDs are not in the catalog; accept any (like ollama).
        return True


class VertexGeminiClient(_VertexClientBase):
    """Gemini on Vertex via ChatVertexAI."""

    def get_llm(self) -> Any:
        from langchain_google_vertexai import ChatVertexAI

        cls = _normalized_subclass(ChatVertexAI)
        llm_kwargs = {
            "model": self.model,
            "project": vertex_auth.resolve_project(self.project),
            "location": vertex_auth.resolve_location(self.location),
        }
        for key in ("temperature", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return cls(**llm_kwargs)


class VertexAnthropicClient(_VertexClientBase):
    """Claude on Vertex via ChatAnthropicVertex (uses ``model_name=``)."""

    def get_llm(self) -> Any:
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        cls = _normalized_subclass(ChatAnthropicVertex)
        llm_kwargs = {
            "model_name": self.model,
            "project": vertex_auth.resolve_project(self.project),
            "location": vertex_auth.resolve_location(self.location),
        }
        for key in ("temperature", "max_tokens", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return cls(**llm_kwargs)


class VertexGrokClient(_VertexClientBase):
    """Grok on Vertex via the OpenAI-compatible MaaS endpoint.

    Reuses NormalizedChatOpenAI so the per-model capability table and
    structured-output dispatch carry over. A Google OAuth access token is
    fetched at build time and passed as the OpenAI api_key (valid ~1h).
    """

    def get_llm(self) -> Any:
        from .openai_client import NormalizedChatOpenAI

        project = vertex_auth.resolve_project(self.project)
        location = vertex_auth.resolve_location(self.location)
        base_url = self.base_url or vertex_auth.openapi_base_url(project, location)
        token = vertex_auth.get_access_token()
        llm_kwargs = {
            "model": self.model,
            "base_url": base_url,
            "api_key": token,
        }
        for key in ("temperature", "max_retries", "reasoning_effort", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatOpenAI(**llm_kwargs)
```

- [ ] **Step 4: Add the factory dispatch branches in `tradingagents/llm_clients/factory.py`**

Insert these three branches immediately after the existing `google` branch (after line 51, before the `azure` branch at line 53):

```python
    if provider_lower == "vertex_gemini":
        from .vertex_clients import VertexGeminiClient
        return VertexGeminiClient(model, base_url, **kwargs)

    if provider_lower == "vertex_anthropic":
        from .vertex_clients import VertexAnthropicClient
        return VertexAnthropicClient(model, base_url, **kwargs)

    if provider_lower == "vertex_grok":
        from .vertex_clients import VertexGrokClient
        return VertexGrokClient(model, base_url, **kwargs)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_clients.py -q`
Expected: PASS (all TestVertexAuth + TestVertexClients tests green)

- [ ] **Step 6: Commit**

```bash
git add tradingagents/llm_clients/vertex_clients.py tradingagents/llm_clients/factory.py tests/test_vertex_clients.py
git commit -m "feat(vertex): add Gemini/Claude/Grok Vertex clients + factory dispatch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Config keys + role→model resolver

**Files:**
- Modify: `tradingagents/default_config.py` (insert after line 72, the `temperature` key)
- Modify: `tradingagents/graph/trading_graph.py` (constants after line 49; `__init__` lines 81-102; resolver methods after line 165)
- Test: `tests/test_role_model_resolver.py`

- [ ] **Step 1: Add the new config keys in `tradingagents/default_config.py`**

Insert immediately after the `"temperature": None,` line (line 72) and its comment block, before `# Checkpoint/resume`:

```python
    # Multi-model debate: maps a graph role key to {"provider","model"[,"location",
    # "temperature", ...]}. None/{} => current quick/deep tier behavior (fully
    # backward compatible). Set by the CLI "Vertex Model Garden" preset.
    "role_models": None,
    # Vertex AI Model Garden access — used only when a role routes to a vertex_*
    # provider. project falls back to env GOOGLE_CLOUD_PROJECT; location falls back
    # to GOOGLE_CLOUD_LOCATION then "global". Credentials come from ADC (gcloud auth
    # application-default login) or GOOGLE_APPLICATION_CREDENTIALS — never an API key.
    "vertex_project": None,
    "vertex_location": None,
```

- [ ] **Step 2: Write the failing resolver tests** (create `tests/test_role_model_resolver.py`)

```python
"""Per-role LLM resolver + client dedup (the multi-model debate engine).

Exercised without building the full graph (TradingAgentsGraph.__new__), with
create_llm_client patched to a fake so dedup is observed via call count and
object identity — no provider SDKs or network.
"""
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.trading_graph import DEEP_ROLES, ROLE_KEYS


def _graph(config):
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    g = TradingAgentsGraph.__new__(TradingAgentsGraph)
    g.config = config
    g.callbacks = []
    g._llm_cache = {}
    return g


def _patch_factory(monkeypatch):
    calls = []

    def fake_create(provider, model, base_url=None, **kw):
        calls.append({"provider": provider, "model": model, "base_url": base_url, **kw})
        client = MagicMock()
        client.get_llm.return_value = object()  # fresh identity per build
        return client

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client",
        MagicMock(side_effect=fake_create),
    )
    return calls


_BASE = {
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": None,
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "temperature": None,
    "vertex_project": None,
    "vertex_location": None,
    "role_models": None,
}


@pytest.mark.unit
class TestBackwardCompatTierDefaults:
    def test_role_keys_and_deep_roles(self):
        assert DEEP_ROLES == frozenset({"research_manager", "portfolio_manager"})
        assert {"market_analyst", "trader", "bull_researcher"} <= ROLE_KEYS

    def test_unspecified_quick_roles_share_one_client(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(dict(_BASE))
        a = g._llm_for("market_analyst")
        b = g._llm_for("news_analyst")
        assert a is b  # dedup: same quick tier
        quick_calls = [c for c in calls if c["model"] == "gpt-5.4-mini"]
        assert len(quick_calls) == 1
        assert quick_calls[0]["provider"] == "openai"

    def test_deep_roles_use_deep_model_distinct_from_quick(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(dict(_BASE))
        quick = g._llm_for("market_analyst")
        deep = g._llm_for("research_manager")
        assert deep is not quick
        assert any(c["model"] == "gpt-5.5" for c in calls)


@pytest.mark.unit
class TestRoleModelsPreset:
    def _vertex_config(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        cfg = dict(_BASE)
        cfg.update(
            llm_provider="vertex_gemini",
            quick_think_llm="gemini-3.5-flash",
            deep_think_llm="gemini-3.5-flash",
            vertex_project="tpmn-dev",
            vertex_location="global",
            role_models=dict(VERTEX_DEBATE_PRESET),
        )
        return cfg

    def test_roles_route_to_their_provider(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._vertex_config())
        g._llm_for("bull_researcher")
        g._llm_for("bear_researcher")
        g._llm_for("research_manager")
        by_role = {(c["provider"], c["model"]) for c in calls}
        assert ("vertex_gemini", "gemini-3.5-flash") in by_role
        assert ("vertex_grok", "xai/grok-4.20-reasoning") in by_role
        assert ("vertex_anthropic", "claude-opus-4-8") in by_role

    def test_vertex_specs_get_project_and_location(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._vertex_config())
        g._llm_for("bear_researcher")
        grok = next(c for c in calls if c["provider"] == "vertex_grok")
        assert grok["project"] == "tpmn-dev"
        assert grok["location"] == "global"

    def test_same_spec_roles_dedup_to_one_client(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._vertex_config())
        # two Grok roles, two Gemini roles, two+one Claude roles
        assert g._llm_for("bear_researcher") is g._llm_for("aggressive_debator")
        assert g._llm_for("bull_researcher") is g._llm_for("conservative_debator")
        assert g._llm_for("research_manager") is g._llm_for("portfolio_manager")
        assert g._llm_for("neutral_debator") is g._llm_for("research_manager")
        # 3 distinct Vertex clients total
        providers = [c["provider"] for c in calls]
        assert providers.count("vertex_grok") == 1
        assert providers.count("vertex_gemini") == 1
        assert providers.count("vertex_anthropic") == 1

    def test_unspecified_role_falls_back_to_quick_gemini(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._vertex_config())
        # analysts + trader are not in the preset -> quick tier (vertex_gemini)
        trader = g._llm_for("trader")
        bull = g._llm_for("bull_researcher")
        assert trader is bull  # same vertex_gemini/gemini-3.5-flash/global client
```

- [ ] **Step 3: Run to verify failure**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_role_model_resolver.py -q`
Expected: FAIL — `ImportError: cannot import name 'DEEP_ROLES' from 'tradingagents.graph.trading_graph'`

- [ ] **Step 4a: Add module-level constants in `tradingagents/graph/trading_graph.py`**

Insert after the imports block (after line 49, the `from .signal_processing import SignalProcessor` line) and before `class TradingAgentsGraph`:

```python
# Roles that synthesize the debates (the two judges) default to the deep tier;
# every other role defaults to the quick tier. The role->model resolver uses this
# to pick a tier-default model for any role that role_models does not specify.
DEEP_ROLES = frozenset({"research_manager", "portfolio_manager"})

# Canonical graph role keys that can take a per-role model via role_models.
ROLE_KEYS = frozenset({
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator", "portfolio_manager",
})
```

- [ ] **Step 4b: Replace the 2-LLM construction in `__init__`**

Replace lines 81-102 (from `# Initialize LLMs with provider-specific thinking configuration` through `self.quick_thinking_llm = quick_client.get_llm()`) with:

```python
        # Per-role LLM resolution with client dedup. role_models (when set) maps a
        # role to its own provider/model; unset roles fall back to the quick/deep
        # tier defaults below, so an unconfigured run behaves exactly as before.
        # The Reflector / SignalProcessor reuse the quick tier client.
        self._llm_cache = {}
        self.deep_thinking_llm = self._llm_for_tier("deep")
        self.quick_thinking_llm = self._llm_for_tier("quick")
```

- [ ] **Step 4c: Replace the `GraphSetup(...)` call**

Replace lines 114-120 (the `self.graph_setup = GraphSetup(...)` block) with:

```python
        self.graph_setup = GraphSetup(
            self._llm_for,
            self.tool_nodes,
            self.conditional_logic,
            analyst_concurrency_limit=self.config.get("analyst_concurrency_limit", 1),
        )
```

- [ ] **Step 4d: Add the resolver methods**

Insert immediately after the `_get_provider_kwargs` method (after line 165, its `return kwargs`) and before `_create_tool_nodes`:

```python
    def _provider_kwargs_for(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Thinking/sampling kwargs for a role_models spec (per-spec wins, else
        run-level). Vertex providers get only sampling kwargs in v1 — their
        thinking-config param names are pending live SDK verification and the
        Vertex clients forward a minimal kwarg set."""
        provider = str(spec.get("provider", "")).lower()
        kwargs: Dict[str, Any] = {}
        if provider == "google":
            level = spec.get("google_thinking_level", self.config.get("google_thinking_level"))
            if level:
                kwargs["thinking_level"] = level
        elif provider == "openai":
            effort = spec.get("openai_reasoning_effort", self.config.get("openai_reasoning_effort"))
            if effort:
                kwargs["reasoning_effort"] = effort
        elif provider == "anthropic":
            eff = spec.get("anthropic_effort", self.config.get("anthropic_effort"))
            if eff:
                kwargs["effort"] = eff
        temperature = spec.get("temperature", self.config.get("temperature"))
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)
        return kwargs

    def _base_url_for(self, provider: str) -> Optional[str]:
        """Base URL for a provider. None for vertex_* (Gemini/Claude use
        project+location; the Grok client builds its own endpoints/openapi URL);
        the run-level backend_url otherwise (single-provider vendor-direct runs)."""
        if str(provider).lower().startswith("vertex_"):
            return None
        return self.config.get("backend_url")

    def _build_cached(self, provider, model, location, kwargs):
        """Build (or reuse) the LLM for a (provider, model, location, kwargs) key.

        Roles sharing a spec share one client — the two Claude judges, the two
        Gemini debaters, the two Grok debaters each build a single client (one
        Vertex OAuth token fetch), and unspecified roles reuse the quick tier.
        Callbacks are run-global and excluded from the key but passed to the build.
        """
        build_kwargs = dict(kwargs)
        if str(provider).lower().startswith("vertex_"):
            build_kwargs["project"] = self.config.get("vertex_project")
            build_kwargs["location"] = location
        key = (str(provider).lower(), model, location, frozenset(build_kwargs.items()))
        if key not in self._llm_cache:
            if self.callbacks:
                build_kwargs["callbacks"] = self.callbacks
            self._llm_cache[key] = create_llm_client(
                provider, model, base_url=self._base_url_for(provider), **build_kwargs
            ).get_llm()
        return self._llm_cache[key]

    def _llm_for_tier(self, tier: str):
        """Build the tier-default LLM (the backward-compatible quick/deep path)."""
        provider = self.config["llm_provider"]
        model = (
            self.config["deep_think_llm"] if tier == "deep"
            else self.config["quick_think_llm"]
        )
        kwargs = self._get_provider_kwargs()
        location = self.config.get("vertex_location")
        return self._build_cached(provider, model, location, kwargs)

    def _llm_for(self, role: str):
        """Resolve the LLM for a graph role. Falls back to the quick/deep tier
        default when role_models is unset or omits the role (backward compatible)."""
        spec = (self.config.get("role_models") or {}).get(role)
        if spec is None:
            return self._llm_for_tier("deep" if role in DEEP_ROLES else "quick")
        kwargs = self._provider_kwargs_for(spec)
        location = spec.get("location") or self.config.get("vertex_location")
        return self._build_cached(spec["provider"], spec["model"], location, kwargs)
```

- [ ] **Step 5: Run the resolver tests to verify they pass**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_role_model_resolver.py -q`
Expected: PASS (all backward-compat + preset + dedup tests green)

Note: `tests/test_role_model_resolver.py` imports `from cli.presets import VERTEX_DEBATE_PRESET`, created in Task 5. If running this task in isolation before Task 5, the `TestRoleModelsPreset` class errors on import — that's expected; it goes green after Task 5. The `TestBackwardCompatTierDefaults` class passes now. (When executing top-to-bottom, do Task 5 before the final full-suite run.)

- [ ] **Step 6: Commit**

```bash
git add tradingagents/default_config.py tradingagents/graph/trading_graph.py tests/test_role_model_resolver.py
git commit -m "feat(graph): role->model resolver with client dedup + vertex config keys

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: GraphSetup per-role wiring

**Files:**
- Modify: `tradingagents/graph/setup.py` (imports line 3; `__init__` lines 17-30; `setup_graph` lines 49-66)
- Test: `tests/test_graph_setup_wiring.py`

- [ ] **Step 1: Write the failing wiring test** (create `tests/test_graph_setup_wiring.py`)

```python
"""GraphSetup asks llm_for(role) per node and builds a valid graph.

We don't run the LLMs; we assert GraphSetup requests the right role keys and
that the graph compiles. llm_for returns a MagicMock so agent factories that
bind tools / structured output at creation time work; tool nodes are simple
callables so LangGraph's add_node accepts them.
"""
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic

_ROLES = {
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator", "portfolio_manager",
}


@pytest.mark.unit
def test_setup_graph_requests_each_role_and_compiles():
    requested = []

    def llm_for(role):
        requested.append(role)
        return MagicMock()

    setup = GraphSetup(
        llm_for,
        tool_nodes={k: (lambda state: state) for k in ("market", "social", "news", "fundamentals")},
        conditional_logic=ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
    )
    workflow = setup.setup_graph(["market", "social", "news", "fundamentals"])
    # All 12 role nodes (analysts via factory lambdas, the rest eagerly) are
    # built during setup_graph, so every role key was requested.
    assert _ROLES <= set(requested)
    assert workflow.compile() is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_graph_setup_wiring.py -q`
Expected: FAIL — `TypeError: __init__() ... ` (old signature requires quick/deep LLMs positionally)

- [ ] **Step 3a: Update the `typing` import in `tradingagents/graph/setup.py`**

Change line 3 from:

```python
from typing import Any, Dict
```

to:

```python
from typing import Any, Callable, Dict
```

- [ ] **Step 3b: Replace `GraphSetup.__init__` (lines 17-30)**

```python
    def __init__(
        self,
        llm_for: Callable[[str], Any],
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
    ):
        """Initialize with required components.

        ``llm_for(role_key)`` resolves the LLM for a graph role (see
        ``TradingAgentsGraph._llm_for``); it lets each node run on its own model
        in multi-model debate mode while staying identical to the old quick/deep
        wiring when ``role_models`` is unset.
        """
        self.llm_for = llm_for
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit
```

- [ ] **Step 3c: Replace the node construction in `setup_graph` (lines 49-66)**

```python
        analyst_factories = {
            "market": lambda: create_market_analyst(self.llm_for("market_analyst")),
            "social": lambda: create_sentiment_analyst(self.llm_for("sentiment_analyst")),
            "news": lambda: create_news_analyst(self.llm_for("news_analyst")),
            "fundamentals": lambda: create_fundamentals_analyst(self.llm_for("fundamentals_analyst")),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.llm_for("bull_researcher"))
        bear_researcher_node = create_bear_researcher(self.llm_for("bear_researcher"))
        research_manager_node = create_research_manager(self.llm_for("research_manager"))
        trader_node = create_trader(self.llm_for("trader"))

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.llm_for("aggressive_debator"))
        neutral_analyst = create_neutral_debator(self.llm_for("neutral_debator"))
        conservative_analyst = create_conservative_debator(self.llm_for("conservative_debator"))
        portfolio_manager_node = create_portfolio_manager(self.llm_for("portfolio_manager"))
```

- [ ] **Step 4: Run the wiring test to verify it passes**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_graph_setup_wiring.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/graph/setup.py tests/test_graph_setup_wiring.py
git commit -m "refactor(graph): GraphSetup wires each node via llm_for(role)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CLI "Vertex Model Garden" preset

**Files:**
- Create: `cli/presets.py`
- Modify: `cli/utils.py` (`_llm_provider_table` line 355-367; add `ask_vertex_config`)
- Modify: `cli/main.py` (imports ~line 35; `get_user_selections` lines 571-681; `run_analysis` ~line 1014)
- Test: `tests/test_vertex_cli_preset.py`

- [ ] **Step 1: Write the failing CLI/preset tests** (create `tests/test_vertex_cli_preset.py`)

```python
"""Vertex multi-model CLI preset: provider table entry, preset shape, and the
pure config-mapping helper (run_analysis logic, testable without the graph)."""
import pytest

from tradingagents.graph.trading_graph import ROLE_KEYS


@pytest.mark.unit
class TestProviderTable:
    def test_vertex_entry_present_with_no_default_url(self):
        from cli.utils import _llm_provider_table, provider_default_url
        keys = {pk for _, pk, _ in _llm_provider_table()}
        assert "vertex_model_garden" in keys
        assert provider_default_url("vertex_model_garden") is None


@pytest.mark.unit
class TestPresetShape:
    def test_preset_keys_are_valid_roles(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        assert set(VERTEX_DEBATE_PRESET) <= ROLE_KEYS

    def test_judges_are_claude(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        for judge in ("research_manager", "portfolio_manager"):
            assert VERTEX_DEBATE_PRESET[judge] == {
                "provider": "vertex_anthropic", "model": "claude-opus-4-8"
            }

    def test_debaters_span_three_families(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        debater_providers = {
            VERTEX_DEBATE_PRESET[r]["provider"]
            for r in ("bull_researcher", "bear_researcher", "aggressive_debator",
                      "conservative_debator", "neutral_debator")
        }
        assert debater_providers == {"vertex_gemini", "vertex_grok", "vertex_anthropic"}


@pytest.mark.unit
class TestApplyVertexConfig:
    def test_noop_when_not_selected(self):
        from cli.presets import apply_vertex_multimodel_config
        cfg = {"llm_provider": "openai", "role_models": None}
        apply_vertex_multimodel_config(cfg, {"enable_vertex_multimodel": False})
        assert cfg["llm_provider"] == "openai"
        assert cfg["role_models"] is None

    def test_applies_preset_and_vertex_config(self):
        from cli.presets import apply_vertex_multimodel_config, VERTEX_DEBATE_PRESET
        cfg = {"llm_provider": "openai", "role_models": None}
        apply_vertex_multimodel_config(cfg, {
            "enable_vertex_multimodel": True,
            "vertex_project": "tpmn-dev",
            "vertex_location": "global",
        })
        assert cfg["llm_provider"] == "vertex_gemini"
        assert cfg["quick_think_llm"] == "gemini-3.5-flash"
        assert cfg["deep_think_llm"] == "gemini-3.5-flash"
        assert cfg["role_models"] == VERTEX_DEBATE_PRESET
        assert cfg["vertex_project"] == "tpmn-dev"
        assert cfg["vertex_location"] == "global"

    def test_location_defaults_to_global(self):
        from cli.presets import apply_vertex_multimodel_config
        cfg = {}
        apply_vertex_multimodel_config(cfg, {
            "enable_vertex_multimodel": True, "vertex_project": "p", "vertex_location": None,
        })
        assert cfg["vertex_location"] == "global"
```

- [ ] **Step 2: Run to verify failure**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_cli_preset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.presets'`

- [ ] **Step 3a: Create `cli/presets.py`**

```python
"""Presets that populate role_models for the CLI's multi-model debate options.

Kept separate from cli/main.py so the mapping (and the config it produces) is
unit-testable without importing the interactive CLI.
"""
from __future__ import annotations

from typing import Any, Dict

# Vertex Model Garden multi-model debate (Gemini / Claude / Grok), all on the
# Vertex `global` endpoint. Judges (research_manager, portfolio_manager) run on
# Claude; the five debate roles are diversified across the three families; the
# four analysts and the trader are omitted and fall back to the quick-tier
# default (Gemini) via the role->model resolver.
VERTEX_DEBATE_PRESET: Dict[str, Dict[str, str]] = {
    "bull_researcher":      {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "bear_researcher":      {"provider": "vertex_grok",      "model": "xai/grok-4.20-reasoning"},
    "aggressive_debator":   {"provider": "vertex_grok",      "model": "xai/grok-4.20-reasoning"},
    "conservative_debator": {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "neutral_debator":      {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    "research_manager":     {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    "portfolio_manager":    {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
}

# Default model for roles not in the preset (analysts + trader) and for the
# quick/deep tier fallback in Vertex mode.
VERTEX_DEFAULT_MODEL = "gemini-3.5-flash"


def apply_vertex_multimodel_config(
    config: Dict[str, Any], selections: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the Vertex multi-model debate preset to ``config`` in place.

    No-op unless ``selections['enable_vertex_multimodel']`` is truthy, so any
    other provider choice leaves ``role_models`` unset and the run on its
    single-model path.
    """
    if not selections.get("enable_vertex_multimodel"):
        return config
    config["llm_provider"] = "vertex_gemini"
    config["quick_think_llm"] = VERTEX_DEFAULT_MODEL
    config["deep_think_llm"] = VERTEX_DEFAULT_MODEL
    config["role_models"] = dict(VERTEX_DEBATE_PRESET)
    config["vertex_project"] = selections.get("vertex_project")
    config["vertex_location"] = selections.get("vertex_location") or "global"
    return config
```

- [ ] **Step 3b: Add the provider-table row in `cli/utils.py`**

In `_llm_provider_table()` (the returned list, lines 355-367), insert the Vertex row immediately after the `("Google", "google", None),` line:

```python
        ("Google", "google", None),
        ("Vertex Model Garden (multi-model debate)", "vertex_model_garden", None),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
```

- [ ] **Step 3c: Add `ask_vertex_config()` in `cli/utils.py`**

Add this function right after `provider_default_url` (after line 376):

```python
def ask_vertex_config() -> tuple[str, str]:
    """Prompt for the Vertex AI project and location (multi-model debate mode).

    Credentials come from ADC / GOOGLE_APPLICATION_CREDENTIALS, not an API key,
    so no key prompt is needed here.
    """
    project = questionary.text(
        "Enter your GCP project ID for Vertex AI:",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        validate=lambda x: len(x.strip()) > 0 or "Please enter a project ID.",
    ).ask().strip()
    location = questionary.text(
        "Enter the Vertex AI location:",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        validate=lambda x: len(x.strip()) > 0 or "Please enter a location.",
    ).ask().strip()
    return project, location
```

- [ ] **Step 4: Run the preset tests to verify they pass**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest tests/test_vertex_cli_preset.py -q`
Expected: PASS (TestProviderTable, TestPresetShape, TestApplyVertexConfig all green)

- [ ] **Step 5: Wire the preset into `cli/main.py` (interactive flow — not unit-tested, manually verified in Task 7)**

5a. Add the import after line 35 (`from cli.stats_handler import StatsCallbackHandler`):

```python
from cli.presets import apply_vertex_multimodel_config
```

5b. In `get_user_selections`, initialize the Vertex flags. Insert immediately before line 575 (`provider_from_env = bool(...)`):

```python
    is_vertex_multimodel = False
    vertex_project = None
    vertex_location = None
```

5c. In the interactive provider branch, after the Ollama confirm (after line 604, `confirm_ollama_endpoint(backend_url)`) and before `ensure_api_key(selected_llm_provider)` at line 609, add:

```python
        is_vertex_multimodel = selected_llm_provider == "vertex_model_garden"
        if is_vertex_multimodel:
            console.print(
                create_question_box(
                    "Vertex AI Configuration",
                    "GCP project + location for Model Garden (ADC auth, no API key)",
                )
            )
            vertex_project, vertex_location = ask_vertex_config()
            console.print(
                "[yellow]Multi-model debate uses 3 distinct Vertex models "
                "(incl. Claude Opus judges): slower and more expensive than a "
                "single model — debate rounds multiply the cost.[/yellow]"
            )
```

(`ensure_api_key("vertex_model_garden")` at line 609 is a safe no-op: the provider has no entry in `PROVIDER_API_KEY_ENV`, so it returns `None` without prompting.)

5d. Replace the Step 7 leading condition. Change the `if` at line 612 so the Vertex branch comes first. Replace lines 612-626 (`if os.environ.get(...) ... select_deep_thinking_agent(...)`) with:

```python
    if is_vertex_multimodel:
        selected_shallow_thinker = "gemini-3.5-flash"
        selected_deep_thinker = "gemini-3.5-flash"
        console.print(
            "[green]✓ Vertex multi-model debate preset selected "
            "(per-role models applied; analysts/trader default to Gemini).[/green]"
        )
    elif os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)
```

5e. Add the three keys to the returned dict. In the `return { ... }` block (lines 666-681), add after the `"enable_kr_sources": enable_kr_sources,` line:

```python
        "enable_vertex_multimodel": is_vertex_multimodel,
        "vertex_project": vertex_project,
        "vertex_location": vertex_location,
```

5f. In `run_analysis`, apply the preset. Insert immediately after line 1014 (`config["checkpoint_enabled"] = checkpoint`), before the KR-sources block:

```python
    # Vertex multi-model debate preset (no-op unless that provider was selected).
    apply_vertex_multimodel_config(config, selections)
```

- [ ] **Step 6: Run the full suite to confirm nothing regressed**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest -q`
Expected: PASS — the pre-existing baseline plus all new tests (no failures introduced).

- [ ] **Step 7: Commit**

```bash
git add cli/presets.py cli/utils.py cli/main.py tests/test_vertex_cli_preset.py
git commit -m "feat(cli): Vertex Model Garden multi-model debate preset + project/location prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Dependencies + documentation

**Files:**
- Modify: `pyproject.toml` (after the `[project]` dependencies block, line 33)
- Modify: `CLAUDE.md` (add a Vertex multi-model section)

- [ ] **Step 1: Add the optional `vertex` extra in `pyproject.toml`**

Insert after the closing `]` of the `dependencies` list (after line 33) and before `[project.scripts]` (line 35):

```toml
[project.optional-dependencies]
# Vertex AI Model Garden multi-model debate (Gemini/Claude/Grok via Vertex).
# Lazy-imported, so the base install and the test suite do not need these.
# Version floors are conservative; bump if your GCP/SDK requires newer.
vertex = [
    "langchain-google-vertexai>=2.0.0",
    "anthropic[vertex]>=0.40.0",
]
```

- [ ] **Step 2: Verify the package metadata still parses**

Run: `.venv/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Document the mode in `CLAUDE.md`**

Add this subsection at the end of the "### LLM client factory" section (after the `Model catalog ...` line). Use the exact text:

```markdown
### Multi-model debate via Vertex Model Garden (v0.2.6)

`role_models` (config; default `None` = current quick/deep tier behavior) maps a
graph role to its own `{"provider","model"[,"location",...]}`. The resolver lives
in `trading_graph.py` (`_llm_for(role)` + client dedup keyed on
`(provider, model, location, kwargs)`); `GraphSetup` calls `llm_for(role)` per node
(node factory signatures unchanged). `DEEP_ROLES = {research_manager,
portfolio_manager}`; all other roles default to the quick tier.

Three Vertex providers (`tradingagents/llm_clients/vertex_clients.py`, lazy SDK
imports): `vertex_gemini` (`ChatVertexAI`), `vertex_anthropic`
(`ChatAnthropicVertex`, uses `model_name=`), `vertex_grok` (`ChatOpenAI` against the
Vertex `endpoints/openapi` URL with a Google OAuth token as `api_key`). Auth is
Google ADC/service-account — **no vendor API key** (`vertex_auth.py`). Install the
optional deps with `pip install -e ".[vertex]"`.

Enable from the CLI by picking **"Vertex Model Garden (multi-model debate)"** as the
provider; it applies `cli/presets.py:VERTEX_DEBATE_PRESET` (judges=Claude, debaters
diversified across Gemini/Claude/Grok, analysts+trader=Gemini) and prompts for the
GCP project + location. Required env: `GOOGLE_CLOUD_PROJECT` (e.g. `tpmn-dev`),
optional `GOOGLE_CLOUD_LOCATION` (default `global`), and ADC via
`gcloud auth application-default login` (or `GOOGLE_APPLICATION_CREDENTIALS`).
v1 forwards a minimal kwarg set to the Vertex clients; thinking-config plumbing is
deferred. Don't remove the vendor-direct providers — they stay for single-model runs.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml CLAUDE.md
git commit -m "build+docs: add [vertex] extra and document Vertex multi-model debate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full suite, manual live-verification checklist, and merge

- [ ] **Step 1: Run the entire test suite**

Run: `DEEPSEEK_API_KEY=placeholder .venv/bin/python -m pytest -q`
Expected: PASS — pre-existing baseline (was 383 passed) plus the new tests, zero failures.

- [ ] **Step 2: Sanity-import the new modules without the Vertex extra installed**

Run:
```bash
.venv/bin/python -c "import tradingagents.llm_clients.vertex_clients, tradingagents.llm_clients.vertex_auth, cli.presets; print('lazy imports ok')"
```
Expected: `lazy imports ok` (no langchain-google-vertexai required at import time).

- [ ] **Step 3: Manual live verification (user-run; requires GCP `tpmn-dev` + ADC + enabled models)**

This is **not** automated. Provide the user this checklist:

1. `pip install -e ".[vertex]"`
2. `gcloud auth application-default login` (or set `GOOGLE_APPLICATION_CREDENTIALS`), and ensure Gemini, Claude (`claude-opus-4-8`), and Grok (`xai/grok-4.20-reasoning`) are **enabled** in Model Garden for `tpmn-dev`.
3. `export GOOGLE_CLOUD_PROJECT=tpmn-dev` (location defaults to `global`).
4. Run `tradingagents`, pick **"Vertex Model Garden (multi-model debate)"**, enter project/location, run a small ticker.
5. Confirm: the run completes; the judges' (Research Manager / Portfolio Manager) structured output renders correctly (Claude tool-use); the Trader proposal renders (Gemini).
6. **Known live-verify items (from the spec §13):** if Vertex rejects `claude-opus-4-8`, retry with a date-suffixed id (`claude-opus-4-8@<date>`) — set it in `VERTEX_DEBATE_PRESET`. If a Grok run exceeds ~1h, the OAuth token can expire (documented limitation).

- [ ] **Step 4: Final commit / branch state**

Confirm the branch builds cleanly and all task commits are present:
```bash
git log --oneline feature/vertex-multimodel-debate -10
git status
```

- [ ] **Step 5: Merge (only after the user approves and live verification passes)**

Per the fork workflow, merge with `--no-ff` and push to origin **only when the user asks**:
```bash
git checkout main
git merge --no-ff feature/vertex-multimodel-debate
# git push origin main   # ONLY when the user explicitly requests it
```

---

## Notes for the implementer

- **Line numbers are pre-edit hints, not absolute targets.** They reference the
  unedited files. Editing a block (e.g. shrinking `__init__` in Task 3) shifts all
  later line numbers, so **always match by the quoted code content** (use it as the
  `old_string` for the Edit tool), not by the line number. Every replacement in this
  plan shows the exact surrounding content to match.
- **TDD order matters across tasks:** `tests/test_role_model_resolver.py` (Task 3) imports `cli.presets` (Task 5) in its `TestRoleModelsPreset` class. When executing top-to-bottom, Task 3's `TestBackwardCompatTierDefaults` passes immediately; the preset class goes green after Task 5. The Task 7 full-suite run is the gate that everything is green together.
- **Do not** forward `thinking_level` / `effort` to the Vertex clients in v1 (the test `test_gemini_client_forwards_only_safe_kwargs` locks this in). It is a deliberate risk-reduction choice pending live SDK verification.
- **Do not** modify `model_catalog.py` — `validate_model` already accepts unregistered providers, and each Vertex client's `validate_model()` returns `True`.
- **Backward compatibility is the contract:** with `role_models` unset, `_llm_for` returns the same cached tier-default object for every quick role (and the deep object for judges), so existing single-model runs are unchanged. The `TestBackwardCompatTierDefaults` tests guard this.
```
