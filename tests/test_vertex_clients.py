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

    def test_anthropic_client_omits_model_kwargs_when_unset(self, monkeypatch):
        # Opt-in: a plain build must not inject any thinking/effort config, so a
        # default Vertex Claude run is byte-for-byte unchanged.
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient("claude-opus-4-8", project="p", location="global").get_llm()
        assert "model_kwargs" not in captured["claude"]
        assert "effort" not in captured["claude"]
        assert "thinking" not in captured["claude"]

    def test_anthropic_client_forwards_max_tokens(self, monkeypatch):
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient(
            "claude-opus-4-8", project="p", location="global", max_tokens=8192,
        ).get_llm()
        assert captured["claude"]["max_tokens"] == 8192

    def test_anthropic_client_maps_effort_into_output_config(self, monkeypatch):
        # effort is not a ChatAnthropicVertex field; it must ride in model_kwargs
        # as output_config.effort so it reaches messages.create(**params).
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient(
            "claude-opus-4-8", project="p", location="global", effort="high",
        ).get_llm()
        assert captured["claude"]["model_kwargs"]["output_config"] == {"effort": "high"}
        assert "effort" not in captured["claude"]

    def test_anthropic_client_wraps_thinking_string_into_dict(self, monkeypatch):
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient(
            "claude-opus-4-8", project="p", location="global", thinking="adaptive",
        ).get_llm()
        assert captured["claude"]["model_kwargs"]["thinking"] == {"type": "adaptive"}
        assert "thinking" not in captured["claude"]

    def test_anthropic_client_passes_thinking_dict_through(self, monkeypatch):
        # A programmatic caller may pass the full dict; keep it verbatim.
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        VertexAnthropicClient(
            "claude-opus-4-8", project="p", location="global",
            thinking={"type": "enabled", "budget_tokens": 4096},
        ).get_llm()
        assert captured["claude"]["model_kwargs"]["thinking"] == {
            "type": "enabled", "budget_tokens": 4096
        }

    def test_grok_client_builds_openai_compat_with_token(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.setattr(
            vertex_auth, "get_access_token", lambda credentials=None: "FAKE_TOKEN"
        )
        from tradingagents.llm_clients.vertex_clients import VertexGrokClient
        llm = VertexGrokClient(
            "xai/grok-4.3", project="tpmn-dev", location="global"
        ).get_llm()
        assert llm.model_name == "xai/grok-4.3"
        base = str(llm.openai_api_base)
        assert "endpoints/openapi" in base
        assert base.startswith("https://aiplatform.googleapis.com/")  # global host
        assert llm.openai_api_key.get_secret_value() == "FAKE_TOKEN"

    def test_validate_model_accepts_any(self, monkeypatch):
        _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.vertex_clients import (
            VertexAnthropicClient,
            VertexGeminiClient,
            VertexGrokClient,
        )
        assert VertexGeminiClient("anything").validate_model() is True
        assert VertexAnthropicClient("anything").validate_model() is True
        assert VertexGrokClient("anything").validate_model() is True

    def test_end_to_end_config_knobs_reach_chatanthropicvertex(self, monkeypatch):
        """Full chain: config knobs -> _llm_for -> real VertexAnthropicClient ->
        ChatAnthropicVertex constructor. create_llm_client is NOT mocked here, so a
        key-name mismatch between the graph builder and the client would surface."""
        captured = _install_fake_vertexai(monkeypatch)
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        g = TradingAgentsGraph.__new__(TradingAgentsGraph)
        g.callbacks = []
        g._llm_cache = {}
        g.config = {
            "llm_provider": "vertex_anthropic",
            "quick_think_llm": "claude-opus-4-8",
            "deep_think_llm": "claude-opus-4-8",
            "vertex_project": "tpmn-dev",
            "vertex_location": "global",
            "backend_url": None,
            "temperature": None,
            "role_models": None,
            "anthropic_effort": "high",
            "anthropic_max_tokens": 8192,
            "anthropic_thinking": "adaptive",
        }
        g._llm_for("research_manager")  # deep tier -> vertex_anthropic
        claude = captured["claude"]
        assert claude["model_name"] == "claude-opus-4-8"
        assert claude["max_tokens"] == 8192
        assert claude["model_kwargs"]["output_config"] == {"effort": "high"}
        assert claude["model_kwargs"]["thinking"] == {"type": "adaptive"}

    def test_factory_dispatches_vertex_providers(self, monkeypatch):
        _install_fake_vertexai(monkeypatch)
        from tradingagents.llm_clients.factory import create_llm_client
        from tradingagents.llm_clients.vertex_clients import (
            VertexAnthropicClient,
            VertexGeminiClient,
            VertexGrokClient,
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
            create_llm_client("vertex_grok", "xai/grok-4.3", project="p"),
            VertexGrokClient,
        )


@pytest.mark.unit
class TestMissingVertexSDK:
    """When the optional [vertex] extra is absent, raise an actionable error."""

    def _force_import_error(self, monkeypatch, missing_name):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == missing_name or name.startswith(missing_name + "."):
                raise ImportError(f"No module named '{missing_name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_gemini_missing_sdk_points_to_extra(self, monkeypatch):
        self._force_import_error(monkeypatch, "langchain_google_vertexai")
        from tradingagents.llm_clients.vertex_clients import VertexGeminiClient
        with pytest.raises(ImportError, match=r'pip install -e ".\[vertex\]"'):
            VertexGeminiClient("gemini-3.5-flash", project="p", location="global").get_llm()

    def test_anthropic_missing_sdk_points_to_extra(self, monkeypatch):
        self._force_import_error(monkeypatch, "langchain_google_vertexai")
        from tradingagents.llm_clients.vertex_clients import VertexAnthropicClient
        with pytest.raises(ImportError, match=r'pip install -e ".\[vertex\]"'):
            VertexAnthropicClient("claude-opus-4-8", project="p", location="global").get_llm()

    def test_get_access_token_missing_google_auth_points_to_extra(self, monkeypatch):
        self._force_import_error(monkeypatch, "google.auth")
        from tradingagents.llm_clients import vertex_auth
        with pytest.raises(ImportError, match=r'pip install -e ".\[vertex\]"'):
            vertex_auth.get_access_token()
