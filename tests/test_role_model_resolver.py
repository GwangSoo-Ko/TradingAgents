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
        assert frozenset({"research_manager", "portfolio_manager"}) == DEEP_ROLES
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
        assert ("vertex_grok", "xai/grok-4.3") in by_role
        assert ("vertex_anthropic", "claude-opus-5") in by_role

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
        assert g._llm_for("bear_researcher") is g._llm_for("aggressive_debator")
        assert g._llm_for("bull_researcher") is g._llm_for("conservative_debator")
        assert g._llm_for("research_manager") is g._llm_for("portfolio_manager")
        assert g._llm_for("neutral_debator") is g._llm_for("research_manager")
        providers = [c["provider"] for c in calls]
        assert providers.count("vertex_grok") == 1
        assert providers.count("vertex_gemini") == 1
        assert providers.count("vertex_anthropic") == 1

    def test_unspecified_role_falls_back_to_quick_gemini(self, monkeypatch):
        _patch_factory(monkeypatch)
        g = _graph(self._vertex_config())
        trader = g._llm_for("trader")
        bull = g._llm_for("bull_researcher")
        assert trader is bull  # same vertex_gemini/gemini-3.5-flash/global client


@pytest.mark.unit
class TestVertexAnthropicThinkingKwargs:
    """anthropic_effort / anthropic_max_tokens / anthropic_thinking must reach the
    create_llm_client kwargs for vertex_anthropic (both the single-model tier path
    and the role_models debate path). Opt-in: unset => nothing forwarded."""

    def _single(self, **over):
        cfg = dict(_BASE)
        cfg.update(
            llm_provider="vertex_anthropic",
            quick_think_llm="claude-opus-4-8",
            deep_think_llm="claude-opus-4-8",
            vertex_project="tpmn-dev",
            vertex_location="global",
        )
        cfg.update(over)
        return cfg

    def test_single_model_forwards_effort_max_tokens_thinking(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._single(
            anthropic_effort="high",
            anthropic_max_tokens=8192,
            anthropic_thinking="adaptive",
        ))
        g._llm_for("market_analyst")  # role_models unset -> quick tier default
        call = next(c for c in calls if c["provider"] == "vertex_anthropic")
        assert call["effort"] == "high"
        assert call["max_tokens"] == 8192
        assert call["thinking"] == "adaptive"

    def test_single_model_defaults_forward_nothing(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        g = _graph(self._single())  # all knobs default (None)
        g._llm_for("market_analyst")
        call = next(c for c in calls if c["provider"] == "vertex_anthropic")
        assert "effort" not in call
        assert "max_tokens" not in call
        assert "thinking" not in call

    def test_max_tokens_env_string_coerced_to_int(self, monkeypatch):
        # Env vars arrive as strings; keep the same behavior as temperature.
        calls = _patch_factory(monkeypatch)
        g = _graph(self._single(anthropic_max_tokens="16000"))
        g._llm_for("market_analyst")
        call = next(c for c in calls if c["provider"] == "vertex_anthropic")
        assert call["max_tokens"] == 16000
        assert isinstance(call["max_tokens"], int)

    def test_role_models_spec_forwards_from_config(self, monkeypatch):
        from cli.presets import VERTEX_DEBATE_PRESET
        calls = _patch_factory(monkeypatch)
        cfg = dict(_BASE)
        cfg.update(
            llm_provider="vertex_gemini",
            quick_think_llm="gemini-3.5-flash",
            deep_think_llm="gemini-3.5-flash",
            vertex_project="tpmn-dev",
            vertex_location="global",
            role_models=dict(VERTEX_DEBATE_PRESET),
            anthropic_effort="max",
            anthropic_max_tokens=16000,
            anthropic_thinking="adaptive",
        )
        g = _graph(cfg)
        g._llm_for("research_manager")  # vertex_anthropic spec in the preset
        call = next(c for c in calls if c["provider"] == "vertex_anthropic")
        assert call["effort"] == "max"
        assert call["max_tokens"] == 16000
        assert call["thinking"] == "adaptive"

    def test_role_spec_override_wins_over_config(self, monkeypatch):
        calls = _patch_factory(monkeypatch)
        cfg = dict(_BASE)
        cfg.update(
            llm_provider="vertex_gemini",
            quick_think_llm="gemini-3.5-flash",
            deep_think_llm="gemini-3.5-flash",
            vertex_project="tpmn-dev",
            vertex_location="global",
            anthropic_effort="low",  # run-level
            role_models={
                "research_manager": {
                    "provider": "vertex_anthropic",
                    "model": "claude-opus-4-8",
                    "anthropic_effort": "max",  # per-spec wins
                },
            },
        )
        g = _graph(cfg)
        g._llm_for("research_manager")
        call = next(c for c in calls if c["provider"] == "vertex_anthropic")
        assert call["effort"] == "max"
