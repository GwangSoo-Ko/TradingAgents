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
        from cli.presets import VERTEX_DEBATE_PRESET, apply_vertex_multimodel_config
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


@pytest.mark.unit
class TestVertexSingleModel:
    def test_table_has_claude_and_grok_single_options(self):
        from cli.utils import _llm_provider_table, provider_default_url
        keys = {pk for _, pk, _ in _llm_provider_table()}
        assert "vertex_anthropic" in keys and "vertex_grok" in keys
        assert provider_default_url("vertex_anthropic") is None
        assert provider_default_url("vertex_grok") is None

    def test_registry_models(self):
        from cli.presets import VERTEX_SINGLE_MODELS
        assert VERTEX_SINGLE_MODELS["vertex_anthropic"] == "claude-opus-4-8"
        assert VERTEX_SINGLE_MODELS["vertex_grok"] == "xai/grok-4.3"

    def test_apply_noop_when_not_selected(self):
        from cli.presets import apply_vertex_single_model_config
        cfg = {"llm_provider": "openai", "role_models": None}
        apply_vertex_single_model_config(cfg, {"vertex_single_provider": None})
        assert cfg["llm_provider"] == "openai"
        assert cfg["role_models"] is None

    def test_apply_claude_single(self):
        from cli.presets import apply_vertex_single_model_config
        cfg = {}
        apply_vertex_single_model_config(cfg, {
            "vertex_single_provider": "vertex_anthropic",
            "vertex_project": "tpmn-dev", "vertex_location": None,
        })
        assert cfg["llm_provider"] == "vertex_anthropic"
        assert cfg["quick_think_llm"] == "claude-opus-4-8"
        assert cfg["deep_think_llm"] == "claude-opus-4-8"
        assert cfg["role_models"] is None
        assert cfg["vertex_project"] == "tpmn-dev"
        assert cfg["vertex_location"] == "global"


@pytest.mark.unit
class TestVertexPreflight:
    """ensure_vertex_extra fails fast when a vertex_* provider is picked but the
    optional [vertex] extra is not installed — before the project/model prompts."""

    def test_noop_for_non_vertex_provider(self):
        import cli.main as m
        assert m.ensure_vertex_extra("openai") is None

    def test_exits_when_extra_missing(self, monkeypatch):
        import importlib.util

        import typer

        import cli.main as m
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(typer.Exit):
            m.ensure_vertex_extra("vertex_anthropic")

    def test_passes_when_extra_present(self, monkeypatch):
        import importlib.util

        import cli.main as m
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        assert m.ensure_vertex_extra("vertex_grok") is None


class _FakeText:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


@pytest.mark.unit
class TestVertexConfigDefaults:
    """The GCP project prompt defaults to tpmn-dev when GOOGLE_CLOUD_PROJECT is
    unset, but the env var still wins when present."""

    def _project_default(self, monkeypatch, env):
        import cli.utils as u
        for k in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        defaults = []

        def fake_text(msg, default=None, validate=None):
            defaults.append(default)
            return _FakeText("x")  # non-empty so the None-guards pass

        monkeypatch.setattr(u.questionary, "text", fake_text)
        u.ask_vertex_config()
        return defaults[0]  # the project prompt's default

    def test_project_defaults_to_tpmn_dev_without_env(self, monkeypatch):
        assert self._project_default(monkeypatch, {}) == "tpmn-dev"

    def test_project_honors_env_when_set(self, monkeypatch):
        assert self._project_default(
            monkeypatch, {"GOOGLE_CLOUD_PROJECT": "other-proj"}
        ) == "other-proj"

    def test_apply_grok_single(self):
        from cli.presets import apply_vertex_single_model_config
        cfg = {}
        apply_vertex_single_model_config(cfg, {
            "vertex_single_provider": "vertex_grok",
            "vertex_project": "p", "vertex_location": "global",
        })
        assert cfg["llm_provider"] == "vertex_grok"
        assert cfg["quick_think_llm"] == "xai/grok-4.3"
        assert cfg["deep_think_llm"] == "xai/grok-4.3"

    def test_multimodel_preset_uses_grok_4_3(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        assert VERTEX_DEBATE_PRESET["bear_researcher"]["model"] == "xai/grok-4.3"
        assert VERTEX_DEBATE_PRESET["aggressive_debator"]["model"] == "xai/grok-4.3"
