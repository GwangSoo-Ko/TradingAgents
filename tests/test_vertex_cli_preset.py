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
