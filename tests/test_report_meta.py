"""Report metadata: analysis-mode tag (folder name) + config block (report header)."""
import pytest

from cli.report_meta import analysis_mode_tag, analysis_config_block


@pytest.mark.unit
class TestAnalysisModeTag:
    def test_single_model(self):
        cfg = {"llm_provider": "openai", "deep_think_llm": "gpt-5.5", "quick_think_llm": "gpt-5.4-mini"}
        assert analysis_mode_tag(cfg) == "openai-gpt-5.5"

    def test_single_model_sanitizes_unsafe_chars(self):
        cfg = {"llm_provider": "xai", "deep_think_llm": "grok/4.20:reasoning"}
        tag = analysis_mode_tag(cfg)
        assert "/" not in tag and ":" not in tag
        assert tag.startswith("xai-grok-4.20")

    def test_vertex_multimodel(self):
        cfg = {"role_models": {
            "bull_researcher": {"provider": "vertex_gemini", "model": "gemini-3.5-flash"},
            "research_manager": {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
        }}
        assert analysis_mode_tag(cfg) == "vertex-multimodel"

    def test_mixed_multimodel(self):
        cfg = {"role_models": {
            "bull_researcher": {"provider": "openai", "model": "gpt-5.5"},
            "bear_researcher": {"provider": "vertex_grok", "model": "xai/grok-4.20-reasoning"},
        }}
        assert analysis_mode_tag(cfg) == "multimodel"

    def test_empty_role_models_is_single(self):
        cfg = {"role_models": None, "llm_provider": "google", "deep_think_llm": "gemini-3.5-flash"}
        assert analysis_mode_tag(cfg) == "google-gemini-3.5-flash"


@pytest.mark.unit
class TestAnalysisConfigBlock:
    def test_single_model_block(self):
        cfg = {"llm_provider": "openai", "quick_think_llm": "gpt-5.4-mini", "deep_think_llm": "gpt-5.5"}
        block = analysis_config_block(cfg)
        assert "single-model" in block
        assert "gpt-5.5" in block and "gpt-5.4-mini" in block

    def test_multimodel_block_lists_all_roles_with_tier_default_marker(self):
        from cli.presets import VERTEX_DEBATE_PRESET
        cfg = {
            "role_models": dict(VERTEX_DEBATE_PRESET),
            "llm_provider": "vertex_gemini",
            "quick_think_llm": "gemini-3.5-flash",
            "deep_think_llm": "gemini-3.5-flash",
        }
        block = analysis_config_block(cfg)
        assert "vertex-multimodel" in block
        assert "research_manager" in block and "claude-opus-4-8" in block
        # trader is NOT in the preset -> shown as a tier default (Gemini)
        assert "trader" in block and "tier default" in block
        # all 12 roles appear
        for role in ("market_analyst", "bull_researcher", "portfolio_manager", "neutral_debator"):
            assert role in block
