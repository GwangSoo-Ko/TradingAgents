"""Report metadata: analysis-mode tag (folder name) + config block (report header)."""
import pytest

from cli.report_meta import analysis_config_block, analysis_mode_tag, build_report_header


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
            "bear_researcher": {"provider": "vertex_grok", "model": "xai/grok-4.3"},
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
        assert "research_manager" in block and "claude-opus-5-5" in block
        # trader is NOT in the preset -> shown as a tier default (Gemini)
        assert "trader" in block and "tier default" in block
        # all 12 roles appear
        for role in ("market_analyst", "bull_researcher", "portfolio_manager", "neutral_debator"):
            assert role in block


@pytest.mark.unit
class TestBuildReportHeader:
    """The title block of complete_report.md that main.py and the CLI write.

    alpha-pulse shows it (web report view, discovery input): the title names the
    company and the ticker, then the time, then the model behind every role. The
    expected strings are the layout the fork's own writer produced before the
    writer moved to tradingagents.reporting (title, blank, Generated, blank,
    config block).
    """

    @pytest.fixture(autouse=True)
    def _clock_and_identity(self, monkeypatch):
        import datetime as dt

        import cli.report_meta as report_meta
        import tradingagents.agents.context as context

        class _Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 8, 19, 21, 34, 5)

        monkeypatch.setattr(report_meta, "datetime", _Clock)
        monkeypatch.setattr(context, "resolve_instrument_identity",
                            lambda t: {"company_name": "하나글로벌리츠"} if t == "417310.KS" else {})

    def test_single_model_header(self):
        cfg = {"llm_provider": "openai", "quick_think_llm": "gpt-5.4-mini", "deep_think_llm": "gpt-5.5"}
        assert build_report_header("417310.KS", cfg) == (
            "# Trading Analysis Report: 하나글로벌리츠 (417310.KS)\n\n"
            "Generated: 2026-08-19 21:34:05\n\n"
            "**Analysis mode:** single-model\n\n"
            "**Provider:** `openai` · **quick-tier:** `gpt-5.4-mini` · **deep-tier:** `gpt-5.5`\n\n"
        )

    def test_without_a_name_or_a_config_the_title_is_the_bare_ticker(self):
        assert build_report_header("ZZZZ") == (
            "# Trading Analysis Report: ZZZZ\n\nGenerated: 2026-08-19 21:34:05\n\n"
        )

    def test_main_py_config_lists_opus_judges_and_sonnet_for_every_other_role(self):
        import main

        header = build_report_header("417310.KS", main.build_config())

        assert header.startswith(
            "# Trading Analysis Report: 하나글로벌리츠 (417310.KS)\n\n"
            "Generated: 2026-08-19 21:34:05\n\n"
            "**Analysis mode:** vertex-multimodel\n\n"
            "| Role | Provider | Model |\n| --- | --- | --- |\n"
        )
        assert header.endswith("` |\n\n")
        rows = [line for line in header.splitlines() if line.startswith("| ") and "`" in line]
        judges = {"research_manager", "portfolio_manager"}
        assert len(rows) == 12
        for role in judges:
            assert f"| {role} | `vertex_anthropic` | `claude-opus-5-5` |" in rows
        for row in rows:
            if row.split(" ")[1] not in judges:
                assert row.endswith("*(tier default)* | `vertex_anthropic` | `claude-sonnet-5-5` |")

    def test_the_writer_puts_it_above_the_sections(self, tmp_path):
        from tradingagents.reporting import write_report_tree

        header = build_report_header("417310.KS")
        out = write_report_tree({"market_report": "MKT"}, "417310.KS", tmp_path, header=header)

        assert out.read_text(encoding="utf-8") == (
            header + "## I. Analyst Team Reports\n\n### Market Analyst\nMKT")
