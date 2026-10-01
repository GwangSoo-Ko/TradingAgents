"""Vertex multi-model CLI preset: provider table entry, preset shape, the pure
config-mapping helper, and the interactive flow that applies them (cli.selections
collects the choice, cli.run applies it to the run config before the graph is built)."""
import pytest

from tradingagents.graph.trading_graph import ROLE_KEYS


@pytest.mark.unit
class TestProviderTable:
    def test_vertex_entry_present_with_no_default_url(self):
        from cli.prompts import _llm_provider_table, provider_default_url
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
                "provider": "vertex_anthropic", "model": "claude-opus-5-5"
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
        from cli.prompts import _llm_provider_table, provider_default_url
        keys = {pk for _, pk, _ in _llm_provider_table()}
        assert "vertex_anthropic" in keys and "vertex_grok" in keys
        assert provider_default_url("vertex_anthropic") is None
        assert provider_default_url("vertex_grok") is None

    def test_registry_models(self):
        from cli.presets import VERTEX_SINGLE_MODELS
        assert VERTEX_SINGLE_MODELS["vertex_anthropic"] == "claude-opus-5-5"
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
        assert cfg["quick_think_llm"] == "claude-opus-5-5"
        assert cfg["deep_think_llm"] == "claude-opus-5-5"
        assert cfg["role_models"] is None
        assert cfg["vertex_project"] == "tpmn-dev"
        assert cfg["vertex_location"] == "global"


@pytest.mark.unit
class TestVertexPreflight:
    """ensure_vertex_extra fails fast when a vertex_* provider is picked but the
    optional [vertex] extra is not installed — before the project/model prompts."""

    def test_noop_for_non_vertex_provider(self):
        import cli.selections as m
        assert m.ensure_vertex_extra("openai") is None

    def test_exits_when_extra_missing(self, monkeypatch):
        import importlib.util

        import typer

        import cli.selections as m
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(typer.Exit):
            m.ensure_vertex_extra("vertex_anthropic")

    def test_passes_when_extra_present(self, monkeypatch):
        import importlib.util

        import cli.selections as m
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
        import cli.prompts as u
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


# --- the interactive flow: cli.selections collects, cli.run applies ------------

def _never(name):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be prompted here")
    return _fail


def _answer_prompts(monkeypatch, ticker="NVDA", provider=("openai", None)):
    """Drive the real cli.selections flow with fixed answers.

    Returns the module and the list of providers the [vertex] preflight saw.
    """
    import cli.selections as sel
    from cli.models import AnalystType

    monkeypatch.setattr(sel, "fetch_announcements", lambda: [])
    monkeypatch.setattr(sel, "display_announcements", lambda *a: None)
    monkeypatch.setattr(sel, "get_ticker", lambda: ticker)
    monkeypatch.setattr(sel, "get_analysis_date", lambda: "2026-09-01")
    monkeypatch.setattr(sel, "ask_output_language", lambda default=None: "English")
    monkeypatch.setattr(sel, "select_analysts",
                        lambda asset_type, default=None: [AnalystType.MARKET])
    monkeypatch.setattr(sel, "select_research_depth", lambda default=None: 1)
    monkeypatch.setattr(sel, "select_llm_provider", lambda default=None: provider)
    monkeypatch.setattr(sel, "select_shallow_thinking_agent", lambda p, default=None: "gpt-5.4-mini")
    monkeypatch.setattr(sel, "select_deep_thinking_agent", lambda p, default=None: "gpt-5.5")
    monkeypatch.setattr(sel, "ask_openai_reasoning_effort", lambda: "medium")
    monkeypatch.setattr(sel, "ask_kr_data_sources", _never("ask_kr_data_sources"))
    monkeypatch.setattr(sel, "ask_vertex_config", _never("ask_vertex_config"))
    preflight = []
    monkeypatch.setattr(sel, "ensure_vertex_extra", preflight.append)
    return sel, preflight


def _provider_from_env(monkeypatch, sel, provider, **config):
    """TRADINGAGENTS_LLM_PROVIDER=<provider>, as DEFAULT_CONFIG reads it at import."""
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", provider)
    monkeypatch.setattr(sel, "DEFAULT_CONFIG", dict(
        sel.DEFAULT_CONFIG, llm_provider=provider, backend_url=None, **config))
    for name in ("select_llm_provider", "select_shallow_thinking_agent",
                 "select_deep_thinking_agent"):
        monkeypatch.setattr(sel, name, _never(name))


@pytest.mark.unit
class TestVertexSelectionFlow:
    def test_env_multimodel_debate_runs_without_a_prompt(self, monkeypatch):
        sel, preflight = _answer_prompts(monkeypatch)
        _provider_from_env(monkeypatch, sel, "vertex_model_garden")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "tpmn-dev")
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        chosen = sel.get_user_selections()

        assert preflight == ["vertex_model_garden"]
        assert chosen["enable_vertex_multimodel"] is True
        assert chosen["vertex_single_provider"] is None
        assert (chosen["vertex_project"], chosen["vertex_location"]) == ("tpmn-dev", "global")
        assert (chosen["quick_think_llm"], chosen["deep_think_llm"]) == (
            "gemini-3.5-flash", "gemini-3.5-flash")

    def test_env_single_claude_takes_its_model_and_step8_knobs_from_the_config(self, monkeypatch):
        sel, preflight = _answer_prompts(monkeypatch)
        _provider_from_env(monkeypatch, sel, "vertex_anthropic", anthropic_effort="high",
                           anthropic_max_tokens=20000, anthropic_thinking="adaptive")
        for name in ("ask_anthropic_effort", "ask_anthropic_max_tokens", "ask_anthropic_thinking"):
            monkeypatch.setattr(sel, name, _never(name))
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east5")

        chosen = sel.get_user_selections()

        assert preflight == ["vertex_anthropic"]
        assert chosen["enable_vertex_multimodel"] is False
        assert chosen["vertex_single_provider"] == "vertex_anthropic"
        assert (chosen["vertex_project"], chosen["vertex_location"]) == ("p", "us-east5")
        assert (chosen["quick_think_llm"], chosen["deep_think_llm"]) == (
            "claude-opus-5-5", "claude-opus-5-5")
        assert (chosen["anthropic_effort"], chosen["anthropic_max_tokens"],
                chosen["anthropic_thinking"]) == ("high", 20000, "adaptive")

    def test_env_vertex_without_a_project_stops_before_the_run(self, monkeypatch):
        import typer

        sel, _ = _answer_prompts(monkeypatch)
        _provider_from_env(monkeypatch, sel, "vertex_grok")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        with pytest.raises(typer.Exit):
            sel.get_user_selections()

    def test_menu_single_claude_asks_the_project_and_step8_not_the_models(self, monkeypatch):
        sel, preflight = _answer_prompts(monkeypatch, provider=("vertex_anthropic", None))
        monkeypatch.setattr(sel, "ask_vertex_config", lambda: ("tpmn-dev", "global"))
        for name in ("select_shallow_thinking_agent", "select_deep_thinking_agent"):
            monkeypatch.setattr(sel, name, _never(name))
        monkeypatch.setattr(sel, "ask_anthropic_effort", lambda: "max")
        monkeypatch.setattr(sel, "ask_anthropic_max_tokens", lambda: 20000)
        monkeypatch.setattr(sel, "ask_anthropic_thinking", lambda: "adaptive")

        chosen = sel.get_user_selections()

        assert preflight == ["vertex_anthropic"]
        assert chosen["llm_provider"] == "vertex_anthropic"
        assert chosen["vertex_single_provider"] == "vertex_anthropic"
        assert (chosen["vertex_project"], chosen["vertex_location"]) == ("tpmn-dev", "global")
        assert (chosen["quick_think_llm"], chosen["deep_think_llm"]) == (
            "claude-opus-5-5", "claude-opus-5-5")
        assert (chosen["anthropic_effort"], chosen["anthropic_max_tokens"],
                chosen["anthropic_thinking"]) == ("max", 20000, "adaptive")

    @pytest.mark.parametrize(("ticker", "offered"), [("005930.KS", True), ("NVDA", False)])
    def test_korean_data_sources_are_offered_for_kr_tickers_only(self, monkeypatch, ticker,
                                                                 offered):
        sel, _ = _answer_prompts(monkeypatch, ticker=ticker)
        asked = []
        monkeypatch.setattr(sel, "ask_kr_data_sources", lambda: asked.append(ticker) or True)

        chosen = sel.get_user_selections()

        assert asked == ([ticker] if offered else [])
        assert chosen["enable_kr_sources"] is offered


class _NullLive:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Buffer:
    """Stands in for cli.display.message_buffer."""

    def __init__(self):
        self.messages = []
        self.tool_calls = []
        self.report_sections = {}
        self.agent_status = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        self.selected_analysts = [a.lower() for a in selected_analysts]

    def add_message(self, kind, content):
        self.messages.append((0.0, kind, content))

    def add_tool_call(self, name, args):
        self.tool_calls.append((0.0, name, args))

    def update_report_section(self, *a):
        pass

    def update_agent_status(self, agent, status):
        self.agent_status[agent] = status


class _Graph:
    """Stands in for TradingAgentsGraph and keeps the config run_analysis built."""

    def __init__(self, selected_analysts, config=None, **kwargs):
        self.config = config
        self.graph = self
        self.propagator = self
        self._resuming = False

    def create_run_state(self, ticker, trade_date, asset_type="stock", portfolio=None):
        return {"messages": [], "company_of_interest": ticker}

    def get_graph_args(self, callbacks=None):
        return {}

    def begin_checkpoint(self, *a, **k):
        return None

    def checkpoint_input(self, state):
        return state

    def stream(self, graph_input, **kwargs):
        yield {"messages": [], "market_report": "MKT"}
        yield {"messages": [], "risk_debate_state": {"judge_decision": "**Rating**: Hold"},
               "final_trade_decision": "**Rating**: Hold"}

    def record_decision(self, *a, **k):
        pass

    def clear_checkpoint_on_success(self, *a, **k):
        pass

    def end_checkpoint(self):
        pass

    def process_signal(self, text):
        from tradingagents.agents.rating import parse_rating
        return parse_rating(text)


@pytest.mark.unit
def test_run_applies_the_vertex_preset_and_kr_sources_and_tags_the_report(monkeypatch, tmp_path):
    """cli.run applies the choices cli.selections collected to the run config
    before the graph is built, and saves the report under a mode-tagged folder
    with the company-and-model header."""
    import typer

    import cli.run as cli_run
    import tradingagents.agents.context as context
    from cli.models import AnalystType
    from cli.presets import VERTEX_DEBATE_PRESET

    default_vendors = dict(cli_run.DEFAULT_CONFIG["data_vendors"])
    built = []

    def build_graph(*args, **kwargs):
        built.append(_Graph(*args, **kwargs))
        return built[-1]

    def prompt(text, default=None, **kwargs):
        if text == "Save report?":
            return "Y"
        if "Display full report" in text:
            return "N"
        return default  # the save path: accept the offered default

    monkeypatch.setattr(cli_run, "DEFAULT_CONFIG", dict(
        cli_run.DEFAULT_CONFIG, results_dir=str(tmp_path / "results"),
        data_cache_dir=str(tmp_path / "cache")))
    monkeypatch.setattr(cli_run, "TradingAgentsGraph", build_graph)
    monkeypatch.setattr(cli_run, "message_buffer", _Buffer())
    monkeypatch.setattr(cli_run, "create_layout", lambda: None)
    monkeypatch.setattr(cli_run, "update_display", lambda *a, **k: None)
    monkeypatch.setattr(cli_run, "Live", _NullLive)
    monkeypatch.setattr(typer, "prompt", prompt)
    monkeypatch.setattr(context, "resolve_instrument_identity",
                        lambda ticker: {"company_name": "Samsung Electronics Co., Ltd."})
    monkeypatch.setattr(cli_run, "get_user_selections", lambda: {
        "ticker": "005930.KS", "analysis_date": "2026-09-01", "asset_type": "stock",
        "analysts": [AnalystType.MARKET], "research_depth": 1,
        "llm_provider": "vertex_model_garden", "backend_url": None,
        "quick_think_llm": "gemini-3.5-flash", "deep_think_llm": "gemini-3.5-flash",
        "output_language": "English", "enable_kr_sources": True,
        "enable_vertex_multimodel": True, "vertex_single_provider": None,
        "vertex_project": "tpmn-dev", "vertex_location": "global",
    })

    cli_run.run_analysis()

    (graph,) = built
    config = graph.config
    assert config["llm_provider"] == "vertex_gemini"
    assert config["role_models"] == VERTEX_DEBATE_PRESET
    assert (config["vertex_project"], config["vertex_location"]) == ("tpmn-dev", "global")
    assert config["data_vendors"]["news_data"] == "naver,yfinance"
    assert config["data_vendors"]["fundamental_data"] == "wisereport,yfinance"
    assert config["enable_kr_discussion_sentiment"] is True
    assert cli_run.DEFAULT_CONFIG["data_vendors"] == default_vendors  # shared default untouched

    (report_dir,) = (tmp_path / "results" / "reports").iterdir()
    assert report_dir.name.startswith("005930.KS_")
    assert report_dir.name.endswith("_vertex-multimodel")
    report = (report_dir / "complete_report.md").read_text(encoding="utf-8")
    assert report.splitlines()[0] == (
        "# Trading Analysis Report: Samsung Electronics Co., Ltd. (005930.KS)")
    assert "**Analysis mode:** vertex-multimodel" in report
    assert "| portfolio_manager | `vertex_anthropic` | `claude-opus-5-5` |" in report
    assert (report_dir / "1_analysts" / "market.md").read_text(encoding="utf-8") == "MKT"
    assert (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8") == (
        "**Rating**: Hold")
