"""The account snapshot reaches the Portfolio Manager, and only it, through the real graph.

AlphaPulse runs ``python main.py`` with TRADINGAGENTS_POSITION_CONTEXT set to the
account: holdings, cash and NAV, and the plan the position was opened on (the
Founding Thesis). test_position_context.py pins each seam on its own -- the state
key, _run_graph, the PM node, the scrub. This runs the compiled graph end to end
with scripted models and every real node, as main.py does (debug=True), so a merge
that rewires any of it -- a node that starts reading the account, a PM that loses
it, a notice that toggles on its presence, an archive that keeps the balances --
fails here. Offline: every vendor is stubbed at the router.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from tradingagents.agents import context, schemas
from tradingagents.agents.analysts import sentiment_analyst
from tradingagents.dataflows import router
from tradingagents.dataflows.vendors.yahoo import market as yahoo_market, snapshot
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph

pytestmark = pytest.mark.unit

TICKER = "NVDA"
TRADE_DATE = "2026-01-09"

ACCOUNT = {
    "held_qty": 2697, "avg_price": 10900, "current_price": 10950,
    "unrealized_pnl_pct": 0.46, "current_weight_pct": 5.9,
    "cash": 456535870, "total_nav": 500769000, "currency": "KRW",
    "founding_thesis": {
        "as_of": "20251201", "rating": "Overweight", "price_target": 14000,
        "time_horizon": "6-12 months", "kill_switch_price": 9000,
        "entry_price": 10900, "entry_date": "20251202",
        "trading_days_held": 25, "last_night_rating": "Hold",
    },
    "founding_thesis_absent_reason": "",
}
# How the account shows up in text: the holding and the balances, bare and grouped.
ACCOUNT_FIGURES = ("2697", "10900", "456535870", "456,535,870", "500769000", "500,769,000")
ACCOUNT_MARKERS = ("Current Position", "Founding Thesis", "Portfolio context")

TEXT = "Report.\n\n**Rating**: Overweight\n\nFINAL TRANSACTION PROPOSAL: **BUY**"
# The PM restates the holding and the cash, as its executive summary tends to.
DECISION = schemas.PortfolioDecision(
    rating=schemas.PortfolioRating.UNDERWEIGHT,
    executive_summary="Trim the 2697 shares; keep the 456,535,870 KRW cash in reserve.",
    investment_thesis="The founding catalyst has played out.",
    revision=schemas.PlanRevision(kind="price_action", note="Ran past the target band."),
)
STRUCTURED = {
    schemas.ResearchPlan: schemas.ResearchPlan(
        recommendation=schemas.PortfolioRating.HOLD, rationale="r", strategic_actions="a"),
    schemas.TraderProposal: schemas.TraderProposal(action=schemas.TraderAction.SELL, reasoning="r"),
    schemas.PortfolioDecision: DECISION,
    schemas.SentimentReport: schemas.SentimentReport(
        overall_band=schemas.SentimentBand.NEUTRAL, overall_score=5.0, confidence="low",
        narrative="n"),
}
ARGS = {"symbol": TICKER, "ticker": TICKER, "curr_date": TRADE_DATE, "start_date": "2026-01-02",
        "end_date": TRADE_DATE, "indicator": "rsi", "topic": "Fed rate cut", "freq": "quarterly"}


class PromptLog(list):
    """(role, prompt text) for every model call of a run."""

    def of(self, role: str) -> list[str]:
        return [text for who, text in self if who == role]


def _prompt_text(prompt) -> str:
    """A prompt's text: a string as is, messages by content (their ids differ per run)."""
    if isinstance(prompt, (list, tuple)):
        return "\n".join(str(getattr(message, "content", message)) for message in prompt)
    return str(prompt)


class ScriptedModel(BaseChatModel):
    """Calls every bound tool once, then answers TEXT; records each prompt by role."""

    role: str
    log: Any  # the run's PromptLog itself (a `list` field would be validated into a copy)
    tools: tuple = ()

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": tuple(tools)})

    def with_structured_output(self, schema, **kwargs):
        def answer(prompt):
            self.log.append((self.role, _prompt_text(prompt)))
            return STRUCTURED[schema]
        return RunnableLambda(answer)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.log.append((self.role, _prompt_text(messages)))
        if self.tools and not isinstance(messages[-1], ToolMessage):
            calls = [{"name": t.name, "id": f"call_{i}",
                      "args": {k: v for k, v in ARGS.items()
                               if k in t.tool_call_schema.model_json_schema()["properties"]}}
                     for i, t in enumerate(self.tools)]
            message = AIMessage(content="", tool_calls=calls)
        else:
            message = AIMessage(content=TEXT)
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def offline(monkeypatch):
    for method, vendors in router.VENDOR_METHODS.items():
        for vendor in vendors:
            monkeypatch.setitem(vendors, vendor, lambda *a, _m=method, **k: f"{_m} data")
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=60),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000,
    })
    monkeypatch.setattr(snapshot, "load_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", lambda *a, **k: "no posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: "no posts")
    monkeypatch.setattr(yahoo_market.yf, "Ticker",
                        lambda s: type("T", (), {"info": {"longName": "NVIDIA"}})())
    context.resolve_instrument_identity.cache_clear()


def _run(monkeypatch, tmp_path, position_context: str):
    """One propagate() over the real graph; returns (final_state, prompts, memory log text)."""
    log = PromptLog()
    monkeypatch.setenv("TRADINGAGENTS_POSITION_CONTEXT", position_context)
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: type(
        "Client", (), {"get_llm": lambda self: ScriptedModel(role=k["model"], log=log)})())
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "memory.md"),
        # Every role on a model of its own name, so each prompt is attributable.
        role_models={role: {"provider": "openai", "model": role}
                     for role in trading_graph.ROLE_KEYS},
        # As main.py runs it: the account channel on, the absent-portfolio notice off.
        position_context_from_env=True, portfolio_notice_when_absent=False,
    )
    graph = trading_graph.TradingAgentsGraph(debug=True, config=cfg)
    final_state, _ = graph.propagate(TICKER, TRADE_DATE)
    return final_state, log, (tmp_path / "memory.md").read_text(encoding="utf-8")


def test_the_account_reaches_the_portfolio_manager_only(monkeypatch, tmp_path, offline):
    final_state, log, memory = _run(monkeypatch, tmp_path / "with", json.dumps(ACCOUNT))

    assert {role for role, _ in log} >= trading_graph.ROLE_KEYS, "a role never ran"
    (pm_prompt,) = log.of("portfolio_manager")
    for shown in ("**Current Position and Account**", "Holding: 2697 shares",
                  "Cash available: 456535870 KRW", "**Founding Thesis**",
                  "On 20251201 you rated this **Overweight**", "Most recent prior rating: Hold",
                  "you MUST fill the `revision` field"):
        assert shown in pm_prompt, shown
    assert "Portfolio context" not in pm_prompt  # one account section, never two

    leaks = [(role, mark) for role, text in log if role != "portfolio_manager"
             for mark in ACCOUNT_FIGURES + ACCOUNT_MARKERS if mark in text]
    assert leaks == []

    # The state carries the channel and the typed decision to the end of the run.
    assert final_state["position_context"] == json.dumps(ACCOUNT)
    assert final_state["portfolio_decision_obj"] == DECISION
    # The operator's copy keeps the model's words; the archived copy loses the figures.
    assert "2697 shares" in final_state["final_trade_decision"]
    assert "[redacted]" in memory
    assert not [figure for figure in ACCOUNT_FIGURES if figure in memory]


def test_no_other_prompt_changes_with_the_account(monkeypatch, tmp_path, offline):
    """Every prompt but the PM's is the same with and without an account, so
    nothing upstream of the PM -- no notice, no hint -- depends on whether an
    account was injected."""
    _, with_account, _ = _run(monkeypatch, tmp_path / "with", json.dumps(ACCOUNT))
    _, without, _ = _run(monkeypatch, tmp_path / "without", "")

    def others(log):
        return [(role, text) for role, text in log if role != "portfolio_manager"]

    assert others(with_account) == others(without)
    (pm_prompt,) = without.of("portfolio_manager")
    assert not [mark for mark in ACCOUNT_MARKERS if mark in pm_prompt]
