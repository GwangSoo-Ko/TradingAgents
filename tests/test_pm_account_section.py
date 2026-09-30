"""The Portfolio Manager's one account section, and the switch for upstream's
absent-portfolio notice (fork on top of upstream 6436d1f).

Two account channels meet at the PM. ``position_context`` is AlphaPulse's
account JSON (env TRADINGAGENTS_POSITION_CONTEXT): holdings, cash, NAV and the
Founding Thesis, rendered for the PM alone. ``portfolio_context`` is upstream's
caller portfolio (--portfolio / propagate(portfolio=)), which upstream renders --
or replaces with a "not provided" notice -- for the Trader, the three risk
debators and the PM. The PM shows exactly one of them: the position block when it
was injected, otherwise the caller portfolio or the notice. Two blocks would put
"you hold 1250 shares" next to "you do not know the caller's holdings".

AlphaPulse's main.py switches the notice off (``portfolio_notice_when_absent:
False``): it never passes a caller portfolio, so with the notice on it would
rewrite every Trader and debator prompt of every run. Every other caller keeps
upstream's behaviour.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.context import build_position_block, get_portfolio_context_from_state
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.config import set_config

pytestmark = pytest.mark.unit

NOTICE = "Portfolio context: not provided"
INSTRUMENT = "INSTRUMENT_CONTEXT_MARKER"
CALLER_PORTFOLIO = "PORTFOLIO_BLOCK_MARKER"
POSITION = json.dumps({
    "held_qty": 1250, "avg_price": 10480.0, "current_price": 10800.0,
    "cash": 421675870.0, "total_nav": 451015870.0, "currency": "KRW",
})

_RISK_DEBATE = {
    "history": "h", "aggressive_history": "", "conservative_history": "",
    "neutral_history": "", "latest_speaker": "", "current_aggressive_response": "",
    "current_conservative_response": "", "current_neutral_response": "",
    "judge_decision": "", "count": 3,
}


def _pm_prompt(**state) -> str:
    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or (
        PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t"))
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    create_portfolio_manager(llm)({
        "company_of_interest": "417310.KS", "instrument_context": INSTRUMENT,
        "investment_plan": "plan", "trader_investment_plan": "proposal", "past_context": "",
        "risk_debate_state": dict(_RISK_DEBATE), **state,
    })
    return captured["prompt"]


def test_an_injected_position_is_the_only_account_section():
    prompt = _pm_prompt(position_context=POSITION, portfolio_context=CALLER_PORTFOLIO)
    block = build_position_block({"position_context": POSITION})
    assert block and prompt.count(block) == 1
    assert f"{INSTRUMENT}\n\n{block}\n---" in prompt  # a2981a7's layout, byte for byte
    assert CALLER_PORTFOLIO not in prompt
    assert NOTICE not in prompt


def test_the_notice_never_sits_next_to_injected_holdings():
    """Default config (notice on), no caller portfolio: the holdings block alone."""
    prompt = _pm_prompt(position_context=POSITION)
    assert NOTICE not in prompt
    assert "Current Position and Account" in prompt


def test_without_a_position_the_caller_portfolio_takes_the_section():
    prompt = _pm_prompt(portfolio_context=CALLER_PORTFOLIO)
    assert f"{INSTRUMENT}\n\n{CALLER_PORTFOLIO}\n\n---" in prompt  # upstream's layout
    assert NOTICE not in prompt


def test_by_default_an_absent_portfolio_is_announced():
    """Upstream behaviour, unchanged for every caller that does not switch it off."""
    prompt = _pm_prompt()
    notice = get_portfolio_context_from_state({})
    assert notice.startswith(NOTICE)
    assert prompt.count(NOTICE) == 1
    assert f"{INSTRUMENT}\n\n{notice}\n\n---" in prompt


def test_a_switched_off_notice_leaves_an_empty_account_section():
    set_config({"portfolio_notice_when_absent": False})
    prompt = _pm_prompt(position_context="")
    assert "portfolio context" not in prompt.lower()
    assert f"{INSTRUMENT}\n\n\n---" in prompt  # a2981a7's layout with nothing injected


def test_the_switch_silences_only_the_notice():
    set_config({"portfolio_notice_when_absent": False})
    assert get_portfolio_context_from_state({}) == ""
    assert get_portfolio_context_from_state({"portfolio_context": "  "}) == ""
    assert get_portfolio_context_from_state(
        {"portfolio_context": CALLER_PORTFOLIO}) == CALLER_PORTFOLIO


class _RecordingLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt, *a, **k):
        from langchain_core.messages import AIMessage

        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return AIMessage("argument")

    def with_structured_output(self, *a, **k):
        raise NotImplementedError  # free-text path: the prompt is all we read


_DECISION_STATE = {
    "company_of_interest": "417310.KS", "trade_date": "2026-08-19", "asset_type": "stock",
    "instrument_context": INSTRUMENT, "market_report": "M", "sentiment_report": "S",
    "news_report": "N", "fundamentals_report": "F", "investment_plan": "P",
    "trader_investment_plan": "T", "past_context": "", "portfolio_context": "",
    "investment_debate_state": {"history": "", "judge_decision": "", "count": 0},
    "risk_debate_state": _RISK_DEBATE,
}


@pytest.mark.parametrize("module, factory", [
    ("tradingagents.agents.trader.trader", "create_trader"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
])
def test_a_switched_off_notice_reaches_no_other_decision_agent(module, factory):
    node_factory = getattr(importlib.import_module(module), factory)

    default = _RecordingLLM()
    node_factory(default)(dict(_DECISION_STATE))
    assert NOTICE in " ".join(default.prompts)  # premise: upstream injects it by default

    set_config({"portfolio_notice_when_absent": False})
    silenced = _RecordingLLM()
    node_factory(silenced)(dict(_DECISION_STATE))
    assert silenced.prompts and "portfolio context" not in " ".join(silenced.prompts).lower()
