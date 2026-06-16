"""GraphSetup asks llm_for(role) per node and builds a valid graph.

We don't run the LLMs; we assert GraphSetup requests the right role keys and
that the graph compiles. llm_for returns a MagicMock so agent factories that
bind tools / structured output at creation time work; tool nodes are simple
callables so LangGraph's add_node accepts them.
"""
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup

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
    assert set(requested) >= _ROLES
    assert workflow.compile() is not None
