from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.state import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


def _tools_or_clear(spec):
    """Route an analyst's turn: run its tool calls, or finish its report."""
    def route(state) -> str:
        return spec.tool_node if state["messages"][-1].tool_calls else spec.clear_node
    return route


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        llm_for: Callable[[str], Any],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components.

        ``llm_for(role_key)`` resolves the LLM for a graph role (see
        ``TradingAgentsGraph._llm_for``); it lets each node run on its own model
        in multi-model debate mode while staying identical to the old quick/deep
        wiring when ``role_models`` is unset. Tool nodes come from each analyst's
        own ``TOOLS`` (see ``analyst_execution``), not from the caller.
        """
        self.llm_for = llm_for
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.llm_for("market_analyst")),
            "social": lambda: create_sentiment_analyst(self.llm_for("sentiment_analyst")),
            "news": lambda: create_news_analyst(self.llm_for("news_analyst")),
            "fundamentals": lambda: create_fundamentals_analyst(self.llm_for("fundamentals_analyst")),
        }

        # Every node gets its role's LLM from the resolver -- the two judges too,
        # so a role_models entry for them is honoured instead of a fixed deep tier.
        bull_researcher_node = create_bull_researcher(self.llm_for("bull_researcher"))
        bear_researcher_node = create_bear_researcher(self.llm_for("bear_researcher"))
        research_manager_node = create_research_manager(self.llm_for("research_manager"))
        trader_node = create_trader(self.llm_for("trader"))

        aggressive_analyst = create_aggressive_debator(self.llm_for("aggressive_debator"))
        neutral_analyst = create_neutral_debator(self.llm_for("neutral_debator"))
        conservative_analyst = create_conservative_debator(self.llm_for("conservative_debator"))
        portfolio_manager_node = create_portfolio_manager(self.llm_for("portfolio_manager"))

        workflow = StateGraph(AgentState)

        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            if spec.tools:
                workflow.add_node(spec.tool_node, ToolNode(list(spec.tools)))

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        workflow.add_edge(START, plan.specs[0].agent_node)

        for i, spec in enumerate(plan.specs):
            if spec.tools:
                workflow.add_conditional_edges(
                    spec.agent_node, _tools_or_clear(spec), [spec.tool_node, spec.clear_node]
                )
                workflow.add_edge(spec.tool_node, spec.agent_node)
            else:
                workflow.add_edge(spec.agent_node, spec.clear_node)

            # The last analyst hands over to the research debate.
            following = plan.specs[i + 1].agent_node if i < len(plan.specs) - 1 else "Bull Researcher"
            workflow.add_edge(spec.clear_node, following)

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
