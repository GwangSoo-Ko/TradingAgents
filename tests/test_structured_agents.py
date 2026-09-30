"""Tests for structured-output agents (Trader, Research Manager, Sentiment Analyst).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader, Research Manager, and Sentiment Analyst
so they share the same deterministic output shape.
"""

import inspect
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    ExitTarget,
    KillSwitch,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    Tranche,
    TrancheTrigger,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader

# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_are_named_as_not_provided(self):
        """An omitted line reads as a field nobody asked for; the reader cannot
        tell it from a level the trader declined to set."""
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        for field in ("Entry Price", "Stop Loss", "Position Sizing"):
            assert f"**{field}**: not provided" in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestNullishFloatCoercion:
    """A weak LLM may write "None"/"N/A" into an optional float field (#1058);
    coerce those to None so the structured call validates instead of erroring."""

    def test_trader_nullish_strings_coerce_to_none(self):
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            p = TraderProposal(
                action=TraderAction.HOLD,
                reasoning="x",
                entry_price=sentinel,
                stop_loss=sentinel,
            )
            assert p.entry_price is None
            assert p.stop_loss is None

    def test_trader_real_numeric_string_still_parses(self):
        p = TraderProposal(action=TraderAction.BUY, reasoning="x", entry_price="189.5")
        assert p.entry_price == 189.5

    def test_pm_nullish_price_target_coerces_to_none(self):
        d = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            price_target="N/A",
        )
        assert d.price_target is None

    def test_pm_nullish_trade_plan_floats_coerce_to_none(self):
        # The trade-plan floats ride the same validator as price_target. Without
        # them listed, a model writing "N/A" into a sizing field would blow up
        # the whole structured call and drop the run to free text -- i.e. no plan.
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            d = PortfolioDecision(
                rating=PortfolioRating.OVERWEIGHT,
                executive_summary="s",
                investment_thesis="t",
                total_weight_pct=sentinel,
                stop_loss=sentinel,
            )
            assert d.total_weight_pct is None
            assert d.stop_loss is None

    def test_pm_real_trade_plan_numeric_strings_still_parse(self):
        d = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="s",
            investment_thesis="t",
            total_weight_pct="3.0",
            stop_loss="12050",
        )
        assert d.total_weight_pct == 3.0
        assert d.stop_loss == 12050.0

    def test_tranche_nullish_price_band_coerces_to_none(self):
        # An event-triggered tranche has no band yet; a model often writes "N/A"
        # there rather than omitting the key.
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            t = Tranche(
                seq=1,
                pct=100,
                trigger="immediate",
                price_low=sentinel,
                price_high=sentinel,
            )
            assert t.price_low is None
            assert t.price_high is None

    def test_tranche_real_numeric_band_still_parses(self):
        t = Tranche(seq=1, pct=100, trigger="immediate",
                    price_low="13200", price_high="13400.5")
        assert t.price_low == 13200.0
        assert t.price_high == 13400.5

    def test_trigger_nullish_floats_coerce_to_none(self):
        # Same failure mode one level down. A sell plan's triggers carry three
        # optional numbers; "N/A" in any one of them raises, and the raise is
        # swallowed into a free-text fallback -- the whole plan disappears, not
        # just the field.
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            t = TrancheTrigger(
                kind="stop",
                price=sentinel,
                trail_pct=sentinel,
                reference_price=sentinel,
            )
            assert t.price is None
            assert t.trail_pct is None
            assert t.reference_price is None

    def test_trigger_real_numeric_strings_still_parse(self):
        t = TrancheTrigger(kind="stop", price="10600",
                           trail_pct="8.0", reference_price="14628.5")
        assert t.price == 10600.0
        assert t.trail_pct == 8.0
        assert t.reference_price == 14628.5

    def test_exit_target_nullish_weight_coerces_to_none(self):
        # kind='full'/'cost_recovery' have no residual weight, and a model that
        # writes "N/A" rather than omitting the key must not take the plan down.
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            assert ExitTarget(kind="full", remaining_weight_pct=sentinel).remaining_weight_pct is None
        assert ExitTarget(kind="weight", remaining_weight_pct="1.5").remaining_weight_pct == 1.5

    def test_kill_switch_nullish_price_coerces_to_none(self):
        # A kill switch can be condition-only ("임시주총 연기 공시 시"), so the
        # price slot is exactly where a placeholder string lands.
        for sentinel in ("None", "N/A", "null", "-", "", "TBD"):
            ks = KillSwitch(price=sentinel, condition="200SMA 훼손")
            assert ks.price is None
        assert KillSwitch(price="9100", condition="c").price == 9100.0

    # The three below pin upstream's salvaging coercer, which stays on the
    # Trader's proposal only. The Portfolio Manager's plan numbers are
    # fail-closed instead -- see the fork tests at the end of this file.
    def test_percentage_answer_to_a_price_field_becomes_none(self):
        # The Trader is asked for concrete levels and may answer a price field
        # with a distance ("15%"), which failed the whole proposal (#1288).
        # A percentage cannot be salvaged: 15% must not become a $15 stop.
        for pct in ("15%", " 7.5% ", "-10%"):
            p = TraderProposal(
                action=TraderAction.BUY,
                reasoning="x",
                entry_price=pct,
                stop_loss=pct,
            )
            assert p.entry_price is None
            assert p.stop_loss is None

    def test_human_formatted_price_is_reduced_to_its_number(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="x",
            entry_price="$1,234.50",
            stop_loss="1,180",
        )
        assert p.entry_price == 1234.50
        assert p.stop_loss == 1180.0

    def test_one_bad_field_no_longer_fails_the_whole_proposal(self):
        # Previously a single '15%' raised, forcing a free-text retry that lost
        # the action and reasoning; now the rest of the proposal survives.
        p = TraderProposal(
            action=TraderAction.SELL,
            reasoning="downgrade on margin compression",
            entry_price="612.40",
            stop_loss="15%",
        )
        assert p.action is TraderAction.SELL
        assert p.entry_price == 612.40
        assert p.stop_loss is None


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
        "market_report": "Current price $189.5; 14-day ATR 4.2; support $178, resistance $196.",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_invoke_structured_falls_back_when_result_is_none():
    # A thinking model can answer in plain text, leaving the parser with None.
    # That must fall back to free text, not crash on render(None) (#1051).
    from tradingagents.agents.structured import invoke_structured_or_freetext

    structured = MagicMock()
    structured.invoke.return_value = None
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content="FREETEXT")

    # The helper now returns (markdown, obj); the fallback path yields obj=None
    # so callers can tell "no structured plan" from "plan present".
    out, obj = invoke_structured_or_freetext(
        structured, plain, "prompt", render=lambda r: r.rating, agent_name="t"
    )
    assert out == "FREETEXT"
    assert obj is None
    plain.invoke.assert_called_once()


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Proposed Investment Plan" in m["content"] for m in prompt)

    def test_prompt_includes_market_report_for_price_levels(self):
        # #1167: the Trader must see the technical market report so entry/stop
        # levels are grounded in real price structure, not just the digested plan.
        captured = {}
        trader = create_trader(_structured_trader_llm(captured))
        trader(_make_trader_state())
        user = " ".join(m["content"] for m in captured["prompt"] if m["role"] == "user")
        system = " ".join(m["content"] for m in captured["prompt"] if m["role"] == "system")
        assert "Technical Market Report:" in user
        assert "14-day ATR 4.2" in user            # the actual report content reached the Trader
        assert "support $178, resistance $196" in user
        assert "Ground concrete price levels" in system

    def test_empty_market_report_omits_the_section_and_grounding(self):
        # #1167: when the market analyst wasn't selected the report is empty, so
        # don't tell the Trader to ground levels in a report it doesn't have.
        captured = {}
        state = _make_trader_state()
        state["market_report"] = ""
        create_trader(_structured_trader_llm(captured))(state)
        text = " ".join(m["content"] for m in captured["prompt"])
        assert "Technical Market Report:" not in text
        assert "Ground concrete price levels" not in text
        assert "Proposed Investment Plan" in text  # still present

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(captured: dict, report: SentimentReport | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    @pytest.fixture(autouse=True)
    def _stub_prefetched_sources(self, monkeypatch):
        """Stub the sources the analyst pre-fetches before prompting.

        create_sentiment_analyst fetches news, StockTwits and Reddit itself, so
        without this these tests hit the live network. A real Reddit 429 then
        backs the fetcher off for a minute per subreddit, which is what turned
        this file into a multi-minute hang.
        """
        from tradingagents.agents.analysts import sentiment_analyst as sentiment

        monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "st")
        monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")
        monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)
        # The fork's Reddit recall looks the company name up through the
        # identity resolver (yfinance, whose curl_cffi transport the socket
        # guard in conftest cannot see), so stub it too.
        monkeypatch.setattr(sentiment, "resolve_instrument_identity", lambda ticker: {})

    def test_reddit_fetch_gets_the_company_name_and_the_screen(self, monkeypatch):
        """Fork 2215b60 widens Reddit recall with the company name, on top of
        upstream's Jev screen (c924f84). Every other test stubs the fetcher
        with ``lambda *a, **k``, which accepts a keyword the real fetcher may
        not take and so hides a TypeError that would kill the node on every
        live run. Bind each call to the real signature so that cannot pass."""
        from tradingagents.agents.analysts import sentiment_analyst as sentiment
        from tradingagents.dataflows.vendors import reddit as reddit_vendor

        real = inspect.signature(reddit_vendor.fetch_reddit_posts)
        seen = {}

        def fetch(*args, **kwargs):
            real.bind(*args, **kwargs)  # TypeError if the real fetcher rejects the call
            seen.update(kwargs, ticker=args[0])
            return "rd"

        screen = object()
        monkeypatch.setattr(sentiment, "fetch_reddit_posts", fetch)
        monkeypatch.setattr(sentiment, "jev_screen", lambda ticker: screen)
        monkeypatch.setattr(sentiment, "resolve_instrument_identity",
                            lambda ticker: {"company_name": "NVIDIA Corporation"})

        create_sentiment_analyst(_structured_sentiment_llm({}))(_make_sentiment_state())

        assert seen["ticker"] == "NVDA"
        assert seen["company_name"] == "NVIDIA Corporation"
        assert seen["screen"] is screen
        assert (seen["start_date"], seen["end_date"]) == ("2026-01-08", "2026-01-15")

    def test_reddit_fetch_falls_back_to_the_ticker_without_a_company_name(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sentiment

        seen = {}
        monkeypatch.setattr(sentiment, "fetch_reddit_posts",
                            lambda *a, **k: seen.update(k) or "rd")
        create_sentiment_analyst(_structured_sentiment_llm({}))(_make_sentiment_state())
        assert seen["company_name"] is None

    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain


@pytest.mark.unit
@pytest.mark.parametrize("source", [
    pytest.param(lambda: ResearchPlan.model_fields["recommendation"].description, id="ResearchPlan.recommendation"),
    pytest.param(lambda: PortfolioDecision.model_fields["rating"].description, id="PortfolioDecision.rating"),
    pytest.param(lambda: inspect.getsource(create_research_manager), id="research_manager prompt"),
    pytest.param(lambda: inspect.getsource(create_portfolio_manager), id="portfolio_manager prompt"),
])
def test_conflict_alone_is_not_a_hold_trigger(source):
    # The debate always contains conflicting arguments, so a Hold condition that
    # conflict satisfies fires on every run and swallows directional calls
    # (#1321). All four decision sites must state the same rule.
    text = " ".join(source().split())
    assert "conflict alone is not a reason to Hold" in text or \
        "Conflicting arguments alone are not a reason to Hold" in text
    assert "materially conflicting" not in text


@pytest.mark.unit
@pytest.mark.parametrize("written", ["150-160", "150 to 160", "around 150", "150/160", "~150"])
def test_a_price_written_as_a_range_drops_only_that_field(written):
    """Anything that is not a single number becomes None. Letting it through
    fails the whole proposal's validation, and the run falls back to free text,
    losing every other field the model got right.

    Upstream pins this on PortfolioDecision.price_target; in this fork it holds
    for the Trader's proposal only (see the fail-closed tests below)."""
    proposal = TraderProposal(action=TraderAction.BUY, reasoning="r",
                              entry_price=written, stop_loss="172")
    assert proposal.entry_price is None
    assert proposal.stop_loss == 172.0
    assert proposal.action is TraderAction.BUY


# ---------------------------------------------------------------------------
# Portfolio Manager plan numbers: fail-closed (fork, AlphaPulse contract)
#
# Inverts upstream's expectation for PortfolioDecision (a range or a formatted
# price salvaged into None / a number). AlphaPulse turns the PM's plan into
# order drafts from the TRADE_PLAN_JSON line (auto-execution included), and
# there a null is not "unknown": a missing band trades at the current price with
# the band gate off, a missing stop means no stop, a missing price_target drops
# the take-profit, and a separator-stripped number is a different number
# ('1,5' -> 15.0, '11.500,00' -> 11.5). So a number pydantic cannot read as one
# plain number must fail the whole decision: the structured call falls back to
# free text, main.py prints no plan line, and AlphaPulse stores 'unparsed' with
# zero drafts. Only placeholders mean "not provided" (#1058, tested above).
# ---------------------------------------------------------------------------

_UNREADABLE_PLAN_NUMBERS = ["150-160", "150 to 160", "around 150", "150/160", "~150",
                            "$1,150.25", "12,050원", "₩12,050", "15%", "1,5", "11.500,00"]


@pytest.mark.unit
@pytest.mark.parametrize("field", ["price_target", "stop_loss", "total_weight_pct"])
@pytest.mark.parametrize("written", _UNREADABLE_PLAN_NUMBERS)
def test_an_unreadable_plan_number_fails_the_whole_decision(field, written):
    with pytest.raises(ValidationError):
        PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s",
                          investment_thesis="t", **{field: written})


@pytest.mark.unit
@pytest.mark.parametrize("build", [
    pytest.param(lambda v: Tranche(seq=1, pct=100, trigger="immediate", price_low=v),
                 id="Tranche.price_low"),
    pytest.param(lambda v: Tranche(seq=1, pct=100, trigger="immediate", price_high=v),
                 id="Tranche.price_high"),
    pytest.param(lambda v: TrancheTrigger(kind="stop", price=v), id="TrancheTrigger.price"),
    pytest.param(lambda v: TrancheTrigger(kind="trailing", trail_pct=v),
                 id="TrancheTrigger.trail_pct"),
    pytest.param(lambda v: TrancheTrigger(kind="stop", reference_price=v),
                 id="TrancheTrigger.reference_price"),
    pytest.param(lambda v: ExitTarget(kind="weight", remaining_weight_pct=v),
                 id="ExitTarget.remaining_weight_pct"),
    pytest.param(lambda v: KillSwitch(price=v, condition="c"), id="KillSwitch.price"),
])
@pytest.mark.parametrize("written", ["12,050원", "3%", "1,5", "150-160"])
def test_an_unreadable_number_inside_the_plan_fails_it(build, written):
    with pytest.raises(ValidationError):
        build(written)


@pytest.mark.unit
def test_a_plan_price_that_is_a_plain_number_survives():
    decision = PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s",
                                 investment_thesis="t", price_target="1150.25")
    assert decision.price_target == 1150.25


@pytest.mark.unit
def test_a_field_the_model_did_not_give_says_so():
    """An omitted line and a line never asked for read the same to an analyst."""
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, render_pm_decision

    rendered = render_pm_decision(PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t"))
    assert "Price Target" in rendered and "not provided" in rendered.lower()


@pytest.mark.unit
def test_the_trader_names_the_levels_it_did_not_give():
    from tradingagents.agents.schemas import TraderAction, TraderProposal, render_trader_proposal

    rendered = render_trader_proposal(TraderProposal(action=TraderAction.HOLD, reasoning="r"))
    for field in ("Entry Price", "Stop Loss", "Position Sizing"):
        assert field in rendered
    assert rendered.lower().count("not provided") == 3
