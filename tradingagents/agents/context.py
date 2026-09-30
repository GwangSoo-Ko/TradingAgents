"""Prompt context shared by the agents: instrument identity, output language,
portfolio, and the message reset between analysts."""

import functools
import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """Opponent's latest argument, or an explicit opening marker when empty.

    The first speaker in each debate round receives an empty opponent response;
    interpolating it into a "refute the opponent" prompt makes the model
    fabricate the other side's position. Returning a clear "has not spoken yet"
    marker instead lets it open with its own case (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    Identity resolves for the same instrument the price path fetches
    (``XAUUSD`` -> ``GC=F``, #983).
    """
    try:
        info = get_company_profile(ticker)
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
        # Listing currency anchors price/valuation reasoning. Without it the
        # pipeline (US-equity-first by origin) silently assumes USD and can
        # misread a KRW figure like 299500 as dollars for non-US tickers.
        ("currency", "currency"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def instrument_display_label(ticker: str) -> str:
    """Return ``'Company Name (TICKER)'`` for display, or the bare ticker when
    no company name resolves. Uses the lru_cached identity lookup, so during a
    run (after the identity is resolved) this is a free cache hit.
    """
    name = (resolve_instrument_identity(ticker) or {}).get("company_name")
    return f"{name} ({ticker})" if name else ticker


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
    curr_date: str | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).

    That profile carries no historical vintage: it describes the company today.
    For a run dated earlier, the context says so, since a company that has since
    renamed or been reclassified would otherwise anchor the whole graph to an
    identity it did not have on the analysis date.
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    resolved_name = None
    if identity:
        resolved_name = identity.get("company_name") or identity.get("name")
        name = resolved_name
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")
        if identity.get("currency"):
            details.append(
                f"Prices and financials are quoted in {identity['currency']}"
            )

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )
        if resolved_name:
            # Make reports readable: name the instrument as "Company Name (TICKER)"
            # in headings and prose, while still using the exact TICKER in tool
            # calls and the FINAL TRANSACTION PROPOSAL line.
            context += (
                f" When naming the {instrument_label} in your report headings and "
                f"prose, write it as \"{resolved_name} ({ticker})\" rather than the "
                "bare ticker; keep using the exact ticker in tool calls and any "
                "FINAL TRANSACTION PROPOSAL line."
            )
        today = get_current_date()
        if curr_date and str(curr_date) < today:
            context += (
                f" This identity is how the vendor describes the instrument today, "
                f"not necessarily on {curr_date}: a name or classification changed "
                f"since then would read as the current one."
            )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def build_position_block(state: Mapping[str, Any]) -> str:
    """Render the account snapshot for decision-stage agents.

    Returns an empty string -- heading included -- when no context was injected.
    A bare heading with nothing under it reads to the model as "something should
    be here but isn't", which is worse than silence.

    Analysts never call this: knowing we hold the name biases them toward the
    position (disposition effect), and the Bull/Bear debate is built on the two
    sides arguing from the same unbiased ground.

    **The Portfolio Manager is the only caller.** It is the last node in the
    graph, so nothing it writes can reach another agent's prompt -- the isolation
    is structural rather than defended. The Trader deliberately does NOT call
    this even though it also sizes: its TraderProposal renders into
    ``trader_investment_plan``, which the aggressive, conservative, and neutral
    debators all embed verbatim, so injecting there would leak the account into
    the exact three modules the isolation test pins. A source scan cannot see
    position data arriving inside a different field.
    """
    raw = str(state.get("position_context") or "").strip()
    if not raw:
        return ""
    try:
        ctx = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    if not isinstance(ctx, dict):
        return ""

    cur = ctx.get("currency") or ""
    qty = ctx.get("held_qty")
    lines = ["**Current Position and Account**"]
    if not qty:
        lines.append("- This name is **not currently held**. Any plan is a fresh entry.")
    else:
        lines.append(f"- Holding: {qty} shares at an average cost of {ctx.get('avg_price')} {cur}")
        if ctx.get("current_price") is not None:
            lines.append(f"- Last price: {ctx['current_price']} {cur}")
        if ctx.get("unrealized_pnl_pct") is not None:
            lines.append(f"- Unrealised P&L: {ctx['unrealized_pnl_pct']}%")
        if ctx.get("current_weight_pct") is not None:
            lines.append(f"- Current weight: {ctx['current_weight_pct']}% of NAV")
    if ctx.get("cash") is not None:
        lines.append(f"- Cash available: {ctx['cash']} {cur}")
    if ctx.get("total_nav") is not None:
        lines.append(f"- Total account NAV: {ctx['total_nav']} {cur}")
    ft = ctx.get("founding_thesis")
    if isinstance(ft, dict):
        lines.append("")
        lines.append("**Founding Thesis** -- the plan this position was opened on")
        lines.append(
            f"- On {ft.get('as_of')} you rated this **{ft.get('rating')}**, "
            f"price target {ft.get('price_target')} {cur}, horizon {ft.get('time_horizon')}."
        )
        lines.append(f"- Entered at {ft.get('entry_price')} {cur} on {ft.get('entry_date')}.")
        lines.append(f"- Trading days held since entry: {ft.get('trading_days_held')}.")
        if ft.get("kill_switch_price") is not None:
            lines.append(f"- That plan's own kill switch: {ft['kill_switch_price']} {cur}.")
        if ft.get("last_night_rating"):
            lines.append(f"- Most recent prior rating: {ft['last_night_rating']}.")
        lines.append(
            "If today's rating leaves the buy side (Buy/Overweight), you MUST fill the "
            "`revision` field: pick new_information / price_action / thesis_error / "
            "tactical and name what specifically changed since that plan."
        )
    elif ctx.get("founding_thesis_absent_reason") == "lookup_failed":
        lines.append("- Founding thesis: **could not be retrieved** (infrastructure issue, not absence).")
    elif ctx.get("founding_thesis_absent_reason") == "no_source_plan":
        lines.append("- Founding thesis: none on record -- this position was not opened from a plan.")
    lines.append(
        "Use these figures to size and direct the decision. **Do not quote the raw "
        "account numbers in your written output** -- refer to sizing in percentages "
        "of NAV. The written decision is archived and reused as context on later runs, "
        "where these balances will be stale."
    )
    return "\n".join(lines) + "\n"


def report_or_absent(text: str, source: str) -> str:
    """An analyst's report, or a marker saying it was never produced.

    A report is empty when its analyst was not selected, refused, or returned
    nothing. Interpolating that into a labelled section presents an absence as a
    blank finding, and the reading agent fills it in from nothing, the same way
    an empty opponent argument used to invite an invented rebuttal (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(No {source} report in this run: it is not available, not an empty finding.)"


def get_portfolio_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the caller's portfolio block, or a notice that none was given.

    A run without portfolio context must not read as a flat book: the agents
    would otherwise size as if the caller held nothing, which is a claim about
    an account we were never told about.

    A caller that never passes a portfolio can switch the notice off with
    ``portfolio_notice_when_absent: False``; the result is then ''. AlphaPulse's
    main.py does: its account reaches the Portfolio Manager alone, through
    ``position_context``, and the notice would otherwise rewrite every Trader and
    risk-debator prompt and contradict the PM's own holdings block. A portfolio
    that was given is always returned, whatever the setting.
    """
    context = state.get("portfolio_context")
    if isinstance(context, str) and context.strip():
        return context
    from tradingagents.dataflows.config import get_config
    if not get_config().get("portfolio_notice_when_absent", True):
        return ""
    return (
        "Portfolio context: not provided. You do not know the caller's current "
        "holdings or cash, so do not assume a flat book; give direction and "
        "sizing guidance in terms the caller can apply to their own position."
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages



