# TradingAgents/graph/trading_graph.py

import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG, news_region_for_ticker
from tradingagents.llm_clients import create_llm_client
from tradingagents.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

# Roles that synthesize the debates (the two judges) default to the deep tier;
# every other role defaults to the quick tier. The role->model resolver uses this
# to pick a tier-default model for any role that role_models does not specify.
DEEP_ROLES = frozenset({"research_manager", "portfolio_manager"})

# Canonical graph role keys that can take a per-role model via role_models.
ROLE_KEYS = frozenset({
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator", "portfolio_manager",
})


def _coerce_max_retries(value):
    """Validate an ``llm_max_retries`` value to a non-negative int.

    Accepts an int or a numeric string (env vars arrive as strings). Rejects
    booleans and negatives loudly so a misconfiguration fails at startup rather
    than silently disabling retries.
    """
    if isinstance(value, bool):
        raise ValueError(f"llm_max_retries must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_max_retries must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"llm_max_retries must be >= 0, got {n}")
    return n


def _read_position_context() -> str:
    """env 에서 계좌 스냅샷 JSON 을 읽는다. 없거나 깨졌으면 빈 문자열.

    파싱 실패로 분석을 죽이지 않는다 -- 계좌 맥락은 판단의 질을 높이는 부가
    정보이고, 없으면 예전 품질로 떨어질 뿐이다.
    """
    raw = os.environ.get("TRADINGAGENTS_POSITION_CONTEXT", "").strip()
    if not raw:
        return ""
    try:
        json.loads(raw)
    except (TypeError, ValueError):
        print(
            "warning: TRADINGAGENTS_POSITION_CONTEXT is not valid JSON -- ignoring",
            file=sys.stderr,
        )
        return ""
    return raw


# Account figures the archive must never carry. Percentages (weight, P&L) and the
# last traded price are deliberately absent: the prompt tells the model to express
# sizing in percent, and a price level is public market data, not a balance.
_ACCOUNT_NUMBER_KEYS = ("cash", "total_nav", "held_qty", "avg_price")


# A figure with fewer digits than this is not account data in any useful sense --
# it is guessable, it is not a balance, and redacting it wrecks the archive: a
# 1-share holding turns "R:R 1 to 3; Phase 1 entry" into two redactions. Every
# realistic cash/NAV figure, and every avg_price above 9.99, clears the bar.
_MIN_REDACTED_DIGITS = 3


def _account_number_forms(value: Any) -> set[str]:
    """Every textual shape one injected figure can take in the model's prose.

    Three shapes, not one. Models re-render numbers for humans, so ``456535870``
    comes back as ``456,535,870`` about as often as it comes back bare -- and a
    producer that hands us ``456535870.0`` (JSON has one number type; whether the
    caller sends int or float is not pinned) must still redact the bare integer
    the model actually writes.
    """
    forms = {str(value).strip()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            forms.add(str(int(value)))
            forms.add(f"{int(value):,}")
        else:
            forms.add(f"{value:,}")
    return {
        f for f in forms
        if sum(c.isdigit() for c in f) >= _MIN_REDACTED_DIGITS
    }


def scrub_account_numbers(text: str, position_context: str) -> str:
    """Remove injected account figures from text about to be archived.

    The prompt asks the model not to quote them, but the Portfolio Manager prompt
    also asks its executive summary to cover position sizing -- the two pull in
    opposite directions, so the instruction alone is not enough. Archived
    decisions feed ``get_past_context(n_same=5)``, so one leak colours the next
    five runs with balances that are, by then, wrong.

    Only the memory-log copy is scrubbed. The saved report and the state the
    operator sees keep the model's original wording.
    """
    if not position_context or not text:
        return text
    try:
        ctx = json.loads(position_context)
    except (TypeError, ValueError):
        return text
    if not isinstance(ctx, dict):
        return text

    forms: set[str] = set()
    for key in _ACCOUNT_NUMBER_KEYS:
        value = ctx.get(key)
        if value is None or isinstance(value, bool) or value == 0:
            continue
        forms |= _account_number_forms(value)

    out = text
    # Longest first, and only on a standalone number: a short figure must not eat
    # part of a longer one (avg_price 10900 inside total_nav 109000, or inside
    # 110900). The trailing `.`/`,` must NOT count as continuation on its own --
    # "Cash of 456535870." and "Cash 456535870, plus room." are the likeliest
    # phrasings, and treating the punctuation as part of the number let exactly
    # those through. Only a digit *after* the separator continues the number
    # (10900 inside 10900.50).
    for form in sorted(forms, key=len, reverse=True):
        out = re.sub(rf"(?<![\d.,]){re.escape(form)}(?![\d]|[.,]\d)", "[redacted]", out)
    return out


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Per-role LLM resolution with client dedup. role_models (when set) maps a
        # role to its own provider/model; unset roles fall back to the quick/deep
        # tier defaults below, so an unconfigured run behaves exactly as before.
        # The Reflector / SignalProcessor reuse the quick tier client.
        self._llm_cache = {}
        self.deep_thinking_llm = self._llm_for_tier("deep")
        self.quick_thinking_llm = self._llm_for_tier("quick")

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self._llm_for,
            self.tool_nodes,
            self.conditional_logic,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Graph-shape-affecting run choices, kept for the checkpoint signature.
        self.selected_analysts = tuple(selected_analysts)

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider in ("anthropic", "vertex_anthropic"):
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort
            # max_tokens/thinking are wired only for the Vertex Claude client
            # (VertexAnthropicClient routes them into model_kwargs); the
            # vendor-direct anthropic path is left untouched.
            if provider == "vertex_anthropic":
                max_tokens = self.config.get("anthropic_max_tokens")
                if max_tokens is not None and max_tokens != "":
                    kwargs["max_tokens"] = int(max_tokens)
                thinking = self.config.get("anthropic_thinking")
                if thinking:
                    kwargs["thinking"] = thinking

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a TRADINGAGENTS_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        # SDK retry budget is cross-provider. Forward it only when explicitly set
        # so each provider keeps its own default (usually 2) otherwise (#1091).
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None and max_retries != "":
            kwargs["max_retries"] = _coerce_max_retries(max_retries)

        return kwargs

    def _provider_kwargs_for(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Thinking/sampling kwargs for a role_models spec (per-spec wins, else
        run-level). vertex_anthropic gets effort/max_tokens/thinking (routed into
        the client's model_kwargs); vertex_gemini/vertex_grok still get only
        sampling kwargs (their thinking-config param names are unverified)."""
        provider = str(spec.get("provider", "")).lower()
        kwargs: dict[str, Any] = {}
        if provider == "google":
            level = spec.get("google_thinking_level", self.config.get("google_thinking_level"))
            if level:
                kwargs["thinking_level"] = level
        elif provider == "openai":
            effort = spec.get("openai_reasoning_effort", self.config.get("openai_reasoning_effort"))
            if effort:
                kwargs["reasoning_effort"] = effort
        elif provider in ("anthropic", "vertex_anthropic"):
            eff = spec.get("anthropic_effort", self.config.get("anthropic_effort"))
            if eff:
                kwargs["effort"] = eff
            if provider == "vertex_anthropic":
                mt = spec.get("anthropic_max_tokens",
                              self.config.get("anthropic_max_tokens"))
                if mt is not None and mt != "":
                    kwargs["max_tokens"] = int(mt)
                th = spec.get("anthropic_thinking",
                              self.config.get("anthropic_thinking"))
                if th:
                    kwargs["thinking"] = th
        temperature = spec.get("temperature", self.config.get("temperature"))
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)
        return kwargs

    def _base_url_for(self, provider: str) -> str | None:
        """Base URL for a provider. None for vertex_* (Gemini/Claude use
        project+location; the Grok client builds its own endpoints/openapi URL);
        the run-level backend_url otherwise (single-provider vendor-direct runs)."""
        if str(provider).lower().startswith("vertex_"):
            return None
        return self.config.get("backend_url")

    def _build_cached(self, provider, model, location, kwargs):
        """Build (or reuse) the LLM for a (provider, model, location, kwargs) key.

        Roles sharing a spec share one client — the two Claude judges, the two
        Gemini debaters, the two Grok debaters each build a single client (one
        Vertex OAuth token fetch), and unspecified roles reuse the quick tier.
        Callbacks are run-global and excluded from the key but passed to the build.
        """
        build_kwargs = dict(kwargs)
        if str(provider).lower().startswith("vertex_"):
            build_kwargs["project"] = self.config.get("vertex_project")
            build_kwargs["location"] = location
        key = (str(provider).lower(), model, location, frozenset(build_kwargs.items()))
        if key not in self._llm_cache:
            if self.callbacks:
                build_kwargs["callbacks"] = self.callbacks
            self._llm_cache[key] = create_llm_client(
                provider, model, base_url=self._base_url_for(provider), **build_kwargs
            ).get_llm()
        return self._llm_cache[key]

    def _llm_for_tier(self, tier: str):
        """Build the tier-default LLM (the backward-compatible quick/deep path)."""
        provider = self.config["llm_provider"]
        model = (
            self.config["deep_think_llm"] if tier == "deep"
            else self.config["quick_think_llm"]
        )
        kwargs = self._get_provider_kwargs()
        location = self.config.get("vertex_location")
        return self._build_cached(provider, model, location, kwargs)

    def _llm_for(self, role: str):
        """Resolve the LLM for a graph role. Falls back to the quick/deep tier
        default when role_models is unset or omits the role (backward compatible)."""
        spec = (self.config.get("role_models") or {}).get(role)
        if spec is None:
            return self._llm_for_tier("deep" if role in DEEP_ROLES else "quick")
        kwargs = self._provider_kwargs_for(spec)
        location = spec.get("location") or self.config.get("vertex_location")
        return self._build_cached(spec["provider"], spec["model"], location, kwargs)

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Deterministic verification snapshot (bound to the analyst
                    # LLM and required by its prompt; must be executable here or
                    # the call fails and the model reports it "unavailable").
                    get_verified_market_snapshot,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_macro_indicators,
                    get_prediction_markets,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        from tradingagents.dataflows.symbol_utils import normalize_symbol

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            stock = yf.Ticker(normalize_symbol(ticker)).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.
        """
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity)

    def _run_signature(self, asset_type: str) -> str:
        """Graph-shape inputs that must invalidate a checkpoint if changed.

        Keyed into the checkpoint thread ID so a resume under a different analyst
        selection, debate/risk depth, or asset mode starts fresh instead of
        silently continuing the previous graph (#1089).
        """
        return "|".join([
            "analysts=" + ",".join(self.selected_analysts),
            f"debate={self.config['max_debate_rounds']}",
            f"risk={self.config['max_risk_discuss_rounds']}",
            f"asset={asset_type}",
        ])

    def propagate(self, company_name, trade_date, asset_type: str = "stock"):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Make macro/global news region-aware: stash this ticker's region so
        # get_global_news_* selects region-appropriate queries (e.g. Bank of
        # Korea / KOSPI for .KS/.KQ instead of only Fed / S&P). None = US/default.
        self.config["news_region"] = news_region_for_ticker(company_name)
        set_config(self.config)

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(asset_type),
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date, asset_type=asset_type)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock"):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM and the
        # deterministically resolved instrument identity for all agents.
        past_context = self.memory_log.get_past_context(company_name)
        instrument_context = self.resolve_instrument_context(company_name, asset_type)
        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            instrument_context=instrument_context,
            position_context=_read_position_context(),
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date+graph-shape resumes; a different
        # date or graph shape starts fresh (#1089).
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date), self._run_signature(asset_type))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            last_printed = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk["messages"]:
                    msg = chunk["messages"][-1]
                    # Nodes after the trader don't append to messages, so the
                    # same trailing message repeats across chunks. Print it only
                    # when it changes (#1027); the trace/state merge is unchanged.
                    signature = (type(msg).__name__, getattr(msg, "content", None))
                    if signature != last_printed:
                        msg.pretty_print()
                        last_printed = signature
                    trace.append(chunk)
            # Streamed chunks are per-node deltas. Merge them so the returned
            # state matches what graph.invoke() yields in the non-debug path.
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        # Scrub the archived copy only -- `final_state` (and therefore the saved
        # report and the operator's view) keeps the model's original wording.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=scrub_account_numbers(
                final_state["final_trade_decision"],
                final_state.get("position_context", ""),
            ),
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(asset_type),
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
