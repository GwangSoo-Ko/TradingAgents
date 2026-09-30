import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity
from tradingagents.agents.rating import parse_rating
from tradingagents.dataflows.config import run_config, set_config
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.decision_log import TradingMemoryLog
from tradingagents.default_config import DEFAULT_CONFIG, news_region_for_ticker
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.reporting import write_report_tree

from . import settlement
from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup

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


def _validate_trade_date(trade_date) -> str:
    """The run date as a canonical ``YYYY-MM-DD`` string no later than today."""
    value = str(trade_date)
    try:
        canonical = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
    except ValueError:
        canonical = False
    if not canonical:
        raise ValueError(f"trade_date must be a date in YYYY-MM-DD format, got {trade_date!r}")
    if value > get_current_date():
        raise ValueError(f"trade_date cannot be in the future: {value}")
    return value


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

        set_config(self.config)

        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Per-role LLM resolution with client dedup. role_models (when set) maps a
        # role to its own provider/model; unset roles fall back to the quick/deep
        # tier defaults below, so an unconfigured run behaves exactly as before.
        # The Reflector reuses the quick tier client.
        self._llm_cache = {}
        self.deep_thinking_llm = self._llm_for_tier("deep")
        self.quick_thinking_llm = self._llm_for_tier("quick")

        self.memory_log = TradingMemoryLog(self.config)

        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        # GraphSetup asks the resolver for each node's LLM by role, so a
        # role_models entry reaches its node (the two judges included); a role
        # role_models leaves unset gets its quick/deep tier default the same way.
        self.graph_setup = GraphSetup(
            self._llm_for,
            self.conditional_logic,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)

        # Graph-shape-affecting run choices, kept for the checkpoint signature.
        self.selected_analysts = tuple(selected_analysts)

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None
        self._resuming = False

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
                provider=provider,
                model=model,
                base_url=self._base_url_for(provider),
                **build_kwargs,
            ).get_llm()
        return self._llm_cache[key]

    def _llm_for_tier(self, tier: str):
        """Build the tier-default LLM (the backward-compatible quick/deep path)."""
        provider = self.config["llm_provider"]
        model = (
            self.config["deep_think_llm"] if tier == "deep"
            else self.config["quick_think_llm"]
        )
        kwargs = build_llm_kwargs(self.config)
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

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock",
                                   curr_date: str | None = None) -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.
        """
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity, curr_date)

    def _memory_as_of(self, trade_date) -> str | None:
        """Point-in-time cutoff for past-context lessons (#1251).

        A historical/backtest run (trade date before today) filters lessons to
        those already resolved by the trade date. A current-date run returns
        None, disabling the filter so live behavior and pre-migration entries
        (which have no stored resolution date) are unaffected.
        """
        td = str(trade_date)
        return td if td < datetime.now().strftime("%Y-%m-%d") else None

    def _run_signature(self, asset_type: str, portfolio=None) -> str:
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
            # None, an empty book and a changed book are three different runs.
            f"portfolio={portfolio.fingerprint() if portfolio is not None else 'none'}",
        ])

    def propagate(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.

        Returns ``(final_state, signal)`` where ``signal`` is one of the 5-tier
        ratings (Buy / Overweight / Hold / Underweight / Sell) or ``"REVIEW"``
        when the decision had no parseable rating (#1170); guard with
        ``tradingagents.agents.rating.is_review`` before mapping it to the
        PortfolioRating enum.
        """
        trade_date = _validate_trade_date(trade_date)

        # Make macro/global news region-aware: the run's config carries this
        # ticker's region so get_global_news_* selects region-appropriate queries
        # (e.g. Bank of Korea / KOSPI for .KS/.KQ instead of only Fed / S&P).
        # None = US/default. It lives in the run's scope only: the caller's config
        # dict (DEFAULT_CONFIG itself when none was given) and the process-wide
        # config stay untouched, so no other graph or later run inherits it.
        run_cfg = {**self.config, "news_region": news_region_for_ticker(company_name)}

        # Pending decisions are settled in create_run_state() (via _run_graph),
        # inside the run's config scope.
        with run_config(run_cfg), \
                self.checkpoint_scope(company_name, trade_date, asset_type, portfolio) as thread_id_value:
            return self._run_graph(
                company_name, trade_date, asset_type=asset_type,
                checkpoint_thread_id=thread_id_value, portfolio=portfolio,
            )

    def begin_checkpoint(self, company_name, trade_date, asset_type: str = "stock", portfolio=None) -> str | None:
        """Recompile the graph with a per-ticker checkpointer and return the
        ``thread_id`` to inject into the stream/invoke ``config`` (or ``None``
        when checkpointing is disabled).

        Pair every call with :meth:`end_checkpoint` in a ``finally``. Both
        ``propagate`` (via :meth:`checkpoint_scope`) and the CLI stream path use
        this so ``--checkpoint`` actually resumes (#1249); previously the setup
        lived only inside ``propagate`` and the CLI streamed the checkpointer-less
        graph, making the flag a no-op.
        """
        self._resuming = False
        if not self.config.get("checkpoint_enabled"):
            return None
        signature = self._run_signature(asset_type, portfolio)
        self._checkpointer_ctx = get_checkpointer(self.config["data_cache_dir"], company_name)
        saver = self._checkpointer_ctx.__enter__()
        self.graph = self.workflow.compile(checkpointer=saver)

        step = checkpoint_step(
            self.config["data_cache_dir"], company_name, str(trade_date), signature
        )
        self._resuming = step is not None
        if step is not None:
            logger.info("Resuming from step %d for %s on %s", step, company_name, trade_date)
        else:
            logger.info("Starting fresh for %s on %s", company_name, trade_date)
        return thread_id(company_name, str(trade_date), signature)

    def checkpoint_input(self, init_state):
        """The value to stream/invoke: ``None`` to resume an existing checkpoint,
        else the initial state for a fresh run.

        LangGraph resumes an interrupted thread when invoked with ``None``;
        re-passing the initial state instead appends it through the message
        reducer, duplicating messages in the resumed state (#1249).
        """
        return None if self._resuming else init_state

    def end_checkpoint(self):
        """Restore the plain uncheckpointed graph after a checkpointed run."""
        if self._checkpointer_ctx is not None:
            self._checkpointer_ctx.__exit__(None, None, None)
            self._checkpointer_ctx = None
            self.graph = self.workflow.compile()
        self._resuming = False

    @contextmanager
    def checkpoint_scope(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """Context-manager form of begin/end_checkpoint for the propagate path."""
        try:
            yield self.begin_checkpoint(company_name, trade_date, asset_type, portfolio)
        finally:
            self.end_checkpoint()

    def clear_checkpoint_on_success(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """Drop a completed run's checkpoint so a later run starts fresh (#1249)."""
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(asset_type, portfolio),
            )

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

    def create_run_state(self, company_name, trade_date, asset_type: str = "stock", portfolio=None):
        """Build a run's initial state; propagate() and the CLI both start here.

        Settles this ticker's pending decisions first, then injects the lessons
        known by the trade date for the Portfolio Manager (#1251) and the
        resolved instrument identity for every agent (#814). An entry point that
        assembled the state itself would skip the decision log.
        """
        self.settle_pending(company_name)
        return self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=self.memory_log.get_past_context(
                company_name, as_of=self._memory_as_of(trade_date)
            ),
            instrument_context=self.resolve_instrument_context(company_name, asset_type, trade_date),
            portfolio_context=portfolio.render(company_name) if portfolio is not None else "",
            # The caller's account snapshot (TRADINGAGENTS_POSITION_CONTEXT JSON);
            # only the Portfolio Manager reads it, and record_decision() scrubs it
            # from the archived copy. The CLI and the backtest switch the read off
            # (position_context_from_env), so a variable left in the environment
            # or .env reaches only the runs that are meant to carry an account.
            position_context=(
                _read_position_context()
                if self.config.get("position_context_from_env", True) else ""
            ),
        )

    def settle_pending(self, company_name):
        """Settle this ticker's decisions whose holding window has now traded.

        A run settles the ticker's earlier decisions on its way in, so the most
        recent one stays pending until the next run for that ticker. A caller
        that is done analyzing a ticker (a backtest sweep, a scheduled job) calls
        this to settle it now.
        """
        with run_config(self.config):
            settlement.settle_pending(company_name, self.memory_log, self.reflector, self.config)

    def record_decision(self, company_name, trade_date, final_state):
        """Log a finished run's decision for reflection on the next same-ticker run.

        Scrub the archived copy only -- ``final_state`` (and therefore the saved
        report and the operator's view) keeps the model's original wording.
        propagate() and the CLI both record here, so every archive is scrubbed.
        """
        decision = final_state.get("final_trade_decision")
        if not decision:
            logger.warning("No final decision for %s on %s; nothing logged", company_name, trade_date)
            return
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=scrub_account_numbers(
                decision, final_state.get("position_context", ""),
            ),
        )

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock",
                   checkpoint_thread_id: str | None = None, portfolio=None):
        """Execute the graph and write the resulting state to disk and memory log."""
        init_agent_state = self.create_run_state(company_name, trade_date, asset_type, portfolio)
        args = self.propagator.get_graph_args()

        # Inject the checkpoint thread_id (from checkpoint_scope) so the same
        # ticker+date+graph-shape resumes; a different one starts fresh (#1089).
        if checkpoint_thread_id is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_thread_id

        # None resumes an existing checkpoint; init_agent_state starts fresh (#1249).
        graph_input = self.checkpoint_input(init_agent_state)
        if self.debug:
            trace = []
            last_printed = None
            for chunk in self.graph.stream(graph_input, **args):
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
            final_state = self.graph.invoke(graph_input, **args)

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run
        # (scrubbed of account figures inside record_decision).
        self.record_decision(company_name, trade_date, final_state)

        # Clear checkpoint on successful completion to avoid stale state.
        self.clear_checkpoint_on_success(company_name, trade_date, asset_type, portfolio)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Write a run's final state to JSON under the run's own ticker."""
        entry = {
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

        # A ticker that would escape the results directory is rejected.
        safe_ticker = safe_ticker_component(final_state["company_of_interest"])
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            # Reports can be in any language and this file is read by a person.
            json.dump(entry, f, indent=4, ensure_ascii=False)

    def process_signal(self, full_signal):
        """The decision's 5-tier rating, or REVIEW when it has none."""
        return parse_rating(full_signal)
