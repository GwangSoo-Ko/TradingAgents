# Embedding / Internalizing TradingAgents

A guide for pulling TradingAgents **source in-tree (vendoring)** into another
project — e.g. `alpha-pulse` — rather than `pip install tradingagents`. It maps
the public surface, the module boundaries you must copy, the configuration
surface, the runtime side-effects that trip embedders, and the dependency
footprint.

> Scope: this is the integration/architecture reference. For end-user CLI usage
> see the top-level `README.md`; for contributor conventions see `CLAUDE.md`.

---

## TL;DR

- **One public entry point.** Construct `TradingAgentsGraph(config=...)` and call
  `propagate(ticker, date)` → returns `(final_state, signal)`, where `signal` is a
  5-tier rating or `"REVIEW"`.
- **Copy `tradingagents/`** (the library). `cli/` is optional — only needed for
  the interactive Typer CLI and for `main.py`'s report header; a host app drives
  the graph directly instead.
- **Config is one dict** (`DEFAULT_CONFIG`) overlaid by `TRADINGAGENTS_*` env
  vars at import time. Pass your own dict to the constructor to fully control it.
- **Runtime side-effects to plan for:** reads provider **API-key env vars**,
  writes under **`~/.tradingagents/`** (decision log, checkpoints, caches,
  reports), and makes **outbound calls** (yfinance + any configured data vendor +
  the chosen LLM provider). All are relocatable/opt-out — see §4.

---

## 1. The public entry point

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()          # or build your own dict (see §3)
ta = TradingAgentsGraph(config=config)  # debug=False, callbacks=None,
                                        # selected_analysts=("market","social","news","fundamentals")
final_state, decision = ta.propagate("NVDA", "2024-05-10")   # (company_name, trade_date, asset_type="stock", portfolio=None)
print(decision)                         # "Buy" / "Overweight" / "Hold" / "Underweight" / "Sell", or "REVIEW"
```

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `TradingAgentsGraph` | `(selected_analysts=("market","social","news","fundamentals"), debug=False, config: dict\|None=None, callbacks: list\|None=None)` | Ctor **has side-effects**: `set_config(config)`, `makedirs(data_cache_dir, results_dir)`, builds every node's LLM (tier defaults + `role_models`), opens `TradingMemoryLog`. |
| `.propagate` | `(company_name, trade_date, asset_type="stock", portfolio=None) -> (final_state: dict, signal: str)` | The one call you need. `trade_date` must be `YYYY-MM-DD` and not in the future, else `ValueError` before any model call. Runs inside `run_config(self.config)` (§3). `final_state` is the full `AgentState` dict (all reports + `final_trade_decision` + `portfolio_decision_obj`); `signal` is the 5-tier rating or `"REVIEW"` when none is readable (guard with `tradingagents.agents.rating.is_review`). |
| `.create_run_state` | `(company_name, trade_date, asset_type="stock", portfolio=None) -> dict` | Settles the ticker's pending decisions, then builds the initial state (past context, instrument identity, `portfolio_context`, `position_context`). `propagate()` and the CLI start here. |
| `.record_decision` | `(company_name, trade_date, final_state)` | Appends the run's `pending` decision-log entry, account figures scrubbed (§4). |
| `.settle_pending` | `(company_name)` | Settles pending entries whose holding window has traded (Reflector LLM call per entry). |
| `.save_reports` | `(final_state, ticker, save_path=None) -> Path` | Writes the per-section + consolidated markdown tree under `results_dir` (default header). Optional. |
| `.process_signal` | `(full_signal) -> str` | Rating from rendered markdown via `tradingagents.agents.rating.parse_rating` (no extra LLM call; `"REVIEW"` when none). |
| `begin_checkpoint` / `checkpoint_input` / `end_checkpoint` / `clear_checkpoint_on_success` | — | For a host that streams the compiled graph itself, as the CLI does. |

`portfolio` is an optional `tradingagents.portfolio.PortfolioContext`
(`load_portfolio(path)` reads one from JSON): the caller's holdings and cash,
rendered for the Trader, the risk debaters and the Portfolio Manager.
`propagate()` also settles and appends decision-log entries and (if
`--checkpoint`-style resume is enabled) manages per-ticker SQLite checkpoints —
see §4.

---

## 1b. Running `main.py` and consuming its output

`main.py` is a **runnable, arg-driven entry**: `python main.py TICKER [DATE]` —
`TICKER` is required (argparse errors if missing), `DATE` is optional
(`YYYY-MM-DD`, validated; Python 3.11+ also accepts `YYYYMMDD`) and **defaults to
today** (process-local date, so set `TZ`). A future date fails the run (rc 1)
before any model call. It runs a full analysis, prints the decision, then writes
the report tree and prints its path:

```
$ python main.py MU 2026-01-15
...                                  # debug trace
Buy
TRADE_PLAN_JSON: {"rating":"Buy","price_target":...,"revision":null}
Report saved: /Users/you/.tradingagents/logs/reports/MU_20260115_140233/complete_report.md
```

The decision line is the Portfolio Manager's typed rating (the same value as
`TRADE_PLAN_JSON.rating`); when its structured call fell back to free text it is
the parsed signal, `REVIEW` when no rating is readable.

> ⚠️ `main.py` carries a **hard-coded run config** (`build_config()` — currently a
> tiered Vertex Claude setup: Opus 5 at effort max for the Research and Portfolio
> Managers via `role_models`, Sonnet 5 at effort high for every other role,
> `max_tokens` 20000, adaptive thinking, Korean output, KR vendor chains, all four
> analysts, `portfolio_notice_when_absent: False`). Shelling out to `main.py` uses
> *that* config; to vary provider/models/language per run, either edit
> `build_config()` or use the import path (mode B) below. After the run `main.py`
> imports `cli.report_meta.build_report_header` and
> `tradingagents.reporting.write_report_tree` (it does not import `cli.main`), so it
> needs the `cli/` package present.

### The report tree (what to parse)

Written under **`{results_dir}/reports/{safe_ticker}_{YYYYMMDD_HHMMSS}/`**
(`results_dir` defaults to `~/.tradingagents/logs`; override with
`TRADINGAGENTS_RESULTS_DIR`, or relocate the whole home with
`TRADINGAGENTS_CACHE_DIR`):

```
complete_report.md          # consolidated report; header has company label + a
                            # per-role provider/model table (the run's provenance)
1_analysts/{market,sentiment,news,fundamentals}.md
2_research/{bull,bear,manager}.md
3_trading/trader.md
4_risk/{aggressive,conservative,neutral}.md
5_portfolio/decision.md     # the Portfolio Manager's final decision (rendered)
```

The **final decision** is available three ways: stdout (the decision line before
`Report saved:` — a `TRADE_PLAN_JSON:` line may follow it),
`5_portfolio/decision.md`, and `final_state["final_trade_decision"]`.
The 5-tier (or `REVIEW`) **signal** is the second return of `propagate()` /
`process_signal()`.

The **machine-readable trade plan** is available two ways:

- `final_state["portfolio_decision_obj"]` — the typed `PortfolioDecision`
  (rating, `price_target`, `time_horizon`, `total_weight_pct`, `stop_loss`,
  `tranches[]`, `exit_target`, `kill_switch`, `revision`), or `None` when the
  Portfolio Manager's structured call fell back to free text.
- `main.py` prints one line `TRADE_PLAN_JSON: {...}` right after the decision
  when that object exists (`executive_summary` / `investment_thesis` excluded —
  they are already in the report). **The line is absent when there is no
  structured plan**; "no plan" is a valid outcome and a consumer must not
  synthesise one by parsing the prose.

#### The plan contract

The tranches are a phased *execution* plan whose direction follows `rating`: on
a Buy/Overweight they scale in, on an Underweight/Sell they scale out. **Read
`rating` before turning an `immediate` tranche into an order.**

**`tranches[]`** — `pct` partitions the move from the *current* position to the
target and sums to **100** across the list (on a Buy the move is the whole
target; on a reduction it is only the amount being reduced). `trigger` is
`immediate` or `conditional`. **Only `immediate` tranches are executable** —
`conditional` ones are for display, and nothing in this framework watches them.

> ⚠️ **Breaking, 2026-08:** `trigger` was `immediate` / `price` / `event`. The
> *kind* of condition now lives in `triggers[].kind`; `price` and `event` are
> rejected by validation. A consumer comparing `trigger == "immediate"` is
> unaffected.

`price_low` / `price_high` are the tranche's execution band. On a `conditional`
tranche the band is frequently just the span between the extreme triggers (a
real run produced `10321~11600`, i.e. stop-to-take-profit), so **do not read it
as an order band unless `trigger == "immediate"`.**

**`tranches[].triggers[]`** — the conditions that fire a `conditional` tranche;
empty on `immediate`. More than one means **whichever comes first wins (OCO)**.
OCO legs are never split across tranches — splitting would double-count the same
quantity while still summing to 100.

| field | meaning |
|---|---|
| `kind` | `take_profit` / `stop` / `trailing` / `event` |
| `price` | **the only tradeable number in this object** — where an order would be placed |
| `trail_pct` | `trailing` only: percent drop from the running high (`8.0` = −8%) |
| `reference_price` | the indicator level the condition watches. **Never place an order at this value.** |
| `reference_label` | what `reference_price` is (`10EMA`, `200SMA`, `range low`, …) |
| `condition` | human-readable text, including technical criteria |

**`exit_target`** — `{kind, remaining_weight_pct}` or null. What a *reduction*
plan converges to: the sell-side counterpart of `total_weight_pct`. `kind` is
`weight` (keep `remaining_weight_pct` % of NAV), `cost_recovery` (sell enough to
take the original capital back out, hold the rest), or `full` (exit entirely).
`Sell` ⇒ `full`; `weight` / `cost_recovery` belong to `Underweight`.

> ⚠️ **`total_weight_pct` is buy-side only** as of 2026-08. It used to double as
> "the remaining exposure to converge down to" on an Underweight; it is now null
> on reductions. A consumer that still reads it for the reduction target gets
> **no target**, which typically degrades to "produce no orders" with a
> plausible-looking reason rather than an error. Read `exit_target` instead.

**`kill_switch`** — `{price, condition}` or null. A full-exit condition watched
independently of the tranche schedule; it is not part of the `pct` partition.
`price` may be null, because the condition can be an event rather than a level
(e.g. a delayed disclosure).

**`revision`** — `{kind, note}` or null: why the decision departs from the plan
the position was opened on (see the Founding Thesis in §4).

Every numeric field above is optional and may be null; a model that cannot
justify a number omits it. Numbers are **all-or-nothing**: only placeholders
(`N/A`, `none`, `-`, `TBD`, empty) become null; any other value that is not a
plain number (`12,050원`, `3%`, `150-160`, `1,5`) fails the whole structured
decision, so there is no plan line rather than a plan with a silently nulled or
reinterpreted number (`PortfolioDecision`'s `_coerce_plan_float`). Treat each one
as something that could become an order — see `TRADINGAGENTS_POSITION_CONTEXT` in
§4 for the account snapshot that lets the Portfolio Manager size a reduction
against what is actually held.

### Two consumption modes for a host app (e.g. alpha-pulse)

- **(A) Shell out + read reports** — matches "run `main.py`, use the reports".
  `subprocess.run([sys.executable, "main.py", ticker], capture_output=True)`, then
  read the path from the `Report saved:` line of stdout (don't glob by timestamp —
  parse the printed path) and consume `complete_report.md` / the per-section files.
  Simplest, but locked to `main.py`'s baked config and pays subprocess + a fresh
  graph build per run.
- **(B) Import + drive directly** — tighter and configurable. Build your own
  `config`, call `final_state, signal = ta.propagate(ticker, date)`, consume the
  structured `final_state` dict (all `*_report` keys + `final_trade_decision`)
  and/or call `ta.save_reports(final_state, ticker)` for the file tree. Preferred
  when alpha-pulse needs per-run config, structured access, or to avoid a
  subprocess. See §1.

---

## 2. Package layout & what to copy

Copy the **`tradingagents/`** package. Copy **`cli/`** only if you want the
interactive terminal UI (Typer/questionary/rich) or run `main.py`; a host app
normally replaces it.

```
tradingagents/
├── graph/          # orchestration — the entry point, settlement, checkpoints
├── agents/         # LLM agent nodes + schemas + tools + context/rating + render helpers
├── dataflows/      # market-data vendor abstraction (router + vendors/)
├── llm_clients/    # provider-agnostic chat-model factory (dependency-free leaf)
├── decision_log.py # TradingMemoryLog (the decision log)
├── reporting.py    # write_report_tree (the report files)
├── portfolio.py    # caller portfolio (PortfolioContext, load_portfolio)
├── backtest.py     # date-grid backtest (CLI `tradingagents backtest`)
└── default_config.py   # the single canonical config dict + env overlay
cli/                # OPTIONAL interactive driver (Typer app); main.py uses cli/report_meta.py
main.py             # OPTIONAL runner (the alpha-pulse entry, §1b)
```

**Dependency direction** (what imports what) — copy in this order, or stub the
arrows you don't want:

```
cli ─────────────┐
                 ▼
graph ──▶ agents ──▶ dataflows ──▶ default_config
   │        │            ▲              ▲
   └────────┴──▶ llm_clients (leaf, imports nothing from tradingagents)
graph ──▶ decision_log ──▶ agents.rating;   graph ──▶ reporting (stdlib only)
```

- `llm_clients/` — **leaf**, imports nothing from `tradingagents`. Vendors alone.
- `dataflows/` — **near-leaf**; only hard coupling is `default_config` (via
  `dataflows/config.py`). One soft/lazy import of
  `agents.context.resolve_instrument_identity` (the CLI's ticker resolver).
- `agents/` — depends on `dataflows` (the tools in `agents/tools.py` funnel through
  `dataflows.router.route_to_vendor`; identity comes from
  `dataflows.vendors.yahoo.fundamentals.get_company_profile`) + LangChain chat
  models. Only `dataflows/` imports `yfinance` (`tests/test_layering.py`).
- `graph/` — the orchestrator; depends on `agents`, `dataflows`, `llm_clients`,
  `decision_log`, `reporting`, `default_config`.

---

## 3. Configuration surface

Two layers, one source of truth:

1. **`tradingagents/default_config.py:DEFAULT_CONFIG`** — the canonical dict,
   built at import time and overlaid with `TRADINGAGENTS_*` env vars (type-coerced
   by `_coerce`; a bad int/bool raises at import). This is the **programmatic
   path** (`main.py` starts from `DEFAULT_CONFIG.copy()`).
2. **`tradingagents/dataflows/config.py`** — what every agent/dataflow reads via
   `get_config()`. `propagate()` wraps its run in `run_config(self.config)`, a
   `ContextVar` scope that LangGraph carries into tool calls, so the data tools read
   that graph's config even when several graphs share a process. Outside such a
   scope `get_config()` returns a **mutable process-global copy**, which
   `TradingAgentsGraph.__init__` updates via `set_config(config)` and the CLI's
   stream path reads; there the last `set_config` wins.

> **`.env` ordering:** `tradingagents/__init__.py` loads `.env` (python-dotenv)
> so that `default_config`'s env overlay sees it. If you drop that `__init__`,
> load your `.env` **before** importing `default_config`.

Config-key groups an embedder cares about (full list in `default_config.py`):

| Group | Keys |
|-------|------|
| **LLM** | `llm_provider`, `deep_think_llm`, `quick_think_llm`, `backend_url`, `temperature`, `llm_max_retries`, `max_tokens`, `role_models`, `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`, `anthropic_max_tokens`, `anthropic_thinking`, `vertex_project`, `vertex_location` |
| **Debate depth** | `max_debate_rounds`, `max_risk_discuss_rounds` |
| **Data routing** | `data_vendors` (category default), `tool_vendors` (per-tool override), `enable_alpha_vantage_price_crosscheck`, news/global-news knobs |
| **Persistence** | `data_cache_dir`, `results_dir`, `memory_log_path`, `memory_log_max_entries`, `checkpoint_enabled` |
| **Settlement** | `holding_period_days`, `benchmark_ticker`, `benchmark_map` (suffix → index, e.g. `.KS` → `^KS11`, `.KQ` → `^KQ11`, default `SPY`) |
| **Behavior** | `output_language`, `enable_kr_discussion_sentiment`, `portfolio_notice_when_absent` |

Every key has a `TRADINGAGENTS_*` env override (see `_ENV_OVERRIDES`), e.g.
`TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_DEEP_THINK_LLM`,
`TRADINGAGENTS_ANTHROPIC_EFFORT`, `TRADINGAGENTS_CACHE_DIR`. The tier LLMs' kwargs
come from `tradingagents.llm_clients.build_llm_kwargs(config)`; a `role_models`
entry's from `TradingAgentsGraph._provider_kwargs_for(spec)`.

---

## 4. Runtime footprint — the vendoring gotcha list

The single most important section for embedding. TradingAgents is **not a pure
function**: it reads env, writes to disk, and calls the network.

### Environment variables
- **Provider API keys** (read lazily when a client is built, via
  `llm_clients/api_key_env.py:PROVIDER_API_KEY_ENV`): `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`,
  `OPENROUTER_API_KEY`, `AZURE_OPENAI_*`, … `OLLAMA_BASE_URL` for local.
- **Data-vendor keys** (only if that vendor is routed): `ALPHA_VANTAGE_API_KEY`,
  `FRED_API_KEY`, `DART_API_KEY` (KR), etc. — vendors quiet-skip when unconfigured.
- **Vertex (optional `[vertex]`):** `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  (default `global`), ADC via `GOOGLE_APPLICATION_CREDENTIALS` — **no vendor key**.
- **Config overlay:** all `TRADINGAGENTS_*` (see §3).
- **`TRADINGAGENTS_POSITION_CONTEXT`** — a one-line JSON account/position
  snapshot, read once per run by `create_run_state()`
  (`graph/trading_graph.py:_read_position_context`) and carried on the
  `position_context` state channel declared in `agents/state.py`. ⚠️ That
  declaration is **load-bearing** — langgraph silently drops any key a node
  returns that the state schema doesn't list, so removing it makes the injection
  vanish with no exception and no warning; plan quality just quietly degrades.
  Invalid JSON is a warning to stderr, not a crash — the run degrades to "no
  position information available", same as unset. The CLI and `backtest` build
  their state through `create_run_state()` as well, so the variable reaches them
  too; unset it for those runs.

  **Only the Portfolio Manager reads it**, via
  `agents/context.py:build_position_block` — it is the last node, so
  nothing it writes can reach another agent's prompt; the isolation is
  structural, not just a prompt instruction. The Trader deliberately does
  **not** read it even though it also sizes a transaction: its `TraderProposal`
  renders into `trader_investment_plan`, which the aggressive/conservative/
  neutral risk debators all embed verbatim, so injecting there would leak the
  account into the three modules whose debate is supposed to run on unbiased
  ground. Analysts and researchers never see it either — knowing a position is
  held biases toward the disposition effect.

  Expected shape (all keys optional; a missing key just narrows what the PM can
  reason about):
  ```json
  {"held_qty":2697,"avg_price":10900,"current_price":10950,
   "unrealized_pnl_pct":0.46,"current_weight_pct":5.90,
   "cash":456535870,"total_nav":500769000,"currency":"KRW"}
  ```
  Without it the PM writes reduction plans blind: it does not know whether
  anything is held, so `exit_target` and the tranche `pct` split are reasoned
  from the prose alone. `record_decision()` applies
  `graph/trading_graph.py:scrub_account_numbers` before the Portfolio Manager's
  rendered output is archived to the decision log, stripping the raw
  `cash`/`total_nav`/`held_qty`/`avg_price` figures back out on every path
  (`propagate()` and the CLI) — the prompt already asks the model not to quote
  them, but the PM's own executive-summary instructions pull toward citing
  position sizing, so the instruction alone isn't reliable enough for an archive
  that `get_past_context` replays into later runs.

  This is a separate channel from upstream's caller portfolio
  (`propagate(portfolio=)` / `tradingagents --portfolio`, state key
  `portfolio_context`), which the Trader and the risk debaters read as well. When
  no caller portfolio is given, those agents (and the PM, when no position is
  injected) are told "Portfolio context: not provided" unless the config sets
  `portfolio_notice_when_absent: False`, as `main.py` does. When a position is
  injected, the PM shows only the position block.

### Filesystem — the `~/.tradingagents/` home tree
Rooted at `~/.tradingagents/`; override the base with **`TRADINGAGENTS_CACHE_DIR`**
(and individual paths via `TRADINGAGENTS_RESULTS_DIR` / `_MEMORY_LOG_PATH`).

| Path | Written by | Purpose |
|------|-----------|---------|
| `memory/trading_memory.md` | `tradingagents/decision_log.py:TradingMemoryLog` | Append-only decision log; the next same-ticker run settles outcomes (`graph/settlement.py`). Always on — the interactive CLI records here too, so give each consumer its own `TRADINGAGENTS_MEMORY_LOG_PATH`. |
| `cache/checkpoints/<TICKER>.db` | `graph/checkpointer.py` | Per-ticker SQLite LangGraph resume. Opt-in (`checkpoint_enabled`). |
| `cache/` | dataflows vendors | yfinance / DART data caches. |
| `results_dir` (logs) | `graph/_log_state`, `reporting.py:write_report_tree` | JSON state logs + markdown reports. |

⚠️ **Tickers become path components** and must go through
`dataflows/symbols.py:safe_ticker_component` (already enforced in `_log_state` and
`checkpointer._db_path`). Preserve that if you touch those paths.

### Outbound network
- **yfinance** (default vendor: prices/indicators/fundamentals/news/Search) — always.
- **Reddit (RSS) + StockTwits** — keyless; fetched by the sentiment analyst every run.
- **Keyed vendors** when routed: Alpha Vantage, FRED, Polymarket, SEC EDGAR, KR (Naver/DART/wisereport).
- **LLM provider APIs** per `llm_provider` (or Vertex via ADC).

### Global state to isolate
`propagate()` scopes its config per run (§3); the process-global copy still backs
the constructor and the CLI path. The decision log and checkpoints are keyed by
ticker on disk. For concurrent/multi-tenant embedding, give each run its own
`TRADINGAGENTS_CACHE_DIR` (and memory-log path), or run in separate processes.

---

## 5. Dependency footprint

**Python ≥ 3.10.** Base runtime deps the code imports:

`langchain-core`, `langchain-openai`, `langchain-anthropic`,
`langchain-google-genai`, `langgraph`, `langgraph-checkpoint-sqlite`, `pandas`,
`yfinance`, `stockstats`, `requests`, `python-dotenv`, `pytz`, `typing-extensions`
— plus CLI-only `typer`, `questionary`, `rich`.

**Optional extras (lazy-imported — file can exist without the package):**
- `[vertex]` → `langchain-google-vertexai`, `anthropic[vertex]` (pulls google-auth)
- `[bedrock]` → `langchain-aws` (pulls boto3)

**Declared but unused:** the fork's `pyproject.toml` still lists `backtrader`,
`redis`, `parsel`, `tqdm`, `langchain-experimental` and `setuptools` — no code
imports them (upstream dropped them in v0.5.1; the fork keeps its pyproject
unchanged until a separate dependency sync, so the production venv is not
rebuilt by a code merge). The
`beautifulsoup4` used by KR `wisereport.py` arrives transitively — declare it if
you keep KR vendors. **`certifi`** (transitive via `requests`) is required by the
stdlib-`urllib` vendors — `dataflows/net.py:default_ssl_context` builds their TLS
context from certifi's CA bundle so reddit/stocktwits work on macOS installs
whose OS CA bundle isn't linked; keep certifi if you vendor those vendors.

**Per-provider LangChain packages** are only needed for the provider you actually
build: keep `langchain-openai` (imported at module top of `openai_client.py`);
`langchain-anthropic` / `langchain-google-genai` are needed only if you keep those
client files. `llm_clients/` couples to **nothing** in `tradingagents`, so you can
vendor just the providers you use.

---

## 6. Subsystem reference

### orchestration — `tradingagents/graph/`
- **Purpose:** the LangGraph pipeline + the single public entry point. Pipeline
  (`setup.py`): Analysts (market→social→news→fundamentals, each with a `ToolNode`
  loop + message-clear node) → Bull/Bear debate → Research Manager → Trader →
  Aggressive/Conservative/Neutral risk debate → Portfolio Manager → END.
- **Key files:** `trading_graph.py` (`TradingAgentsGraph`, `propagate`,
  `create_run_state`, `record_decision`, `settle_pending`, `save_reports`, the
  checkpoint helpers, `_llm_for(role)` + `_llm_for_tier` resolver + client-dedup
  cache, `DEEP_ROLES`, `scrub_account_numbers`), `setup.py` (node wiring),
  `analyst_execution.py` (each analyst's `ToolNode` from its `TOOLS`),
  `conditional_logic.py` (debate/tool routing), `settlement.py` (benchmark,
  returns, `settle_pending`), `reflection.py` (`Reflector`), `checkpointer.py`
  (opt-in SQLite resume). State: `agents/state.py:AgentState`.
- **Public API:** `TradingAgentsGraph`, `DEEP_ROLES`, `ROLE_KEYS`.
- **Deps:** `langgraph`, `langgraph-checkpoint-sqlite`. **Couples to:** `agents`,
  `dataflows` (`run_config`/`set_config`, `safe_ticker_component`,
  `normalize_symbol`, `get_closes`), `llm_clients`, `decision_log`, `reporting`.

### agents — `tradingagents/agents/`
- **Purpose:** the LLM agent "nodes" + Pydantic decision schemas + `@tool`
  wrappers + render helpers. Factory pattern: `create_X(llm) -> node(state) -> dict`.
- **Public API:** `create_{market,sentiment,news,fundamentals}_analyst`,
  `create_{bull,bear}_researcher`, `create_{aggressive,conservative,neutral}_debator`,
  `create_{research_manager,trader,portfolio_manager}`, `create_msg_delete`,
  `AgentState`/`InvestDebateState`/`RiskDebateState`, schemas
  (`ResearchPlan`/`TraderProposal`/`PortfolioDecision`/`SentimentReport`),
  `render_*` helpers, `bind_structured`/`invoke_structured_or_freetext`, the data
  `@tool`s, `build_instrument_context`, `build_position_block`,
  `RATINGS_5_TIER`/`parse_rating`/`is_review`.
- **Key files:** `__init__.py` (re-export surface), `schemas.py`, `structured.py`,
  `context.py` (identity, language, position and portfolio blocks), `state.py`,
  `rating.py`, `tools.py`, the `analysts/` + `managers/` + `researchers/` +
  `risk_mgmt/` + `trader/` node modules.
- **Deps:** `langchain-core`, `langgraph`, `pydantic>=2`. **Couples to:**
  `dataflows.router.route_to_vendor` (all tool data access), `dataflows.config`,
  `dataflows.vendors` (identity lookup, fails open; the sentiment analyst's
  Reddit/StockTwits fetch), `graph` (owns the tool loop).
- **Embed note:** structured agents render typed instances back to a **fixed
  markdown shape** — don't bypass the `render_*` helpers; downstream (rating
  parser, memory log, reports) parse that shape.

### dataflows — `tradingagents/dataflows/`
- **Purpose:** market-data access layer. Two-level vendor router
  (`router.route_to_vendor`) maps six categories → configured vendors, returning
  LLM-ready markdown/CSV. Vendor modules live in `vendors/` (`yahoo/`,
  `alpha_vantage/`, `sec_edgar`, `fred`, `polymarket`, `reddit`, `stocktwits`); the
  fork's KR vendors (`naver_news`, `opendart_*`, `wisereport`, `naver_discussion`)
  sit at the package top level. A leaf subsystem.
- **Public API:** `router.route_to_vendor`/`get_vendor`/`get_category_for_method`,
  `vendors.yahoo.snapshot.build_verified_market_snapshot`,
  `config.set_config`/`get_config`/`run_config`/`initialize_config`,
  error types in `errors.py` (`VendorError`/`NoMarketDataError`/`VendorRateLimitError`…),
  `symbols.normalize_symbol`/`crypto_base`/`safe_ticker_component`,
  `kr_utils.is_kr_ticker`/`to_krx_code`, `vendors.yahoo.ohlcv.load_ohlcv`/`yf_retry`,
  `vendors.reddit.fetch_reddit_posts`/`vendors.stocktwits.fetch_stocktwits_messages`,
  `ticker_resolver.resolve_query`/`looks_like_ticker`, `net.get_scrubbed`/`default_ssl_context`.
- **Deps:** `pandas`, `yfinance`, `stockstats`, `requests`, `python-dateutil`;
  `beautifulsoup4` (KR wisereport only, transitive). Reddit/StockTwits/OpenDART are
  **stdlib-only**.
- **Couples to:** `default_config` (HARD, one import in `config.py`) —
  vendor or replace. Soft/lazy: `agents.context.resolve_instrument_identity`.

### llm_clients — `tradingagents/llm_clients/`
- **Purpose:** `create_llm_client(provider, model, base_url=None, **kwargs)` →
  `BaseLLMClient`; `.get_llm()` reads the API key from env and returns a configured
  LangChain chat model. Providers: `anthropic`; `google`; `azure`; `bedrock`;
  `vertex_gemini`/`vertex_anthropic`/`vertex_grok`; every other key goes to
  `OpenAIClient` through the OpenAI-compatible registry
  (`openai_client.py:OPENAI_COMPATIBLE_PROVIDERS` — `openai`, `xai`, `deepseek`,
  `qwen`, `glm`, `minimax`, `openrouter`, `mistral`, `kimi`, `groq`, `nvidia`,
  `ollama`, `openai_compatible`, …). `build_llm_kwargs(config)` turns a config into
  the client kwargs. The `role_models` multi-model resolver lives in
  `graph/trading_graph.py`, not here.
- **Public API:** `create_llm_client`, `build_llm_kwargs`, `BaseLLMClient.get_llm`,
  `normalize_content`, the provider-string keys, `get_api_key_env`/`PROVIDER_API_KEY_ENV`,
  `get_model_options`/model catalog, `vertex_auth` helpers.
- **Deps:** `langchain-core` (always); `langchain-openai` (module-top in
  `openai_client.py`); others per kept file; `[vertex]`/`[bedrock]` lazy.
- **Couples to:** **NOTHING** in `tradingagents` — pure leaf. Vendors cleanest.

### config_cli — `tradingagents/default_config.py` + `cli/`
- **Purpose:** the config surface + entry scripts. `default_config.py` =
  programmatic path; `cli/` = interactive Typer path: `main.py` (the app: bare
  `tradingagents` = `analyze`, plus `backtest`), `prompts.py` (questions),
  `selections.py` (the selection flow, Vertex/KR choices), `run.py` (builds the
  config, applies presets, streams the graph, writes reports), `display.py`,
  `prefs.py` (last-run answers), `presets.py`, `report_meta.py` (report header).
- **Public API:** `DEFAULT_CONFIG`, `get_config`/`set_config`,
  `apply_vertex_multimodel_config`/`apply_vertex_single_model_config` (presets),
  `write_report_tree`, `build_report_header`, `run_analysis`/`app` (CLI).
  `news_region_for_ticker`.
- **Deps:** `typer`, `questionary`, `rich`, `python-dotenv`, `requests`,
  `langchain-core` (all effectively CLI-side). **Embed note:** a host app skips
  `cli/` and builds `DEFAULT_CONFIG.copy()` + overrides itself.

### runtime_footprint (cross-cutting)
See §4 — the consolidated env / filesystem / network / global-state surface.

---

## 7. Internalization checklist

1. **Copy** `tradingagents/` (the four subpackages plus the top-level
   `default_config.py`, `decision_log.py`, `reporting.py`, `portfolio.py`;
   `backtest.py` only for grid backtests). Add `cli/` only for the interactive UI
   or `main.py`.
2. **Pin deps** from §5; drop the declared-but-unused packages only after
   confirming your routed features don't need them. Add `[vertex]`/`[bedrock]`
   only if used.
3. **Handle `.env` ordering** — load env before importing `default_config`
   (keep `tradingagents/__init__.py` or replicate its dotenv load).
4. **Redirect the home tree** — set `TRADINGAGENTS_CACHE_DIR` to a path your app
   owns; decide whether the decision log / checkpoints belong in your data model.
5. **Provide credentials** — set the provider API-key env var for your
   `llm_provider` (or ADC for Vertex) + any routed data-vendor keys.
6. **Isolate per tenant** — `propagate()` already scopes its config per run; give
   concurrent tenants distinct cache dirs and memory-log paths (or separate
   processes).
7. **Drive it:** `TradingAgentsGraph(config).propagate(ticker, date)`. Consume
   `final_state` (structured reports, `portfolio_decision_obj`) and/or the
   `signal` string (`REVIEW` is not a tradeable rating).
8. **Keep the render seam** — if you extend decision agents, render typed schemas
   to the existing markdown shape (don't bypass `render_*`).

> When the public surface changes (the `propagate` signature, a config key, an
> entry point), update this file — external consumers rely on it.
