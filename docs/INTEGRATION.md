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
  `propagate(ticker, date)` → returns `(final_state, processed_signal)`.
- **Copy `tradingagents/`** (the library). `cli/` is optional — only needed for
  the interactive Typer CLI; a host app drives the graph directly instead.
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
final_state, decision = ta.propagate("NVDA", "2024-05-10")   # (company_name, trade_date, asset_type="stock")
print(decision)                         # "Buy" / "Hold" / "Sell" (heuristic signal string)
```

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `TradingAgentsGraph` | `(selected_analysts=("market","social","news","fundamentals"), debug=False, config: dict\|None=None, callbacks: list\|None=None)` | Ctor **has side-effects**: `set_config(config)`, `makedirs(data_cache_dir, results_dir)`, builds the two LLM tiers, opens `TradingMemoryLog`. |
| `.propagate` | `(company_name, trade_date, asset_type="stock") -> (final_state: dict, processed_signal: str)` | The one call you need. `final_state` is the full `AgentState` dict (all reports + `final_trade_decision`); `processed_signal` is the heuristic Buy/Hold/Sell. |
| `.save_reports` | `(final_state, ticker, save_path=None) -> Path` | Writes the per-section + consolidated markdown tree under `results_dir`. Optional. |
| `.process_signal` | `(full_signal) -> str` | Extract the rating from rendered markdown (no extra LLM call; via `SignalProcessor`). |

`propagate()` also appends a `pending` entry to the decision log and (if
`--checkpoint`-style resume is enabled) manages per-ticker SQLite checkpoints —
see §4.

---

## 1b. Running `main.py` and consuming its output

`main.py` is a **runnable, arg-driven entry**: `python main.py TICKER [DATE]` —
`TICKER` is required (argparse errors if missing), `DATE` is optional
(`YYYY-MM-DD`, validated) and **defaults to today**. It runs a full analysis,
prints the decision, then writes the report tree and prints its path:

```
$ python main.py MU 2026-01-15
Buy
Report saved: /Users/you/.tradingagents/logs/reports/MU_20260115_140233/complete_report.md
```

> ⚠️ `main.py` carries a **hard-coded run config** (`build_config()` — currently a
> tiered Vertex Claude setup: Sonnet 5 quick / Opus 4.8 deep judges, Korean
> output, all four analysts). Shelling out to `main.py` uses *that* config; to
> vary provider/models/language per run, either edit `build_config()` or use the
> import path (mode B) below. `main.py` imports `cli.main` (for the rich report
> writer), so it needs the `cli/` package present.

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
The heuristic **Buy/Hold/Sell** signal is the second return of `propagate()` /
`process_signal()`.

The **machine-readable trade plan** is available two ways:

- `final_state["portfolio_decision_obj"]` — the typed `PortfolioDecision`
  (rating, `price_target`, `time_horizon`, `total_weight_pct`, `stop_loss`,
  `tranches[]`), or `None` when the Portfolio Manager's structured call fell
  back to free text.
- `main.py` prints one line `TRADE_PLAN_JSON: {...}` right after the decision
  when that object exists (`executive_summary` / `investment_thesis` excluded —
  they are already in the report). **The line is absent when there is no
  structured plan**; "no plan" is a valid outcome and a consumer must not
  synthesise one by parsing the prose.

`tranches[].trigger` is one of `immediate` / `price` / `event`. The tranches are
a phased *execution* plan whose direction follows `rating`: on a Buy/Overweight
they scale in, on an Underweight/Sell they scale out. Read `rating` before
turning an `immediate` tranche into an order.

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
interactive terminal UI (Typer/questionary/rich); a host app normally replaces it.

```
tradingagents/
├── graph/          # orchestration — the entry point lives here
├── agents/         # LLM agent nodes + schemas + tools + render helpers
├── dataflows/      # market-data vendor abstraction (leaf; internalizes cleanly)
├── llm_clients/    # provider-agnostic chat-model factory (dependency-free leaf)
└── default_config.py   # the single canonical config dict + env overlay
cli/                # OPTIONAL interactive driver (Typer app)
main.py             # OPTIONAL programmatic entry example
```

**Dependency direction** (what imports what) — copy in this order, or stub the
arrows you don't want:

```
cli ─────────────┐
                 ▼
graph ──▶ agents ──▶ dataflows ──▶ default_config
   │        │            ▲              ▲
   └────────┴──▶ llm_clients (leaf, imports nothing from tradingagents)
```

- `llm_clients/` — **leaf**, imports nothing from `tradingagents`. Vendors alone.
- `dataflows/` — **near-leaf**; only hard coupling is `default_config` (via
  `dataflows/config.py`). One soft/lazy import of `agents…resolve_instrument_identity`.
- `agents/` — depends on `dataflows` (all data access funnels through
  `dataflows.interface.route_to_vendor`) + LangChain chat models.
- `graph/` — the orchestrator; depends on `agents`, `dataflows`, `llm_clients`,
  `default_config`.

---

## 3. Configuration surface

Two layers, one source of truth:

1. **`tradingagents/default_config.py:DEFAULT_CONFIG`** — the canonical dict,
   built at import time and overlaid with `TRADINGAGENTS_*` env vars (type-coerced
   by `_coerce`; a bad int/bool raises at import). This is the **programmatic
   path** (`main.py` uses `DEFAULT_CONFIG.copy()`).
2. **`tradingagents/dataflows/config.py`** — a **mutable process-global copy**
   that every agent/dataflow reads via `get_config()`. `TradingAgentsGraph.__init__`
   pushes the run config into it via `set_config(config)`. ⚠️ This is a **global
   singleton** — a host app running multiple graphs in one process shares it; the
   last `set_config` wins. Isolate per-run if you parallelize.

> **`.env` ordering:** `tradingagents/__init__.py` loads `.env` (python-dotenv)
> so that `default_config`'s env overlay sees it. If you drop that `__init__`,
> load your `.env` **before** importing `default_config`.

Config-key groups an embedder cares about (full list in `default_config.py`):

| Group | Keys |
|-------|------|
| **LLM** | `llm_provider`, `deep_think_llm`, `quick_think_llm`, `backend_url`, `temperature`, `llm_max_retries`, `role_models`, `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`, `anthropic_max_tokens`, `anthropic_thinking`, `vertex_project`, `vertex_location` |
| **Debate depth** | `max_debate_rounds`, `max_risk_discuss_rounds` |
| **Data routing** | `data_vendors` (category default), `tool_vendors` (per-tool override), `enable_alpha_vantage_price_crosscheck`, news/global-news knobs |
| **Persistence** | `data_cache_dir`, `results_dir`, `memory_log_path`, `memory_log_max_entries`, `checkpoint_enabled` |
| **Behavior** | `output_language`, `benchmark_ticker`, `enable_kr_discussion_sentiment` |

Every key has a `TRADINGAGENTS_*` env override (see `_ENV_OVERRIDES`), e.g.
`TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_DEEP_THINK_LLM`,
`TRADINGAGENTS_ANTHROPIC_EFFORT`, `TRADINGAGENTS_CACHE_DIR`.

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

### Filesystem — the `~/.tradingagents/` home tree
Rooted at `~/.tradingagents/`; override the base with **`TRADINGAGENTS_CACHE_DIR`**
(and individual paths via `TRADINGAGENTS_RESULTS_DIR` / `_MEMORY_LOG_PATH`).

| Path | Written by | Purpose |
|------|-----------|---------|
| `memory/trading_memory.md` | `agents/utils/memory.py:TradingMemoryLog` | Append-only decision log; next same-ticker run resolves outcomes. Always on. |
| `cache/checkpoints/<TICKER>.db` | `graph/checkpointer.py` | Per-ticker SQLite LangGraph resume. Opt-in (`checkpoint_enabled`). |
| `cache/` | dataflows vendors | yfinance / DART data caches. |
| `results_dir` (logs) | `graph/_log_state`, `reporting.py:write_report_tree` | JSON state logs + markdown reports. |

⚠️ **Tickers become path components** and must go through
`dataflows/utils.py:safe_ticker_component` (already enforced in `_log_state` and
`checkpointer._db_path`). Preserve that if you touch those paths.

### Outbound network
- **yfinance** (default vendor: prices/indicators/fundamentals/news/Search) — always.
- **Keyed vendors** when routed: Alpha Vantage, FRED, Polymarket, KR (Naver/DART/wisereport).
- **LLM provider APIs** per `llm_provider` (or Vertex via ADC).

### Global state to isolate
`dataflows.config` is process-global (see §3). The decision log and checkpoints
are keyed by ticker on disk. For concurrent/multi-tenant embedding, give each run
its own `TRADINGAGENTS_CACHE_DIR` and serialize `set_config` or run in separate
processes.

---

## 5. Dependency footprint

**Python ≥ 3.10.** Base runtime deps (`pyproject.toml`):

`langchain-core`, `langchain-openai`, `langchain-anthropic`,
`langchain-google-genai`, `langgraph`, `langgraph-checkpoint-sqlite`, `pandas`,
`yfinance`, `stockstats`, `requests`, `python-dotenv`, `pytz`, `typing-extensions`,
`tqdm` — plus CLI-only `typer`, `questionary`, `rich`.

**Optional extras (lazy-imported — file can exist without the package):**
- `[vertex]` → `langchain-google-vertexai`, `anthropic[vertex]` (pulls google-auth)
- `[bedrock]` → `langchain-aws` (pulls boto3)

**Trim-for-embed candidates** (present in `pyproject` but not on the core
`propagate` path — verify against your routed features before dropping):
`backtrader`, `redis`, `parsel`, `langchain-experimental`, `setuptools`. The
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
  `save_reports`, `_llm_for(role)` + `_llm_for_tier` resolver + client-dedup cache,
  `DEEP_ROLES`), `setup.py` (node wiring), `conditional_logic.py` (debate/tool
  routing), `signal_processing.py` (`SignalProcessor`, heuristic rating),
  `checkpointer.py` (opt-in SQLite resume). State: `agents/utils/agent_states.py:AgentState`.
- **Public API:** `TradingAgentsGraph`, `SignalProcessor`, `DEEP_ROLES`, `ROLE_KEYS`.
- **Deps:** `langgraph`, `langgraph-checkpoint-sqlite`. **Couples to:** `agents`,
  `dataflows` (`set_config`, `safe_ticker_component`, `normalize_symbol`), `llm_clients`.

### agents — `tradingagents/agents/`
- **Purpose:** the LLM agent "nodes" + Pydantic decision schemas + `@tool`
  wrappers + render helpers. Factory pattern: `create_X(llm) -> node(state) -> dict`.
- **Public API:** `create_{market,sentiment,news,fundamentals}_analyst`,
  `create_{bull,bear}_researcher`, `create_{aggressive,conservative,neutral}_debator`,
  `create_{research_manager,trader,portfolio_manager}`, `create_msg_delete`,
  `AgentState`/`InvestDebateState`/`RiskDebateState`, schemas
  (`ResearchPlan`/`TraderProposal`/`PortfolioDecision`/`SentimentReport`),
  `render_*` helpers, `bind_structured`/`invoke_structured_or_freetext`, the data
  `@tool`s, `build_instrument_context`, `RATINGS_5_TIER`/`parse_rating`.
- **Key files:** `__init__.py` (re-export surface), `schemas.py`, `utils/structured.py`,
  `utils/agent_utils.py`, `utils/agent_states.py`, `utils/rating.py`, the `analysts/`
  + `managers/` + `researchers/` + `risk_mgmt/` + `trader/` node modules.
- **Deps:** `langchain-core`, `langgraph`, `pydantic>=2`, `yfinance` (identity
  lookup, fails open). **Couples to:** `dataflows.interface.route_to_vendor` (all
  data access), `dataflows.config`, `graph` (owns the tool loop), `agents.utils.memory`.
- **Embed note:** structured agents render typed instances back to a **fixed
  markdown shape** — don't bypass the `render_*` helpers; downstream (signal
  processor, memory log, reports) parse that shape.

### dataflows — `tradingagents/dataflows/`
- **Purpose:** market-data access layer. Two-level vendor router
  (`interface.route_to_vendor`) maps six categories → configured vendors, returning
  LLM-ready markdown/CSV. A leaf subsystem.
- **Public API:** `route_to_vendor`, `build_verified_market_snapshot`,
  `set_config`/`get_config`/`initialize_config`, `get_vendor`/`get_category_for_method`,
  error types (`VendorError`/`NoMarketDataError`/`VendorRateLimitError`…),
  `normalize_symbol`/`crypto_base`, `is_kr_ticker`/`to_krx_code`, `load_ohlcv`,
  `StockstatsUtils`, `fetch_reddit_posts`/`fetch_stocktwits_messages`,
  `resolve_query`/`looks_like_ticker`, `safe_ticker_component`.
- **Deps:** `pandas`, `yfinance`, `stockstats`, `requests`, `python-dateutil`;
  `beautifulsoup4` (KR wisereport only, transitive). Reddit/StockTwits/OpenDART are
  **stdlib-only**.
- **Couples to:** `default_config` (HARD, one import in `config.py`) —
  vendor or replace. Soft/lazy: `agents…resolve_instrument_identity`.

### llm_clients — `tradingagents/llm_clients/`
- **Purpose:** `create_llm_client(provider, model, base_url=None, **kwargs)` →
  `BaseLLMClient`; `.get_llm()` reads the API key from env and returns a configured
  LangChain chat model. Providers: `openai`/`xai`/`deepseek`/`qwen`/`glm`/`ollama`/
  `openrouter` → `OpenAIClient`; `anthropic`; `google`; `azure`; `bedrock`;
  `vertex_gemini`/`vertex_anthropic`/`vertex_grok`. `role_models` multi-model
  resolver lives in `graph/trading_graph.py`, not here.
- **Public API:** `create_llm_client`, `BaseLLMClient.get_llm`, `normalize_content`,
  the provider-string keys, `get_api_key_env`/`PROVIDER_API_KEY_ENV`,
  `get_model_options`/model catalog, `vertex_auth` helpers.
- **Deps:** `langchain-core` (always); `langchain-openai` (module-top in
  `openai_client.py`); others per kept file; `[vertex]`/`[bedrock]` lazy.
- **Couples to:** **NOTHING** in `tradingagents` — pure leaf. Vendors cleanest.

### config_cli — `tradingagents/default_config.py` + `cli/`
- **Purpose:** the config surface + entry scripts. `default_config.py` =
  programmatic path; `cli/` = interactive Typer path (provider/model/key prompts,
  vertex presets, progress UI, report writing).
- **Public API:** `DEFAULT_CONFIG`, `get_config`/`set_config`,
  `apply_vertex_multimodel_config`/`apply_vertex_single_model_config` (presets),
  `write_report_tree`, `run_analysis`/`app` (CLI). `news_region_for_ticker`.
- **Deps:** `typer`, `questionary`, `rich`, `python-dotenv`, `requests`,
  `langchain-core` (all effectively CLI-side). **Embed note:** a host app skips
  `cli/` and builds `DEFAULT_CONFIG.copy()` + overrides itself.

### runtime_footprint (cross-cutting)
See §4 — the consolidated env / filesystem / network / global-state surface.

---

## 7. Internalization checklist

1. **Copy** `tradingagents/` (all four subpackages + `default_config.py`). Add
   `cli/` only for the interactive UI.
2. **Pin deps** from §5; drop the trim candidates only after confirming your
   routed features don't need them. Add `[vertex]`/`[bedrock]` only if used.
3. **Handle `.env` ordering** — load env before importing `default_config`
   (keep `tradingagents/__init__.py` or replicate its dotenv load).
4. **Redirect the home tree** — set `TRADINGAGENTS_CACHE_DIR` to a path your app
   owns; decide whether the decision log / checkpoints belong in your data model.
5. **Provide credentials** — set the provider API-key env var for your
   `llm_provider` (or ADC for Vertex) + any routed data-vendor keys.
6. **Isolate the global config** if you run graphs concurrently (separate
   processes or serialized `set_config`; distinct cache dirs).
7. **Drive it:** `TradingAgentsGraph(config).propagate(ticker, date)`. Consume
   `final_state` (structured reports) and/or the `processed_signal` string.
8. **Keep the render seam** — if you extend decision agents, render typed schemas
   to the existing markdown shape (don't bypass `render_*`).

> When the public surface changes (the `propagate` signature, a config key, an
> entry point), update this file — external consumers rely on it.
