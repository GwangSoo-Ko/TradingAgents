# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Install (Python ≥ 3.10):
```bash
pip install .            # editable: pip install -e .
```

Run the framework:
```bash
tradingagents                                  # interactive CLI (a bare command runs the `analyze` callback)
tradingagents --checkpoint                     # opt-in LangGraph resume from per-ticker SQLite
tradingagents --clear-checkpoints              # wipe ~/.tradingagents/cache/checkpoints/*.db first
tradingagents --portfolio book.json            # size decisions against a caller portfolio (holdings + cash)
tradingagents backtest NVDA,AAPL --start 2026-01-05 --end 2026-03-30   # score past decisions over a date grid
python -m cli.main                             # equivalent to running the CLI from source
python main.py TICKER [DATE]                   # the alpha-pulse runner (tiered Vertex Claude, Korean, KR vendors)
```

Tests (pytest config in `pyproject.toml`, `testpaths = ["tests"]`):
```bash
pytest                                  # full suite — runs without API keys or network (conftest stubs them)
pytest -m unit                          # markers: unit / integration / smoke
pytest tests/test_memory_log.py -k append   # single test
TZ=Asia/Seoul python3.11 -m pytest -q   # the production runtime (alpha-pulse runs Python 3.11 in KST)
```

Vertex Claude smoke (live, ADC auth, 2 tiny calls — checks the endpoint accepts the
thinking/effort/max_tokens request shape `VertexAnthropicClient` builds):
```bash
GOOGLE_CLOUD_PROJECT=tpmn-dev python scripts/smoke_vertex_thinking.py
```

Docker (multi-stage; runtime image is `python:3.12-slim` with non-root `appuser`):
```bash
cp .env.example .env                                    # add API keys first
docker compose run --rm tradingagents                   # default
docker compose --profile ollama run --rm tradingagents-ollama  # local models
```

## Architecture

This is a LangGraph-orchestrated multi-agent pipeline. The single entry point is `TradingAgentsGraph.propagate(company_name, trade_date, asset_type="stock", portfolio=None)` in `tradingagents/graph/trading_graph.py`, which returns `(final_state, signal)` — `signal` is a 5-tier rating or `"REVIEW"` when the decision has no readable rating. A future or non-`YYYY-MM-DD` `trade_date` raises `ValueError` before any model call (`_validate_trade_date`). `propagate()` runs inside `dataflows.config.run_config(self.config)`, a per-run scope over the process-wide `set_config` copy that the CLI stream path still reads. `create_run_state()` builds the initial state and `record_decision()` logs the result; `propagate()` and the CLI (`cli/run.py`) both go through them.

**Embedding this package in another project (vendoring):** see `docs/INTEGRATION.md` — the external-consumer integration guide (public surface, module boundaries + dependency direction, full config-key reference, runtime footprint, dependency footprint, internalization checklist). Keep it in sync when the public API (`propagate` signature, config keys, entry points) changes.

### Pipeline (`tradingagents/graph/setup.py`)

Selectable analysts run sequentially, each looping with its `ToolNode` and a message-clear node before passing to the next:

```
Analysts (market → social → news → fundamentals)
   → Bull / Bear Researcher debate (max_debate_rounds)
   → Research Manager (deep LLM, structured output)
   → Trader (quick LLM, structured output)
   → Aggressive / Conservative / Neutral risk debate (max_risk_discuss_rounds)
   → Portfolio Manager (deep LLM, structured output) → END
```

Each analyst module declares its data tools as `TOOLS` (`agents/tools.py`); `graph/analyst_execution.py` turns them into the analyst's `ToolNode`. The sentiment analyst fetches its sources before calling the model and has no tools.

State threads through `AgentState` (`tradingagents/agents/state.py`), a `MessagesState` extension carrying per-section reports, two debate sub-states, `final_trade_decision`, `portfolio_decision_obj` (the PM's typed decision), `past_context` (memory-log injection), `portfolio_context` (caller portfolio) and `position_context` (alpha-pulse account snapshot). langgraph silently drops keys a node returns that the schema does not declare. Conditional routing (debate continuation, tool calls vs. clear) lives in `graph/conditional_logic.py`.

Two LLM tiers: `quick_thinking_llm` for analysts/researchers/risk debaters/Trader, `deep_thinking_llm` for Research Manager and Portfolio Manager. Every node gets its LLM from `TradingAgentsGraph._llm_for(role)`, which falls back to these tiers when `role_models` leaves the role unset (see below).

### Structured-output decision agents (v0.2.4, #434)

Research Manager, Trader, and Portfolio Manager use `llm.with_structured_output(Schema)` and return typed Pydantic instances from `tradingagents/agents/schemas.py`. **The provider-specific mode matters** and is encoded in the agent factories: `json_schema` (OpenAI/xAI/DeepSeek/Qwen/GLM), `response_schema` (Gemini), tool-use (Anthropic), `function_calling` (OpenAI default to silence noisy `PydanticSerializationUnexpectedValue` warnings from langchain-openai's Responses-API parser).

`agents/structured.py:invoke_structured_or_freetext` returns `(markdown, obj)` — `obj` is `None` on the free-text fallback. Render helpers (`render_research_plan`, `render_trader_proposal`, `render_pm_decision`) turn the Pydantic instance back into the legacy markdown shape so the rest of the system (memory log, CLI display, saved reports) keeps working unchanged. **Don't bypass the render helpers** — downstream consumers expect that exact shape. The PM's typed object is `portfolio_decision_obj`, which `main.py` prints as the `TRADE_PLAN_JSON:` line. Its plan numbers are fail-closed (`_coerce_plan_float`: only placeholders become `None`, an unreadable number fails the whole decision); upstream's salvaging `_coerce_optional_float` applies to `TraderProposal` only.

`tradingagents/agents/rating.py:parse_rating` reads the rating from rendered markdown — no extra LLM call. The first line that opens with a `Rating:` label wins (so a rating the prose quotes later does not), else the last label, else a single rating word, else `REVIEW`. `process_signal()` and the memory-log tag use it; `main.py` prints the PM's typed rating when it has one. The 5-tier scale (Buy/Overweight/Hold/Underweight/Sell) is used by Research Manager and Portfolio Manager; Trader keeps 3-tier (Buy/Hold/Sell).

### LLM client factory (`tradingagents/llm_clients/`)

`create_llm_client(provider, model, base_url, **kwargs)` dispatches to a client class. Imports are lazy so test collection doesn't pull heavy SDKs.

- `anthropic` → `AnthropicClient`, `google` → `GoogleClient`, `azure` → `AzureOpenAIClient`, `bedrock` → `BedrockClient`
- `vertex_gemini` / `vertex_anthropic` / `vertex_grok` → `vertex_clients.py` (see below)
- everything else → `OpenAIClient` through the OpenAI-compatible registry (`OPENAI_COMPATIBLE_PROVIDERS` in `openai_client.py`: `openai`, `xai`, `deepseek`, `qwen`, `glm`, `minimax`, `openrouter`, `mistral`, `kimi`, `groq`, `nvidia`, `ollama`, `openai_compatible`, …)

**`backend_url` default is `None`** so each provider falls back to its native endpoint. Setting an OpenAI URL globally previously leaked into Gemini and produced malformed requests — never hardcode a provider URL in `DEFAULT_CONFIG`.

Provider-specific kwargs for the tier LLMs come from `llm_clients/factory.py:build_llm_kwargs(config)`: `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`, the `vertex_anthropic` knobs below, `temperature`, `llm_max_retries`, `max_tokens`. Model catalog (CLI options + validation source of truth) is `llm_clients/model_catalog.py`.

### Multi-model debate via Vertex Model Garden (v0.2.6)

`role_models` (config; default `None` = current quick/deep tier behavior) maps a
graph role to its own `{"provider","model"[,"location",...]}`. The resolver lives
in `trading_graph.py` (`_llm_for(role)` + client dedup keyed on
`(provider, model, location, kwargs)`); `GraphSetup` calls `llm_for(role)` per node
(node factory signatures unchanged). `DEEP_ROLES = {research_manager,
portfolio_manager}`; all other roles default to the quick tier.

Three Vertex providers (`tradingagents/llm_clients/vertex_clients.py`, lazy SDK
imports): `vertex_gemini` (`ChatVertexAI`), `vertex_anthropic`
(`ChatAnthropicVertex`, uses `model_name=`), `vertex_grok` (`ChatOpenAI` against the
Vertex `endpoints/openapi` URL with a Google OAuth token as `api_key`). Auth is
Google ADC/service-account — **no vendor API key** (`vertex_auth.py`). Install the
optional deps with `pip install -e ".[vertex]"`.

Enable from the CLI by picking **"Vertex Model Garden (multi-model debate)"** as the
provider; it applies `cli/presets.py:VERTEX_DEBATE_PRESET` (judges=Claude, debaters
diversified across Gemini/Claude/Grok, analysts+trader=Gemini) and prompts for the
GCP project + location. Required env: `GOOGLE_CLOUD_PROJECT` (e.g. `tpmn-dev`),
optional `GOOGLE_CLOUD_LOCATION` (default `global`), and ADC via
`gcloud auth application-default login` (or `GOOGLE_APPLICATION_CREDENTIALS`). It also
works non-interactively via `TRADINGAGENTS_LLM_PROVIDER=vertex_model_garden` +
`GOOGLE_CLOUD_PROJECT`. **`vertex_anthropic` thinking-config is wired**: config knobs
`anthropic_effort`, `anthropic_max_tokens`, `anthropic_thinking` (env
`TRADINGAGENTS_ANTHROPIC_{EFFORT,MAX_TOKENS,THINKING}`; also per-role in `role_models`)
are computed in `build_llm_kwargs` (tier path) / `_provider_kwargs_for` (role path) and
translated by `VertexAnthropicClient.get_llm` — `max_tokens` direct, `effort`→`output_config.effort`,
`thinking` (`"adaptive"`/`"disabled"` shorthand or a dict)→`model_kwargs.thinking`.
`anthropic_max_tokens` wins over the generic `max_tokens`. All
opt-in (unset ⇒ ChatAnthropicVertex defaults, incl. `max_tokens` 4096). `vertex_gemini`/
`vertex_grok` thinking-config stays deferred (they forward only sampling kwargs).
Don't remove the vendor-direct providers — they stay for single-model runs.

For users without an Anthropic/xAI API key, two CLI options run the **whole
pipeline on a single Vertex-hosted model** (no vendor key, ADC auth): **"Vertex
Model Garden — Claude (claude-opus-5)"** and **"Vertex Model Garden — Grok
(xai/grok-4.3)"**. Their provider key IS the real `vertex_anthropic` /
`vertex_grok` client key; `cli/presets.py:VERTEX_SINGLE_MODELS` maps it to the
fixed model and `apply_vertex_single_model_config` sets `llm_provider` +
quick/deep think models + project/location with `role_models` unset (the normal
single-model path). They also work non-interactively via
`TRADINGAGENTS_LLM_PROVIDER=vertex_anthropic|vertex_grok` + `GOOGLE_CLOUD_PROJECT`.
The multi-model preset's Grok role also uses `xai/grok-4.3`. The CLI flow lives in
`cli/selections.py` (prompts) and `cli/run.py` (applies the presets before the graph is built).

### Account context (alpha-pulse)

Two separate channels. `position_context` is the alpha-pulse account snapshot
(env `TRADINGAGENTS_POSITION_CONTEXT` JSON, read in `create_run_state` when
`position_context_from_env` is on — `main.py` pins it on, the interactive CLI and
`backtest` switch it off); **only the
Portfolio Manager reads it** (`agents/context.py:build_position_block`, Founding
Thesis included), and `record_decision()` scrubs its figures from the archived
memory-log copy (`scrub_account_numbers`). `portfolio_context` is upstream's caller
portfolio (`--portfolio` / `propagate(portfolio=)`), which the Trader and risk
debaters read too. With no caller portfolio, upstream tells those agents
"Portfolio context: not provided"; `portfolio_notice_when_absent: False` (set by
`main.py`) removes that notice.

### Data vendor abstraction (`tradingagents/dataflows/`)

Tools route to vendors through `dataflows/router.py:route_to_vendor` and two-level config:
- `data_vendors` — category default (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`, `macro_data`, `prediction_markets`)
- `tool_vendors` — per-tool override

Vendors live in `dataflows/vendors/` (`yahoo/`, `alpha_vantage/`, `sec_edgar`, `fred`, `polymarket`, `reddit`, `stocktwits`); the fork's KR vendors (`naver_news`, `opendart_fundamentals`, `wisereport`) sit at `dataflows/` top level, are opt-in chain members (`main.py` uses `naver,yfinance` news and `wisereport,yfinance` fundamentals) and raise `VendorOutOfScopeError` (a `NoMarketDataError`) for non-KR tickers, which the router passes over so the covering vendor's verdict (e.g. Yahoo's `DATA_UNAVAILABLE`) stands. `safe_get(..., secret=)` masks OpenDART's query-string key in logs and errors. `naver_discussion` (종목토론방) feeds the sentiment analyst when `enable_kr_discussion_sentiment` is on. The analyst tools are in `agents/tools.py`.

**Latest-close cross-check (verified snapshot).** `dataflows/vendors/yahoo/snapshot.py:build_verified_market_snapshot` (the `get_verified_market_snapshot` tool) cross-checks the primary feed's latest close against Alpha Vantage (`dataflows/vendors/alpha_vantage/stock.py:get_latest_close_on_or_before`, `TIME_SERIES_DAILY` compact, filtered `<= curr_date` so it stays look-ahead-safe). When Alpha Vantage has a more recent close than yfinance — the common case where yfinance lags the latest session (returns a NaN/missing last close) — the snapshot flags the primary feed as STALE and surfaces the newer close. Best-effort: gated by `enable_alpha_vantage_price_crosscheck` (default True; env `TRADINGAGENTS_AV_PRICE_CROSSCHECK`), needs `ALPHA_VANTAGE_API_KEY`, and returns nothing (no behavior change) without a key or on any error. yfinance stays the primary vendor — this only adds a one-call verification, not a vendor switch.

### Persistence

Two independent mechanisms, both rooted at `~/.tradingagents/` (override base dir with `TRADINGAGENTS_CACHE_DIR`):

**Decision log (always on)** — `tradingagents/decision_log.py:TradingMemoryLog`. Append-only markdown at `memory/trading_memory.md` (override with `TRADINGAGENTS_MEMORY_LOG_PATH`). `record_decision()` appends a `pending` entry at the end of each run (one entry per ticker+date). The next same-ticker run settles pending entries in `create_run_state()` via `graph/settlement.py:settle_pending`: realised return + alpha over `holding_period_days` trading days against `benchmark_map` (SPY for US, `^KS11`/`^KQ11` for `.KS`/`.KQ`), `Reflector.reflect_on_final_decision()`, then `batch_update_with_outcomes()`; a reflection error leaves the entry pending. Resolved context is injected into the Portfolio Manager prompt via `get_past_context()`. **Pending entries are never pruned**; only resolved entries respect `memory_log_max_entries`. Hard delimiter is the HTML comment `<!-- ENTRY_END -->` (cannot appear in LLM prose).

**Checkpoint resume (opt-in via `--checkpoint`)** — `graph/checkpointer.py`. Per-ticker SQLite at `cache/checkpoints/<TICKER>.db`. `thread_id(ticker, date, signature)` is a sha256 prefix over the ticker, date and run signature (analysts, debate depths, asset type, portfolio), so the same run resumes and anything else starts fresh. The workflow is compiled with the `SqliteSaver` only when checkpointing is enabled, and the checkpoint is cleared on successful completion.

**Reports** — `tradingagents/reporting.py:write_report_tree(final_state, ticker, save_path, *, header=None)` writes `complete_report.md` plus the per-section files. `main.py` and the CLI pass `header=cli/report_meta.py:build_report_header(...)` (company label + per-role model table).

The old per-agent BM25 memory (`FinancialSituationMemory`) and `reflect_and_remember()` are removed — don't re-introduce per-agent memory; everything goes through `TradingMemoryLog`.

## Conventions to preserve

- **All `open()` calls pass `encoding="utf-8"` explicitly.** This is the Windows cp1252 fix from v0.2.4 (#543, #550, #576). The earlier process-level approach in v0.2.2 didn't actually take effect.
- **Tickers used as path components must go through `safe_ticker_component()`** (`dataflows/symbols.py`) — see `_log_state` and `checkpointer._db_path`. This is the patch from #618.
- **Exchange-qualified tickers** (`7203.T`, `BRK.B`, `.HK`, `.L`, `.TO`) must round-trip unchanged through prompts and tool calls. `build_instrument_context()` in `agents/context.py` enforces this in prompts.
- **Only `tradingagents/dataflows/` imports vendor libraries (`yfinance`)** — enforced by `tests/test_layering.py`. Agents reach data through the router or `agents/context.py`.
- **`output_language`** applies to every agent whose output reaches the saved report (analysts, researchers, debaters, Research Manager, Trader, Portfolio Manager) via `get_language_instruction()`.
- **`risk_manager` was renamed to `portfolio_manager`** in v0.2.2 — match the file/role naming when adding code.
- Cache and log dirs live under `~/.tradingagents/` (not the project dir) — this is the Docker permissions fix (#519).
- Test fixtures in `tests/conftest.py` stub all provider API keys with `placeholder`, blank every `TRADINGAGENTS_*` setting, keep CLI prefs in `tmp_path`, and refuse network access (`_no_network` for sockets, `_no_curl_network` for curl_cffi/yfinance) unless a test is marked `integration`. Tests should not require real keys; stub the model by monkeypatching `create_llm_client` where it is used (e.g. `tradingagents.graph.trading_graph.create_llm_client`).
