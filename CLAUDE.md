# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Install (Python ≥ 3.10):
```bash
pip install .            # editable: pip install -e .
```

Run the framework:
```bash
tradingagents                                  # interactive CLI (Typer single command, no subcommand)
tradingagents --checkpoint                     # opt-in LangGraph resume from per-ticker SQLite
tradingagents --clear-checkpoints              # wipe ~/.tradingagents/cache/checkpoints/*.db first
python -m cli.main                             # equivalent to running the CLI from source
python main.py                                 # programmatic entry point (uses DEFAULT_CONFIG + .env)
python test.py                                 # ad-hoc dataflow probe (yfinance indicator latency)
```

Tests (pytest config in `pyproject.toml`, `testpaths = ["tests"]`):
```bash
pytest                                  # full suite — runs without API keys (conftest stubs them)
pytest -m unit                          # markers: unit / integration / smoke
pytest tests/test_memory_log.py -k append   # single test
```

Provider smoke (exercises the three structured-output agents only — no full propagate, low cost):
```bash
OPENAI_API_KEY=... python scripts/smoke_structured_output.py openai
GOOGLE_API_KEY=... python scripts/smoke_structured_output.py google
ANTHROPIC_API_KEY=... python scripts/smoke_structured_output.py anthropic
```

Docker (multi-stage; runtime image is `python:3.12-slim` with non-root `appuser`):
```bash
cp .env.example .env                                    # add API keys first
docker compose run --rm tradingagents                   # default
docker compose --profile ollama run --rm tradingagents-ollama  # local models
```

## Architecture

This is a LangGraph-orchestrated multi-agent pipeline. The single entry point is `TradingAgentsGraph.propagate(ticker, date)` in `tradingagents/graph/trading_graph.py`, which returns `(final_state, processed_signal)`.

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

State threads through `AgentState` (`tradingagents/agents/utils/agent_states.py`), a `MessagesState` extension carrying per-section reports, two debate sub-states, `final_trade_decision`, and `past_context` (memory-log injection). Conditional routing (debate continuation, tool calls vs. clear) lives in `graph/conditional_logic.py`.

Two LLM tiers are created once and reused: `quick_thinking_llm` for analysts/researchers/risk debaters/Trader, `deep_thinking_llm` for Research Manager and Portfolio Manager.

### Structured-output decision agents (v0.2.4, #434)

Research Manager, Trader, and Portfolio Manager use `llm.with_structured_output(Schema)` and return typed Pydantic instances from `tradingagents/agents/schemas.py`. **The provider-specific mode matters** and is encoded in the agent factories: `json_schema` (OpenAI/xAI/DeepSeek/Qwen/GLM), `response_schema` (Gemini), tool-use (Anthropic), `function_calling` (OpenAI default to silence noisy `PydanticSerializationUnexpectedValue` warnings from langchain-openai's Responses-API parser).

Render helpers (`render_research_plan`, `render_trader_proposal`, `render_portfolio_decision`) turn the Pydantic instance back into the legacy markdown shape so the rest of the system (memory log, CLI display, saved reports) keeps working unchanged. **Don't bypass the render helpers** — downstream consumers expect that exact shape.

`SignalProcessor` (`graph/signal_processing.py`) reads the rating heuristically from the Portfolio Manager's rendered markdown — there is no extra LLM call. The 5-tier scale (Buy/Overweight/Hold/Underweight/Sell) is used by Research Manager and Portfolio Manager; Trader keeps 3-tier (Buy/Hold/Sell).

### LLM client factory (`tradingagents/llm_clients/`)

`create_llm_client(provider, model, base_url, **kwargs)` dispatches to one of four real client classes. Imports are lazy so test collection doesn't pull heavy SDKs.

- `openai`, `xai`, `deepseek`, `qwen`, `glm`, `ollama`, `openrouter` → `OpenAIClient` (OpenAI-compatible chat completions)
- `anthropic` → `AnthropicClient`
- `google` → `GoogleClient`
- `azure` → `AzureOpenAIClient`

**`backend_url` default is `None`** so each provider falls back to its native endpoint. Setting an OpenAI URL globally previously leaked into Gemini and produced malformed requests — never hardcode a provider URL in `DEFAULT_CONFIG`.

Provider-specific thinking config lives in `_get_provider_kwargs()` in `trading_graph.py`: `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`. Model catalog (CLI options + validation source of truth) is `llm_clients/model_catalog.py`.

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
`GOOGLE_CLOUD_PROJECT`. v1 forwards a minimal kwarg set to the Vertex clients;
thinking-config plumbing is deferred. Don't remove the vendor-direct providers —
they stay for single-model runs.

For users without an Anthropic/xAI API key, two CLI options run the **whole
pipeline on a single Vertex-hosted model** (no vendor key, ADC auth): **"Vertex
Model Garden — Claude (claude-opus-4-8)"** and **"Vertex Model Garden — Grok
(xai/grok-4.3)"**. Their provider key IS the real `vertex_anthropic` /
`vertex_grok` client key; `cli/presets.py:VERTEX_SINGLE_MODELS` maps it to the
fixed model and `apply_vertex_single_model_config` sets `llm_provider` +
quick/deep think models + project/location with `role_models` unset (the normal
single-model path). They also work non-interactively via
`TRADINGAGENTS_LLM_PROVIDER=vertex_anthropic|vertex_grok` + `GOOGLE_CLOUD_PROJECT`.
The multi-model preset's Grok role also uses `xai/grok-4.3`.

### Data vendor abstraction (`tradingagents/dataflows/`)

Tools route to vendors through two-level config:
- `data_vendors` — category default (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`)
- `tool_vendors` — per-tool override

Currently `yfinance` and `alpha_vantage`. Tools are created in `agents/utils/agent_utils.py` (re-exporting from `core_stock_tools.py`, `technical_indicators_tools.py`, `fundamental_data_tools.py`, `news_data_tools.py`) and grouped into `ToolNode`s by analyst type in `_create_tool_nodes()`.

**Latest-close cross-check (verified snapshot).** `dataflows/market_data_validator.py:build_verified_market_snapshot` (the `get_verified_market_snapshot` tool) cross-checks the primary feed's latest close against Alpha Vantage (`alpha_vantage_stock.get_latest_close_on_or_before`, `TIME_SERIES_DAILY` compact, filtered `<= curr_date` so it stays look-ahead-safe). When Alpha Vantage has a more recent close than yfinance — the common case where yfinance lags the latest session (returns a NaN/missing last close) — the snapshot flags the primary feed as STALE and surfaces the newer close. Best-effort: gated by `enable_alpha_vantage_price_crosscheck` (default True; env `TRADINGAGENTS_AV_PRICE_CROSSCHECK`), needs `ALPHA_VANTAGE_API_KEY`, and returns nothing (no behavior change) without a key or on any error. yfinance stays the primary vendor — this only adds a one-call verification, not a vendor switch.

### Persistence

Two independent mechanisms, both rooted at `~/.tradingagents/` (override base dir with `TRADINGAGENTS_CACHE_DIR`):

**Decision log (always on)** — `agents/utils/memory.py:TradingMemoryLog`. Append-only markdown at `memory/trading_memory.md` (override with `TRADINGAGENTS_MEMORY_LOG_PATH`). Each `propagate()` ends with `store_decision()` writing a `pending` entry. The next same-ticker run resolves pending entries via `_resolve_pending_entries()`: fetches realised return + alpha vs SPY through yfinance, calls `Reflector.reflect_on_final_decision()`, then `batch_update_with_outcomes()`. Resolved context is injected into the Portfolio Manager prompt via `get_past_context()`. **Pending entries are never pruned**; only resolved entries respect `memory_log_max_entries`. Hard delimiter is the HTML comment `<!-- ENTRY_END -->` (cannot appear in LLM prose).

**Checkpoint resume (opt-in via `--checkpoint`)** — `graph/checkpointer.py`. Per-ticker SQLite at `cache/checkpoints/<TICKER>.db`. `thread_id(ticker, date)` is a sha256 prefix so same ticker+date resumes, different date starts fresh. `propagate()` recompiles the workflow with the `SqliteSaver` only when checkpointing is enabled, and clears the checkpoint on successful completion.

The old per-agent BM25 memory (`FinancialSituationMemory`) and `reflect_and_remember()` are removed — don't re-introduce per-agent memory; everything goes through `TradingMemoryLog`.

## Conventions to preserve

- **All `open()` calls pass `encoding="utf-8"` explicitly.** This is the Windows cp1252 fix from v0.2.4 (#543, #550, #576). The earlier process-level approach in v0.2.2 didn't actually take effect.
- **Tickers used as path components must go through `safe_ticker_component()`** (`dataflows/utils.py`) — see `_log_state` and `checkpointer._db_path`. This is the patch from #618.
- **Exchange-qualified tickers** (`7203.T`, `BRK.B`, `.HK`, `.L`, `.TO`) must round-trip unchanged through prompts and tool calls. `build_instrument_context()` in `agents/utils/agent_utils.py` enforces this in prompts.
- **Internal agent debate stays in English** for reasoning quality; `output_language` only affects user-facing agents (analysts + Portfolio Manager) via `get_language_instruction()`.
- **`risk_manager` was renamed to `portfolio_manager`** in v0.2.2 — match the file/role naming when adding code.
- Cache and log dirs live under `~/.tradingagents/` (not the project dir) — this is the Docker permissions fix (#519).
- Test fixtures in `tests/conftest.py` stub all provider API keys with `placeholder` so the suite runs without credentials. Tests should not require real keys; mock `create_llm_client` via the `mock_llm_client` fixture instead.
