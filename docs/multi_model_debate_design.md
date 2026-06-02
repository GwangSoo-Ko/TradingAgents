# Design: Multi-Model Debate (role→model assignment) + Vertex Model Garden preset

> Status: **DESIGN ONLY — not implemented.** This document is the implementation
> blueprint agreed in design review. Build in the phases at the end.

## 1. Goal & motivation

Raise answer quality by having **different foundation models debate each other**
(rather than one model role-playing every side). The pipeline already contains
debate roles (bull vs bear; aggressive/conservative/neutral risk debaters; two
judge-managers) — today they all run on the **same** model with different
prompts, so they share that model's blind spots and biases. Assigning *diverse*
models to these roles lets each side challenge the others' reasoning and
hallucinations in-context.

**Honest framing (carried from review):** this is the *debate* variant of the
multi-model idea — distinct from the earlier "parallel + ensemble vote" that was
discarded. It is more defensible (adversarial in-context challenge ≠ output
averaging) and architecturally cheap (reuses existing debate nodes). But the
alpha gain is **unproven** in this domain; frontier models partly converge, and
a single judge model re-introduces single-model bias at the decision point. So
the design is built to be **measurable** and **opt-in**, started as a small
pilot — not shipped as a default with assumed benefit.

## 2. Core design principles

1. **Generic engine, not Vertex-specific.** The real feature is a
   **role → model mapping** that works with *any* provider (direct APIs today).
   "Vertex Model Garden" is just a **CLI preset** that fills that mapping with
   Vertex-hosted models. This avoids coupling "access method" to "multi-model"
   and lets us pilot now with direct APIs (GPT vs Claude) before any Vertex work.
2. **Opt-in, fully backward compatible.** When no role mapping is configured, the
   system behaves *exactly* as today (single provider, `quick`/`deep` tiers).
   The mapping is an *override layer* on top of the existing two tiers.
3. **Measurable.** Ship with an A/B path: compare realized alpha (memory log)
   for single-model vs multi-model on the same tickers/dates.
4. **Graceful degradation.** A role's model that fails to build, or lacks
   structured-output support, falls back (to its tier default / to free text)
   exactly like the existing per-agent fallbacks — never blocks `propagate()`.

## 3. Current architecture (the parts this touches)

- `TradingAgentsGraph.__init__` (`tradingagents/graph/trading_graph.py`):
  builds **two** clients via `create_llm_client(provider, model, …)` →
  `self.deep_thinking_llm`, `self.quick_thinking_llm`. Provider is single
  (`config["llm_provider"]`); thinking kwargs from `_get_provider_kwargs()`.
- `GraphSetup` (`tradingagents/graph/setup.py`): receives `quick_thinking_llm`
  and `deep_thinking_llm` and passes them into node factories:
  - **quick tier:** `create_market_analyst`, `create_social_media_analyst`
    (sentiment), `create_news_analyst`, `create_fundamentals_analyst`,
    `create_bull_researcher`, `create_bear_researcher`, `create_trader`,
    `create_aggressive_debator`, `create_conservative_debator`,
    `create_neutral_debator`.
  - **deep tier:** `create_research_manager`, `create_portfolio_manager`.
  - Every factory has the signature `create_X(llm)` (the three decision agents
    additionally call `bind_structured(llm, Schema, name)`).
- `create_llm_client` (`tradingagents/llm_clients/factory.py`): provider →
  client class. Structured-output **mode is a property of each client/model**
  (json_schema / response_schema / tool-use / function_calling), so a
  per-role model "just works" for structured output as long as its client is
  built through the factory. `bind_structured` already falls back to free text
  when a model lacks support.
- CLI: `cli/utils.py:_llm_provider_table()` is the single source of providers;
  `select_llm_provider()` renders it; `select_shallow_thinking_agent` /
  `select_deep_thinking_agent` pick the two models; `cli/main.py:get_user_selections`
  assembles the dict and `run_analysis` maps it into `config`.

## 4. Target design

### 4.1 The role → model mapping (config schema)

New optional config key `role_models` (default `None` → current behavior):

```python
# default_config.py (illustrative)
"role_models": None,   # None/{} => use quick/deep tiers as today (backward compat)
# When set, maps a ROLE KEY to a model spec; unspecified roles fall back to
# their tier default (deep for judges, quick otherwise).
# Example:
# "role_models": {
#     "bull_researcher":     {"provider": "openai",    "model": "gpt-5.5"},
#     "bear_researcher":     {"provider": "anthropic", "model": "claude-opus-4-7"},
#     "research_manager":    {"provider": "google",    "model": "gemini-3.1-pro-preview"},
#     "portfolio_manager":   {"provider": "google",    "model": "gemini-3.1-pro-preview"},
# },
```

**Role keys** (canonical, one per node that can take a model):
`market_analyst`, `sentiment_analyst`, `news_analyst`, `fundamentals_analyst`,
`bull_researcher`, `bear_researcher`, `research_manager`, `trader`,
`aggressive_debator`, `conservative_debator`, `neutral_debator`,
`portfolio_manager`.

A model spec is `{"provider", "model"}` plus optional per-model thinking knobs
(`google_thinking_level` / `openai_reasoning_effort` / `anthropic_effort` /
`temperature`). When omitted, inherit the run-level thinking config.

Tier classification (for fallback of unspecified roles):
`DEEP_ROLES = {"research_manager", "portfolio_manager"}`; everything else is quick.

### 4.2 LLM resolution & client deduplication

In `TradingAgentsGraph.__init__`, replace "build 2 LLMs" with "build an
**LLM-per-role resolver**":

```python
# pseudocode
self._llm_cache = {}                     # (provider, model, frozenset(kwargs)) -> llm
def _llm_for(self, role: str):
    spec = (self.config.get("role_models") or {}).get(role)
    if spec is None:
        # backward-compatible tier default
        provider = self.config["llm_provider"]
        model = self.config["deep_think_llm"] if role in DEEP_ROLES else self.config["quick_think_llm"]
        kwargs = self._get_provider_kwargs()           # existing
    else:
        provider = spec["provider"]; model = spec["model"]
        kwargs = self._provider_kwargs_for(spec)       # per-spec thinking knobs, else run-level
    key = (provider, model, _hashable(kwargs))
    if key not in self._llm_cache:
        self._llm_cache[key] = create_llm_client(provider, model,
                                  base_url=self._base_url_for(provider), **kwargs).get_llm()
    return self._llm_cache[key]
```

- **Dedup** so two roles on the same (provider, model) share one client (saves
  setup cost / connections).
- `base_url` per provider must be resolved per-provider (not the single run-level
  `backend_url`) — reuse the CLI `provider_default_url()` logic or a shared map,
  so a Claude role and a Gemini role each hit their own endpoint. **This is the
  one real subtlety:** today `backend_url` is single; multi-model needs
  per-provider base URLs. Keep `backend_url` as an override only when the run is
  single-provider.

`GraphSetup` then asks the resolver per node instead of holding two LLMs:

```python
# setup.py (illustrative)
analyst_nodes["market"] = create_market_analyst(self.llm_for("market_analyst"))
...
research_manager_node = create_research_manager(self.llm_for("research_manager"))
bull_researcher_node  = create_bull_researcher(self.llm_for("bull_researcher"))
```

`GraphSetup.__init__` changes from `(quick_thinking_llm, deep_thinking_llm, …)`
to `(llm_for: Callable[[str], Any], …)` (or pass the graph and call back). Node
factory signatures are **unchanged** (`create_X(llm)`), so the blast radius is
confined to `trading_graph.py` + `setup.py`.

### 4.3 Structured output in multi-model mode

No special handling needed: the three decision agents already call
`bind_structured(llm, Schema, name)`, and each role's `llm` is built through the
factory, so it carries its own provider-native structured-output mode. If a
role's model lacks support (e.g. an immature Vertex MaaS model), `bind_structured`
returns `None` and the agent uses free text — existing behavior. **Caveat to
verify at impl:** Vertex Grok/Qwen structured-output support; acceptable to rely
on the free-text fallback initially.

### 4.4 The judge-bottleneck (explicit design stance)

`research_manager` and `portfolio_manager` synthesize the debates; a single
judge model re-introduces single-model bias at the decision point. v1 stance:
**make the judges configurable** (they're roles in the mapping) and document the
limitation. Future enhancement (out of v1): **judge rotation or a small judge
panel** (run the PM decision under 2–3 models, reconcile) — only after the basic
pilot shows signal.

### 4.5 Vertex integration layer (prerequisite for the preset)

(From the earlier Vertex investigation — see §7.) Adds providers
`vertex_anthropic` (and `vertex_grok` / `vertex_qwen` / Gemini-on-Vertex) backed
by `langchain-google-vertexai` + GCP auth (ADC or service-account JSON),
`project` + `location`, and Vertex model names (`claude-…@date`, MaaS ids). The
multi-model engine (§4.1–4.2) is **independent** of this and can ship first with
direct-API providers.

### 4.6 CLI: "Vertex Model Garden" as a preset

- Add one row to `_llm_provider_table()`:
  `("Vertex Model Garden (multi-model debate)", "vertex_model_garden", None)`.
- In `get_user_selections`, when the chosen provider is `vertex_model_garden`:
  - **Skip** the single `select_shallow/deep_thinking_agent` prompts.
  - Apply a **default role→Vertex-model preset** (a constant, e.g.
    `VERTEX_DEBATE_PRESET`), optionally letting advanced users override per role.
  - Collect Vertex auth/config: `project`, `location` (and confirm ADC or a
    service-account path).
  - Print a clear **cost/latency notice** (N models, slower, pricier).
- In `run_analysis`, when `vertex_model_garden` selected, set
  `config["role_models"] = <resolved preset/override>` and the Vertex auth keys;
  otherwise leave `role_models` unset → existing single-model path.

Any non-`vertex_model_garden` choice is **completely unchanged**.

Illustrative preset (exact models chosen at impl, balancing cost/diversity):

```python
VERTEX_DEBATE_PRESET = {
    "bull_researcher":   {"provider": "vertex_anthropic", "model": "claude-sonnet-4-5@..."},
    "bear_researcher":   {"provider": "vertex_grok",      "model": "grok-..."},
    "aggressive_debator":   {"provider": "google",        "model": "gemini-3.1-pro-preview"},   # via Vertex
    "conservative_debator": {"provider": "vertex_qwen",   "model": "qwen-..."},
    "neutral_debator":      {"provider": "vertex_anthropic", "model": "claude-haiku-…@..."},
    "research_manager":  {"provider": "google",           "model": "gemini-3.1-pro-preview"},   # judge
    "portfolio_manager": {"provider": "google",           "model": "gemini-3.1-pro-preview"},   # judge
    # analysts + trader: omitted -> fall back to a designated default model
}
```

## 5. Measurement / evaluation framework (required, not optional)

Because the quality gain is unproven, ship the ability to measure it:

- **A/B harness** (`scripts/`): run the same {ticker, date} set under (a) a
  single-model config and (b) a multi-model config; compare realized alpha vs
  the benchmark (already computed by the memory-log resolution layer) over N
  decisions. Tag runs (e.g. a `run_label` in the memory-log entry) so resolved
  alpha can be grouped single vs multi.
- **Honest power caveat:** decision-level alpha is noisy; tens–hundreds of
  resolved decisions are needed for any signal. Document that early results are
  directional only.
- **Disagreement as a secondary signal** (cheap to capture): log when the debate
  models disagree (e.g. bull/bear or risk debaters reach opposite ratings) — a
  candidate uncertainty/position-sizing signal independent of alpha.

## 6. Phased implementation plan

1. **Phase 1 — generic role→model engine (direct APIs).** `role_models` config +
   `_llm_for` resolver + client dedup + per-provider base_url + `GraphSetup`
   wiring. Backward compatible (unset → identical to today). Unit tests for
   resolution & dedup. **Pilot:** bull=GPT / bear=Claude, rest single; everything
   else default.
2. **Phase 2 — measurement harness.** A/B script + run tagging in the memory log.
   Gather a baseline before investing in Vertex.
3. **Phase 3 — Vertex provider integration.** `langchain-google-vertexai`
   dependency, `vertex_anthropic` (+ grok/qwen) clients, GCP auth + project/
   location config, Vertex model names in the catalog. Mock tests; live
   verification needs the user's GCP project + Model-Garden enablement.
4. **Phase 4 — CLI "Vertex Model Garden" preset.** Provider-table entry + preset
   + role-mapping UX + cost notice.
5. **Phase 5 (optional) — judge rotation/panel** if Phase 2 shows signal.

## 7. Risks & open questions

- **Per-provider base_url:** the run-level single `backend_url` must become
  per-provider in multi-model mode (resolver concern, §4.2).
- **Model convergence:** frontier models may "debate" but converge — diversity
  gain uncertain. Mitigated by measurement, not assumption.
- **Judge bottleneck:** single judge limits the benefit (§4.4).
- **Vertex maturity:** Claude-on-Vertex is solid; Grok/Qwen MaaS integration +
  structured-output support must be verified; region/model availability varies.
  Token-based Vertex auth (vs static keys) for OpenAI-compatible MaaS endpoints
  may need refresh handling.
- **Cost / latency / rate limits:** N distinct models multiply cost and failure
  surface; debate rounds amplify it. Surface cost at CLI selection.
- **Determinism:** more models = less reproducibility; pair with low temperature
  where supported.
- **Verification:** alpha attribution is statistically hard — set expectations.

## 8. Backward-compatibility guarantees

- `role_models` unset (default) ⇒ byte-for-byte current behavior; existing tests
  and single-provider runs unaffected.
- Node factory signatures unchanged; the change is confined to
  `trading_graph.py` (LLM resolver) and `setup.py` (per-role wiring), plus
  additive CLI/config and the (separate) Vertex provider layer.
