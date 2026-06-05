# Design: Vertex Model Garden multi-model debate (Gemini / Claude / Grok)

> Status: **DESIGN — approved in brainstorming, not yet implemented.**
> Supersedes the generic blueprint `docs/multi_model_debate_design.md` for *this*
> implementation: the same role→model engine, now constrained to **Vertex Model
> Garden as the only model-access path** (no vendor-direct APIs).

## 1. Goal & constraint

Raise answer quality by having **different foundation models debate each other**
in the existing debate roles (bull vs bear; aggressive/conservative/neutral risk
debaters; the two judge-managers), instead of one model role-playing every side.

**Hard constraint (from the user):** every model must be reached through **Google
Cloud Vertex AI Model Garden / Agent platform**. Vendor-direct APIs
(api.openai.com, api.anthropic.com, the Gemini Developer API, api.x.ai) **must not
be used**. GCP project for live use: `tpmn-dev`.

This moves the Vertex provider layer from "later phase" (in the generic doc) to a
**prerequisite delivered in this work**.

## 2. Scope

**In scope (this deliverable):**
1. Vertex provider layer: 3 new providers in the LLM client factory —
   `vertex_gemini`, `vertex_anthropic`, `vertex_grok` — plus a shared Vertex auth
   helper.
2. Generic role→model engine: `role_models` config + per-role LLM resolver +
   client dedup + per-model `project`/`location`, wired through `GraphSetup`.
   Backward compatible (unset ⇒ byte-for-byte current behavior).
3. CLI: a `Vertex Model Garden (multi-model debate)` provider entry that applies
   the debate preset, collects `project`/`location`, and prints a cost notice.
4. Tests: unit/mocked for clients, resolver, dedup, preset, backward compat.

**Out of scope (deferred, noted in §13):**
- A/B alpha-measurement harness (generic doc §5).
- Judge rotation / judge panel (generic doc §4.4).
- Qwen on Vertex (explicitly dropped by the user).
- Removing or changing the existing vendor-direct providers (kept intact for
  tests + upstream parity; simply unused in Vertex mode).

## 3. Model & role mapping (concrete)

Three models, exact Vertex Model Garden IDs (provided by the user):

| Family | Provider key | Model ID | LangChain class | Structured output |
|---|---|---|---|---|
| Gemini | `vertex_gemini` | `gemini-3.5-flash` | `ChatVertexAI` | function_calling / json |
| Claude | `vertex_anthropic` | `claude-opus-4-8` | `ChatAnthropicVertex` | tool-use |
| Grok | `vertex_grok` | `xai/grok-4.20-reasoning` | `ChatOpenAI` → Vertex `endpoints/openapi` | json_schema / function_calling |

**`VERTEX_DEBATE_PRESET` (role → model):**

```python
VERTEX_DEBATE_PRESET = {
    # debaters — diversified (all three families appear; bull↔bear and the
    # risk trio are each on different models). Grok is served on the Vertex
    # `global` endpoint, so its entries pin location explicitly.
    "bull_researcher":      {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "bear_researcher":      {"provider": "vertex_grok",      "model": "xai/grok-4.20-reasoning", "location": "global"},
    "aggressive_debator":   {"provider": "vertex_grok",      "model": "xai/grok-4.20-reasoning", "location": "global"},
    "conservative_debator": {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "neutral_debator":      {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    # judges — Claude (user's choice)
    "research_manager":     {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    "portfolio_manager":    {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    # market/sentiment/news/fundamentals analysts + trader: omitted ->
    # fall back to the quick tier default = vertex_gemini / gemini-3.5-flash
}
```

Unspecified roles reuse the **existing quick/deep tier fallback**: the CLI sets
`llm_provider="vertex_gemini"`, `quick_think_llm="gemini-3.5-flash"`,
`deep_think_llm="gemini-3.5-flash"`, so the 4 analysts + trader resolve to Gemini
with no special-casing. (The two deep-tier judges are explicitly overridden to
Claude in the preset, so the deep default is irrelevant for them.)

**Structured-output safety property:** the three structured-output decision
agents — `research_manager`, `portfolio_manager` (Claude, tool-use) and `trader`
(Gemini, function_calling) — all land on the **native LangChain Vertex classes**,
which support `with_structured_output()` reliably. Grok only does **free-form
debate** (bear / aggressive), so Grok's MaaS structured-output maturity is **not
on the critical path**; if it were ever needed, `bind_structured` already falls
back to free text.

## 4. Architecture — Vertex provider layer

Chosen approach (of three considered): **one provider key → one dedicated client
class**, matching the existing factory pattern; auth isolated in a shared helper.
(Rejected: a single fat `vertex` client with model-prefix branching; and a
`vertex=True` flag bolted onto the existing google/anthropic/openai clients —
both tangle responsibilities the codebase deliberately keeps separate.)

### 4.1 New files

- `tradingagents/llm_clients/vertex_auth.py` — shared Vertex auth/config:
  - `get_credentials_and_project()` → `google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])`,
    returning `(credentials, default_project)`. Handles **both** ADC
    (`gcloud auth application-default login`) and service-account JSON
    (`GOOGLE_APPLICATION_CREDENTIALS`) transparently — no code branch.
  - `get_access_token(credentials)` → refreshes via
    `google.auth.transport.requests.Request()` and returns `credentials.token`
    (used only by the Grok OpenAI-compatible path).
  - `resolve_project(explicit)` / `resolve_location(explicit)` → explicit arg →
    config → env (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) → sensible
    default. **All `google.*` imports are lazy** (function-local) so importing the
    module never pulls the Vertex SDK during test collection.
  - `openapi_base_url(project, location)` → builds the OpenAI-compatible base:
    `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi`
    (and the `global` host form `https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi`
    when `location == "global"`).

- `tradingagents/llm_clients/vertex_clients.py` — three small client classes
  (all extend `BaseLLMClient`, all lazy-import their SDK in `get_llm()`):
  - `VertexGeminiClient` → `NormalizedChatVertexAI(model=..., project=..., location=...)`
    from `langchain_google_vertexai`. Maps `thinking_level` exactly like the
    existing `GoogleClient` (Gemini 3 → `thinking_level`; Gemini 2.5 →
    `thinking_budget`). Forwards `temperature`, `timeout`, `max_retries`.
  - `VertexAnthropicClient` → `NormalizedChatAnthropicVertex(model_name=...,
    project=..., location=...)` from `langchain_google_vertexai.model_garden`.
    Forwards `effort` (gated by the existing `_supports_effort` rule),
    `temperature`, `max_tokens`, `timeout`, `max_retries`.
  - `VertexGrokClient` → builds `endpoints/openapi` base_url + an OAuth access
    token (via `vertex_auth`), then returns a `NormalizedChatOpenAI(model=...,
    base_url=..., api_key=<token>, reasoning_effort=...)`. Reuses
    `NormalizedChatOpenAI` so the per-model **capability table** and
    structured-output dispatch carry over unchanged.
  - `NormalizedChatVertexAI` / `NormalizedChatAnthropicVertex` subclass their base
    and apply `normalize_content` in `invoke` (same pattern as the existing
    `NormalizedChat*` classes), so list-of-blocks content is flattened to a
    string for downstream agents.

### 4.2 Factory wiring (`factory.py`)

Add three dispatch branches (lazy import inside each, like today):

```python
if provider_lower == "vertex_gemini":
    from .vertex_clients import VertexGeminiClient
    return VertexGeminiClient(model, base_url, **kwargs)
if provider_lower == "vertex_anthropic":
    from .vertex_clients import VertexAnthropicClient
    return VertexAnthropicClient(model, base_url, **kwargs)
if provider_lower == "vertex_grok":
    from .vertex_clients import VertexGrokClient
    return VertexGrokClient(model, base_url, **kwargs)
```

`create_llm_client` gains the ability to receive `project` and `location` via
`**kwargs` (injected by the resolver, §5). Vendor-direct branches are untouched.

### 4.3 Model catalog / validation

Register the three Vertex model IDs in `model_catalog.py` under their new provider
keys so `warn_if_unknown_model()` stays quiet, and so the CLI/validation has a
source of truth. (Catalog is additive; existing entries unchanged.)

## 5. Architecture — generic role→model engine

Confined to `trading_graph.py` (resolver) and `setup.py` (per-role wiring); node
factory signatures `create_X(llm)` are **unchanged**.

### 5.1 Config schema (`default_config.py`, additive)

```python
"role_models": None,        # None/{} => current quick/deep tier behavior (backward compat)
"vertex_project": None,     # else env GOOGLE_CLOUD_PROJECT
"vertex_location": None,    # else env GOOGLE_CLOUD_LOCATION, else default "us-central1"
                            # (Grok pins "global" per-model in the preset)
```

A model spec is `{"provider", "model"}` plus optional `location` and optional
per-model thinking knobs (`google_thinking_level` / `openai_reasoning_effort` /
`anthropic_effort` / `temperature`). Omitted knobs inherit the run-level config.

`DEEP_ROLES = {"research_manager", "portfolio_manager"}`; everything else is quick.

### 5.2 Resolver + dedup (`trading_graph.py`)

Replace "build 2 LLMs" with a per-role resolver:

```python
self._llm_cache = {}  # (provider, model, location, hashable(kwargs)) -> llm

def _llm_for(self, role: str):
    spec = (self.config.get("role_models") or {}).get(role)
    if spec is None:                                  # backward-compatible tier default
        provider = self.config["llm_provider"]
        model = self.config["deep_think_llm"] if role in DEEP_ROLES else self.config["quick_think_llm"]
        kwargs = self._get_provider_kwargs()          # existing
        location = self.config.get("vertex_location")
    else:
        provider = spec["provider"]; model = spec["model"]
        kwargs = self._provider_kwargs_for(spec)      # per-spec knobs, else run-level
        location = spec.get("location") or self.config.get("vertex_location")
    if provider.startswith("vertex_"):
        kwargs["project"] = self.config.get("vertex_project")
        kwargs["location"] = location
    key = (provider, model, location, _hashable(kwargs))
    if key not in self._llm_cache:
        self._llm_cache[key] = create_llm_client(
            provider, model, base_url=self._base_url_for(provider), **kwargs
        ).get_llm()
    return self._llm_cache[key]
```

- **Dedup:** two roles on the same `(provider, model, location, kwargs)` share one
  client (e.g. the two Claude judges build one client; the two Gemini debaters
  one; the two Grok roles one) — fewer connections, fewer token fetches.
- **Per-model project/location** replaces the single run-level `backend_url` for
  Vertex (the one real subtlety the generic doc flagged). `_base_url_for(provider)`
  returns `None` for every `vertex_*` provider (Gemini/Claude take project+location;
  the Grok client builds its own `endpoints/openapi` URL from project+location), and
  returns the run-level `backend_url` only for single-provider vendor-direct runs.
- `_provider_kwargs_for(spec)` generalizes `_get_provider_kwargs()` to a spec:
  map `vertex_gemini`→google-style `thinking_level`, `vertex_anthropic`→
  anthropic-style `effort`, `vertex_grok`→openai-style `reasoning_effort`, plus
  cross-provider `temperature`.

### 5.3 GraphSetup wiring (`setup.py`)

`GraphSetup.__init__` changes from `(quick_thinking_llm, deep_thinking_llm, …)` to
receiving a `llm_for: Callable[[str], Any]` (the bound `graph._llm_for`). Each node
is built as `create_market_analyst(self.llm_for("market_analyst"))`, etc., for all
12 roles. The hardcoded quick/deep lambdas are replaced by `llm_for(<role key>)`.

## 6. Structured output

No new mechanism. The three decision agents already call
`bind_structured(llm, Schema, name)`; each role's `llm` is built through the
factory, so it carries its provider-native structured-output mode:
- Claude judges → `ChatAnthropicVertex` tool-use.
- Gemini trader → `ChatVertexAI` function_calling/json.
- Grok (debate only) → `NormalizedChatOpenAI` capability dispatch; not used for
  structured output. `bind_structured` returns `None` → free-text fallback if a
  model ever lacks support.

## 7. CLI integration

- `cli/utils.py:_llm_provider_table()` gains one row:
  `("Vertex Model Garden (multi-model debate)", "vertex_model_garden", None)`.
- `cli/main.py:get_user_selections()`: when `vertex_model_garden` is chosen —
  - **Skip** `select_shallow_thinking_agent` / `select_deep_thinking_agent`.
  - Prompt for `project` (default env `GOOGLE_CLOUD_PROJECT`, e.g. `tpmn-dev`) and
    `location` (default region; `global` offered).
  - Print a clear **cost/latency notice** (3 distinct models incl. Opus judges;
    slower and pricier; debate rounds multiply it).
  - Return the selection carrying `enable_vertex_multimodel=True` + project +
    location.
- `cli/main.py` config assembly / `run_analysis`: when selected, set
  `config["role_models"] = VERTEX_DEBATE_PRESET`,
  `config["llm_provider"]="vertex_gemini"`,
  `config["quick_think_llm"]="gemini-3.5-flash"`,
  `config["deep_think_llm"]="gemini-3.5-flash"`,
  `config["vertex_project"]`, `config["vertex_location"]`. Any other provider
  choice leaves `role_models` unset → existing single-model path, **unchanged**.
- `VERTEX_DEBATE_PRESET` is defined as a plain constant in a new `cli/presets.py`
  module (imported by `cli/main.py`), so it is isolated and unit-testable.

## 8. Authentication (details)

- All four call paths authenticate with **one** Google credential
  (`google.auth.default`, scope `cloud-platform`). No vendor API keys.
- Gemini (`ChatVertexAI`) and Claude (`ChatAnthropicVertex`) consume ADC
  internally from `project` + `location`.
- Grok (`ChatOpenAI`) needs a bearer token: obtain via `vertex_auth.get_access_token`
  at client **build time**. Valid ~1 hour — sufficient for a single `propagate()`
  (minutes). **Known limitation:** a run exceeding ~1h could see Grok-token
  expiry; documented, with a future refreshing-wrapper as the fix.
- Project/location/credentials are supplied via `.env` / CLI input and **never
  printed to chat**.

## 9. Dependencies & install

- New optional extra in `pyproject.toml`: `vertex = ["langchain-google-vertexai",
  "anthropic[vertex]"]`. Install with `pip install -e ".[vertex]"`.
- All Vertex SDK imports are **lazy** (inside `get_llm()` / function-local in
  `vertex_auth`), so the default install and the test suite run without the extra.
- `.env.example`: add `GOOGLE_CLOUD_PROJECT=`, `GOOGLE_CLOUD_LOCATION=`, and a note
  to run `gcloud auth application-default login` (or set
  `GOOGLE_APPLICATION_CREDENTIALS`).

## 10. Testing strategy

- **Vertex clients:** mock the lazy-imported LangChain classes; assert the right
  class is built with the right `model`/`project`/`location`, that `thinking_level`
  maps correctly for Gemini, that `effort` gating holds for Claude, and that the
  Grok client builds the correct `endpoints/openapi` base_url and passes a token
  as `api_key` (token fetch mocked).
- **Resolver:** `role_models=None` ⇒ identical (provider, model) as today for
  every role (backward compat lock); a preset routes each role correctly; dedup
  returns the **same object** for two roles sharing a spec; per-model `location`
  override beats the run default; `_provider_kwargs_for` maps knobs per family.
- **Preset:** every role key in `VERTEX_DEBATE_PRESET` is a valid node role; the 5
  unspecified roles fall back to Gemini quick tier.
- **CLI:** selecting `vertex_model_garden` skips the model prompts, sets
  `role_models` + project/location; selecting anything else leaves `role_models`
  unset.
- **Live verification (user-run):** against `tpmn-dev` with ADC + the three models
  enabled in Model Garden — not in CI (needs real credentials).

## 11. Backward compatibility

- `role_models` unset (default) ⇒ byte-for-byte current behavior; existing tests
  and single-provider vendor-direct runs unaffected.
- Node factory signatures unchanged; blast radius confined to `trading_graph.py`
  (resolver) + `setup.py` (wiring), plus additive new files (vertex clients/auth),
  additive config keys, additive catalog entries, and additive CLI rows.
- Vendor-direct providers remain fully functional.

## 12. Build order (within this deliverable)

1. **Vertex provider layer** — `vertex_auth.py`, `vertex_clients.py`, factory
   branches, catalog entries, optional extra. Mocked unit tests. (Land
   Gemini + Claude native classes first, then Grok MaaS.)
2. **role→model engine** — `role_models`/`vertex_*` config, `_llm_for` resolver +
   dedup + `_provider_kwargs_for` + per-model project/location, `GraphSetup`
   rewiring. Resolver/dedup/backward-compat unit tests.
3. **CLI preset** — provider-table row, `get_user_selections` branch, project/
   location prompts, cost notice, `run_analysis` config mapping, `VERTEX_DEBATE_PRESET`.
   CLI tests.
4. **Docs** — README/CLAUDE.md note + `.env.example`. User runs live verification
   against `tpmn-dev`.

Per the fork workflow ([[fork-upstream-sync]]): a feature branch, `--no-ff` merge,
push to origin.

## 13. Risks & open items

- **Claude-on-Vertex model id:** the user gave `claude-opus-4-8` (no date suffix).
  Vertex Anthropic publisher IDs are often date-suffixed (`...@YYYYMMDD`). Use the
  given value, keep it config-overridable, and **verify at live test**; append a
  date suffix if Vertex rejects the bare id.
- **Region/location per model:** Gemini, Claude, and Grok may be enabled in
  different regions (Grok commonly on `global`). `location` is per-model
  overridable; the user sets `vertex_location` (or per-spec `location`) to match
  `tpmn-dev` enablement. Resolve exact regions at live test.
- **Grok token expiry (~1h)** for very long runs — documented limitation (§8).
- **Cost/latency:** three distinct models (two Opus judges) multiply cost and the
  failure surface; debate rounds amplify it. Surfaced at CLI selection.
- **Quality gain unproven:** model diversity may help or may converge; this
  deliverable ships the capability, not a proven alpha lift. Measurement harness
  is deferred (§2 out-of-scope).
- **Determinism:** more models = less reproducibility; pair with low temperature
  where supported.
