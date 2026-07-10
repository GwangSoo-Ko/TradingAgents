from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Tiered Vertex Model Garden run on Claude (no vendor API key — ADC auth + the
# optional [vertex] extra; needs GOOGLE_CLOUD_PROJECT and `gcloud auth
# application-default login`). The two deep judges (Research/Portfolio Manager)
# run Opus 4.8 at max effort; every other role runs Sonnet 5 at high effort.
# thinking/max_tokens are shared across tiers. vertex_project/location resolve
# from GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION (default "global").
# max_tokens stays <= ~21.3k so the non-streaming node calls don't trip the
# Anthropic SDK's "streaming required" guard. Comment this block out to fall
# back to DEFAULT_CONFIG (OpenAI + .env).
config.update({
    "llm_provider": "vertex_anthropic",
    "deep_think_llm": "claude-opus-4-8",
    "quick_think_llm": "claude-sonnet-5",
    "anthropic_thinking": "adaptive",
    "anthropic_max_tokens": 20000,
    "anthropic_effort": "high",          # quick tier (Sonnet 5) default
    "output_language": "Korean",         # localize the user-facing report/decision

    "role_models": {                     # override the two deep judges -> Opus / max
        "research_manager": {
            "provider": "vertex_anthropic", "model": "claude-opus-4-8",
            "anthropic_effort": "max",
        },
        "portfolio_manager": {
            "provider": "vertex_anthropic", "model": "claude-opus-4-8",
            "anthropic_effort": "max",
        },
    },
})

# Initialize with custom config. selected_analysts defaults to all four; listed
# explicitly here so it's obvious what runs and easy to trim.
ta = TradingAgentsGraph(
    selected_analysts=("market", "social", "news", "fundamentals"),
    debug=True,
    config=config,
)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
