
from typing import Any

from .base_client import BaseLLMClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    # Native (non-OpenAI) APIs are matched first so their string check doesn't
    # import the OpenAI client. Everything else is OpenAI-compatible and routes
    # through the provider registry (single source of truth).
    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "vertex_gemini":
        from .vertex_clients import VertexGeminiClient
        return VertexGeminiClient(model, base_url, **kwargs)

    if provider_lower == "vertex_anthropic":
        from .vertex_clients import VertexAnthropicClient
        return VertexAnthropicClient(model, base_url, **kwargs)

    if provider_lower == "vertex_grok":
        from .vertex_clients import VertexGrokClient
        return VertexGrokClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)

    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(provider_lower):
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")


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


def _coerce_max_tokens(value):
    """Validate a ``max_tokens`` value to a positive int (env vars are strings)."""
    if isinstance(value, bool):
        raise ValueError(f"max_tokens must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_tokens must be an integer, got {value!r}") from exc
    if n <= 0:
        raise ValueError(f"max_tokens must be > 0, got {n}")
    return n


def build_llm_kwargs(config: dict) -> dict[str, Any]:
    """Keyword arguments for ``create_llm_client`` from a TradingAgents config."""
    kwargs = {}
    provider = config.get("llm_provider", "").lower()

    if provider == "google":
        thinking_level = config.get("google_thinking_level")
        if thinking_level:
            kwargs["thinking_level"] = thinking_level

    elif provider == "openai":
        reasoning_effort = config.get("openai_reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

    elif provider in ("anthropic", "vertex_anthropic"):
        effort = config.get("anthropic_effort")
        if effort:
            kwargs["effort"] = effort
        # max_tokens/thinking are wired only for the Vertex Claude client
        # (VertexAnthropicClient routes them into model_kwargs); the
        # vendor-direct anthropic path is left untouched.
        if provider == "vertex_anthropic":
            max_tokens = config.get("anthropic_max_tokens")
            if max_tokens is not None and max_tokens != "":
                kwargs["max_tokens"] = _coerce_max_tokens(max_tokens)
            thinking = config.get("anthropic_thinking")
            if thinking:
                kwargs["thinking"] = thinking

    # Sampling temperature is cross-provider: forward it whenever set.
    # float() here so a value coming from a TRADINGAGENTS_TEMPERATURE env
    # string ("0.2") works the same as a programmatic float.
    temperature = config.get("temperature")
    if temperature is not None and temperature != "":
        kwargs["temperature"] = float(temperature)

    # SDK retry budget is cross-provider. Forward it only when explicitly set
    # so each provider keeps its own default (usually 2) otherwise (#1091).
    max_retries = config.get("llm_max_retries")
    if max_retries is not None and max_retries != "":
        kwargs["max_retries"] = _coerce_max_retries(max_retries)

    # Output-token cap is cross-provider, but Gemini names it
    # ``max_output_tokens``; forward under the right key when set (#1204).
    max_tokens = config.get("max_tokens")
    if max_tokens is not None and max_tokens != "":
        key = "max_output_tokens" if provider == "google" else "max_tokens"
        # A provider-specific cap set above wins. vertex_anthropic takes its cap
        # from ``anthropic_max_tokens``, a policy value sized for thinking plus the
        # answer (too low and a judge's thinking spends it: no tool call, no
        # TRADE_PLAN_JSON); letting the generic setting overwrite it would let one
        # TRADINGAGENTS_MAX_TOKENS env var silently defeat that policy.
        # The per-role path (TradingAgentsGraph._provider_kwargs_for) never
        # forwards the generic cap, so this guard gives both paths one meaning.
        if key not in kwargs:
            kwargs[key] = _coerce_max_tokens(max_tokens)

    return kwargs
