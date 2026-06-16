"""Report metadata helpers: a short mode tag for the report folder name and a
markdown configuration block for the report header, so multi-model and
single-model runs are distinguishable at a glance and self-document the exact
models used.
"""
from __future__ import annotations

import re
from typing import Any

# Pipeline order, for a readable per-role table.
_ROLE_ORDER = [
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator", "portfolio_manager",
]


def _sanitize(value: str) -> str:
    """Make a string a safe, compact filesystem path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def analysis_mode_tag(config: dict[str, Any]) -> str:
    """Short, filesystem-safe tag describing how a run was configured.

    Multi-model (``role_models`` set) -> ``"vertex-multimodel"`` when every role's
    provider is a ``vertex_*`` one, else ``"multimodel"``. Single-model ->
    ``"<provider>-<deep_model>"`` (e.g. ``"openai-gpt-5.5"``).
    """
    role_models = config.get("role_models") or {}
    if role_models:
        providers = {str(spec.get("provider", "")) for spec in role_models.values()}
        if providers and all(p.startswith("vertex_") for p in providers):
            return "vertex-multimodel"
        return "multimodel"
    provider = str(config.get("llm_provider", "") or "")
    deep = str(config.get("deep_think_llm", "") or "")
    return _sanitize(f"{provider}-{deep}" if (provider or deep) else "single")


def _resolved_role(config: dict[str, Any], role: str):
    """Return (provider, model) for a role: its role_models spec, else the tier
    default (deep model for the two judges, quick model otherwise)."""
    spec = (config.get("role_models") or {}).get(role)
    if spec:
        return str(spec.get("provider", "")), str(spec.get("model", "")), False
    from tradingagents.graph.trading_graph import DEEP_ROLES
    provider = str(config.get("llm_provider", "") or "")
    model = str(
        (config.get("deep_think_llm") if role in DEEP_ROLES else config.get("quick_think_llm")) or ""
    )
    return provider, model, True


def analysis_config_block(config: dict[str, Any]) -> str:
    """Markdown block for the report header: analysis mode + the models used.

    Multi-model runs get a full 12-role table (roles not in the preset are marked
    as tier defaults); single-model runs get a compact provider/quick/deep line.
    Returns text ending in a blank line, or '' for an empty config.
    """
    if not config:
        return ""
    role_models = config.get("role_models") or {}
    if not role_models:
        provider = config.get("llm_provider", "")
        quick = config.get("quick_think_llm", "")
        deep = config.get("deep_think_llm", "")
        return (
            "**Analysis mode:** single-model\n\n"
            f"**Provider:** `{provider}` · **quick-tier:** `{quick}` · "
            f"**deep-tier:** `{deep}`\n\n"
        )
    lines = [
        f"**Analysis mode:** {analysis_mode_tag(config)}",
        "",
        "| Role | Provider | Model |",
        "| --- | --- | --- |",
    ]
    for role in _ROLE_ORDER:
        provider, model, is_default = _resolved_role(config, role)
        marker = " *(tier default)*" if is_default else ""
        lines.append(f"| {role}{marker} | `{provider}` | `{model}` |")
    return "\n".join(lines) + "\n\n"
