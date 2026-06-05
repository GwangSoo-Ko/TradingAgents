"""Presets that populate role_models for the CLI's multi-model debate options.

Kept separate from cli/main.py so the mapping (and the config it produces) is
unit-testable without importing the interactive CLI.
"""
from __future__ import annotations

from typing import Any, Dict

# Vertex Model Garden multi-model debate (Gemini / Claude / Grok), all on the
# Vertex `global` endpoint. Judges (research_manager, portfolio_manager) run on
# Claude; the five debate roles are diversified across the three families; the
# four analysts and the trader are omitted and fall back to the quick-tier
# default (Gemini) via the role->model resolver.
VERTEX_DEBATE_PRESET: Dict[str, Dict[str, str]] = {
    "bull_researcher":      {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "bear_researcher":      {"provider": "vertex_grok",      "model": "xai/grok-4.3"},
    "aggressive_debator":   {"provider": "vertex_grok",      "model": "xai/grok-4.3"},
    "conservative_debator": {"provider": "vertex_gemini",    "model": "gemini-3.5-flash"},
    "neutral_debator":      {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    "research_manager":     {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
    "portfolio_manager":    {"provider": "vertex_anthropic", "model": "claude-opus-4-8"},
}

# Default model for roles not in the preset (analysts + trader) and for the
# quick/deep tier fallback in Vertex mode.
VERTEX_DEFAULT_MODEL = "gemini-3.5-flash"


def apply_vertex_multimodel_config(
    config: Dict[str, Any], selections: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the Vertex multi-model debate preset to ``config`` in place.

    No-op unless ``selections['enable_vertex_multimodel']`` is truthy, so any
    other provider choice leaves ``role_models`` unset and the run on its
    single-model path.
    """
    if not selections.get("enable_vertex_multimodel"):
        return config
    config["llm_provider"] = "vertex_gemini"
    config["quick_think_llm"] = VERTEX_DEFAULT_MODEL
    config["deep_think_llm"] = VERTEX_DEFAULT_MODEL
    config["role_models"] = dict(VERTEX_DEBATE_PRESET)
    config["vertex_project"] = selections.get("vertex_project")
    config["vertex_location"] = selections.get("vertex_location") or "global"
    return config


# Single-model runs hosted on Vertex Model Garden (no vendor API key needed):
# the CLI provider key -> the fixed model the whole pipeline runs on. Lets users
# without an Anthropic/xAI API key run a normal single-model analysis on a
# Vertex-hosted Claude or Grok. The provider key IS the real vertex_* client key.
VERTEX_SINGLE_MODELS: Dict[str, str] = {
    "vertex_anthropic": "claude-opus-4-8",
    "vertex_grok":      "xai/grok-4.3",
}


def apply_vertex_single_model_config(
    config: Dict[str, Any], selections: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply a single-model Vertex run (Claude or Grok) to ``config`` in place.

    No-op unless ``selections['vertex_single_provider']`` is one of
    ``VERTEX_SINGLE_MODELS``. Runs the whole pipeline on that one Vertex-hosted
    model (``role_models`` stays unset → the normal single-model path).
    """
    provider = selections.get("vertex_single_provider")
    if provider not in VERTEX_SINGLE_MODELS:
        return config
    model = VERTEX_SINGLE_MODELS[provider]
    config["llm_provider"] = provider
    config["quick_think_llm"] = model
    config["deep_think_llm"] = model
    config["role_models"] = None
    config["vertex_project"] = selections.get("vertex_project")
    config["vertex_location"] = selections.get("vertex_location") or "global"
    return config
