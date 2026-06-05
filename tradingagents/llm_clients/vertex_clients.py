"""LLM clients for foundation models hosted on Vertex AI Model Garden.

Three providers, two integration patterns:
  - vertex_gemini   -> ChatVertexAI            (langchain_google_vertexai)
  - vertex_anthropic -> ChatAnthropicVertex    (langchain_google_vertexai.model_garden)
  - vertex_grok     -> ChatOpenAI against the Vertex OpenAI-compatible endpoint
                       (langchain_openai), Google OAuth token as api_key

All SDK imports are lazy (inside get_llm) so the default install / test suite
runs without the optional ``[vertex]`` extra. v1 forwards a minimal, SDK-safe
kwarg set; thinking-config (thinking_level / effort) plumbing for the Vertex SDK
is deferred until param names are confirmed against a live project.
"""

from __future__ import annotations

from typing import Any, Optional

from . import vertex_auth
from .base_client import BaseLLMClient, normalize_content

# Cache of dynamically-created Normalized<Base> subclasses, keyed on the base
# class, so the subclass identity is stable across calls (and lazy imports).
_NORMALIZED_CLASSES: dict = {}


def _normalized_subclass(base_cls: type) -> type:
    """Return a cached subclass of ``base_cls`` that normalizes invoke() output.

    Mirrors the NormalizedChat* pattern in the other clients: providers that
    return list-of-blocks content (Gemini 3 reasoning blocks, Claude tool/think
    blocks) are flattened to a plain string for downstream agents.
    """
    cached = _NORMALIZED_CLASSES.get(base_cls)
    if cached is None:
        class _Normalized(base_cls):  # type: ignore[misc, valid-type]
            def invoke(self, input, config=None, **kwargs):
                return normalize_content(super().invoke(input, config, **kwargs))

        _Normalized.__name__ = f"Normalized{base_cls.__name__}"
        _NORMALIZED_CLASSES[base_cls] = _Normalized
        cached = _Normalized
    return cached


def _require_vertex_sdk(exc: ImportError) -> ImportError:
    """Return a clear, actionable error when the optional [vertex] extra is absent."""
    return ImportError(
        "Vertex AI Model Garden support requires the optional dependencies. "
        'Install them with:  pip install -e ".[vertex]"'
    )


class _VertexClientBase(BaseLLMClient):
    """Shared ctor for Vertex clients: project/location alongside model/base_url."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.project = project
        self.location = location

    def validate_model(self) -> bool:
        # Vertex model IDs are not in the catalog; accept any (like ollama).
        return True


class VertexGeminiClient(_VertexClientBase):
    """Gemini on Vertex via ChatVertexAI."""

    def get_llm(self) -> Any:
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise _require_vertex_sdk(exc) from exc

        cls = _normalized_subclass(ChatVertexAI)
        llm_kwargs = {
            "model": self.model,
            "project": vertex_auth.resolve_project(self.project),
            "location": vertex_auth.resolve_location(self.location),
        }
        for key in ("temperature", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return cls(**llm_kwargs)


class VertexAnthropicClient(_VertexClientBase):
    """Claude on Vertex via ChatAnthropicVertex (uses ``model_name=``)."""

    def get_llm(self) -> Any:
        try:
            from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        except ImportError as exc:
            raise _require_vertex_sdk(exc) from exc

        cls = _normalized_subclass(ChatAnthropicVertex)
        llm_kwargs = {
            "model_name": self.model,
            "project": vertex_auth.resolve_project(self.project),
            "location": vertex_auth.resolve_location(self.location),
        }
        for key in ("temperature", "max_tokens", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return cls(**llm_kwargs)


class VertexGrokClient(_VertexClientBase):
    """Grok on Vertex via the OpenAI-compatible MaaS endpoint.

    Reuses NormalizedChatOpenAI so the per-model capability table and
    structured-output dispatch carry over. A Google OAuth access token is
    fetched at build time and passed as the OpenAI api_key (valid ~1h).
    """

    def get_llm(self) -> Any:
        from .openai_client import NormalizedChatOpenAI

        project = vertex_auth.resolve_project(self.project)
        location = vertex_auth.resolve_location(self.location)
        base_url = self.base_url or vertex_auth.openapi_base_url(project, location)
        token = vertex_auth.get_access_token()
        llm_kwargs = {
            "model": self.model,
            "base_url": base_url,
            "api_key": token,
        }
        for key in ("temperature", "max_retries", "reasoning_effort", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatOpenAI(**llm_kwargs)
