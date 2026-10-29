import re
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .claude_structured import ClaudeStructuredOutput
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Opus 5.5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",        # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


class NormalizedChatAnthropic(ClaudeStructuredOutput, ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.

    구조화 출력은 공용 ``ClaudeStructuredOutput`` mixin(Vertex Claude 클라이언트와
    같은 것)을 거친다: 스키마가 유일한 tool 이고 ``tool_choice`` 는 ``"auto"`` 다.
    ChatAnthropic 자신의 ``with_structured_output`` 은 (thinking 이 없으면) tool 을
    강제하고, Claude Opus 5.5 / Sonnet 5.5 는 이를 400 으로 돌려준다.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # ``effort`` 는 ChatAnthropic 의 ``reasoning_effort`` 필드 alias 이고, 요청에서는
        # ``output_config.effort`` 로 나간다(langchain-anthropic 1.5.6;
        # tests/test_anthropic_direct.py 가 고정). ``thinking`` 은 API 의 dict 를 받는
        # ChatAnthropic 필드다: 문자열은 {"type": <mode>} 의 약식("adaptive",
        # "disabled"), dict 는 그대로 — VertexAnthropicClient 와 같은 규칙.
        thinking = self.kwargs.get("thinking")
        if thinking:
            llm_kwargs["thinking"] = (
                {"type": thinking} if isinstance(thinking, str) else thinking
            )

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
