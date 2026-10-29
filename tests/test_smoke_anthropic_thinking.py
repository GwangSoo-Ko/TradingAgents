"""scripts/smoke_anthropic_thinking.py 를 오프라인으로 고정한다 (키·네트워크 없음).

스모크는 main.build_config() 의 실제 블록으로 빠른 계층 하나와 깊은 계층(판정) 하나를
만든다. --dry-run 은 요청 쌍과 SDK 가 보낼 본문만 출력하고, 실호출 경로는 여기서는
가짜 전송층으로 돈다(진짜 ChatAnthropic + anthropic SDK, 전송층만 가짜).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_anthropic_thinking.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_anthropic_thinking", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wire(monkeypatch):
    import anthropic

    http = pytest.importorskip("httpx2" if anthropic.__version__.startswith("1.") else "httpx")
    seen = []

    def handle_request(self, request):
        body = json.loads(request.read().decode("utf-8"))
        seen.append({"url": str(request.url), "body": body,
                     "key": request.headers.get("x-api-key")})
        if body.get("tools"):
            content = [{"type": "tool_use", "id": "toolu_1", "name": body["tools"][0]["name"],
                        "input": {"ok": True}}]
            stop = "tool_use"
        else:
            content, stop = [{"type": "text", "text": "OK"}], "end_turn"
        return http.Response(200, request=request, json={
            "id": "msg_smoke", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 2}})

    monkeypatch.setattr(http.HTTPTransport, "handle_request", handle_request)
    return seen


@pytest.mark.unit
def test_pairs_are_main_py_quick_tier_and_judge():
    pairs = _load().request_pairs()
    assert [(p["provider"], p["model"], p["kwargs"], p["structured"]) for p in pairs] == [
        ("anthropic", "claude-sonnet-5-5",
         {"effort": "high", "max_tokens": 32000, "thinking": "adaptive"}, False),
        ("anthropic", "claude-opus-5-5",
         {"effort": "xhigh", "max_tokens": 32000, "thinking": "adaptive"}, True),
    ]


@pytest.mark.unit
def test_dry_run_prints_the_request_pairs_offline(monkeypatch, capsys, wire):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # 키가 필요 없다
    assert _load().main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert wire == []  # 아무것도 보내지 않았다
    pairs = [line for line in out.splitlines() if line.startswith("[pair] ")]
    assert len(pairs) == 2, out
    assert "claude-sonnet-5-5 effort=high max_tokens=32000 thinking=adaptive" in pairs[0]
    assert "claude-opus-5-5 effort=xhigh max_tokens=32000 thinking=adaptive structured" in pairs[1]
    bodies = [json.loads(line.split("body=", 1)[1]) for line in out.splitlines()
              if line.strip().startswith("body=")]
    assert bodies == [
        {"model": "claude-sonnet-5-5", "max_tokens": 32000, "thinking": {"type": "adaptive"},
         "output_config": {"effort": "high"}},
        {"model": "claude-opus-5-5", "max_tokens": 32000, "thinking": {"type": "adaptive"},
         "output_config": {"effort": "xhigh"}, "tool_choice": {"type": "auto"}},
    ]
    assert out.count("sdk_timeout=None") == 2  # SDK 의 비스트리밍 가드가 걸리지 않는다


@pytest.mark.unit
def test_live_path_without_a_key_is_a_setup_error(monkeypatch, capsys, wire):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _load().main([]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert wire == []


@pytest.mark.unit
def test_live_path_sends_main_py_requests_and_never_prints_the_key(monkeypatch, capsys, wire):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-smoke-test-not-a-key")
    assert _load().main([]) == 0
    out = capsys.readouterr().out
    assert "sk-smoke-test-not-a-key" not in out
    assert [r["url"] for r in wire] == ["https://api.anthropic.com/v1/messages"] * 2
    assert {r["key"] for r in wire} == {"sk-smoke-test-not-a-key"}
    quick, judge = (r["body"] for r in wire)
    assert (quick["model"], quick["max_tokens"], quick["output_config"], quick["thinking"]) == (
        "claude-sonnet-5-5", 32000, {"effort": "high"}, {"type": "adaptive"})
    assert (judge["model"], judge["max_tokens"], judge["output_config"], judge["thinking"]) == (
        "claude-opus-5-5", 32000, {"effort": "xhigh"}, {"type": "adaptive"})
    assert judge["tool_choice"] == {"type": "auto"} and "tool_choice" not in quick
    assert out.count("[ok]") == 2 and "all accepted" in out


@pytest.mark.unit
def test_a_rejected_call_exits_1(monkeypatch, capsys):
    import anthropic

    http = pytest.importorskip("httpx2" if anthropic.__version__.startswith("1.") else "httpx")

    def reject(self, request):
        return http.Response(400, request=request, json={
            "type": "error", "error": {"type": "invalid_request_error", "message": "nope"}})

    monkeypatch.setattr(http.HTTPTransport, "handle_request", reject)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-smoke-test-not-a-key")
    assert _load().main([]) == 1
    out = capsys.readouterr().out
    assert out.count("[FAIL]") == 2 and "[400]" in out
