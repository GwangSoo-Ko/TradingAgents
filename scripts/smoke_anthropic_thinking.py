"""main.py 가 쓰는 직접 Anthropic API 경로(ANTHROPIC_API_KEY)의 실호출 스모크.

api.anthropic.com 이 main.py 의 Claude 블록이 보내는 요청을 그대로 받는지 확인한다.
요청은 fork 의 코드가 ``main.build_config()`` 로 만든다: 빠른 계층(``quick_think_llm`` +
``build_llm_kwargs(config)``)과 깊은 계층 심판(``role_models["portfolio_manager"]`` +
그래프의 역할 경로 kwargs). 둘 다 ``create_llm_client("anthropic", ...).get_llm()`` 즉
main.py 가 만드는 ``NormalizedChatAnthropic`` 이라, 운영과 같은 것을 보낸다: 모델,
비스트리밍 ``max_tokens`` 32000, ``thinking: {"type": "adaptive"}``,
``output_config.effort``. 빠른 계층은 일반 호출, 심판은 구조화 호출(Research/Portfolio
Manager 의 꼴: 스키마가 유일한 tool, ``tool_choice`` auto)이다.

사용법:
    ANTHROPIC_API_KEY=... python scripts/smoke_anthropic_thinking.py
    python scripts/smoke_anthropic_thinking.py --dry-run   # 오프라인: 요청 쌍만 출력

키는 환경 변수(또는 ``tradingagents`` 가 읽는 .env)에서 오고 출력하지 않는다. 비용은
작다: 한 줄 프롬프트에 호출 2번(심판이 산문으로 답해 한 번 더 물으면 3번). max_tokens 와
상관없이 생성된 토큰만 청구된다. 종료 코드 0 = 모두 수락, 1 = 거절된 호출 있음,
2 = 설정 문제(키 없음). ``--dry-run`` 은 키가 필요 없고 요청을 보내지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPT = "Reply with exactly: OK"
JUDGE_ROLE = "portfolio_manager"
# dry-run 에 보여 줄 본문 필드 (헤더는 키가 실리므로 절대 보여 주지 않는다).
_SHOWN = ("model", "max_tokens", "thinking", "output_config", "stream", "tool_choice")


def request_pairs(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """main.py 의 실제 블록에서 만든 두 쌍 (tier, provider, model, 클라이언트 kwargs)."""
    import main
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.llm_clients.factory import build_llm_kwargs

    config = main.build_config() if config is None else config
    spec = (config.get("role_models") or {})[JUDGE_ROLE]
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)  # 역할 경로 kwargs 만드는 부분만 쓴다
    graph.config = config
    return [
        {"tier": "quick", "provider": config["llm_provider"],
         "model": config["quick_think_llm"], "kwargs": build_llm_kwargs(config),
         "structured": False},
        {"tier": f"deep ({JUDGE_ROLE})", "provider": spec["provider"], "model": spec["model"],
         "kwargs": graph._provider_kwargs_for(spec), "structured": True},
    ]


def _build(pair: dict[str, Any], config: dict[str, Any]):
    from tradingagents.llm_clients.factory import create_llm_client

    return create_llm_client(pair["provider"], pair["model"],
                             base_url=config.get("backend_url"), **pair["kwargs"]).get_llm()


def _smoke_schema():
    from pydantic import BaseModel

    class SmokeAnswer(BaseModel):
        """스모크 프롬프트에 대한 답."""

        ok: bool

    return SmokeAnswer


def _body(llm, pair: dict[str, Any]) -> dict[str, Any]:
    """이 쌍에 대해 SDK 가 보낼 요청 본문 (오프라인, langchain-anthropic)."""
    from langchain_core.messages import HumanMessage

    extra: dict[str, Any] = {}
    if pair["structured"]:
        # fork 의 구조화 경로(claude_structured)가 만드는 것과 같은 묶음.
        extra = dict(llm.bind_tools([_smoke_schema()], tool_choice="auto").kwargs)
    payload = llm._get_request_payload([HumanMessage(PROMPT)], **extra)
    return {k: payload[k] for k in _SHOWN if k in payload}


def _label(pair: dict[str, Any]) -> str:
    kw = pair["kwargs"]
    return (f"{pair['tier']}: {pair['provider']} {pair['model']} effort={kw.get('effort')} "
            f"max_tokens={kw.get('max_tokens')} thinking={kw.get('thinking')}"
            f"{' structured' if pair['structured'] else ''}")


def _versions() -> str:
    from importlib import metadata

    out = []
    for dist in ("anthropic", "langchain-anthropic"):
        try:
            out.append(f"{dist}={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            out.append(f"{dist}=missing")
    return " ".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="요청 쌍과 본문을 오프라인으로 출력한다 (키·요청 없음)")
    args = parser.parse_args(argv)

    import main as runner  # main.py 처럼 tradingagents 를 거쳐 .env 를 읽는다

    config = runner.build_config()
    pairs = request_pairs(config)
    print(_versions())

    if args.dry_run:
        print("dry run: no request is sent\n")
        for pair in pairs:
            llm = _build(pair, config)
            print(f"[pair] {_label(pair)}")
            print(f"       client={type(llm).__name__} streaming={llm.streaming} "
                  f"sdk_timeout={llm._client.timeout}")
            print(f"       body={json.dumps(_body(llm, pair), sort_keys=True, default=str)}")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY (environment or .env)", file=sys.stderr)
        return 2

    print()
    rc = 0
    for pair in pairs:
        label = _label(pair)
        try:
            llm = _build(pair, config)
            if pair["structured"]:
                out = llm.with_structured_output(_smoke_schema(), include_raw=True).invoke(PROMPT)
                if out["parsing_error"] is not None:
                    raise out["parsing_error"]
                msg, text = out["raw"], f"parsed={out['parsed']!r}"
            else:
                msg = llm.invoke(PROMPT)
                text = f"text={msg.content!r}"
            stop = (msg.response_metadata or {}).get("stop_reason")
            tokens = (msg.usage_metadata or {}).get("output_tokens")
            print(f"[ok]   {label}")
            print(f"       stop_reason={stop} out_tokens={tokens} {text}")
        except Exception as exc:  # noqa: BLE001
            rc = 1
            status = getattr(exc, "status_code", None)
            suffix = f" [{status}]" if status else ""
            print(f"[FAIL] {label}  {type(exc).__name__}{suffix}: {str(exc)[:300]}")

    print("\nRESULT:", "all accepted" if rc == 0 else "at least one call rejected — see above")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
