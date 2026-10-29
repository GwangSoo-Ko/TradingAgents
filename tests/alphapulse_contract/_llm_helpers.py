"""SDK 생성자 kwargs 를 진짜 ``ChatAnthropic``(직접 API)로 오프라인 재생한다.

하네스(``llm_boundary: "sdk"``)는 fork 가 ``langchain_anthropic.ChatAnthropic`` 에 넘기는
키워드 인자를 기록한다(Claude 호출이 Vertex 에서 운영자의 ANTHROPIC_API_KEY 로 옮겨진 뒤
main.py 의 ``anthropic`` 공급자). 이 모듈은 그다음 질문에 답한다: 그 인자로, 운영이
만드는 클래스를 거쳐, 진짜 SDK 가 Anthropic API 에 무엇을 보내는가? 따로 된
인터프리터에서 돈다(기본은 하네스 인터프리터라 하네스 하위 프로세스와 같은 SDK 판):

    <python> -I -B tests/alphapulse_contract/_llm_helpers.py <in.json> <out.json>

각 경우는 fork 의 ``tradingagents.llm_clients.anthropic_client.NormalizedChatAnthropic``
(``repo_root`` 에서 import, 진짜 langchain-anthropic + anthropic SDK 위)을 기록된 kwargs 로
만들고 두 번 부른다: 일반 ``invoke("ping")`` 하나, 그 클래스 자신의
``with_structured_output``(탐침 스키마)으로 구조화 호출 하나. 바꾸는 것은 HTTP 전송층
(``httpx`` / ``httpx2`` 의 ``HTTPTransport.handle_request``, 설치된 SDK 가 쓰는 쪽)뿐이고,
미리 만든 Messages 응답을 돌려준다 — 요청에 tool 이 있으면 묶인 tool 의 ``tool_use``,
없으면 텍스트. 그리고 진짜 API 가 Claude Opus 5.5 / Sonnet 5.5 에 하듯,
``rejects_forced`` 모델에 tool 을 강제하는 요청(``tool_choice`` ``{"type": "tool"}`` /
``"any"``)에는 400 으로 답한다. 생성자 인자와 요청 본문 사이는 모두 진짜 fork +
langchain-anthropic + anthropic 코드다. 소켓은 거절하고(경우마다 기록), ``time.sleep`` 은
자지 않고 기록하며(400 뒤 SDK 재시도 대기가 여기 드러난다), ``tradingagents`` 가 import 때
하는 ``.env`` 찾기는 울타리로 막는다(아무것도 읽지 않음).

in.json:  {"repo_root": str, "rejects_forced": [model], "forced_error": str,
           "cases": [{"key": str, "kwargs": {...}}]}
out.json: {"versions": {...}, "patched_transports": [...], "blocked_before_cases": [...],
           "import_error": str | None, "fork_class": {"module", "file", "mro"},
           "dotenv": [...], "results": [{"key": str, "error": str | None,
           "structured_error": str | None, "structured_parsed": {...} | None,
           "blocked": [...], "sleeps": int,
           "requests": [{"phase", "method", "url", "headers", "body", "status"}]}]}

스크립트 쪽은 모듈 수준에서 표준 라이브러리만 import 한다. 계약 시험이 부르는 것은
시험 쪽(:func:`replay_on_real_anthropic_sdk`)이다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_HTTP_LIBS = ("httpx", "httpx2")
# 시험이 읽는 헤더만 남긴다 (user agent, 재시도 횟수 등은 뺀다).
_KEPT_HEADERS = ("x-api-key", "anthropic-version", "anthropic-beta", "authorization")
PROBE_SCHEMA = "ContractProbe"


def _message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {"id": "msg_alphapulse_contract_harness", "type": "message", "role": "assistant",
            "model": "harness", "content": content, "stop_reason": stop_reason,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


# ============================================================================ test side

def replay_on_real_anthropic_sdk(cases: dict[str, dict[str, Any]], work_dir: Path, *,
                                 python: str, repo_root: Path, api_key: str,
                                 rejects_forced: list[str], forced_error: str,
                                 timeout: float = 300) -> dict[str, Any]:
    """하위 프로세스에서 각 kwargs 를 진짜 SDK 위의 fork 클래스로 만들고 요청을 잡는다.

    ``cases`` 는 호출자가 고른 키 -> 생성자 kwargs. 하위 프로세스 환경은 하네스처럼
    처음부터 만든다: HOME/TMPDIR 는 ``work_dir`` 아래, TZ=Asia/Seoul, 운영처럼
    ANTHROPIC_API_KEY, GOOGLE_* 는 없음. 읽은 out.json 에 ``rc`` 와 ``stderr`` 끝부분을
    더해 돌려준다. 재생이 실패해도 예외를 올리지 않는다.
    """
    work_dir = Path(work_dir)
    for sub in ("home", "tmp"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "anthropic_replay_in.json"
    out_path = work_dir / "anthropic_replay_out.json"
    in_path.write_text(json.dumps({
        "repo_root": str(Path(repo_root).resolve()),
        "rejects_forced": sorted(rejects_forced),
        "forced_error": forced_error,
        "cases": [{"key": k, "kwargs": v} for k, v in cases.items()],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(work_dir / "home"),
        "TMPDIR": str(work_dir / "tmp"),
        "LANG": "C.UTF-8",
        "TZ": "Asia/Seoul",
        "ANTHROPIC_API_KEY": api_key,
    }
    # -I: sys.path 에 스크립트 디렉터리·user site·PYTHON* env 없음; -B: 바이트코드를 쓰지 않음.
    cmd = [python, "-I", "-B", str(Path(__file__).resolve()), str(in_path), str(out_path)]
    proc = subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True, timeout=timeout)
    out: dict[str, Any] = {}
    if out_path.exists():
        out = json.loads(out_path.read_text(encoding="utf-8"))
    out["rc"] = proc.returncode
    out["stderr"] = proc.stderr.decode(errors="replace")[-4000:]
    return out


# ============================================================================ script side

def _guard_sockets(blocked: list[dict[str, str]]) -> None:
    import socket

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _refuse(api: str, target: Any) -> OSError:
        blocked.append({"api": api, "target": repr(target)[:200]})
        return OSError(f"alphapulse contract replay: network access blocked ({api} {target!r})")

    def connect(self, address):
        if getattr(self, "family", None) == getattr(socket, "AF_UNIX", object()):
            return real_connect(self, address)
        raise _refuse("socket.connect", address)

    def connect_ex(self, address):
        if getattr(self, "family", None) == getattr(socket, "AF_UNIX", object()):
            return real_connect_ex(self, address)
        raise _refuse("socket.connect_ex", address)

    def create_connection(address, *a, **k):
        raise _refuse("socket.create_connection", address)

    def getaddrinfo(host, *a, **k):
        raise _refuse("socket.getaddrinfo", host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo


def _forces_a_tool(tool_choice: Any) -> bool:
    return isinstance(tool_choice, dict) and tool_choice.get("type") in ("tool", "any")


def _patch_http_transports(requests: list[dict[str, Any]], phase: list[str],
                           rejects_forced: set[str], forced_error: str) -> list[str]:
    import importlib

    patched = []
    for name in _HTTP_LIBS:
        try:
            lib = importlib.import_module(name)
        except ImportError:
            continue

        def handle_request(self, request, _lib=lib):
            raw = request.read()
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                body = {"__unparsed__": raw.decode("utf-8", errors="replace")[:2000]}
            headers = {k: v for k, v in request.headers.items() if k.lower() in _KEPT_HEADERS}
            entry = {"phase": phase[0], "lib": _lib.__name__, "method": request.method,
                     "url": str(request.url), "headers": headers, "body": body}
            requests.append(entry)
            body = body if isinstance(body, dict) else {}
            tools = body.get("tools") or []
            if tools and _forces_a_tool(body.get("tool_choice")) \
                    and body.get("model") in rejects_forced:
                entry["status"] = 400
                return _lib.Response(400, request=request, json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": forced_error}})
            entry["status"] = 200
            if tools:
                reply = _message([{"type": "tool_use", "id": "toolu_harness",
                                   "name": tools[0].get("name"), "input": {"ok": True}}],
                                 "tool_use")
            else:
                reply = _message([{"type": "text", "text": "ok"}], "end_turn")
            return _lib.Response(200, json=reply, request=request)

        lib.HTTPTransport.handle_request = handle_request
        patched.append(name)
    return patched


def _fence_dotenv(lookups: list[dict[str, Any]]) -> None:
    """``import tradingagents`` 는 load_dotenv(find_dotenv(usecwd=True)) 를 한다: 아무것도 주지 않는다."""
    import dotenv
    import dotenv.main

    def find_dotenv(filename: str = ".env", *args: Any, **kwargs: Any) -> str:
        lookups.append({"filename": filename, "served": ""})
        return ""

    dotenv.find_dotenv = find_dotenv
    dotenv.main.find_dotenv = find_dotenv


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for dist in ("anthropic", "langchain-anthropic", "langchain-core", "httpx", "httpx2"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _record_sleeps(sleeps: list[float]) -> None:
    import time

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    time.sleep = sleep


def _main(argv: list[str]) -> int:
    in_path, out_path = Path(argv[1]), Path(argv[2])
    spec = json.loads(in_path.read_text(encoding="utf-8"))
    blocked: list[dict[str, str]] = []
    sleeps: list[float] = []
    requests: list[dict[str, Any]] = []
    phase = ["chat"]
    lookups: list[dict[str, Any]] = []
    _guard_sockets(blocked)
    _record_sleeps(sleeps)
    out: dict[str, Any] = {"versions": _versions(), "patched_transports": [],
                           "blocked_before_cases": [], "import_error": None,
                           "fork_class": None, "dotenv": lookups, "results": []}
    try:
        out["patched_transports"] = _patch_http_transports(
            requests, phase, set(spec["rejects_forced"]), spec["forced_error"])
        _fence_dotenv(lookups)
        sys.path.insert(0, spec["repo_root"])  # main.py 가 sys.path[0] 으로 받는 것
        import inspect

        from pydantic import BaseModel

        from tradingagents.llm_clients.anthropic_client import NormalizedChatAnthropic

        class ContractProbe(BaseModel):
            """Answer through this tool."""

            ok: bool

        out["fork_class"] = {
            "module": NormalizedChatAnthropic.__module__,
            "file": inspect.getfile(NormalizedChatAnthropic),
            "mro": [f"{c.__module__}.{c.__qualname__}" for c in NormalizedChatAnthropic.__mro__],
        }
    except Exception as exc:  # noqa: BLE001 — 시험에 보고하고, 시험이 크게 실패한다
        out["import_error"] = f"{type(exc).__name__}: {exc}"
    out["blocked_before_cases"] = list(blocked)
    if out["import_error"] is None:
        for case in spec["cases"]:
            requests.clear()
            blocked_at, sleeps_at = len(blocked), len(sleeps)
            entry: dict[str, Any] = {"key": case["key"], "error": None,
                                     "structured_error": None, "structured_parsed": None}
            try:
                llm = NormalizedChatAnthropic(**case["kwargs"])
                phase[0] = "chat"
                llm.invoke("ping")
                phase[0] = "structured"
                try:
                    parsed = llm.with_structured_output(ContractProbe).invoke("ping")
                    entry["structured_parsed"] = parsed.model_dump() if parsed else None
                except Exception as exc:  # noqa: BLE001 — 강제 호출의 400 이 여기로 온다
                    entry["structured_error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            except Exception as exc:  # noqa: BLE001 — SDK 가 kwargs 를 거절하는 것도 발견이다
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            entry["blocked"] = blocked[blocked_at:]
            entry["sleeps"] = len(sleeps) - sleeps_at
            entry["requests"] = list(requests)
            out["results"].append(entry)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
