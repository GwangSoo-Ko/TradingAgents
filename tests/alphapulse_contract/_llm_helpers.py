"""Replay Vertex SDK constructor kwargs through the REAL ``ChatAnthropicVertex``, offline.

The harness (``llm_boundary: "sdk"``) records the keyword arguments the fork hands
to ``langchain_google_vertexai.model_garden.ChatAnthropicVertex``. This module
answers the next question: what request would the real SDK send to Vertex with
those arguments? It runs in its own interpreter (by default the harness
interpreter, so the SDK versions are the ones the harness subprocess uses):

    <python> -I -B tests/alphapulse_contract/_llm_helpers.py <in.json> <out.json>

Only the HTTP transport is replaced (``httpx`` / ``httpx2``
``HTTPTransport.handle_request``, whichever the installed anthropic SDK uses) with
a canned Anthropic Messages reply, so everything between the constructor
arguments and the request body is the real langchain-google-vertexai + anthropic
code. Sockets are refused (and recorded per case), ``time.sleep`` is recorded
instead of slept (google-auth / retry backoff), and an explicit ``access_token``
stands in for ADC so google-auth never needs to look for credentials when the
kwargs are complete (production authenticates with ADC; the token only changes the
Authorization header, not the request). Kwargs that are NOT complete — no project,
say — make the SDK go looking for ADC / the GCE metadata server; that shows up as
the case's ``blocked`` attempts and its ``error``.

in.json:  {"cases": [{"key": str, "kwargs": {...}}]}
out.json: {"versions": {...}, "patched_transports": [...],
           "blocked_before_cases": [...], "import_error": str | None,
           "results": [{"key": str, "error": str | None, "blocked": [...],
                        "sleeps": int, "requests": [{"method", "url", "body"}]}]}

The script half imports nothing but the standard library at module level; the
test half (:func:`replay_on_real_vertex_sdk`) is what the contract tests call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ACCESS_TOKEN = "alphapulse-contract-harness-token"
_HTTP_LIBS = ("httpx", "httpx2")
_CANNED_REPLY = {
    "id": "msg_alphapulse_contract_harness",
    "type": "message",
    "role": "assistant",
    "model": "harness",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


# ============================================================================ test side

def replay_on_real_vertex_sdk(cases: dict[str, dict[str, Any]], work_dir: Path, *,
                              python: str, project_env: str = "tpmn-dev",
                              timeout: float = 300) -> dict[str, Any]:
    """Build each kwargs set with the real SDK in a subprocess and capture its request.

    ``cases`` maps a caller-chosen key to constructor kwargs. The subprocess
    environment is built from scratch like the harness's: HOME/TMPDIR under
    ``work_dir``, TZ=Asia/Seoul, GOOGLE_CLOUD_PROJECT as production sets it
    and GOOGLE_CLOUD_LOCATION unset (as in production). Returns the parsed
    out.json plus ``rc`` and the tail of ``stderr``; never raises for a failed replay.
    """
    work_dir = Path(work_dir)
    for sub in ("home", "tmp"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "vertex_replay_in.json"
    out_path = work_dir / "vertex_replay_out.json"
    in_path.write_text(json.dumps({"cases": [{"key": k, "kwargs": v} for k, v in cases.items()]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(work_dir / "home"),
        "TMPDIR": str(work_dir / "tmp"),
        "LANG": "C.UTF-8",
        "TZ": "Asia/Seoul",
        "GOOGLE_CLOUD_PROJECT": project_env,
    }
    # -I: no script dir / user site / PYTHON* env on sys.path; -B: write no bytecode.
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


def _patch_http_transports(requests: list[dict[str, Any]]) -> list[str]:
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
            requests.append({"lib": _lib.__name__, "method": request.method,
                             "url": str(request.url), "body": body})
            return _lib.Response(200, json=_CANNED_REPLY, request=request)

        lib.HTTPTransport.handle_request = handle_request
        patched.append(name)
    return patched


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for dist in ("anthropic", "langchain-google-vertexai", "langchain-core", "httpx", "httpx2"):
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
    _guard_sockets(blocked)
    _record_sleeps(sleeps)
    out: dict[str, Any] = {"versions": _versions(), "patched_transports": [],
                           "blocked_before_cases": [], "import_error": None, "results": []}
    try:
        out["patched_transports"] = _patch_http_transports(requests)
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
    except Exception as exc:  # noqa: BLE001 — reported to the test, which fails loudly
        out["import_error"] = f"{type(exc).__name__}: {exc}"
    out["blocked_before_cases"] = list(blocked)
    if out["import_error"] is None:
        for case in spec["cases"]:
            requests.clear()
            blocked_at, sleeps_at = len(blocked), len(sleeps)
            entry: dict[str, Any] = {"key": case["key"], "error": None}
            try:
                llm = ChatAnthropicVertex(**case["kwargs"], access_token=ACCESS_TOKEN)
                llm.invoke("ping")
            except Exception as exc:  # noqa: BLE001 — the SDK rejecting the kwargs is a finding
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            entry["blocked"] = blocked[blocked_at:]
            entry["sleeps"] = len(sleeps) - sleeps_at
            entry["requests"] = list(requests)
            out["results"].append(entry)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
