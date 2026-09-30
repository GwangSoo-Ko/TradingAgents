"""Helpers for test_alphapulse_merge_expectations.py (test side only).

That module pins behaviour the upstream v0.5.x merge is EXPECTED to change, as
strict xfails: each test asserts the new behaviour and raises :class:`ExpectedDrift`
when it sees today's (a2981a7) behaviour instead. Any other failure — a broken
precondition, or a merge that lands only half of the change — is a plain
AssertionError and fails the suite for real (the xfail marker names
``raises=ExpectedDrift``).

What lives here:

* :func:`exit_info` and :func:`assert_ran_this_checkout`;
* an LLM fault shim: a ``sitecustomize`` put on the subprocess's PYTHONPATH (the
  harness runs ``main.py`` with one extra PYTHONPATH entry, as alpha-pulse does)
  that makes a scripted chat reply fail the way the Vertex Anthropic SDK fails on
  HTTP 429. The harness's scripted model only fails *structured* calls; the
  settlement reflector makes a plain chat call. The shim wraps the harness's
  model, never fork code (see :data:`LLM_FAULT_SITECUSTOMIZE`);
* weekday arithmetic for the harness's Mon-Fri synthetic price calendar and a
  reader for the verified market snapshot the market analyst receives.

:class:`ExpectedDrift` lives in ``_compat`` and the memory-log layout helpers (the
documented ``<!-- ENTRY_END -->`` layout) in ``_memory_helpers``, shared with the
memory contract. Nothing here imports the fork.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

from .harness import REPO_ROOT, RunResult


def exit_info(res: RunResult) -> dict[str, Any]:
    """How main.py ended (``captures.exit``): kind, and for an exception type/message."""
    return dict((res.captures or {}).get("exit") or {})


def assert_ran_this_checkout(res: RunResult) -> None:
    """Precondition: the subprocess imported the fork from THIS checkout (not the
    main checkout an editable install may point at), so a verdict is about this tree."""
    imported = ((res.captures or {}).get("imports") or {}).get("tradingagents")
    assert imported, f"the run never imported tradingagents\n{res.describe()}"
    assert Path(imported).resolve().is_relative_to(REPO_ROOT.resolve()), (
        f"tradingagents came from {imported}, not from {REPO_ROOT}")


# ============================================================================ prices

def last_weekday_on_or_before(day: dt.date) -> dt.date:
    """The harness serves Mon-Fri bars only (fixtures are holiday-free by design)."""
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def previous_weekday(day: dt.date) -> dt.date:
    return last_weekday_on_or_before(day - dt.timedelta(days=1))


_LATEST_ROW_RE = re.compile(r"Latest trading row used:\s*(\d{4}-\d{2}-\d{2})")
_REQUESTED_RE = re.compile(r"Requested analysis date:\s*(\d{4}-\d{2}-\d{2})")
_CLOSE_ROW_RE = re.compile(r"^\|\s*Close\s*\|\s*([^|]+?)\s*\|\s*$", re.MULTILINE)


def snapshot_reading(text: str) -> dict[str, Any]:
    """What the verified snapshot tells the market analyst: requested date, the row it
    used and that row's close (None where the snapshot does not say)."""
    latest = _LATEST_ROW_RE.search(text)
    requested = _REQUESTED_RE.search(text)
    close_cell = _CLOSE_ROW_RE.search(text)
    close = None
    if close_cell:
        try:
            close = float(close_cell.group(1).replace(",", ""))
        except ValueError:
            close = None
    return {"requested": requested.group(1) if requested else None,
            "latest_row": latest.group(1) if latest else None,
            "close": close,
            "close_cell": close_cell.group(1) if close_cell else None}


# ============================================================================ LLM fault shim

FAULT_KEY = "chat_fault"  # role-script key read by the shim (ignored by the harness)
FAULT_MARKER = "harness-merge-expectations: RESOURCE_EXHAUSTED"

# Installed as ``sitecustomize`` through PYTHONPATH. It waits for the harness's
# scripted model module to be imported by _bootstrap.py, then wraps its reply
# function: a chat call whose role script carries ``chat_fault`` (and whose prompt
# contains ``prompt_contains``, when given) raises what ChatAnthropicVertex raises
# on HTTP 429 — ``anthropic.RateLimitError`` (RuntimeError if the SDK is missing).
# The harness records the call as ``reply: {"raised": ...}`` like any provider error.
# A sitecustomize the interpreter already had is found further down sys.path and run
# first, so interpreter startup is otherwise unchanged. It never writes to stdout.
LLM_FAULT_SITECUSTOMIZE = r'''
"""Test-only LLM fault injector (tests/alphapulse_contract/_merge_expectations_helpers.py)."""
import importlib.abc as _abc
import importlib.machinery as _machinery
import importlib.util as _util
import os as _os
import sys as _sys

_TARGET = "tests.alphapulse_contract._fake_llm"
_KEY = "chat_fault"


def _chain():
    here = _os.path.dirname(_os.path.realpath(__file__))
    rest = [p for p in _sys.path if _os.path.realpath(p or _os.getcwd()) != here]
    spec = _machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None or not spec.origin:
        return
    if _os.path.realpath(spec.origin) == _os.path.realpath(__file__):
        return
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)


_chain()


def _provider_error(message):
    try:
        import anthropic
        import httpx
        url = ("https://aiplatform.googleapis.com/v1/projects/tpmn-dev/locations/global/"
               "publishers/anthropic/models/claude-sonnet-5:rawPredict")
        body = {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}}
        response = httpx.Response(429, request=httpx.Request("POST", url), json=body)
        return anthropic.RateLimitError(message, response=response, body=body)
    except Exception:  # SDK not installed: a plain provider-style error
        return RuntimeError(message)


def _prompt(messages):
    parts = []
    for m in messages or []:
        content = getattr(m, "content", "")
        parts.append(content if isinstance(content, str) else repr(content))
    return "\n".join(parts)


def _patch(module):
    cls = module.HarnessChatModel
    original = cls._reply
    fired = {}

    def _reply(self, kind, script, schema, tool_names, messages, index, record):
        fault = (script or {}).get(_KEY)
        if fault and kind == "chat":
            needle = fault.get("prompt_contains")
            if needle is None or needle in _prompt(messages):
                key = (record.get("role"), needle)
                times = fault.get("times")
                if times is None or fired.get(key, 0) < int(times):
                    fired[key] = fired.get(key, 0) + 1
                    raise _provider_error(fault.get("message") or "harness: provider error")
        return original(self, kind, script, schema, tool_names, messages, index, record)

    cls._reply = _reply
    module.MERGE_EXPECTATIONS_FAULT_SHIM = True


class _Loader(_abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


class _Finder(_abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = _machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


_sys.meta_path.insert(0, _Finder())
'''


def write_llm_fault_site(directory: Path) -> Path:
    """Write the shim as ``<directory>/sitecustomize.py``; pass the directory as the
    run's PYTHONPATH (``env_overrides={"PYTHONPATH": str(directory)}``)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(LLM_FAULT_SITECUSTOMIZE.lstrip("\n"),
                                                encoding="utf-8")
    return directory


def chat_fault(prompt_contains: str | None = None, times: int | None = None,
               message: str = FAULT_MARKER) -> dict[str, Any]:
    """Role-script value for :data:`FAULT_KEY`: fail matching chat calls (all when
    ``times`` is None) with a provider 429 carrying ``message``."""
    return {"prompt_contains": prompt_contains, "times": times, "message": message}


def raised_calls(res: RunResult, role: str) -> list[dict[str, Any]]:
    """``role``'s LLM calls that ended in an exception (the harness records ``raised``)."""
    return [c for c in res.calls_for(role) if "raised" in (c.get("reply") or {})]


def answered_calls(res: RunResult, role: str) -> list[dict[str, Any]]:
    return [c for c in res.calls_for(role) if "raised" not in (c.get("reply") or {})]
