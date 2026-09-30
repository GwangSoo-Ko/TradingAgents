"""Subprocess-side shared state of the alpha-pulse contract harness.

Imported only inside the harness subprocess (by ``_bootstrap.py`` and the fakes),
never by the fork. Holds the scenario spec, the capture document that is written
to ``<run_dir>/captures.json``, and small helpers every fake needs: which graph
node is currently executing, which role that node plays, the trade date that
reached ``propagate``, and token rendering for scripted replies.

Nothing here imports ``tradingagents`` at module level — the fakes must be in
place before the fork is imported.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import re
import threading
import traceback
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
FIXTURES = PACKAGE_DIR / "fixtures"

CAPTURE_VERSION = 1

# Top-level graph node -> the role key main.py / cli.report_meta use. Upstream
# names are the same today; a renamed node shows up as role None in captures.
NODE_ROLES = {
    "Market Analyst": "market_analyst",
    "Sentiment Analyst": "sentiment_analyst",
    "Social Analyst": "sentiment_analyst",
    "News Analyst": "news_analyst",
    "Fundamentals Analyst": "fundamentals_analyst",
    "Bull Researcher": "bull_researcher",
    "Bear Researcher": "bear_researcher",
    "Research Manager": "research_manager",
    "Trader": "trader",
    "Aggressive Analyst": "aggressive_debator",
    "Conservative Analyst": "conservative_debator",
    "Neutral Analyst": "neutral_debator",
    "Portfolio Manager": "portfolio_manager",
}

_lock = threading.RLock()
SPEC: dict[str, Any] = {}
RUN_DIR: Path | None = None
_seq = 0
_trade_date: str | None = None
_role_counters: dict[tuple[str, str], int] = {}

CAPTURES: dict[str, Any] = {
    "capture_version": CAPTURE_VERSION,
    "scenario": None,
    "python": None,
    "cwd": None,
    "sys_path0": None,
    "argv": None,
    "imports": {},
    "env_at_start": {},
    "env_at_propagate": None,
    "dotenv": [],
    "propagate": [],
    "graph_config": None,
    "llm_constructions": [],
    "structured_bindings": [],
    "llm_calls": [],
    "config_by_node": {},
    "config_at_first_tool_call": None,
    "data_calls": [],
    "network": {"blocked": [], "unrouted": []},
    "sleeps": [],
    "notes": [],
    "exit": None,
}


def init(spec: dict[str, Any], run_dir: Path) -> None:
    global RUN_DIR
    SPEC.clear()
    SPEC.update(spec)
    RUN_DIR = run_dir
    CAPTURES["scenario"] = spec.get("name")


def next_seq() -> int:
    global _seq
    with _lock:
        _seq += 1
        return _seq


def json_safe(value: Any, _depth: int = 0) -> Any:
    """Best-effort JSON-able copy (pydantic models dumped, objects repr'd)."""
    if _depth > 12:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, _depth + 1) for v in value]
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return {"__type__": type(value).__name__, "dump": json_safe(dump(mode="json"), _depth + 1)}
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, type):
        return f"<class {value.__module__}.{value.__qualname__}>"
    return repr(value)


def flush() -> None:
    """Write captures.json atomically (safe to call repeatedly)."""
    if RUN_DIR is None:
        return
    with _lock:
        text = json.dumps(json_safe(CAPTURES), ensure_ascii=False, indent=1)
    target = RUN_DIR / "captures.json"
    tmp = RUN_DIR / "captures.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def note(message: str) -> None:
    with _lock:
        CAPTURES["notes"].append(message)


def short_stack(limit: int = 8) -> list[str]:
    """The innermost frames outside this harness package, for attributing calls."""
    frames = []
    for fs in traceback.extract_stack()[:-1]:
        if str(PACKAGE_DIR) in fs.filename:
            continue
        frames.append(f"{_rel(fs.filename)}:{fs.lineno}:{fs.name}")
    return frames[-limit:]


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        parts = Path(path).parts
        if "site-packages" in parts:
            return "site-packages/" + "/".join(parts[parts.index("site-packages") + 1:])
        return path


# --------------------------------------------------------------------------- nodes

def current_node() -> tuple[str | None, str | None]:
    """(top-level graph node, innermost node) of the running LangGraph task, if any.

    LangGraph puts ``langgraph_node`` and ``langgraph_checkpoint_ns`` into the
    runnable config it sets for each task; the config travels in a contextvar that
    tool threads inherit. The first checkpoint-namespace segment is the top-level
    node even when a node is itself a subgraph (upstream v0.5.2 analysts).
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config
    except Exception:  # noqa: BLE001
        return None, None
    cfg = var_child_runnable_config.get() or {}
    md = cfg.get("metadata") or {}
    inner = md.get("langgraph_node")
    ns = md.get("langgraph_checkpoint_ns") or ""
    top = ns.split("|")[0].split(":")[0] if ns else inner
    return (top or None), (inner or None)


def role_for_node(top: str | None) -> str | None:
    if top is None:
        return None
    return NODE_ROLES.get(top)


def role_outside_graph() -> str:
    """Label for an LLM call made outside any graph node (settlement reflection)."""
    for fs in traceback.extract_stack():
        name = Path(fs.filename).name
        if name == "reflection.py" or "reflect" in fs.name:
            return "reflector"
    return "outside_graph"


def role_script(role: str | None) -> dict[str, Any]:
    roles = SPEC.get("roles") or {}
    return roles.get(role or "", {}) or {}


def bump_counter(role: str | None, kind: str) -> int:
    """0-based index of this (role, kind) call within the run."""
    key = (role or "?", kind)
    with _lock:
        n = _role_counters.get(key, 0)
        _role_counters[key] = n + 1
        return n


# --------------------------------------------------------------------------- dates

def set_trade_date(value: Any) -> None:
    global _trade_date
    _trade_date = None if value is None else str(value)


def trade_date() -> _dt.date:
    """The trade date that reached propagate, else KST/local today."""
    if _trade_date:
        try:
            return _dt.date.fromisoformat(_trade_date[:10])
        except ValueError:
            pass
    return _dt.date.today()


_TOKEN_RE = re.compile(r"\$\{(ticker|company|trade_date)(?:-(\d+))?\}")


def render(value: Any) -> Any:
    """Replace ${ticker} ${company} ${trade_date} ${trade_date-N} in strings, recursively."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            name, minus = m.group(1), m.group(2)
            if name == "ticker":
                return str((SPEC.get("argv") or [""])[0])
            if name == "company":
                return str(SPEC.get("company") or (SPEC.get("argv") or [""])[0])
            day = trade_date() - _dt.timedelta(days=int(minus or 0))
            return day.isoformat()
        return _TOKEN_RE.sub(_sub, value)
    if isinstance(value, list):
        return [render(v) for v in value]
    if isinstance(value, dict):
        return {k: render(v) for k, v in value.items()}
    return copy.deepcopy(value)


# --------------------------------------------------------------------------- config

def config_snapshot() -> dict[str, Any] | None:
    """``get_config()`` of the fork's dataflows layer, wherever it lives now."""
    import importlib
    for dotted in ("tradingagents.dataflows.config",):
        try:
            mod = importlib.import_module(dotted)
        except ImportError:
            continue
        getter = getattr(mod, "get_config", None)
        if callable(getter):
            try:
                return json_safe(getter())
            except Exception as exc:  # noqa: BLE001
                return {"__error__": f"{type(exc).__name__}: {exc}"}
    return None


def note_node_config(top: str | None) -> None:
    if top is None:
        return
    with _lock:
        if top in CAPTURES["config_by_node"]:
            return
    snap = config_snapshot()
    with _lock:
        CAPTURES["config_by_node"].setdefault(top, snap)


def record_data_call(entry: dict[str, Any]) -> None:
    """One library-level data access (yfinance API, HTTP request)."""
    top, inner = current_node()
    entry = dict(entry)
    entry["seq"] = next_seq()
    entry["node"] = top
    entry["inner_node"] = inner
    with _lock:
        CAPTURES["data_calls"].append(entry)
        first_tool = CAPTURES["config_at_first_tool_call"] is None and top is not None
    if first_tool:
        snap = config_snapshot()
        with _lock:
            if CAPTURES["config_at_first_tool_call"] is None:
                CAPTURES["config_at_first_tool_call"] = {"node": top, "config": snap}
