"""Helpers for the memory-log / settlement contract tests (test side only).

The memory log is the fork's own feedback loop: every ``main.py`` run appends its
decision as a ``pending`` entry to the per-ticker log alpha-pulse names in
``TRADINGAGENTS_MEMORY_LOG_PATH``; the next run of that ticker settles it (5 trading
days, alpha vs the listing market's index), asks the reflector for a lesson, and
hands the lessons to the Portfolio Manager. These helpers read that log the way a
person reads it (split on the documented ``<!-- ENTRY_END -->`` delimiter, look at
the tag line), write production-shaped seed logs, and observe file writes of a
subprocess run without patching fork code.

Nothing here imports the fork at module import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from ._compat import get_class_by_interface
from .harness import FIXTURES, REPO_ROOT

# The documented hard delimiter between entries (CLAUDE.md "Persistence").
SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"

# Production-shaped logs written by the fork at a2981a7 and their reviewed reading.
LOG_FIXTURES = (
    "memory_log_a2981a7_005930.KS.md",   # KR per-ticker (the layout alpha-pulse uses)
    "memory_log_a2981a7_AAPL.md",        # US per-ticker
    "memory_log_a2981a7_shared.md",      # legacy shared file (mixed tickers, cross-ticker lessons)
)
EXPECTED_FIXTURE = "memory_log_a2981a7_expected.json"

# Tag lines exactly as a2981a7 writes them (and reads them back after a rollback).
PENDING_TAG_RE = re.compile(r"\[(?P<date>\d{4}-\d{2}-\d{2}) \| (?P<ticker>[^|\]]+?) \| "
                            r"(?P<rating>[A-Za-z]+) \| pending\]")
RESOLVED_TAG_RE = re.compile(r"\[(?P<date>\d{4}-\d{2}-\d{2}) \| (?P<ticker>[^|\]]+?) \| "
                             r"(?P<rating>[A-Za-z]+) \| (?P<raw>[+-]\d+\.\d%) \| "
                             r"(?P<alpha>[+-]\d+\.\d%) \| (?P<holding>\d+d)"
                             r"(?: \| resolved:(?P<resolved>\d{4}-\d{2}-\d{2}))?\]")

# Where the memory-log class has lived / may live. The graph module comes first on
# purpose: the class under test must be the one the graph instantiates, not an
# orphaned copy a merge left behind.
MEMORY_LOG_HOMES = (
    "tradingagents.graph.trading_graph",   # the graph's own binding (every version so far)
    "tradingagents.agents.utils.memory",   # fork a2981a7
    "tradingagents.decision_log",          # upstream v0.5.1
    "tradingagents.memory",                # upstream v0.5.2 (package)
    "tradingagents.memory.log",
)
# The memory-log interface the graph drives (the methods propagate() calls on it); the
# class is found by these, not by its name (TradingMemoryLog at a2981a7, a merge may rename
# it). The harness applies the same test to the live graph (_bootstrap._memory_log_classes).
MEMORY_LOG_METHODS = ("get_past_context", "store_decision")

# Symbols the settlement may use as the alpha benchmark (listing market's index).
BENCHMARKS = ("^KS11", "^KQ11", "SPY")
PRICE_APIS = ("Ticker.history", "download")


# ============================================================================ log text

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_entries(text: str | None) -> list[str]:
    """Entry blocks (stripped, non-empty) in file order."""
    return [b.strip() for b in (text or "").split(SEPARATOR) if b.strip()]


def tag_line(block: str) -> str:
    return block.splitlines()[0].strip()


def blocks_for(text: str | None, date: str, ticker: str) -> list[str]:
    prefix = f"[{date} | {ticker} |"
    return [b for b in split_entries(text) if tag_line(b).startswith(prefix)]


def entry_text(tag: str, decision: str, reflection: str | None = None) -> str:
    """One entry exactly as a2981a7's writer lays it out (without the delimiter)."""
    text = f"{tag}\n\nDECISION:\n{decision}"
    if reflection is not None:
        text += f"\n\nREFLECTION:\n{reflection}"
    return text


def pending_tag(date: str, ticker: str, rating: str) -> str:
    return f"[{date} | {ticker} | {rating} | pending]"


def resolved_tag(date: str, ticker: str, rating: str, raw: str, alpha: str,
                 holding: str = "5d", resolved: str | None = None) -> str:
    """``resolved=None`` gives the legacy 6-field tag written before #1251."""
    tail = f" | resolved:{resolved}" if resolved else ""
    return f"[{date} | {ticker} | {rating} | {raw} | {alpha} | {holding}{tail}]"


def seed_log(*entries: str) -> str:
    """A whole log file: every entry followed by the delimiter (append-only layout)."""
    return "".join(e + SEPARATOR for e in entries)


# ============================================================================ fixtures

def expected_readings() -> dict[str, Any]:
    return json.loads((FIXTURES / EXPECTED_FIXTURE).read_text(encoding="utf-8"))


def memory_log_class() -> type:
    """The memory-log class the graph module binds, found by its interface (not its name) and
    asserted to come from THIS checkout. That it is the class the graph instantiates is
    checked against the live graph (test_memory_log_class_under_test_is_the_one_the_graph_
    instantiates)."""
    cls = get_class_by_interface(MEMORY_LOG_METHODS, *MEMORY_LOG_HOMES)
    module_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
    assert module_file and Path(module_file).resolve().is_relative_to(REPO_ROOT.resolve()), (
        f"{cls.__qualname__} resolved to {module_file!r}, not under {REPO_ROOT} — the test would "
        "judge another checkout (editable install / site-packages copy)")
    return cls


def file_state(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return sha256_file(path), st.st_size, st.st_mtime_ns


def dir_listing(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


# ============================================================================ run captures

def first_seq(calls: list[dict[str, Any]]) -> int:
    assert calls, "no calls"
    return min(int(c["seq"]) for c in calls)


def settlement_price_calls(res, since: str, before_seq: int) -> list[dict[str, Any]]:
    """Daily-price fetches for a settlement window starting at ``since`` (library layer),
    made before ``before_seq`` (normally the Portfolio Manager's first LLM call)."""
    out = []
    for d in res.data_calls:
        if d.get("lib") != "yfinance" or d.get("api") not in PRICE_APIS:
            continue
        if str(d.get("start") or "")[:10] != since:
            continue
        if int(d["seq"]) < before_seq:
            out.append(d)
    return out


def sqlite_files(root: Path) -> list[str]:
    """Every SQLite database file under ``root`` (checkpoints are the fork's only SQLite use)."""
    found = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as fh:
                head = fh.read(16)
        except OSError:
            continue
        if head == b"SQLite format 3\x00":
            found.append(str(p.relative_to(root)))
    return found


# ============================================================================ file-write observer

AUDIT_OUT_ENV = "APMEM_AUDIT_OUT"
AUDIT_WATCH_ENV = "APMEM_AUDIT_WATCH"

# Installed as ``sitecustomize`` through PYTHONPATH (alpha-pulse also runs the
# subprocess with one extra PYTHONPATH entry). It only OBSERVES: a
# sys.addaudithook that records file writes, renames and removals under one
# directory to a JSON-lines file. It patches nothing and never writes to stdout.
# Any sitecustomize the interpreter already had (Homebrew Python ships one) is
# found further down sys.path and executed first, so startup is otherwise unchanged.
AUDIT_SITECUSTOMIZE = r'''
"""Test-only observer for the alpha-pulse memory contract (see _memory_helpers.py)."""
import importlib.machinery as _machinery
import importlib.util as _util
import json as _json
import os as _os
import sys as _sys


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

_OUT = _os.environ.get("APMEM_AUDIT_OUT")
_WATCH = _os.environ.get("APMEM_AUDIT_WATCH")
if _OUT and _WATCH:
    _watch = _os.path.realpath(_WATCH)
    _marker = _os.path.basename(_watch)
    _fh = open(_OUT, "a", encoding="utf-8")
    _busy = [False]
    _WRITE_FLAGS = _os.O_WRONLY | _os.O_RDWR | _os.O_APPEND | _os.O_CREAT | _os.O_TRUNC
    _EVENTS = frozenset({"open", "os.rename", "os.remove", "os.truncate",
                         "shutil.copyfile", "shutil.move"})

    def _norm(p):
        try:
            p = _os.fsdecode(_os.fspath(p))
        except TypeError:
            return None
        if _marker not in p:
            return None
        rp = _os.path.realpath(p)
        if rp == _watch or rp.startswith(_watch + _os.sep):
            return rp
        return None

    def _hook(event, args):
        if event not in _EVENTS or _busy[0]:
            return
        _busy[0] = True
        try:
            rec = None
            if event == "open":
                path = args[0] if len(args) > 0 else None
                mode = args[1] if len(args) > 1 else None
                flags = args[2] if len(args) > 2 else None
                if path is None or isinstance(path, int):
                    return
                writes = bool(isinstance(flags, int) and flags & _WRITE_FLAGS) or bool(
                    isinstance(mode, str) and any(c in mode for c in "wax+"))
                if not writes:
                    return
                p = _norm(path)
                if p:
                    rec = {"event": "open-write", "path": p}
            elif event in ("os.rename", "shutil.copyfile", "shutil.move"):
                src, dst = _norm(args[0]), _norm(args[1])
                if src or dst:
                    rec = {"event": event, "path": src, "dst": dst}
            else:
                p = _norm(args[0])
                if p:
                    rec = {"event": event, "path": p}
            if rec is not None:
                rec["pid"] = _os.getpid()
                _fh.write(_json.dumps(rec) + "\n")
                _fh.flush()
        except Exception:
            pass
        finally:
            _busy[0] = False

    _sys.addaudithook(_hook)
'''


def write_audit_site(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(AUDIT_SITECUSTOMIZE.lstrip("\n"), encoding="utf-8")
    return directory


def read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def written_paths(records: list[dict[str, Any]]) -> set[str]:
    """Every path a run created, wrote, renamed (either side) or removed."""
    out: set[str] = set()
    for r in records:
        for key in ("path", "dst"):
            if r.get(key):
                out.add(os.path.realpath(r[key]))
    return out
