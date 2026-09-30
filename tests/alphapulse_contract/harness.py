"""Run the fork's main.py the way alpha-pulse does, and read what it prints.

    from tests.alphapulse_contract.harness import run_main
    from tests.alphapulse_contract import scenarios

    res = run_main(tmp_path, scenarios.get("s1_nightly_kr_holding_sell"))
    res.assert_ok()
    assert res.decision == res.plan["rating"]

``run_main`` launches ``<python> _bootstrap.py <run_dir>`` with cwd = repo root and
an environment built from scratch (nothing inherited from the developer's shell):
HOME, TMPDIR and every TradingAgents path under ``run_dir``; TZ=Asia/Seoul;
GOOGLE_CLOUD_PROJECT=tpmn-dev (the fork's documented example project);
PYTHONUNBUFFERED=1; PYTHONDONTWRITEBYTECODE=1 (a deployed source tree need not be
writable); a per-ticker TRADINGAGENTS_MEMORY_LOG_PATH and
TRADINGAGENTS_CHECKPOINT_ENABLED=false, as alpha-pulse sets them for every run; and
TRADINGAGENTS_POSITION_CONTEXT from the scenario.

stdout is read with ``contract_reader`` -- a clean-room reader of the output grammar the
fork documents itself (docs/INTEGRATION.md §1b, main.py's comments): ``decision`` and
``report_path`` come from ``read_stdout``, ``plan`` from ``read_trade_plan``, and
``stdout_lines`` / ``tail`` / ``trade_plan_lines`` cut stdout the same way
(``contract_reader.lines``).
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import contract_reader

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
BOOTSTRAP = PACKAGE_DIR / "_bootstrap.py"
FIXTURES = PACKAGE_DIR / "fixtures"

DEFAULT_TIMEOUT_S = 600
DECISION_WORDS = contract_reader.DECISION_WORDS
REPORT_SAVED_LINE_RE = re.compile(
    r"^Report saved: (?P<path>.+/reports/(?P<dir>[^/]+)_(?P<stamp>\d{8}_\d{6})/complete_report\.md)$"
)
# Directories a run can never legitimately own but other tools touch concurrently
# (caches, virtualenvs, other worktrees nested under .claude/, editor/plugin state).
_SKIP_SNAPSHOT_DIRS = {"__pycache__", ".pytest_cache", ".git", ".ruff_cache", ".mypy_cache",
                       ".venv", "venv", ".tox", ".nox", "node_modules", ".claude", ".remember",
                       ".benchmarks", ".idea", ".vscode"}
_SKIP_SNAPSHOT_FILES = {".DS_Store"}


def kst_today() -> datetime.date:
    """Today's date in Asia/Seoul (what main.py defaults to under TZ=Asia/Seoul)."""
    return datetime.datetime.now(ZoneInfo("Asia/Seoul")).date()


def harness_python() -> str:
    """Interpreter for the subprocess: $ALPHAPULSE_HARNESS_PYTHON, else this one."""
    return os.environ.get("ALPHAPULSE_HARNESS_PYTHON") or sys.executable


def memory_log_name(ticker: str) -> str:
    """One memory log per ticker, as production names it in TRADINGAGENTS_MEMORY_LOG_PATH."""
    return f"trading_memory_{ticker}.md"


def base_env(run_dir: Path, ticker: str) -> dict[str, str]:
    """The environment every harness run starts from (before scenario/env overrides)."""
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(run_dir / "home"),
        "TMPDIR": str(run_dir / "tmp"),
        "LANG": "C.UTF-8",
        "TZ": "Asia/Seoul",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # production runs the subprocess with one extra PYTHONPATH entry (its own
        # application directory); mimic an extra (empty) entry so import resolution
        # has the same shape.
        "PYTHONPATH": str(run_dir / "fake_app"),
        "GOOGLE_CLOUD_PROJECT": "tpmn-dev",
        "TRADINGAGENTS_RESULTS_DIR": str(run_dir / "results"),
        "TRADINGAGENTS_CACHE_DIR": str(run_dir / "cache"),
        "TRADINGAGENTS_MEMORY_LOG_PATH": str(run_dir / "memory" / memory_log_name(ticker)),
        "TRADINGAGENTS_CHECKPOINT_ENABLED": "false",
    }


def _instrument_company(ticker: str) -> str | None:
    data = json.loads((FIXTURES / "instruments.json").read_text(encoding="utf-8"))
    info = (data.get(ticker) or {}).get("info") or {}
    return info.get("longName") or info.get("shortName")


def _unique_dir(parent: Path, name: str) -> Path:
    candidate = parent / name
    n = 2
    while candidate.exists():
        candidate = parent / f"{name}-{n}"
        n += 1
    return candidate


def tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """{relative path: (size, mtime_ns)} for every file under root (caches excluded)."""
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_SNAPSHOT_DIRS]
        for fn in filenames:
            if fn in _SKIP_SNAPSHOT_FILES:
                continue
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


@dataclass
class RunResult:
    """Everything one main.py run produced; stdout read with ``contract_reader``."""

    scenario: dict[str, Any]
    argv: list[str]
    rc: int
    stdout: str
    stderr: str
    run_dir: Path
    env: dict[str, str]
    captures: dict[str, Any]
    duration_s: float
    decision: str | None = None
    report_path: str | None = None
    plan: dict[str, Any] | None = None
    report_dir: Path | None = None
    report_files: list[str] = field(default_factory=list)
    complete_report: str | None = None
    memory_log_path: Path | None = None
    memory_log: str | None = None
    memory_log_seed: str | None = None

    def __repr__(self) -> str:
        # pytest prints the repr of every object in a failing assert; the generated
        # dataclass repr would dump stdout and the whole capture document. describe()
        # has the diagnostics.
        return (f"RunResult(scenario={self.scenario.get('name')!r}, argv={self.argv!r}, "
                f"rc={self.rc}, decision={self.decision!r}, run_dir='{self.run_dir}')")

    # ------------------------------------------------------------------ stdout
    @property
    def stdout_lines(self) -> list[str]:
        """stdout cut into the lines print() wrote (``contract_reader.lines``: at "\\n" only)."""
        return contract_reader.lines(self.stdout)

    @property
    def tail(self) -> list[str]:
        """The last three stdout lines (decision, TRADE_PLAN_JSON, Report saved)."""
        return self.stdout_lines[-3:]

    @property
    def trade_plan_lines(self) -> list[str]:
        """Every stdout line of the documented ``TRADE_PLAN_JSON: {...}`` form."""
        return [ln for ln in self.stdout_lines if contract_reader.is_plan_line(ln)]

    @property
    def results_dir(self) -> Path:
        return Path(self.env["TRADINGAGENTS_RESULTS_DIR"])

    @property
    def cache_dir(self) -> Path:
        return Path(self.env["TRADINGAGENTS_CACHE_DIR"])

    @property
    def home(self) -> Path:
        return Path(self.env["HOME"])

    def describe(self) -> str:
        """Short diagnostic for assertion messages."""
        exit_info = (self.captures or {}).get("exit")
        return (f"scenario={self.scenario.get('name')} argv={self.argv} rc={self.rc} "
                f"duration={self.duration_s:.1f}s\n--- stdout tail ---\n"
                + "\n".join(self.stdout_lines[-5:])
                + f"\n--- stderr tail ---\n{self.stderr[-3000:]}\n--- exit ---\n{exit_info}")

    def assert_ok(self) -> RunResult:
        assert self.rc == 0, self.describe()
        return self

    # ------------------------------------------------------------------ reports
    def report_file(self, relative: str) -> str:
        assert self.report_dir is not None, "no report dir (no 'Report saved:' line)"
        return (self.report_dir / relative).read_text(encoding="utf-8")

    # ------------------------------------------------------------------ captures
    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        return list((self.captures or {}).get("llm_calls") or [])

    def calls_for(self, role: str, kind: str | None = None) -> list[dict[str, Any]]:
        """LLM calls made by ``role`` (role keys as in cli.report_meta, plus 'reflector')."""
        return [c for c in self.llm_calls if c.get("role") == role and (kind is None or c.get("kind") == kind)]

    @staticmethod
    def prompt_text(call: dict[str, Any]) -> str:
        """All message contents of one call joined (system + human + tool messages)."""
        parts = []
        for m in call.get("messages") or []:
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in content)
            parts.append(str(content))
        return "\n\n".join(parts)

    def prompts_for(self, role: str, kind: str | None = None) -> list[str]:
        return [self.prompt_text(c) for c in self.calls_for(role, kind)]

    def prompt_for(self, role: str, kind: str | None = None) -> str:
        """The first prompt ``role`` sent (for PM/RM/Trader: the structured call)."""
        prompts = self.prompts_for(role, kind)
        assert prompts, f"role {role!r} made no LLM call; roles seen: {sorted(self.roles_called())}"
        return prompts[0]

    def roles_called(self) -> set[str]:
        return {c.get("role") for c in self.llm_calls if c.get("role")}

    @property
    def llm_constructions(self) -> list[dict[str, Any]]:
        return list((self.captures or {}).get("llm_constructions") or [])

    def llm_for(self, role: str) -> dict[str, Any]:
        """Construction record of the (single) LLM instance that served ``role``'s calls."""
        ids = {c.get("llm_id") for c in self.calls_for(role)}
        assert ids, f"role {role!r} made no LLM call; roles seen: {sorted(self.roles_called())}"
        assert len(ids) == 1, f"role {role!r} was served by several LLM instances: {ids}"
        (llm_id,) = ids
        records = self.llm_constructions
        # -1 (a harness model built outside both fakes) must not index the last record.
        assert isinstance(llm_id, int) and 0 <= llm_id < len(records), (
            f"role {role!r} was served by an LLM the fakes did not construct "
            f"(llm_id={llm_id!r}, {len(records)} construction records)")
        return records[llm_id]

    def llm_by_role(self) -> dict[str, dict[str, Any]]:
        return {role: self.llm_for(role) for role in sorted(self.roles_called())}

    @property
    def data_calls(self) -> list[dict[str, Any]]:
        return list((self.captures or {}).get("data_calls") or [])

    def tool_results(self, role: str | None = None) -> list[dict[str, Any]]:
        """Tool calls the analysts made and what the real tools returned (deduplicated).

        Each item: {role, name, args, content, tool_call_id}. Reconstructed from the
        messages the analyst sent back to the model after its ToolNode ran.
        """
        out: dict[str, dict[str, Any]] = {}
        for call in self.llm_calls:
            if role is not None and call.get("role") != role:
                continue
            requested = {}
            for m in call.get("messages") or []:
                for tc in m.get("tool_calls") or []:
                    requested[tc.get("id")] = tc
                if m.get("type") == "tool" and m.get("tool_call_id") in requested:
                    tc = requested[m["tool_call_id"]]
                    out.setdefault(m["tool_call_id"], {
                        "role": call.get("role"), "name": tc.get("name"), "args": tc.get("args"),
                        "content": m.get("content"), "tool_call_id": m["tool_call_id"],
                    })
        return list(out.values())

    @property
    def propagate_calls(self) -> list[dict[str, Any]]:
        return list((self.captures or {}).get("propagate") or [])

    @property
    def trade_date_seen(self) -> str | None:
        """The trade_date main.py handed to TradingAgentsGraph.propagate (as str)."""
        calls = self.propagate_calls
        return calls[-1].get("trade_date") if calls else None

    @property
    def final_state(self) -> dict[str, Any] | None:
        calls = self.propagate_calls
        return calls[-1].get("final_state") if calls else None

    @property
    def network_attempts(self) -> list[dict[str, Any]]:
        return list(((self.captures or {}).get("network") or {}).get("blocked") or [])

    @property
    def unrouted_http(self) -> list[dict[str, Any]]:
        return list(((self.captures or {}).get("network") or {}).get("unrouted") or [])

    @property
    def env_seen(self) -> dict[str, str]:
        """os.environ inside the subprocess when propagate was entered (after .env loading);
        falls back to the environment at bootstrap start if propagate never ran."""
        caps = self.captures or {}
        return dict(caps.get("env_at_propagate") or caps.get("env_at_start") or {})

    @property
    def graph_config(self) -> dict[str, Any] | None:
        """``TradingAgentsGraph.config`` at propagate entry (what main.build_config() built)."""
        return (self.captures or {}).get("graph_config")

    def config_seen(self, node: str | None = None) -> dict[str, Any] | None:
        """get_config() as the data layer saw it during the run.

        ``node=None``: at the first data access made inside a graph node (the first
        tool call). Otherwise: at that top-level node's first LLM call.
        """
        caps = self.captures or {}
        if node is None:
            first = caps.get("config_at_first_tool_call") or {}
            return first.get("config")
        return (caps.get("config_by_node") or {}).get(node)


def run_main(tmp_path: Path, scenario: dict[str, Any] | str, argv: list[str] | None = None,
             env_overrides: dict[str, str | None] | None = None, *, python: str | None = None,
             timeout: float | None = None) -> RunResult:
    """Run ``main.py`` for ``scenario`` in a fresh directory under ``tmp_path``.

    ``scenario``: a dict (see scenarios.py) or a scenario name. ``argv`` replaces the
    scenario's argv (what alpha-pulse passes after main.py: [TICKER] or [TICKER, DATE]).
    ``env_overrides`` is applied last; a value of None removes the variable.
    Never raises for a failing run — inspect ``rc`` (use ``assert_ok()``).
    """
    if isinstance(scenario, str):
        from . import scenarios as _scenarios
        scenario = _scenarios.get(scenario)
    spec = copy.deepcopy(scenario)
    if argv is not None:
        spec["argv"] = [str(a) for a in argv]
    assert spec.get("argv"), "scenario needs argv: [TICKER] or [TICKER, DATE]"
    ticker = str(spec["argv"][0])
    spec.setdefault("company", _instrument_company(ticker))

    run_dir = _unique_dir(Path(tmp_path), str(spec.get("name") or "run"))
    for sub in ("home", "tmp", "results", "cache", "memory", "fake_app"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    env = base_env(run_dir, ticker)
    for key, value in (spec.get("env") or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    if spec.get("position_context") is not None:
        env["TRADINGAGENTS_POSITION_CONTEXT"] = str(spec["position_context"])
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    memory_log_path = Path(env["TRADINGAGENTS_MEMORY_LOG_PATH"]) if env.get(
        "TRADINGAGENTS_MEMORY_LOG_PATH") else None
    seed = spec.get("memory_log_seed")
    if seed is not None and memory_log_path is not None:
        memory_log_path.parent.mkdir(parents=True, exist_ok=True)
        memory_log_path.write_text(seed, encoding="utf-8")
    if spec.get("dotenv") is not None:
        (run_dir / "scenario.env").write_text(spec["dotenv"], encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    cmd = [python or harness_python(), str(BOOTSTRAP), str(run_dir)]
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True,
                          timeout=timeout or spec.get("timeout_s") or DEFAULT_TIMEOUT_S)
    duration = time.monotonic() - started
    # bytes -> text the way a subprocess consumer reads them: undecodable bytes are
    # replaced, never an error.
    stdout = proc.stdout.decode(errors="replace")
    stderr = proc.stderr.decode(errors="replace")
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

    captures_path = run_dir / "captures.json"
    captures = json.loads(captures_path.read_text(encoding="utf-8")) if captures_path.exists() else {}

    res = RunResult(scenario=spec, argv=list(spec["argv"]), rc=proc.returncode, stdout=stdout,
                    stderr=stderr, run_dir=run_dir, env=env, captures=captures,
                    duration_s=duration, memory_log_path=memory_log_path, memory_log_seed=seed)
    res.decision, res.report_path = contract_reader.read_stdout(stdout)
    res.plan = contract_reader.read_trade_plan(stdout)
    if res.report_path:
        rp = Path(res.report_path)
        res.report_dir = rp.parent if rp.suffix == ".md" else rp
        if res.report_dir.is_dir():
            res.report_files = sorted(p.relative_to(res.report_dir).as_posix()
                                      for p in res.report_dir.rglob("*") if p.is_file())
            complete = res.report_dir / "complete_report.md"
            if complete.is_file():
                res.complete_report = complete.read_text(encoding="utf-8")
    if memory_log_path is not None and memory_log_path.is_file():
        res.memory_log = memory_log_path.read_text(encoding="utf-8")
    return res
