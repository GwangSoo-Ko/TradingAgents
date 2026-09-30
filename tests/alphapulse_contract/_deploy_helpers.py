"""Private helpers of test_alphapulse_deploy_contract.py (not part of the harness API).

How alpha-pulse gets this code onto its interpreter: it runs a clone of the deployed
branch without writing into it; when ``sha256(pyproject.toml)`` differs from the hash of
the file its environment was built from, it builds a FRESH venv on CPython 3.11 and
installs ``'<copy of the clone>[vertex]'`` into it (``uv pip install``). Runs are then
``<venv>/bin/python <clone>/main.py TICKER [DATE]`` with ``cwd=<clone>``, so the source
tree (``sys.path[0]``) shadows the installed copy and the venv only supplies the
third-party dependencies.

Static half (standard library only at import time; ``packaging`` is imported
lazily -- it is a dependency of langchain-core, so it is present wherever the
fork is installed):

* :func:`read_pyproject` -- ``tomllib`` (``tomli`` on the Python 3.10 CI lane).
* :func:`py311_syntax_offenses` -- does a file parse as Python 3.11? On a newer
  interpreter ``ast.parse(feature_version=(3, 11))`` still accepts PEP 701
  f-strings, so those are checked with the tokenizer.
* :func:`scan_imports` -- every absolute import of a file, with whether it runs at
  import time and whether a failure is tolerated (a guard that recovers).
* :func:`dependency_closure` / :func:`provider_of` -- what ``pip``/``uv`` would
  install for a set of declared requirements, and which distribution provides an
  import name.

Subprocess half (run BY PATH, the way production runs main.py, so ``sys.path[0]``
is set to the repo root before anything of the fork is imported)::

    <python> tests/alphapulse_contract/_deploy_helpers.py <repo_root> <run_dir>

reads ``<run_dir>/probe_spec.json`` (``{"modules": [...], "optional": [...],
"walk": [package, ...], "named": {label: [candidate module, ...]}}``), installs the
harness's network guards (``_fake_data``: sockets and curl_cffi refused and
recorded, ``.env`` lookup fenced) plus an audit hook that records every
filesystem write, network and process event, imports the modules and writes
``<run_dir>/probe.json``. Nothing but the harness guards is faked: no data or LLM
fake is installed, because importing must not need either.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI lane: pytest pulls in tomli there
    import tomli as tomllib

FIRST_PARTY = frozenset({"tradingagents", "cli", "main"})
_IMPORT_ERROR_NAMES = frozenset({"ImportError", "ModuleNotFoundError", "Exception",
                                 "BaseException"})
# Extras that exist for development only; they are never installed in production.
DEV_EXTRAS = frozenset({"dev", "test", "tests", "lint", "docs", "doc", "typing"})
# Marker environment the dependency closure is evaluated for: CPython 3.11 on Linux.
PRODUCTION_MARKER_ENV = {
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "os_name": "posix",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "aarch64",
    "python_version": "3.11",
    "python_full_version": "3.11.11",
}


# =============================================================================== pyproject

def read_pyproject(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def python_sources(repo_root: Path) -> list[Path]:
    """Every .py file production can import: tradingagents/, cli/ and main.py."""
    files = [p for pkg in ("tradingagents", "cli")
             for p in (repo_root / pkg).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files) + [repo_root / "main.py"]


def module_name(repo_root: Path, path: Path) -> str:
    parts = list(path.relative_to(repo_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# =============================================================================== syntax

_QUOTE_RE = re.compile(r"^[A-Za-z]*('''|\"\"\"|'|\")")


def _quote_of(token_text: str) -> str:
    m = _QUOTE_RE.match(token_text)
    return m.group(1) if m else ""


def _quote_conflicts(inner: str, outer: str) -> bool:
    """Would ``inner`` have ended the ``outer`` f-string literal under the 3.11 lexer?

    Up to 3.11 an f-string was first lexed as an ordinary string literal, so a
    quote inside a replacement field ended it: any use of the outer quote character
    ends a single-quoted literal; only the same triple quote ends a triple-quoted one.
    """
    if not inner or not outer:
        return False
    if len(outer) == 1:
        return outer in inner
    return inner == outer


def pep701_offenses(source: str) -> list[tuple[int, str]]:
    """f-string forms that Python >= 3.12 accepts (PEP 701) and Python 3.11 rejects.

    Only meaningful on 3.12+, where the tokenizer splits f-strings into
    FSTRING_START / FSTRING_MIDDLE / FSTRING_END; on 3.11 the parser itself refuses
    these forms, so this returns [] there.
    """
    if sys.version_info < (3, 12):
        return []
    import token
    import tokenize

    offenses: list[tuple[int, str]] = []
    # One entry per open f-string: its quote and the brace depth of its replacement
    # fields (0 = in its literal text, > 0 = in a replacement field).
    stack: list[dict[str, Any]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == token.FSTRING_START:
            quote = _quote_of(tok.string)
            if any(ctx["depth"] > 0 and _quote_conflicts(quote, ctx["quote"]) for ctx in stack):
                offenses.append((tok.start[0], "nested f-string reuses an enclosing quote"))
            stack.append({"quote": quote, "depth": 0})
            continue
        if not stack:
            continue
        if tok.type == token.FSTRING_END:
            stack.pop()
            continue
        top = stack[-1]
        if tok.type == token.OP and tok.string in ("{", "}"):
            top["depth"] += 1 if tok.string == "{" else -1
            continue
        if tok.type == token.FSTRING_MIDDLE:
            # Literal text (or format spec) of an f-string that itself sits inside an
            # enclosing replacement field: 3.11 lexed it as part of that expression.
            if any(ctx["depth"] > 0 for ctx in stack[:-1]):
                if "\\" in tok.string:
                    offenses.append((tok.start[0], "backslash inside an f-string replacement "
                                                   "field"))
                if "#" in tok.string:
                    offenses.append((tok.start[0], "'#' inside an f-string replacement field"))
            continue
        if top["depth"] == 0:
            continue
        fields = [ctx for ctx in stack if ctx["depth"] > 0]
        if tok.type == token.STRING:
            quote = _quote_of(tok.string)
            if any(_quote_conflicts(quote, ctx["quote"]) for ctx in fields):
                offenses.append((tok.start[0], "string inside a replacement field reuses the "
                                               "enclosing f-string quote"))
            if "\\" in tok.string:
                offenses.append((tok.start[0], "backslash inside an f-string replacement field"))
            if "#" in tok.string:
                offenses.append((tok.start[0], "'#' inside an f-string replacement field"))
        elif tok.type == token.COMMENT:
            offenses.append((tok.start[0], "comment inside an f-string replacement field"))
        elif tok.type in (token.NL, token.NEWLINE) and any(len(ctx["quote"]) == 1
                                                            for ctx in fields):
            offenses.append((tok.start[0], "line break inside the replacement field of a "
                                           "single-quoted f-string"))
    return offenses


def py311_syntax_offenses(path: Path, display: str) -> list[str]:
    """Reasons ``path`` would not compile on CPython 3.11 (empty when it does)."""
    source = path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # invalid-escape SyntaxWarnings are not errors
        try:
            tree = ast.parse(source, filename=display, feature_version=(3, 11))
            compile(tree, display, "exec", dont_inherit=True)
        except SyntaxError as exc:
            return [f"{display}:{exc.lineno}: SyntaxError: {exc.msg}"]
    return [f"{display}:{line}: {why} (Python >= 3.12 only, PEP 701)"
            for line, why in pep701_offenses(source)]


# =============================================================================== imports

def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any((isinstance(n, ast.Name) and n.id in _IMPORT_ERROR_NAMES)
               or (isinstance(n, ast.Attribute) and n.attr in _IMPORT_ERROR_NAMES)
               for n in names)


def _handler_recovers(handler: ast.ExceptHandler) -> bool:
    """False when the handler re-raises (the 'explain what to install' pattern)."""
    return not any(isinstance(stmt, ast.Raise) for stmt in handler.body)


def _suppresses_import_error(node: ast.With) -> bool:
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "suppress" and any(
                isinstance(a, (ast.Name, ast.Attribute))
                and (getattr(a, "id", None) or getattr(a, "attr", None)) in _IMPORT_ERROR_NAMES
                for a in call.args):
            return True
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return "TYPE_CHECKING" in ast.unparse(test)


def scan_imports(path: Path, display: str) -> list[dict[str, Any]]:
    """Absolute imports of one file: {module, line, where, at_import_time, tolerated}.

    ``at_import_time``: the statement runs when the module is imported (not inside a
    function). ``tolerated``: it sits in a ``try`` whose ImportError handler recovers,
    or in ``contextlib.suppress(ImportError)`` -- a missing package is then not fatal.
    A handler that re-raises (with install instructions) does NOT make it tolerated.
    Imports under ``if TYPE_CHECKING:`` never run and are left out; relative imports
    are first-party and left out.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=display)
    found: list[dict[str, Any]] = []

    def record(node: ast.stmt, module: str, names: list[str], in_function: bool,
               tolerated: bool) -> None:
        found.append({"module": module, "names": names, "line": node.lineno,
                      "where": f"{display}:{node.lineno}", "at_import_time": not in_function,
                      "tolerated": tolerated})

    def visit(nodes: list[ast.AST], in_function: bool, tolerated: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(node, alias.name, [], in_function, tolerated)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    record(node, node.module, [a.name for a in node.names], in_function,
                           tolerated)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(list(ast.iter_child_nodes(node)), True, tolerated)
            elif isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
                guard = any(_handler_catches_import_error(h) and _handler_recovers(h)
                            for h in node.handlers)
                visit(node.body, in_function, tolerated or guard)
                visit([*node.handlers, *node.orelse, *node.finalbody], in_function, tolerated)
            elif isinstance(node, ast.With):
                visit(node.body, in_function, tolerated or _suppresses_import_error(node))
                visit([i.context_expr for i in node.items], in_function, tolerated)
            elif isinstance(node, ast.If) and _is_type_checking(node.test):
                visit(node.orelse, in_function, tolerated)
            else:
                visit(list(ast.iter_child_nodes(node)), in_function, tolerated)

    visit(tree.body, False, False)
    return found


def main_py_fork_imports(repo_root: Path) -> list[dict[str, Any]]:
    """Fork modules main.py imports -- at import time or lazily after propagate() --
    in source order: [{module, tolerated, where}]. ``from pkg import submodule``
    contributes the submodule too."""
    out: list[dict[str, Any]] = []
    for imp in scan_imports(repo_root / "main.py", "main.py"):
        if imp["module"].split(".")[0] not in FIRST_PARTY - {"main"}:
            continue
        targets = [imp["module"]]
        base = repo_root.joinpath(*imp["module"].split("."))
        targets += [f"{imp['module']}.{name}" for name in imp["names"]
                    if (base / name).is_dir() or (base / f"{name}.py").is_file()]
        out += [{"module": t, "tolerated": imp["tolerated"], "where": imp["where"]}
                for t in targets]
    return out


# =============================================================================== dependencies

def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_groups(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    """{'base': [...], <extra>: [...]} requirement strings as declared."""
    project = pyproject.get("project") or {}
    groups = {"base": list(project.get("dependencies") or [])}
    for extra, reqs in (project.get("optional-dependencies") or {}).items():
        groups[extra] = list(reqs or [])
    return groups


def dependency_closure(requirements: list[str], marker_env: dict[str, str] | None = None
                       ) -> dict[str, list[str]]:
    """{canonical distribution name: requirement chain} that installing ``requirements``
    pulls in, resolved through the metadata of the distributions installed here.

    Markers are evaluated for ``marker_env`` (production by default). A distribution
    that is required but not installed here is still listed (its own dependencies are
    then unknown and not followed).
    """
    import importlib.metadata as md

    from packaging.requirements import Requirement

    env = dict(PRODUCTION_MARKER_ENV if marker_env is None else marker_env)
    seen: dict[str, set[str]] = {}
    chain: dict[str, list[str]] = {}
    todo: list[tuple[Requirement, set[str], list[str]]] = [
        (Requirement(r), {""}, []) for r in requirements]
    while todo:
        req, parent_extras, via = todo.pop()
        if req.marker is not None and not any(
                req.marker.evaluate({**env, "extra": e}) for e in parent_extras):
            continue
        name = canonical(req.name)
        extras = set(req.extras)
        if name in seen and extras <= seen[name]:
            continue
        seen.setdefault(name, set()).update(extras)
        chain.setdefault(name, [*via, str(req)])
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        for text in dist.requires or []:
            todo.append((Requirement(text), {""} | seen[name], chain[name]))
    return chain


def provider_of(module: str) -> list[str]:
    """Canonical names of the installed distributions that provide ``module``.

    Uses the top-level map; for a namespace shared by several distributions
    (``google``) the second component decides, from each candidate's file list.
    """
    import importlib.metadata as md

    top, _, rest = module.partition(".")
    dists = sorted({canonical(d) for d in _packages_distributions().get(top, [])})
    if len(dists) <= 1 or not rest:
        return dists
    second = rest.split(".")[0]
    narrowed = []
    for dist in dists:
        try:
            files = md.files(dist) or []
        except md.PackageNotFoundError:
            continue
        for f in files:
            parts = Path(str(f)).parts
            if len(parts) >= 2 and parts[0] == top and (
                    parts[1] == second or parts[1].split(".")[0] == second):
                narrowed.append(dist)
                break
    return narrowed or dists


_PD_CACHE: dict[str, list[str]] = {}


def _packages_distributions() -> dict[str, list[str]]:
    if not _PD_CACHE:
        import importlib.metadata as md
        _PD_CACHE.update(md.packages_distributions())
    return _PD_CACHE


def is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


# =============================================================================== subprocess

_WRITE_MODE = re.compile(r"[wax+]")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_FS_EVENTS = frozenset({
    "os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.symlink", "os.link", "os.truncate",
    "os.chmod", "os.chown", "os.utime", "os.chflags", "os.lchflags", "os.setxattr",
    "os.removexattr", "os.mkfifo", "os.mknod", "shutil.copyfile", "shutil.copymode",
    "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.chown",
    "shutil.make_archive", "shutil.unpack_archive", "tempfile.mkstemp", "tempfile.mkdtemp",
    "sqlite3.connect",
})
_NET_EVENTS = frozenset({
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
    "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto", "socket.sendmsg",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
    "webbrowser.open",
})
_PROC_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn",
                          "os.exec", "os.fork", "os.forkpty", "pty.spawn"})


class _Audit:
    """sys.addaudithook recorder of write / network / process events (cannot be removed,
    so ``stop()`` just ends the recording)."""

    def __init__(self) -> None:
        self.recording = True
        self.writes: list[dict[str, Any]] = []
        self.network: list[dict[str, Any]] = []
        self.processes: list[dict[str, Any]] = []
        self._busy = threading.local()
        self._self_file = str(Path(__file__).resolve())
        sys.addaudithook(self._hook)

    def stop(self) -> None:
        self.recording = False

    def _stack(self) -> list[str]:
        frames = []
        frame = sys._getframe(2)
        while frame is not None and len(frames) < 8:
            filename = frame.f_code.co_filename
            if filename != self._self_file and not filename.startswith("<frozen"):
                frames.append(f"{filename}:{frame.f_lineno}:{frame.f_code.co_name}")
            frame = frame.f_back
        return frames

    def _hook(self, event: str, args: tuple) -> None:
        if not self.recording or getattr(self._busy, "on", False):
            return
        self._busy.on = True
        try:
            self._handle(event, args)
        finally:
            self._busy.on = False

    def _handle(self, event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            if isinstance(path, int) or path is None:
                return
            path = os.fsdecode(path)
            writing = (isinstance(mode, str) and bool(_WRITE_MODE.search(mode))) or (
                isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            if writing and path != os.devnull:
                self.writes.append({"event": "open", "target": path, "mode": mode,
                                    "stack": self._stack()})
        elif event in _FS_EVENTS:
            target = [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else repr(a)
                      for a in args[:2]]
            if event == "sqlite3.connect" and target and target[0] in (":memory:", ""):
                return
            self.writes.append({"event": event, "target": target, "stack": self._stack()})
        elif event in _NET_EVENTS:
            detail = repr(args[1] if event == "socket.connect" and len(args) > 1 else args[:2])
            self.network.append({"event": event, "target": detail[:200], "stack": self._stack()})
            if event == "socket.connect" and len(args) > 1 and isinstance(args[1], (str, bytes)):
                return  # AF_UNIX: local, recorded only
            raise PermissionError(f"alpha-pulse deploy probe: network refused ({event})")
        elif event in _PROC_EVENTS:
            self.processes.append({"event": event, "target": repr(args[:2])[:200],
                                   "stack": self._stack()})


def _walk_modules(repo_root: Path, package: str) -> list[str]:
    return [module_name(repo_root, p)
            for p in sorted((repo_root / package).rglob("*.py")) if "__pycache__" not in p.parts]


def _import(name: str) -> dict[str, Any]:
    import importlib
    before = set(sys.modules)
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 -- report every failure by name
        # A failed import leaves half-initialised fork packages behind; drop them so the
        # next module reports its own root cause instead of a KeyError on the parent.
        for loaded in set(sys.modules) - before:
            if loaded.split(".")[0] in FIRST_PARTY:
                del sys.modules[loaded]
        return {"module": name, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=-6)}
    return {"module": name, "ok": True, "file": getattr(mod, "__file__", None)}


def _self_test(audit: _Audit, rt: Any, scratch: Path) -> dict[str, Any]:
    """Make one write, one mkdir and two network attempts on purpose; report what the
    audit hook and the harness guard saw, then forget them (so they cannot mask or
    pollute the real observation). The address is TEST-NET-1: never routable."""
    import _socket
    import socket

    canary = scratch / "deploy-probe-canary.txt"
    canary_dir = scratch / "deploy-probe-canary.d"
    with canary.open("w", encoding="utf-8") as fh:
        fh.write("canary")
    os.mkdir(canary_dir)
    errors = {}
    try:
        raw = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            raw.connect(("192.0.2.1", 9))
        finally:
            raw.close()
    except BaseException as exc:  # noqa: BLE001 -- the refusal is the point
        errors["c_level_connect"] = type(exc).__name__
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=1)
    except BaseException as exc:  # noqa: BLE001
        errors["create_connection"] = type(exc).__name__
    seen = {"writes": [(w["event"], w["target"]) for w in audit.writes],
            "network": [n["event"] for n in audit.network],
            "blocked": [b["api"] for b in rt.CAPTURES["network"]["blocked"]],
            "errors": errors, "canary": str(canary), "canary_dir": str(canary_dir)}
    audit.recording = False
    canary.unlink()
    canary_dir.rmdir()
    audit.writes.clear()
    audit.network.clear()
    rt.CAPTURES["network"]["blocked"].clear()
    audit.recording = True
    return seen


def run_probe(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    spec = json.loads((run_dir / "probe_spec.json").read_text(encoding="utf-8"))
    sys.path[0] = str(repo_root)  # what `python <repo>/main.py` gets
    audit = _Audit()
    from tests.alphapulse_contract import _fake_data, _runtime as rt
    from tests.alphapulse_contract._compat import import_first

    rt.init({"name": "deploy-probe"}, run_dir)
    _fake_data.install_socket_guard()
    _fake_data.install_curl_guard()
    _fake_data.install_dotenv_fence()

    result: dict[str, Any] = {"python": list(sys.version_info[:3]), "cwd": os.getcwd(),
                              "sys_path0": sys.path[0], "imports": [], "walked": [], "named": {}}
    if spec.get("self_test"):
        result["self_test"] = _self_test(audit, rt, Path(spec["self_test"]))
    optional = set(spec.get("optional") or [])
    for name in spec.get("modules") or []:
        entry = _import(name)
        entry["optional"] = name in optional
        result["imports"].append(entry)
    for package in spec.get("walk") or []:
        for name in _walk_modules(repo_root, package):
            result["walked"].append(_import(name))
    for label, candidates in (spec.get("named") or {}).items():
        try:
            mod = import_first(*candidates)
            result["named"][label] = {"ok": True, "module": mod.__name__,
                                      "file": getattr(mod, "__file__", None)}
        except BaseException as exc:  # noqa: BLE001
            result["named"][label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    audit.stop()
    result.update(writes=audit.writes, network=audit.network, processes=audit.processes,
                  blocked=rt.CAPTURES["network"]["blocked"], notes=rt.CAPTURES["notes"],
                  dotenv=rt.CAPTURES["dotenv"])
    return result


if __name__ == "__main__":
    _repo, _run_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    _result = run_probe(_repo, _run_dir)
    (_run_dir / "probe.json").write_text(json.dumps(_result, ensure_ascii=False, indent=1,
                                                    default=repr), encoding="utf-8")
