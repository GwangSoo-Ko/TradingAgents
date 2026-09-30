"""Private helpers of test_alphapulse_output_contract.py (not part of the harness API).

1. ``main_py_imports(main_py)`` lists every import statement in main.py (AST): the
   module, the imported name, the function it runs in and whether it runs after the
   ``propagate(...)`` call (main.py imports its report writer lazily, i.e. only once
   the paid LLM run is over).
2. Run as a module, it resolves each of those imports exactly the way
   ``python <repo>/main.py`` would find them (``sys.path[0]`` = repo root, cwd = repo
   root, the harness's fenced ``.env`` and network guard installed) and writes the
   outcome to ``<run_dir>/imports.json``:

       cd <repo_root> && <python> -m tests.alphapulse_contract._output_helpers <repo_root> <run_dir>

   A missing module or name is then reported by name instead of as an rc 1 at the end
   of a run that already paid for every LLM call.

Only the standard library is imported at module level: the script mode installs the
harness fakes before anything from the fork is imported.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_IMPORT_ERROR_HANDLERS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """True when an ``except`` clause would swallow an ImportError (an optional import)."""
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERROR_HANDLERS for n in names)


def _propagate_line(func: ast.AST) -> int | None:
    """Line of the first ``<something>.propagate(...)`` call inside ``func``."""
    lines = [n.lineno for n in ast.walk(func)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "propagate"]
    return min(lines) if lines else None


def main_py_imports(main_py: Path) -> list[dict[str, Any]]:
    """Every import statement in ``main_py`` with where (and when) it runs."""
    tree = ast.parse(main_py.read_text(encoding="utf-8"), filename=str(main_py))
    records: list[dict[str, Any]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[tuple[str, int | None]] = []
            self.guarded = 0

        def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.functions.append((node.name, _propagate_line(node)))
            self.generic_visit(node)
            self.functions.pop()

        visit_FunctionDef = _function
        visit_AsyncFunctionDef = _function

        def visit_Try(self, node: ast.Try) -> None:
            guarded = any(_catches_import_error(h) for h in node.handlers)
            self.guarded += int(guarded)
            for stmt in node.body:
                self.visit(stmt)
            self.guarded -= int(guarded)
            for part in (*node.handlers, *node.orelse, *node.finalbody):
                self.visit(part)

        def _record(self, node: ast.stmt, module: str | None, name: str | None,
                    level: int = 0) -> None:
            function, propagate_at = self.functions[-1] if self.functions else (None, None)
            records.append({
                "line": node.lineno,
                "function": function,
                "after_propagate": bool(propagate_at and node.lineno > propagate_at),
                "module": module,
                "name": name,
                "level": level,
                "optional": self.guarded > 0,
            })

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self._record(node, alias.name, None)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                self._record(node, node.module, alias.name, node.level)

    _Visitor().visit(tree)
    return records


def _resolve(module: str, name: str | None) -> tuple[Any, Any]:
    """(module, object) for ``import module`` / ``from module import name``."""
    mod = importlib.import_module(module)
    if name is None or name == "*":
        return mod, mod
    if hasattr(mod, name):
        return mod, getattr(mod, name)
    if hasattr(mod, "__path__"):
        # ``from package import submodule`` binds a submodule that is not yet an attribute.
        try:
            return mod, importlib.import_module(f"{module}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{module}.{name}":
                raise
    # The same failure Python itself reports for ``from module import name``.
    raise ImportError(f"cannot import name {name!r} from {module!r} "
                      f"({getattr(mod, '__file__', None)})")


def probe_imports(repo_root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Resolve every main.py import like production would; record the outcome of each."""
    sys.path[0] = str(repo_root)  # exactly what `python <repo>/main.py` gets
    from tests.alphapulse_contract import _fake_data, _runtime as rt

    rt.init({"name": "main-py-import-probe"}, run_dir)
    _fake_data.install_all()  # fenced .env, no network, library-level data fakes

    records = main_py_imports(repo_root / "main.py")
    for rec in records:
        if rec["level"]:
            rec.update(ok=False, error="relative import in main.py: production runs it as a "
                                       "script, where a relative import cannot resolve")
            continue
        try:
            mod, obj = _resolve(rec["module"], rec["name"])
        except BaseException as exc:  # noqa: BLE001 — report every failure by name
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                       missing_module=getattr(exc, "name", None),
                       traceback=traceback.format_exc(limit=-4))
            continue
        # Where the imported object itself was defined (a name may be re-exported).
        home = obj.__name__ if isinstance(obj, type(sys)) else getattr(obj, "__module__", None)
        rec.update(ok=True, module_file=getattr(mod, "__file__", None),
                   object_module=home,
                   object_file=getattr(sys.modules.get(home or ""), "__file__", None))
    return records


if __name__ == "__main__":
    # python -m tests.alphapulse_contract._output_helpers <repo_root> <run_dir>
    _repo, _run_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    _records = probe_imports(_repo, _run_dir)
    (_run_dir / "imports.json").write_text(json.dumps(_records, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
