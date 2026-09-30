"""alpha-pulse deployment contract: what it takes for THIS tree to run where alpha-pulse runs it.

alpha-pulse does not import the fork; it installs and spawns it:

* it runs a clone of the deployed branch and never writes into that clone;
* whenever ``sha256(pyproject.toml)`` differs from the hash of the file its current
  environment was built from, it builds a FRESH virtualenv on CPython 3.11 and installs
  ``'<copy of the clone>[vertex]'`` into it (``uv pip install``) -- an isolated,
  non-editable PEP 517 build; if that fails the environment stays EMPTY and every spawn
  fails until someone notices;
* every analysis is ``<venv>/bin/python <clone>/main.py TICKER [DATE]`` with
  ``cwd=<clone>``, ``PYTHONDONTWRITEBYTECODE=1``, a HOME that does not persist and the
  three ``TRADINGAGENTS_*`` path variables pointing at persistent storage. The source tree
  (``sys.path[0]``) shadows the installed copy, so the venv supplies the third-party
  packages only.

The tests below freeze exactly those preconditions, reading pyproject with ``tomllib``,
building the tree the way uv does, compiling every file as Python 3.11 and importing
every fork module in a subprocess under the harness's network guard. Expected values are
literals taken from production (the hash of the pyproject.toml the production venv was
built from, Python 3.11, the ``vertex`` extra, the two Vertex SDKs) -- nothing is
recomputed by the code under test.

The static and dependency analysis lives in ``_deploy_helpers.py`` (private to this file).
"""

from __future__ import annotations

import ast
import email.parser
import fnmatch
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from . import _deploy_helpers as dh
from .harness import PACKAGE_DIR, REPO_ROOT, base_env, harness_python

# sha256 of a2981a7's pyproject.toml -- the tree production runs. After each successful
# rebuild the deployment stores the hash of the file it built from; before anything is
# promoted, that stored value is checked against this literal.
PRODUCTION_PYPROJECT_SHA256 = "1b270b35113358ddc995322fbf582c5cafd833399b238f1528a007a7f35feabe"
BUILD_BACKEND = "setuptools.build_meta"
PRODUCTION_EXTRA = "vertex"
# What main.build_config() runs on (vertex_anthropic -> ChatAnthropicVertex, which the
# anthropic SDK's Vertex client backs): {distribution: extras it must be installed with}.
VERTEX_SDKS = {"langchain-google-vertexai": frozenset(), "anthropic": frozenset({"vertex"})}
# The production image tracks the latest 3.11 patch release; admit all of them.
PRODUCTION_PYTHONS = [f"3.11.{patch}" for patch in range(21)]

# Fork-only modules a merge with upstream can drop or break (upstream has no KR vendors
# and no ticker resolver). Current home first, then where a merge could move them.
FORK_ONLY_MODULES = {
    "wisereport": ("tradingagents.dataflows.wisereport",
                   "tradingagents.dataflows.vendors.wisereport"),
    "naver_news": ("tradingagents.dataflows.naver_news",
                   "tradingagents.dataflows.vendors.naver_news"),
    "naver_discussion": ("tradingagents.dataflows.naver_discussion",
                         "tradingagents.dataflows.vendors.naver_discussion"),
    "opendart_fundamentals": ("tradingagents.dataflows.opendart_fundamentals",
                              "tradingagents.dataflows.vendors.opendart_fundamentals"),
    "opendart_common": ("tradingagents.dataflows.opendart_common",
                        "tradingagents.dataflows.vendors.opendart_common"),
    "kr_utils": ("tradingagents.dataflows.kr_utils",
                 "tradingagents.dataflows.vendors.kr_utils"),
    "ticker_resolver": ("tradingagents.dataflows.ticker_resolver",
                        "tradingagents.dataflows.vendors.ticker_resolver"),
}
WALKED_PACKAGES = ["tradingagents", "cli"]  # includes tradingagents/dataflows + /agents

# What the copy of the clone leaves out (VCS, venvs, nested worktrees, local build and run
# output at the top; caches anywhere). The build reads pyproject, README, LICENSE and the
# packages only.
_TOP_LEVEL_SKIP = {".git", ".venv", "venv", ".claude", ".remember", ".benchmarks",
                   "node_modules", "build", "dist", "results"}
_ANYWHERE_SKIP = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".DS_Store"}
_BUILD_WHEEL = (
    "import importlib, json, sys\n"
    "out, backend_name = sys.argv[1], sys.argv[2]\n"
    "backend = importlib.import_module(backend_name)\n"
    "hook = getattr(backend, 'get_requires_for_build_wheel', None)\n"
    "extra = list(hook()) if hook else []\n"
    "wheel = backend.build_wheel(out)\n"
    "print('BUILD-RESULT ' + json.dumps({'extra_requires': extra, 'wheel': wheel}))\n"
)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    top = Path(directory).resolve() == REPO_ROOT.resolve()
    return {n for n in names if n in _ANYWHERE_SKIP
            or (top and (n in _TOP_LEVEL_SKIP or n.endswith(".egg-info")))}


def _requirement(text: str):
    from packaging.requirements import Requirement
    return Requirement(text)


def _applies(req, extra: str = "") -> bool:
    """Does ``req`` get installed on production when ``extra`` is requested?"""
    return req.marker is None or req.marker.evaluate({**dh.PRODUCTION_MARKER_ENV,
                                                      "extra": extra})


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return dh.read_pyproject(REPO_ROOT)


def _run_probe(tmp_path: Path, spec: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
    """Run ``_deploy_helpers.py`` like production runs main.py: by path, cwd outside the
    checkout, HOME/TMPDIR/TRADINGAGENTS_* under tmp (the path dirs are NOT created)."""
    run_dir = tmp_path / name
    for sub in ("home", "tmp", "cwd", "fake_app", "selftest"):
        (run_dir / sub).mkdir(parents=True)
    spec = {**spec, "self_test": str(run_dir / "selftest")}
    (run_dir / "probe_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    proc = subprocess.run(
        [harness_python(), str(PACKAGE_DIR / "_deploy_helpers.py"), str(REPO_ROOT),
         str(run_dir)],
        cwd=str(run_dir / "cwd"), env=base_env(run_dir, "005930.KS"), capture_output=True,
        timeout=600)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-4000:]
    return json.loads((run_dir / "probe.json").read_text(encoding="utf-8")), run_dir


def _assert_probe_watched(probe: dict[str, Any]) -> None:
    """The probe's own guards observed the deliberate write, mkdir and network attempts --
    otherwise an empty observation below would prove nothing."""
    seen = probe["self_test"]
    assert ("open", seen["canary"]) in [tuple(w) for w in seen["writes"]], seen
    assert any(w[0] == "os.mkdir" and seen["canary_dir"] in w[1] for w in seen["writes"]), seen
    assert "socket.connect" in seen["network"], seen
    assert "socket.create_connection" in seen["blocked"], seen
    assert seen["errors"] == {"c_level_connect": "PermissionError",
                              "create_connection": "HarnessNetworkBlocked"}, seen


def _files_under(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) if root.exists() \
        else []


# ============================================================================ the venv rebuild

def test_pyproject_is_byte_identical_to_the_one_the_production_venv_was_built_from():
    """Breaks if the merge changes pyproject.toml in ANY byte (dependency list, version, even
    ruff settings). The deployment compares ``sha256sum pyproject.toml`` with the hash of the
    last successful rebuild; on any difference it deletes the venv and re-resolves every
    dependency unpinned (``uv pip install '<clone>[vertex]'``): minutes in which every
    spawned analysis fails with ModuleNotFoundError, then SDK versions nobody reviewed --
    and reverting the file re-resolves once more instead of restoring the old venv. Phase A
    of the merge keeps a2981a7's bytes; a dependency change is Phase B, promoted on its own
    -- that commit updates this literal (and needs a backup/rollout checklist)."""
    digest = hashlib.sha256((REPO_ROOT / "pyproject.toml").read_bytes()).hexdigest()
    assert digest == PRODUCTION_PYPROJECT_SHA256, (
        f"pyproject.toml changed (sha256 {digest}): pushing this rebuilds the production venv "
        "from scratch with fresh, unpinned resolution")


def test_build_backend_is_setuptools_and_the_isolated_build_env_gets_it(pyproject):
    """Breaks if the build backend changes (e.g. to hatchling, uv_build or poetry-core, or the
    [build-system] table disappears): uv then builds with a different backend that ignores
    [tool.setuptools] (package discovery, dynamic version), so the rebuild after the next
    pyproject change produces different metadata -- or fails and leaves the venv empty."""
    build = pyproject.get("build-system") or {}
    assert build.get("build-backend") == BUILD_BACKEND, build
    requires = [_requirement(r) for r in build.get("requires") or []]
    assert any(dh.canonical(r.name) == "setuptools" for r in requires), build


def test_vertex_extra_installs_the_vertex_sdks_production_runs_on(pyproject):
    """Breaks if the ``vertex`` extra disappears or loses an SDK -- upstream's pyproject
    (v0.5.1/v0.5.2) has no [vertex] extra at all, so resolving pyproject 'theirs' does it
    silently. ``uv pip install '<clone>[vertex]'`` only warns about an unknown extra and
    installs no Vertex SDK; every run then dies at its first LLM construction
    (vertex_anthropic -> ChatAnthropicVertex import error, rc 1) from the next rebuild on."""
    extras = (pyproject.get("project") or {}).get("optional-dependencies") or {}
    assert PRODUCTION_EXTRA in extras, sorted(extras)
    declared = {dh.canonical(r.name): r for r in map(_requirement, extras[PRODUCTION_EXTRA])}
    for name, needed_extras in VERTEX_SDKS.items():
        assert name in declared, (name, sorted(declared))
        req = declared[name]
        assert needed_extras <= set(req.extras), (name, str(req))
        assert _applies(req, PRODUCTION_EXTRA), (name, str(req))


def test_python_dotenv_is_an_unconditional_dependency(pyproject):
    """Breaks if python-dotenv stops being a base dependency. cli/utils.py imports ``dotenv``
    at module level and main.py imports cli.main only AFTER propagate(), so production would
    exit 1 after every paid run; upstream's tradingagents/__init__.py imports it unguarded,
    which would make every run fail at its first import instead."""
    base = [_requirement(r) for r in (pyproject.get("project") or {}).get("dependencies") or []]
    dotenv = [r for r in base if dh.canonical(r.name) == "python-dotenv"]
    assert dotenv, [str(r) for r in base]
    assert all(_applies(r) for r in dotenv), [str(r) for r in dotenv]


def test_requires_python_admits_every_python_3_11_release(pyproject):
    """Breaks if requires-python stops admitting 3.11 (upstream raising its floor to 3.12,
    say): uv then refuses to install the tree into the Python 3.11 image's venv -- after the
    deployment already deleted it -- and every analysis fails until the image changes."""
    from packaging.specifiers import SpecifierSet

    spec = SpecifierSet((pyproject.get("project") or {}).get("requires-python") or "")
    rejected = [v for v in PRODUCTION_PYTHONS if not spec.contains(v, prereleases=True)]
    assert rejected == [], f"requires-python {str(spec)!r} rejects {rejected}"


def test_package_discovery_is_explicit_and_covers_tradingagents_and_cli(pyproject):
    """Breaks if [tool.setuptools.packages.find] (include tradingagents*, cli*) is dropped or
    narrowed: this repo has main.py, tests/, scripts/ and assets/ beside the packages, so
    setuptools' automatic flat-layout discovery refuses to build ('Multiple top-level
    packages discovered') and the rebuild leaves the venv empty; a narrowed include
    silently ships a venv without ``cli`` or a subpackage."""
    packages = ((pyproject.get("tool") or {}).get("setuptools") or {}).get("packages")
    assert packages is not None, "no [tool.setuptools.packages]: automatic discovery fails here"
    wanted = ["tradingagents", "tradingagents.dataflows", "tradingagents.agents", "cli"]
    if isinstance(packages, list):
        missing = [p for p in wanted if p not in packages]
    else:
        find = packages.get("find") or {}
        include = find.get("include") or ["*"]
        exclude = find.get("exclude") or []
        missing = [p for p in wanted
                   if not any(fnmatch.fnmatchcase(p, pat) for pat in include)
                   or any(fnmatch.fnmatchcase(p, pat) for pat in exclude)]
    assert missing == [], f"package discovery leaves out {missing}: {packages}"


def test_version_is_readable_without_importing_the_package(pyproject):
    """Breaks if the version stops being statically readable -- e.g. upstream v0.5.2's
    ``dynamic = ["version"]`` + ``attr = "tradingagents.__version__"`` merged while the fork's
    tradingagents/__init__.py has no ``__version__`` literal (or computes it). setuptools then
    imports ``tradingagents`` inside uv's isolated build env, where only setuptools is
    installed: ImportError (dotenv) or AttributeError, the rebuild fails, the venv is left
    empty and every analysis fails."""
    from packaging.version import Version

    project = pyproject.get("project") or {}
    if "version" not in (project.get("dynamic") or []):
        assert isinstance(project.get("version"), str), project.get("version")
        Version(project["version"])
        return
    dynamic = (((pyproject.get("tool") or {}).get("setuptools") or {}).get("dynamic") or {})
    source = dynamic.get("version") or {}
    if "file" in source:
        files = source["file"] if isinstance(source["file"], list) else [source["file"]]
        for name in files:
            Version((REPO_ROOT / name).read_text(encoding="utf-8").strip())
        return
    assert "attr" in source, f"dynamic version without attr/file: {source}"
    module, _, attr = source["attr"].rpartition(".")
    base = REPO_ROOT.joinpath(*module.split("."))
    path = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
    assert path.is_file(), f"version attr {source['attr']!r}: no module file {path}"
    literals = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:  # what setuptools reads
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        if any(isinstance(t, ast.Name) and t.id == attr for t in targets) and isinstance(
                node.value, ast.Constant) and isinstance(node.value.value, str):
            literals.append(node.value.value)
    assert literals, (f"{_rel(path)} has no top-level `{attr} = \"<literal>\"`: setuptools "
                      "would import the package inside the isolated build env")
    Version(literals[-1])


def test_isolated_build_emits_the_metadata_and_files_uv_installs(pyproject, tmp_path):
    """Breaks if the tree no longer builds the way the deployment builds it -- a copy of the
    clone, built by the declared PEP 517 backend in an environment holding nothing but the
    build requirements -- or if the built metadata stops carrying what production installs:
    the ``vertex`` extra with both SDKs, python-dotenv, a Requires-Python admitting 3.11, and
    every module of ``tradingagents`` and ``cli``. Covers the failure modes the static tests
    above name (backend, discovery, dynamic version, extras) as the real build sees them,
    plus build errors they do not model -- metadata setuptools rejects, such as an SPDX
    ``license`` expression next to a legacy license classifier -- any of which leaves the
    production venv empty after the next rebuild. (A missing README is only a setuptools
    warning, not a failure.)"""
    src = tmp_path / "clone-copy"
    shutil.copytree(REPO_ROOT, src, ignore=_copy_ignore, symlinks=True)
    build_env = tmp_path / "build-env"
    build_env.mkdir()
    build = pyproject.get("build-system") or {}
    closure = dh.dependency_closure(list(build.get("requires") or []), marker_env={})
    for name in sorted(closure):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"build requirement {name!r} is not installed in the test interpreter "
                        f"({sys.executable}); install it there to emulate uv's build env")
        for top in sorted({Path(str(f)).parts[0] for f in dist.files or []}):
            if top.endswith(".pth") or top == ".." or (build_env / top).exists():
                continue
            target = Path(dist.locate_file(top))
            if target.exists():
                (build_env / top).symlink_to(target)
    out, home, tmp = tmp_path / "wheel", tmp_path / "home", tmp_path / "tmp"
    for d in (out, home, tmp):
        d.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(tmp), "LANG": "C.UTF-8",
           "PYTHONPATH": str(build_env), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(  # -S: no site-packages -- only the build requirements are importable
        [sys.executable, "-S", "-c", _BUILD_WHEEL, str(out), str(build.get("build-backend"))],
        cwd=str(src), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"isolated build failed:\n{proc.stderr[-4000:]}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("BUILD-RESULT ")][-1]
    wheel = out / json.loads(line[len("BUILD-RESULT "):])["wheel"]
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = email.parser.Parser().parsestr(zf.read(meta_name).decode("utf-8"))

    from packaging.specifiers import SpecifierSet

    requires_python = SpecifierSet(meta.get("Requires-Python") or "")
    assert all(requires_python.contains(v, prereleases=True) for v in PRODUCTION_PYTHONS), \
        meta.get("Requires-Python")
    assert PRODUCTION_EXTRA in {dh.canonical(e) for e in meta.get_all("Provides-Extra") or []}
    requires = [_requirement(r) for r in meta.get_all("Requires-Dist") or []]
    for name, needed_extras in VERTEX_SDKS.items():
        assert any(dh.canonical(r.name) == name and needed_extras <= set(r.extras)
                   and _applies(r, PRODUCTION_EXTRA) for r in requires), \
            (name, [str(r) for r in requires])
    assert any(dh.canonical(r.name) == "python-dotenv" and _applies(r) for r in requires), \
        [str(r) for r in requires]
    sources = {_rel(p) for p in dh.python_sources(REPO_ROOT) if p.name != "main.py"}
    assert sorted(sources - names) == [], "modules missing from the built wheel"


# ============================================================================ Python 3.11

def test_every_fork_source_file_compiles_as_python_3_11():
    """Breaks if any file production imports uses syntax newer than 3.11 -- PEP 695 generics
    or ``type`` aliases, or PEP 701 f-strings such as ``f"{row["name"]}"`` that the 3.12+
    developer venv accepts silently (conflict resolution done there, or upstream code). On
    the Python 3.11 image that is a SyntaxError at import: rc 1 before any LLM call, or --
    for a module main.py imports lazily -- after the paid run. On 3.11 the check is the
    interpreter's own parser; on 3.12+ PEP 701 forms are found with the tokenizer."""
    sources = dh.python_sources(REPO_ROOT)
    assert REPO_ROOT / "main.py" in sources and len(sources) > 50, len(sources)
    offenses = [o for path in sources for o in dh.py311_syntax_offenses(path, _rel(path))]
    assert offenses == [], "\n".join(offenses)


# ============================================================================ importing

def test_importing_what_main_py_imports_writes_nothing_and_needs_no_network(tmp_path):
    """Breaks if importing the fork -- ``tradingagents``, main.py itself and every fork module
    main.py imports, eagerly or after propagate() (the report writer) -- starts to write
    files, create directories, open network connections or spawn processes, or fails from a
    working directory outside the checkout. alpha-pulse imports from a source tree it never
    writes to, with a HOME that does not persist: an import-time ``makedirs`` of
    ~/.tradingagents/... or of a relative path, a cache warm-up, or a vendor reachability
    probe is a crash on an unwritable tree, state silently lost, or a hard dependency on
    egress before the first LLM call.
    Also proves the imports come from THIS checkout (the 3.11 venv holds a built copy)."""
    fork_imports = dh.main_py_fork_imports(REPO_ROOT)
    modules = list(dict.fromkeys(["tradingagents", "main", *(i["module"] for i in fork_imports)]))
    optional = sorted({i["module"] for i in fork_imports if i["tolerated"]})
    assert len(modules) > 2, fork_imports  # main.py still imports the fork
    probe, run_dir = _run_probe(tmp_path, {"modules": modules, "optional": optional},
                                "main-imports")
    _assert_probe_watched(probe)
    assert Path(probe["sys_path0"]) == REPO_ROOT
    failed = [r for r in probe["imports"] if not r["ok"] and not r["optional"]]
    assert failed == [], "\n".join(
        [f"{r['module']}: {r['error']}" for r in failed]
        + [f"--- first failure ({failed[0]['module']}) ---\n{failed[0].get('traceback', '')}"])
    for r in probe["imports"]:
        if r["ok"] and r.get("file"):
            assert Path(r["file"]).resolve().is_relative_to(REPO_ROOT.resolve()), r
    assert [r["module"] for r in probe["imports"]] == modules
    assert probe["writes"] == [], json.dumps(probe["writes"], indent=1)[:4000]
    assert probe["network"] == [] and probe["blocked"] == [], (probe["network"],
                                                               probe["blocked"])
    assert probe["processes"] == [], probe["processes"]
    for sub in ("cwd", "home", "tmp", "fake_app"):
        assert _files_under(run_dir / sub) == [], (sub, _files_under(run_dir / sub))
    for key in ("TRADINGAGENTS_RESULTS_DIR", "TRADINGAGENTS_CACHE_DIR"):
        assert not Path(base_env(run_dir, "005930.KS")[key]).exists(), key


def test_every_fork_module_imports_including_the_kr_vendors_and_ticker_resolver(tmp_path):
    """Breaks if any module of the fork -- every file under tradingagents/ (dataflows and
    agents included) and cli/ -- no longer imports on this interpreter, or if a fork-only
    module (the KR vendors wisereport / naver_news / naver_discussion / opendart_*, kr_utils,
    the ticker resolver) disappears. Upstream renamed dataflows/symbol_utils.py to symbols.py
    while the KR vendors still import ``.symbol_utils``: main.py then dies at import for
    every ticker, while naver_discussion -- imported lazily by the sentiment analyst behind a
    broad except -- degrades to a placeholder without a sound. Failures are named here."""
    probe, _ = _run_probe(tmp_path, {"walk": WALKED_PACKAGES, "named": FORK_ONLY_MODULES},
                          "walk")
    walked = {r["module"] for r in probe["walked"]}
    # Precondition: the probe walked every source file of the two packages. Derived from the
    # tree, not from a list of module names, so a module a merge renames (upstream's
    # dataflows/interface.py is dataflows/router.py) is simply walked under its new name.
    on_disk = {dh.module_name(REPO_ROOT, path) for path in dh.python_sources(REPO_ROOT)
               if path.parent != REPO_ROOT}  # everything but main.py
    assert {"tradingagents", "cli"} <= on_disk and len(on_disk) > 50, sorted(on_disk)[:20]
    assert walked == on_disk, sorted(walked ^ on_disk)[:20]
    failed = [r for r in probe["walked"] if not r["ok"]]
    missing = {label: v["error"] for label, v in probe["named"].items() if not v["ok"]}
    problems = [f"fork-only module {label} does not import: {error}"
                for label, error in missing.items()]
    problems += [f"{r['module']}: {r['error']}" for r in failed]
    if failed:
        problems.append(f"--- first failure ({failed[0]['module']}) ---\n"
                        f"{failed[0].get('traceback', '')}")
    assert problems == [], "\n".join(problems)
    for label, v in probe["named"].items():
        assert Path(v["file"]).resolve().is_relative_to(REPO_ROOT.resolve()), (label, v)
        assert v["module"] in walked, (label, v)  # a fork-only module is a module of the tree
    assert probe["network"] == [] and probe["blocked"] == [], (probe["network"],
                                                               probe["blocked"])


# ============================================================================ dependencies

def test_every_import_the_fork_cannot_do_without_is_installed_by_its_declared_dependencies(
        pyproject):
    """Breaks if fork code needs a third-party package that installing the tree does not
    bring: a new import of an undeclared library (installed only in a developer venv), a
    dependency dropped from pyproject while code still imports it (upstream's pyproject
    removes parsel/redis/tqdm/backtrader/langchain-experimental/setuptools and the whole
    [vertex] extra), or an import-time import of a package only a non-production extra
    provides. Production's venv has exactly what ``uv pip install '.[vertex]'`` resolves, so
    such an import is a ModuleNotFoundError there: at import (rc 1 for every run) or on
    the Vertex LLM path. The allowed set is derived from pyproject plus the installed
    metadata of those requirements (the transitive closure, markers evaluated for the
    production image); imports behind a guard that recovers are optional and skipped."""
    groups = dh.requirement_groups(pyproject)
    runtime = [r for group, reqs in groups.items() if group not in dh.DEV_EXTRAS for r in reqs]
    installed_by_production = dh.dependency_closure(groups["base"] + groups.get(
        PRODUCTION_EXTRA, []))
    installed_by_some_extra = dh.dependency_closure(runtime)
    offenders, checked = [], 0
    for path in dh.python_sources(REPO_ROOT):
        for imp in dh.scan_imports(path, _rel(path)):
            top = imp["module"].split(".")[0]
            if imp["tolerated"] or dh.is_stdlib(imp["module"]) or top in dh.FIRST_PARTY:
                continue
            checked += 1
            # Not installed in this venv: the distribution of the same name, if declared.
            providers = dh.provider_of(imp["module"]) or [dh.canonical(top)]
            allowed = installed_by_production if imp["at_import_time"] else \
                installed_by_some_extra
            if not any(p in allowed for p in providers):
                when = "import-time" if imp["at_import_time"] else "lazy"
                scope = (f"`pip install .[{PRODUCTION_EXTRA}]`" if imp["at_import_time"]
                         else "any runtime dependency group")
                offenders.append(f"{imp['where']}: {when} import of {imp['module']!r} (provided "
                                 f"by {providers}) is not installed by {scope}")
    assert checked > 30, checked  # the scan really saw the fork's third-party imports
    assert offenders == [], "\n".join(offenders)


# ============================================================================ a real run

@pytest.mark.parametrize("name", ["s1_nightly_kr_holding_sell", "s4_us_not_held_discovery"])
def test_a_production_run_keeps_everything_it_writes_out_of_home(ap_run, name):
    """Breaks if a nightly KR run or a US discovery run writes anything under HOME -- a cache,
    a log, a checkpoint DB or a memory file whose path no longer follows
    TRADINGAGENTS_{RESULTS_DIR,CACHE_DIR,MEMORY_LOG_PATH} (a renamed variable, a new default
    under ~/.tradingagents). alpha-pulse's HOME does not persist and it reads the three
    configured paths, not HOME (without the three variables, output lands in HOME and a
    report is gone after a restart)."""
    res = ap_run(name).assert_ok()
    assert _files_under(res.home) == [], _files_under(res.home)
    assert res.report_path and Path(res.report_path).resolve().is_relative_to(
        res.results_dir.resolve()), res.report_path
    assert res.memory_log_path is not None and res.memory_log_path.is_file()
    assert _files_under(res.cache_dir), "nothing was cached under TRADINGAGENTS_CACHE_DIR"
