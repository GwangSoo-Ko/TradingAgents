"""Locate internals that an upstream merge may move — loudly, never by skipping.

The contract tests freeze what alpha-pulse consumes, not where the fork keeps
it. When a test needs an internal module (a schema, a render helper, a data
vendor), it names every known home of that module in order of preference:

    schemas = import_first(
        "tradingagents.agents.schemas",            # a2981a7 (fork today)
        "tradingagents.agents.structured.schemas",  # hypothetical future home
    )

If none of the candidates resolves, the test FAILS with an AssertionError that
lists every candidate and the import error each one raised. A contract test
that silently skips after a merge is a contract test that no longer exists,
so there is deliberately no skip path here.

:func:`get_class_by_interface` does the same for a class a merge may also RENAME
(the memory log): it finds the class by the methods it implements, not by its name.

Only an ImportError of the candidate module itself (or one of its parent
packages) moves on to the next candidate. An ImportError raised *inside* a
module that does exist (a broken import in its body) is re-raised as a
failure, because that is exactly the kind of breakage a merge introduces.

:class:`ExpectedDrift` is the one exception type of the behaviours the merge is
EXPECTED to change (strict xfails); ``grep -rn ExpectedDrift`` lists them all.

:func:`project_to_expected` / :func:`canonical_json` compare a JSON payload the way
alpha-pulse reads it: by the keys it knows. A merge that adds an unrelated optional
field changes nothing alpha-pulse consumes, so it must not fail a contract test.
"""

from __future__ import annotations

import importlib
import json
from types import ModuleType
from typing import Any

# What project_to_expected() puts where the payload lacks a key the expectation names.
ABSENT = "<absent from the line>"


def _is_missing_module(exc: ImportError, dotted: str) -> bool:
    """True when ``exc`` means ``dotted`` (or a parent package) does not exist.

    ``ModuleNotFoundError.name`` names the module that could not be found. If it
    is the candidate itself or one of its parents, the candidate is absent. If it
    is some *other* module, the candidate exists but its body failed to import —
    that is a real breakage and must not be hidden behind the next candidate.
    """
    missing = getattr(exc, "name", None)
    if not isinstance(exc, ModuleNotFoundError) or not missing:
        return False
    parts = dotted.split(".")
    parents = {".".join(parts[: i + 1]) for i in range(len(parts))}
    return missing in parents


def import_first(*module_paths: str) -> ModuleType:
    """Import and return the first module in ``module_paths`` that exists.

    Raises AssertionError naming every candidate when none exists. A candidate
    that exists but fails while importing its own body is reported immediately
    (AssertionError chained to the original error) instead of being skipped.
    """
    if not module_paths:
        raise AssertionError("import_first() needs at least one candidate module path")
    tried: list[str] = []
    for dotted in module_paths:
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:
            if _is_missing_module(exc, dotted):
                tried.append(f"  - {dotted}: {type(exc).__name__}: {exc}")
                continue
            raise AssertionError(
                f"module {dotted!r} exists but failed to import: {type(exc).__name__}: {exc}"
            ) from exc
    raise AssertionError(
        "none of the candidate modules could be imported (the code moved again? "
        "add its new home to the candidate list):\n" + "\n".join(tried)
    )


def get_symbol(name: str, *module_paths: str) -> Any:
    """Return attribute ``name`` from the first candidate module that defines it.

    Candidates that are missing entirely, or that exist but do not define
    ``name``, are skipped; if no candidate defines it an AssertionError lists
    what was tried. A candidate whose body fails to import is reported as a
    failure (see :func:`import_first`).
    """
    if not module_paths:
        raise AssertionError(f"get_symbol({name!r}) needs at least one candidate module path")
    tried: list[str] = []
    for dotted in module_paths:
        try:
            module = importlib.import_module(dotted)
        except ImportError as exc:
            if _is_missing_module(exc, dotted):
                tried.append(f"  - {dotted}: missing ({exc})")
                continue
            raise AssertionError(
                f"module {dotted!r} exists but failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        if hasattr(module, name):
            return getattr(module, name)
        tried.append(f"  - {dotted}: imported, but has no attribute {name!r}")
    raise AssertionError(
        f"symbol {name!r} not found in any candidate module:\n" + "\n".join(tried)
    )


def get_class_by_interface(methods: tuple[str, ...], *module_paths: str) -> type:
    """Return the one class implementing ``methods`` that the first candidate module binds.

    Found by what it does, not by its name (a merge may rename it): each candidate module's
    namespace is scanned for classes on which every name in ``methods`` is callable. The
    first module binding exactly one such class wins; a module binding several fails (the
    pick would be a guess), and so does finding none in any candidate. Missing candidates are
    skipped; a candidate whose body fails to import is a failure (see :func:`import_first`).
    """
    if not methods or not module_paths:
        raise AssertionError("get_class_by_interface() needs methods and candidate modules")
    tried: list[str] = []
    for dotted in module_paths:
        try:
            module = importlib.import_module(dotted)
        except ImportError as exc:
            if _is_missing_module(exc, dotted):
                tried.append(f"  - {dotted}: missing ({exc})")
                continue
            raise AssertionError(
                f"module {dotted!r} exists but failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        found = sorted({obj for obj in vars(module).values()
                        if isinstance(obj, type)
                        and all(callable(getattr(obj, name, None)) for name in methods)},
                       key=lambda cls: (cls.__module__, cls.__qualname__))
        if len(found) == 1:
            return found[0]
        if found:
            raise AssertionError(
                f"{dotted} binds several classes with {list(methods)}: "
                + ", ".join(f"{c.__module__}.{c.__qualname__}" for c in found))
        tried.append(f"  - {dotted}: imported, but binds no class with {list(methods)}")
    raise AssertionError(
        f"no candidate module binds a class with {list(methods)}:\n" + "\n".join(tried))


def project_to_expected(actual: Any, expected: Any) -> Any:
    """``actual`` cut down to the keys ``expected`` names, recursively.

    Keys alpha-pulse does not read are dropped; a key it reads that ``actual`` lacks
    shows up as :data:`ABSENT` (so a rename or a drop still fails). Lists are projected
    item by item only when both have the same length; otherwise ``actual`` is kept whole
    so the comparison fails and shows it. Compare with :func:`canonical_json`.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return actual
        return {key: project_to_expected(actual[key], value) if key in actual else ABSENT
                for key, value in expected.items()}
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        return [project_to_expected(a, e) for a, e in zip(actual, expected, strict=True)]
    return actual


def canonical_json(value: Any) -> str:
    """JSON text that tells 1 from 1.0 and null from a missing key (dict ``==`` would not)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=1)


class ExpectedDrift(AssertionError):
    """a2981a7 behaviour that the upstream v0.5.x merge is expected to change.

    A test pinning such a change asserts the NEW behaviour, raises this when it sees
    today's behaviour instead, and is marked ``xfail(strict=True, raises=ExpectedDrift)``:
    XFAIL on a2981a7, a strict XPASS failure (delete the marker in the merge commit) once
    the merge lands, and any other failure -- a broken precondition, half a change -- is
    a real failure.
    """
