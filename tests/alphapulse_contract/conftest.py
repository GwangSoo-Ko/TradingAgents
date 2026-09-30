"""Fixtures for the alpha-pulse contract tests.

``ap_run`` runs a scenario through the real main.py once per pytest session and
caches the RunResult, so several test modules can assert on the same run (a full
graph run costs seconds). Cached results are shared: treat them as read-only and
never write into their run directories. Use ``run_main(tmp_path, ...)`` directly
for a private run.

The whole package is skipped on interpreters older than production's CPython 3.11
(see ``pytest_collection_modifyitems``): the fork's CI matrix still runs 3.10, where
the contract does not hold by construction.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from . import scenarios
from .harness import RunResult, run_main

# alpha-pulse runs the fork on a CPython 3.11 image.
PRODUCTION_PYTHON = (3, 11)
_PACKAGE_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip this package (and only this package) below CPython 3.11.

    The lock describes the interpreter production runs. On 3.10 it cannot hold: main.py
    accepts the nightly batch's KST ``YYYYMMDD`` date only because Python 3.11's
    ``date.fromisoformat`` reads the basic ISO format (3.10 rejects it: rc 2 at argparse).
    A red 3.10 lane would report the interpreter, not a merge. conftest hooks see every
    collected item, so the marker is applied by path.
    """
    if sys.version_info >= PRODUCTION_PYTHON:
        return
    found = ".".join(map(str, sys.version_info[:3]))
    marker = pytest.mark.skip(reason=(
        f"alpha-pulse contract lock: production runs CPython 3.11+, this is {found} "
        "(e.g. main.py reads the nightly YYYYMMDD date via 3.11's date.fromisoformat)"))
    for item in items:
        if Path(str(item.path)).resolve().is_relative_to(_PACKAGE_DIR):
            item.add_marker(marker)


@pytest.fixture(scope="session")
def ap_run(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., RunResult]:
    """ap_run(scenario_or_name, argv=None, env_overrides=None) -> cached RunResult."""
    cache: dict[str, RunResult] = {}

    def run(scenario: dict[str, Any] | str, argv: list[str] | None = None,
            env_overrides: dict[str, str | None] | None = None) -> RunResult:
        spec = scenarios.get(scenario) if isinstance(scenario, str) else scenario
        key = json.dumps([spec, argv, env_overrides], sort_keys=True, ensure_ascii=False,
                         default=str)
        if key not in cache:
            base = tmp_path_factory.mktemp("ap_runs", numbered=True)
            cache[key] = run_main(base, spec, argv=argv, env_overrides=env_overrides)
        return cache[key]

    return run
