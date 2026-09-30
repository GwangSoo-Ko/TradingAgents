"""Harness subprocess entry point: fake the boundaries, then run main.py as __main__.

    <python> tests/alphapulse_contract/_bootstrap.py <run_dir>

``<run_dir>/spec.json`` is the scenario (see scenarios.py); the harness
(harness.run_main) has already written the per-ticker memory-log seed and the
optional scenario ``.env`` and set the environment. This process then behaves
like alpha-pulse's ``<venv>/bin/python <clone>/main.py TICKER [DATE]``:
``sys.path[0]`` is the repo root, the cwd is the repo root, and ``main.py`` runs
unmodified via ``runpy.run_path(..., run_name="__main__")``.

Before main.py runs, the boundary fakes are installed (``_fake_data`` for data,
network, clock and .env; ``_fake_llm`` for the model) and
``TradingAgentsGraph.propagate`` — the documented public entry point — is wrapped
by a pass-through spy that records its arguments and the returned state.

The harness never writes to stdout: stdout is the contract under test. Every
observation goes to ``<run_dir>/captures.json`` (written even when main.py
fails; the exit code and traceback are those main.py itself produces).
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_files() -> dict[str, str | None]:
    names = ("tradingagents", "cli", "cli.main", "tradingagents.graph.trading_graph",
             "tradingagents.agents.schemas", "tradingagents.reporting",
             "tests.alphapulse_contract")
    out = {}
    for name in names:
        mod = sys.modules.get(name)
        out[name] = getattr(mod, "__file__", None) if mod is not None else None
    return out


def _memory_log_classes(graph) -> list[dict]:
    """Classes of the objects the graph holds that implement the memory-log interface.

    Read from the live instance, so a test can compare the class production actually
    instantiates with the class it exercises -- a lookup by a familiar name would happily
    return an orphaned copy left behind by a merge."""
    out = []
    for attr, value in sorted(vars(graph).items()):
        if callable(getattr(value, "get_past_context", None)) and callable(
                getattr(value, "store_decision", None)):
            cls = type(value)
            module = sys.modules.get(cls.__module__)
            out.append({"attr": attr, "module": cls.__module__, "qualname": cls.__qualname__,
                        "file": getattr(module, "__file__", None)})
    return out


def _install_propagate_spy(rt, graph_cls) -> None:
    original = graph_cls.propagate
    signature = inspect.signature(original)
    params = [p for p in signature.parameters if p != "self"]

    @functools.wraps(original)
    def propagate(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
        except TypeError:
            arguments = {"args": list(args), "kwargs": dict(kwargs)}
        ticker = arguments.get(params[0]) if params else None
        trade_date = arguments.get(params[1]) if len(params) > 1 else None
        rt.set_trade_date(trade_date)
        entry = {"arguments": rt.json_safe(arguments), "ticker": ticker,
                 "trade_date": None if trade_date is None else str(trade_date),
                 "trade_date_type": type(trade_date).__name__,
                 "memory_log_classes": _memory_log_classes(self)}
        with rt._lock:
            rt.CAPTURES["propagate"].append(entry)
            rt.CAPTURES["env_at_propagate"] = dict(os.environ)
            rt.CAPTURES["graph_config"] = rt.json_safe(getattr(self, "config", None))
            rt.CAPTURES["graph_debug"] = getattr(self, "debug", None)
        rt.flush()
        result = original(self, *args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            final_state, signal = result
            entry["signal"] = rt.json_safe(signal)
            entry["signal_type"] = type(signal).__name__
            if isinstance(final_state, dict):
                excerpt = {k: rt.json_safe(v) for k, v in final_state.items() if k != "messages"}
                excerpt["messages_count"] = len(final_state.get("messages") or [])
                entry["final_state"] = excerpt
        rt.flush()
        return result

    graph_cls.propagate = propagate


def _self_test_network(rt) -> None:
    """Deliberately attempt real network I/O; every attempt must be refused and recorded.

    Enabled by the scenario key ``self_test_network``. Targets TEST-NET-1
    (192.0.2.1, never routable) so even a broken guard cannot reach a real host.
    """
    import socket

    def raw_socket():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(1)
            s.connect(("192.0.2.1", 80))
        finally:
            s.close()

    def create_connection():
        socket.create_connection(("192.0.2.1", 80), timeout=1)

    def curl_session():
        from curl_cffi import requests as cr
        cr.get("https://192.0.2.1/", timeout=1)

    def curl_low_level():
        from curl_cffi import Curl, CurlOpt
        c = Curl()
        c.setopt(CurlOpt.URL, b"https://192.0.2.1/")
        c.perform()

    results = []
    for name, probe in (("raw_socket", raw_socket), ("create_connection", create_connection),
                        ("curl_session", curl_session), ("curl_low_level", curl_low_level)):
        try:
            probe()
            results.append({"probe": name, "error_type": None, "error": None})
        except Exception as exc:  # noqa: BLE001 — the point is to observe the refusal
            results.append({"probe": name, "error_type": type(exc).__name__, "error": str(exc)})
    rt.CAPTURES["self_test_network"] = results


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))

    # Exactly what `python <clone>/main.py` gets as sys.path[0].
    sys.path[0] = str(REPO_ROOT)

    from tests.alphapulse_contract import _fake_data, _fake_llm, _runtime as rt
    from tests.alphapulse_contract._compat import get_symbol

    rt.init(spec, run_dir)
    rt.CAPTURES["python"] = {"version": sys.version, "executable": sys.executable,
                             "version_info": list(sys.version_info[:3])}
    rt.CAPTURES["cwd"] = os.getcwd()
    rt.CAPTURES["sys_path0"] = sys.path[0]
    rt.CAPTURES["env_at_start"] = dict(os.environ)
    rt.CAPTURES["llm_boundary"] = spec.get("llm_boundary", "factory")

    _fake_data.install_all()
    _fake_llm.install_sdk_fakes()
    if spec.get("self_test_network"):
        _self_test_network(rt)

    main_py = REPO_ROOT / "main.py"
    try:
        # Pre-import only to install the propagate spy and the factory fake. If the
        # fork cannot even be imported, record it and let main.py hit the same error
        # itself, so rc, traceback and stdout are exactly what production would see.
        modules_before = set(sys.modules)
        try:
            graph_cls = get_symbol("TradingAgentsGraph", "tradingagents.graph.trading_graph",
                                   "tradingagents.graph")
        except BaseException as exc:  # noqa: BLE001
            graph_cls = None
            rt.note(f"pre-import of TradingAgentsGraph failed ({type(exc).__name__}: {exc}); "
                    "main.py runs unwrapped and should fail on its own import")
            # A failed import leaves half-initialised fork packages behind; drop them so
            # main.py imports from a clean slate, like a fresh production process.
            for name in list(sys.modules):
                if name not in modules_before and (
                        name in ("tradingagents", "cli")
                        or name.startswith(("tradingagents.", "cli."))):
                    del sys.modules[name]
        if graph_cls is not None:
            if spec.get("llm_boundary", "factory") == "factory":
                try:
                    rt.CAPTURES["factory_patched_in"] = _fake_llm.install_factory_fake()
                except Exception as exc:  # noqa: BLE001
                    rt.note(f"factory fake not installed ({type(exc).__name__}: {exc}); the "
                            "fake Vertex SDK class still stands in for the model")
            _install_propagate_spy(rt, graph_cls)
        rt.CAPTURES["imports"].update(_module_files())
        rt.flush()

        sys.argv = [str(main_py), *spec["argv"]]
        rt.CAPTURES["argv"] = list(sys.argv)
        runpy.run_path(str(main_py), run_name="__main__")
    except SystemExit as exc:
        rt.CAPTURES["exit"] = {"kind": "SystemExit", "code": rt.json_safe(exc.code)}
        raise
    except BaseException as exc:
        rt.CAPTURES["exit"] = {"kind": "exception", "type": type(exc).__name__,
                               "message": str(exc), "traceback": traceback.format_exc()}
        raise
    else:
        rt.CAPTURES["exit"] = {"kind": "returned"}
    finally:
        rt.CAPTURES["imports"].update(_module_files())
        rt.flush()


if __name__ == "__main__":
    main()
