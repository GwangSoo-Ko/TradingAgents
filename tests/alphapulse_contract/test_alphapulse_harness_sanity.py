"""Sanity of the alpha-pulse contract harness itself.

Every other test in this package trusts the harness to (1) run THIS checkout's
main.py the way alpha-pulse does, (2) never touch the network, (3) leave the
repository and the developer's home untouched, and (4) read stdout with a strict
reader of the output grammar the fork documents (``contract_reader``). These tests
prove each of those; if one fails, the contract tests' verdicts mean nothing until
it is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import contract_reader, scenarios
from .harness import (
    DECISION_WORDS,
    PACKAGE_DIR,
    REPO_ROOT,
    REPORT_SAVED_LINE_RE,
    run_main,
    tree_snapshot,
)

S1 = "s1_nightly_kr_holding_sell"

_PLAN_TEXT = '{"rating": "Sell", "stop_loss": 10380.0, "kill_switch": {"price": null, "condition": "x"}}'
_PLAN = {"rating": "Sell", "stop_loss": 10380.0, "kill_switch": {"price": None, "condition": "x"}}
# Prose with the characters str.splitlines() treats as line breaks although print() does not:
# pydantic's model_dump_json writes U+2028 / U+2029 / U+0085 inside a JSON string raw.
_SEPARATORS = "a\u2028b\u2029c\x85d"
_PLAN_WITH_SEPARATORS_TEXT = ('{"rating": "Sell", "kill_switch": {"price": null, "condition": "'
                              + _SEPARATORS + '"}}')
_PLAN_WITH_SEPARATORS = {"rating": "Sell", "kill_switch": {"price": None, "condition": _SEPARATORS}}
# Hand-written stdout -> (decision, report path, plan) by the documented grammar.
READER_CASES: dict[str, tuple[str, str | None, str | None, dict | None]] = {
    "closing_lines_after_a_trace": (
        "== trace ==\nHold is not a line\nSell\nTRADE_PLAN_JSON: " + _PLAN_TEXT
        + "\nReport saved: /r/reports/X_20260819_010203/complete_report.md\n",
        "Sell", "/r/reports/X_20260819_010203/complete_report.md", _PLAN),
    "decoys_before_the_last_report_line": (
        "Hold\nReport saved: /tmp/decoy/complete_report.md\nanalysis\nOverweight\n"
        "Report saved: /r/B/complete_report.md\n",
        "Overweight", "/r/B/complete_report.md", None),
    "review_is_a_decision_word": (
        "Buy\nREVIEW\nReport saved: /r/C/complete_report.md\n",
        "REVIEW", "/r/C/complete_report.md", None),
    "whitespace_case_and_crlf": (
        "  underweight \r\nReport saved:   /r/D/complete_report.md  \r\n",
        "underweight", "/r/D/complete_report.md", None),
    "a_rating_inside_prose_is_not_a_decision": (
        "I would Hold here\nReport saved: /r/E/complete_report.md\n",
        None, "/r/E/complete_report.md", None),
    "a_decision_after_the_last_report_line_is_ignored": (
        "Buy\nReport saved: /r/F/complete_report.md\nSell\n",
        "Buy", "/r/F/complete_report.md", None),
    "no_report_line": ("Sell\nsome trace\n", "Sell", None, None),
    "the_last_plan_line_wins": (
        'TRADE_PLAN_JSON: {"rating": "Buy"}\nTRADE_PLAN_JSON: {"rating": "Sell"}\n',
        None, None, {"rating": "Sell"}),
    "an_unreadable_last_plan_line_is_no_plan": (
        'TRADE_PLAN_JSON: {"rating": "Buy"}\nTRADE_PLAN_JSON: {"rating": Sell}\n',
        None, None, None),
    "nan_is_not_json": ('TRADE_PLAN_JSON: {"stop_loss": NaN}\n', None, None, None),
    "infinity_is_not_json": ('TRADE_PLAN_JSON: {"stop_loss": -Infinity}\n', None, None, None),
    "a_plan_spread_over_lines_is_no_plan": (
        'TRADE_PLAN_JSON: {"rating":\n "Sell"}\n', None, None, None),
    "an_array_is_no_plan": ('TRADE_PLAN_JSON: [{"rating": "Sell"}]\n', None, None, None),
    "the_prefix_must_start_the_line": (' TRADE_PLAN_JSON: {"rating": "Sell"}\n', None, None, None),
    # Only "\n" ends a line: U+2028/U+2029/U+0085 inside the plan's JSON are part of it...
    "unicode_separators_inside_the_plan_line": (
        "Sell\nTRADE_PLAN_JSON: " + _PLAN_WITH_SEPARATORS_TEXT
        + "\nReport saved: /r/G/complete_report.md\n",
        "Sell", "/r/G/complete_report.md", _PLAN_WITH_SEPARATORS),
    # ...and a lone "\r" does not start a new line either.
    "a_report_line_must_start_after_a_newline": (
        "Sell\nprogress 100%\rReport saved: /r/H/complete_report.md\n", "Sell", None, None),
    # 'Report saved:' with no path names no report: the last line that names one wins.
    "a_report_line_without_a_path_is_skipped": (
        "Sell\nReport saved: /r/I/complete_report.md\nReport saved:\n",
        "Sell", "/r/I/complete_report.md", None),
    "a_report_line_without_a_path_names_no_report": ("Hold\nReport saved:\n", "Hold", None, None),
    # Only 'TRADE_PLAN_JSON: {...}' is a plan line: nothing after the prefix, an array or text
    # after the object makes a line of another form, so the last plan LINE is the first one.
    "only_an_object_makes_a_plan_line": (
        'TRADE_PLAN_JSON: {"rating": "Sell"}\nTRADE_PLAN_JSON: [1, 2]\n'
        'TRADE_PLAN_JSON: {"rating": "Hold"} and more\nTRADE_PLAN_JSON:\n',
        None, None, {"rating": "Sell"}),
}

# stdout -> the lines print() wrote (contract_reader.lines): split at "\n" only, a "\r" before
# it dropped, no empty last line after a final newline.
LINE_CASES: dict[str, tuple[str, list[str]]] = {
    "empty": ("", []),
    "no_final_newline": ("Sell", ["Sell"]),
    "final_newline": ("Sell\n", ["Sell"]),
    "crlf": ("Sell\r\nReport saved: /r\r\n", ["Sell", "Report saved: /r"]),
    "blank_line_kept": ("a\n\nb\n", ["a", "", "b"]),
    "only_newline_ends_a_line": ("a\u2028b\u2029c\x85d\x0be\x0cf\x1cg\rh\n",
                                 ["a\u2028b\u2029c\x85d\x0be\x0cf\x1cg\rh"]),
}


def _under(path: str | None, root: Path) -> bool:
    return path is not None and Path(path).resolve().is_relative_to(root.resolve())


def test_harness_package_derives_the_repo_root_from_its_own_location():
    """Breaks if: the harness hardcodes a checkout path (e.g. the worktree) instead of
    deriving it, so a merged tree or a scratch copy would run someone else's main.py."""
    assert Path(__file__).resolve().parents[2] == REPO_ROOT
    assert (REPO_ROOT / "main.py").is_file()
    assert (REPO_ROOT / "tradingagents" / "__init__.py").is_file()
    assert Path(__file__).resolve().parent == PACKAGE_DIR


def test_subprocess_imports_tradingagents_cli_and_main_from_this_checkout(ap_run):
    """Breaks if: the subprocess resolves ``tradingagents``/``cli`` from an editable install
    of another checkout (the developer .venv points at the MAIN checkout) or from
    site-packages (the Python 3.11 venv has a built copy), or runs a different main.py —
    every contract test would then silently test the wrong code."""
    res = ap_run(S1)
    caps = res.captures
    imports = caps["imports"]
    for module in ("tradingagents", "cli", "cli.main", "tradingagents.graph.trading_graph"):
        assert _under(imports.get(module), REPO_ROOT), (module, imports.get(module), REPO_ROOT)
    assert Path(caps["sys_path0"]).resolve() == REPO_ROOT.resolve()
    assert Path(caps["cwd"]).resolve() == REPO_ROOT.resolve()
    assert Path(caps["argv"][0]).resolve() == (REPO_ROOT / "main.py").resolve()
    assert caps["argv"][1:] == ["417310.KS", "20260819"]
    # main.py builds its own graph with debug=True (the streaming trace path alpha-pulse
    # parses around), not a stand-in graph.
    assert caps["graph_debug"] is True
    assert res.propagate_calls, "TradingAgentsGraph.propagate was never called"


def test_no_network_was_attempted_and_every_http_request_was_served_by_fixtures(ap_run):
    """Breaks if: a code path reaches the network (a new vendor endpoint, a yfinance API the
    fake does not model, an SDK bypassing the LLM fake) — the run would then depend on the
    outside world and could hit production services from a test."""
    res = ap_run(S1)
    assert res.network_attempts == [], res.network_attempts
    assert res.unrouted_http == [], res.unrouted_http
    assert res.captures["notes"] == [], res.captures["notes"]
    # The developer's .env above the checkout is fenced off; nothing was loaded from it.
    for lookup in res.captures["dotenv"]:
        assert lookup["served"] == "", lookup
    # The data layer really ran through the library fakes (not a stubbed vendor function).
    libs = {(d["lib"], d.get("node")) for d in res.data_calls}
    assert ("yfinance", "tools_market") in libs
    assert ("http", "tools_fundamentals") in libs


def test_network_guards_refuse_and_record_every_attempt(tmp_path):
    """Breaks if: the socket or curl_cffi guard stops working (curl_cffi talks to libcurl
    directly, so Python socket patches alone would not stop yfinance's transport)."""
    spec = scenarios.get(S1)
    spec["self_test_network"] = True
    # An invalid date makes main.py exit at argparse, so this check stays fast.
    res = run_main(tmp_path, spec, argv=["417310.KS", "2026-13-45"])
    assert res.rc == 2, res.describe()
    probes = {p["probe"]: p for p in res.captures["self_test_network"]}
    assert set(probes) == {"raw_socket", "create_connection", "curl_session", "curl_low_level"}
    for name, probe in probes.items():
        assert probe["error_type"] == "HarnessNetworkBlocked", (name, probe)
    blocked_apis = {b["api"] for b in res.network_attempts}
    assert {"socket.connect", "socket.create_connection",
            "curl_cffi.requests.Session.request", "curl_cffi.Curl.perform"} <= blocked_apis


def test_s1_exits_zero_with_the_three_line_stdout_contract(ap_run):
    """Breaks if: main.py stops ending stdout with decision / TRADE_PLAN_JSON / 'Report saved:',
    prints after the report line, or fails after the paid run — alpha-pulse takes the LAST
    'Report saved:' line and the decision line right before it, and discards any rc != 0 run."""
    res = ap_run(S1)
    res.assert_ok()
    assert "Traceback" not in res.stderr, res.stderr[-2000:]
    decision_line, plan_line, report_line = res.tail
    assert decision_line.strip().lower() in DECISION_WORDS, res.tail
    assert plan_line.startswith("TRADE_PLAN_JSON: "), plan_line[:80]
    assert len(res.trade_plan_lines) == 1
    json.loads(plan_line[len("TRADE_PLAN_JSON: "):])
    m = REPORT_SAVED_LINE_RE.match(report_line)
    assert m, report_line
    assert Path(m.group("path")).parent.parent == res.results_dir / "reports"
    assert m.group("dir") == "417310.KS"
    assert Path(m.group("path")).is_file()
    # The documented last-line grammar reads exactly these lines, ignoring the decoy 'Hold' /
    # 'Report saved:' lines the analyst printed earlier in the debug trace.
    assert res.stdout.count("Report saved: /tmp/decoy/complete_report.md") == 1
    assert contract_reader.read_stdout(res.stdout) == (decision_line.strip(), m.group("path"))
    assert contract_reader.read_trade_plan(res.stdout) == \
        json.loads(plan_line[len("TRADE_PLAN_JSON: "):])


@pytest.mark.parametrize("case", sorted(READER_CASES))
def test_stdout_reader_follows_the_documented_last_line_grammar(case):
    """Breaks if: ``contract_reader`` drifts from the grammar the fork documents for main.py's
    output (docs/INTEGRATION.md §1b, main.py's comments) -- it takes the first 'Report saved:'
    line or a decision printed after the last one, takes a 'Report saved:' line that names no
    path, matches a rating word inside prose, stops counting REVIEW as a decision word, accepts
    NaN/Infinity, an array or a multi-line object as a plan, counts a 'TRADE_PLAN_JSON:' line of
    another form as the last plan line, falls back to an earlier plan line when the last one is
    unreadable, or cuts lines anywhere but at "\\n" (``str.splitlines`` splits a plan whose
    prose holds U+2028, which main.py prints on one line, and finds a report line after a lone
    "\\r"). Every other test reads main.py's stdout through it; a lenient reader would pass
    output a strict consumer rejects, a stricter one would miss a plan it reads. Expected
    values are hand-derived from the grammar."""
    stdout, decision, report_path, plan = READER_CASES[case]
    assert contract_reader.read_stdout(stdout) == (decision, report_path)
    assert contract_reader.read_trade_plan(stdout) == plan
    decision_words = contract_reader.DECISION_WORDS
    tradeable_words = contract_reader.TRADEABLE_WORDS
    assert decision_words == {"buy", "overweight", "hold", "underweight", "sell", "review"}
    assert tradeable_words == decision_words - {"review"}


@pytest.mark.parametrize("case", sorted(LINE_CASES))
def test_a_stdout_line_is_what_print_ended(case):
    """Breaks if: the harness's line cutting (``RunResult.stdout_lines`` / ``tail`` /
    ``trade_plan_lines``, all ``contract_reader.lines``) drifts from print()'s lines -- splitting
    at U+2028, a form feed or a lone "\\r" (``str.splitlines``), keeping the "\\r" of a CRLF, or
    reporting an empty last line after the final newline: the three closing lines would then be
    misread as the tail, and a plan line counted as two."""
    stdout, expected = LINE_CASES[case]
    assert contract_reader.lines(stdout) == expected


@pytest.mark.parametrize(("decision", "review"), [
    ("REVIEW", True), (" review ", True), ("Review", True),
    ("Hold", False), ("Sell", False), ("", False), (None, False),
])
def test_review_sentinel_is_recognised_in_any_case(decision, review):
    """Breaks if: the reader's REVIEW check stops matching the fork's sentinel (``REVIEW``,
    tradingagents/agents/utils/rating.py) as the decision line prints it, or starts treating a
    tradeable word as "look at it" -- the REVIEW contract test reads the decision through it."""
    assert contract_reader.is_review(decision) is review


def test_a_run_writes_only_under_its_tmp_directory(tmp_path):
    """Breaks if: a run writes into the checkout (reports, caches, logs, pyc — a deployed
    source tree need not be writable) or into the developer's real ~/.tradingagents."""
    home_ta = Path.home() / ".tradingagents"
    repo_before = tree_snapshot(REPO_ROOT)
    home_before = tree_snapshot(home_ta)
    res = run_main(tmp_path, scenarios.get(S1))
    res.assert_ok()
    repo_after = tree_snapshot(REPO_ROOT)
    changed = sorted(set(repo_before.items()) ^ set(repo_after.items()))
    assert changed == [], changed[:20]
    assert tree_snapshot(home_ta) == home_before
    for path in (res.report_path, res.memory_log_path, res.env["HOME"], res.env["TMPDIR"],
                 res.env["TRADINGAGENTS_CACHE_DIR"]):
        assert _under(str(path), tmp_path), path
    produced = [p for p in res.run_dir.rglob("*") if p.is_file()]
    assert any(p.name == "complete_report.md" for p in produced)
    assert any(p.suffix == ".csv" for p in (res.run_dir / "cache").rglob("*"))


@pytest.mark.parametrize("name", scenarios.names())
def test_every_ready_scenario_runs_green_and_offline(ap_run, name):
    """Breaks if: a ready scenario no longer completes on this tree (a merge broke a path the
    scenario exercises) or needs the network — later contract tests build on all of them."""
    res = ap_run(name)
    res.assert_ok()
    assert res.network_attempts == [] and res.unrouted_http == [], name
    assert REPORT_SAVED_LINE_RE.match(res.stdout_lines[-1]), res.stdout_lines[-1]
    assert (res.decision or "").lower() in DECISION_WORDS, res.decision
    assert res.complete_report and res.complete_report.startswith("# ")
    roles = res.roles_called()
    assert {"market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
            "bull_researcher", "bear_researcher", "research_manager", "trader",
            "aggressive_debator", "conservative_debator", "neutral_debator",
            "portfolio_manager"} <= roles, roles


def test_sdk_boundary_mode_builds_every_llm_through_the_fake_vertex_class(ap_run):
    """Breaks if: the SDK-boundary mode stops intercepting construction (the fork would then
    build a real ChatAnthropicVertex) — the LLM-kwargs contract tests depend on it."""
    spec = scenarios.derive(scenarios.get(S1), {"llm_boundary": "sdk"}, name=S1 + "_sdk")
    res = ap_run(spec)
    res.assert_ok()
    records = res.llm_constructions
    assert records and {r["boundary"] for r in records} == {"sdk"}
    assert {r["sdk_class"] for r in records} == {"ChatAnthropicVertex"}
    for role in res.roles_called():
        assert res.llm_for(role)["boundary"] == "sdk", role


# The one-line account JSON's keys in the order alpha-pulse writes them: the documented
# snapshot (docs/INTEGRATION.md §4) plus the two keys build_position_block reads.
POSITION_CONTEXT_KEYS = ["held_qty", "avg_price", "current_price", "unrealized_pnl_pct",
                         "current_weight_pct", "cash", "total_nav", "currency",
                         "founding_thesis", "founding_thesis_absent_reason"]


def test_position_context_fixtures_are_one_line_json_in_the_shape_alpha_pulse_sends():
    """Breaks if: a committed TRADINGAGENTS_POSITION_CONTEXT sample stops being one line of JSON
    with the keys, key order and founding context production sends (a hand edit that drops the
    absent reason, pretty-prints the value or detaches it from its case's founding context) --
    the account tests would then feed the fork a shape production never sends. The explicit
    empty value must stay the empty string (production SETS it, it does not unset it)."""
    data = scenarios.fixture_json("position_contexts.json")
    for key in ("holding_founding_buy", "not_held"):
        entry = data[key]
        value = entry["env_value"]
        assert "\n" not in value and value == scenarios.position_context(key), key
        parsed = json.loads(value)
        assert list(parsed) == POSITION_CONTEXT_KEYS, (key, list(parsed))
        assert (parsed["founding_thesis"], parsed["founding_thesis_absent_reason"]) == (
            entry["founding"]["thesis"], entry["founding"]["absent_reason"]), key
    assert data["explicit_empty"]["env_value"] == "" == scenarios.position_context("explicit_empty")
